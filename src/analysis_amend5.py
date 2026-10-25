"""PREREG Amendment 5: white-box analysis of persona homogenization (W1-W5) on cached decision-position residuals.

Primary: layer 32, every Phase 2 cell, h = mean over the 16 order x paraphrase cells per (persona, frame, instance).
Exploratory sweep: single cell (order 0, paraphrase (0,0)) at layers 4, 8, ..., 60, 63 (recaptured) and 46-53 (Phase 2).
Bootstrap, intervals and Holm as Amendments 3-4.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import yaml

from . import stats
from .analysis_amend3 import SUBST, _ci, cosv
from .analysis_phase2 import FAMS, NB, per_order, pilot_instances
from .prompts import ROOT

F13 = ["F1", "F3"]
FR4 = ("pick", "guess", "coord", "anti")
COLS = ["persona", "frame", "family", "instance_id", "order_id", "persona_par", "frame_par"]


def _ci_ref(est, draws, side, ref):
    """_ci on (x - ref), reported back on the original scale."""
    r = _ci(est - ref, np.asarray(draws, float) - ref, side)
    return {**r, "est": r["est"] + ref, "lo": r["lo"] + ref, "hi": r["hi"] + ref}


def substantive() -> tuple[list[str], dict[str, str]]:
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    subst = [p for p in pcfg if pcfg[p]["stratum"] in SUBST and p != "facilitator"]
    return subst, {p: pcfg[p]["stratum"] for p in subst}


# ------------------------------------------------------------------ loading
def load_l32(phase2_dir: str, subst: list[str]):
    """Per (frame, family): h (I, K, d) cell means, plug-in distances (I, K, K) and cross-cell unbiased squared
    distances (I, K, K); also the order-0 / paraphrase-(0,0) cell (I, K, d) for the recapture check."""
    pil, kidx, K = pilot_instances(), {p: i for i, p in enumerate(subst)}, len(subst)
    metas = []
    for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet")):
        d = pd.read_parquet(f, columns=COLS)
        metas.append((f, d[d.persona.isin(kidx) & ~d.instance_id.isin(pil) & d.frame.isin(FR4)]))
    keys = sorted({k for _, d in metas for k in zip(d.frame, d.family, d.instance_id)})
    gidx = {k: i for i, k in enumerate(keys)}
    X = None
    filled = np.zeros((len(keys), K, 16), bool)
    for f, d in metas:
        a = np.load(f[:-8] + "_l32.npy", mmap_mode="r")
        if X is None:
            X = np.zeros((len(keys), K, 16, a.shape[1]), np.float16)
        g = np.array([gidx[k] for k in zip(d.frame, d.family, d.instance_id)])
        k = d.persona.map(kidx).to_numpy()
        c = (d.order_id * 4 + d.persona_par * 2 + d.frame_par).to_numpy()
        X[g, k, c] = a[d.index.to_numpy()]
        filled[g, k, c] = True
    assert filled.all(), "missing layer-32 cells"
    H, Dp, Du, H0 = {}, {}, {}, {}
    for (fr, fam) in sorted({(k[0], k[1]) for k in keys}):
        gs = [gidx[k] for k in keys if k[0] == fr and k[1] == fam]          # sorted by instance id
        hs, dp, du = [], [], []
        for g in gs:
            Hc = X[g].astype(np.float64)                                    # (K, 16, d)
            n = Hc.shape[1]
            h = Hc.mean(1)
            hs.append(h.astype(np.float32))
            dp.append(_pdist(h))
            S = Hc.sum(1)
            q = (Hc ** 2).sum(2)                                            # (K, 16)
            G = np.einsum("isd,ksd->ik", Hc, Hc)                            # sum over cells of cross inner products
            within = q.sum(1)[:, None] + q.sum(1)[None, :] - 2 * G          # sum_s ||h_is - h_ks||^2
            du.append((_pdist(S) ** 2 - within) / (n * (n - 1)))
        H[(fr, fam)], Dp[(fr, fam)], Du[(fr, fam)] = np.stack(hs), np.stack(dp), np.stack(du)
        H0[(fr, fam)] = X[gs][:, :, 0].astype(np.float32)
    insts = {fam: sorted({k[2] for k in keys if k[1] == fam and k[0] == "pick"}) for fam in FAMS}
    return H, Dp, Du, H0, insts


def _pdist(h):
    sq = (h ** 2).sum(1)
    return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * h @ h.T, 0, None))


def behavioral_alignment(phase2_dir: str, subst: list[str], insts: dict) -> dict:
    """A3-A2 per instance: cos(pbar^coord - pbar^pick, pbar^guess - pbar^pick), (I, K) per family, T = 1."""
    df = pd.concat([pd.read_parquet(f, columns=COLS + ["p"]) for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet"))],
                   ignore_index=True)
    df = df[df.persona.isin(subst) & ~df.instance_id.isin(pilot_instances()) & df.frame.isin(("pick", "guess", "coord"))]
    PO = per_order(df)
    out = {}
    for fam in FAMS:
        A = np.full((len(insts[fam]), len(subst)), np.nan)
        for j, iid in enumerate(insts[fam]):
            for i, p in enumerate(subst):
                pp, pg, pc = (PO[(p, fam, iid, x)].mean(0) for x in ("pick", "guess", "coord"))
                A[j, i] = cosv(pc - pp, pg - pp)
        out[fam] = A
    return out


# ------------------------------------------------------------------ analysis
def core(H: dict, insts: dict, subst: list[str], stratum: dict, nb: int = NB, nsplit: int = 200, seed: int = 0,
         Dp: dict | None = None, Du: dict | None = None, beh: dict | None = None, a_hat: np.ndarray | None = None) -> dict:
    """W1-W3 on any (frame, family) -> (I, K, d) residual set; W4 and W5 when beh / a_hat are given."""
    K = len(subst)
    strata = [[i for i, p in enumerate(subst) if stratum[p] == s] for s in SUBST]
    strata = [v for v in strata if v]
    rng = np.random.default_rng(seed)
    nI = {f: len(insts[f]) for f in FAMS}
    ALL = {f: np.arange(nI[f]) for f in FAMS}
    allK = np.arange(K)

    def resample():
        return {f: rng.integers(0, nI[f], nI[f]) for f in FAMS}

    def presample():
        return np.concatenate([rng.choice(v, len(v)) for v in strata])

    def two_level(fn, side=None):
        return _ci(fn(allK, ALL), [fn(presample(), resample()) for _ in range(nb)], side)

    def pp(A, s, fams):
        """per-persona mean over families of instance means: (K,)"""
        return np.mean([np.nanmean(A[f][s[f]], 0) for f in fams], 0)

    out: dict = {"n_personas": K, "instances_per_family": nI}

    # ---------------- W1: mean pairwise distance between personas
    if Dp is None:
        Dp = {k: np.stack([_pdist(h.astype(np.float64)) for h in v]) for k, v in H.items()}

    def dmean(D, fr, pid, s, fams):
        mask = ~np.eye(len(pid), dtype=bool)
        return float(np.mean([D[(fr, f)][s[f]].mean(0)[np.ix_(pid, pid)][mask].mean() for f in fams]))

    def ratio_ci(D, fr, fams, side, sqrt=False):
        fn = lambda pid, s: dmean(D, fr, pid, s, fams) / dmean(D, "pick", pid, s, fams)
        tr = np.sqrt if sqrt else (lambda x: x)
        est = tr(fn(allK, ALL))
        return _ci_ref(est, [tr(fn(presample(), resample())) for _ in range(nb)], side, 1.0)

    W1 = {"D_pick": dmean(Dp, "pick", allK, ALL, FAMS), "D_coord": dmean(Dp, "coord", allK, ALL, FAMS),
          "D_guess": dmean(Dp, "guess", allK, ALL, FAMS), "D_pick_F13": dmean(Dp, "pick", allK, ALL, F13),
          "D_anti_F13": dmean(Dp, "anti", allK, ALL, F13),
          "coord_over_pick": ratio_ci(Dp, "coord", FAMS, "less"),
          "anti_over_pick_F13": ratio_ci(Dp, "anti", F13, None),
          "guess_over_pick": ratio_ci(Dp, "guess", FAMS, None)}
    if Du is not None:
        W1["robust_unbiased"] = {"coord_over_pick": ratio_ci(Du, "coord", FAMS, None, True),
                                 "anti_over_pick_F13": ratio_ci(Du, "anti", F13, None, True),
                                 "guess_over_pick": ratio_ci(Du, "guess", FAMS, None, True)}
    out["W1"] = W1

    # ---------------- W2: activation shift alignment
    Dl = {fr: {f: H[(fr, f)] - H[("pick", f)] for f in (F13 if fr == "anti" else FAMS)} for fr in ("coord", "anti", "guess")}

    def cosrows(A, B):
        num = (A * B).sum(-1)
        den = np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

    Ac = {f: cosrows(Dl["coord"][f], Dl["guess"][f]) for f in FAMS}
    Aa = {f: cosrows(Dl["anti"][f], Dl["guess"][f]) for f in F13}
    contrast = lambda pid, s: float((pp(Ac, s, F13) - pp(Aa, s, F13))[pid].mean())
    W2 = {"contrast_F13": two_level(contrast, "greater"),
          "coord_alignment_F13": two_level(lambda pid, s: float(pp(Ac, s, F13)[pid].mean())),
          "anti_alignment_F13": two_level(lambda pid, s: float(pp(Aa, s, F13)[pid].mean())),
          "coord_alignment_F1F5": two_level(lambda pid, s: float(pp(Ac, s, FAMS)[pid].mean()))}
    e_c, e_a = pp(Ac, ALL, F13), pp(Aa, ALL, F13)
    dr = [resample() for _ in range(nb)]
    dc = np.array([pp(Ac, s, F13) for s in dr])
    da = np.array([pp(Aa, s, F13) for s in dr])
    W2["per_persona"] = {p: {"coord": float(e_c[i]), "anti": float(e_a[i]), "contrast": float(e_c[i] - e_a[i]),
                             "lo": float(np.quantile(dc[:, i] - da[:, i], .025)), "hi": float(np.quantile(dc[:, i] - da[:, i], .975))}
                         for i, p in enumerate(subst)}
    out["W2"] = W2

    # ---------------- W3 (+ W4, W5-rho): generic-AI direction from Delta^guess, split-half
    def g_of(pid, s, half):
        """mean Delta^guess over persona positions pid and resampled instances s[f] that fall in half[f]."""
        wK = np.bincount(pid, minlength=K).astype(float)
        num, den = 0.0, 0.0
        for f in FAMS:
            sel = s[f][half[f][s[f]]]
            W = np.outer(np.bincount(sel, minlength=nI[f]), wK)
            num = num + np.tensordot(W, Dl["guess"][f], axes=([0, 1], [0, 1]))
            den += W.sum()
        return num / den

    def proj(fr, gh, s, held, fams):
        return np.mean([(Dl[fr][f][s[f][held[f][s[f]]]] @ gh).mean(0) for f in fams], 0)   # (K,)

    def split():
        A = {}
        for f in FAMS:
            m = np.zeros(nI[f], bool)
            m[rng.choice(nI[f], nI[f] // 2, replace=False)] = True
            A[f] = m
        return A

    def w3_once(pid, s, A):
        g = g_of(pid, s, A)
        gh = g / np.linalg.norm(g)
        B = {f: ~A[f] for f in FAMS}
        return {"pc": proj("coord", gh, s, B, FAMS), "pc13": proj("coord", gh, s, B, F13),
                "pa13": proj("anti", gh, s, B, F13), "pg": proj("guess", gh, s, B, FAMS)}

    ests, rel = [], []
    for _ in range(nsplit):
        A = split()
        ests.append(w3_once(allK, ALL, A))
        gA, gB = g_of(allK, ALL, A), g_of(allK, ALL, {f: ~A[f] for f in FAMS})
        rel.append(cosv(gA, gB))
    E = {k: np.mean([e[k] for e in ests], 0) for k in ests[0]}               # per persona, averaged over splits
    if a_hat is not None:
        Pk = {f: H[("pick", f)] @ a_hat for f in FAMS}

        def alpha_of(s):
            a = pp(Pk, s, FAMS)
            return a - a[subst.index("assistant")]
    draws = {"a": [], "b": [], "rho_beh": [], "rho_alpha": [], "rho_absalpha": []}
    for _ in range(nb):
        pid, s, A = presample(), resample(), split()
        r = w3_once(pid, s, A)
        draws["a"].append(r["pc"][pid].mean())
        draws["b"].append((r["pc13"] - r["pa13"])[pid].mean())
        if beh is not None:
            draws["rho_beh"].append(stats.spearman(r["pc"][pid], pp(beh, s, FAMS)[pid]))
        if a_hat is not None:
            al = alpha_of(s)
            draws["rho_alpha"].append(stats.spearman(r["pc"][pid], al[pid]))
            draws["rho_absalpha"].append(stats.spearman(r["pc"][pid], np.abs(al)[pid]))
    W3 = {"a_coord_proj": _ci(E["pc"].mean(), draws["a"], "greater"),
          "b_coord_minus_anti_F13": _ci((E["pc13"] - E["pa13"]).mean(), draws["b"], "greater"),
          "coord_proj_F13": float(E["pc13"].mean()), "anti_proj_F13": float(E["pa13"].mean()),
          "guess_proj_heldout": float(E["pg"].mean()),
          "relative_pull_coord": float(E["pc"].mean() / E["pg"].mean()),
          "reliability": {"mean": float(np.mean(rel)), "q025": float(np.quantile(rel, .025)), "q975": float(np.quantile(rel, .975))},
          "per_persona": {p: {"coord": float(E["pc"][i]), "coord_F13": float(E["pc13"][i]), "anti_F13": float(E["pa13"][i]),
                              "guess": float(E["pg"][i])} for i, p in enumerate(subst)}}
    out["W3"] = W3

    if beh is not None:
        b_pp = pp(beh, ALL, FAMS)
        rho = stats.spearman(E["pc"], b_pp)
        prng = np.random.default_rng(seed + 1)
        perm = np.array([stats.spearman(E["pc"], prng.permutation(b_pp)) for _ in range(10000)])
        W4 = _ci(rho, draws["rho_beh"], None)
        W4["p"] = float(((perm >= rho).sum() + 1) / (len(perm) + 1))
        W4["p_kind"] = "one-sided permutation (10,000)"
        W4["per_persona"] = {p: {"proj": float(E["pc"][i]), "behavioral_alignment": float(b_pp[i])} for i, p in enumerate(subst)}
        out["W4"] = W4

    if a_hat is not None:
        Pa = {fr: {f: Dl[fr][f] @ a_hat for f in Dl[fr]} for fr in Dl}
        gfull = lambda pid, s: g_of(pid, s, {f: np.ones(nI[f], bool) for f in FAMS})
        al = alpha_of(ALL)
        W5 = {"cos_g_axis": two_level(lambda pid, s: cosv(gfull(pid, s), a_hat)),
              "proj_axis_coord": two_level(lambda pid, s: float(pp(Pa["coord"], s, FAMS)[pid].mean())),
              "proj_axis_guess": two_level(lambda pid, s: float(pp(Pa["guess"], s, FAMS)[pid].mean())),
              "proj_axis_anti_F13": two_level(lambda pid, s: float(pp(Pa["anti"], s, F13)[pid].mean())),
              "proj_axis_coord_F13": two_level(lambda pid, s: float(pp(Pa["coord"], s, F13)[pid].mean())),
              "rho_pull_alpha": _ci(stats.spearman(E["pc"], al), draws["rho_alpha"]),
              "rho_pull_absalpha": _ci(stats.spearman(E["pc"], np.abs(al)), draws["rho_absalpha"]),
              "alpha": {p: float(al[i]) for i, p in enumerate(subst)},
              "per_persona_axis": {p: {fr: float(pp(Pa[fr], ALL, F13 if fr == "anti" else FAMS)[i]) for fr in Pa}
                                   for i, p in enumerate(subst)}}
        out["W5"] = W5
    return out


def load_sweep(sweep_dir: str, phase2_dir: str, subst: list[str], insts: dict):
    """layer -> {(frame, family): (I, K, d)} single-cell residuals (order 0, paraphrase (0,0))."""
    kidx = {p: i for i, p in enumerate(subst)}
    iidx = {f: {iid: j for j, iid in enumerate(insts[f])} for f in FAMS}
    pil = pilot_instances()
    out: dict = {}

    def put(layer, d, arr):
        for fr, fam in {(a, b) for a, b in zip(d.frame, d.family)}:
            m = ((d.frame == fr) & (d.family == fam)).to_numpy()
            dd = d[m]
            buf = out.setdefault(layer, {}).setdefault((fr, fam), np.full((len(insts[fam]), len(subst), arr.shape[-1]), np.nan, np.float32))
            buf[dd.instance_id.map(iidx[fam]).to_numpy(), dd.persona.map(kidx).to_numpy()] = arr[m]

    idx = pd.read_parquet(f"{sweep_dir}/index.parquet")
    acts = np.load(f"{sweep_dir}/acts.npy", mmap_mode="r")
    layers = yaml.safe_load(open(f"{sweep_dir}/summary.json"))["layers"]
    for li, L in enumerate(layers):
        put(L, idx, np.asarray(acts[:, li], np.float32))
    for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet")):
        d = pd.read_parquet(f, columns=COLS)
        z = np.load(f[:-8] + "_extra.npz")
        rows = z["rows"]
        dd = d.iloc[rows].reset_index(drop=True)
        m = (dd.persona.isin(kidx) & ~dd.instance_id.isin(pil) & dd.frame.isin(FR4)).to_numpy()
        for k in z.files:
            if k != "rows" and int(k[1:]) not in layers:  # 48, 52: the recaptured copy is used
                put(int(k[1:]), dd[m].reset_index(drop=True), np.asarray(z[k][m], np.float32))
    for L in out:
        for k, v in out[L].items():
            assert not np.isnan(v).any(), (L, k)
    return out, layers


def analyze(phase2_dir: str, sweep_dir: str, axis_l32: np.ndarray, nb_sweep: int = 500, seed: int = 0, log=print) -> dict:
    subst, stratum = substantive()
    a_hat = axis_l32 / np.linalg.norm(axis_l32)
    H, Dp, Du, H0, insts = load_l32(phase2_dir, subst)
    log("loaded layer 32")
    beh = behavioral_alignment(phase2_dir, subst, insts)
    log("behavioral alignment done")
    rep = {"substantive": subst, "primary_l32": core(H, insts, subst, stratum, NB, seed=seed, Dp=Dp, Du=Du, beh=beh, a_hat=a_hat)}
    P = rep["primary_l32"]
    adj = stats.holm([P["W1"]["coord_over_pick"]["p"], P["W2"]["contrast_F13"]["p"], P["W3"]["a_coord_proj"]["p"],
                      P["W3"]["b_coord_minus_anti_F13"]["p"], P["W4"]["p"]])
    rep["holm"] = dict(zip(["W1", "W2", "W3-a", "W3-b", "W4"], map(float, adj)))
    log("primary done")
    del Dp, Du
    sweep, slayers = load_sweep(sweep_dir, phase2_dir, subst, insts)
    # recapture check: recaptured layer 32 vs stored layer 32, order-0 / paraphrase-(0,0) cell
    a = np.concatenate([sweep[32][k].reshape(-1, sweep[32][k].shape[-1]) for k in sorted(H0)])
    b = np.concatenate([H0[k].reshape(-1, H0[k].shape[-1]) for k in sorted(H0)])
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    cs = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    rep["recapture_check_l32"] = {"cells": int(len(a)), "rel_err_median": float(np.median(rel)), "rel_err_max": float(rel.max()),
                                  "cos_min": float(cs.min()), "cos_median": float(np.median(cs))}
    del H, H0
    SW = {}
    for L in sorted(sweep):
        r = core(sweep[L], insts, subst, stratum, nb_sweep, nsplit=50, seed=seed + L)
        SW[str(L)] = {"W1_coord_over_pick": r["W1"]["coord_over_pick"], "W1_anti_over_pick_F13": r["W1"]["anti_over_pick_F13"],
                      "W2_contrast_F13": r["W2"]["contrast_F13"], "W2_coord_alignment_F1F5": r["W2"]["coord_alignment_F1F5"],
                      "W3_a": r["W3"]["a_coord_proj"], "W3_b": r["W3"]["b_coord_minus_anti_F13"],
                      "W3_relative_pull": r["W3"]["relative_pull_coord"], "W3_reliability": r["W3"]["reliability"],
                      "source": "recaptured" if L in slayers else "phase2_extra"}
        log(f"sweep layer {L} done")
    rep["sweep"] = SW
    return rep
