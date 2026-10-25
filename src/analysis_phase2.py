"""Phase 2 analysis (PREREG.md Amendment 1, Part B, Amendment 2). Reads cached distributions only.

Primary estimator: cross-order U-statistic C^{f,g}_ij = mean_{s != s'} <p^f_{i,s}, p^g_{j,s'}>, with p_{i,s}
persona i's distribution under order s mixed over the 2x2 paraphrases. Mixture estimator <pbar_i, pbar_j> for
PSD checks and eigen-decompositions. Population quantities: per-family instance-mean matrices, pooled as the
mean over F1-F5; two-level bootstrap (personas within stratum, instances within family).

python -m src.analysis_phase2 RESULTS_DIR OUT.json [--keep-pilot]
"""
from __future__ import annotations

import glob
import json
import random
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from . import gram, prompts
from .prompts import ROOT

FAMS = ["F1", "F2", "F3", "F4", "F5"]
SUBST = ("assistant", "stratified", "theory", "realistic")
R_DEFAULT = -4.544155597686768
NB = 2000


def load(d: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/dist_shard*.parquet"))], ignore_index=True)


def pilot_instances() -> set[str]:
    out = set()
    for fam in FAMS:
        out |= {g["instance_id"] for g in random.Random(f"pilot:{fam}").sample(prompts.load_games(fam), 4)}
    return out


def per_order(df: pd.DataFrame, col: str = "p") -> dict:
    """(persona, family, instance, frame) -> [O, n], mixed over persona and frame paraphrases."""
    out = {}
    for key, g in df.groupby(["persona", "family", "instance_id", "frame"]):
        O, n = int(g.order_id.max()) + 1, len(g.iloc[0][col])
        acc, cnt = np.zeros((O, n)), np.zeros(O)
        for o, p in zip(g.order_id.values, g[col].values):
            acc[o] += p
            cnt[o] += 1
        out[key] = acc / cnt[:, None]
    return out


def cross(A, B):
    G = A @ B.T
    O = len(G)
    return (G.sum() - np.trace(G)) / (O * (O - 1))


def cross_pairs(Ps, Qs):
    """Ps, Qs: (K, O, n). Returns (K, K) cross-order inner products C_ij = mean_{o != q} <P_io, Q_jq>."""
    G = np.einsum("ion,jqn->ijoq", Ps, Qs)
    O = Ps.shape[1]
    return (G.sum((2, 3)) - np.einsum("ijoo->ij", G)) / (O * (O - 1))


def temper_col(df, T):
    def f(z):
        z = np.asarray(z) / T
        e = np.exp(z - z.max())
        return e / e.sum()
    return df.logprobs.apply(f)


def spearman(x, y):
    return float(spearmanr(x, y).statistic)


def kl(p, q):
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.clip(q[m], 1e-300, None)))).sum())


def ci(est, draws):
    """Percentile interval plus a two-sided bootstrap p-value against 0."""
    draws = np.asarray(draws, float)
    d = draws[~np.isnan(draws)]
    p0 = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / (len(d) + 1)))
    return {"est": float(est), "lo": float(np.nanquantile(draws, .025)), "hi": float(np.nanquantile(draws, .975)), "p0": p0}


def analyze(d: str, exclude_pilot: bool = True) -> dict:
    df = load(d)
    excluded = sorted(pilot_instances()) if exclude_pilot else []
    if excluded:
        df = df[~df.instance_id.isin(excluded)].reset_index(drop=True)
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    personas = [p for p in pcfg if p in set(df.persona)]
    stratum = {p: pcfg[p]["stratum"] for p in personas}
    idx = {p: i for i, p in enumerate(personas)}
    K = len(personas)
    rng = np.random.default_rng(0)

    # ---------------- axis position and manipulation check (decides the substantive set)
    alpha_raw = df[df.frame == "pick"].groupby("persona").proj_l32.mean()
    alpha = {p: float(alpha_raw[p] - alpha_raw["assistant"]) for p in personas}
    manip_pass = {}
    for p in personas:
        r = pcfg[p].get("released_projection_l32")
        if r is not None:
            manip_pass[p] = bool(abs(r - R_DEFAULT) < 2 or np.sign(alpha[p]) == np.sign(r - R_DEFAULT))
    dropped = [p for p, ok in manip_pass.items() if not ok]
    subst = [p for p in personas if stratum[p] in SUBST and p not in dropped]
    sidx = np.array([idx[p] for p in subst])
    a0 = idx["assistant"]
    a_vec = np.array([alpha[p] for p in personas])

    PO = per_order(df)
    insts = {f: sorted({k[2] for k in PO if k[1] == f}) for f in FAMS}
    frames = ["pick", "guess", "coord", "anti"]

    # ---------------- per-family instance arrays
    selfC = {fr: {} for fr in frames}
    pairC = {fr: {} for fr in ("pick", "coord", "guess", "coord_pick")}
    pairM = {fr: {} for fr in ("pick", "coord")}
    psd_fail, ident_max = 0, 0.0
    for f in FAMS:
        for fr in frames:
            if (personas[0], f, insts[f][0], fr) not in PO:
                continue
            X = np.full((K, len(insts[f])), np.nan)
            for j, iid in enumerate(insts[f]):
                for p in personas:
                    A = PO[(p, f, iid, fr)]
                    X[idx[p], j] = cross(A, A)
            selfC[fr][f] = X
        stacks = {fr: [np.stack([PO[(p, f, iid, fr)] for p in personas]) for iid in insts[f]] for fr in ("pick", "coord", "guess")}
        for fr in ("pick", "coord", "guess"):
            pairC[fr][f] = np.stack([cross_pairs(S, S) for S in stacks[fr]])
        pairC["coord_pick"][f] = np.stack([cross_pairs(Sc, Sp) for Sc, Sp in zip(stacks["coord"], stacks["pick"])])
        for fr in ("pick", "coord"):
            Cm = []
            for S in stacks[fr]:
                Pbar = S.mean(1)
                M = gram.gram(Pbar)
                ok, _ = gram.psd_check(M)
                psd_fail += not ok
                ident_max = max(ident_max, gram.identity_residual(Pbar, M))
                Cm.append(M)
            pairM[fr][f] = np.stack(Cm)

    ALL = {f: np.arange(len(insts[f])) for f in FAMS}

    def resample_inst():
        return {f: rng.integers(0, len(insts[f]), len(insts[f])) for f in FAMS}

    strata_lists = {s: [idx[p] for p in subst if stratum[p] == s] for s in SUBST}

    def two_level(fn, strata_keys=SUBST, B=NB):
        base = np.array([i for s in strata_keys for i in strata_lists[s]])
        est = fn(base, ALL)
        draws = []
        for _ in range(B):
            pid = np.concatenate([rng.choice(strata_lists[s], len(strata_lists[s])) for s in strata_keys if strata_lists[s]])
            draws.append(fn(pid, resample_inst()))
        return ci(est, draws)

    def fam_mats(fr, s, which=pairC):
        return {f: which[fr][f][s[f]].mean(0) for f in FAMS}

    def pooled(fr, s, which=pairC):
        return np.mean(list(fam_mats(fr, s, which).values()), 0)

    def offdiag_mean(M, pid):
        sub = M[np.ix_(pid, pid)]
        k = len(pid)
        return (sub.sum() - np.trace(sub)) / (k * (k - 1))

    def cosm(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        return M / np.outer(d, d)

    # ================= A2: convergence
    eps = None
    ap = idx.get("assistant_paraphrase")
    if ap is not None:
        others = [idx[p] for p in subst if p != "assistant"]
        dv = lambda s: np.abs(pooled("coord", s)[others, ap] - pooled("coord", s)[others, a0]).mean()
        eps = ci(dv(ALL), [dv(resample_inst()) for _ in range(NB)])

    phi = lambda fr, pid, s: offdiag_mean(pooled(fr, s), pid)
    A2a = two_level(lambda pid, s: phi("coord", pid, s) - phi("pick", pid, s))
    A2a["Phi_pick"] = two_level(lambda pid, s: phi("pick", pid, s))
    A2a["Phi_coord"] = two_level(lambda pid, s: phi("coord", pid, s))
    A2a["eps_wording"] = eps
    A2a["pass"] = bool(A2a["lo"] > 0 and eps is not None and A2a["est"] > eps["hi"])

    def abar(fr, pid, s):
        return np.mean([offdiag_mean(cosm(M), pid) for M in fam_mats(fr, s).values()])

    def angle(fr, pid, s):
        k = len(pid)
        mask = ~np.eye(k, dtype=bool)
        return float(np.mean([np.degrees(np.arccos(np.clip(cosm(M)[np.ix_(pid, pid)][mask], -1, 1))).mean()
                              for M in fam_mats(fr, s).values()]))

    A2b = {"Abar_pick": two_level(lambda pid, s: abar("pick", pid, s)),
           "Abar_coord": two_level(lambda pid, s: abar("coord", pid, s)),
           "dAbar": two_level(lambda pid, s: abar("coord", pid, s) - abar("pick", pid, s)),
           "frac_disagreement_removed": two_level(lambda pid, s: 1 - (1 - abar("coord", pid, s)) / (1 - abar("pick", pid, s))),
           "angle_pick_deg": two_level(lambda pid, s: angle("pick", pid, s)),
           "angle_coord_deg": two_level(lambda pid, s: angle("coord", pid, s)),
           "d_angle_deg": two_level(lambda pid, s: angle("coord", pid, s) - angle("pick", pid, s))}
    A2b["pass"] = bool(A2b["dAbar"]["lo"] > 0)
    # one-sided reading for directional criteria is not used; Holm runs on two-sided bootstrap p-values

    def decomp(pid, s):
        Fp, Fc = fam_mats("pick", s), fam_mats("coord", s)
        tot, dec, agr, dlphi = [], [], [], []
        k = len(pid)
        mask = ~np.eye(k, dtype=bool)
        for f in FAMS:
            P, Cc = Fp[f][np.ix_(pid, pid)], Fc[f][np.ix_(pid, pid)]
            dlogC = np.log(Cc) - np.log(P)
            dd = 0.5 * (np.log(np.diag(Cc)) - np.log(np.diag(P)))
            D = dd[:, None] + dd[None, :]
            dlc = np.log(cosm(Cc)) - np.log(cosm(P))
            tot.append(dlogC[mask].mean()); dec.append(D[mask].mean()); agr.append(dlc[mask].mean())
            dlphi.append(np.log(Cc[mask].mean()) - np.log(P[mask].mean()))
        return np.array([np.mean(tot), np.mean(dec), np.mean(agr), np.mean(dlphi)])

    base = sidx
    est_c = decomp(base, ALL)
    draws_c = []
    for _ in range(NB):
        pid = np.concatenate([rng.choice(strata_lists[s], len(strata_lists[s])) for s in SUBST if strata_lists[s]])
        draws_c.append(decomp(pid, resample_inst()))
    draws_c = np.array(draws_c)
    A2c = {name: ci(est_c[i], draws_c[:, i]) for i, name in enumerate(["mean_dlogC", "decisiveness", "agreement", "dlogPhi"])}

    nonA_idx = [idx[p] for p in subst if p != "assistant"]

    def gains(s):
        g = []
        for f, Mcp in fam_mats("coord_pick", s).items():
            Mp, Mc = fam_mats("pick", s)[f], fam_mats("coord", s)[f]
            cos_c = Mcp[nonA_idx, a0] / np.sqrt(np.diag(Mc)[nonA_idx] * Mp[a0, a0])
            cos_p = Mp[nonA_idx, a0] / np.sqrt(np.diag(Mp)[nonA_idx] * Mp[a0, a0])
            g.append(cos_c - cos_p)
        return np.mean(g, 0)

    g_est = gains(ALL)
    absal = np.abs(a_vec[nonA_idx])
    gd, sd = [], []
    nonA_strata = {s: [nonA_idx.index(i) for i in strata_lists[s]] for s in SUBST if s != "assistant"}
    for _ in range(NB):
        gb = gains(resample_inst())
        pick = np.concatenate([rng.choice(v, len(v)) for v in nonA_strata.values() if v])
        gd.append(gb[pick].mean())
        sd.append(spearman(gb[pick], absal[pick]))
    A2d = {"mean_gain": ci(g_est.mean(), gd), "spearman_gain_absalpha": ci(spearman(g_est, absal), sd),
           "per_persona": {personas[i]: float(v) for i, v in zip(nonA_idx, g_est)}}

    # A2-(e): convergence target (instance level, mixture distributions)
    human = {g["instance_id"]: g for fam in ("F1", "F4") for g in prompts.load_games(fam) if "human" in g}
    rows_e = []
    for f in FAMS:
        for iid in insts[f]:
            qbar = np.mean([PO[(p, f, iid, "coord")].mean(0) for p in subst], 0)
            c1 = PO[("assistant", f, iid, "pick")].mean(0)
            c2 = np.mean([PO[(p, f, iid, "guess")].mean(0) for p in subst], 0)
            cosf = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            r = {"family": f, "id": iid, "cos1": cosf(qbar, c1), "cos2": cosf(qbar, c2), "kl1": kl(c1, qbar), "kl2": kl(c2, qbar)}
            if iid in human:
                k = int(np.argmax(human[iid]["human"]["coordinate"]))
                e = np.zeros_like(qbar); e[k] = 1
                r.update({"cos3": cosf(qbar, e), "kl3": float(-np.log(max(qbar[k], 1e-300)))})
            rows_e.append(r)
    E = pd.DataFrame(rows_e)

    def fam_boot(col_fn, sub):
        """mean over families of family means (families present in `sub`); instance bootstrap within family."""
        fams = sorted(sub.family.unique())
        groups = {f: sub[sub.family == f] for f in fams}
        est = np.mean([col_fn(groups[f]).mean() for f in fams])
        dr = [np.mean([col_fn(groups[f].iloc[rng.integers(0, len(groups[f]), len(groups[f]))]).mean() for f in fams]) for _ in range(NB)]
        return ci(est, dr)

    A2e = {"all": {k: fam_boot(lambda g, k=k: g[k], E) for k in ("cos1", "cos2", "kl1", "kl2")},
           "all_diff_cos_2_minus_1": fam_boot(lambda g: g.cos2 - g.cos1, E),
           "all_diff_kl_2_minus_1": fam_boot(lambda g: g.kl2 - g.kl1, E)}
    B = E.dropna(subset=["cos3"])
    if len(B):
        A2e["bardsley"] = {"n": int(len(B)), **{k: fam_boot(lambda g, k=k: g[k], B) for k in ("cos1", "cos2", "cos3", "kl1", "kl2", "kl3")}}
        for a, b in (("3", "1"), ("3", "2")):
            A2e["bardsley"][f"diff_cos_{a}_minus_{b}"] = fam_boot(lambda g, a=a, b=b: g[f"cos{a}"] - g[f"cos{b}"], B)
            A2e["bardsley"][f"diff_kl_{a}_minus_{b}"] = fam_boot(lambda g, a=a, b=b: g[f"kl{a}"] - g[f"kl{b}"], B)
    cl_all = "Assistant pick" if A2e["all"]["cos1"]["est"] > A2e["all"]["cos2"]["est"] else "population guess"
    A2e["closest_by_cos_all"] = cl_all

    # A2-(f): instruction effect / axis gap
    def axis_gap(fr, pid, s):
        k = len(pid)
        iu = np.triu_indices(k, 1)
        dist = np.abs(a_vec[pid][:, None] - a_vec[pid][None, :])[iu]
        lo_t, hi_t = np.quantile(dist, [1 / 3, 2 / 3])
        vals = []
        for M in fam_mats(fr, s).values():
            c = cosm(M)[np.ix_(pid, pid)][iu]
            vals.append(c[dist <= lo_t].mean() - c[dist >= hi_t].mean())
        return float(np.mean(vals))

    A2f = {"axis_gap_pick": two_level(lambda pid, s: axis_gap("pick", pid, s)),
           "axis_gap_coord": two_level(lambda pid, s: axis_gap("coord", pid, s)),
           "ratio": two_level(lambda pid, s: (abar("coord", pid, s) - abar("pick", pid, s)) / axis_gap("pick", pid, s))}

    # ================= persona x frame
    def pooled_self(fr, s):
        return np.mean([np.nanmean(selfC[fr][f][:, s[f]], 1) for f in FAMS], 0)

    S_fn = lambda s: pooled_self("coord", s) - pooled_self("pick", s)
    S_est = S_fn(ALL)
    S_draws = np.array([S_fn(resample_inst()) for _ in range(NB)])
    S_lo, S_hi = np.quantile(S_draws, [.025, .975], 0)
    fam_S = {f: np.nanmean(selfC["coord"][f] - selfC["pick"][f], 1) for f in FAMS}
    anti = {}
    for f in ("F1", "F3"):
        D = selfC["anti"][f] - selfC["pick"][f]
        dr = np.array([D[:, rng.integers(0, D.shape[1], D.shape[1])].mean(1) for _ in range(NB)])
        anti[f] = (D.mean(1), np.quantile(dr, .025, 0), np.quantile(dr, .975, 0))
    omega = df.groupby(["persona", "frame"]).offmenu_mass.mean().unstack()
    h10 = {}
    for p in personas:
        g = [kl(PO[(p, f, iid, "coord")].mean(0), PO[(p, f, iid, "pick")].mean(0)) -
             kl(PO[(p, f, iid, "coord")].mean(0), PO[(p, f, iid, "guess")].mean(0)) for f in FAMS for iid in insts[f]]
        h10[p] = float(np.mean(g))
    rows = []
    for p in personas:
        i = idx[p]
        rows.append({
            "persona": p, "stratum": stratum[p], "alpha": alpha[p], "released_proj": pcfg[p].get("released_projection_l32"),
            "manip_pass": manip_pass.get(p), "dropped": p in dropped, "substantive": p in subst,
            "omega": {fr: float(omega.loc[p, fr]) for fr in omega.columns if not np.isnan(omega.loc[p, fr])},
            "C_self": {fr: {**{f: float(np.nanmean(selfC[fr][f][i])) for f in selfC[fr]},
                            "pooled": float(np.mean([np.nanmean(selfC[fr][f][i]) for f in selfC[fr]]))} for fr in frames if selfC[fr]},
            "S": {"est": float(S_est[i]), "lo": float(S_lo[i]), "hi": float(S_hi[i]), **{f: float(fam_S[f][i]) for f in FAMS}},
            "anti_minus_pick": {f: {"est": float(anti[f][0][i]), "lo": float(anti[f][1][i]), "hi": float(anti[f][2][i])} for f in anti},
            "H10_kl_gain": h10[p],
            "replace_flag": bool(stratum[p] == "stratified" and (omega.loc[p] > 0.5).any()),
            "gain_toward_assistant": A2d["per_persona"].get(p),
        })
    lu = [r for r in rows if r["released_proj"] is not None]
    manip = {"spearman_alpha_vs_released": spearman([r["alpha"] for r in lu], [r["released_proj"] for r in lu]),
             "failed": dropped, "n_lu_roles": len(lu)}

    # A1 secondaries (on the substantive set)
    S_sub_draws = S_draws
    def a1a(pid, s):
        return spearman(S_fn(s)[pid], a_vec[pid])
    A1a = two_level(a1a)
    gt = idx.get("game_theorist")
    A1b = {"est": float(S_est[gt]), "lo": float(S_lo[gt]), "hi": float(S_hi[gt])} if gt is not None else None
    A1c = {f: float(np.mean([anti[f][2][idx[p]] < 0 for p in subst])) for f in anti}

    # ================= Gram analysis
    pop = {}
    for fr in ("pick", "coord"):
        pop[f"Phi_{fr}_by_family"] = {f: float(offdiag_mean(pairC[fr][f].mean(0), sidx)) for f in FAMS}
        pop[f"Phi_{fr}_by_stratum"] = {st: float(offdiag_mean(pooled(fr, ALL), np.array(strata_lists[st])))
                                       for st in ("stratified", "theory", "realistic") if len(strata_lists[st]) > 1}
        pop[f"mixture_Phi_{fr}"] = float(offdiag_mean(pooled(fr, ALL, pairM), sidx))
    df03 = df[df.frame.isin(["pick", "coord"])].copy()
    df03["p"] = temper_col(df03, 0.3)
    PO3 = per_order(df03)
    for fr in ("pick", "coord"):
        vals = [offdiag_mean(np.mean([cross_pairs(*(2 * [np.stack([PO3[(p, f, iid, fr)] for p in subst])])) for iid in insts[f]], 0),
                             np.arange(len(subst))) for f in FAMS]
        pop[f"Phi_{fr}_T0.3"] = float(np.mean(vals))
    nonA = [s for s in SUBST if s != "assistant"]
    H2 = two_level(lambda pid, s: spearman(a_vec[pid], pooled("coord", s)[pid, a0]), strata_keys=nonA)
    H2["pick_version"] = two_level(lambda pid, s: spearman(a_vec[pid], pooled("pick", s)[pid, a0]), strata_keys=nonA)

    def h3_fn(pid, s):
        C = pooled("coord", s, pairM)[np.ix_(pid, pid)]
        k = len(pid)
        Hc = np.eye(k) - 1 / k
        w, V = np.linalg.eigh(Hc @ C @ Hc)
        return abs(spearman(V[:, -1], a_vec[pid]))
    H3 = two_level(h3_fn)
    C3 = pooled("coord", ALL, pairM)[np.ix_(sidx, sidx)]
    Hc = np.eye(len(sidx)) - 1 / len(sidx)
    w, V = np.linalg.eigh(Hc @ C3 @ Hc)
    H3["var_share_u1"] = float(w[-1] / w[w > 0].sum())
    H3["u1"] = dict(zip(subst, V[:, -1].round(4).tolist()))

    # ================= additional outputs
    real = [p for p in personas if stratum[p] == "realistic"]
    ri = [idx[p] for p in real]
    realistic = {"order": real, **{fr: pooled(fr, ALL)[np.ix_(ri, ri)].round(4).tolist() for fr in ("pick", "coord")},
                 "cos": {fr: cosm(pooled(fr, ALL))[np.ix_(ri, ri)].round(4).tolist() for fr in ("pick", "coord")}}
    assets = json.loads((ROOT / "results/assets/qwen3-32b_assets.json").read_text())
    caps = {c["layer"]: c["cap"] for c in assets["capping_released"]}
    ex = df[df.proj_extra.apply(lambda v: v is not None and not np.isnan(v).all())]
    layers = list(range(46, 54))
    Pm = np.stack(ex.proj_extra.values)
    binds = Pm < -np.array([caps[l] for l in layers])[None, :]
    bind = {"layers": layers, "caps": [caps[l] for l in layers],
            "per_persona": {p: binds[(ex.persona == p).values].mean(0).round(4).tolist() for p in personas},
            "per_persona_any": {p: float(binds[(ex.persona == p).values].any(1).mean()) for p in personas},
            "per_stratum_any": {s: float(binds[ex.persona.map(stratum).values == s].any(1).mean()) for s in set(stratum.values())},
            "mean_proj_by_persona": {p: Pm[(ex.persona == p).values].mean(0).round(2).tolist() for p in personas},
            "n_cells_per_persona": int((ex.persona == "assistant").sum())}
    bind["assistant_flag"] = bool(bind["per_persona_any"]["assistant"] > 0.05)

    from .stats import holm
    fam = {"H2": H2["p0"], "H3": H3["p0"], "A1-(a)": A1a["p0"],
           "A2-(a)": A2a["p0"], "A2-(b)": A2b["dAbar"]["p0"],
           "A2-(d) mean": A2d["mean_gain"]["p0"], "A2-(d) spearman": A2d["spearman_gain_absalpha"]["p0"]}
    if A1b is not None:
        A1b_draws_p = float(min(1.0, 2 * min((S_draws[:, gt] <= 0).mean(), (S_draws[:, gt] >= 0).mean()) + 1 / (NB + 1)))
        fam["A1-(b)"] = A1b_draws_p
    adj = holm(list(fam.values()))
    holm_tab = {k: {"p": float(v), "p_holm": float(a)} for (k, v), a in zip(fam.items(), adj)}

    Cc, Cp = pooled("coord", ALL), pooled("pick", ALL)
    return {"holm": holm_tab,
        "n_rows": len(df), "excluded_pilot_instances": excluded, "personas": personas, "stratum": stratum,
        "substantive": subst, "instances_per_family": {f: len(v) for f, v in insts.items()},
        "A2": {"a": A2a, "b": A2b, "c": A2c, "d": A2d, "e": A2e, "f": A2f},
        "persona_rows": rows, "manipulation": manip,
        "A1": {"a_spearman_S_alpha": A1a, "b_game_theorist_S": A1b, "c_anti_share_below0": A1c},
        "population": pop, "H2": H2, "H3": H3, "realistic": realistic, "bind": bind,
        "sanity": {"psd_failures": int(psd_fail), "identity_max_residual": float(ident_max)},
        "gram": {"order": personas, "coord": Cc.round(5).tolist(), "pick": Cp.round(5).tolist()},
        "convergence_instances": rows_e,
    }


if __name__ == "__main__":
    rep = analyze(sys.argv[1], exclude_pilot="--keep-pilot" not in sys.argv)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(rep, f, default=float)
    print(json.dumps({k: rep[k] for k in ("manipulation", "A1", "H2", "sanity")}, indent=1, default=float))
