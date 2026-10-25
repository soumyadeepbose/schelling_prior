"""PREREG Amendment 3: convergence mechanism (Part A, cached Phase 2 data) and the partner-description
experiment (Part B, Phase 2b). Reads cached distributions only.

Population cosines: cross-order estimator on per-family instance-mean matrices (as A2-(b)).
Per-instance vectors: order- and paraphrase-mixed distributions pbar.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import yaml

from . import stats
from .analysis_phase2 import FAMS, NB, cross_pairs, per_order, pilot_instances, temper_col
from .prompts import ROOT

SUBST = ("assistant", "stratified", "theory", "realistic")


def _ci(est, draws, side=None):
    """Interval + p-value. side None: two-sided vs 0; 'less' / 'greater': one-sided share of draws on the wrong side."""
    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    if side == "greater":
        p = (d <= 0).mean() + 1 / (len(d) + 1)
    elif side == "less":
        p = (d >= 0).mean() + 1 / (len(d) + 1)
    else:
        p = 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / (len(d) + 1)
    return {"est": float(est), "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975)), "p": float(min(1.0, p))}


def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def analyze(phase2_dir: str, phase2b_dir: str | None = None, T: float = 1.0, seed: int = 0) -> dict:
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet"))], ignore_index=True)
    if phase2b_dir:
        df = pd.concat([df] + [pd.read_parquet(f) for f in sorted(glob.glob(f"{phase2b_dir}/dist_shard*.parquet"))], ignore_index=True)
    df = df[~df.instance_id.isin(pilot_instances())].reset_index(drop=True)
    if T != 1.0:
        df["p"] = temper_col(df, T)
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    subst = [p for p in pcfg if pcfg[p]["stratum"] in SUBST and p != "facilitator" and p in set(df.persona)]
    stratum = {p: pcfg[p]["stratum"] for p in subst}
    K = len(subst)
    strata = {s: [i for i, p in enumerate(subst) if stratum[p] == s] for s in SUBST}
    PO = per_order(df[df.persona.isin(subst)])
    frames = sorted({k[3] for k in PO})
    coordlike = [f for f in ("coord", "coord_d2", "coord_d3") if f in frames]
    insts = {f: sorted({k[2] for k in PO if k[1] == f}) for f in FAMS}
    rng = np.random.default_rng(seed)

    # ---------------- per-family instance arrays
    pair, xframe, mix = {}, {}, {}
    for f in FAMS:
        fr_here = [fr for fr in frames if (subst[0], f, insts[f][0], fr) in PO]
        stk = {fr: [np.stack([PO[(p, f, iid, fr)] for p in subst]) for iid in insts[f]] for fr in fr_here}
        for fr in fr_here:
            pair[(fr, f)] = np.stack([cross_pairs(S, S) for S in stk[fr]])
            mix[(fr, f)] = [S.mean(1) for S in stk[fr]]                               # list over instances of (K, n); n varies in F4
        for fr in coordlike:
            xframe[(fr, f)] = np.stack([cross_pairs(Sc, Sp) for Sc, Sp in zip(stk[fr], stk["pick"])])

    ALL = {f: np.arange(len(insts[f])) for f in FAMS}

    def resample():
        return {f: rng.integers(0, len(insts[f]), len(insts[f])) for f in FAMS}

    def presample():
        return np.concatenate([rng.choice(v, len(v)) for v in strata.values() if v])

    def two_level(fn, side=None):
        est = fn(np.arange(K), ALL)
        return _ci(est, [fn(presample(), resample()) for _ in range(NB)], side)

    def cosm(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        return M / np.outer(d, d)

    def abar(fr, pid, s, fams=FAMS):
        k = len(pid)
        mask = ~np.eye(k, dtype=bool)
        return float(np.mean([cosm(pair[(fr, f)][s[f]].mean(0))[np.ix_(pid, pid)][mask].mean() for f in fams]))

    def retention(fr, s):
        """r_i per persona (K,), pooled over families."""
        vals = []
        for f in FAMS:
            X = xframe[(fr, f)][s[f]].mean(0)
            Cc = pair[(fr, f)][s[f]].mean(0)
            Cp = pair[("pick", f)][s[f]].mean(0)
            vals.append(np.diag(X) / np.sqrt(np.diag(Cc) * np.diag(Cp)))
        return np.mean(vals, 0)

    def selfC(fr, s):
        return np.mean([np.diagonal(pair[(fr, f)][s[f]].mean(0)) for f in FAMS], 0)

    out: dict = {"T": T, "n_personas": K, "substantive": subst, "instances_per_family": {f: len(v) for f, v in insts.items()}}

    # ================= Part A
    F13 = ["F1", "F3"]
    A1 = {fr: two_level(lambda pid, s, fr=fr: abar(fr, pid, s, F13)) for fr in ("pick", "coord", "anti")}
    frac = lambda fr, pid, s: 1 - (1 - abar(fr, pid, s, F13)) / (1 - abar("pick", pid, s, F13))
    A1["frac_removed_coord"] = two_level(lambda pid, s: frac("coord", pid, s))
    A1["frac_removed_anti"] = two_level(lambda pid, s: frac("anti", pid, s))
    A1["anti_minus_pick"] = two_level(lambda pid, s: abar("anti", pid, s, F13) - abar("pick", pid, s, F13))
    A1["coord_minus_anti"] = two_level(lambda pid, s: abar("coord", pid, s, F13) - abar("anti", pid, s, F13))
    A1["gain_coord_minus_gain_anti"] = A1["coord_minus_anti"]  # identical by algebra: (c - p) - (a - p)
    out["A1_anti_control"] = A1

    # A2: shift alignment (instance level)
    align = {}  # f -> (I, K)
    for f in FAMS:
        Mc, Mp, Mg = mix[("coord", f)], mix[("pick", f)], mix[("guess", f)]
        A = np.full((len(Mc), K), np.nan)
        for j in range(len(Mc)):
            for i in range(K):
                A[j, i] = cosv(Mc[j][i] - Mp[j][i], Mg[j][i] - Mp[j][i])
        align[f] = A
    per_p = lambda s: np.mean([np.nanmean(align[f][s[f]], 0) for f in FAMS], 0)
    est_pp = per_p(ALL)
    dr_pp = np.array([per_p(resample()) for _ in range(NB)])
    out["A2_shift_alignment"] = {
        "population": two_level(lambda pid, s: per_p(s)[pid].mean()),
        "per_persona": {p: {"est": float(est_pp[i]), "lo": float(np.quantile(dr_pp[:, i], .025)), "hi": float(np.quantile(dr_pp[:, i], .975))}
                        for i, p in enumerate(subst)},
    }

    # A3: guess homogeneity
    out["A3_guess_homogeneity"] = {fr: two_level(lambda pid, s, fr=fr: abar(fr, pid, s)) for fr in ("pick", "coord", "guess")}
    out["A3_guess_homogeneity"]["guess_minus_coord"] = two_level(lambda pid, s: abar("guess", pid, s) - abar("coord", pid, s))
    out["A3_guess_homogeneity"]["guess_minus_pick"] = two_level(lambda pid, s: abar("guess", pid, s) - abar("pick", pid, s))

    # A4: retention baseline
    r_est = retention("coord", ALL)
    r_dr = np.array([retention("coord", resample()) for _ in range(NB)])
    out["A4_retention"] = {"population": two_level(lambda pid, s: retention("coord", s)[pid].mean()),
                           "per_persona": {p: {"est": float(r_est[i]), "lo": float(np.quantile(r_dr[:, i], .025)),
                                               "hi": float(np.quantile(r_dr[:, i], .975))} for i, p in enumerate(subst)}}

    # ================= Part B
    if "coord_d3" in frames or "coord_d2" in frames:
        gbar = {f: [M.mean(0) for M in mix[("guess", f)]] for f in FAMS}             # per instance: population guess

        def d_stat(fr, s):
            vals = []
            for f in FAMS:
                M = [mix[(fr, f)][j] for j in s[f]]
                Mp = [mix[("pick", f)][j] for j in s[f]]
                G = [gbar[f][j] for j in s[f]]
                D = np.array([[cosv(M[j][i], Mp[j][i]) - cosv(M[j][i], G[j]) for i in range(K)] for j in range(len(M))])
                vals.append(D.mean(0))
            return np.mean(vals, 0)

        beta = {}
        for fr in coordlike:
            beta[fr] = {f: np.array([[stats.tempered_fit(mix[("pick", f)][j][i], mix[(fr, f)][j][i])["beta"] for i in range(K)]
                                     for j in range(len(insts[f]))]) for f in FAMS}

        def med_beta(fr, pid, s):
            return float(np.median(np.concatenate([beta[fr][f][s[f]][:, pid].ravel() for f in FAMS])))

        B = {"reference_D1": {
            "Abar": two_level(lambda pid, s: abar("coord", pid, s)),
            "retention_mean": two_level(lambda pid, s: retention("coord", s)[pid].mean()),
            "S_mean": two_level(lambda pid, s: (selfC("coord", s) - selfC("pick", s))[pid].mean()),
            "median_beta": two_level(lambda pid, s: med_beta("coord", pid, s) - 1),
            "d_mean": two_level(lambda pid, s: d_stat("coord", s)[pid].mean())}}
        for fr, tag, side in (("coord_d3", "D3", "directional"), ("coord_d2", "D2", "two")):
            if fr not in frames:
                continue
            sd = lambda dirn: (dirn if side == "directional" else None)
            res = {
                "a1_dAbar": two_level(lambda pid, s: abar(fr, pid, s) - abar("coord", pid, s), sd("less")),
                "a2_dRetention": two_level(lambda pid, s: (retention(fr, s) - retention("coord", s))[pid].mean(), sd("greater")),
                "b1_dS": two_level(lambda pid, s: (selfC(fr, s) - selfC("coord", s))[pid].mean(), sd("greater")),
                "b2_median_beta_minus_1": two_level(lambda pid, s: med_beta(fr, pid, s) - 1, sd("greater")),
                "c_d_mean": two_level(lambda pid, s: d_stat(fr, s)[pid].mean(), sd("greater")),
                "Abar": two_level(lambda pid, s: abar(fr, pid, s)),
                "retention_mean": two_level(lambda pid, s: retention(fr, s)[pid].mean()),
                "S_mean": two_level(lambda pid, s: (selfC(fr, s) - selfC("pick", s))[pid].mean()),
            }
            r_fr = retention(fr, ALL)
            res["per_persona"] = {p: {"retention": float(r_fr[i]), "retention_D1": float(r_est[i]),
                                      "S": float((selfC(fr, ALL) - selfC("pick", ALL))[i]),
                                      "S_D1": float((selfC("coord", ALL) - selfC("pick", ALL))[i])} for i, p in enumerate(subst)}
            B[tag] = res
        tests = {f"{tag} {k}": B[tag][k]["p"] for tag in ("D3", "D2") if tag in B
                 for k in ("a1_dAbar", "a2_dRetention", "b1_dS", "b2_median_beta_minus_1", "c_d_mean")}
        adj = stats.holm(list(tests.values()))
        B["holm"] = {k: {"p": float(v), "p_holm": float(a)} for (k, v), a in zip(tests.items(), adj)}
        if "D3" in B:
            ok = {k: B["holm"][f"D3 {k}"]["p_holm"] < 0.05 for k in ("a1_dAbar", "a2_dRetention", "b1_dS", "b2_median_beta_minus_1", "c_d_mean")}
            restore = ok["a1_dAbar"] and ok["a2_dRetention"]
            sharpen = ok["b1_dS"] or ok["b2_median_beta_minus_1"]
            B["mechanism"] = ("strategic persona-dropping (D3 restores personas)" if restore else
                              "mixed (D3 sharpens but personas do not return)" if sharpen else
                              "fixed reflex of the coordination framing" if not any(ok.values()) else "other: see tests")
            B["passed"] = ok
        out["partB"] = B
    return out
