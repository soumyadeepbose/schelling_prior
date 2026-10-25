"""PREREG Amendment 4: anti-frame shift alignment (A4-1, A4-2) and centred retention (A4-3).
Same data, substantive set, estimators and bootstrap as Amendment 3. Reads cached distributions only."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import yaml

from . import stats
from .analysis_amend3 import SUBST, _ci, cosv
from .analysis_phase2 import FAMS, NB, cross_pairs, per_order, pilot_instances, temper_col
from .prompts import ROOT


def analyze(phase2_dir: str, phase2b_dir: str, T: float = 1.0, seed: int = 0) -> dict:
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet"))
                    + sorted(glob.glob(f"{phase2b_dir}/dist_shard*.parquet"))], ignore_index=True)
    df = df[~df.instance_id.isin(pilot_instances())].reset_index(drop=True)
    if T != 1.0:
        df["p"] = temper_col(df, T)
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    subst = [p for p in pcfg if pcfg[p]["stratum"] in SUBST and p != "facilitator" and p in set(df.persona)]
    stratum = {p: pcfg[p]["stratum"] for p in subst}
    K = len(subst)
    strata = {s: [i for i, p in enumerate(subst) if stratum[p] == s] for s in SUBST}
    PO = per_order(df[df.persona.isin(subst)])
    insts = {f: sorted({k[2] for k in PO if k[1] == f}) for f in FAMS}
    rng = np.random.default_rng(seed)

    def resample(fams):
        return {f: rng.integers(0, len(insts[f]), len(insts[f])) for f in fams}

    def presample():
        return np.concatenate([rng.choice(v, len(v)) for v in strata.values() if v])

    def two_level(fn, fams, side=None):
        est = fn(np.arange(K), {f: np.arange(len(insts[f])) for f in fams})
        return _ci(est, [fn(presample(), resample(fams)) for _ in range(NB)], side)

    # ---------------- A4-1 / A4-2: shift alignment on F1, F3 (instance level, mixed distributions)
    F13 = ["F1", "F3"]
    align = {"anti": {}, "coord": {}}
    for f in F13:
        for fr in align:
            A = np.full((len(insts[f]), K), np.nan)
            for j, iid in enumerate(insts[f]):
                for i, p in enumerate(subst):
                    pp, pg, px = (PO[(p, f, iid, x)].mean(0) for x in ("pick", "guess", fr))
                    A[j, i] = cosv(px - pp, pg - pp)
            align[fr][f] = A

    def per_p(fr, s):
        return np.mean([np.nanmean(align[fr][f][s[f]], 0) for f in F13], 0)

    out: dict = {"T": T}
    est_a = per_p("anti", {f: np.arange(len(insts[f])) for f in F13})
    dr_a = np.array([per_p("anti", resample(F13)) for _ in range(NB)])
    est_c = per_p("coord", {f: np.arange(len(insts[f])) for f in F13})
    out["A4_1_anti_alignment"] = {
        "population": two_level(lambda pid, s: per_p("anti", s)[pid].mean(), F13, "less"),
        "per_persona": {p: {"est": float(est_a[i]), "lo": float(np.quantile(dr_a[:, i], .025)),
                            "hi": float(np.quantile(dr_a[:, i], .975)), "coord": float(est_c[i])} for i, p in enumerate(subst)},
    }
    out["A4_2_anti_minus_coord"] = two_level(lambda pid, s: (per_p("anti", s) - per_p("coord", s))[pid].mean(), F13, "less")
    out["coord_alignment_F1F3"] = two_level(lambda pid, s: per_p("coord", s)[pid].mean(), F13)
    adj = stats.holm([out["A4_1_anti_alignment"]["population"]["p"], out["A4_2_anti_minus_coord"]["p"]])
    out["holm"] = {"A4-1": float(adj[0]), "A4-2": float(adj[1])}

    # ---------------- A4-3: centred retention (cross-order, C - 1/n), pooled over F1-F5
    dframes = [x for x in ("coord", "coord_d2", "coord_d3") if (subst[0], "F1", insts["F1"][0], x) in PO]
    selfc, xc = {}, {}
    for f in FAMS:
        for fr in ["pick"] + dframes:
            vals = []
            for iid in insts[f]:
                S = np.stack([PO[(p, f, iid, fr)] for p in subst])
                vals.append(np.diagonal(cross_pairs(S, S)) - 1.0 / S.shape[-1])
            selfc[(fr, f)] = np.stack(vals)                                       # (I, K)
        for fr in dframes:
            vals = []
            for iid in insts[f]:
                Sd = np.stack([PO[(p, f, iid, fr)] for p in subst])
                Sp = np.stack([PO[(p, f, iid, "pick")] for p in subst])
                vals.append(np.diagonal(cross_pairs(Sd, Sp)) - 1.0 / Sd.shape[-1])
            xc[(fr, f)] = np.stack(vals)

    def ret_c(fr, s):
        return np.mean([xc[(fr, f)][s[f]].mean(0) / np.sqrt(np.clip(selfc[(fr, f)][s[f]].mean(0) * selfc[("pick", f)][s[f]].mean(0), 1e-300, None))
                        for f in FAMS], 0)

    A3: dict = {}
    for fr in dframes:
        A3[fr] = two_level(lambda pid, s, fr=fr: ret_c(fr, s)[pid].mean(), FAMS)
    for fr in ("coord_d2", "coord_d3"):
        if fr in dframes:
            A3[f"{fr}_minus_coord"] = two_level(lambda pid, s, fr=fr: (ret_c(fr, s) - ret_c("coord", s))[pid].mean(), FAMS)
    allI = {f: np.arange(len(insts[f])) for f in FAMS}
    A3["per_persona"] = {p: {fr: float(ret_c(fr, allI)[i]) for fr in dframes} for i, p in enumerate(subst)}
    out["A4_3_centred_retention"] = A3
    return out
