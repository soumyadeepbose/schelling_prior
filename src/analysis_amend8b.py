"""PREREG Amendment 8 Part B (Tier B): AI-specificity, length, non-social goal, guess target, g specificity, none persona.

Phase 2 (pick, guess, coord, anti) + Phase 2c (coord_human, anti_human, guess_human, pick_long, goal_nonsocial; none).
Cross-order estimators, per-family instance means pooled over F1-F5, two-level bootstrap (B = 2000).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .amend6 import split
from .analysis_amend3 import SUBST, cosv
from .analysis_amend7 import WAuc
from .analysis_amend8a import COLS, NB, Pop, _ci, per_order_cells
from .analysis_phase2 import FAMS, cross_pairs, pilot_instances, temper_col
from .prompts import ROOT

F13 = ["F1", "F3"]
NEW = ["coord_human", "anti_human", "guess_human", "pick_long", "goal_nonsocial"]


def tost(est, draws, margin=0.03):
    d = np.asarray(draws, float)
    p_lo = (d <= -margin).mean() + 1 / (len(d) + 1)
    p_hi = (d >= margin).mean() + 1 / (len(d) + 1)
    return {"est": float(est), "lo90": float(np.quantile(d, .05)), "hi90": float(np.quantile(d, .95)),
            "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975)), "p": float(min(1.0, max(p_lo, p_hi))),
            "p_lower": float(p_lo), "p_upper": float(p_hi), "margin": margin}


def load(base: str, T: float = 1.0):
    cols = [c for c in COLS if c not in ("options", "shown")]
    dfs = [pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet"))]
    dfs += [pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{base}/phase2c/dist_shard*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    df = df[~df.instance_id.isin(pilot_instances())].reset_index(drop=True)
    if T != 1.0:
        df["p"] = temper_col(df, T)
    return df


def behaviour(base: str, T: float = 1.0, seed: int = 0) -> dict:
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    stratum = {p: pcfg[p]["stratum"] for p in pcfg}
    stratum["none"] = "none"
    subst = [p for p in pcfg if stratum[p] in SUBST and p != "facilitator"]
    df = load(base, T)
    df = df[df.persona.isin(subst + ["none"])]
    PO = per_order_cells(df)
    frames = ["pick", "guess", "coord", "anti"] + NEW
    pop = Pop(PO, subst, stratum, frames, seed=seed)
    K, ALL, allK = pop.K, pop.ALL, np.arange(pop.K)
    A = lambda fr, pid, s, fams=None: pop.abar(fr, pid, s, fams)
    out = {"T": T, "n_personas": K}
    out["Abar"] = {fr: pop.two_level(lambda pid, s, fr=fr: A(fr, pid, s, F13 if fr in ("anti", "anti_human") else None))
                   for fr in frames}
    out["dAbar_vs_pick"] = {fr: pop.two_level(lambda pid, s, fr=fr: A(fr, pid, s) - A("pick", pid, s))
                            for fr in ("coord", "coord_human", "pick_long", "goal_nonsocial", "guess", "guess_human")}
    # B1
    est = A("coord", allK, ALL) - A("coord_human", allK, ALL)
    dr = []
    dr_b2, dr_b3 = [], []
    for _ in range(NB):
        pid, s = pop.presample(), pop.resample()
        ap, ac, ah, al, ag = (A(fr, pid, s) for fr in ("pick", "coord", "coord_human", "pick_long", "goal_nonsocial"))
        dr.append(ac - ah)
        dr_b2.append((al - ap) / (ac - ap))
        dr_b3.append((ag - ap) - (ac - ap))
    ap, ac, al, ag = (A(fr, allK, ALL) for fr in ("pick", "coord", "pick_long", "goal_nonsocial"))
    out["B1_superiority"] = _ci(est, dr, "greater")
    out["B1_TOST"] = tost(est, dr)
    R = (al - ap) / (ac - ap)
    d2 = np.asarray(dr_b2)
    out["B2"] = {"est": float(R), "lo": float(np.quantile(d2, .025)), "hi": float(np.quantile(d2, .975)),
                 "p": float((d2 >= 0.25).mean() + 1 / (NB + 1))}
    out["B3_TOST"] = tost((ag - ap) - (ac - ap), dr_b3)
    # B4 guess-target alignment
    def align(fr, target):
        return {f: np.array([[cosv(pop.mix(fr, f)[j][i] - pop.mix("pick", f)[j][i], pop.mix(target, f)[j][i] - pop.mix("pick", f)[j][i])
                              for i in range(K)] for j in range(len(pop.insts[f]))]) for f in pop.fams_of(fr)}
    al_ = {(fr, t): align(fr, t) for fr in ("coord", "coord_human") for t in ("guess", "guess_human")}
    al_[("anti", "guess")] = align("anti", "guess")
    al_[("anti_human", "guess_human")] = align("anti_human", "guess_human")
    pp = lambda Am, s, fams: np.mean([np.nanmean(Am[f][s[f]], 0) for f in fams], 0)
    out["B4_AI"] = pop.two_level(lambda pid, s: float((pp(al_[("coord", "guess")], s, FAMS) - pp(al_[("coord", "guess_human")], s, FAMS))[pid].mean()), "greater")
    out["B4_human"] = pop.two_level(lambda pid, s: float((pp(al_[("coord_human", "guess_human")], s, FAMS) - pp(al_[("coord_human", "guess")], s, FAMS))[pid].mean()), "greater")
    out["B4_levels"] = {f"{a}->{b}": pop.two_level(lambda pid, s, k=(a, b): float(pp(al_[k], s, FAMS)[pid].mean())) for a, b in al_ if a.startswith("coord")}
    out["anti_human_vs_anti"] = {"Abar_anti_F13": out["Abar"]["anti"], "Abar_anti_human_F13": out["Abar"]["anti_human"],
                                 "alignment_anti_to_AI_guess": pop.two_level(lambda pid, s: float(pp(al_[("anti", "guess")], s, F13)[pid].mean())),
                                 "alignment_anti_human_to_human_guess": pop.two_level(lambda pid, s: float(pp(al_[("anti_human", "guess_human")], s, F13)[pid].mean()))}
    # B6 none persona
    popn = Pop(PO, subst + ["none"], stratum, ["pick", "guess", "coord", "coord_human", "pick_long"], seed=seed + 1)
    n = K
    def cos_to_none(s):
        vals = []
        for f in FAMS:
            M = popn.mat("pick", s, f)
            vals.append(M[:n, n] / np.sqrt(np.diag(M)[:n] * M[n, n]))
        return np.mean(vals, 0)
    Xcp = {f: np.stack([cross_pairs(c, p) for c, p in zip(popn.S[("coord", f)], popn.S[("pick", f)])]) for f in FAMS}
    def g_none(s):
        vals = []
        for f in FAMS:
            Mp, Mc, X = popn.mat("pick", s, f), popn.mat("coord", s, f), Xcp[f][s[f]].mean(0)
            vals.append(X[:n, n] / np.sqrt(np.diag(Mc)[:n] * Mp[n, n]) - Mp[:n, n] / np.sqrt(np.diag(Mp)[:n] * Mp[n, n]))
        return np.mean(vals, 0)
    tn = {f: np.array([[cosv(popn.mix("coord", f)[j][i] - popn.mix("pick", f)[j][i], popn.mix("pick", f)[j][n] - popn.mix("pick", f)[j][i])
                        for i in range(n)] for j in range(len(popn.insts[f]))]) for f in FAMS}
    cn = cos_to_none(popn.ALL)
    gn = g_none(popn.ALL)
    out["B6"] = {"cos_to_none_pick_mean": pop.two_level(lambda pid, s: float(cos_to_none(s)[pid].mean())),
                 "cos_to_none_pick_per_persona": dict(zip(subst, map(float, cn))),
                 "g_none_mean": pop.two_level(lambda pid, s: float(g_none(s)[pid].mean())),
                 "g_none_per_persona": dict(zip(subst, map(float, gn))),
                 "align_toward_none": pop.two_level(lambda pid, s: float(pp(tn, s, FAMS)[pid].mean())),
                 "align_toward_own_guess": pop.two_level(lambda pid, s: float(pp(al_[("coord", "guess")], s, FAMS)[pid].mean())),
                 "toward_none_minus_toward_guess": pop.two_level(lambda pid, s: float((pp(tn, s, FAMS) - pp(al_[("coord", "guess")], s, FAMS))[pid].mean())),
                 "none_C_self": {fr: float(np.mean([np.diag(popn.mat(fr, popn.ALL, f))[n] for f in FAMS])) for fr in ("pick", "guess", "coord", "coord_human", "pick_long")},
                 "none_vs_assistant_pick_cos": float(np.mean([popn.cosm(popn.mat("pick", popn.ALL, f))[n, subst.index("assistant")] for f in FAMS]))}
    return out


def activations(base: str, seed: int = 0) -> dict:
    """B5: Amendment 6 g_A on half-B cells; projections relative to coord (cell-matched, 16-cell means, half B)."""
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    subst = [p for p in pcfg if pcfg[p]["stratum"] in SUBST and p != "facilitator"]
    sp = split()
    half = {iid: h for f in sp for h in ("A", "B") for iid in sp[f][h]}
    fam_of = {iid: f for f in sp for h in ("A", "B") for iid in sp[f][h]}
    g = np.load(f"{base}/amend6/directions.npz")["ghat"]
    cols = ["persona", "frame", "family", "instance_id", "order_id", "persona_par", "frame_par"]
    rows, S = [], []
    for sub in ("phase2", "phase2c"):
        for fn in sorted(glob.glob(f"{base}/{sub}/dist_shard*.parquet")):
            d = pd.read_parquet(fn, columns=cols)
            m = np.array(d.persona.isin(subst) & (d.instance_id.map(half) == "B"), dtype=bool)
            if sub == "phase2":
                m &= d.frame.isin(("pick", "coord", "guess")).to_numpy()
            a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
            S.append(np.asarray(a[np.flatnonzero(m)], np.float32) @ g)
            rows.append(d[m].reset_index(drop=True))
    meta, s = pd.concat(rows, ignore_index=True), np.concatenate(S)
    rng = np.random.default_rng(seed)
    ids = sorted(meta.instance_id.unique())
    ix = {i: k for k, i in enumerate(ids)}
    code = meta.instance_id.map(ix).to_numpy()
    groups = [[ix[i] for i in ids if fam_of[i] == f] for f in FAMS]
    W = []
    for _ in range(NB):
        w = np.zeros(len(ids))
        for gr in groups:
            np.add.at(w, rng.choice(gr, len(gr)), 1)
        W.append(w)
    out = {"auroc_vs_pick": {}}
    for fr in ("coord", "coord_human", "pick_long", "guess", "guess_human", "goal_nonsocial", "anti_human"):
        sel = np.array(meta.frame.isin((fr, "pick")), dtype=bool)
        if fr == "anti_human":
            sel = sel & np.array(meta.family.isin(F13), dtype=bool)
        a = WAuc(s[sel], (meta.frame[sel] == fr).to_numpy())
        dr = [a(w[code][sel]) for w in W]
        out["auroc_vs_pick"][fr] = {"est": a(), "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975))}
    # projections: 16-cell means per persona-instance, minus pick, relative to coord
    cm = meta.assign(s=s).groupby(["frame", "persona", "instance_id"]).s.mean()
    pk = cm["pick"]
    proj = {fr: (cm[fr] - pk.reindex(cm[fr].index)).dropna() for fr in ("coord", "coord_human", "guess", "guess_human", "goal_nonsocial", "pick_long")}
    inst_of = {fr: v.index.get_level_values("instance_id").map(ix).to_numpy() for fr, v in proj.items()}
    res = {}
    for fr in proj:
        est = proj[fr].mean() / proj["coord"].mean()
        dr = []
        for w in W:
            num = np.average(proj[fr].to_numpy(), weights=w[inst_of[fr]] + 1e-12)
            den = np.average(proj["coord"].to_numpy(), weights=w[inst_of["coord"]] + 1e-12)
            dr.append(num / den)
        res[fr] = {"mean_projection": float(proj[fr].mean()), "ratio_to_coord": float(est),
                   "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975))}
    out["projection_relative_to_coord"] = res
    return out


def analyze(base: str) -> dict:
    cache = Path(f"{base}/amend8_partB_behaviour_cache.json")
    if cache.exists():
        beh = json.loads(cache.read_text())
    else:
        beh = {"T1.0": behaviour(base, 1.0), "T0.3": behaviour(base, 0.3)}
        cache.write_text(json.dumps(beh, default=float))
    return {**beh, "B5": activations(base)}
