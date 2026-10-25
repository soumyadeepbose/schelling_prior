"""PREREG Amendment 6: causal steering along g (S1 induction, S2 removal) and the layer-53 goal direction (S3, exploratory).

Steered readouts: amend6/steer_shard*.parquet (reuse path, dose 0 included). Guess target: cached Phase 2 readouts of the
same cells (orders 1-2, paraphrase (0,0)). Menu-renormalized distributions throughout.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import stats
from .amend6 import DOSES, N_RANDOM, ORDERS, split
from .analysis_amend3 import SUBST, _ci, cosv
from .analysis_amend5 import COLS, substantive
from .analysis_phase2 import FAMS, NB, cross_pairs

RAND = [f"r{k}" for k in range(N_RANDOM)]


def _softmax(lp):
    x = np.asarray(lp, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def load(a6_dir: str, phase2_dir: str):
    subst, stratum = substantive()
    st = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a6_dir}/steer_shard*.parquet"))], ignore_index=True)
    sp = split()
    insts = {f: [i for i in sp[f]["B"] if i in set(st.instance_id)] for f in FAMS}
    kidx = {p: i for i, p in enumerate(subst)}
    oidx = {o: k for k, o in enumerate(ORDERS)}
    D: dict = {}
    for key, g in st.groupby(["frame", "part", "direction", "dose"]):
        for f in FAMS:
            gf = g[g.family == f]
            arrs = []
            for iid in insts[f]:
                gi = gf[gf.instance_id == iid]
                n = len(gi.logprobs.iloc[0])
                A = np.full((len(subst), len(ORDERS), n), np.nan)
                for p_, o, lp in zip(gi.persona, gi.order_id, gi.logprobs):
                    A[kidx[p_], oidx[o]] = _softmax(lp)
                assert not np.isnan(A).any(), (key, f, iid)
                arrs.append(A)
            D[(key, f)] = arrs
        D[(key, "omega")] = float(g.offmenu.mean())
    ph = pd.concat([pd.read_parquet(f, columns=COLS + ["p"]) for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet"))],
                   ignore_index=True)
    ph = ph[(ph.frame == "guess") & ph.persona.isin(subst) & ph.order_id.isin(ORDERS) & (ph.persona_par == 0) & (ph.frame_par == 0)]
    G = {}
    for f in FAMS:
        arrs = []
        for iid in insts[f]:
            gi = ph[ph.instance_id == iid]
            n = len(gi.p.iloc[0])
            A = np.full((len(subst), len(ORDERS), n), np.nan)
            for p_, o, p in zip(gi.persona, gi.order_id, gi.p):
                A[kidx[p_], oidx[o]] = np.asarray(p) / np.sum(p)
            assert not np.isnan(A).any()
            arrs.append(A.mean(1))
        G[f] = arrs
    return D, G, insts, subst, stratum


def analyze(a6_dir: str, phase2_dir: str, seed: int = 0) -> dict:
    D, G, insts, subst, stratum = load(a6_dir, phase2_dir)
    K = len(subst)
    nI = {f: len(insts[f]) for f in FAMS}
    strata = [v for v in ([i for i, p in enumerate(subst) if stratum[p] == s] for s in SUBST) if v]
    rng = np.random.default_rng(seed)
    ALL = {f: np.arange(nI[f]) for f in FAMS}
    allK = np.arange(K)

    def key(frame, part, direction, dose):
        return (frame, "dose0", "none", 0.0) if dose == 0 else (frame, part, direction, float(dose))

    pick0, coord0 = key("pick", "", "", 0), key("coord", "", "", 0)
    conds = sorted({k[0] for k in D if k[1] in FAMS})

    # ---------------- per-condition, per-family instance arrays
    C, AL, RET, PB = {}, {}, {}, {}
    for c in conds:
        for f in FAMS:
            C[(c, f)] = np.stack([cross_pairs(S, S) for S in D[(c, f)]])                                    # (I, K, K)
            PB[(c, f)] = [S.mean(1) for S in D[(c, f)]]
    for c in conds:
        for f in FAMS:
            pb0 = PB[(pick0, f)]
            AL[(c, f)] = np.array([[0.0 if c == pick0 else cosv(PB[(c, f)][j][i] - pb0[j][i], G[f][j][i] - pb0[j][i])
                                    for i in range(K)] for j in range(nI[f])])
            if c[0] == "coord":
                x = np.stack([np.diagonal(cross_pairs(Sc, Sp)) - 1 / Sc.shape[-1] for Sc, Sp in zip(D[(c, f)], D[(pick0, f)])])
                sc_ = np.stack([np.diagonal(cross_pairs(S, S)) - 1 / S.shape[-1] for S in D[(c, f)]])
                sp_ = np.stack([np.diagonal(cross_pairs(S, S)) - 1 / S.shape[-1] for S in D[(pick0, f)]])
                RET[(c, f)] = (x, sc_, sp_)

    def cosm(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        return M / np.outer(d, d)

    def offd(M, pid):
        M = M[np.ix_(pid, pid)]
        k = len(pid)
        return (M.sum() - np.trace(M)) / (k * (k - 1))

    def metrics(c, pid, s):
        Ms = {f: C[(c, f)][s[f]].mean(0) for f in FAMS}
        out = {"Abar": float(np.mean([offd(cosm(Ms[f]), pid) for f in FAMS])),
               "Phi": float(np.mean([offd(Ms[f], pid) for f in FAMS])),
               "H2": float(np.mean([-np.log(np.clip(np.diag(Ms[f])[pid], 1e-300, None)).mean() for f in FAMS])),
               "align": float(np.mean([np.nanmean(AL[(c, f)][s[f]], 0) for f in FAMS], 0)[pid].mean())}
        if c[0] == "coord":
            r = []
            for f in FAMS:
                x, sc_, sp_ = (a[s[f]].mean(0) for a in RET[(c, f)])
                r.append(x / np.sqrt(np.clip(sc_ * sp_, 1e-300, None)))
            out["retention"] = float(np.mean(r, 0)[pid].mean())
        return out

    def slope(ds, ys):
        ds, ys = np.asarray(ds, float), np.asarray(ys, float)
        return float(((ds - ds.mean()) * (ys - ys.mean())).sum() / ((ds - ds.mean()) ** 2).sum())

    parts = {"S1": ("pick", "g", +1), "S2": ("coord", "g", -1), "S3": ("pick", "m53", +1)}
    admissible = {}
    for name, (fr, dr, _) in parts.items():
        ok = []
        for d in DOSES:
            if D[(key(fr, name, dr, d), "omega")] > 0.2:
                break
            ok.append(d)
        admissible[name] = ok

    def curves(name, pid, s):
        fr, dr, _ = parts[name]
        ds = [0.0] + admissible[name]
        cg = [metrics(key(fr, name, dr, d), pid, s) for d in ds]
        cr = [[metrics(key(fr, name, rk, d), pid, s) for d in ds] for rk in (RAND if name != "S3" else [])]
        return ds, cg, cr

    def summarize(name, ds, cg, cr, mets):
        o = {}
        for m in mets:
            yg = [x[m] for x in cg]
            o[f"slope_g_{m}"] = slope(ds, yg)
            if cr:
                o[f"slope_r_{m}"] = float(np.mean([slope(ds, [x[m] for x in cur]) for cur in cr]))
                o[f"diff_{m}"] = o[f"slope_g_{m}"] - o[f"slope_r_{m}"]
            o[f"curve_g_{m}"] = yg
            if cr:
                o[f"curve_r_{m}"] = [float(np.mean([cur[j][m] for cur in cr])) for j in range(len(ds))]
        a0 = cg[0]["Abar"]
        o["curve_g_frac"] = [1 - (1 - x["Abar"]) / (1 - a0) for x in cg]
        if cr:
            o["curve_r_frac"] = [float(np.mean([1 - (1 - cur[j]["Abar"]) / (1 - a0) for cur in cr])) for j in range(len(ds))]
        if name == "S1" and cr:   # matched-H2 comparison against the random-mean curve
            hr = np.array(o["curve_r_H2"])
            order = np.argsort(hr)
            for m in ("Abar", "align"):
                yr = np.array(o[f"curve_r_{m}"])
                vals = []
                for j in range(1, len(ds)):
                    h = cg[j]["H2"]
                    vals.append(float(cg[j][m] - np.interp(h, hr[order], yr[order])) if hr.min() <= h <= hr.max() else np.nan)
                o[f"matched_{m}"] = vals
        return o

    mets = {"S1": ["Abar", "align", "Phi", "H2"], "S2": ["retention", "Abar", "align", "Phi", "H2"], "S3": ["Abar", "align", "Phi", "H2"]}
    est = {n: summarize(n, *curves(n, allK, ALL), mets[n]) for n in parts}
    draws = {n: [] for n in parts}
    for _ in range(NB):
        pid = np.concatenate([rng.choice(v, len(v)) for v in strata])
        s = {f: rng.integers(0, nI[f], nI[f]) for f in FAMS}
        for n in parts:
            draws[n].append(summarize(n, *curves(n, pid, s), mets[n]))

    def ci(n, k, side=None):
        dv = [d[k] for d in draws[n]]
        if isinstance(est[n][k], list):
            arr = np.array(dv, float)
            return [{"est": float(e), "lo": float(np.nanquantile(arr[:, j], .025)), "hi": float(np.nanquantile(arr[:, j], .975)),
                     "matchable_share": float(np.mean(~np.isnan(arr[:, j])))} for j, e in enumerate(est[n][k])]
        return _ci(est[n][k], dv, side)

    out: dict = {"admissible_doses": admissible, "u": None, "n_personas": K, "instances_per_family": nI,
                 "omega": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": D[(k, "omega")] for k in conds}}
    for n in parts:
        r = {}
        for k in est[n]:
            if k.startswith("diff_"):
                m = k[5:]
                side = {"S1": {"Abar": "greater", "align": "greater"},
                        "S2": {"retention": "greater", "Abar": "less", "align": "less"}}.get(n, {}).get(m)
                r[k] = ci(n, k, side)
            elif k.startswith("slope_"):
                r[k] = ci(n, k)
            else:
                r[k] = ci(n, k)
        r["doses"] = [0.0] + admissible[n]
        out[n] = r
    tests = [("S1-a Abar", out["S1"]["diff_Abar"]), ("S1-b align", out["S1"]["diff_align"]),
             ("S2-a retention", out["S2"]["diff_retention"]), ("S2-b Abar", out["S2"]["diff_Abar"]), ("S2-c align", out["S2"]["diff_align"])]
    adj = stats.holm([t[1]["p"] for t in tests])
    out["holm"] = {t[0]: float(a) for t, a in zip(tests, adj)}
    # S3 against the unsteered coordinate frame
    out["S3"]["coord_dose0_Abar"] = metrics(coord0, allK, ALL)["Abar"]
    # per-persona: S1 alignment and S2 retention at the highest analysed dose
    for n, m, fr, dr in (("S1", "align", "pick", "g"), ("S2", "retention", "coord", "g")):
        if admissible[n]:
            dmax = admissible[n][-1]
            per = {}
            for i, p in enumerate(subst):
                g_ = metrics(key(fr, n, dr, dmax), np.array([i, i]), ALL)[m]
                r_ = float(np.mean([metrics(key(fr, n, rk, dmax), np.array([i, i]), ALL)[m] for rk in RAND]))
                b_ = metrics(key(fr, n, dr, 0), np.array([i, i]), ALL)[m]
                per[p] = {"dose0": b_, "g": g_, "random": r_}
            out[n]["per_persona_max_dose"] = {"dose": dmax, "values": per}
    return out
