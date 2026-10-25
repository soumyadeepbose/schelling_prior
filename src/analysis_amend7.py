"""PREREG Amendment 7: the generic-AI direction as a detector of partner framing (cached residuals, CPU only).

Unit of detection: one prompt (cell). Directions from half A (Amendment 6 split); AUROCs on half B, bootstrapped over
half-B instances within family with the directions fixed.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import stats
from .amend6 import split
from .analysis_amend3 import _ci
from .analysis_amend5 import COLS, substantive
from .analysis_phase2 import FAMS, NB
from .pricing import CONDITIONS

FR4 = ("pick", "guess", "coord", "anti")


class WAuc:
    """AUROC with per-cell weights; scores sorted once, each weighting is O(n). Ties are negligible (continuous scores)."""

    def __init__(self, score, pos):
        self.o = np.argsort(score, kind="mergesort")
        self.pos = pos[self.o].astype(bool)

    def __call__(self, w=None):
        ws = np.ones(len(self.o)) if w is None else w[self.o]
        neg = np.where(self.pos, 0.0, ws)
        below = np.cumsum(neg) - neg
        wp = np.where(self.pos, ws, 0.0)
        return float((wp * below).sum() / (wp.sum() * neg.sum()))


def signfree(a):
    return max(a, 1 - a)


def load_proj(phase2_dir: str, a6_dir: str, axis_l32: np.ndarray, seed: int = 7):
    """Pass 1: half-A per-family (guess - pick) means. Pass 2: projections of every relevant cell."""
    subst, _ = substantive()
    sp = split()
    half = {iid: h for f in sp for h in ("A", "B") for iid in sp[f][h]}
    files = sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet"))
    metas = []
    acc = {}
    for fn in files:
        d = pd.read_parquet(fn, columns=COLS)
        d["half"] = d.instance_id.map(half)
        d = d[d.persona.isin(subst) & d.half.notna() & d.frame.isin(FR4)]
        metas.append((fn, d))
        a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        for (f, fr), g in d[(d.half == "A") & d.frame.isin(("pick", "guess"))].groupby(["family", "frame"]):
            rows = g.index.to_numpy()
            s = np.zeros(a.shape[1])
            for k in range(0, len(rows), 4096):
                s += np.asarray(a[rows[k: k + 4096]], np.float64).sum(0)
            n0, s0 = acc.get((f, fr), (0, 0.0))
            acc[(f, fr)] = (n0 + len(rows), s0 + s)
    gf = {f: acc[(f, "guess")][1] / acc[(f, "guess")][0] - acc[(f, "pick")][1] / acc[(f, "pick")][0] for f in FAMS}
    vec = np.load(f"{a6_dir}/directions.npz")
    unit = lambda v: v / np.linalg.norm(v)
    rnd = np.random.default_rng(seed).standard_normal((100, len(vec["ghat"])))
    V = np.vstack([vec["ghat"], unit(axis_l32)] + [unit(np.mean([gf[o] for o in FAMS if o != f], 0)) for f in FAMS]
                  + list(rnd / np.linalg.norm(rnd, axis=1, keepdims=True))).astype(np.float32)
    names = ["ghat", "axis"] + [f"lofo_{f}" for f in FAMS] + [f"rand{k}" for k in range(100)]
    frames, projs = [], []
    for fn, d in metas:
        d = d[d.half == "B"]
        a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        rows = d.index.to_numpy()
        P = np.concatenate([np.asarray(a[rows[k: k + 4096]], np.float32) @ V.T for k in range(0, len(rows), 4096)])
        frames.append(d.reset_index(drop=True))
        projs.append(P)
    meta = pd.concat(frames, ignore_index=True)
    P = np.concatenate(projs)
    vecs = {"ghat": vec["ghat"], "m53hat": vec["m53hat"],
            "cos_lofo_ghat": {f: float(V[2 + i] @ V[0]) for i, f in enumerate(FAMS)}, "cos_axis_ghat": float(V[1] @ V[0])}
    return meta, P, names, vecs


def analyze(phase2_dir: str, a6_dir: str, pricing_dir: str, axis_l32: np.ndarray, seed: int = 0) -> dict:
    meta, P, names, vecs = load_proj(phase2_dir, a6_dir, axis_l32)
    col = {n: i for i, n in enumerate(names)}
    subst, _ = substantive()
    sp = split()
    rng = np.random.default_rng(seed)
    iidx = {iid: (f, j) for f in FAMS for j, iid in enumerate(sp[f]["B"])}
    cell_f = meta.instance_id.map(lambda x: iidx[x][0]).to_numpy()
    cell_j = meta.instance_id.map(lambda x: iidx[x][1]).to_numpy()
    fam_code = {f: k for k, f in enumerate(FAMS)}
    cell_key = np.array([fam_code[f] for f in cell_f]) * 100 + cell_j        # unique instance code
    pos = (meta.frame != "pick").to_numpy()

    def draw_weights():
        w = np.zeros(len(FAMS) * 100)
        for f in FAMS:
            c = np.bincount(rng.integers(0, 18, 18), minlength=18)
            w[fam_code[f] * 100: fam_code[f] * 100 + 18] = c
        return w[cell_key]

    W = [draw_weights() for _ in range(NB)]

    def boot(sel, v, sign_free=False):
        a = WAuc(P[sel, col[v]], pos[sel])
        est = a()
        dr = np.array([a(w[sel]) for w in W])
        if sign_free:
            est, dr = signfree(est), np.maximum(dr, 1 - dr)
        return est, dr

    def ci_thr(est, dr, thr):
        r = _ci(est - thr, dr - thr, "greater")
        return {**r, "est": est, "lo": r["lo"] + thr, "hi": r["hi"] + thr}

    allc = np.ones(len(meta), bool)
    out: dict = {"n_cells_halfB": int(len(meta)), "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()), "vectors": {k: v for k, v in vecs.items() if k.startswith("cos")}}
    e1, d1 = boot(allc, "ghat")
    out["M1a_overall"] = ci_thr(e1, d1, 0.90)
    per_p = {}
    pd_ = {}
    for p in subst:
        sel = (meta.persona == p).to_numpy()
        e, d = boot(sel, "ghat")
        per_p[p] = {"est": e, "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}
        pd_[p] = d
    mins = np.min(np.stack(list(pd_.values())), 0)
    out["M1b_min_persona"] = ci_thr(min(v["est"] for v in per_p.values()), mins, 0.80)
    out["M1_per_persona"] = per_p
    out["M1_per_family"] = {}
    for f in FAMS:
        e, d = boot(cell_f == f, "ghat")
        out["M1_per_family"][f] = {"est": e, "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}
    out["M1_per_frame"] = {}
    for fr in ("coord", "anti", "guess"):
        sel = ((meta.frame == fr) | (meta.frame == "pick")).to_numpy() & (np.isin(cell_f, ["F1", "F3"]) if fr == "anti" else True)
        e, d = boot(sel, "ghat")
        out["M1_per_frame"][fr] = {"est": e, "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}
    # M2
    m2, m2d = {}, []
    for f in FAMS:
        e, d = boot(cell_f == f, f"lofo_{f}")
        m2[f] = {"est": e, "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}
        m2d.append(d)
    out["M2_lofo"] = m2
    out["M2_min_family"] = ci_thr(min(v["est"] for v in m2.values()), np.min(np.stack(m2d), 0), 0.85)
    # M3
    ea, da = boot(allc, "axis", sign_free=True)
    out["M3_axis_auroc"] = {"est": ea, "lo": float(np.quantile(da, .025)), "hi": float(np.quantile(da, .975))}
    out["M3a_ghat_minus_axis"] = _ci(e1 - ea, d1 - da, "greater")
    rand = np.array([signfree(WAuc(P[:, col[f"rand{k}"]], pos)()) for k in range(100)])
    q99 = float(np.quantile(rand, .99))
    out["M3_random"] = {"q99": q99, "max": float(rand.max()), "median": float(np.median(rand)), "values": rand.tolist()}
    out["M3b_ghat_minus_q99"] = _ci(e1 - q99, d1 - q99, "greater")
    tests = [("M1a", out["M1a_overall"]), ("M1b", out["M1b_min_persona"]), ("M2", out["M2_min_family"]),
             ("M3a", out["M3a_ghat_minus_axis"]), ("M3b", out["M3b_ghat_minus_q99"])]
    out["holm"] = {k: float(a) for (k, _), a in zip(tests, stats.holm([t["p"] for _, t in tests]))}

    # M4: coord vs anti
    sel = meta.frame.isin(("coord", "anti")).to_numpy() & np.isin(cell_f, ["F1", "F3"])
    a4 = WAuc(P[sel, col["ghat"]], (meta.frame[sel] == "coord").to_numpy())
    dr = np.array([a4(w[sel]) for w in W])
    out["M4_ghat_coord_vs_anti_all_cells"] = {"est": a4(), "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975))}
    out["M4_layer53"] = m4_layer53(phase2_dir, vecs, sp, subst, rng)

    # M5: pricing
    out["M5_pricing"] = m5_pricing(pricing_dir, vecs["ghat"], rng)
    return out


def m4_layer53(phase2_dir, vecs, sp, subst, rng):
    """Order-0 / paraphrase-(0,0) cells of half B, F1/F3: coord vs anti on <h32, ghat> and <h53, m53hat>."""
    B = {iid for f in ("F1", "F3") for iid in sp[f]["B"]}
    s32, s53, lab, inst, famc = [], [], [], [], []
    for fn in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet")):
        d = pd.read_parquet(fn, columns=COLS)
        z = np.load(fn[:-8] + "_extra.npz")
        a32 = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        rows = z["rows"]
        dd = d.iloc[rows].reset_index(drop=True)
        m = (dd.persona.isin(subst) & dd.instance_id.isin(B) & dd.frame.isin(("coord", "anti"))).to_numpy()
        s53.append(z["l53"][m].astype(np.float32) @ vecs["m53hat"])
        s32.append(np.asarray(a32[rows[m]], np.float32) @ vecs["ghat"])
        lab.append((dd.frame[m] == "coord").to_numpy())
        inst.append(dd.instance_id[m].to_numpy())
        famc.append(dd.family[m].to_numpy())
    s32, s53, lab, inst, famc = map(np.concatenate, (s32, s53, lab, inst, famc))
    ids = sorted(set(inst))
    code = np.array([ids.index(i) for i in inst])
    fam = dict(zip(inst, famc))
    groups = {f: [k for k, i in enumerate(ids) if fam[i] == f] for f in set(fam.values())}
    out = {"n_cells": int(len(lab))}
    for name, s in (("ghat_l32", s32), ("m53hat_l53", s53)):
        a = WAuc(s, lab)
        dr = []
        for _ in range(NB):
            w = np.zeros(len(ids))
            for g in groups.values():
                np.add.at(w, rng.choice(g, len(g)), 1)
            dr.append(a(w[code]))
        out[name] = {"est": a(), "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975))}
    return out


def m5_pricing(pricing_dir, ghat, rng):
    d = pd.concat([pd.read_parquet(f, columns=["persona", "frame", "instance_id", "persona_par", "frame_par"])
                   for f in sorted(glob.glob(f"{pricing_dir}/dist_shard*.parquet"))], ignore_index=True)
    a = np.concatenate([np.load(f[:-8] + "_l32.npy") for f in sorted(glob.glob(f"{pricing_dir}/dist_shard*.parquet"))]).astype(np.float32)
    s = a @ ghat
    sids = sorted(d.instance_id.unique())
    code = d.instance_id.map({x: k for k, x in enumerate(sids)}).to_numpy()
    W = [np.bincount(rng.integers(0, len(sids), len(sids)), minlength=len(sids))[code] for _ in range(NB)]
    out = {}
    for c in [x for x in CONDITIONS if x != "pick"] + ["all"]:
        sel = (d.frame != "pick").to_numpy() | (d.frame == "pick").to_numpy() if c == "all" else d.frame.isin((c, "pick")).to_numpy()
        au = WAuc(s[sel], (d.frame[sel] != "pick").to_numpy())
        dr = [au(w[sel]) for w in W]
        out[c] = {"est": au(), "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975))}
    agents = sorted(d.persona.unique())
    cellmean = d.assign(s=s).groupby(["frame", "persona", "instance_id"]).s.mean()
    for a_, b_ in (("compete_R2", "compete_R0"), ("compete_R3", "compete_R0"), ("compete_R1", "compete_R0")):
        diff = (cellmean[a_] - cellmean[b_]).unstack()            # persona x scenario
        M = diff.loc[agents, sids].to_numpy()
        est = float(M.mean())
        dr = [float(M[:, rng.integers(0, len(sids), len(sids))].mean()) for _ in range(NB)]
        out[f"{a_[-2:]}-{b_[-2:]}"] = _ci(est, dr)
    return out
