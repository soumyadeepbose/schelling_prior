"""PREREG Amendment 6: causal steering along the generic-AI direction g (layer 32) and the late goal direction (layer 53).

Split: per family, the 36 confirmatory instances are shuffled (seed "a6split:<family>"); the first 18 are the
estimation half A (the steering directions come from cached Phase 2 residuals on A), the last 18 the held-out half B,
in shuffled order. The dose unit u is the mean unsteered coord projection on half B (cached data). Steered outcomes are
collected on half B only; the held-out sample of size I is the first I of B.
"""
from __future__ import annotations

import glob
import json
import random

import numpy as np
import pandas as pd

from . import run
from .analysis_amend5 import COLS, substantive
from .analysis_phase2 import FAMS, pilot_instances
from .prompts import load_games

ORDERS = (1, 2)                     # two permuted Phase 2 orders, paraphrase (0, 0)
DOSES = (0.5, 1.0, 1.5, 2.0)        # in units of u (S1, S2) or u53 (S3)
N_RANDOM = 5
H100_PER_MIN = 0.0657               # observed Phase 2 rate ($7.10 / 108 min)


def split() -> dict:
    pil = pilot_instances()
    out = {}
    for fam in FAMS:
        ids = sorted(g["instance_id"] for g in load_games(fam) if g["instance_id"] not in pil)
        random.Random(f"a6split:{fam}").shuffle(ids)
        out[fam] = {"A": ids[:18], "B": ids[18:]}
    return out


def directions(phase2_dir: str) -> tuple[dict, dict]:
    """ghat: unit mean (guess - pick) at layer 32 over the 23 personas and half-A instances (all 16 cells each).
    u: mean held-out (half B) projection of (coord - pick) on ghat. m53hat: unit mean (coord - anti) at layer 53 over
    half-A F1/F3 instances (order 0, paraphrase (0,0), the cells where layer 53 exists); u53: its half-B projection."""
    subst, _ = substantive()
    sp = split()
    half = {iid: h for fam in sp for h in ("A", "B") for iid in sp[fam][h]}
    acc, cnt = {}, {}
    acc53, cnt53 = {}, {}
    norms = []
    for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet")):
        d = pd.read_parquet(f, columns=COLS)
        d["half"] = d.instance_id.map(half)
        ok = d.persona.isin(subst) & d.half.notna()
        a = np.load(f[:-8] + "_l32.npy", mmap_mode="r")
        for (fr, h), g in d[ok & d.frame.isin(("pick", "guess", "coord"))].groupby(["frame", "half"]):
            rows = g.index.to_numpy()
            for s in range(0, len(rows), 4096):
                blk = np.asarray(a[rows[s: s + 4096]], np.float64)
                acc[(fr, h)] = acc.get((fr, h), 0.0) + blk.sum(0)
                if fr == "pick" and s == 0:
                    norms.append(np.linalg.norm(blk, axis=1).mean())
            cnt[(fr, h)] = cnt.get((fr, h), 0) + len(rows)
        z = np.load(f[:-8] + "_extra.npz")
        dd = d.iloc[z["rows"]].reset_index(drop=True)
        m = (dd.persona.isin(subst) & dd.half.notna() & dd.frame.isin(("coord", "anti")) & dd.family.isin(("F1", "F3"))).to_numpy()
        x53 = z["l53"][m].astype(np.float64)
        for (fr, h), g in dd[m].reset_index(drop=True).groupby(["frame", "half"]):
            acc53[(fr, h)] = acc53.get((fr, h), 0.0) + x53[g.index.to_numpy()].sum(0)
            cnt53[(fr, h)] = cnt53.get((fr, h), 0) + len(g)
    mean = lambda k: acc[k] / cnt[k]
    mean53 = lambda k: acc53[k] / cnt53[k]
    gA = mean(("guess", "A")) - mean(("pick", "A"))
    ghat = gA / np.linalg.norm(gA)
    u = float((mean(("coord", "B")) - mean(("pick", "B"))) @ ghat)
    mA = mean53(("coord", "A")) - mean53(("anti", "A"))
    m53hat = mA / np.linalg.norm(mA)
    u53 = float((mean53(("coord", "B")) - mean53(("anti", "B"))) @ m53hat)
    gB = mean(("guess", "B")) - mean(("pick", "B"))
    rnd = np.random.default_rng(6).standard_normal((N_RANDOM, len(ghat)))
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    vec = {"ghat": ghat.astype(np.float32), "m53hat": m53hat.astype(np.float32), "random": rnd.astype(np.float32)}
    info = {"u": u, "u53": u53, "norm_gA": float(np.linalg.norm(gA)), "cos_gA_gB": float(gA @ gB / np.linalg.norm(gA) / np.linalg.norm(gB)),
            "mean_resid_norm_l32_pick": float(np.mean(norms)), "norm_m53A": float(np.linalg.norm(mA)),
            "counts": {f"{k[0]}|{k[1]}": v for k, v in cnt.items()}, "counts53": {f"{k[0]}|{k[1]}": v for k, v in cnt53.items()},
            "split": sp}
    return vec, info


def cells(frames: tuple[str, ...], half: str, n_per_family: int | None = None):
    """Cells for the 23 substantive personas, F1-F5, orders 1-2, paraphrase (0,0), on the first n instances of a half."""
    subst, _ = substantive()
    sp = split()
    keep = {iid for fam in FAMS for iid in sp[fam][half][: n_per_family or 18]}
    sel = run.load_personas(subst)
    cs = run.phase_cells(sel, FAMS, list(frames), n_orders=4, anti_families=())
    return [c for c in cs if c.instance_id in keep and c.order_id in ORDERS and c.persona_par == 0 and c.frame_par == 0]


def predicted_cost(bench: dict, I: int, shards: int = 4) -> dict:
    """Size rule cost model from measured per-prompt costs (seconds): two prompt passes per frame (stage 1: dose 0 and
    the g / goal doses; stage 2: random controls), 29 steered evaluations per pick prompt (1 + 4 g + 4 goal + 20 random)
    and 25 per coord prompt (1 + 4 g + 20 random), plus one model load per shard and the audit."""
    N = 23 * 5 * I * len(ORDERS)
    sec = 0.0
    for fr, D in (("pick", 29), ("coord", 25)):
        sec += N * (2 * bench[fr]["c_prefix"] + D * bench[fr]["c_delta"])
    sec += shards * bench["load_s"] + bench.get("audit_s", 60.0)
    return {"I": I, "prompts_per_frame": N, "seconds": sec, "usd": sec / 60 * H100_PER_MIN}


def save_directions(vec: dict, info: dict, out_dir: str):
    np.savez(f"{out_dir}/directions.npz", **vec)
    with open(f"{out_dir}/directions.json", "w") as f:
        json.dump(info, f, indent=1)
