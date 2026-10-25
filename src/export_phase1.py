"""Export Phase 1 per-instance quantities for figures (reads cached distributions only).

uv run python -m src.export_phase1 results/qwen3-32b/phase1/dist.parquet OUT.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import prompts, stats
from .analysis_phase1 import analyze, c00_permuted, c00_shared, instance_tensors, kl


def export(path: str) -> dict:
    df = pd.read_parquet(path)
    T = instance_tensors(df)
    rep = analyze(path)
    human = {}
    for fam in ("F1", "F4"):
        for g in prompts.load_games(fam):
            if "human" in g:
                human[g["instance_id"]] = g
    inst = []
    for (fam, iid, frame), v in T.items():
        if frame != "pick":
            continue
        row = {"family": fam, "id": iid, "options": v["options"], "n": len(v["options"])}
        for f in ("pick", "guess", "coord", "anti"):
            if (fam, iid, f) in T:
                arr = T[(fam, iid, f)]["p"]
                c, pbar = c00_permuted(arr)
                row[f] = {"C00": c, "C00_shared": c00_shared(arr), "p": pbar.round(5).tolist(),
                          "omega": float(np.nanmean(T[(fam, iid, f)]["omega"]))}
        tf = stats.tempered_fit(np.array(row["pick"]["p"]), np.array(row["coord"]["p"]))
        row["beta"], row["rel"] = tf["beta"], tf["rel"]
        if "guess" in row:
            pc = np.array(row["coord"]["p"])
            row["kl_coord_pick"] = kl(pc, np.array(row["pick"]["p"]))
            row["kl_coord_guess"] = kl(pc, np.array(row["guess"]["p"]))
        if iid in human:
            h = human[iid]["human"]
            row["human"] = {}
            for f, key in (("pick", "pick"), ("guess", "guess"), ("coord", "coordinate")):
                c = np.array(h[key], float)
                q = c / c.sum()
                # Mehta-Starmer-Sugden index: chance two distinct respondents match (without replacement)
                N = c.sum()
                row["human"][f] = {"p": q.round(4).tolist(), "C00": float(((c * (c - 1)).sum()) / (N * (N - 1))),
                                   "top": human[iid]["options"][int(c.argmax())]}
            row["source"] = human[iid]["source"]
        inst.append(row)
    return {"report": rep, "instances": inst}


if __name__ == "__main__":
    out = export(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, default=float)
    print("instances", len(out["instances"]), "G1", out["report"]["G1"])
