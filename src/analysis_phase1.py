"""Phase 1 analysis: Schelling component of the Assistant, G1 / H1 decision (PREREG.md Part A).

Reads cached distributions only.  uv run python -m src.analysis_phase1 results/qwen3-32b/phase1/dist.parquet
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from . import gram, stats

GATE = ["F1", "F2", "F3", "F4", "F5"]


def instance_tensors(df: pd.DataFrame) -> dict:
    """(family, instance, frame) -> array [order, persona_par, frame_par, n] of choice distributions."""
    out = {}
    for (fam, inst, frame), g in df.groupby(["family", "instance_id", "frame"]):
        O, n = g.order_id.max() + 1, len(g.iloc[0].p)
        arr = np.full((O, 2, 2, n), np.nan)
        om = np.full((O, 2, 2), np.nan)
        for r in g.itertuples():
            arr[r.order_id, r.persona_par, r.frame_par] = r.p
            om[r.order_id, r.persona_par, r.frame_par] = r.offmenu_mass
        out[(fam, inst, frame)] = {"p": arr, "omega": om, "options": list(g.iloc[0].options)}
    return out


def c00_permuted(arr):
    pbar = arr.reshape(-1, arr.shape[-1]).mean(0)
    return float(pbar @ pbar), pbar


def c00_shared(arr):
    per_order = arr.reshape(arr.shape[0], -1, arr.shape[-1]).mean(1)
    return float(np.mean((per_order**2).sum(-1)))


def kl(p, q):
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.clip(q[m], 1e-300, None)))).sum())


def analyze(path: str) -> dict:
    df = pd.read_parquet(path)
    T = instance_tensors(df)
    fams = sorted({k[0] for k in T})
    report: dict = {"families": {}, "omega": {}}
    for fam, g in df.groupby("family"):
        report["omega"][fam] = {f: {"mean": float(x.offmenu_mass.mean()), "frac_gt_0.2": float((x.offmenu_mass > 0.2).mean()),
                                    "max_pruned_bound": float(x.pruned_bound.max())}
                                for f, x in g.groupby("frame")}
    for fam in fams:
        insts = sorted({k[1] for k in T if k[0] == fam})
        frames = sorted({k[2] for k in T if k[0] == fam})
        rows = defaultdict(list)
        for inst in insts:
            pb = {}
            for f in frames:
                arr = T[(fam, inst, f)]["p"]
                c, pbar = c00_permuted(arr)
                pb[f] = pbar
                rows[f"C00_{f}"].append(c)
                rows[f"C00shared_{f}"].append(c00_shared(arr))
                rows[f"top_{f}"].append(T[(fam, inst, f)]["options"][int(pbar.argmax())])
            n = len(pb["pick"])
            rows["chance"].append(1.0 / n)
            tf = stats.tempered_fit(pb["pick"], pb["coord"])
            rows["beta"].append(tf["beta"])
            rows["rel"].append(tf["rel"])
            a = T[(fam, inst, "coord")]["p"]
            pa, pbp = a[:, :, 0].reshape(-1, n).mean(0), a[:, :, 1].reshape(-1, n).mean(0)
            rows["rel_floor"].append(stats.tempered_fit(pbp, pa)["rel"])
            if "guess" in pb:
                rows["h10_kl_pick"].append(kl(pb["coord"], pb["pick"]))
                rows["h10_kl_guess"].append(kl(pb["coord"], pb["guess"]))
            sp = gram.match_split(pb["coord"], pb["pick"])
            rows["agree_pick_coord"].append(float(2 ** sp["agreement"]))
        R = {k: np.array(v) for k, v in rows.items() if k[:4] != "top_"}
        fam_rep = {"n_instances": len(insts), "chance_mean": float(R["chance"].mean())}
        for f in frames:
            fam_rep[f"C00_{f}_mean"] = float(R[f"C00_{f}"].mean())
            fam_rep[f"positional_prior_{f}"] = float((R[f"C00shared_{f}"] - R[f"C00_{f}"]).mean())
        S = R["C00_coord"] - R["C00_pick"]
        fam_rep["S00"] = {k: v for k, v in stats.bootstrap(np.mean, S).items() if k != "draws"}
        if "anti" in frames:
            fam_rep["anti_minus_pick"] = {k: v for k, v in stats.bootstrap(np.mean, R["C00_anti"] - R["C00_pick"]).items() if k != "draws"}
        fam_rep["beta_median"] = float(np.median(R["beta"]))
        fam_rep["rel_median"] = float(np.median(R["rel"]))
        fam_rep["rel_floor_median"] = float(np.median(R["rel_floor"]))
        fam_rep["frac_relocation"] = float((R["rel"] > np.quantile(R["rel_floor"], 0.95)).mean())
        if "h10_kl_pick" in R:
            d = R["h10_kl_pick"] - R["h10_kl_guess"]
            fam_rep["H10_guess_better_by_KL"] = {k: v for k, v in stats.bootstrap(np.mean, d).items() if k != "draws"}
        fam_rep["top_options"] = {f: pd.Series(rows[f"top_{f}"]).value_counts().head(5).to_dict() for f in frames}
        report["families"][fam] = fam_rep

    gate = [f for f in GATE if f in report["families"]]
    F = report["families"]
    c1 = sum(F[f]["S00"]["lo"] > 0 for f in gate)
    c2 = sum(F[f]["C00_coord_mean"] < 0.9 for f in gate)
    c3 = all(F[f]["anti_minus_pick"]["lo"] <= 0 for f in ("F1", "F3") if f in F and "anti_minus_pick" in F[f])
    sat_pick = sum(F[f]["C00_pick_mean"] >= 0.9 for f in gate)
    report["G1"] = {"S00_clears_0_families": c1, "coord_C00_below_0.9_families": c2, "anti_not_sharpening_F1_F3": c3,
                    "pick_saturated_families": sat_pick, "stop_rule_triggered": sat_pick >= 3,
                    "H1_pass": bool(c1 >= 3 and c2 >= 3 and c3)}
    return report


def table(rep: dict) -> str:
    lines = ["| family | chance | C00 pick | C00 guess | C00 coord | C00 anti | S00 [95% CI] | anti-pick [CI] | beta* med | reloc frac | pos. prior (coord) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam, r in rep["families"].items():
        s = r["S00"]
        a = r.get("anti_minus_pick")
        lines.append(f"| {fam} | {r['chance_mean']:.3f} | {r['C00_pick_mean']:.3f} | {r.get('C00_guess_mean', float('nan')):.3f} | "
                     f"{r['C00_coord_mean']:.3f} | {r.get('C00_anti_mean', float('nan')):.3f} | {s['est']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | "
                     + (f"{a['est']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}]" if a else "–")
                     + f" | {r['beta_median']:.2f} | {r['frac_relocation']:.2f} | {r['positional_prior_coord']:+.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = analyze(sys.argv[1])
    print(table(rep))
    print(json.dumps({"G1": rep["G1"], "omega": rep["omega"]}, indent=1))
    out = sys.argv[1].replace("dist.parquet", "analysis.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, default=float)
