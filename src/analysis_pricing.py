"""Pricing pilot analysis (exploratory, not pre-registered).

C: cross-cell estimator over the 4 paraphrase cells (persona x condition paraphrase), C_ij = mean_{s != s'} <p_is, p_js'>,
per scenario, then mean over scenarios. Per-scenario vectors: pbar = mean over the 4 cells.
Bootstrap: agents resampled within group (pricing agents / Assistant / stratified roles), scenarios resampled; B = 2000.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .analysis_amend3 import _ci, cosv
from .analysis_amend5 import COLS, substantive
from .analysis_phase2 import NB, cross_pairs, pilot_instances
from .pricing import CONDITIONS, load_cfg

COMPETE = ["compete_R0", "compete_R1", "compete_R2", "compete_R3"]


def generic_ai_direction(phase2_dir: str) -> np.ndarray:
    """Amendment 5 g at layer 32: mean (guess - pick) residual over the 23 substantive personas and confirmatory
    instances. The design is balanced, so this equals the mean of the per-instance Delta^guess."""
    subst, _ = substantive()
    pil = pilot_instances()
    acc = {"pick": 0.0, "guess": 0.0}
    cnt = {"pick": 0, "guess": 0}
    for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet")):
        d = pd.read_parquet(f, columns=["persona", "frame", "instance_id"])
        a = np.load(f[:-8] + "_l32.npy", mmap_mode="r")
        for fr in acc:
            rows = np.flatnonzero((d.persona.isin(subst) & ~d.instance_id.isin(pil) & (d.frame == fr)).to_numpy())
            for s in range(0, len(rows), 4096):
                acc[fr] = acc[fr] + np.asarray(a[rows[s: s + 4096]], np.float64).sum(0)
            cnt[fr] += len(rows)
    return acc["guess"] / cnt["guess"] - acc["pick"] / cnt["pick"]


def analyze(pilot_dir: str, g: np.ndarray | None = None, seed: int = 0) -> dict:
    cfg = load_cfg()
    agents = cfg["agents"]
    df = pd.concat([pd.read_parquet(f, columns=COLS + ["p", "offmenu_mass"]) for f in sorted(glob.glob(f"{pilot_dir}/dist_shard*.parquet"))],
                   ignore_index=True)
    acts = np.concatenate([np.load(f[:-8] + "_l32.npy") for f in sorted(glob.glob(f"{pilot_dir}/dist_shard*.parquet"))]).astype(np.float32)
    sids = sorted(df.instance_id.unique())
    K, S = len(agents), len(sids)
    aidx, sidx = {a: i for i, a in enumerate(agents)}, {s: j for j, s in enumerate(sids)}
    n = len(df.p.iloc[0])
    P = {c: np.zeros((S, K, 4, n)) for c in CONDITIONS}        # per-cell distributions
    H = {c: np.zeros((S, K, acts.shape[1])) for c in CONDITIONS}
    for c in CONDITIONS:
        m = (df.frame == c).to_numpy()
        dd = df[m]
        j, i, s = dd.instance_id.map(sidx).to_numpy(), dd.persona.map(aidx).to_numpy(), (dd.persona_par * 2 + dd.frame_par).to_numpy()
        P[c][j, i, s] = np.stack(dd.p.to_numpy())
        np.add.at(H[c], (j, i), acts[m] / 4)
    Pbar = {c: P[c].mean(2) for c in CONDITIONS}                # (S, K, n)
    Cs = {c: np.stack([cross_pairs(P[c][j], P[c][j]) for j in range(S)]) for c in CONDITIONS}   # (S, K, K)

    groups = [[aidx[a] for a in agents if a.startswith("pricing_")], [aidx["assistant"]],
              [aidx[a] for a in agents if not a.startswith("pricing_") and a != "assistant"]]
    pricing3 = groups[0]
    rng = np.random.default_rng(seed)
    allK, allS = np.arange(K), np.arange(S)

    def boot(fn, side=None):
        draws = [fn(np.concatenate([rng.choice(g_, len(g_)) for g_ in groups]), rng.integers(0, S, S)) for _ in range(NB)]
        return _ci(fn(allK, allS), draws, side)

    def cosm(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        return M / np.outer(d, d)

    def offd(M, pid):
        M = M[np.ix_(pid, pid)]
        k = len(pid)
        return (M.sum() - np.trace(M)) / (k * (k - 1))

    Phi = lambda c, pid, s: float(offd(Cs[c][s].mean(0), pid))
    Abar = lambda c, pid, s: float(offd(cosm(Cs[c][s].mean(0)), pid))
    frac = lambda c, ref, pid, s: 1 - (1 - Abar(c, pid, s)) / (1 - Abar(ref, pid, s))

    tiers = np.arange(1, n + 1)
    tier = {c: (Pbar[c] / Pbar[c].sum(-1, keepdims=True)) @ tiers for c in CONDITIONS}       # (S, K)
    align = {c: np.array([[cosv(Pbar[c][j, i] - Pbar["pick"][j, i], Pbar["guess"][j, i] - Pbar["pick"][j, i]) for i in range(K)]
                          for j in range(S)]) for c in CONDITIONS if c not in ("pick", "guess")}
    mtier = lambda c, pid, s: float(tier[c][s].mean(0)[pid].mean())
    malign = lambda c, pid, s: float(np.nanmean(align[c][s], 0)[pid].mean())

    out: dict = {"agents": agents, "n_scenarios": S, "n_tiers": n, "offmenu_mean": float(df.offmenu_mass.mean()),
                 "offmenu_max": float(df.offmenu_mass.max()), "per_condition": {}, "contrasts": {}}
    for c in CONDITIONS:
        r = {"Phi": boot(lambda pid, s: Phi(c, pid, s)), "Abar": boot(lambda pid, s: Abar(c, pid, s)),
             "Abar_pricing3": Abar(c, pricing3, allS), "Phi_pricing3": Phi(c, pricing3, allS),
             "mean_tier": boot(lambda pid, s: mtier(c, pid, s)),
             "per_agent_tier": {a: float(tier[c].mean(0)[i]) for i, a in enumerate(agents)}}
        if c != "pick":
            r["frac_removed_vs_pick"] = boot(lambda pid, s: frac(c, "pick", pid, s))
        if c not in ("pick", "compete_R1"):
            r["frac_removed_vs_R1"] = boot(lambda pid, s: frac(c, "compete_R1", pid, s))
        if c in align:
            r["alignment"] = boot(lambda pid, s: malign(c, pid, s))
            r["per_agent_alignment"] = {a: float(np.nanmean(align[c], 0)[i]) for i, a in enumerate(agents)}
        out["per_condition"][c] = r

    for name, a, b in [("R2-R1", "compete_R2", "compete_R1"), ("R3-R1", "compete_R3", "compete_R1"),
                       ("R3-R2", "compete_R3", "compete_R2"), ("R0-R1", "compete_R0", "compete_R1"), ("coord-pick", "coord", "pick"),
                       ("R1-pick", "compete_R1", "pick")]:
        r = {"dPhi": boot(lambda pid, s: Phi(a, pid, s) - Phi(b, pid, s)),
             "dAbar": boot(lambda pid, s: Abar(a, pid, s) - Abar(b, pid, s)),
             "dTier": boot(lambda pid, s: mtier(a, pid, s) - mtier(b, pid, s))}
        if a in align and b in align:
            r["dAlign"] = boot(lambda pid, s: malign(a, pid, s) - malign(b, pid, s))
        out["contrasts"][name] = r

    # activation projections on the Amendment 5 generic-AI direction (layer 32)
    if g is not None:
        gh = g / np.linalg.norm(g)
        proj = {c: H[c] @ gh for c in CONDITIONS}                                            # (S, K)
        mp = lambda a, b: (lambda pid, s: float((proj[a][s] - proj[b][s]).mean(0)[pid].mean()))
        G = {"g_norm": float(np.linalg.norm(g)),
             "R2-R1": boot(mp("compete_R2", "compete_R1")), "R3-R1": boot(mp("compete_R3", "compete_R1")),
             "R0-R1": boot(mp("compete_R0", "compete_R1"))}
        for c in CONDITIONS[1:]:
            G[f"{c}-pick"] = boot(mp(c, "pick"))
        gp = (H["guess"] - H["pick"]).reshape(-1, H["pick"].shape[-1]).mean(0)
        G["cos_g_pricing_guess_dir"] = cosv(gp, g)
        G["per_agent_R2-R1"] = {a: float((proj["compete_R2"] - proj["compete_R1"]).mean(0)[i]) for i, a in enumerate(agents)}
        out["g_projection"] = G
    return out
