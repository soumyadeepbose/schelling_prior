"""PREREG Amendment 9, Run 1: persona-relevant pricing with 4 option orders.

Distributions are in price order (the readout's canonical order). Per agent x scenario there are 16 cells (4 orders x
2 persona x 2 frame paraphrases); C_ij = mean over cells s != s' of <p_is, p_js'> (the Phase 2 estimator), per scenario,
then averaged over scenarios. Centred cosine uses C - 1/n. Bootstrap: all 9 agents = agents within groups (pricing /
Assistant / others) and scenarios; the 3 pricing agents = scenarios only.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .analysis_amend3 import _ci, cosv
from .analysis_amend5 import COLS
from .analysis_amend7 import WAuc
from .analysis_amend8b import tost
from .analysis_phase2 import NB, cross_pairs
from .pricing import A9_FRAMES, load_cfg

GOAL_FRAMES = ["coord_ai", "coord_human", "goal_nonsocial", "compete_ai"]


def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def _spearman3(a, b):
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze(run_dir: str, ghat: np.ndarray, seed: int = 0) -> dict:
    agents = load_cfg()["agents"]
    files = sorted(glob.glob(f"{run_dir}/dist_shard*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=COLS + ["p", "offmenu_mass"]) for f in files], ignore_index=True)
    has_acts = all(os.path.exists(f[:-8] + "_l32.npy") for f in files)
    acts = np.concatenate([np.load(f[:-8] + "_l32.npy") for f in files]).astype(np.float32) if has_acts else None
    sids = sorted(df.instance_id.unique())
    K, S = len(agents), len(sids)
    aidx, sidx = {a: i for i, a in enumerate(agents)}, {s: j for j, s in enumerate(sids)}
    n = len(df.p.iloc[0])
    P = {f: np.full((S, K, 16, n), np.nan) for f in A9_FRAMES}
    for f in A9_FRAMES:
        m = (df.frame == f).to_numpy()
        d = df[m]
        cell = (d.order_id * 4 + d.persona_par * 2 + d.frame_par).to_numpy()
        P[f][d.instance_id.map(sidx).to_numpy(), d.persona.map(aidx).to_numpy(), cell] = np.stack(d.p.to_numpy())
    assert all(not np.isnan(P[f]).any() for f in A9_FRAMES), "missing cells"
    Pbar = {f: P[f].mean(2) for f in A9_FRAMES}                                     # (S, K, n)
    Cs = {f: np.stack([cross_pairs(P[f][j], P[f][j]) for j in range(S)]) for f in A9_FRAMES}

    pricing3 = [aidx[a] for a in agents if a.startswith("pricing_")]
    groups = [pricing3, [aidx["assistant"]], [aidx[a] for a in agents if not a.startswith("pricing_") and a != "assistant"]]
    allK, allS = np.arange(K), np.arange(S)
    rng = np.random.default_rng(seed)
    draws_all = [(np.concatenate([rng.choice(g, len(g)) for g in groups]), rng.integers(0, S, S)) for _ in range(NB)]
    draws_p3 = [(np.array(pricing3), rng.integers(0, S, S)) for _ in range(NB)]

    def boot(fn, side=None, p3=False):
        est = fn(np.array(pricing3) if p3 else allK, allS)
        return _ci(est, [fn(pid, s) for pid, s in (draws_p3 if p3 else draws_all)], side)

    def offd(M, pid):
        M = M[np.ix_(pid, pid)]
        k = len(pid)
        off = ~np.eye(k, dtype=bool)
        # duplicate agents in a bootstrap draw: drop pairs of an agent with itself
        same = pid[:, None] == pid[None, :]
        return float(M[off & ~same].mean())

    def cosm(M, centred=False):
        M = M - 1.0 / n if centred else M
        d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
        return M / np.outer(d, d)

    Cm = lambda f, s: Cs[f][s].mean(0)
    Phi = lambda f, pid, s: offd(Cm(f, s), pid)
    Abar = lambda f, pid, s: offd(cosm(Cm(f, s)), pid)
    Acen = lambda f, pid, s: offd(cosm(Cm(f, s), True), pid)
    frac = lambda f, pid, s, A=Abar: 1 - (1 - A(f, pid, s)) / (1 - A("pick", pid, s))

    def top_agree(f, pid, s):
        am = Pbar[f][s].argmax(-1)[:, pid]                                            # (S', k)
        same = (am[:, :, None] == am[:, None, :])
        k = len(pid)
        mask = ~np.eye(k, dtype=bool) & ~(pid[:, None] == pid[None, :])
        return float(same[:, mask].mean())

    tiers = np.arange(1, n + 1)
    tier = {f: Pbar[f] @ tiers for f in A9_FRAMES}                                  # (S, K)
    mtier = lambda f, pid, s: float(tier[f][s].mean(0)[pid].mean())

    def decomp(f, pid, s):
        Cp, Cc = Cm("pick", s)[np.ix_(pid, pid)], Cm(f, s)[np.ix_(pid, pid)]
        k = len(pid)
        mask = ~np.eye(k, dtype=bool) & ~(pid[:, None] == pid[None, :])
        dd = 0.5 * (np.log(np.diag(Cc)) - np.log(np.diag(Cp)))
        D = dd[:, None] + dd[None, :]
        agr = np.log(cosm(Cc)) - np.log(cosm(Cp))
        return np.array([(np.log(Cc) - np.log(Cp))[mask].mean(), D[mask].mean(), agr[mask].mean()])

    def boot_vec(fn, p3=False):
        est = fn(np.array(pricing3) if p3 else allK, allS)
        dr = np.array([fn(pid, s) for pid, s in (draws_p3 if p3 else draws_all)])
        return {name: _ci(est[i], dr[:, i]) for i, name in enumerate(["dlogC", "decisiveness", "agreement"])}

    align = {f: np.array([[cosv(Pbar[f][j, i] - Pbar["pick"][j, i], Pbar["guess_ai"][j, i] - Pbar["pick"][j, i])
                           for i in range(K)] for j in range(S)]) for f in A9_FRAMES if f not in ("pick", "guess_ai")}
    malign = lambda f, pid, s: float(np.nanmean(align[f][s], 0)[pid].mean())

    out: dict = {"agents": agents, "n_scenarios": S, "n_tiers": n, "n_prompts": len(df),
                 "offmenu_mean": float(df.offmenu_mass.mean()), "offmenu_max": float(df.offmenu_mass.max()),
                 "per_frame": {}}
    for f in A9_FRAMES:
        r = {}
        for tag, p3 in (("all9", False), ("pricing3", True)):
            q = {"Abar": boot(lambda pid, s: Abar(f, pid, s), p3=p3),
                 "Abar_centred": boot(lambda pid, s: Acen(f, pid, s), p3=p3),
                 "Phi": boot(lambda pid, s: Phi(f, pid, s), p3=p3),
                 "top_choice_agreement": boot(lambda pid, s: top_agree(f, pid, s), p3=p3),
                 "mean_tier": boot(lambda pid, s: mtier(f, pid, s), p3=p3)}
            if f != "pick":
                q["frac_removed_vs_pick"] = boot(lambda pid, s: frac(f, pid, s), p3=p3)
                q["frac_removed_vs_pick_centred"] = boot(lambda pid, s: frac(f, pid, s, Acen), p3=p3)
                q["decomposition_vs_pick"] = boot_vec(lambda pid, s: decomp(f, pid, s), p3=p3)
                q["dTier_vs_pick"] = boot(lambda pid, s: mtier(f, pid, s) - mtier("pick", pid, s), p3=p3)
            if f in align:
                q["alignment_to_guess"] = boot(lambda pid, s: malign(f, pid, s), p3=p3)
            r[tag] = q
        r["per_agent_tier"] = {a: float(tier[f].mean(0)[i]) for i, a in enumerate(agents)}
        r["per_agent_tier_by_order"] = {a: [float((P[f][:, i, o * 4:(o + 1) * 4].mean(1) @ tiers).mean()) for o in range(4)]
                                        for i, a in enumerate(agents)}
        out["per_frame"][f] = r

    # ---------------- confirmatory tests
    F = lambda pid, s: frac("coord_ai", pid, s)
    estF = F(np.array(pricing3), allS)
    dF = np.array([F(pid, s) for pid, s in draws_p3])
    tests: dict = {"P1a": {"est": float(estF), "lo": float(np.quantile(dF, .025)), "hi": float(np.quantile(dF, .975)),
                     "p": float(min(1, (dF <= 0.50).mean() + 1 / (NB + 1)))},
             "P1b": {"est": float(estF), "lo": float(np.quantile(dF, .025)), "hi": float(np.quantile(dF, .975)),
                     "p": float(min(1, (dF >= 0.25).mean() + 1 / (NB + 1)))}}
    tests["P1_reading"] = ("goal framing collapses persona-relevant diversity" if estF >= 0.50 else
                           "persona-relevant diversity is retained" if estF <= 0.25 else "partial")

    p3 = np.array(pricing3)
    rho = lambda f, s: _spearman3(tier["pick"][s].mean(0)[p3], tier[f][s].mean(0)[p3])
    est2 = np.mean([rho(f, allS) for f in GOAL_FRAMES])
    d2 = np.array([np.mean([rho(f, s) for f in GOAL_FRAMES]) for _, s in draws_p3])
    order_ok = lambda s: bool(tier["pick"][s].mean(0)[aidx["pricing_premium"]] > tier["pick"][s].mean(0)[aidx["pricing_sustainable"]]
                              > tier["pick"][s].mean(0)[aidx["pricing_discount"]])
    tests["P2"] = {**_ci(est2, d2, "greater"), "per_frame_rho": {f: rho(f, allS) for f in GOAL_FRAMES},
                   "pick_order_premium_gt_sustainable_gt_discount": order_ok(allS),
                   "share_draws_pick_order": float(np.mean([order_ok(s) for _, s in draws_p3])),
                   "pricing_tiers_pick": {agents[i]: float(tier["pick"].mean(0)[i]) for i in pricing3},
                   "spearman_all9": {f: float(np.corrcoef(_rank(tier["pick"].mean(0)), _rank(tier[f].mean(0)))[0, 1])
                                     for f in GOAL_FRAMES}}
    d3 = lambda pid, s: Abar("coord_ai", pid, s) - Abar("coord_human", pid, s)
    tests["P3"] = tost(d3(allK, allS), [d3(pid, s) for pid, s in draws_all])
    d4 = lambda pid, s: mtier("compete_ai", pid, s) - mtier("pick", pid, s)
    tests["P4"] = {**_ci(d4(allK, allS), [d4(pid, s) for pid, s in draws_all]),
                   "by_order": [float(np.mean([(P["compete_ai"][:, i, o * 4:(o + 1) * 4].mean(1) @ tiers).mean()
                                               - (P["pick"][:, i, o * 4:(o + 1) * 4].mean(1) @ tiers).mean() for i in range(K)]))
                                for o in range(4)],
                   "per_agent": {a: float(tier["compete_ai"].mean(0)[i] - tier["pick"].mean(0)[i]) for i, a in enumerate(agents)}}
    out["tests"] = tests

    # ---------------- P5: g-hat AUROC per frame vs pick (pooled cells), random-direction 99th percentile
    if acts is None:
        out["P5"] = "not computed: layer-32 residuals missing"
        return out
    g = ghat / np.linalg.norm(ghat)
    R = np.random.default_rng(9).standard_normal((100, acts.shape[1])).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    P5 = {}
    for f in A9_FRAMES[1:]:
        m = df.frame.isin((f, "pick")).to_numpy()
        y = (df.frame[m] == f).to_numpy()
        X = acts[m]
        a = WAuc(X @ g, y)()
        rs = [WAuc(X @ R[k], y)() for k in range(100)]
        rs = [max(x, 1 - x) for x in rs]
        P5[f] = {"auroc_ghat": float(a), "random_q99": float(np.quantile(rs, .99)), "random_median": float(np.median(rs))}
    out["P5"] = P5
    return out
