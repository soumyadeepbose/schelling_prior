"""PREREG Amendment 10 (cached readouts only): A10-1 default-model centrality, A10-2 pricing flattening-robust measures and
brand-gap compression, A10-3 one estimator for the menu headline numbers, A10-4 number checks.

Estimator: one distribution per option order (cells for pricing), mixed over paraphrases; C_ij = mean over o != o' of
<p_io, p_jo'>; instance-mean C per family; mean over families. Duplicate-persona pairs are excluded from pairwise means.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import yaml

from .analysis_amend3 import SUBST
from .analysis_amend8a import NB, Pop, _ci, per_order_cells
from .analysis_amend8b import F13, load
from .analysis_phase2 import FAMS, cross, cross_pairs, pilot_instances, spearman
from .pricing import A9_FRAMES, load_cfg
from .prompts import ROOT, load_games

NONE_FRAMES = ["pick", "guess", "coord", "coord_human"]
A10_1_FRAMES = ["coord", "coord_human", "guess", "guess_human", "goal_nonsocial", "anti", "anti_human"]


def _onehot(v):
    return np.eye(len(v))[int(np.argmax(v))]


def _holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for r, (k, p) in enumerate(items):
        run = max(run, min(1.0, p * (len(items) - r)))
        out[k] = run
    return out


class Est:
    """C_B measures on a Pop, duplicate-persona pairs excluded."""

    def __init__(self, pop):
        self.pop = pop
        self._am, self._inv = {}, {}

    @staticmethod
    def fams(fr):
        return F13 if fr in ("anti", "anti_human") else FAMS

    def _mask(self, pid):
        return ~np.eye(len(pid), dtype=bool) & (pid[:, None] != pid[None, :])

    def offd(self, M, pid):
        return float(M[np.ix_(pid, pid)][self._mask(pid)].mean())

    def abar(self, fr, pid, s, fams=None):
        return float(np.mean([self.offd(self.pop.cosm(self.pop.mat(fr, s, f)), pid) for f in fams or self.fams(fr)]))

    def phi(self, fr, pid, s, fams=None):
        return float(np.mean([self.offd(self.pop.mat(fr, s, f), pid) for f in fams or self.fams(fr)]))

    def acen(self, fr, pid, s, fams=None):
        vals = []
        for f in fams or self.fams(fr):
            if (fr, f) not in self._inv:
                self._inv[(fr, f)] = np.array([1.0 / x.shape[-1] for x in self.pop.S[(fr, f)]])
            M = (self.pop.C[(fr, f)][s[f]] - self._inv[(fr, f)][s[f]][:, None, None]).mean(0)
            vals.append(self.offd(self.pop.cosm(M), pid))
        return float(np.mean(vals))

    def top(self, fr, pid, s, fams=None):
        vals = []
        for f in fams or self.fams(fr):
            if (fr, f) not in self._am:
                self._am[(fr, f)] = np.stack([x.argmax(-1) for x in self.pop.mix(fr, f)])
            am = self._am[(fr, f)][s[f]][:, pid]
            vals.append((am[:, :, None] == am[:, None, :])[:, self._mask(pid)].mean())
        return float(np.mean(vals))


def _centrality(pop, fr, pid, s, agents, fams):
    """t_j for each agent index in `agents`; q-bar over the resampled personas pid minus every copy of j."""
    K = pop.K
    w = np.bincount(pid, minlength=K).astype(float)
    W = np.repeat(w[None, :], len(agents), 0)
    for a, j in enumerate(agents):
        if j < len(w):
            W[a, j] = 0.0
    n = W.sum(1)
    t = np.zeros(len(agents))
    for f in fams:
        M = pop.mat(fr, s, f)
        Mw = M[:len(w), :len(w)]
        num = np.array([W[a] @ M[:len(w), j] for a, j in enumerate(agents)]) / n
        qq = np.einsum("ak,kl,al->a", W, Mw, W) / n ** 2
        jj = np.array([M[j, j] for j in agents])
        t += num / np.sqrt(qq * jj)
    return t / len(fams)


def a10_1(PO, subst, stratum, seed):
    out = {}
    popN = Pop(PO, subst + ["none"], stratum, NONE_FRAMES, seed=seed)
    popS = Pop({k: v for k, v in PO.items() if k[0] != "none"}, subst, stratum,
               ["pick", "guess_human", "goal_nonsocial", "anti", "anti_human"], seed=seed + 1)
    ia = subst.index("assistant")
    tests = {}
    for fr in A10_1_FRAMES:
        pop = popN if fr in NONE_FRAMES else popS
        has_none = pop is popN
        fams = F13 if fr in ("anti", "anti_human") else FAMS
        agents = list(range(len(subst))) + ([len(subst)] if has_none else [])
        names = subst + (["none"] if has_none else [])

        def T(pid, s, fr=fr, pop=pop, fams=fams, agents=agents):
            return _centrality(pop, fr, pid, s, agents, fams), _centrality(pop, "pick", pid, s, agents, fams)

        allK = np.arange(len(subst))
        t, tp = T(allK, pop.ALL)
        dt = t - tp
        order = np.argsort(-t)
        rank = {names[a]: int(np.where(order == a)[0][0]) + 1 for a in range(len(names))}
        pct = lambda v, a: float((np.delete(v, a) < v[a]).mean())
        r = {"n_agents": len(names), "t": dict(zip(names, map(float, t))), "dt": dict(zip(names, map(float, dt))),
             "median_t_personas": float(np.median(t[:len(subst)])), "median_dt_personas": float(np.median(dt[:len(subst)])),
             "assistant": {"t": float(t[ia]), "dt": float(dt[ia]), "rank_t": rank["assistant"], "pct_t": pct(t, ia),
                           "rank_dt": int(np.where(np.argsort(-dt) == ia)[0][0]) + 1, "pct_dt": pct(dt, ia)}}
        if has_none:
            jn = len(subst)
            r["none"] = {"t": float(t[jn]), "dt": float(dt[jn]), "rank_t": rank["none"], "pct_t": pct(t, jn),
                         "rank_dt": int(np.where(np.argsort(-dt) == jn)[0][0]) + 1, "pct_dt": pct(dt, jn)}

        def diffs(pid, s):
            tt, ttp = T(pid, s)
            dd = tt - ttp
            oth = np.array([k for k in pid if k != ia])        # resampled multiset of non-Assistant personas
            per = np.array(list(pid))
            res = [tt[ia] - np.median(tt[oth]), dd[ia] - np.median(dd[oth])]
            if has_none:
                res += [tt[len(subst)] - np.median(tt[per]), dd[len(subst)] - np.median(dd[per])]
            return np.array(res)

        est = diffs(allK, pop.ALL)
        dr = np.array([diffs(pop.presample(), pop.resample()) for _ in range(NB)])
        keys = ["assistant_minus_median_t", "assistant_minus_median_dt"] + (["none_minus_median_t", "none_minus_median_dt"] if has_none else [])
        r["differences"] = {k: _ci(est[i], dr[:, i], "greater") for i, k in enumerate(keys)}
        out[fr] = r
    tests["A10-1a"] = out["coord"]["differences"]["assistant_minus_median_t"]
    tests["A10-1b"] = out["coord"]["differences"]["assistant_minus_median_dt"]
    return out, tests


def a10_2(base, seed):
    agents = load_cfg()["agents"]
    files = sorted(glob.glob(f"{base}/amend9/pricing/dist_shard*.parquet"))
    cols = ["persona", "frame", "instance_id", "order_id", "persona_par", "frame_par", "p"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    sids = sorted(df.instance_id.unique())
    S, K = len(sids), len(agents)
    aidx, sidx = {a: i for i, a in enumerate(agents)}, {s: j for j, s in enumerate(sids)}
    n = len(df.p.iloc[0])
    P = {f: np.full((S, K, 16, n), np.nan) for f in A9_FRAMES}
    for f in A9_FRAMES:
        d = df[df.frame == f]
        P[f][d.instance_id.map(sidx).to_numpy(), d.persona.map(aidx).to_numpy(),
             (d.order_id * 4 + d.persona_par * 2 + d.frame_par).to_numpy()] = np.stack(d.p.to_numpy())
    assert all(not np.isnan(P[f]).any() for f in A9_FRAMES), "missing pricing cells"
    G = {f: np.eye(n)[P[f].argmax(-1)] for f in A9_FRAMES}
    C = {f: np.stack([cross_pairs(P[f][j], P[f][j]) for j in range(S)]) for f in A9_FRAMES}
    CG = {f: np.stack([cross_pairs(G[f][j], G[f][j]) for j in range(S)]) for f in A9_FRAMES}
    top = {f: P[f].mean(2).argmax(-1) for f in A9_FRAMES}                          # (S, K)
    tiers = np.arange(1, n + 1)
    tier = {f: P[f].mean(2) @ tiers for f in A9_FRAMES}                             # (S, K)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, S, S) for _ in range(NB)]
    allS = np.arange(S)

    def offd(M, pid):
        M = M[np.ix_(pid, pid)]
        return float(M[~np.eye(len(pid), dtype=bool)].mean())

    def cosm(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
        return M / np.outer(d, d)

    meas = {"Abar": lambda f, pid, s: offd(cosm(C[f][s].mean(0)), pid),
            "Abar_centred": lambda f, pid, s: offd(cosm(C[f][s].mean(0) - 1.0 / n), pid),
            "top_choice_agreement": lambda f, pid, s: float((top[f][s][:, pid][:, :, None] == top[f][s][:, pid][:, None, :])[:, ~np.eye(len(pid), dtype=bool)].mean()),
            "greedy_match": lambda f, pid, s: offd(CG[f][s].mean(0), pid)}

    def boot(fn, side=None):
        return _ci(fn(allS), [fn(s) for s in draws], side)

    out = {"n_products": S, "agents": agents}
    sets = {"pricing3": np.array([aidx[a] for a in agents if a.startswith("pricing_")]), "all9": np.arange(K)}
    for tag, pid in sets.items():
        r = {}
        for f in A9_FRAMES:
            q = {m: boot(lambda s, f=f, m=m: meas[m](f, pid, s)) for m in meas}
            if f != "pick":
                q["frac_removed"] = {m: boot(lambda s, f=f, m=m: 1 - (1 - meas[m](f, pid, s)) / (1 - meas[m]("pick", pid, s))) for m in meas}
            r[f] = q
        out[tag] = r
    ip, idd = aidx["pricing_premium"], aidx["pricing_discount"]
    gap = lambda f, s: float(tier[f][s].mean(0)[ip] - tier[f][s].mean(0)[idd])
    out["tiers"] = {f: {a: boot(lambda s, f=f, i=aidx[a]: float(tier[f][s].mean(0)[i]))
                        for a in agents if a.startswith("pricing_")} for f in A9_FRAMES}
    out["gap_premium_minus_discount"] = {f: boot(lambda s, f=f: gap(f, s)) for f in A9_FRAMES}
    out["gap_frac_removed"] = {f: boot(lambda s, f=f: 1 - gap(f, s) / gap("pick", s)) for f in A9_FRAMES if f != "pick"}
    p3 = sets["pricing3"]
    tests = {"A10-2a": boot(lambda s: 1 - (1 - meas["Abar_centred"]("coord_ai", p3, s)) / (1 - meas["Abar_centred"]("pick", p3, s)), "greater"),
             "A10-2b": boot(lambda s: gap("pick", s) - gap("coord_ai", s), "greater")}
    return out, tests


def a10_3(PO, POg, subst, stratum, seed):
    pop = Pop({k: v for k, v in PO.items() if k[0] in subst}, subst, stratum, ["pick", "coord", "anti", "anti_human"], seed=seed + 3)
    popg = Pop({k: v for k, v in POg.items() if k[0] in subst}, subst, stratum, ["pick", "coord"], seed=seed + 4)
    e, eg = Est(pop), Est(popg)
    frac = lambda est, m: (lambda pid, s: 1 - (1 - getattr(est, m)("coord", pid, s)) / (1 - getattr(est, m)("pick", pid, s)))
    out = {"coord_frac_removed_F1_F5": {"Abar": pop.two_level(frac(e, "abar")), "Abar_centred": pop.two_level(frac(e, "acen")),
                                        "top_choice_agreement": pop.two_level(frac(e, "top")),
                                        "greedy_match": popg.two_level(frac(eg, "phi"))},
           "levels_F1_F5": {fr: {"Abar": pop.two_level(lambda pid, s, fr=fr: e.abar(fr, pid, s)),
                                 "Abar_centred": pop.two_level(lambda pid, s, fr=fr: e.acen(fr, pid, s)),
                                 "top_choice_agreement": pop.two_level(lambda pid, s, fr=fr: e.top(fr, pid, s))} for fr in ("pick", "coord")},
           "greedy_match_F1_F5": {fr: popg.two_level(lambda pid, s, fr=fr: eg.phi(fr, pid, s)) for fr in ("pick", "coord")},
           "F1_F3": {fr: {"Abar": pop.two_level(lambda pid, s, fr=fr: e.abar(fr, pid, s, F13)),
                          "Abar_centred": pop.two_level(lambda pid, s, fr=fr: e.acen(fr, pid, s, F13)),
                          "top_choice_agreement": pop.two_level(lambda pid, s, fr=fr: e.top(fr, pid, s, F13))}
                     for fr in ("pick", "coord", "anti", "anti_human")}}
    return out


def a10_4(base, PO, subst, stratum, seed):
    out = {}
    names = subst + ["assistant_paraphrase", "none"]
    pop = Pop({k: v for k, v in PO.items() if k[0] in names and k[3] in ("pick", "coord")}, names, stratum, ["pick", "coord"], seed=seed + 5)
    e = Est(pop)
    ia, iap, inn = names.index("assistant"), names.index("assistant_paraphrase"), names.index("none")
    cosf = lambda a, b: (lambda pid, s: float(np.mean([pop.cosm(pop.mat("pick", s, f))[a, b] for f in FAMS])))
    out["i_cos_assistant_vs_paraphrase_pick"] = pop.two_level(cosf(ia, iap))
    out["i_cos_assistant_vs_none_pick"] = pop.two_level(cosf(ia, inn))
    Sii = lambda i: (lambda s: float(np.mean([pop.mat("coord", s, f)[i, i] - pop.mat("pick", s, f)[i, i] for f in FAMS])))
    npers = len(subst)                       # the point estimate's pid is arange(K); keep personas only
    Spop = lambda pid, s: e.phi("coord", pid[pid < npers], s) - e.phi("pick", pid[pid < npers], s)
    dSw = lambda s: Sii(iap)(s) - Sii(ia)(s)
    Spair = lambda s: float(np.mean([pop.mat("coord", s, f)[ia, iap] - pop.mat("pick", s, f)[ia, iap] for f in FAMS]))
    out["ii_S_assistant"] = pop.two_level(lambda pid, s: Sii(ia)(s))
    out["ii_S_paraphrase"] = pop.two_level(lambda pid, s: Sii(iap)(s))
    out["ii_dS_wording_primary"] = pop.two_level(lambda pid, s: dSw(s))
    out["ii_S_pair_secondary"] = pop.two_level(lambda pid, s: Spair(s))
    out["ii_S_pop"] = pop.two_level(Spop)
    out["ii_ratio_S_pop_over_abs_dS_wording"] = pop.two_level(lambda pid, s: Spop(pid, s) / abs(dSw(s)))

    # (iii) Bardsley items: Assistant S_00 vs human, on the 24 confirmatory and on all 28 items
    human = {g["instance_id"]: (fam, g) for fam in ("F1", "F4") for g in load_games(fam) if "human" in g}
    pil = set(pilot_instances())
    cols = ["persona", "frame", "family", "instance_id", "order_id", "persona_par", "frame_par", "p"]
    dfa = pd.concat([pd.read_parquet(f, columns=cols, filters=[("persona", "==", "assistant")])
                     for f in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet"))], ignore_index=True)
    dfa = dfa[dfa.instance_id.isin(human) & dfa.frame.isin(("pick", "coord"))]
    PA = per_order_cells(dfa)
    mss = lambda c: float((c * (c - 1)).sum() / (c.sum() * (c.sum() - 1)))
    items = []
    for iid, (fam, g) in sorted(human.items()):
        if ("assistant", fam, iid, "coord") not in PA:
            continue
        c_c, c_p = PA[("assistant", fam, iid, "coord")], PA[("assistant", fam, iid, "pick")]
        h = g["human"]
        items.append({"instance_id": iid, "family": fam, "pilot": iid in pil,
                      "model_S00": float(cross(c_c, c_c) - cross(c_p, c_p)),
                      "human_S00": mss(np.array(h["coordinate"], float)) - mss(np.array(h["pick"], float))})
    conf = [x for x in items if not x["pilot"]]
    out["iii_bardsley"] = {"n_items_found": len(items), "n_confirmatory": len(conf),
                           "spearman_confirmatory": spearman([x["model_S00"] for x in conf], [x["human_S00"] for x in conf]),
                           "spearman_all": spearman([x["model_S00"] for x in items], [x["human_S00"] for x in items]),
                           "items": items}
    return out


def analyze(base: str, seed: int = 0) -> dict:
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    stratum = {p: pcfg[p]["stratum"] for p in pcfg}
    stratum["none"] = "none"
    subst = [p for p in pcfg if stratum[p] in SUBST and p != "facilitator"]
    df = load(base)
    df = df[df.persona.isin(subst + ["none", "assistant_paraphrase"])]
    PO = per_order_cells(df)
    dg = df[df.persona.isin(subst) & df.frame.isin(("pick", "coord"))].copy()
    dg["p"] = dg.p.apply(_onehot)
    POg = per_order_cells(dg)
    r1, t1 = a10_1(PO, subst, stratum, seed)
    r2, t2 = a10_2(base, seed + 2)
    tests = {**t1, **t2}
    holm = _holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["holm"] = holm[k]
        tests[k]["pass"] = bool(holm[k] < 0.05)
    a, b = tests["A10-1a"]["pass"], tests["A10-1b"]["pass"]
    reading = ("personas converge on a common frame-specific answer, which the default model also gives" if not (a or b)
               else "toward the default model's answer" if a and b
               else f"not covered by the pre-stated rule (A10-1a {'pass' if a else 'fail'}, A10-1b {'pass' if b else 'fail'})")
    return {"tests": tests, "A10_1_reading": reading, "A10_1": r1, "A10_2": r2,
            "A10_3": a10_3(PO, POg, subst, stratum, seed), "A10_4": a10_4(base, PO, subst, stratum, seed)}
