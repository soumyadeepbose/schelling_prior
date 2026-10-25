"""PREREG Amendment 9 add-on (cached Phase 2 + Tier B readouts): C-A same-frame default comparison, C-B flattening-robust
measures for every control frame, C-C greedy-decoding agreement. Estimators as Phase 2: one distribution per option order
(mixed over paraphrases), cross-order U-statistics, two-level bootstrap."""
from __future__ import annotations

import numpy as np
import yaml

from .analysis_amend3 import SUBST
from .analysis_amend8a import NB, Pop, _ci, per_order_cells
from .analysis_amend8b import F13, load, tost
from .analysis_phase2 import FAMS
from .prompts import ROOT

CA_FRAMES = ["pick", "coord", "coord_human", "goal_nonsocial", "guess"]
CB_FRAMES = ["pick_long", "guess", "guess_human", "goal_nonsocial", "coord", "coord_human", "anti", "anti_human"]
NONE_FRAMES = ["pick", "guess", "coord", "coord_human", "pick_long"]


def _cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


class _M:
    """Measures on a Pop; fams restricted to F1/F3 for the anti frames."""

    def __init__(self, pop):
        self.pop = pop
        self._am = {}
        self._inv_n = {}

    @staticmethod
    def fams(fr):
        return F13 if fr in ("anti", "anti_human") else FAMS

    def offd(self, M, pid):
        return self.pop.offd(M, pid)

    def abar(self, fr, pid, s, fams=None):
        return float(np.mean([self.offd(self.pop.cosm(self.pop.mat(fr, s, f)), pid) for f in (fams or self.fams(fr))]))

    def acen(self, fr, pid, s, fams=None):
        vals = []
        for f in fams or self.fams(fr):
            if (fr, f) not in self._inv_n:      # instances of one family can have different menu sizes
                self._inv_n[(fr, f)] = np.array([1.0 / x.shape[-1] for x in self.pop.S[(fr, f)]])
            M = (self.pop.C[(fr, f)][s[f]] - self._inv_n[(fr, f)][s[f]][:, None, None]).mean(0)
            vals.append(self.offd(self.pop.cosm(M), pid))
        return float(np.mean(vals))

    def phi(self, fr, pid, s, fams=None):
        return float(np.mean([self.offd(self.pop.mat(fr, s, f), pid) for f in (fams or self.fams(fr))]))

    def top(self, fr, pid, s, fams=None):
        vals = []
        for f in fams or self.fams(fr):
            if (fr, f) not in self._am:
                self._am[(fr, f)] = np.stack([x.argmax(-1) for x in self.pop.mix(fr, f)])
            am = self._am[(fr, f)][s[f]][:, pid]
            mask = ~np.eye(len(pid), dtype=bool)
            vals.append((am[:, :, None] == am[:, None, :])[:, mask].mean())
        return float(np.mean(vals))

    def decomp(self, fr, pid, s, fams=None):
        tot, dec, agr = [], [], []
        k = len(pid)
        mask = ~np.eye(k, dtype=bool)
        for f in fams or self.fams(fr):
            P, C = self.pop.mat("pick", s, f)[np.ix_(pid, pid)], self.pop.mat(fr, s, f)[np.ix_(pid, pid)]
            dd = 0.5 * (np.log(np.diag(C)) - np.log(np.diag(P)))
            D = dd[:, None] + dd[None, :]
            tot.append((np.log(C) - np.log(P))[mask].mean())
            dec.append(D[mask].mean())
            agr.append((np.log(self.pop.cosm(C)) - np.log(self.pop.cosm(P)))[mask].mean())
        return np.array([np.mean(tot), np.mean(dec), np.mean(agr)])


def _boot_vec(pop, fn, names):
    est = fn(np.arange(pop.K), pop.ALL)
    dr = np.array([fn(pop.presample(), pop.resample()) for _ in range(NB)])
    return {n: _ci(est[i], dr[:, i]) for i, n in enumerate(names)}


def analyze(base: str, seed: int = 0) -> dict:
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    stratum = {p: pcfg[p]["stratum"] for p in pcfg}
    stratum["none"] = "none"
    subst = [p for p in pcfg if stratum[p] in SUBST and p != "facilitator"]
    df = load(base)
    df = df[df.persona.isin(subst + ["none"])]
    frames = ["pick"] + sorted(set(CA_FRAMES + CB_FRAMES) - {"pick"})     # Pop takes instance lists from frames[0]
    pop = Pop(per_order_cells(df[df.persona != "none"]), subst, stratum, frames, seed=seed)
    m = _M(pop)
    K, ALL, allK = pop.K, pop.ALL, np.arange(pop.K)
    ia = subst.index("assistant")
    out: dict = {"n_personas": K}

    # ---------------- C-A
    popn = Pop(per_order_cells(df[df.persona.isin(subst + ["none"])
                                  & df.frame.isin(NONE_FRAMES)]), subst + ["none"], stratum, NONE_FRAMES, seed=seed + 1)
    inn = K
    assert popn.insts == pop.insts, "instance lists differ between the two populations"
    pbar = {fr: {f: pop.mix(fr, f) for f in FAMS} for fr in CA_FRAMES}                        # per instance (K, n_i)
    pnone = {fr: {f: [x[inn] for x in popn.mix(fr, f)] for f in FAMS} for fr in CA_FRAMES if fr in NONE_FRAMES}

    def qbar(fr, f, pid, s):
        keep = pid[pid != ia]
        return [pbar[fr][f][j][keep].mean(0) for j in s[f]]

    def cos_rows(A, B):
        return float(np.mean([_cos(a, b) for a, b in zip(A, B)]))

    def q_vs(fr, target, pid, s):
        vals = []
        for f in FAMS:
            q = qbar(fr, f, pid, s)
            if target == "assistant":
                t = [pbar[fr][f][j][ia] for j in s[f]]
            elif target == "none":
                t = [pnone[fr][f][j] for j in s[f]]
            else:
                t = qbar("pick", f, pid, s)
            vals.append(cos_rows(q, t))
        return float(np.mean(vals))

    def cos_persona_to(fr, target, pid, s):
        if target == "assistant":
            vals = [pop.cosm(pop.mat(fr, s, f))[:, ia] for f in FAMS]
            keep = pid[pid != ia]
        else:
            vals = [popn.cosm(popn.mat(fr, s, f))[:K, inn] for f in FAMS]
            keep = pid
        return float(np.mean(vals, 0)[keep].mean())

    CA = {}
    for fr in CA_FRAMES:
        r = {"mean_cos_persona_to_assistant": pop.two_level(lambda pid, s, fr=fr: cos_persona_to(fr, "assistant", pid, s)),
             "cos_qbar_to_assistant": pop.two_level(lambda pid, s, fr=fr: q_vs(fr, "assistant", pid, s)),
             "cos_qbar_to_qbar_pick": pop.two_level(lambda pid, s, fr=fr: q_vs(fr, "pick", pid, s))}
        if fr in NONE_FRAMES:
            r["mean_cos_persona_to_none"] = popn.two_level(lambda pid, s, fr=fr: cos_persona_to(fr, "none", pid[pid < K], s))
            r["cos_qbar_to_none"] = pop.two_level(lambda pid, s, fr=fr: q_vs(fr, "none", pid, s))
        CA[fr] = r
    out["C_A"] = CA
    tests = {"CA-A": pop.two_level(lambda pid, s: q_vs("coord", "assistant", pid, s) - q_vs("coord", "pick", pid, s), "greater"),
             "CA-none": pop.two_level(lambda pid, s: q_vs("coord", "none", pid, s) - q_vs("coord", "pick", pid, s), "greater")}
    ps = sorted((v["p"], k) for k, v in tests.items())
    adj, run = {}, 0.0
    for r_, (p_, k) in enumerate(ps):
        run = max(run, min(1.0, p_ * (len(ps) - r_)))
        adj[k] = run
    for k in tests:
        tests[k]["holm"] = adj[k]
    tests["reading"] = ("the default model's answer in the same frame" if min(adj.values()) < 0.05
                        else "a population-level common choice")
    out["C_A_tests"] = tests

    # ---------------- C-B
    CB = {}
    for fr in ["pick"] + CB_FRAMES:
        r = {"Abar": pop.two_level(lambda pid, s, fr=fr: m.abar(fr, pid, s)),
             "Abar_centred": pop.two_level(lambda pid, s, fr=fr: m.acen(fr, pid, s)),
             "top_choice_agreement": pop.two_level(lambda pid, s, fr=fr: m.top(fr, pid, s))}
        if fr != "pick":
            fams = m.fams(fr)
            r["pick_same_families"] = {"Abar": m.abar("pick", allK, ALL, fams), "Abar_centred": m.acen("pick", allK, ALL, fams),
                                       "top_choice_agreement": m.top("pick", allK, ALL, fams)}
            r["decomposition_vs_pick"] = _boot_vec(pop, lambda pid, s, fr=fr: m.decomp(fr, pid, s), ["dlogC", "decisiveness", "agreement"])
        CB[fr] = r
    out["C_B"] = CB
    rob = {}
    for name, fn in (("centred", m.acen), ("top_choice", m.top)):
        b1 = lambda pid, s, fn=fn: fn("coord", pid, s) - fn("coord_human", pid, s)
        b3 = lambda pid, s, fn=fn: fn("goal_nonsocial", pid, s) - fn("coord", pid, s)
        for tn, f_ in (("B1", b1), ("B3", b3)):
            dr = [f_(pop.presample(), pop.resample()) for _ in range(NB)]
            rob[f"{tn}_TOST_{name}"] = tost(f_(allK, ALL), dr)
    ps = sorted((v["p"], k) for k, v in rob.items())
    run = 0.0
    for r_, (p_, k) in enumerate(ps):
        run = max(run, min(1.0, p_ * (len(ps) - r_)))
        rob[k]["holm4"] = run
    out["C_B_robustness_TOST"] = rob

    # ---------------- C-C greedy decoding
    dg = df[(df.persona != "none") & df.frame.isin(CA_FRAMES)].copy()
    dg["p"] = dg.p.apply(lambda v: np.eye(len(v))[int(np.argmax(v))])
    popg = Pop(per_order_cells(dg), subst, stratum, CA_FRAMES, seed=seed + 2)
    mg = _M(popg)
    CC = {}
    for fr in CA_FRAMES:
        CC[fr] = {"Phi_argmax": popg.two_level(lambda pid, s, fr=fr: mg.phi(fr, pid, s)),
                  "self_consistency": float(np.mean([np.diag(popg.mat(fr, popg.ALL, f)).mean() for f in FAMS]))}
        if fr != "pick":
            CC[fr]["dPhi_argmax_vs_pick"] = popg.two_level(lambda pid, s, fr=fr: mg.phi(fr, pid, s) - mg.phi("pick", pid, s))
            CC[fr]["frac_disagreement_removed_argmax"] = popg.two_level(
                lambda pid, s, fr=fr: 1 - (1 - mg.phi(fr, pid, s)) / (1 - mg.phi("pick", pid, s)))
    out["C_C"] = CC
    return out
