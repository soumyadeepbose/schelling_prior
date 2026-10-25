"""PREREG Amendment 8 Part A, behavioural items (A1-A7, A12) on cached Phase 1 / 2 / 2b distributions.

Conventions as Amendments 3-7: 23 substantive personas, 36 confirmatory instances per family, cross-order U-statistic
estimators (mixture estimator where stated), per-family instance means pooled over families, two-level bootstrap
(personas within stratum, instances within family; B = 2000). `dedup=True` drops pairs whose resampled indices refer
to the same persona (A2 fix).
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
import yaml

from . import gram, prompts
from .analysis_amend3 import SUBST, cosv
from .analysis_phase2 import FAMS, NB, cross_pairs, kl, pilot_instances, spearman, temper_col
from .prompts import ROOT

COLS = ["persona", "frame", "family", "instance_id", "order_id", "persona_par", "frame_par", "options", "shown", "p", "logprobs", "offmenu_mass"]


def _ci(est, draws, side=None):
    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    if side == "greater":
        p = (d <= 0).mean() + 1 / (len(d) + 1)
    elif side == "less":
        p = (d >= 0).mean() + 1 / (len(d) + 1)
    else:
        p = 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / (len(d) + 1)
    return {"est": float(est), "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975)), "p": float(min(1.0, p))}


def per_order_cells(df, col="p"):
    """(persona, family, instance, frame) -> (O, n) mixed over the paraphrases present in df."""
    out = {}
    for key, g in df.groupby(["persona", "family", "instance_id", "frame"]):
        O, n = int(g.order_id.max()) + 1, len(g.iloc[0][col])
        acc, cnt = np.zeros((O, n)), np.zeros(O)
        for o, p in zip(g.order_id.values, g[col].values):
            acc[o] += p
            cnt[o] += 1
        out[key] = acc / np.maximum(cnt, 1)[:, None]
    return out


class Pop:
    """Per-family instance stacks for a persona list, with bootstrap helpers."""

    def __init__(self, PO, personas, stratum, frames, fams=FAMS, seed=0):
        self.PO, self.P, self.stratum, self.fams = PO, list(personas), stratum, list(fams)
        self.K = len(self.P)
        self.insts = {f: sorted({k[2] for k in PO if k[1] == f and k[0] == self.P[0] and k[3] == frames[0]}) for f in self.fams}
        self.rng = np.random.default_rng(seed)
        self.strata = [v for v in ([i for i, p in enumerate(self.P) if stratum[p] == s] for s in SUBST) if v]
        self.S, self.C, self.M = {}, {}, {}
        for fr in frames:
            for f in self.fams:
                if (self.P[0], f, self.insts[f][0], fr) not in PO:
                    continue
                st = [np.stack([PO[(p, f, iid, fr)] for p in self.P]) for iid in self.insts[f]]
                self.S[(fr, f)] = st
                self.C[(fr, f)] = np.stack([cross_pairs(s, s) for s in st])
                self.M[(fr, f)] = np.stack([gram.gram(s.mean(1)) for s in st])
        self.ALL = {f: np.arange(len(self.insts[f])) for f in self.fams}

    def fams_of(self, fr):
        return [f for f in self.fams if (fr, f) in self.C]

    def resample(self):
        return {f: self.rng.integers(0, len(self.insts[f]), len(self.insts[f])) for f in self.fams}

    def presample(self):
        return np.concatenate([self.rng.choice(v, len(v)) for v in self.strata])

    def two_level(self, fn, side=None, B=NB):
        est = fn(np.arange(self.K), self.ALL)
        return _ci(est, [fn(self.presample(), self.resample()) for _ in range(B)], side)

    @staticmethod
    def offd(M, pid, dedup=False):
        sub = M[np.ix_(pid, pid)]
        mask = ~np.eye(len(pid), dtype=bool)
        if dedup:
            mask &= pid[:, None] != pid[None, :]
        return float(sub[mask].mean())

    @staticmethod
    def cosm(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        return M / np.outer(d, d)

    def mat(self, fr, s, f, which="C"):
        return getattr(self, which)[(fr, f)][s[f]].mean(0)

    def phi(self, fr, pid, s, fams=None, dedup=False, which="C"):
        return float(np.mean([self.offd(self.mat(fr, s, f, which), pid, dedup) for f in (fams or self.fams_of(fr))]))

    def abar(self, fr, pid, s, fams=None, dedup=False, which="C"):
        return float(np.mean([self.offd(self.cosm(self.mat(fr, s, f, which)), pid, dedup) for f in (fams or self.fams_of(fr))]))

    def angle(self, fr, pid, s, dedup=False):
        vals = []
        for f in self.fams_of(fr):
            c = self.cosm(self.mat(fr, s, f))[np.ix_(pid, pid)]
            mask = ~np.eye(len(pid), dtype=bool)
            if dedup:
                mask &= pid[:, None] != pid[None, :]
            vals.append(np.degrees(np.arccos(np.clip(c[mask], -1, 1))).mean())
        return float(np.mean(vals))

    def mix(self, fr, f):
        return [s.mean(1) for s in self.S[(fr, f)]]


def load_phase2(phase2_dir, T=1.0):
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet"))], ignore_index=True)
    df = df[~df.instance_id.isin(pilot_instances())].reset_index(drop=True)
    if T != 1.0:
        df["p"] = temper_col(df, T)
    return df


def analyze(phase2_dir, phase2b_dir, phase1_path, old: dict, seed=0) -> dict:
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    stratum = {p: pcfg[p]["stratum"] for p in pcfg}
    subst = [p for p in pcfg if stratum[p] in SUBST and p != "facilitator"]
    df = load_phase2(phase2_dir)
    df2b = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(f"{phase2b_dir}/dist_shard*.parquet"))], ignore_index=True)
    PO = per_order_cells(pd.concat([df, df2b], ignore_index=True))
    out: dict = {}
    pop = Pop(PO, subst, stratum, ["pick", "guess", "coord", "anti", "coord_d2", "coord_d3"], seed=seed)
    K, ALL, allK = pop.K, pop.ALL, np.arange(pop.K)
    F13 = ["F1", "F3"]

    # ================= A1: convergence target vs the population centre
    rows = []
    a0 = subst.index("assistant")
    for f in FAMS:
        for j, iid in enumerate(pop.insts[f]):
            mc, mp, mg = pop.mix("coord", f)[j], pop.mix("pick", f)[j], pop.mix("guess", f)[j]
            q = mc.mean(0)
            cands = {"pop_pick": mp.mean(0), "pop_guess": mg.mean(0), "assistant_pick": mp[a0]}
            r = {"family": f, **{f"cos_{k}": cosv(q, c) for k, c in cands.items()}, **{f"kl_{k}": kl(c, q) for k, c in cands.items()}}
            rows.append(r)
    E = pd.DataFrame(rows)

    def fam_boot(fn):
        groups = {f: E[E.family == f] for f in FAMS}
        est = np.mean([fn(g).mean() for g in groups.values()])
        rng = np.random.default_rng(seed + 1)
        dr = [np.mean([fn(g.iloc[rng.integers(0, len(g), len(g))]).mean() for g in groups.values()]) for _ in range(NB)]
        return _ci(est, dr)

    A1 = {"target": {k: fam_boot(lambda g, k=k: g[k]) for k in E.columns if k != "family"}}
    for a, b in (("pop_guess", "pop_pick"), ("pop_guess", "assistant_pick"), ("pop_pick", "assistant_pick")):
        A1["target"][f"cos_{a}_minus_{b}"] = fam_boot(lambda g, a=a, b=b: g[f"cos_{a}"] - g[f"cos_{b}"])
        A1["target"][f"kl_{a}_minus_{b}"] = fam_boot(lambda g, a=a, b=b: g[f"kl_{a}"] - g[f"kl_{b}"])

    # per-persona regressions via per-instance cross-products (difference vectors sum to 0, so stacked means are 0)
    XP = {}
    for f in FAMS:
        st = []
        for j in range(len(pop.insts[f])):
            mc, mp, mg = pop.mix("coord", f)[j], pop.mix("pick", f)[j], pop.mix("guess", f)[j]
            y, xg = mc - mp, mg - mp
            xc = (mp.sum(0, keepdims=True) - mp) / (K - 1) - mp
            st.append(np.stack([(y * y).sum(1), (y * xg).sum(1), (y * xc).sum(1), (xg * xg).sum(1), (xg * xc).sum(1),
                                (xc * xc).sum(1), np.full(K, y.shape[1])], 1))        # (K, 7)
        XP[f] = np.stack(st)                                                          # (I, K, 7)

    def reg(s):
        S = sum(XP[f][s[f]].sum(0) for f in FAMS)                                     # (K, 7)
        yy, yg, yc, gg, gc, cc, n = S.T
        det = gg * cc - gc ** 2
        bg, bc = (yg * cc - yc * gc) / det, (yc * gg - yg * gc) / det
        sse_full = yy - bg * yg - bc * yc
        sse_c = yy - yc ** 2 / cc
        sse_g = yy - yg ** 2 / gg
        sd = lambda q: np.sqrt(q / n)
        return {"std_beta_guess": bg * sd(gg) / sd(yy), "std_beta_centre": bc * sd(cc) / sd(yy),
                "partial_r2_guess": (sse_c - sse_full) / sse_c, "partial_r2_centre": (sse_g - sse_full) / sse_g,
                "r2_full": 1 - sse_full / yy}

    est_r = reg(ALL)
    A1["regression"] = {k: pop.two_level(lambda pid, s, k=k: float(reg(s)[k][pid].mean())) for k in est_r}
    A1["regression"]["per_persona"] = {p: {k: float(v[i]) for k, v in est_r.items()} for i, p in enumerate(subst)}
    A1["regression"]["n_personas_guess_beta_gt_centre"] = int((est_r["std_beta_guess"] > est_r["std_beta_centre"]).sum())

    own, other = {}, {}
    for f in FAMS:
        o1, o2 = [], []
        for j in range(len(pop.insts[f])):
            mc, mp, mg = pop.mix("coord", f)[j], pop.mix("pick", f)[j], pop.mix("guess", f)[j]
            D = mc - mp
            o1.append([cosv(D[i], mg[i] - mp[i]) for i in range(K)])
            o2.append([np.nanmean([cosv(D[i], mg[k] - mp[i]) for k in range(K) if k != i]) for i in range(K)])
        own[f], other[f] = np.array(o1), np.array(o2)
    pp = lambda A, s: np.mean([np.nanmean(A[f][s[f]], 0) for f in FAMS], 0)
    A1["alignment"] = {"own_guess": pop.two_level(lambda pid, s: float(pp(own, s)[pid].mean())),
                       "other_guess": pop.two_level(lambda pid, s: float(pp(other, s)[pid].mean())),
                       "own_minus_other": pop.two_level(lambda pid, s: float((pp(own, s) - pp(other, s))[pid].mean()))}
    out["A1"] = A1

    # ================= A2: bootstrap fix (duplicate-persona pairs dropped) + jackknife
    new = {}
    for dd in (False, True):
        tag = "dedup" if dd else "with_duplicates"
        new[tag] = {
            "Phi_pick": pop.two_level(lambda pid, s: pop.phi("pick", pid, s, dedup=dd)),
            "Phi_coord": pop.two_level(lambda pid, s: pop.phi("coord", pid, s, dedup=dd)),
            "S_pop": pop.two_level(lambda pid, s: pop.phi("coord", pid, s, dedup=dd) - pop.phi("pick", pid, s, dedup=dd)),
            "Abar_pick": pop.two_level(lambda pid, s: pop.abar("pick", pid, s, dedup=dd)),
            "Abar_coord": pop.two_level(lambda pid, s: pop.abar("coord", pid, s, dedup=dd)),
            "dAbar": pop.two_level(lambda pid, s: pop.abar("coord", pid, s, dedup=dd) - pop.abar("pick", pid, s, dedup=dd)),
            "frac_removed": pop.two_level(lambda pid, s: 1 - (1 - pop.abar("coord", pid, s, dedup=dd)) / (1 - pop.abar("pick", pid, s, dedup=dd))),
            "d_angle_deg": pop.two_level(lambda pid, s: pop.angle("coord", pid, s, dd) - pop.angle("pick", pid, s, dd)),
            "A3A1_Abar_pick_F13": pop.two_level(lambda pid, s: pop.abar("pick", pid, s, F13, dd)),
            "A3A1_Abar_coord_F13": pop.two_level(lambda pid, s: pop.abar("coord", pid, s, F13, dd)),
            "A3A1_Abar_anti_F13": pop.two_level(lambda pid, s: pop.abar("anti", pid, s, F13, dd)),
            "A3A1_coord_minus_anti": pop.two_level(lambda pid, s: pop.abar("coord", pid, s, F13, dd) - pop.abar("anti", pid, s, F13, dd)),
            "A3A1_frac_removed_anti": pop.two_level(lambda pid, s: 1 - (1 - pop.abar("anti", pid, s, F13, dd)) / (1 - pop.abar("pick", pid, s, F13, dd))),
            "A3A3_Abar_guess": pop.two_level(lambda pid, s: pop.abar("guess", pid, s, dedup=dd)),
        }
    # statistics without pairwise terms (identity check): A3-A2 and A4-1/A4-2 alignments
    al = {fr: {f: np.array([[cosv(pop.mix(fr, f)[j][i] - pop.mix("pick", f)[j][i], pop.mix("guess", f)[j][i] - pop.mix("pick", f)[j][i])
                             for i in range(K)] for j in range(len(pop.insts[f]))]) for f in pop.fams_of(fr)} for fr in ("coord", "anti")}
    ppf = lambda A, s, fams: np.mean([np.nanmean(A[f][s[f]], 0) for f in fams], 0)
    new["no_pair_terms"] = {"A3A2_alignment": pop.two_level(lambda pid, s: float(ppf(al["coord"], s, FAMS)[pid].mean())),
                            "A4_1_anti_alignment_F13": pop.two_level(lambda pid, s: float(ppf(al["anti"], s, F13)[pid].mean()), "less"),
                            "A4_2_anti_minus_coord_F13": pop.two_level(lambda pid, s: float((ppf(al["anti"], s, F13) - ppf(al["coord"], s, F13))[pid].mean()), "less")}
    jk = {}
    for name, fn in (("Abar_pick", lambda pid: pop.abar("pick", pid, ALL)), ("Abar_coord", lambda pid: pop.abar("coord", pid, ALL)),
                     ("dAbar", lambda pid: pop.abar("coord", pid, ALL) - pop.abar("pick", pid, ALL))):
        full = fn(allK)
        loo = np.array([fn(np.delete(allK, k)) for k in range(K)])
        se = float(np.sqrt((K - 1) / K * ((loo - loo.mean()) ** 2).sum()))
        k_inf = int(np.argmax(np.abs(loo - full)))
        jk[name] = {"est": full, "jack_se": se, "lo": full - 1.96 * se, "hi": full + 1.96 * se,
                    "loo_min": float(loo.min()), "loo_max": float(loo.max()),
                    "most_influential": subst[k_inf], "loo_without_most_influential": float(loo[k_inf])}
    new["jackknife"] = jk
    new["old"] = {"A2a": {k: old["phase2"]["A2"]["a"][k] for k in ("est", "lo", "hi")} | {"Phi_pick": old["phase2"]["A2"]["a"]["Phi_pick"], "Phi_coord": old["phase2"]["A2"]["a"]["Phi_coord"]},
                  "A2b": {k: old["phase2"]["A2"]["b"][k] for k in ("Abar_pick", "Abar_coord", "dAbar", "frac_disagreement_removed", "d_angle_deg")},
                  "A3A1": {k: old["amend3"]["A1_anti_control"][k] for k in ("pick", "coord", "anti", "coord_minus_anti", "frac_removed_anti")},
                  "A3A2": old["amend3"]["A2_shift_alignment"]["population"],
                  "A4_1": old["amend4"]["A4_1_anti_alignment"]["population"], "A4_2": old["amend4"]["A4_2_anti_minus_coord"]}
    out["A2"] = new

    # ================= A3: estimator audit
    aud = {}
    for fr in ("pick", "coord"):
        per_f = {}
        for f in FAMS:
            M = pop.mat(fr, ALL, f)
            c = pop.cosm(M)
            off = c[~np.eye(K, dtype=bool)]
            per_f[f] = {"min_eig": float(np.linalg.eigvalsh(M).min()), "max_offdiag_cos": float(off.max()), "n_cos_gt_1": int((off > 1).sum())}
        Mp = np.mean([pop.mat(fr, ALL, f) for f in FAMS], 0)
        cp = pop.cosm(Mp)[~np.eye(K, dtype=bool)]
        aud[fr] = {"per_family": per_f, "pooled": {"min_eig": float(np.linalg.eigvalsh(Mp).min()), "max_offdiag_cos": float(cp.max()),
                                                    "n_cos_gt_1": int((cp > 1).sum())}}
    # the saved Gram matrices (all 26 personas, pooled U-statistic)
    g = old["phase2"]["gram"]
    for fr in ("pick", "coord"):
        M = np.array(g[fr])
        c = pop.cosm(M)[~np.eye(len(M), dtype=bool)]
        aud[f"saved_gram_{fr}"] = {"order": g["order"], "min_eig": float(np.linalg.eigvalsh(M).min()), "max_offdiag_cos": float(c.max()),
                                   "n_cos_gt_1": int((c > 1).sum())}
    for which in ("C", "M"):
        tag = "ustat" if which == "C" else "mixture"
        aud[f"headline_{tag}"] = {"Abar_pick": pop.two_level(lambda pid, s: pop.abar("pick", pid, s, which=which)),
                                  "Abar_coord": pop.two_level(lambda pid, s: pop.abar("coord", pid, s, which=which)),
                                  "Phi_pick": pop.two_level(lambda pid, s: pop.phi("pick", pid, s, which=which)),
                                  "Phi_coord": pop.two_level(lambda pid, s: pop.phi("coord", pid, s, which=which))}

        def decomp(pid, s, which=which):
            tot, dec, agr = [], [], []
            mask = ~np.eye(len(pid), dtype=bool)
            for f in FAMS:
                Pm, Cm = pop.mat("pick", s, f, which)[np.ix_(pid, pid)], pop.mat("coord", s, f, which)[np.ix_(pid, pid)]
                dl = np.log(np.clip(Cm, 1e-300, None)) - np.log(np.clip(Pm, 1e-300, None))
                d = 0.5 * (np.log(np.diag(Cm)) - np.log(np.diag(Pm)))
                agr_m = np.log(np.clip(pop.cosm(Cm), 1e-300, None)) - np.log(np.clip(pop.cosm(Pm), 1e-300, None))
                tot.append(dl[mask].mean()); dec.append((d[:, None] + d[None, :])[mask].mean()); agr.append(agr_m[mask].mean())
            return np.mean(tot), np.mean(dec), np.mean(agr)

        aud[f"decomposition_{tag}"] = {k: pop.two_level(lambda pid, s, i=i: float(decomp(pid, s)[i])) for i, k in enumerate(("mean_dlogC", "decisiveness", "agreement"))}
    out["A3"] = aud

    # ================= A4: position bias by frame
    frames_pos = ["pick", "guess", "coord", "anti", "coord_d2", "coord_d3"]
    dfa = pd.concat([df, df2b], ignore_index=True)
    dfa = dfa[dfa.persona.isin(subst)]
    pos = {}
    for (fr, f), g in dfa[dfa.frame.isin(frames_pos)].groupby(["frame", "family"]):
        acc = {}
        for opts, shown, p in zip(g.options, g.shown, g.p):
            n = len(opts)
            ix = {o: k for k, o in enumerate(opts)}
            m = np.array([p[ix[s]] for s in shown])
            a, c = acc.get(n, (np.zeros(n), 0))
            acc[n] = (a + m, c + 1)
        pos[f"{fr}|{f}"] = {str(n): (a / c).round(4).tolist() for n, (a, c) in acc.items()}
    # same-order estimator: C_same_ij = mean over orders of <p_io, p_jo>
    same = {}
    for (fr, f), st in pop.S.items():
        same[(fr, f)] = np.stack([np.einsum("ion,jon->ijo", s, s).mean(2) for s in st])
    def phi_s(fr, pid, s, fams=None, dedup=False):
        return float(np.mean([pop.offd(same[(fr, f)][s[f]].mean(0), pid, dedup) for f in (fams or pop.fams_of(fr))]))
    def abar_s(fr, pid, s, fams=None):
        return float(np.mean([pop.offd(pop.cosm(same[(fr, f)][s[f]].mean(0)), pid) for f in (fams or pop.fams_of(fr))]))
    def cii(fr, pid, s, est="cross"):
        src = pop.C if est == "cross" else same
        return float(np.mean([np.diag(src[(fr, f)][s[f]].mean(0))[pid].mean() for f in pop.fams_of(fr)]))
    tab = {}
    for fr in ("pick", "guess", "coord", "anti", "coord_d2", "coord_d3"):
        fams = F13 if fr == "anti" else None
        tab[fr] = {"Cii_cross": cii(fr, allK, ALL), "Cii_same": cii(fr, allK, ALL, "same"),
                   "Phi_cross": pop.phi(fr, allK, ALL, fams), "Phi_same": phi_s(fr, allK, ALL, fams),
                   "Abar_cross": pop.abar(fr, allK, ALL, fams), "Abar_same": abar_s(fr, allK, ALL, fams)}
    bonus = lambda fr, pid, s: phi_s(fr, pid, s) - pop.phi(fr, pid, s)
    out["A4"] = {"position_mass": pos, "by_frame": tab,
                 "same_order_bonus_Phi_pick": pop.two_level(lambda pid, s: bonus("pick", pid, s)),
                 "same_order_bonus_Phi_coord": pop.two_level(lambda pid, s: bonus("coord", pid, s)),
                 "bonus_coord_minus_pick": pop.two_level(lambda pid, s: bonus("coord", pid, s) - bonus("pick", pid, s)),
                 "dPhi_pick_to_coord_same": pop.two_level(lambda pid, s: phi_s("coord", pid, s) - phi_s("pick", pid, s)),
                 "dAbar_pick_to_coord_same": pop.two_level(lambda pid, s: abar_s("coord", pid, s) - abar_s("pick", pid, s))}

    # ================= A5: Bardsley items
    human = {g["instance_id"]: g for fam in ("F1", "F4") for g in prompts.load_games(fam) if "human" in g}
    bard = {f: [j for j, iid in enumerate(pop.insts[f]) if iid in human] for f in ("F1", "F4")}
    sub = {f: np.array(bard[f]) for f in ("F1", "F4")}
    rngb = np.random.default_rng(seed + 5)

    def rs_b():
        return {f: v[rngb.integers(0, len(v), len(v))] for f, v in sub.items()}

    def tl_b(fn):
        est = fn(allK, sub)
        return _ci(est, [fn(pop.presample(), rs_b()) for _ in range(NB)])
    mss = lambda c: float((c * (c - 1)).sum() / (c.sum() * (c.sum() - 1)))
    hS = {f: np.array([mss(np.array(human[pop.insts[f][j]]["human"]["coordinate"], float)) - mss(np.array(human[pop.insts[f][j]]["human"]["pick"], float))
                       for j in sub[f]]) for f in sub}
    A5 = {"n_items": int(sum(len(v) for v in sub.values())),
          "S_pop": tl_b(lambda pid, s: float(np.mean([pop.offd(pop.mat("coord", s, f), pid) - pop.offd(pop.mat("pick", s, f), pid) for f in sub]))),
          "dAbar": tl_b(lambda pid, s: float(np.mean([pop.offd(pop.cosm(pop.mat("coord", s, f)), pid) - pop.offd(pop.cosm(pop.mat("pick", s, f)), pid) for f in sub]))),
          "human_S00": _ci(np.mean([hS[f].mean() for f in hS]), [np.mean([v[rngb.integers(0, len(v), len(v))].mean() for v in hS.values()]) for _ in range(NB)])}
    for T in (1.0, 0.3):
        dfa_ = df[(df.persona == "assistant") & df.instance_id.isin(human) & df.frame.isin(("pick", "coord"))].copy()
        if T != 1.0:
            dfa_["p"] = temper_col(dfa_, T)
        PA = per_order_cells(dfa_)
        ms, hs, match = [], [], []
        for f in sub:
            for j in sub[f]:
                iid = pop.insts[f][j]
                c_c, c_p = PA[("assistant", f, iid, "coord")], PA[("assistant", f, iid, "pick")]
                from .analysis_phase2 import cross
                ms.append(cross(c_c, c_c) - cross(c_p, c_p))
                hh = human[iid]["human"]
                hs.append(mss(np.array(hh["coordinate"], float)) - mss(np.array(hh["pick"], float)))
                match.append(int(np.argmax(c_c.mean(0)) == int(np.argmax(hh["coordinate"]))))
        A5[f"assistant_T{T}"] = {"S00_mean": float(np.mean(ms)), "spearman_model_vs_human_S00": spearman(ms, hs),
                                 "top_choice_matches_human_modal_coord": int(sum(match)), "n": len(match)}
    out["A5"] = A5

    # ================= A6: outliers and controls
    excl = [p for p in subst if p not in ("contrarian", "competitor", "cartel_executive")]
    pop20 = Pop(PO, excl, stratum, ["pick", "coord"], seed=seed + 6)
    A6 = {"excluded": ["contrarian", "competitor", "cartel_executive"],
          "Abar_pick": pop20.two_level(lambda pid, s: pop20.abar("pick", pid, s)),
          "Abar_coord": pop20.two_level(lambda pid, s: pop20.abar("coord", pid, s)),
          "frac_removed": pop20.two_level(lambda pid, s: 1 - (1 - pop20.abar("coord", pid, s)) / (1 - pop20.abar("pick", pid, s)))}
    ctrl = [p for p in ("random_chooser", "assistant_paraphrase") if (p, "F1", pop.insts["F1"][0], "pick") in PO]
    popc = Pop(PO, subst + ctrl, {**stratum, **{c: "control" for c in ctrl}}, ["pick", "guess", "coord", "anti"], seed=seed + 7)
    rc, ap, ai = popc.P.index("random_chooser"), popc.P.index("assistant_paraphrase"), popc.P.index("assistant")
    om = df.groupby(["persona", "frame"]).offmenu_mass.mean()
    rcr = {}
    for fr in ("pick", "guess", "coord", "anti"):
        fams = popc.fams_of(fr)
        Ms = [popc.mat(fr, popc.ALL, f) for f in fams]
        rcr[fr] = {"C_ii": float(np.mean([M[rc, rc] for M in Ms])),
                   "mean_cos_with_substantive": float(np.mean([popc.cosm(M)[rc, :len(subst)].mean() for M in Ms])),
                   "omega": float(om.get(("random_chooser", fr), np.nan))}
    A6["random_chooser"] = rcr
    A6["assistant_paraphrase_vs_assistant_pick_cos"] = float(np.mean([popc.cosm(popc.mat("pick", popc.ALL, f))[ap, ai] for f in FAMS]))
    A6["assistant_paraphrase_vs_assistant_coord_cos"] = float(np.mean([popc.cosm(popc.mat("coord", popc.ALL, f))[ap, ai] for f in FAMS]))
    A6["gram_heatmap_order"] = old["phase2"]["gram"]["order"]
    out["A6"] = A6

    # ================= A7: noise floor of the partner-description contrasts (D1 frame paraphrase 0 vs 1)
    sel = pd.concat([df, df2b], ignore_index=True)
    sel = sel[sel.persona.isin(subst) & sel.frame.isin(("pick", "coord"))]
    POp = {fp: per_order_cells(sel[sel.frame_par == fp]) for fp in (0, 1)}
    pp_ = {fp: Pop(POp[fp], subst, stratum, ["pick", "coord"], seed=seed + 8) for fp in (0, 1)}
    rng7 = np.random.default_rng(seed + 9)

    def selfc(P_, fr, s):
        return np.mean([np.diagonal(P_.mat(fr, s, f)) for f in FAMS], 0)

    def ret(P_, s):
        vals = []
        for f in FAMS:
            X = np.stack([cross_pairs(a, b) for a, b in zip(P_.S[("coord", f)], P_.S[("pick", f)])])[s[f]].mean(0)
            vals.append(np.diag(X) / np.sqrt(np.diag(P_.mat("coord", s, f)) * np.diag(P_.mat("pick", s, f))))
        return np.mean(vals, 0)

    def a7(fn):
        est = fn(allK, ALL)
        dr = []
        for _ in range(NB):
            pid = np.concatenate([rng7.choice(v, len(v)) for v in pop.strata])
            s = {f: rng7.integers(0, len(pop.insts[f]), len(pop.insts[f])) for f in FAMS}
            dr.append(fn(pid, s))
        return _ci(est, dr)
    A7 = {"dAbar_par1_minus_par0": a7(lambda pid, s: pp_[1].abar("coord", pid, s) - pp_[0].abar("coord", pid, s)),
          "dS_par1_minus_par0": a7(lambda pid, s: float(((selfc(pp_[1], "coord", s) - selfc(pp_[1], "pick", s)) - (selfc(pp_[0], "coord", s) - selfc(pp_[0], "pick", s)))[pid].mean())),
          "dRetention_par1_minus_par0": a7(lambda pid, s: float((ret(pp_[1], s) - ret(pp_[0], s))[pid].mean())),
          "Abar_coord_par0": pp_[0].abar("coord", allK, ALL), "Abar_coord_par1": pp_[1].abar("coord", allK, ALL)}
    pb = old["amend3_2b"]["partB"]
    A7["partB_effects"] = {t: {k: pb[t][k] for k in ("a1_dAbar", "a2_dRetention", "b1_dS", "b2_median_beta_minus_1", "c_d_mean")} for t in ("D3", "D2")}
    A7["partB_holm"] = pb["holm"]
    out["A7"] = A7

    # ================= A12: small items
    h10 = {p: float(np.mean([kl(pop.mix("coord", f)[j][i], pop.mix("pick", f)[j][i]) - kl(pop.mix("coord", f)[j][i], pop.mix("guess", f)[j][i])
                             for f in ("F3", "F4", "F5") for j in range(len(pop.insts[f]))])) for i, p in enumerate(subst)}
    rows_old = {r["persona"]: r for r in old["phase2"]["persona_rows"]}
    a2f = old["phase2"]["A2"]["f"]
    alpha = np.array([rows_old[p]["alpha"] for p in subst])

    def axis_gap(pid, s):
        iu = np.triu_indices(len(pid), 1)
        dist = np.abs(alpha[pid][:, None] - alpha[pid][None, :])[iu]
        lo_t, hi_t = np.quantile(dist, [1 / 3, 2 / 3])
        return float(np.mean([pop.cosm(pop.mat("pick", s, f))[np.ix_(pid, pid)][iu][dist <= lo_t].mean()
                              - pop.cosm(pop.mat("pick", s, f))[np.ix_(pid, pid)][iu][dist >= hi_t].mean() for f in FAMS]))
    dA = lambda pid, s: pop.abar("coord", pid, s) - pop.abar("pick", pid, s)
    out["A12"] = {"H10_kl_gain_F3_F5": h10, "H10_population_mean_F3_F5": float(np.mean(list(h10.values()))),
                  "S_ii_per_persona": {p: rows_old[p]["S"] for p in subst},
                  "A2f_old": a2f,
                  "A2f_unrounded": {"dAbar": pop.two_level(dA), "axis_gap_pick": pop.two_level(axis_gap),
                                    "ratio": pop.two_level(lambda pid, s: dA(pid, s) / axis_gap(pid, s)),
                                    "difference": pop.two_level(lambda pid, s: dA(pid, s) - axis_gap(pid, s))},
                  "F5_story_variants": "not available: only the neutral F5 story was scored (Phase 1 and Phase 2)"}
    return out
