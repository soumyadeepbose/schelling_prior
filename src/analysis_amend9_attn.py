"""PREREG Amendment 9, Run 4 (attention to the persona: K1, K2, K3) and Run 6 (random-direction control for the g ablation:
R1). Run 6 uses the C1 estimators (orders 1-2 remapped to 0-1, cross-order U-statistics). Run 4 has one order, so its
behaviour quantities use same-order (mixture) estimators for conditions and baselines alike."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .amend6 import ORDERS, split
from .amend9 import RUN4_ORDER, SPAN_NAMES
from .analysis_amend3 import cosv
from .analysis_amend8a import Pop, per_order_cells
from .analysis_amend8c import _sm, _subst
from .analysis_phase2 import FAMS, cross_pairs

COLS = ["persona", "family", "instance_id", "order_id", "frame", "p"]


def _phase2(base, subst, orders):
    sp = split()
    B = {iid for f in sp for iid in sp[f]["B"]}
    cols = ["persona", "frame", "family", "instance_id", "order_id", "persona_par", "frame_par", "p"]
    ph = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet"))], ignore_index=True)
    ph = ph[ph.persona.isin(subst) & ph.instance_id.isin(B) & ph.order_id.isin(orders) & (ph.persona_par == 0)
            & (ph.frame_par == 0) & ph.frame.isin(("pick", "coord", "guess"))]
    return ph.assign(frame="p2_" + ph.frame)[COLS]


def _steered(pattern, cond_col="condition"):
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pattern))], ignore_index=True)
    d["p"] = d.logprobs.apply(_sm)
    return d.assign(frame=d[cond_col])[COLS], d


class _Metrics:
    """Behaviour metrics on a Pop; which = 'C' (cross-order) or 'M' (same-order / mixture)."""

    def __init__(self, pop, which):
        self.pop, self.w = pop, which

    def mat(self, c, s, f):
        return self.pop.mat(c, s, f, self.w)

    def abar(self, c, pid, s):
        return self.pop.abar(c, pid, s, which=self.w)

    def phi(self, c, pid, s):
        return self.pop.phi(c, pid, s, which=self.w)

    def acen(self, c, pid, s):
        vals = []
        for f in self.pop.fams_of(c):
            inv_n = np.array([1.0 / x.shape[-1] for x in self.pop.S[(c, f)]])   # menu sizes differ within a family
            M = (getattr(self.pop, self.w)[(c, f)][s[f]] - inv_n[s[f]][:, None, None]).mean(0)
            vals.append(self.pop.offd(self.pop.cosm(M), pid))
        return float(np.mean(vals))

    def h2(self, c, s):
        return float(np.mean([-np.log(np.clip(np.diag(self.mat(c, s, f)), 1e-300, None)).mean() for f in self.pop.fams_of(c)]))

    def top(self, c, pid, s):
        vals = []
        for f in self.pop.fams_of(c):
            am = np.stack([x.argmax(-1) for x in self.pop.mix(c, f)])[s[f]][:, pid]        # (I, k)
            mask = ~np.eye(len(pid), dtype=bool) & (pid[:, None] != pid[None, :])
            vals.append((am[:, :, None] == am[:, None, :])[:, mask].mean())
        return float(np.mean(vals))

    def align_tab(self, c, base_c, guess_c):
        pop = self.pop
        return {f: np.array([[cosv(pop.mix(c, f)[j][i] - pop.mix(base_c, f)[j][i], pop.mix(guess_c, f)[j][i] - pop.mix(base_c, f)[j][i])
                              for i in range(pop.K)] for j in range(len(pop.insts[f]))]) for f in FAMS}

    def ret(self, c, base_c, s):
        pop, vals = self.pop, []
        ip = (lambda a, b: np.diagonal(cross_pairs(a, b))) if self.w == "C" else (lambda a, b: (a.mean(1) * b.mean(1)).sum(-1))
        for f in FAMS:
            X = np.stack([ip(a, b) - 1 / a.shape[-1] for a, b in zip(pop.S[(c, f)], pop.S[(base_c, f)])])[s[f]].mean(0)
            sc_ = np.stack([ip(a, a) - 1 / a.shape[-1] for a in pop.S[(c, f)]])[s[f]].mean(0)
            sb_ = np.stack([ip(b, b) - 1 / b.shape[-1] for b in pop.S[(base_c, f)]])[s[f]].mean(0)
            vals.append(X / np.sqrt(np.clip(sc_ * sb_, 1e-300, None)))
        return np.mean(vals, 0)


def _pp(Am, s):
    return np.mean([np.nanmean(Am[f][s[f]], 0) for f in FAMS], 0)


def _levels(m, conds, AL, base_c, allK, ALL):
    out = {}
    for c in conds:
        r = {"Abar": m.abar(c, allK, ALL), "Abar_centred": m.acen(c, allK, ALL), "top_choice_agreement": m.top(c, allK, ALL),
             "Phi": m.phi(c, allK, ALL), "H2": m.h2(c, ALL)}
        if c in AL:
            r["alignment"] = float(_pp(AL[c], ALL).mean())
        if base_c and c != base_c:
            r["retention"] = float(m.ret(c, base_c, ALL).mean())
        out[c] = r
    return out


def run6(base: str, seed: int = 0) -> dict:
    subst, stratum = _subst()
    omap = {o: k for k, o in enumerate(ORDERS)}
    ab, ab_raw = _steered(f"{base}/amend8_c1/coord_ablate.parquet")
    ab = ab.assign(frame="ablate_g")
    rd, rd_raw = _steered(f"{base}/amend9/ablate/random_ablate_r*.parquet")
    df = pd.concat([ab, rd, _phase2(base, subst, ORDERS)], ignore_index=True)
    df = df[df.persona.isin(subst)].copy()
    df["order_id"] = df.order_id.map(omap)
    conds = sorted(df.frame.unique())
    pop = Pop(per_order_cells(df), subst, stratum, conds, seed=seed)
    m = _Metrics(pop, "C")
    allK, ALL = np.arange(pop.K), pop.ALL
    rand = sorted(c for c in conds if c.startswith("ablate_r"))
    AL = {c: m.align_tab(c, "p2_pick", "p2_guess") for c in conds if c not in ("p2_pick", "p2_guess")}
    al = lambda c, pid, s: float(_pp(AL[c], s)[pid].mean())
    drop = lambda c, pid, s: al("p2_coord", pid, s) - al(c, pid, s)
    out = {"levels": _levels(m, conds, AL, "p2_pick", allK, ALL)}
    out["drops"] = {c: pop.two_level(lambda pid, s, c=c: drop(c, pid, s)) for c in ["ablate_g"] + rand}
    out["R1"] = pop.two_level(lambda pid, s: drop("ablate_g", pid, s) - max(drop(c, pid, s) for c in rand), "greater")
    lv = out["levels"]
    out["random_summary"] = {k: {"mean": float(np.mean([lv[c][k] for c in rand])), "min": float(np.min([lv[c][k] for c in rand])),
                                 "max": float(np.max([lv[c][k] for c in rand])), "ghat": lv["ablate_g"][k], "coord": lv["p2_coord"][k]}
                             for k in ("alignment", "retention", "Abar", "Abar_centred", "H2", "Phi", "top_choice_agreement")}
    dg = {c: drop(c, allK, ALL) for c in ["ablate_g"] + rand}
    out["rank_of_ghat_drop"] = int(1 + sum(dg[c] > dg["ablate_g"] for c in rand))
    out["omega"] = {**{"ablate_g": float(ab_raw.offmenu.mean())},
                    **{c: float(g.offmenu.mean()) for c, g in rd_raw.groupby("condition")}}
    return out


def run4(base: str, seed: int = 0) -> dict:
    subst, stratum = _subst()
    a9 = f"{base}/amend9/attn"
    # ---------------- 4a
    mass = np.load(f"{a9}/4a_mass.npy")                                       # (N, L, spans, 2)
    md = pd.read_parquet(f"{a9}/4a_meta.parquet")
    r = np.load(f"{a9}/r.npy")
    pi = SPAN_NAMES.index("persona")
    L = mass.shape[1]
    out: dict = {"r": r.tolist(), "span_lengths": {k: {"mean": float(md[k].mean()), "min": int(md[k].min()), "max": int(md[k].max())}
                                                   for k in ("n_persona", "n_user", "n_c_user", "n_c_template", "n_tokens")}}
    curves = {}
    for fr, g in md.groupby("frame"):
        ix = g.index.to_numpy()
        curves[fr] = {f"{sp}_{q}": mass[ix][:, :, k, j].mean(0).tolist()
                      for k, sp in enumerate(SPAN_NAMES) for j, q in enumerate(("dec", "user"))}
    out["curves_by_layer"] = curves
    pk = md.frame == "pick"
    expected = {}
    for k, sp in enumerate(SPAN_NAMES):
        mm = mass[pk.to_numpy()][:, :, k, 0].mean(0)
        for dose in (1, 2):
            rr = r ** dose
            post = mm * rr / (1 - mm + mm * rr)
            expected[f"{sp}_dose{dose}"] = {"pre_mass_16_63": float(mm[16:].mean()), "post_mass_16_63": float(post[16:].mean())}
    out["4b_expected_first_order_mass_change_in_pick"] = expected

    # K1: decision-position persona mass averaged over layers 16-63, coord - pick, paired over persona x instance
    k1 = md.assign(m=mass[:, 16:, pi, 0].mean(1))
    piv = {fr: k1[k1.frame == fr].set_index(["persona", "instance_id"]).m for fr in ("pick", "coord", "coord_human", "goal_nonsocial", "pick_long")}
    # ---------------- 4b
    st, raw = _steered(f"{a9}/4b_*.parquet")
    df = pd.concat([st, _phase2(base, subst, (RUN4_ORDER,))], ignore_index=True)
    df = df[df.persona.isin(subst)].copy()
    df["order_id"] = 0
    conds = sorted(df.frame.unique())
    pop = Pop(per_order_cells(df), subst, stratum, conds, seed=seed)
    m = _Metrics(pop, "M")
    allK, ALL = np.arange(pop.K), pop.ALL

    X = {fr: {f: np.array([[piv[fr].get((p, iid), np.nan) for iid in pop.insts[f]] for p in pop.P]) for f in FAMS} for fr in piv}
    dmass = lambda a, b: (lambda pid, s: float(np.mean([np.nanmean((X[a][f] - X[b][f])[pid][:, s[f]]) for f in FAMS])))
    out["K1"] = {**pop.two_level(dmass("coord", "pick"), "less"),
                 "mass_pick": float(np.nanmean([X["pick"][f] for f in FAMS])), "mass_coord": float(np.nanmean([X["coord"][f] for f in FAMS]))}
    out["4a_contrasts_vs_pick"] = {fr: pop.two_level(dmass(fr, "pick")) for fr in ("coord_human", "goal_nonsocial", "pick_long")}

    AL = {c: m.align_tab(c, "p2_pick", "p2_guess") for c in conds if c not in ("p2_pick", "p2_guess")}
    out["levels"] = _levels(m, conds, AL, "p2_pick", allK, ALL)
    gap = lambda c, pid, s: (m.abar(c, pid, s) - m.abar("p2_pick", pid, s)) / (m.abar("p2_coord", pid, s) - m.abar("p2_pick", pid, s))
    eff = {}
    for c in ("knock_r", "knock_r2", "ctrl_user", "ctrl_template"):
        eff[c] = {"gap_closed": pop.two_level(lambda pid, s, c=c: gap(c, pid, s)),
                  "dAbar_vs_pick": pop.two_level(lambda pid, s, c=c: m.abar(c, pid, s) - m.abar("p2_pick", pid, s)),
                  "dAbar_centred_vs_pick": pop.two_level(lambda pid, s, c=c: m.acen(c, pid, s) - m.acen("p2_pick", pid, s)),
                  "alignment": pop.two_level(lambda pid, s, c=c: float(_pp(AL[c], s)[pid].mean())),
                  "dH2_vs_pick": pop.two_level(lambda pid, s, c=c: m.h2(c, s) - m.h2("p2_pick", s))}
    eff["restore"] = {"dAbar_vs_coord": pop.two_level(lambda pid, s: m.abar("restore", pid, s) - m.abar("p2_coord", pid, s)),
                      "dAbar_centred_vs_coord": pop.two_level(lambda pid, s: m.acen("restore", pid, s) - m.acen("p2_coord", pid, s)),
                      "dAlignment_vs_coord": pop.two_level(lambda pid, s: float((_pp(AL["restore"], s) - _pp(AL["p2_coord"], s))[pid].mean())),
                      "dH2_vs_coord": pop.two_level(lambda pid, s: m.h2("restore", s) - m.h2("p2_coord", s))}
    out["effects"] = eff
    out["K2"] = pop.two_level(lambda pid, s: gap("knock_r", pid, s) - max(gap("ctrl_user", pid, s), gap("ctrl_template", pid, s)), "greater")
    out["K3"] = pop.two_level(lambda pid, s: m.abar("restore", pid, s) - m.abar("p2_coord", pid, s), "less")
    out["omega"] = {c: float(g.offmenu.mean()) for c, g in raw.groupby("condition")}
    out["n_layers"] = L
    return out
