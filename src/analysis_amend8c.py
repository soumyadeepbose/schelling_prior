"""PREREG Amendment 8 Part C: steering positive controls (C1) and the cross-model baseline (C2). Descriptive."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import yaml

from .amend6 import ORDERS, split
from .analysis_amend3 import SUBST, cosv
from .analysis_amend8a import NB, Pop, _ci, per_order_cells
from .analysis_phase2 import FAMS, cross_pairs, pilot_instances
from .prompts import ROOT


def _sm(lp):
    x = np.asarray(lp, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def _subst():
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    stratum = {p: pcfg[p]["stratum"] for p in pcfg}
    return [p for p in pcfg if stratum[p] in SUBST and p != "facilitator"], stratum


def c1(base: str, seed: int = 0) -> dict:
    subst, stratum = _subst()
    sp = split()
    B = {iid for f in sp for iid in sp[f]["B"]}
    omap = {o: k for k, o in enumerate(ORDERS)}
    frames = []
    st = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/amend6/steer_shard*.parquet"))], ignore_index=True)
    d0 = st[st.part == "dose0"].assign(cond=lambda x: "a6_" + x.frame + "0")
    frames.append(d0[["persona", "family", "instance_id", "order_id", "cond", "logprobs"]])
    pc = pd.read_parquet(f"{base}/amend8_c1/pick_patch_dcoord.parquet")
    pc["cond"] = pc.condition.where(~pc.condition.str.startswith("r"), "rand") + "_" + pc.dose.astype(str)
    pc.loc[pc.condition.str.startswith("r"), "cond"] = pc.condition + "_1.0"
    frames.append(pc[["persona", "family", "instance_id", "order_id", "cond", "logprobs"]])
    ab = pd.read_parquet(f"{base}/amend8_c1/coord_ablate.parquet").assign(cond="ablate")
    frames.append(ab[["persona", "family", "instance_id", "order_id", "cond", "logprobs"]])
    cols = ["persona", "frame", "family", "instance_id", "order_id", "persona_par", "frame_par", "p"]
    ph = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet"))], ignore_index=True)
    ph = ph[ph.persona.isin(subst) & ph.instance_id.isin(B) & ph.order_id.isin(ORDERS) & (ph.persona_par == 0) & (ph.frame_par == 0)
            & ph.frame.isin(("pick", "coord", "guess"))]
    ph = ph.assign(cond="p2_" + ph.frame)
    df = pd.concat(frames, ignore_index=True)
    df["p"] = df.logprobs.apply(_sm)
    df = pd.concat([df[["persona", "family", "instance_id", "order_id", "cond", "p"]],
                    ph[["persona", "family", "instance_id", "order_id", "cond", "p"]]], ignore_index=True)
    df = df[df.persona.isin(subst)].copy()
    df["order_id"] = df.order_id.map(omap)
    df = df.rename(columns={"cond": "frame"})
    PO = per_order_cells(df)
    conds = sorted(df.frame.unique())
    pop = Pop(PO, subst, stratum, conds, seed=seed)
    K, ALL, allK = pop.K, pop.ALL, np.arange(pop.K)
    rand = [c for c in conds if c.startswith("r") and c.endswith("_1.0")]

    def h2(c, s):
        return float(np.mean([-np.log(np.clip(np.diag(pop.mat(c, s, f)), 1e-300, None)).mean() for f in FAMS]))

    def align(c, base_c):
        return {f: np.array([[cosv(pop.mix(c, f)[j][i] - pop.mix(base_c, f)[j][i], pop.mix("p2_guess", f)[j][i] - pop.mix(base_c, f)[j][i])
                              for i in range(K)] for j in range(len(pop.insts[f]))]) for f in FAMS}

    def ret_c(c, base_c, s):
        vals = []
        for f in FAMS:
            X = np.stack([np.diagonal(cross_pairs(a, b)) - 1 / a.shape[-1] for a, b in zip(pop.S[(c, f)], pop.S[(base_c, f)])])[s[f]].mean(0)
            sc_ = np.stack([np.diagonal(cross_pairs(a, a)) - 1 / a.shape[-1] for a in pop.S[(c, f)]])[s[f]].mean(0)
            sb_ = np.stack([np.diagonal(cross_pairs(b, b)) - 1 / b.shape[-1] for b in pop.S[(base_c, f)]])[s[f]].mean(0)
            vals.append(X / np.sqrt(np.clip(sc_ * sb_, 1e-300, None)))
        return np.mean(vals, 0)

    pp = lambda Am, s: np.mean([np.nanmean(Am[f][s[f]], 0) for f in FAMS], 0)
    AL = {c: align(c, "a6_pick0") for c in conds if not c.startswith(("p2_", "ablate"))}
    AL["ablate"] = align("ablate", "p2_pick")
    AL["p2_coord"] = align("p2_coord", "p2_pick")

    def metrics(c, pid, s, base_c):
        m = {"Abar": pop.abar(c, pid, s), "Phi": pop.phi(c, pid, s), "H2": h2(c, s)}
        if c in AL:
            m["align"] = float(pp(AL[c], s)[pid].mean())
        return m

    def rmean(pid, s, key):
        return float(np.mean([metrics(r, pid, s, "a6_pick0")[key] for r in rand]))

    out = {"levels": {c: {k: float(v) for k, v in metrics(c, allK, ALL, None).items()} for c in conds}}
    gap = lambda pid, s: pop.abar("a6_coord0", pid, s) - pop.abar("a6_pick0", pid, s)
    res = {}
    for c in ("patch_1.0", "dcoord_1.0", "dcoord_2.0"):
        res[c] = {"dAbar_vs_pick0": pop.two_level(lambda pid, s, c=c: pop.abar(c, pid, s) - pop.abar("a6_pick0", pid, s)),
                  "frac_of_pick_to_coord_gap": pop.two_level(lambda pid, s, c=c: (pop.abar(c, pid, s) - pop.abar("a6_pick0", pid, s)) / gap(pid, s)),
                  "alignment": pop.two_level(lambda pid, s, c=c: float(pp(AL[c], s)[pid].mean())),
                  "dPhi_vs_pick0": pop.two_level(lambda pid, s, c=c: pop.phi(c, pid, s) - pop.phi("a6_pick0", pid, s)),
                  "dH2_vs_pick0": pop.two_level(lambda pid, s, c=c: h2(c, s) - h2("a6_pick0", s))}
    res["random_1.0_mean"] = {"dAbar_vs_pick0": pop.two_level(lambda pid, s: rmean(pid, s, "Abar") - pop.abar("a6_pick0", pid, s)),
                              "alignment": pop.two_level(lambda pid, s: rmean(pid, s, "align")),
                              "dPhi_vs_pick0": pop.two_level(lambda pid, s: rmean(pid, s, "Phi") - pop.phi("a6_pick0", pid, s))}
    res["dcoord_1.0_minus_random"] = {"dAbar": pop.two_level(lambda pid, s: pop.abar("dcoord_1.0", pid, s) - rmean(pid, s, "Abar")),
                                      "alignment": pop.two_level(lambda pid, s: float(pp(AL["dcoord_1.0"], s)[pid].mean()) - rmean(pid, s, "align"))}
    res["reference_coord0_alignment"] = pop.two_level(lambda pid, s: float(pp(AL["a6_coord0"], s)[pid].mean()))
    res["ablate"] = {"dAbar_vs_coord": pop.two_level(lambda pid, s: pop.abar("ablate", pid, s) - pop.abar("p2_coord", pid, s)),
                     "alignment_ablate": pop.two_level(lambda pid, s: float(pp(AL["ablate"], s)[pid].mean())),
                     "alignment_coord": pop.two_level(lambda pid, s: float(pp(AL["p2_coord"], s)[pid].mean())),
                     "dAlignment": pop.two_level(lambda pid, s: float((pp(AL["ablate"], s) - pp(AL["p2_coord"], s))[pid].mean())),
                     "retention_ablate": pop.two_level(lambda pid, s: float(ret_c("ablate", "p2_pick", s)[pid].mean())),
                     "retention_coord": pop.two_level(lambda pid, s: float(ret_c("p2_coord", "p2_pick", s)[pid].mean())),
                     "dPhi_vs_coord": pop.two_level(lambda pid, s: pop.phi("ablate", pid, s) - pop.phi("p2_coord", pid, s)),
                     "dH2_vs_coord": pop.two_level(lambda pid, s: h2("ablate", s) - h2("p2_coord", s))}
    out["effects"] = res
    om = {}
    for nm, d_ in (("pick_patch_dcoord", pc), ("coord_ablate", ab)):
        for c, g in d_.groupby("cond" if "cond" in d_ else "condition"):
            om[f"{nm}|{c}"] = float(g.offmenu.mean())
    out["omega"] = om
    return out


def c2(base: str, base14: str, seed: int = 0) -> dict:
    subst, stratum = _subst()
    pil = pilot_instances()
    cols = ["persona", "frame", "family", "instance_id", "order_id", "p"]
    p32 = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet"))], ignore_index=True)
    p32 = p32[~p32.instance_id.isin(pil) & p32.frame.isin(("pick", "coord")) & p32.persona.isin(subst + ["assistant_paraphrase"])]
    p14 = pd.read_parquet(f"{base14}/dist_shard0.parquet", columns=cols)
    p14 = p14[~p14.instance_id.isin(pil)].assign(persona="qwen3_14b_assistant")
    PO = per_order_cells(pd.concat([p32, p14], ignore_index=True))
    names = subst + ["assistant_paraphrase", "qwen3_14b_assistant"]
    st = {**stratum, "assistant_paraphrase": "control", "qwen3_14b_assistant": "other_model"}
    pop = Pop(PO, names, st, ["pick", "coord"], seed=seed)
    i14, i32, iap = names.index("qwen3_14b_assistant"), names.index("assistant"), names.index("assistant_paraphrase")
    rng = np.random.default_rng(seed + 3)
    out = {}
    for fr in ("pick", "coord"):
        def stats_(s):
            cs = [pop.cosm(pop.mat(fr, s, f)) for f in FAMS]
            c = np.mean(cs, 0)
            sub = c[:len(subst), :len(subst)]
            within = sub[~np.eye(len(subst), dtype=bool)].mean()
            return np.array([c[i14, i32], c[iap, i32], within, c[i14, :len(subst)].mean(), (c[i14, :len(subst)] > 0).mean()])
        est = stats_(pop.ALL)
        dr = np.array([stats_(pop.resample()) for _ in range(NB)])
        keys = ["cos_14b_vs_32b_assistant", "cos_32b_assistant_paraphrase_vs_assistant", "within_32b_between_persona_Abar",
                "mean_cos_14b_assistant_vs_32b_personas"]
        out[fr] = {k: {"est": float(est[i]), "lo": float(np.quantile(dr[:, i], .025)), "hi": float(np.quantile(dr[:, i], .975))} for i, k in enumerate(keys)}
        c = np.mean([pop.cosm(pop.mat(fr, pop.ALL, f)) for f in FAMS], 0)
        out[fr]["cos_14b_vs_each_32b_persona"] = {p: float(c[i14, i]) for i, p in enumerate(subst)}
        pair = c[:len(subst), :len(subst)][np.triu_indices(len(subst), 1)]
        out[fr]["share_of_32b_persona_pairs_less_similar_than_14b_vs_32b_assistant"] = float((pair < c[i14, i32]).mean())
        out[fr]["C_self_14b"] = float(np.mean([np.diag(pop.mat(fr, pop.ALL, f))[i14] for f in FAMS]))
        out[fr]["C_self_32b_assistant"] = float(np.mean([np.diag(pop.mat(fr, pop.ALL, f))[i32] for f in FAMS]))
    out["dAbar_cross_model_coord_minus_pick"] = out["coord"]["cos_14b_vs_32b_assistant"]["est"] - out["pick"]["cos_14b_vs_32b_assistant"]["est"]
    return out


def analyze(base: str, base14: str) -> dict:
    return {"C1": c1(base), "C2": c2(base, base14)}
