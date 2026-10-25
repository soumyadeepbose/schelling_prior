"""PREREG Amendment 8 Part A, activation items (A8-A11) on cached residuals (Phase 2 layer 32, Amendment 5 sweep,
Phase 2 layers 46-53, Amendment 8 C0 recapture, Amendment 6 steered readouts, pricing pilot)."""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from .amend6 import split
from .analysis_amend3 import _ci
from .analysis_amend5 import COLS, core, substantive
from .analysis_amend7 import WAuc, signfree
from .analysis_phase2 import FAMS, NB, pilot_instances

F13 = ["F1", "F3"]
EXTRA = [46, 47, 49, 50, 51, 53]


def _unit(v):
    return v / np.linalg.norm(v)


def _codes(inst_ids, fams_of):
    ids = sorted(set(inst_ids))
    ix = {i: k for k, i in enumerate(ids)}
    code = np.array([ix[i] for i in inst_ids])
    groups = [[ix[i] for i in ids if fams_of[i] == f] for f in sorted(set(fams_of.values())) if any(fams_of[i] == f for i in ids)]
    return code, len(ids), groups


def layer_cells(base: str):
    """layer -> (meta DataFrame, X float32) for order-0 / paraphrase-(0,0) cells of the 23 personas (all frames present),
    plus the C0 pick paraphrase-1 cells per layer."""
    subst, _ = substantive()
    pil = pilot_instances()
    out = {}
    idx = pd.read_parquet(f"{base}/amend5_sweep/index.parquet")
    acts = np.load(f"{base}/amend5_sweep/acts.npy", mmap_mode="r")
    layers = json.load(open(f"{base}/amend5_sweep/summary.json"))["layers"]
    for li, L in enumerate(layers):
        out[L] = (idx, np.asarray(acts[:, li], np.float32))
    idx0 = pd.read_parquet(f"{base}/amend8_c0/sweep_l0_3_index.parquet")
    a0 = np.load(f"{base}/amend8_c0/sweep_l0_3_acts.npy", mmap_mode="r")
    for li, L in enumerate([0, 1, 2, 3]):
        out[L] = (idx0, np.asarray(a0[:, li], np.float32))
    for fn in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet")):
        d = pd.read_parquet(fn, columns=COLS)
        z = np.load(fn[:-8] + "_extra.npz")
        dd = d.iloc[z["rows"]].reset_index(drop=True)
        m = (dd.persona.isin(subst) & ~dd.instance_id.isin(pil) & dd.frame.isin(("pick", "guess", "coord", "anti"))).to_numpy()
        for L in EXTRA:
            prev = out.get(L)
            meta, X = dd[m].reset_index(drop=True), z[f"l{L}"][m].astype(np.float32)
            out[L] = (meta, X) if prev is None else (pd.concat([prev[0], meta], ignore_index=True), np.concatenate([prev[1], X]))
    s1 = json.load(open(f"{base}/amend8_c0/summary.json"))["pick_par1"]["layers"]
    ip = pd.read_parquet(f"{base}/amend8_c0/pick_par1_index.parquet")
    ap = np.load(f"{base}/amend8_c0/pick_par1_acts.npy", mmap_mode="r")
    par1 = {L: (ip, np.asarray(ap[:, li], np.float32)) for li, L in enumerate(s1)}
    return out, par1


def a8(base: str, seed: int = 0) -> dict:
    sp = split()
    half = {iid: h for f in sp for h in ("A", "B") for iid in sp[f][h]}
    fam_of = {iid: f for f in sp for h in ("A", "B") for iid in sp[f][h]}
    cells, par1 = layer_cells(base)
    rng = np.random.default_rng(seed)
    rnd = np.random.default_rng(8).standard_normal((100, 5120)).astype(np.float32)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    res, draws_le32 = {}, {}
    W = None
    for L in sorted(cells):
        meta, X = cells[L]
        h = meta.instance_id.map(half).to_numpy()
        ca = meta.frame.isin(("coord", "anti")).to_numpy() & meta.family.isin(F13).to_numpy()
        A = ca & (h == "A")
        B = ca & (h == "B")
        v = X[A & (meta.frame == "coord").to_numpy()].mean(0) - X[A & (meta.frame == "anti").to_numpy()].mean(0)
        sB = X[B] @ _unit(v)
        posB = (meta.frame[B] == "coord").to_numpy()
        code, nc, groups = _codes(meta.instance_id[B].tolist(), fam_of)
        if W is None or len(W[0]) != nc:
            W = [np.zeros(nc) for _ in range(NB)]
            for w in W:
                for g in groups:
                    np.add.at(w, rng.choice(g, len(g)), 1)
        a = WAuc(sB, posB)
        est = a()
        dr = np.array([a(w[code]) for w in W])
        rand_auc = np.array([signfree(WAuc(X[B] @ r, posB)()) for r in rnd])
        r = {"auroc": est, "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975)),
             "random_q99": float(np.quantile(rand_auc, .99)), "random_median": float(np.median(rand_auc)), "n_B": int(B.sum())}
        if L in par1:   # control contrast: pick paraphrase 0 (sweep cells) vs pick paraphrase 1 (C0)
            m0 = (meta.frame == "pick").to_numpy()
            M1, X1 = par1[L]
            h0, h1 = meta.instance_id[m0].map(half).to_numpy(), M1.instance_id.map(half).to_numpy()
            X0 = X[m0]
            vc = X0[h0 == "A"].mean(0) - X1[h1 == "A"].mean(0)
            sc = np.concatenate([X0[h0 == "B"], X1[h1 == "B"]]) @ _unit(vc)
            pc = np.concatenate([np.ones((h0 == "B").sum(), bool), np.zeros((h1 == "B").sum(), bool)])
            r["control_pick_par0_vs_par1_auroc"] = WAuc(sc, pc)()
        res[str(L)] = r
        if L <= 32:
            draws_le32[L] = (est, dr)
    Ls = sorted(draws_le32)
    T8 = max(draws_le32[L][0] for L in Ls)
    mx = np.max(np.stack([draws_le32[L][1] for L in Ls]), 0)
    test = _ci(T8 - 0.95, mx - 0.95, "greater")
    return {"per_layer": res, "T8_max_auroc_layers_le_32": T8, "T8_argmax_layer": int(Ls[int(np.argmax([draws_le32[L][0] for L in Ls]))]),
            "test": {**test, "est": T8, "lo": test["lo"] + 0.95, "hi": test["hi"] + 0.95},
            "withdraw_goal_late_claim": bool(T8 >= 0.95)}


def a9(base: str, amend5: dict, seed: int = 0) -> dict:
    """Per-layer norms for normalizing the sweep series, and the layer-32 reconciliation."""
    cells, _ = layer_cells(base)
    norms = {}
    for L, (meta, X) in cells.items():
        pk = (meta.frame == "pick").to_numpy()
        Xp, mp = X[pk], meta[pk].reset_index(drop=True)
        dist = []
        for iid, g in mp.groupby("instance_id"):
            Z = Xp[g.index.to_numpy()]
            sq = (Z ** 2).sum(1)
            D = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))
            dist.append(D[~np.eye(len(Z), dtype=bool)].mean())
        norms[str(L)] = {"mean_resid_norm_pick": float(np.linalg.norm(Xp, axis=1).mean()), "mean_between_persona_dist_pick": float(np.mean(dist))}
    sw = amend5["sweep"]
    series = {}
    for L, v in sw.items():
        n1, n2 = norms[L]["mean_resid_norm_pick"], norms[L]["mean_between_persona_dist_pick"]
        series[L] = {"W3_a": v["W3_a"], "W3_b": v["W3_b"], "W2_contrast": v["W2_contrast_F13"],
                     "W3_a_per_norm": {k: v["W3_a"][k] / n1 for k in ("est", "lo", "hi")},
                     "W3_b_per_norm": {k: v["W3_b"][k] / n1 for k in ("est", "lo", "hi")},
                     "W3_a_per_spread": {k: v["W3_a"][k] / n2 for k in ("est", "lo", "hi")},
                     "W3_b_per_spread": {k: v["W3_b"][k] / n2 for k in ("est", "lo", "hi")}, **norms[L]}
    # reconciliation at layer 32 on the stored residuals
    subst, stratum = substantive()
    pil = pilot_instances()
    rows, X = [], []
    for fn in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet")):
        d = pd.read_parquet(fn, columns=COLS)
        m = d.persona.isin(subst) & ~d.instance_id.isin(pil) & d.frame.isin(("pick", "guess", "coord", "anti"))
        a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        rows.append(d[m].reset_index(drop=True))
        X.append(np.asarray(a[d[m].index.to_numpy()], np.float32))
    meta, X = pd.concat(rows, ignore_index=True), np.concatenate(X)
    insts = {f: sorted(meta[meta.family == f].instance_id.unique()) for f in FAMS}
    kidx = {p: i for i, p in enumerate(subst)}

    def H_of(mask):
        H = {}
        mm = meta[mask].reset_index(drop=True)
        Xm = X[mask]
        for (fr, f), g in mm.groupby(["frame", "family"]):
            ii = {iid: j for j, iid in enumerate(insts[f])}
            A = np.zeros((len(insts[f]), len(subst), X.shape[1]), np.float64)
            C = np.zeros((len(insts[f]), len(subst)))
            jj, kk = g.instance_id.map(ii).to_numpy(), g.persona.map(kidx).to_numpy()
            np.add.at(A, (jj, kk), Xm[g.index.to_numpy()])
            np.add.at(C, (jj, kk), 1)
            H[(fr, f)] = (A / C[..., None]).astype(np.float32)
        return H

    def run(mask, s):
        r = core(H_of(mask), insts, subst, stratum, nb=200, nsplit=50, seed=s)
        return {"W2_contrast": r["W2"]["contrast_F13"]["est"], "W3_b": r["W3"]["b_coord_minus_anti_F13"]["est"], "W3_a": r["W3"]["a_coord_proj"]["est"]}

    o0 = (meta.order_id == 0).to_numpy()
    rec = {"sweep_l32_reported": {"W2_contrast": sw["32"]["W2_contrast_F13"]["est"], "W3_b": sw["32"]["W3_b"]["est"], "W3_a": sw["32"]["W3_a"]["est"]},
           "stored_order0_par00": run(o0 & (meta.persona_par == 0).to_numpy() & (meta.frame_par == 0).to_numpy(), 32)}
    for pp in (0, 1):
        for fp in (0, 1):
            pm = (meta.persona_par == pp).to_numpy() & (meta.frame_par == fp).to_numpy()
            rec[f"par{pp}{fp}_order0"] = run(o0 & pm, 32)
            rec[f"par{pp}{fp}_4orders"] = run(pm, 32)
    rec["all16_primary"] = run(np.ones(len(meta), bool), 32)
    return {"series": series, "reconciliation_l32": rec}


def a10(base: str) -> dict:
    sp = split()
    half = {iid: h for f in sp for h in ("A", "B") for iid in sp[f][h]}
    vec = np.load(f"{base}/amend6/directions.npz")
    info = json.load(open(f"{base}/amend6/directions.json"))
    g = vec["ghat"].astype(np.float64)
    subst, _ = substantive()
    pil = pilot_instances()
    norms, dcell, dcell_cos, keyed = [], [], [], {}
    for fn in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet")):
        d = pd.read_parquet(fn, columns=COLS)
        d = d[d.persona.isin(subst) & ~d.instance_id.isin(pil) & d.frame.isin(("pick", "coord"))]
        a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        X = np.asarray(a[d.index.to_numpy()], np.float64)
        d = d.reset_index(drop=True)
        pk = (d.frame == "pick").to_numpy()
        norms.append(np.linalg.norm(X[pk & (d.instance_id.map(half) == "B").to_numpy()], axis=1))
        key = list(zip(d.persona, d.instance_id, d.order_id, d.persona_par, d.frame_par))
        P = {k: i for k, i, f in zip(key, range(len(d)), d.frame) if f == "pick"}
        for i, (k, f) in enumerate(zip(key, d.frame)):
            if f == "coord" and k in P:
                dv = X[i] - X[P[k]]
                dcell.append(np.linalg.norm(dv))
                dcell_cos.append(dv @ g / np.linalg.norm(dv))
                acc = keyed.setdefault((k[0], k[1]), [0.0, 0])
                acc[0] = acc[0] + dv
                acc[1] += 1
    avg = np.stack([v[0] / v[1] for v in keyed.values()])
    popmean = avg.mean(0)
    mean_norm = float(np.concatenate(norms).mean())
    st = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/amend6/steer_shard*.parquet"))], ignore_index=True)

    def sm(lp):
        x = np.asarray(lp, float)
        e = np.exp(x - x.max())
        return e / e.sum()
    st["p"] = st.logprobs.apply(sm)
    base0 = {(r.frame, r.persona, r.instance_id, r.order_id): r.p for r in st[st.part == "dose0"].itertuples()}
    tvkl = {}
    for (part, dr, dose), g_ in st[st.part != "dose0"].groupby(["part", "direction", "dose"]):
        tv, kl = [], []
        for r in g_.itertuples():
            q = base0[(r.frame, r.persona, r.instance_id, r.order_id)]
            tv.append(0.5 * np.abs(r.p - q).sum())
            kl.append(float((r.p * (np.log(np.clip(r.p, 1e-300, None)) - np.log(np.clip(q, 1e-300, None)))).sum()))
        kind = "g" if dr == "g" else ("m53" if dr == "m53" else "random")
        e = tvkl.setdefault(f"{part}|{kind}|{dose}", {"tv": [], "kl": []})
        e["tv"] += tv
        e["kl"] += kl
    tvkl = {k: {"median_tv": float(np.median(v["tv"])), "median_kl_nats": float(np.median(v["kl"])), "n": len(v["tv"])} for k, v in tvkl.items()}
    u = info["u"]
    return {"u": u, "mean_resid_norm_l32_pick_halfB": mean_norm, "u_over_mean_norm": u / mean_norm,
            "u53": info["u53"], "dcoord_cell_matched_mean_norm": float(np.mean(dcell)),
            "dcoord_16cell_avg_mean_norm": float(np.linalg.norm(avg, axis=1).mean()),
            "dcoord_population_mean_norm": float(np.linalg.norm(popmean)),
            "cos_dcoord_g_cell_mean": float(np.mean(dcell_cos)),
            "cos_dcoord_g_16cell_mean": float(np.mean(avg @ g / np.linalg.norm(avg, axis=1))),
            "cos_population_mean_dcoord_g": float(popmean @ g / np.linalg.norm(popmean)),
            "steered_vs_unsteered": tvkl, "H2_units": "nats (np.log)"}


def a11(base: str, seed: int = 0) -> dict:
    subst, _ = substantive()
    sp = split()
    half = {iid: h for f in sp for h in ("A", "B") for iid in sp[f][h]}
    fam_of = {iid: f for f in sp for h in ("A", "B") for iid in sp[f][h]}
    pil = pilot_instances()
    vec = np.load(f"{base}/amend6/directions.npz")
    g = vec["ghat"]
    # pass 1: half-A means for the template direction and the paraphrase-0 g
    acc = {}
    metas = []
    for fn in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet")):
        d = pd.read_parquet(fn, columns=COLS)
        d["half"] = d.instance_id.map(half)
        d = d[d.persona.isin(subst) & d.half.notna() & d.frame.isin(("pick", "guess", "coord", "anti"))]
        metas.append((fn, d))
        a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        for (fr, fp), gg in d[d.half == "A"].groupby(["frame", "frame_par"]):
            rows = gg.index.to_numpy()
            s = np.zeros(a.shape[1])
            for k in range(0, len(rows), 4096):
                s += np.asarray(a[rows[k: k + 4096]], np.float64).sum(0)
            n0, s0 = acc.get((fr, fp), (0, 0.0))
            acc[(fr, fp)] = (n0 + len(rows), s0 + s)
    mean = lambda k: acc[k][1] / acc[k][0]
    tmpl = _unit(mean(("pick", 0)) - mean(("pick", 1))).astype(np.float32)
    g_fp0 = _unit(mean(("guess", 0)) - mean(("pick", 0))).astype(np.float32)
    V = np.stack([g, tmpl, g_fp0])
    rows, P = [], []
    for fn, d in metas:
        d = d[d.half == "B"]
        a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        r_ = d.index.to_numpy()
        P.append(np.concatenate([np.asarray(a[r_[k: k + 4096]], np.float32) @ V.T for k in range(0, len(r_), 4096)]))
        rows.append(d.reset_index(drop=True))
    meta, P = pd.concat(rows, ignore_index=True), np.concatenate(P)
    rng = np.random.default_rng(seed)
    code, nc, groups = _codes(meta.instance_id.tolist(), fam_of)
    W = []
    for _ in range(NB):
        w = np.zeros(nc)
        for gr in groups:
            np.add.at(w, rng.choice(gr, len(gr)), 1)
        W.append(w)
    pos = (meta.frame != "pick").to_numpy()

    def auc(sel, col, posv, sign_free=False):
        a = WAuc(P[sel, col], posv[sel])
        est, dr = a(), np.array([a(w[code][sel]) for w in W])
        if sign_free:
            est, dr = signfree(est), np.maximum(dr, 1 - dr)
        return est, dr
    allc = np.ones(len(meta), bool)
    eg, dg = auc(allc, 0, pos)
    et, dt = auc(allc, 1, pos, sign_free=True)
    pk = (meta.frame == "pick").to_numpy()
    eo, do = auc(pk, 1, (meta.frame_par == 0).to_numpy())
    out = {"i_template": {"own_heldout_auroc_pick_par0_vs_par1": {"est": eo, "lo": float(np.quantile(do, .025)), "hi": float(np.quantile(do, .975))},
                          "partner_vs_pick_signfree": {"est": et, "lo": float(np.quantile(dt, .025)), "hi": float(np.quantile(dt, .975))},
                          "ghat_partner_vs_pick": {"est": eg, "lo": float(np.quantile(dg, .025)), "hi": float(np.quantile(dg, .975))},
                          "test_ghat_minus_template": _ci(eg - et, dg - dt, "greater"),
                          "cos_template_ghat": float(tmpl @ g)}}
    fp1 = (meta.frame_par == 1).to_numpy()
    e3, d3 = auc(fp1, 2, pos)
    t3 = _ci(e3 - 0.90, d3 - 0.90, "greater")
    out["iii_paraphrase_heldout"] = {"auroc": e3, "lo": float(np.quantile(d3, .025)), "hi": float(np.quantile(d3, .975)),
                                     "test": {**t3, "est": e3, "lo": t3["lo"] + 0.90, "hi": t3["hi"] + 0.90}, "cos_gfp0_ghat": float(g_fp0 @ g)}
    # (ii) layers 0-3 (C0) and the sweep: g_L from half-A order-0 / paraphrase-(0,0) cells, partner vs pick on half B
    cells, _ = layer_cells(base)
    det = {}
    for L in sorted(cells):
        mt, X = cells[L]
        h = mt.instance_id.map(half).to_numpy()
        vL = X[(h == "A") & (mt.frame == "guess").to_numpy()].mean(0) - X[(h == "A") & (mt.frame == "pick").to_numpy()].mean(0)
        B = h == "B"
        det[str(L)] = WAuc(X[B] @ _unit(vL), (mt.frame[B] != "pick").to_numpy())()
    out["ii_layerwise_g_detection_auroc"] = det
    out["ii_embedding_layer"] = "AUROC = 0.5 by construction: every prompt's decision token is token id 271 (verified in C0)"
    # bag-of-words baseline on the user-message text
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from . import prompts
    cfg = prompts.load_frames()
    texts, y, grp = [], [], []
    for f in FAMS:
        for gm in prompts.load_games(f):
            if gm["instance_id"] in pil:
                continue
            for oid, perm in enumerate(prompts.orders_for(gm["instance_id"], len(gm["options"]), 4)):
                shown = [gm["options"][k] for k in perm]
                for fr in ("pick", "guess", "coord") + (("anti",) if f in F13 else ()):
                    for ft in cfg["frames"][fr]:
                        texts.append(prompts.user_message(ft, shown, cfg["answer_line"], (gm.get("stories") or {}).get("neutral")))
                        y.append(int(fr != "pick"))
                        grp.append(gm["instance_id"])
    Xb = CountVectorizer().fit_transform(texts)
    y, grp = np.array(y), np.array(grp)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xb, y, grp):
        oof[te] = LogisticRegression(max_iter=2000).fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]
    out["ii_bag_of_words_auroc"] = float(roc_auc_score(y, oof))
    out["ii_bag_of_words_n_prompts"] = int(len(y))
    # (iv) pricing: random directions, per-condition AUROC, template transfer
    d = pd.concat([pd.read_parquet(f, columns=["persona", "frame", "instance_id"]) for f in sorted(glob.glob(f"{base}/pricing_pilot/dist_shard*.parquet"))], ignore_index=True)
    Xp = np.concatenate([np.load(f[:-8] + "_l32.npy") for f in sorted(glob.glob(f"{base}/pricing_pilot/dist_shard*.parquet"))]).astype(np.float32)
    posp = (d.frame != "pick").to_numpy()
    sids = sorted(d.instance_id.unique())
    scode = d.instance_id.map({x: k for k, x in enumerate(sids)}).to_numpy()
    WS = [np.bincount(rng.integers(0, len(sids), len(sids)), minlength=len(sids))[scode] for _ in range(NB)]
    ag = WAuc(Xp @ g, posp)
    eg_p, dg_p = ag(), np.array([ag(w) for w in WS])
    rnd = np.random.default_rng(9).standard_normal((100, Xp.shape[1])).astype(np.float32)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    ra = np.array([signfree(WAuc(Xp @ r, posp)()) for r in rnd])
    q99 = float(np.quantile(ra, .99))
    per = {}
    for c in ("compete_R0", "compete_R1", "compete_R2", "compete_R3", "coord", "guess"):
        sel = d.frame.isin((c, "pick")).to_numpy()
        per[c] = WAuc((Xp @ g)[sel], posp[sel])()
    sel = d.frame.isin(("compete_R2", "compete_R3", "pick")).to_numpy()
    per["compete_R2_R3_pooled"] = WAuc((Xp @ g)[sel], posp[sel])()
    out["iv_pricing"] = {"ghat_pooled_auroc": eg_p, "random_q99": q99, "random_median": float(np.median(ra)),
                         "test_ghat_minus_q99": _ci(eg_p - q99, dg_p - q99, "greater"), "per_condition": per,
                         "template_transfer_signfree": signfree(WAuc(Xp @ tmpl, posp)())}
    return out


def analyze(base: str, amend5: dict) -> dict:
    return {"A8": a8(base), "A9": a9(base, amend5), "A10": a10(base), "A11": a11(base)}
