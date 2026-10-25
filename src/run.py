"""Score cells and write distribution records (Brief Section 9.2). Called from modal_app.py on a GPU."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from . import prompts, stats
from .readout_hf import VARIANT_RULE, Item, Scorer, surface_variants

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_personas(names: list[str] | None = None) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "personas.yaml").read_text(encoding="utf-8"))
    return {k: v for k, v in cfg.items() if names is None or k in names}


def load_model(path: str, dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), attn_implementation="sdpa",
                                                 device_map="cuda" if torch.cuda.is_available() else None)
    model.eval()
    return model, tok


def phase_cells(personas: dict, families: list[str], frames: list[str], n_orders: int,
                anti_families: tuple[str, ...] = ("F1", "F3"), max_instances: int | None = None,
                exclude_instances: set[str] | None = None) -> list[prompts.Cell]:
    """max_instances: seeded random subset per family (pilots only). exclude_instances: instance ids to skip."""
    import random

    pmap = {k: v["prompts"] for k, v in personas.items()}
    cells = []
    for fam in families:
        games = prompts.load_games(fam)
        if max_instances:
            games = sorted(random.Random(f"pilot:{fam}").sample(games, max_instances), key=lambda g: g["instance_id"])
        if exclude_instances:
            games = [g for g in games if g["instance_id"] not in exclude_instances]
        fr = list(frames) + (["anti"] if fam in anti_families and "anti" not in frames else [])
        cells += prompts.build_cells(pmap, fr, games, n_orders)
    return cells


def score_cells(scorer: Scorer, cells: list[prompts.Cell], meta: dict, out_path: pathlib.Path,
                strata: dict[str, str] | None = None, chunk: int = 2048, log=print,
                axis: torch.Tensor | None = None, act_layer: int | None = None,
                extra_layers: list[int] | None = None, extra_keep=None) -> dict:
    """Score cells; optionally save decision-position residuals.

    act_layer: residual saved for every cell (fp16, `<stem>_l<act_layer>.npy`, row = parquet row), plus its
    projection on the unit axis at that layer (`proj_l<act_layer>` column).
    extra_layers / extra_keep(cell): residuals at extra layers saved only for cells where extra_keep is true
    (`<stem>_extra.npz`, keyed by layer, with the parquet rows they belong to).
    """
    tok = scorer.tok
    t0 = time.time()
    n_tok = 0
    writer = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers = ([act_layer] if act_layer is not None else []) + list(extra_layers or [])
    acts, extra, extra_rows = [], {l: [] for l in extra_layers or []}, []
    a_hat = {l: (axis[l].float() / axis[l].float().norm()).numpy() for l in layers} if axis is not None else {}
    for s in range(0, len(cells), chunk):
        part = cells[s: s + chunk]
        items = []
        for i, c in enumerate(part):
            ids, P = prompts.render(tok, c.system, c.user)
            items.append(Item(ids, P, c.options, key=i))
            n_tok += len(ids) - P
        res = scorer.score(items, capture_layers=layers or None)
        recs = []
        for j, (c, it, r) in enumerate(zip(part, items, res)):
            row = s + j
            proj = {}
            if extra_layers and a_hat:
                proj["proj_extra"] = [float("nan")] * len(extra_layers)  # fixed schema across chunks
            if act_layer is not None:
                h = r.residuals[act_layer]
                acts.append(h.astype(np.float16))
                if act_layer in a_hat:
                    proj[f"proj_l{act_layer}"] = float(h @ a_hat[act_layer])
            if extra_layers and extra_keep is not None and extra_keep(c):
                extra_rows.append(row)
                for l in extra_layers:
                    extra[l].append(r.residuals[l].astype(np.float16))
                if a_hat:
                    proj["proj_extra"] = [float(r.residuals[l] @ a_hat[l]) for l in extra_layers]
            recs.append({**proj,
                **meta, "persona": c.persona, "stratum": (strata or {}).get(c.persona, ""), "partner": None,
                "disclosure": None, "frame": c.frame, "story": c.meta.get("story", "neutral"), "family": c.family,
                "instance_id": c.instance_id, "order_id": c.order_id, "persona_par": c.persona_par,
                "frame_par": c.frame_par, "paraphrase_ids": f"{c.persona_par}-{c.frame_par}",
                "options": c.options, "shown": c.shown, "logprobs": r.logprobs.tolist(),
                "logprobs_canonical": r.logprobs_canonical.tolist(), "p": r.p.tolist(),
                "offmenu_mass": r.offmenu, "pruned_bound": r.pruned_bound, "n_prompt_tokens": len(it.prompt_ids),
                "variant_logprobs": json.dumps(r.variant_logprobs),
            })
        table = pa.Table.from_pylist(recs)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        el = time.time() - t0
        log(f"  {s + len(part)}/{len(cells)} cells, {el:.0f}s, {(s + len(part)) / el:.1f} prompts/s, {n_tok / el:.0f} new prompt tok/s")
    if writer:
        writer.close()
    stem = str(out_path)[: -len(".parquet")]
    if acts:
        np.save(f"{stem}_l{act_layer}.npy", np.stack(acts))
    if extra_rows:
        np.savez(f"{stem}_extra.npz", rows=np.array(extra_rows), **{f"l{l}": np.stack(v) for l, v in extra.items()})
    el = time.time() - t0
    return {"cells": len(cells), "seconds": el, "prompts_per_s": len(cells) / el, "prompt_tok_per_s": n_tok / el}


@torch.no_grad()
def gof_gate(scorer: Scorer, cells: list[prompts.Cell], n_samples: int = 64, seed: int = 0, steer=None) -> list[dict]:
    """Section 3.3 readout gate: sampled answers at T = 1 vs the logit distribution, per cell.
    steer: (layer, vector) added at the decision position and every generated position (steered readout audit)."""
    model, tok = scorer.model, scorer.tok
    out = []
    for j, c in enumerate(cells):
        ids, P = prompts.render(tok, c.system, c.user)
        r = scorer.score([Item(ids, P, c.options)])[0] if steer is None else scorer.score_steered([Item(ids, P, c.options)], [steer])[0][0]
        table = {v: k for k, o in enumerate(c.options) for v in surface_variants(o)}
        x = torch.tensor([ids], device=model.device)
        torch.manual_seed(seed + j)
        hooks = []
        if steer is not None:
            def hk(mod, inp, o, v=steer[1]):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[:, -1:] += v.to(h.device, h.dtype)  # prompt pass: decision position; later steps: the new position
                return (h, *o[1:]) if isinstance(o, tuple) else h
            hooks.append(model.model.layers[steer[0]].register_forward_hook(hk))
        try:
            g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                               num_return_sequences=n_samples, max_new_tokens=16)
        finally:
            for h in hooks:
                h.remove()
        counts = np.zeros(len(c.options) + 1)
        for row in g[:, x.shape[1]:]:
            text = tok.decode(row, skip_special_tokens=False)
            ans = text.split("<|im_end|>")[0] if "<|im_end|>" in text else None
            counts[table.get(ans, len(c.options)) if ans is not None else len(c.options)] += 1
        probs = np.append(np.exp(r.logprobs), max(r.offmenu, 0.0))
        res = stats.gof_test(counts, probs, seed=j)
        out.append({"instance_id": c.instance_id, "frame": c.frame, "persona": c.persona, "counts": counts.tolist(),
                    "probs": probs.tolist(), **res})
    return out


def summarize_gof(rows: list[dict], alpha: float = 0.05) -> dict:
    """Gate: rejections at alpha consistent with the nominal rate (one-sided binomial test, p > 0.01)."""
    from scipy.stats import binomtest

    k = sum(r["p"] < alpha for r in rows)
    b = binomtest(k, len(rows), alpha, alternative="greater")
    return {"cells": len(rows), "rejections": k, "expected": alpha * len(rows), "binom_p": b.pvalue,
            "pass": bool(b.pvalue > 0.01), "variant_rule": VARIANT_RULE}
