"""Exact option readout under transformers with prefix reuse (Brief Sections 2.1, 4.5).

score(option a) = log P(tok(variant) <eot> | prompt), merged over the declared surface variants of a
by log-sum-exp (rule `VARIANT_RULE`, recorded in PREREG.md). Off-menu mass omega = 1 - sum over all
variants of all options; it is reported, never renormalized away.

Three passes per batch of prompts that share one persona system block:
  pass 0  system block -> KV, computed once per (system text, intervention) and reused;
  pass 1  the rest of each prompt (right-padded) on top of the cached system block -> next-token
          distribution at the decision position (last prompt token), KV kept;
  pass 2  every non-leaf node of a token trie of all variants of all options, packed with a 4D
          tree mask (each node sees the prompt and its own ancestors) and explicit position ids.
<eot> leaves need no forward pass: their log-probs come from their parent's logits.
Depth-0 trie nodes with P(first token) < prune_eps are skipped (the canonical form never is);
the skipped mass is bounded above by sum P(first token) and reported as `pruned_bound`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import DynamicCache

VARIANT_RULE = "v1: case in {as written, lower, Capitalized, UPPER} x {x, x., ' x', \"x\", 'x'}"


def surface_variants(option: str) -> list[str]:
    """Declared surface forms of one option; index 0 is the option exactly as written."""
    cases = [option, option.lower(), option[:1].upper() + option[1:], option.upper()]
    out: list[str] = []
    for c in dict.fromkeys(cases):
        for v in (c, c + ".", " " + c, f'"{c}"', f"'{c}'"):
            if v not in out:
                out.append(v)
    return out


@dataclass
class Item:
    prompt_ids: list[int]
    prefix_len: int
    options: list[str]
    key: object = None  # caller's handle, returned untouched


@dataclass
class Readout:
    key: object
    options: list[str]
    logprobs: np.ndarray          # merged over variants, per option (natural log)
    logprobs_canonical: np.ndarray  # as-written form only
    variant_logprobs: list[list[float]]
    offmenu: float
    pruned_bound: float
    decision_lp_top: list = field(default_factory=list)  # top next-token (id, logprob) for audits
    residuals: dict = field(default_factory=dict)          # layer -> np.ndarray (d,) at decision position

    @property
    def p(self) -> np.ndarray:
        x = self.logprobs - self.logprobs.max()
        e = np.exp(x)
        return e / e.sum()


class _Trie:
    def __init__(self):
        self.tok: list[int] = []
        self.parent: list[int] = []
        self.depth: list[int] = []
        self.children: list[dict[int, int]] = []
        self.roots: dict[int, int] = {}
        self.paths: list[list[int]] = []  # one per inserted sequence

    def insert(self, seq: list[int]) -> int:
        path, cur, kids = [], -1, self.roots
        for t in seq:
            if t not in kids:
                u = len(self.tok)
                self.tok.append(t)
                self.parent.append(cur)
                self.depth.append(0 if cur < 0 else self.depth[cur] + 1)
                self.children.append({})
                kids[t] = u
            cur = kids[t]
            path.append(cur)
            kids = self.children[cur]
        self.paths.append(path)
        return len(self.paths) - 1


class Scorer:
    def __init__(self, model, tok, eot_token: str = "<|im_end|>", prune_eps: float = 1e-6,
                 max_batch_tokens: int = 16384, max_batch_size: int = 64):
        self.model = model
        self.tok = tok
        self.eot = tok.convert_tokens_to_ids(eot_token)
        self.prune_eps = prune_eps
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self._var_ids: dict[str, list[int]] = {}
        self._prefix_cache: dict[tuple, list] = {}
        self.cache_tag = None  # set by interventions so cached system-block KV is never reused across them
        self.decision_delta: dict[int, torch.Tensor] | None = None  # layer -> added to the decision-position residual
        self.steer: tuple[int, torch.Tensor] | None = None  # (layer, vector) at the decision position and every option-
                                                            # continuation position (reference path for steering)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eot

    # ------------------------------------------------------------ helpers
    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def variant_ids(self, s: str) -> list[int]:
        if s not in self._var_ids:
            self._var_ids[s] = self.tok(s, add_special_tokens=False)["input_ids"] + [self.eot]
        return self._var_ids[s]

    def _neg(self):
        return torch.finfo(self.dtype).min

    def _prefix_kv(self, prefix: tuple[int, ...]):
        key = (prefix, self.cache_tag)
        if key not in self._prefix_cache:
            if len(self._prefix_cache) > 64:
                self._prefix_cache.clear()
            ids = torch.tensor([prefix], device=self.device)
            out = self.model.model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
            kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
            self._prefix_cache[key] = kv
        return self._prefix_cache[key]

    def _build_cache(self, prefix: tuple[int, ...], B: int) -> DynamicCache:
        if not prefix:
            return DynamicCache()
        kv = self._prefix_kv(prefix)
        return DynamicCache(ddp_cache_data=[(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in kv])

    # ------------------------------------------------------------ main entry
    @torch.no_grad()
    def score(self, items: list[Item], capture_layers: list[int] | None = None) -> list[Readout]:
        groups: dict[tuple, list[int]] = {}
        for i, it in enumerate(items):
            groups.setdefault(tuple(it.prompt_ids[: it.prefix_len]), []).append(i)
        results: list[Readout | None] = [None] * len(items)
        for prefix, idx in groups.items():
            prepared = [self._prepare(items[i]) for i in idx]
            widths = [len(items[i].prompt_ids) - len(prefix) + len(prepared[j][0].tok) for j, i in enumerate(idx)]
            order = sorted(range(len(idx)), key=lambda j: (widths[j], items[idx[j]].prompt_ids))
            batches, cur, width = [], [], 0
            for j in order:
                if cur and (len(cur) + 1 > self.max_batch_size or max(width, widths[j]) * (len(cur) + 1) > self.max_batch_tokens):
                    batches.append(cur)
                    cur, width = [], 0
                cur.append(j)
                width = max(width, widths[j])
            if cur:
                batches.append(cur)
            for batch in batches:
                outs = self._score_batch(prefix, [items[idx[b]] for b in batch], [prepared[b] for b in batch], capture_layers)
                for b, o in zip(batch, outs):
                    results[idx[b]] = o
        return results  # type: ignore[return-value]

    def _prepare(self, it: Item):
        trie = _Trie()
        table: list[list[int]] = []  # option -> list of path ids
        seen: dict[str, int] = {}
        for k, opt in enumerate(it.options):
            row = []
            for v in surface_variants(opt):
                if v in seen and seen[v] != k:
                    raise ValueError(f"surface variant {v!r} collides between options {seen[v]} and {k}")
                seen[v] = k
                row.append(trie.insert(self.variant_ids(v)))
            table.append(row)
        return trie, table

    @torch.no_grad()
    def capture(self, items: list[Item], layers: list[int]) -> list[dict]:
        """Decision-position residuals only (pass 1, no option scoring); layer -> np.ndarray (d,) per item."""
        groups: dict[tuple, list[int]] = {}
        for i, it in enumerate(items):
            groups.setdefault(tuple(it.prompt_ids[: it.prefix_len]), []).append(i)
        results: list[dict | None] = [None] * len(items)
        for prefix, idx in groups.items():
            order = sorted(idx, key=lambda i: len(items[i].prompt_ids))
            batches, cur, width = [], [], 0
            for i in order:
                w = len(items[i].prompt_ids) - len(prefix)
                if cur and (len(cur) + 1 > self.max_batch_size or max(width, w) * (len(cur) + 1) > self.max_batch_tokens):
                    batches.append(cur)
                    cur, width = [], 0
                cur.append(i)
                width = max(width, w)
            if cur:
                batches.append(cur)
            for batch in batches:
                _, _, captured, _, _ = self._pass1(prefix, [items[i] for i in batch], layers)
                for b, i in enumerate(batch):
                    results[i] = {l: captured[l][b].numpy() for l in captured}
        return results  # type: ignore[return-value]

    def _pass1(self, prefix, items, capture_layers):
        """Rest of the prompt after the cached system block, with decision-position hooks."""
        dev = self.device
        B, P = len(items), len(prefix)
        rems = [it.prompt_ids[P:] for it in items]
        R = [len(r) for r in rems]
        assert min(R) >= 1, "every prompt needs at least one token after the system block"
        L1 = max(R)

        # ---------------- pass 1: rest of prompt
        ids1 = torch.full((B, L1), self.pad_id, dtype=torch.long)
        pos1 = torch.zeros((B, L1), dtype=torch.long)
        m1 = torch.zeros((B, 1, L1, P + L1), dtype=torch.bool)
        m1[:, :, :, :P] = True
        causal = torch.tril(torch.ones(L1, L1, dtype=torch.bool))
        for b, r in enumerate(rems):
            ids1[b, : R[b]] = torch.tensor(r)
            pos1[b, : R[b]] = torch.arange(P, P + R[b])
            m1[b, 0, : R[b], P : P + R[b]] = causal[: R[b], : R[b]]
            if P == 0:  # padded rows need at least one visible key
                m1[b, 0, R[b]:, 0] = True
        cache = self._build_cache(prefix, B)

        captured: dict[int, torch.Tensor] = {}
        hooks = []
        last = torch.tensor([r - 1 for r in R])
        deltas = dict(self.decision_delta or {})
        if self.steer is not None:
            l_, v_ = self.steer
            deltas[l_] = deltas[l_] + v_ if l_ in deltas else v_
        layers = self.model.model.layers
        for l in sorted(set(capture_layers or []) | set(deltas)):
            def hook(mod, inp, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                rows, cols = torch.arange(B, device=h.device), last.to(h.device)
                if l in deltas:
                    h = h.clone()
                    h[rows, cols] += deltas[l].to(h.device, h.dtype)
                if capture_layers and l in capture_layers:
                    captured[l] = h[rows, cols].float().cpu()
                if l in deltas:
                    return (h, *out[1:]) if isinstance(out, tuple) else h
            hooks.append(layers[l].register_forward_hook(hook))
        try:
            out1 = self.model.model(input_ids=ids1.to(dev), attention_mask=self._addmask(m1), position_ids=pos1.to(dev),
                                    past_key_values=cache, use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        return out1, last, captured, R, L1

    def _score_batch(self, prefix, items, prepared, capture_layers):
        dev = self.device
        B, P = len(items), len(prefix)
        out1, last, captured, R, L1 = self._pass1(prefix, items, capture_layers)
        h_dec = out1.last_hidden_state[torch.arange(B), last.to(dev)]
        lp0 = torch.log_softmax(self.model.lm_head(h_dec).float(), -1)  # (B, V)
        ctx = torch.zeros((B, P + L1), dtype=torch.bool)
        ctx[:, :P] = True
        for b in range(B):
            ctx[b, P: P + R[b]] = True
        sel = self._select(prepared, lp0)
        self._pass2(prepared, sel, ctx, [P + r for r in R], out1.past_key_values, self.steer)
        return self._assemble(items, prepared, sel, lp0, captured)

    def _select(self, prepared, lp0):
        """Which trie nodes to expand, given the decision-position log-probs."""
        node_lp: list[np.ndarray] = []
        inputs: list[list[int]] = []   # per item: trie node ids fed to pass 2, in DFS-compatible order
        pruned = np.zeros(len(prepared))
        keep_root: list[set[int]] = []
        for b, (trie, table) in enumerate(prepared):
            lp = np.full(len(trie.tok), -np.inf)
            canonical_roots = {trie.paths[row[0]][0] for row in table}
            keep = set()
            lp0b = lp0[b]
            for t, u in trie.roots.items():
                lp[u] = float(lp0b[t])
                if u in canonical_roots or np.exp(lp[u]) >= self.prune_eps:
                    keep.add(u)
                else:
                    pruned[b] += np.exp(lp[u])
            keep_root.append(keep)
            stack = sorted(keep, reverse=True)
            feed = []
            while stack:
                u = stack.pop()
                if trie.children[u]:
                    feed.append(u)
                    stack.extend(sorted(trie.children[u].values(), reverse=True))
            inputs.append(feed)
            node_lp.append(lp)
        return {"node_lp": node_lp, "inputs": inputs, "pruned": pruned, "keep_root": keep_root}

    def _pass2(self, prepared, sel, ctx, base_pos, cache, steer=None):
        """Feed the selected trie nodes. ctx (B, C): context keys every node may attend to; node u of item b sits at
        position base_pos[b] + depth(u). steer: (layer, vector) added to every position of this pass."""
        dev = self.device
        inputs, node_lp = sel["inputs"], sel["node_lp"]
        B, C = ctx.shape
        L2 = max((len(f) for f in inputs), default=0)
        if L2 == 0:
            return
        ids2 = torch.full((B, L2), self.pad_id, dtype=torch.long)
        pos2 = torch.zeros((B, L2), dtype=torch.long)
        m2 = torch.zeros((B, 1, L2, C + L2), dtype=torch.bool)
        m2[:, 0, :, :C] = ctx[:, None, :]
        for b, feed in enumerate(inputs):
            trie = prepared[b][0]
            slot = {u: i for i, u in enumerate(feed)}
            for i, u in enumerate(feed):
                ids2[b, i] = trie.tok[u]
                pos2[b, i] = base_pos[b] + trie.depth[u]
                a = u
                while a >= 0:
                    m2[b, 0, i, C + slot[a]] = True
                    a = trie.parent[a]
        hooks = []
        if steer is not None:
            hooks.append(self.model.model.layers[steer[0]].register_forward_hook(_add_hook(steer[1])))
        try:
            out2 = self.model.model(input_ids=ids2.to(dev), attention_mask=self._addmask(m2), position_ids=pos2.to(dev),
                                    past_key_values=cache, use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        h2 = out2.last_hidden_state
        for b, feed in enumerate(inputs):
            if not feed:
                continue
            trie = prepared[b][0]
            lg = self.model.lm_head(h2[b, : len(feed)]).float()
            lse = torch.logsumexp(lg, -1)
            rows, cols, kids = [], [], []
            for i, u in enumerate(feed):
                for t, c in trie.children[u].items():
                    rows.append(i)
                    cols.append(t)
                    kids.append(c)
            vals = (lg[rows, cols] - lse[rows]).cpu().numpy()
            node_lp[b][kids] = vals

    def _assemble(self, items, prepared, sel, lp0, captured):
        res = []
        for b, (trie, table) in enumerate(prepared):
            lp = sel["node_lp"][b]
            vlp = [[float(lp[trie.paths[pid]].sum()) if trie.paths[pid][0] in sel["keep_root"][b] else -np.inf
                    for pid in row] for row in table]
            merged = np.array([np.logaddexp.reduce(np.array(r)) for r in vlp])
            canon = np.array([r[0] for r in vlp])
            top = torch.topk(lp0[b], 5)
            res.append(Readout(
                key=items[b].key, options=items[b].options, logprobs=merged, logprobs_canonical=canon,
                variant_logprobs=vlp, offmenu=float(1.0 - np.exp(merged).sum()), pruned_bound=float(sel["pruned"][b]),
                decision_lp_top=list(zip(top.indices.tolist(), top.values.tolist())),
                residuals={l: captured[l][b].numpy() for l in captured},
            ))
        return res

    # ------------------------------------------------------------ steering with prompt reuse
    @torch.no_grad()
    def score_steered(self, items: list[Item], deltas: list) -> list[list[Readout]]:
        """Readouts with h <- h + v at layer l for each delta = (l, v), at the decision position and every option-continuation position.
        The prompt up to (not including) the decision token is computed once per item and reused for every delta.
        Returns out[i][k] for item i and delta k (None = no steering)."""
        groups: dict[tuple, list[int]] = {}
        for i, it in enumerate(items):
            groups.setdefault(tuple(it.prompt_ids[: it.prefix_len]), []).append(i)
        results: list[list] = [[None] * len(deltas) for _ in items]
        for prefix, idx in groups.items():
            prepared = {i: self._prepare(items[i]) for i in idx}
            widths = {i: len(items[i].prompt_ids) - len(prefix) + len(prepared[i][0].tok) for i in idx}
            order = sorted(idx, key=lambda i: (widths[i], items[i].prompt_ids))
            batches, cur, width = [], [], 0
            for i in order:
                if cur and (len(cur) + 1 > self.max_batch_size or max(width, widths[i]) * (len(cur) + 1) > self.max_batch_tokens):
                    batches.append(cur)
                    cur, width = [], 0
                cur.append(i)
                width = max(width, widths[i])
            if cur:
                batches.append(cur)
            for batch in batches:
                outs = self._steered_batch(prefix, [items[i] for i in batch], [prepared[i] for i in batch], deltas)
                for b, i in enumerate(batch):
                    results[i] = outs[b]
        return results

    def _steered_batch(self, prefix, items, prepared, deltas):
        dev = self.device
        B, P = len(items), len(prefix)
        rems = [it.prompt_ids[P:] for it in items]
        R = [len(r) for r in rems]
        assert min(R) >= 2, "steered scoring needs at least two tokens after the system block"
        LA = max(R) - 1
        # ---------------- pass A: rest of prompt without the decision token (shared by every delta)
        idsA = torch.full((B, LA), self.pad_id, dtype=torch.long)
        posA = torch.zeros((B, LA), dtype=torch.long)
        mA = torch.zeros((B, 1, LA, P + LA), dtype=torch.bool)
        mA[:, :, :, :P] = True
        causal = torch.tril(torch.ones(LA, LA, dtype=torch.bool))
        for b, r in enumerate(rems):
            ra = R[b] - 1
            idsA[b, :ra] = torch.tensor(r[:-1])
            posA[b, :ra] = torch.arange(P, P + ra)
            mA[b, 0, :ra, P: P + ra] = causal[:ra, :ra]
            if P == 0:
                mA[b, 0, ra:, 0] = True
        outA = self.model.model(input_ids=idsA.to(dev), attention_mask=self._addmask(mA), position_ids=posA.to(dev),
                                past_key_values=self._build_cache(prefix, B), use_cache=True)
        kvA = [(l.keys, l.values) for l in outA.past_key_values.layers]
        del outA
        # ---------------- per delta: decision token, then the option trie, both steered
        ctx = torch.zeros((B, P + LA + 1), dtype=torch.bool)
        ctx[:, :P] = True
        for b in range(B):
            ctx[b, P: P + R[b] - 1] = True
            ctx[b, P + LA] = True
        ids1 = torch.tensor([[r[-1]] for r in rems])
        pos1 = torch.tensor([[P + r - 1] for r in R])
        m1 = ctx[:, None, None, :].clone()
        out = [[None] * len(deltas) for _ in range(B)]
        for k, steer in enumerate(deltas):
            cache = DynamicCache(ddp_cache_data=kvA)
            hooks = [self.model.model.layers[steer[0]].register_forward_hook(_add_hook(steer[1]))] if steer is not None else []
            try:
                o1 = self.model.model(input_ids=ids1.to(dev), attention_mask=self._addmask(m1), position_ids=pos1.to(dev),
                                      past_key_values=cache, use_cache=True)
            finally:
                for h in hooks:
                    h.remove()
            lp0 = torch.log_softmax(self.model.lm_head(o1.last_hidden_state[:, 0]).float(), -1)
            sel = self._select(prepared, lp0)
            self._pass2(prepared, sel, ctx, [P + r for r in R], o1.past_key_values, steer)
            for b, r in enumerate(self._assemble(items, prepared, sel, lp0, {})):
                out[b][k] = r
        return out

    def _addmask(self, m: torch.Tensor) -> torch.Tensor:
        add = torch.zeros(m.shape, dtype=self.dtype)
        add.masked_fill_(~m, self._neg())
        return add.to(self.device)


def _add_hook(vec):
    """Forward hook adding vec to every position of a decoder layer's output."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + vec.to(h.device, h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return hook


@torch.no_grad()
def score_uncached(model, tok, item: Item, eot_token: str = "<|im_end|>", steer=None) -> list[list[float]]:
    """Reference: one full forward per (option, variant), no cache, no custom mask. Tests only.
    steer: (layer, vector) added from the decision position (last prompt token) onward."""
    eot = tok.convert_tokens_to_ids(eot_token)
    dev = next(model.parameters()).device
    out = []
    for opt in item.options:
        row = []
        for v in surface_variants(opt):
            cont = tok(v, add_special_tokens=False)["input_ids"] + [eot]
            ids = torch.tensor([item.prompt_ids + cont], device=dev)
            n = len(item.prompt_ids)
            hooks = []
            if steer is not None:
                def hk(mod, inp, o, v=steer[1], n=n):
                    h = (o[0] if isinstance(o, tuple) else o).clone()
                    h[:, n - 1:] += v.to(h.device, h.dtype)
                    return (h, *o[1:]) if isinstance(o, tuple) else h
                hooks.append(model.model.layers[steer[0]].register_forward_hook(hk))
            try:
                lg = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
            finally:
                for hh in hooks:
                    hh.remove()
            row.append(float(sum(lg[n - 1 + i, t] for i, t in enumerate(cont))))
        out.append(row)
    return out


@torch.no_grad()
def score_uncached_batched(model, tok, item: Item, eot_token: str = "<|im_end|>", steer=None, chunk: int = 32) -> list[list[float]]:
    """Reference: full uncached forward of every (option, variant) continuation, all variants of the item in right-padded
    batches with the standard HF attention mask (no cache, no custom mask). steer as in score_uncached."""
    eot = tok.convert_tokens_to_ids(eot_token)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eot
    dev = next(model.parameters()).device
    n = len(item.prompt_ids)
    conts, owner = [], []
    for k, opt in enumerate(item.options):
        for v in surface_variants(opt):
            conts.append(tok(v, add_special_tokens=False)["input_ids"] + [eot])
            owner.append(k)
    hooks = []
    if steer is not None:
        def hk(mod, inp, o, v=steer[1]):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[:, n - 1:] += v.to(h.device, h.dtype)
            return (h, *o[1:]) if isinstance(o, tuple) else h
        hooks.append(model.model.layers[steer[0]].register_forward_hook(hk))
    vals = []
    try:
        for s in range(0, len(conts), chunk):
            part = conts[s: s + chunk]
            m = max(len(c) for c in part)
            ids = torch.full((len(part), n + m), pad, dtype=torch.long)
            att = torch.zeros((len(part), n + m), dtype=torch.long)
            for r, c in enumerate(part):
                ids[r, : n + len(c)] = torch.tensor(item.prompt_ids + c)
                att[r, : n + len(c)] = 1
            h = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state[:, n - 1: n - 1 + m]
            lg = torch.log_softmax(model.lm_head(h).float(), -1)
            for r, c in enumerate(part):
                vals.append(float(sum(lg[r, i, t] for i, t in enumerate(c))))
    finally:
        for hh in hooks:
            hh.remove()
    out = [[] for _ in item.options]
    for k, v in zip(owner, vals):
        out[k].append(v)
    return out
