"""Option-score Jacobian at the decision position (Brief Sections 2.7, 4.1, 8.1).

J[k] = d z_k / d h, where h is the output of decoder layer `layer` at the decision position (last prompt
token) and z_k is the variant-merged log-probability of option k, exactly as in readout_hf.

Only computation downstream of h is differentiated:
  A  system block + prompt, no grad (system block KV reused from the scorer's cache);
  B  trie nodes through layers 0..layer, no grad (they do not depend on h);
  C  [h, trie nodes] through layers layer+1..L-1 with grad, attending to the cached KV of every earlier
     position at those layers. The n options are batched as n copies of C; row k back-propagates z_k,
     so one backward pass yields all n rows of J.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import DynamicCache

from .readout_hf import Item, Scorer


def _run_layers(model, x, layer_ids, mask, pos, cache):
    pe = model.model.rotary_emb(x, pos)
    for i in layer_ids:
        out = model.model.layers[i](x, attention_mask=mask, position_ids=pos, past_key_values=cache,
                                    use_cache=True, position_embeddings=pe)
        x = out[0] if isinstance(out, tuple) else out
    return x


def option_jacobian(sc: Scorer, item: Item, layer: int) -> dict:
    model = sc.model
    model.requires_grad_(False)
    dev = sc.device
    n_layers = len(model.model.layers)
    P = item.prefix_len
    prefix, rem = tuple(item.prompt_ids[:P]), item.prompt_ids[P:]
    R = len(rem)
    trie, table = sc._prepare(item)

    # ---------------- A: prompt, capture h at the decision position
    with torch.no_grad():
        m1 = torch.zeros((1, 1, R, P + R), dtype=torch.bool)
        m1[..., :P] = True
        m1[0, 0, :, P:] = torch.tril(torch.ones(R, R, dtype=torch.bool))
        cache = sc._build_cache(prefix, 1)
        grab = {}

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            grab["h"] = h[0, -1].detach().clone()
        hd = model.model.layers[layer].register_forward_hook(hook)
        try:
            out1 = model.model(input_ids=torch.tensor([rem], device=dev), attention_mask=sc._addmask(m1),
                               position_ids=torch.arange(P, P + R, device=dev)[None], past_key_values=cache, use_cache=True)
        finally:
            hd.remove()
        lp0 = torch.log_softmax(model.lm_head(out1.last_hidden_state[0, -1]).float(), -1)
        full_kv = [(l.keys, l.values) for l in out1.past_key_values.layers]

    # ---------------- which nodes: same pruning rule as the readout
    canonical_roots = {trie.paths[row[0]][0] for row in table}
    keep = {u for t, u in trie.roots.items() if u in canonical_roots or float(lp0[t].exp()) >= sc.prune_eps}
    feed, stack = [], sorted(keep, reverse=True)
    while stack:
        u = stack.pop()
        if trie.children[u]:
            feed.append(u)
            stack.extend(sorted(trie.children[u].values(), reverse=True))
    slot = {u: i for i, u in enumerate(feed)}
    L2 = len(feed)
    node_pos = torch.tensor([P + R + trie.depth[u] for u in feed], dtype=torch.long)
    anc = torch.zeros((L2, L2), dtype=torch.bool)
    for i, u in enumerate(feed):
        a = u
        while a >= 0:
            anc[i, slot[a]] = True
            a = trie.parent[a]

    # ---------------- B: nodes through layers 0..layer (no grad)
    with torch.no_grad():
        if L2:
            mB = torch.zeros((1, 1, L2, P + R + L2), dtype=torch.bool)
            mB[..., : P + R] = True
            mB[0, 0, :, P + R:] = anc
            cacheB = DynamicCache(ddp_cache_data=full_kv)
            x = model.model.embed_tokens(torch.tensor([[trie.tok[u] for u in feed]], device=dev))
            HB = _run_layers(model, x, range(layer + 1), sc._addmask(mB), node_pos[None].to(dev), cacheB)[0]
        else:
            HB = torch.zeros((0, grab["h"].shape[-1]), device=dev, dtype=grab["h"].dtype)

    # ---------------- C: [h, nodes] through layers > layer, n copies, grad w.r.t. h only
    n = len(item.options)
    Lc = 1 + L2
    early = P + R - 1
    kvC = [(k[:, :, :early].expand(n, -1, -1, -1), v[:, :, :early].expand(n, -1, -1, -1)) for k, v in full_kv]
    cacheC = DynamicCache(ddp_cache_data=kvC)
    mC = torch.zeros((n, 1, Lc, early + Lc), dtype=torch.bool)
    mC[..., :early] = True
    mC[:, 0, :, early] = True                    # everything sees the decision position
    mC[:, 0, 1:, early + 1:] = anc               # nodes see their ancestors
    posC = torch.cat([torch.tensor([P + R - 1]), node_pos])[None].expand(n, -1).to(dev)
    h0 = grab["h"].to(dev).expand(n, -1).clone().requires_grad_(True)
    x = torch.cat([h0[:, None], HB[None].expand(n, -1, -1)], 1)
    with torch.enable_grad():
        y = _run_layers(model, x, range(layer + 1, n_layers), sc._addmask(mC), posC, cacheC)
        lg = model.lm_head(model.model.norm(y)).float()            # (n, Lc, V)
        lsm = lg - torch.logsumexp(lg, -1, keepdim=True)
        # node log-probs per copy: roots from position 0, children of feed slot i from position 1 + i
        def node_lp(u):
            par = trie.parent[u]
            src = 0 if par < 0 else 1 + slot[par]
            return lsm[:, src, trie.tok[u]]                         # (n,)
        cache_lp: dict[int, torch.Tensor] = {}

        def lp_of(u):
            if u not in cache_lp:
                cache_lp[u] = node_lp(u)
            return cache_lp[u]
        z_all = []
        for row in table:
            paths = [trie.paths[pid] for pid in row if trie.paths[pid][0] in keep]
            z_all.append(torch.logsumexp(torch.stack([sum(lp_of(u) for u in path) for path in paths]), 0))
        Z = torch.stack(z_all, 1)                                   # (n copies, n options)
        Z.diagonal().sum().backward()
    J = h0.grad.float().cpu().numpy()
    z = Z[0].detach().float().cpu().numpy()
    q = np.exp(z - z.max())
    return {"J": J, "z": z, "q": q / q.sum(), "h": grab["h"].float().cpu().numpy(), "layer": layer}
