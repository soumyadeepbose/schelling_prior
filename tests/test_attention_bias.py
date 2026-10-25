"""PREREG Amendment 9 sanity checks for the attention-bias path and the eager attention-mass measurement (fp32)."""
import math

import numpy as np
import pytest
import torch

from src import amend9, prompts
from src.readout_hf import Item, Scorer

pytestmark = pytest.mark.model

FRAMES = prompts.load_frames()
SYSTEMS = ["You are an AI assistant.",
           "You are a contrarian who consistently takes opposing views to challenge consensus and conventional wisdom."]
MENUS = [["blue", "blueberry", "red"], ["ice cream", "cake", "pie", "cookie", "pudding"]]


def items_spans(tok, frame="pick"):
    out = []
    for s in SYSTEMS:
        for m in MENUS:
            user = prompts.user_message(FRAMES["frames"][frame][0], m, FRAMES["answer_line"])
            ids, P = prompts.render(tok, s, user)
            sp = amend9.spans(tok, s, user, ids)
            out.append((Item(ids, P, list(m), bias_cols=sp["persona"]), sp))
    return out


def test_spans_partition(tok):
    for it, sp in items_spans(tok):
        n = len(it.prompt_ids)
        assert sp["persona"] and max(sp["persona"]) < it.prefix_len
        assert min(sp["user"]) >= it.prefix_len and set(sp["c_user"]) <= set(sp["user"])
        assert 0 not in sp["c_template"] and n - 1 not in sp["c_template"]
        assert not (set(sp["c_template"]) & (set(sp["persona"]) | set(sp["user"])))


def test_zero_bias_reproduces_score(model, tok):
    its = [it for it, _ in items_spans(tok)]
    sc = Scorer(model, tok, prune_eps=0.0)
    ref = sc.score(its)
    sc.key_bias = torch.zeros(model.config.num_hidden_layers, device=model.device)
    got = sc.score(its)
    for a, b in zip(got, ref):
        assert np.max(np.abs(a.logprobs - b.logprobs)) <= 1e-4


def test_bias_scales_only_targeted_keys(model, tok):
    """Layer 0 (input unaffected by the bias at other layers): biased / unbiased post-softmax weights are a common factor c
    on every non-targeted key and r * c on the targeted keys, for every query after the system block."""
    it, sp = items_spans(tok)[1]
    r = 0.4
    sc = Scorer(model, tok, prune_eps=0.0)
    sc.score([it])                                   # fill the prefix cache before hooking
    cfg = model.config
    old = cfg._attn_implementation
    cfg._attn_implementation = "eager"
    caught = []
    h = model.model.layers[0].self_attn.register_forward_hook(lambda m, i, o: caught.append(o[1].detach().double()))
    try:
        sc.score([it])
        sc.key_bias = torch.full((cfg.num_hidden_layers,), math.log(r), device=model.device)
        sc.score([it])
    finally:
        h.remove()
        cfg._attn_implementation = old
    w0, w1 = caught[0][0], caught[2][0]              # pass 1 of each call: (H, Q, K)
    P, n = it.prefix_len, len(it.prompt_ids)
    tgt = np.zeros(n, bool)
    tgt[sp["persona"]] = True
    ratio = (w1[..., :n] / w0[..., :n]).cpu().numpy()  # (H, Q, n), queries are the post-system rows
    for q in range(n - P):
        vis = np.arange(n) <= P + q
        c = ratio[:, q, vis & ~tgt]
        assert np.allclose(c, c[:, :1], rtol=1e-4)
        assert np.allclose(ratio[:, q, vis & tgt], r * c[:, :1], rtol=1e-4)


def test_eager_matches_sdpa(model, tok, lp_tol):
    its = [it for it, _ in items_spans(tok)]
    sc = Scorer(model, tok, prune_eps=0.0)
    ref = sc.score(its)
    sc._prefix_cache.clear()
    cfg = model.config
    old = cfg._attn_implementation
    cfg._attn_implementation = "eager"
    try:
        got = sc.score(its)
    finally:
        cfg._attn_implementation = old
    for a, b in zip(got, ref):
        assert np.max(np.abs(a.logprobs - b.logprobs)) <= lp_tol


def test_attention_mass_totals(model, tok):
    """A span covering every prompt key gets mass 1 from the decision position and from the user queries."""
    pairs = items_spans(tok)
    its = [it for it, _ in pairs]
    spans = [{"keys": {"all": list(range(len(it.prompt_ids))), "persona": sp["persona"]}, "user_q": sp["user"]}
             for it, sp in pairs]
    sc = Scorer(model, tok, prune_eps=0.0)
    m = sc.attention_mass(its, spans)                # names sorted: all, persona
    assert np.allclose(m[:, :, 0, :], 1.0, atol=1e-4)
    assert (m[:, :, 1, :] > 0).all() and (m[:, :, 1, :] < 1).all()
