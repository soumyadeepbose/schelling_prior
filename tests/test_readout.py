"""Brief 9.3 tests #1, #2, #4, #10, #11, test #3 replaced by cached-vs-uncached equivalence, plus
explicit padding / position-id and tokenizer-boundary checks, and the sampling goodness-of-fit gate.
Runs on Qwen3-0.6B, float32, CPU."""
import numpy as np
import pytest
import torch

from src import prompts, stats
from src.readout_hf import Item, Scorer, score_uncached, score_uncached_batched, surface_variants

pytestmark = pytest.mark.model

SYSTEMS = [
    "You are an AI assistant.",
    None,
    "You are a contrarian who consistently takes opposing views to challenge consensus and conventional wisdom.",
]
FRAMES = prompts.load_frames()


def make_item(tok, system, frame, options, par=0, story=None):
    user = prompts.user_message(FRAMES["frames"][frame][par], options, FRAMES["answer_line"], story)
    ids, P = prompts.render(tok, system, user)
    return Item(ids, P, list(options))


MENUS = [
    ["blue", "blueberry", "red"],                       # shared-prefix options
    ["ice cream", "cake", "pie", "cookie", "pudding"],  # multi-token option
    ["26", "31", "100", "7"],                         # numbers
    ["top left", "center", "bottom-right"],
]


def test_variant_rule():
    v = surface_variants("ice cream")
    assert v[0] == "ice cream" and "Ice cream" in v and "ICE CREAM" in v and '"ice cream"' in v
    assert " ice cream" in v and "ice cream." in v and "'ice cream'" in v
    assert len(surface_variants("26")) == 5  # case forms collapse for digits


def test_cached_matches_uncached(model, tok, lp_tol):
    sc = Scorer(model, tok, prune_eps=0.0)
    items = [make_item(tok, s, f, m) for s in SYSTEMS for f, m in zip(["pick", "coord", "guess", "anti"], MENUS)]
    fast = sc.score(items)
    for it, r in zip(items, fast):
        ref = score_uncached(model, tok, it)
        for a, b in zip(r.variant_logprobs, ref):
            a, b = np.array(a), np.array(b)
            ok = b > -40  # beyond this, float32 noise in tiny probabilities is irrelevant
            assert np.allclose(a[ok], b[ok], atol=lp_tol), (it.options, a, b)


def test_batched_equals_single_padding_and_positions(model, tok, lp_tol):
    """Unequal prompt lengths and unequal option lengths in one batch must not change any score."""
    items = [make_item(tok, SYSTEMS[0], "pick", m) for m in MENUS] + [
        make_item(tok, SYSTEMS[0], "coord", MENUS[1], story="Two friends are choosing dessert.")]
    batched = Scorer(model, tok, prune_eps=0.0).score(items)
    single = Scorer(model, tok, prune_eps=0.0, max_batch_size=1)
    for it, rb in zip(items, batched):
        rs = single.score([it])[0]
        assert np.allclose(rb.logprobs, rs.logprobs, atol=lp_tol / 2)
        a, b = np.array(rb.variant_logprobs), np.array(rs.variant_logprobs)
        ok = b > -40
        assert np.allclose(a[ok], b[ok], atol=lp_tol / 2)


def test_mass_accounting(model, tok):
    sc = Scorer(model, tok)
    for r in sc.score([make_item(tok, s, "pick", m) for s in SYSTEMS for m in MENUS]):
        total = np.exp(r.logprobs).sum()
        assert total <= 1 + 1e-4
        assert abs(total + r.offmenu - 1) < 1e-4
        assert abs(r.p.sum() - 1) < 1e-9
        assert r.pruned_bound < 1e-3
        assert np.all(r.logprobs >= r.logprobs_canonical - 1e-9)


def test_pruning_is_bounded(model, tok):
    it = make_item(tok, SYSTEMS[0], "pick", MENUS[1])
    full = Scorer(model, tok, prune_eps=0.0).score([it])[0]
    pr = Scorer(model, tok, prune_eps=1e-6).score([it])[0]
    lost = np.exp(full.logprobs).sum() - np.exp(pr.logprobs).sum()
    assert -1e-6 <= lost <= pr.pruned_bound + 1e-6


def test_terminator_prevents_double_counting(model, tok):
    it = make_item(tok, SYSTEMS[0], "pick", ["blue", "blueberry", "red"])
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    red, reddish = tok("blue", add_special_tokens=False)["input_ids"], tok("blueberry", add_special_tokens=False)["input_ids"]
    assert reddish[: len(red)] == red, "test premise: blueberry starts with the tokens of blue"
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=torch.tensor([it.prompt_ids + reddish + [eot]], device=model.device)).logits[0].float(), -1)
    n = len(it.prompt_ids)
    no_eot_red = sum(float(lp[n - 1 + i, t]) for i, t in enumerate(red))
    no_eot_reddish = sum(float(lp[n - 1 + i, t]) for i, t in enumerate(reddish))
    # without the terminator, 'red' contains all of 'reddish': summing them counts that mass twice
    assert no_eot_red >= no_eot_reddish
    r = Scorer(model, tok, prune_eps=0.0).score([it])[0]
    canon = dict(zip(it.options, r.logprobs_canonical))
    assert canon["blue"] < no_eot_red  # the terminator removes the continuation mass
    assert np.exp(r.logprobs).sum() <= 1 + 1e-4


def test_teacher_forcing_matches_generation(model, tok, lp_tol):
    it = make_item(tok, SYSTEMS[0], "pick", MENUS[0])
    ids = torch.tensor([it.prompt_ids], device=model.device)
    with torch.no_grad():
        g = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=8, output_logits=True,
                           return_dict_in_generate=True, top_k=None, top_p=None, temperature=None)
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    new = g.sequences[0, ids.shape[1]:].tolist()
    assert eot in new, "greedy answer should terminate"
    k = new.index(eot) + 1
    step = sum(float(torch.log_softmax(g.logits[i][0].float(), -1)[new[i]]) for i in range(k))
    text = tok.decode(new[: k - 1])
    assert tok(text, add_special_tokens=False)["input_ids"] == new[: k - 1]
    r = Scorer(model, tok, prune_eps=0.0).score([Item(it.prompt_ids, it.prefix_len, [text])])[0]
    assert abs(r.logprobs_canonical[0] - step) < max(1e-3, lp_tol)


def test_tokenizer_boundary_all_menus(tok):
    """prompt + variant + <eot> tokenizes as prompt_ids + variant_ids + [eot], for every declared variant."""
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    for fam in ["F1", "F2", "F3", "F4", "F5", "F1b", "F2b", "F4b"]:
        for g in prompts.load_games(fam):
            user = prompts.user_message(FRAMES["frames"]["pick"][0], g["options"], FRAMES["answer_line"])
            msgs = [{"role": "system", "content": SYSTEMS[0]}, {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            base = tok(text, add_special_tokens=False)["input_ids"]
            for o in g["options"]:
                for v in surface_variants(o):
                    joint = tok(text + v + "<|im_end|>", add_special_tokens=False)["input_ids"]
                    assert joint == base + tok(v, add_special_tokens=False)["input_ids"] + [eot], (fam, v)


def test_template_reasoning_off(tok):
    ids, P = prompts.render(tok, SYSTEMS[0], "hi")
    text = tok.decode(ids)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(text[-60:])
    assert P > 0 and tok.decode(ids[:P]).startswith("<|im_start|>system")


def test_permutation_roundtrip(model, tok):
    """Scores come back in the caller's canonical order whatever order the menu was shown in."""
    g = {"instance_id": "T-0", "family": "T", "options": MENUS[1]}
    cells = prompts.build_cells({"a": [SYSTEMS[0]]}, ["pick"], [g], n_orders=3)
    for c in cells:
        assert sorted(c.shown) == sorted(c.options) and c.options == MENUS[1]
    sc = Scorer(model, tok, prune_eps=0.0)
    c = cells[1]
    ids, P = prompts.render(tok, c.system, c.user)
    perm = [c.shown.index(o) for o in c.options]
    r_canon = sc.score([Item(ids, P, c.options)])[0]
    r_shown = sc.score([Item(ids, P, c.shown)])[0]
    assert np.allclose(r_canon.logprobs, r_shown.logprobs[perm], atol=1e-6)


def test_determinism(model, tok):
    items = [make_item(tok, s, "coord", m) for s in SYSTEMS[:2] for m in MENUS]
    a = Scorer(model, tok).score(items)
    b = Scorer(model, tok).score(items)
    for x, y in zip(a, b):
        assert x.logprobs.tobytes() == y.logprobs.tobytes()


def test_readout_gof_gate(model, tok):
    """64 samples at T = 1 agree with the logit distribution (Section 3.3 readout gate)."""
    sc = Scorer(model, tok)
    for s, f, m in [(SYSTEMS[0], "pick", MENUS[1]), (SYSTEMS[2], "coord", MENUS[3]), (None, "guess", MENUS[2])]:
        it = make_item(tok, s, f, m)
        r = sc.score([it])[0]
        table = {v: k for k, o in enumerate(m) for v in surface_variants(o)}
        ids = torch.tensor([it.prompt_ids], device=model.device)
        torch.manual_seed(0)
        with torch.no_grad():
            g = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0, top_k=0,
                               top_p=1.0, num_return_sequences=64, max_new_tokens=12)
        counts = np.zeros(len(m) + 1)
        for row in g[:, ids.shape[1]:]:
            text = tok.decode(row, skip_special_tokens=False)
            ans = text.split("<|im_end|>")[0] if "<|im_end|>" in text else None
            counts[table.get(ans, len(m)) if ans is not None else len(m)] += 1
        probs = np.append(np.exp(r.logprobs), max(r.offmenu, 0.0))
        res = stats.gof_test(counts, probs)
        assert res["p"] > 0.001, (m, counts, probs.round(3), res)


def test_capture_matches_score_and_uncached(model, tok):
    """capture() (pass 1 only) gives the same decision-position residuals as score() and as a full uncached forward."""
    items = [make_item(tok, s, f, m) for s in SYSTEMS[:2] for f, m in zip(["pick", "coord"], MENUS[:2])]
    L = [2, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
    sc = Scorer(model, tok, prune_eps=0.0)
    cap = sc.capture(items, L)
    scr = sc.score(items, capture_layers=L)
    for it, c, r in zip(items, cap, scr):
        got = {}
        hooks = [model.model.layers[l].register_forward_hook(
            lambda m, i, o, l=l: got.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy())) for l in L]
        try:
            with torch.no_grad():
                model.model(input_ids=torch.tensor([it.prompt_ids], device=model.device))
        finally:
            for h in hooks:
                h.remove()
        for l in L:
            scale = np.abs(got[l]).max()
            assert np.allclose(c[l], r.residuals[l], atol=1e-3 * scale), l
            assert np.allclose(c[l], got[l], atol=1e-3 * scale), l


def test_steered_reuse_matches_reference(model, tok, lp_tol):
    """score_steered (prompt reused across deltas) = reference path with Scorer.steer = full uncached forward."""
    layer = model.config.num_hidden_layers // 2
    items = [make_item(tok, s, f, m) for s in SYSTEMS[:2] for f, m in zip(["pick", "coord"], MENUS[:2])]
    g = torch.Generator().manual_seed(0)
    d = model.config.hidden_size
    deltas = [None] + [torch.randn(d, generator=g) / d ** 0.5 * s for s in (4.0, 12.0)]
    sc = Scorer(model, tok, prune_eps=0.0)
    fast = sc.score_steered(items, [None if v is None else (layer, v.to(model.device)) for v in deltas])
    plain = sc.score(items)
    for k, delta in enumerate(deltas):
        sc.steer = None if delta is None else (layer, delta.to(model.device))
        ref = sc.score(items)
        sc.steer = None
        for i, it in enumerate(items):
            unc = score_uncached(model, tok, it, steer=None if delta is None else (layer, delta.to(model.device)))
            for a, b, c in zip(fast[i][k].variant_logprobs, ref[i].variant_logprobs, unc):
                a, b, c = np.array(a), np.array(b), np.array(c)
                ok = c > -40
                assert np.allclose(a[ok], b[ok], atol=lp_tol), (k, it.options, a, b)
                assert np.allclose(a[ok], c[ok], atol=lp_tol), (k, it.options, a, c)
            if delta is None:
                assert np.allclose(fast[i][k].logprobs, plain[i].logprobs, atol=lp_tol)
    # steering must actually move the distribution
    assert max(np.abs(fast[i][2].p - fast[i][0].p).sum() for i in range(len(items))) > 1e-3


def test_uncached_batched_matches_uncached(model, tok, lp_tol):
    layer = model.config.num_hidden_layers // 2
    it = make_item(tok, SYSTEMS[0], "coord", MENUS[1])
    v = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(1)).to(model.device) * 0.2
    for st in (None, (layer, v)):
        a, b = score_uncached_batched(model, tok, it, steer=st, chunk=7), score_uncached(model, tok, it, steer=st)
        for x, y in zip(a, b):
            assert np.allclose(np.array(x), np.array(y), atol=lp_tol)


def test_steered_decision_and_replace_modes(model, tok, lp_tol):
    """(l, v, 'decision') = reference decision_delta; (l, targets, 'replace') = decision_delta of (target - live residual)."""
    layer = model.config.num_hidden_layers // 2
    items = [make_item(tok, s, "pick", m) for s in SYSTEMS[:2] for m in MENUS[:2]]
    for k, it in enumerate(items):
        it.key = k
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(2)
    v = (torch.randn(d, generator=g) * 0.3).to(model.device)
    targets = {k: (torch.randn(d, generator=g) * 2.0).to(model.device) for k in range(len(items))}
    sc = Scorer(model, tok, prune_eps=0.0)
    fast = sc.score_steered(items, [(layer, v, "decision"), (layer, targets, "replace")])
    sc.decision_delta = {layer: v}
    ref_dec = sc.score(items)
    sc.decision_delta = None
    live = sc.capture(items, [layer])
    for i, it in enumerate(items):
        sc.decision_delta = {layer: targets[i] - torch.tensor(live[i][layer], device=model.device)}
        ref_rep = sc.score([it])[0]
        sc.decision_delta = None
        for a, b in ((fast[i][0], ref_dec[i]), (fast[i][1], ref_rep)):
            for x, y in zip(a.variant_logprobs, b.variant_logprobs):
                x, y = np.array(x), np.array(y)
                ok = y > -40
                assert np.allclose(x[ok], y[ok], atol=5 * lp_tol), (it.options, x, y)
