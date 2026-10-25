"""Brief 9.3 tests #6 (Jacobian vs finite differences, self-pair zero), #7 (nulls in the tangent space),
#8 (capping identity, steering dose 0). Model tests use Qwen3-0.6B on CPU with a random stand-in axis."""
import numpy as np
import pytest
import torch

from src import geometry, interventions, nulls, prompts
from src.jacobian import option_jacobian
from src.readout_hf import Item, Scorer

FR = prompts.load_frames()
LAYER = 14  # mid-depth layer of the 28-layer dev model


def item(tok, system, frame, options):
    ids, P = prompts.render(tok, system, prompts.user_message(FR["frames"][frame][0], options, FR["answer_line"]))
    return Item(ids, P, options)


# ---------------- pure
def test_self_pair_agreement_gradient_is_zero():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.dirichlet(np.ones(7) * 0.5)
        assert np.abs(geometry.g_agr(q, q)).max() < 1e-12
        J = rng.standard_normal((7, 64))
        assert np.abs(geometry.directions(J, q, q)["agr"]).max() < 1e-10


def test_gradients_match_softmax_finite_difference():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(6)
    p = rng.dirichlet(np.ones(6))
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    q = sm(z)
    for g, f in [(geometry.g_coord(p, q), lambda z: p @ sm(z)),
                 (geometry.g_dec(q), lambda z: np.log(np.linalg.norm(sm(z)))),
                 (geometry.g_agr(p, q), lambda z: np.log(p @ sm(z) / (np.linalg.norm(p) * np.linalg.norm(sm(z)))))]:
        fd = np.array([(f(z + 1e-6 * e) - f(z - 1e-6 * e)) / 2e-6 for e in np.eye(6)])
        assert np.allclose(g, fd, atol=1e-7)


def test_nulls_in_tangent_space():
    rng = np.random.default_rng(2)
    q = rng.dirichlet(np.ones(8))
    p = rng.dirichlet(np.ones(8))
    for g in [nulls.g_random_objective(q, rng, size=200), nulls.g_label_shuffle(p, q, rng, size=200),
              geometry.g_coord(p, q)[None], geometry.g_dec(q)[None], geometry.g_agr(p, q)[None]]:
        assert np.abs(g.sum(-1)).max() < 1e-12


def test_activation_drawn_covariance():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((500, 5)) @ np.diag([3, 1, 1, 0.5, 0.1])
    U = nulls.activation_drawn(A, rng, 20000)
    assert np.allclose(np.cov(U.T), np.cov(A.T), atol=0.15)


def test_split_half_reliability():
    rng = np.random.default_rng(4)
    base = rng.standard_normal(256)
    V = base + 0.5 * rng.standard_normal((100, 256)) * np.linalg.norm(base) / 16
    assert geometry.split_half(V, n_splits=50) > 0.9
    assert abs(geometry.split_half(rng.standard_normal((100, 256)), n_splits=50)) < 0.2


# ---------------- model
@pytest.mark.model
def test_jacobian_scores_match_readout(model, tok):
    sc = Scorer(model, tok)
    it = item(tok, "You are an AI assistant.", "coord", ["ice cream", "cake", "pie", "cookie", "pudding"])
    jac = option_jacobian(sc, it, LAYER)
    r = sc.score([it])[0]
    assert np.allclose(jac["z"], r.logprobs, atol=1e-4)
    assert jac["J"].shape == (5, model.config.hidden_size)


@pytest.mark.model
def test_jacobian_matches_finite_difference(model, tok):
    sc = Scorer(model, tok)
    it = item(tok, "You are an AI assistant.", "pick", ["red", "blue", "green", "orange", "purple"])
    jac = option_jacobian(sc, it, LAYER)
    J, q = jac["J"], jac["q"]
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(5))
    v = geometry.pullback(J, geometry.g_coord(p, q))

    def f(delta):
        sc.decision_delta = {LAYER: torch.tensor(delta, dtype=torch.float32)}
        try:
            return sc.score([it])[0]
        finally:
            sc.decision_delta = None

    eps = 0.05 * np.linalg.norm(jac["h"]) / np.sqrt(len(jac["h"]))
    for u in [v / np.linalg.norm(v), rng.standard_normal(len(v))]:
        u = u / np.linalg.norm(u)
        rp, rm = f(eps * u), f(-eps * u)
        fd_z = (rp.logprobs - rm.logprobs) / (2 * eps)
        assert np.allclose(fd_z, J @ u, rtol=1e-2, atol=1e-2 * np.abs(J @ u).max())
        fd_m = (p @ rp.p - p @ rm.p) / (2 * eps)
        an = v @ u
        assert abs(fd_m - an) <= 1e-2 * abs(an) + 1e-2 * np.linalg.norm(v), (fd_m, an)


@pytest.mark.model
def test_capping_identity_and_steering_dose_zero(model, tok):
    sc = Scorer(model, tok)
    it = item(tok, "You are an AI assistant.", "pick", ["red", "blue", "green"])
    base = sc.score([it])[0].logprobs
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    axis = torch.randn(d, generator=g)
    cfg = {"vectors": {f"layer_{l}/stand_in": {"layer": l, "vector": -axis} for l in (10, 11)},
           "experiments": [{"id": "never_binds", "interventions": [{"vector": f"layer_{l}/stand_in", "cap": 1e9} for l in (10, 11)]},
                           {"id": "always_binds", "interventions": [{"vector": f"layer_{l}/stand_in", "cap": -1e9} for l in (10, 11)]}]}
    with interventions.capped(sc, cfg, "never_binds"):
        assert np.array_equal(sc.score([it])[0].logprobs, base)
    with interventions.capped(sc, cfg, "always_binds"):
        assert not np.allclose(sc.score([it])[0].logprobs, base)
    with interventions.steered(sc, axis / axis.norm(), LAYER, 0.0, "axis"):
        assert np.array_equal(sc.score([it])[0].logprobs, base)
    assert np.array_equal(sc.score([it])[0].logprobs, base)  # hooks removed, cache not polluted
