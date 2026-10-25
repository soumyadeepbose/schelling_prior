"""Brief 9.3 tests #5 and #9, plus the Proposition 3 pipeline check (Section 3.3). Pure numpy."""
import numpy as np
import pytest

from src import gram, stats


def rand_simplex(rng, k, n, conc=0.5):
    return rng.dirichlet(np.full(n, conc), size=k)


# ---- test 5: Gram matrix
def test_gram_psd_and_identity_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        P = rand_simplex(rng, rng.integers(2, 30), rng.integers(2, 12))
        C = gram.gram(P)
        ok, _ = gram.psd_check(C)
        assert ok
        assert gram.identity_residual(P, C) < 1e-8
        d = np.diag(C)
        assert np.all(C <= np.sqrt(np.outer(d, d)) + 1e-12)  # Cauchy-Schwarz


def test_phi_decomposition_exact():
    rng = np.random.default_rng(1)
    P = rand_simplex(rng, 12, 6)
    dec = gram.phi_decomposition(P)
    assert abs(dec["residual"]) < 1e-12


def test_match_split_exact():
    rng = np.random.default_rng(2)
    p, q = rand_simplex(rng, 2, 7)
    s = gram.match_split(p, q)
    assert np.isclose(s["log2_match"], s["decisiveness"] + s["agreement"])
    assert np.isclose(2 ** (-gram.renyi2(p)), (p**2).sum())


def test_temper():
    p = np.array([0.5, 0.3, 0.2])
    assert np.allclose(gram.temper(p, 1.0), p)
    t = gram.temper(p, 0.3)
    assert np.isclose(t.sum(), 1) and t[0] > p[0]


def test_psd_check_catches_bug():
    C = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert not gram.psd_check(C)[0]


# ---- test 9: stats
def test_gof_false_positive_rate():
    rng = np.random.default_rng(3)
    probs = np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 8 options + off-menu
    rejections = 0
    trials = 300
    for t in range(trials):
        counts = rng.multinomial(64, probs)
        rejections += stats.gof_test(counts, probs, n_sim=1000, seed=t)["p"] < 0.05
    rate = rejections / trials
    assert 0.01 <= rate <= 0.10, rate


def test_gof_detects_wrong_distribution():
    rng = np.random.default_rng(4)
    counts = rng.multinomial(64, [0.7, 0.1, 0.1, 0.1])
    assert stats.gof_test(counts, [0.25, 0.25, 0.25, 0.25])["p"] < 0.01


def test_miller_madow_unbiased_on_independent():
    rng = np.random.default_rng(5)
    kx = ky = 6
    plug, mm = [], []
    for _ in range(400):
        x = rng.integers(0, kx, 200)
        y = rng.integers(0, ky, 200)
        plug.append(stats.mi_plugin(x, y, kx, ky))
        mm.append(stats.mi_miller_madow(x, y, kx, ky))
    assert np.mean(plug) > 0.05  # plug-in is biased upward
    assert abs(np.mean(mm)) < 0.01


def test_tempered_fit_recovers_beta():
    rng = np.random.default_rng(6)
    for beta in [0.5, 1.0, 2.5, 6.0]:
        p = rng.dirichlet(np.ones(6))
        res = stats.tempered_fit(p, stats.power_tilt(p, beta))
        assert np.isclose(res["beta"], beta, rtol=1e-3)
        assert res["rel"] < 1e-8


def test_holm():
    adj = stats.holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_bootstrap_ci_covers():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 200)
    x = x - x.mean() + 1.0
    ci = stats.bootstrap(np.mean, x, n_boot=1000)
    assert ci["lo"] < 1.0 < ci["hi"] and stats.clears(ci)


# ---- Proposition 3 check: independent blind play carries zero MI while match rate = <p, q>
def test_prop3_sampled_blind_play():
    rng = np.random.default_rng(8)
    p = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    q = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    N = 4000
    a = rng.choice(5, N, p=p)
    b = rng.choice(5, N, p=q)
    res = stats.mi_permutation_test(a, b, 5, 5, n_perm=500)
    assert res["p"] > 0.01
    match = (a == b).mean()
    se = np.sqrt(p @ q * (1 - p @ q) / N)
    assert abs(match - p @ q) < 4 * se
