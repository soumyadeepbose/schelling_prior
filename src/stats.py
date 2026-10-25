"""Statistics on cached distributions (Brief Sections 3.3, 6.1-6.6). No model calls."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import optimize, stats as sps

EPS = 1e-300


# ---------------------------------------------------------------- bootstrap

def bootstrap(stat: Callable[[np.ndarray], float], units: np.ndarray, n_boot: int = 2000,
              seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap over the first axis of `units` (game instances)."""
    rng = np.random.default_rng(seed)
    units = np.asarray(units)
    m = len(units)
    est = float(stat(units))
    draws = np.array([stat(units[rng.integers(0, m, m)]) for _ in range(n_boot)], dtype=float)
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"est": est, "lo": float(lo), "hi": float(hi), "draws": draws}


def two_level_bootstrap(stat: Callable[[np.ndarray, np.ndarray], float], strata: Sequence[Sequence[int]],
                        n_instances: int, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Resample personas within their stratum, then instances. stat(persona_idx, instance_idx) -> float."""
    rng = np.random.default_rng(seed)
    all_p = np.concatenate([np.asarray(s) for s in strata])
    est = float(stat(all_p, np.arange(n_instances)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pidx = np.concatenate([rng.choice(np.asarray(s), len(s), replace=True) for s in strata])
        iidx = rng.integers(0, n_instances, n_instances)
        draws[b] = stat(pidx, iidx)
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"est": est, "lo": float(lo), "hi": float(hi), "draws": draws}


def clears(ci: dict, null: float = 0.0) -> bool:
    return ci["lo"] > null or ci["hi"] < null


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- readout gate

def gof_test(counts: np.ndarray, probs: np.ndarray, n_sim: int = 4000, seed: int = 0) -> dict:
    """Parametric-bootstrap goodness of fit of multinomial `counts` against `probs`.

    `probs` includes the off-menu category as its last entry when the caller adds one.
    Statistic: total-variation distance between empirical and model distributions; also G.
    p-value = P_sim(stat >= observed) under the model distribution.
    """
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    N = int(counts.sum())

    def tv(c):
        return 0.5 * np.abs(c / N - probs).sum(-1)

    sims = rng.multinomial(N, probs, size=n_sim).astype(float)
    obs = tv(counts)
    sim = tv(sims)
    p = (1 + (sim >= obs - 1e-12).sum()) / (n_sim + 1)
    return {"tv": float(obs), "p": float(p), "tv_null_mean": float(sim.mean())}


# ---------------------------------------------------------------- mutual information

def mi_plugin(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> float:
    """Plug-in MI in bits for integer-coded samples."""
    joint = np.zeros((kx, ky))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / np.outer(px, py)[nz])).sum())


def mi_miller_madow(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> float:
    """Miller-Madow bias-corrected MI (bits): H(X)+H(Y)-H(X,Y), each entropy corrected by (m-1)/(2N ln 2)."""
    x, y = np.asarray(x), np.asarray(y)
    N = len(x)

    def h_mm(codes, k):
        c = np.bincount(codes, minlength=k)
        p = c[c > 0] / N
        return -(p * np.log2(p)).sum() + (len(p) - 1) / (2 * N * np.log(2))

    return float(h_mm(x, kx) + h_mm(y, ky) - h_mm(x * ky + y, kx * ky))


def mi_permutation_test(x: np.ndarray, y: np.ndarray, kx: int, ky: int, n_perm: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs = mi_miller_madow(x, y, kx, ky)
    null = np.array([mi_miller_madow(x, rng.permutation(y), kx, ky) for _ in range(n_perm)])
    return {"mi_mm": obs, "p": float((1 + (null >= obs).sum()) / (n_perm + 1))}


# ---------------------------------------------------------------- tempered fit (Section 2.3)

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.clip(q[m], EPS, None)))).sum())


def power_tilt(p: np.ndarray, beta: float) -> np.ndarray:
    logp = beta * np.log(np.clip(p, EPS, None))
    logp -= logp.max()
    q = np.exp(logp)
    return q / q.sum()


def tempered_fit(p_pick: np.ndarray, p_coord: np.ndarray, beta_bounds=(1e-3, 1e3)) -> dict:
    """beta* = argmin_beta KL(p_coord || p_pick^beta / Z); Rel = the KL left over."""
    p_pick, p_coord = np.asarray(p_pick, float), np.asarray(p_coord, float)
    f = lambda lb: _kl(p_coord, power_tilt(p_pick, np.exp(lb)))
    res = optimize.minimize_scalar(f, bounds=np.log(beta_bounds), method="bounded", options={"xatol": 1e-8})
    return {"beta": float(np.exp(res.x)), "rel": float(res.fun)}


# ---------------------------------------------------------------- correlations

def spearman(x, y) -> float:
    return float(sps.spearmanr(x, y).statistic)
