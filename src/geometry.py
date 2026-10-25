"""Coordination-sensitive directions from cached option-score Jacobians (Brief Sections 2.7-2.8).

J is (n, d): rows are d z_k / d h for option scores z = log P(option) at the decision position.
q = softmax(z) is the agent's choice distribution; p is the partner's (held fixed, stop-gradient).
"""
from __future__ import annotations

import numpy as np


def g_coord(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d<p, q>/dz_k = q_k (p_k - <p, q>)."""
    return q * (p - p @ q)


def g_logmatch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = p @ q
    return q * (p - m) / m


def g_dec(q: np.ndarray) -> np.ndarray:
    """d log||q||_2 / dz_k = q_k (q_k - ||q||^2) / ||q||^2."""
    s = q @ q
    return q * (q - s) / s


def g_agr(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d log cos(p, q) / dz = d log<p,q>/dz - d log||q||/dz (p fixed). Zero at p = q."""
    return g_logmatch(p, q) - g_dec(q)


def pullback(J: np.ndarray, g: np.ndarray) -> np.ndarray:
    """v = J^T g (covector in residual space)."""
    return J.T @ g


def directions(J: np.ndarray, p: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    return {"coord": pullback(J, g_coord(p, q)), "dec": pullback(J, g_dec(q)), "agr": pullback(J, g_agr(p, q))}


def project_out(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / np.linalg.norm(w)
    return v - (v @ w) * w


def unit_mean(V: np.ndarray) -> np.ndarray:
    """vbar = mean of unit vectors (rows of V)."""
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    return (V / np.clip(nrm, 1e-30, None)).mean(0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def split_half(V: np.ndarray, n_splits: int = 200, seed: int = 0) -> float:
    """Mean cosine between unit-mean directions of disjoint random halves of the instances (Section 6.2)."""
    rng = np.random.default_rng(seed)
    m = len(V)
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(m)
        out.append(cos(unit_mean(V[perm[: m // 2]]), unit_mean(V[perm[m // 2:]])))
    return float(np.mean(out))


def directional_derivative(v: np.ndarray, u: np.ndarray, s_u: float) -> float:
    """D(u) = s_u <v, u_hat> (metric-free; s_u = std of decision-position activations along u)."""
    return float(s_u * (v @ (u / np.linalg.norm(u))))


def whitened_cosine(v: np.ndarray, a: np.ndarray, sigma_half: np.ndarray, sigma_half_inv: np.ndarray) -> float:
    """kappa_W = <v, a_hat> / (||Sigma^1/2 v|| ||Sigma^-1/2 a_hat||)."""
    a = a / np.linalg.norm(a)
    return float((v @ a) / (np.linalg.norm(sigma_half @ v) * np.linalg.norm(sigma_half_inv @ a)))
