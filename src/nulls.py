"""Nulls for the axis tests (Brief Sections 2.8(b), 6.3). Every objective-space null is a g vector
pulled back through the same cached Jacobian, so nulls cost one matrix-vector product each."""
from __future__ import annotations

import numpy as np


def g_random_objective(q: np.ndarray, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """g_k = q_k (r_k - <r, q>), r ~ N(0, I_n): same softmax weighting, random target. Sums to 0."""
    n = len(q)
    r = rng.standard_normal((size or 1, n))
    g = q * (r - (r @ q)[:, None])
    return g if size else g[0]


def g_label_shuffle(p: np.ndarray, q: np.ndarray, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """Partner distribution with its option indices permuted within the instance."""
    k = size or 1
    r = np.stack([p[rng.permutation(len(p))] for _ in range(k)])
    g = q * (r - (r @ q)[:, None])
    return g if size else g[0]


def activation_drawn(acts: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """u ~ Sigma^1/2 z, z ~ N(0, I_d), with Sigma the covariance of `acts` (m, d), without forming Sigma:
    u = A_c^T w / sqrt(m - 1), w ~ N(0, I_m), has covariance A_c^T A_c / (m - 1) = Sigma."""
    A = acts - acts.mean(0, keepdims=True)
    w = rng.standard_normal((size, A.shape[0]))
    return w @ A / np.sqrt(A.shape[0] - 1)


def random_unit(d: int, rng: np.random.Generator, size: int) -> np.ndarray:
    u = rng.standard_normal((size, d))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def natural_scale(acts: np.ndarray, U: np.ndarray) -> np.ndarray:
    """s_u: std of decision-position activations along each unit row of U."""
    Uh = U / np.linalg.norm(U, axis=1, keepdims=True)
    return (acts @ Uh.T).std(0, ddof=1)
