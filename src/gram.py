"""Coordination as an inner product (Brief Sections 2.2-2.4).

Rows of P are choice distributions over one menu (one game, one frame); C = P P^T.
Everything here is exact linear algebra on cached distributions; no model calls.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-300


def temper(p: np.ndarray, T: float) -> np.ndarray:
    """Distribution at temperature T from the T = 1 distribution: p^(1/T), renormalized (last axis)."""
    if T == 1.0:
        return p
    logp = np.log(np.clip(p, EPS, None)) / T
    logp -= logp.max(axis=-1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(axis=-1, keepdims=True)


def gram(P: np.ndarray) -> np.ndarray:
    """C_ij = <p_i, p_j>. P: (k, n) -> (k, k)."""
    P = np.asarray(P, dtype=np.float64)
    return P @ P.T


def psd_check(C: np.ndarray, tol: float = 1e-10) -> tuple[bool, float]:
    """(is_psd, min_eigenvalue). A computed C that fails this is a pipeline bug."""
    w = np.linalg.eigvalsh((C + C.T) / 2)
    return bool(w.min() >= -tol * max(1.0, abs(w).max())), float(w.min())


def identity_residual(P: np.ndarray, C: np.ndarray | None = None) -> float:
    """Max |C_ij - (C_ii + C_jj - ||p_i - p_j||^2)/2| (Proposition 2 iii)."""
    P = np.asarray(P, dtype=np.float64)
    C = gram(P) if C is None else C
    d = np.diag(C)
    sq = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return float(np.abs(C - 0.5 * (d[:, None] + d[None, :] - sq)).max())


def renyi2(p: np.ndarray) -> np.ndarray:
    """H_2(p) in bits, last axis."""
    return -np.log2(np.clip((np.asarray(p) ** 2).sum(-1), EPS, None))


def cosine_matrix(P: np.ndarray) -> np.ndarray:
    C = gram(P)
    nrm = np.sqrt(np.clip(np.diag(C), EPS, None))
    return C / np.outer(nrm, nrm)


def phi(C: np.ndarray, idx: np.ndarray | list[int] | None = None) -> float:
    """Population coordination capacity: mean off-diagonal entry over the personas in idx."""
    if idx is not None:
        idx = list(idx)
        C = C[np.ix_(idx, idx)]
    k = C.shape[0]
    return float((C.sum() - np.trace(C)) / (k * (k - 1)))


def phi_decomposition(P: np.ndarray, idx=None) -> dict:
    """Phi = Dbar - dispersion/2 (Section 2.3). Dispersion is the mean off-diagonal ||p_i - p_j||^2.

    Note: Dbar here averages self-coordination over the same k personas, so the identity is exact
    for the mean over ordered off-diagonal pairs.
    """
    P = np.asarray(P, dtype=np.float64)
    if idx is not None:
        P = P[idx]
    k = P.shape[0]
    C = gram(P)
    sq = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    disp = float(sq.sum() / (k * (k - 1)))
    dbar = float(np.trace(C) / k)
    return {"phi": phi(C), "dbar": dbar, "dispersion": disp, "residual": phi(C) - (dbar - disp / 2)}


def mean_cosine(P: np.ndarray, idx=None) -> float:
    """Abar: mean off-diagonal cosine (scale-free agreement, Section 2.9)."""
    P = np.asarray(P, dtype=np.float64)
    if idx is not None:
        P = P[idx]
    return phi(cosine_matrix(P))


def match_split(p: np.ndarray, q: np.ndarray) -> dict:
    """log2 Pr(match) = -H2(p)/2 - H2(q)/2 + log2 cos (Section 2.4). Works on the last axis."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = (p * q).sum(-1)
    dec = -0.5 * renyi2(p) - 0.5 * renyi2(q)
    cos = m / np.sqrt((p**2).sum(-1) * (q**2).sum(-1))
    return {"log2_match": np.log2(np.clip(m, EPS, None)), "decisiveness": dec, "agreement": np.log2(np.clip(cos, EPS, None))}


def mixture(dists: np.ndarray, axis: int = 0) -> np.ndarray:
    """Effective distribution over independently drawn paraphrases / orders (Section 4.4)."""
    return np.asarray(dists, np.float64).mean(axis=axis)
