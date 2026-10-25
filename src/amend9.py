"""PREREG Amendment 9: attention to the persona (Run 4) and the random-direction control for the g ablation (Run 6).

Spans are prompt token positions. Persona span = system-message content tokens; user span = user-message content tokens;
C-user control = the last n_c tokens of the "Options: ..." line plus the answer line (identical across frames for one
instance and order), n_c = min(persona length, tail length); C-template control = chat-template tokens except the first
token (attention sink) and the decision token (last prompt token).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import amend6
from .analysis_amend5 import COLS

RUN4_FRAMES = ("pick", "coord", "coord_human", "goal_nonsocial", "pick_long")
RUN4_ORDER = 1
SPAN_NAMES = ("c_template", "c_user", "persona")      # sorted, as Scorer.attention_mass returns them
N_RANDOM_A9 = 3                                        # owner decision (PREREG Amendment 9)


def run4_cells(frames=RUN4_FRAMES):
    return [c for c in amend6.cells(tuple(frames), "B") if c.order_id == RUN4_ORDER]


def spans(tok, system: str, user: str, ids: list[int]) -> dict:
    """Token positions of the persona, user, C-user and C-template spans for one rendered prompt (ids from render())."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ids, "offsets tokenization differs from render()"
    off = enc["offset_mapping"]
    s0 = text.index(system)
    s1 = s0 + len(system)
    u0 = text.index(user, s1)
    u1 = u0 + len(user)
    t0 = text.index("Options: ", u0)

    def inside(a, b):
        return [k for k, (x, y) in enumerate(off) if y > x and x >= a and y <= b]

    persona, usr, tail = inside(s0, s1), inside(u0, u1), inside(t0, u1)
    n = len(ids)
    content = set(persona) | set(usr)
    template = [k for k in range(1, n - 1) if k not in content]
    c_user = tail[-min(len(persona), len(tail)):]
    return {"persona": persona, "user": usr, "c_user": c_user, "c_template": template}


def sigma_directions(phase2_dir: str, seed: int = 6, k: int = N_RANDOM_A9) -> dict:
    """v = Sigma^{1/2} z / ||.||, Sigma = covariance of layer-32 decision-position residuals over half-A Phase 2 cells
    (pick, coord, guess; 23 substantive personas; all orders and paraphrases)."""
    from .analysis_amend5 import substantive

    subst, _ = substantive()
    sp = amend6.split()
    half_a = {iid for f in sp for iid in sp[f]["A"]}
    n, s1, s2 = 0, None, None
    for fn in sorted(glob.glob(f"{phase2_dir}/dist_shard*.parquet")):
        d = pd.read_parquet(fn, columns=COLS)
        m = np.flatnonzero((d.persona.isin(subst) & d.instance_id.isin(half_a) & d.frame.isin(("pick", "coord", "guess"))).to_numpy())
        X = np.asarray(np.load(fn[:-8] + "_l32.npy", mmap_mode="r")[m], np.float64)
        n += len(X)
        s1 = X.sum(0) if s1 is None else s1 + X.sum(0)
        s2 = X.T @ X if s2 is None else s2 + X.T @ X
    mu = s1 / n
    Sig = s2 / n - np.outer(mu, mu)
    lam, U = np.linalg.eigh(Sig)
    half = (U * np.sqrt(np.clip(lam, 0, None))) @ U.T
    z = np.random.default_rng(seed).standard_normal((k, len(mu)))
    V = z @ half
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    g = np.load(f"{phase2_dir}/../amend6/directions.npz")["ghat"].astype(np.float64)
    g /= np.linalg.norm(g)
    return {"random": V.astype(np.float32), "var_random": np.einsum("kd,de,ke->k", V, Sig, V),
            "var_ghat": float(g @ Sig @ g), "n_rows": n, "trace": float(np.trace(Sig))}
