"""Capping and additive steering, via safety-research/assistant-axis (Brief Sections 2.9, 6.5).

Released Qwen3-32B capping (`layers_46:54-p0.25`) is a ceiling on the projection onto
`contrast_role_pos3_default1` vectors, which have cosine -1 with the Assistant Axis (checked in
results/assets/qwen3-32b_assets.json): it is the brief's floor form along +axis with tau = -cap,
applied at layers 46..53 (eight layers) at every position.

Interventions act at every position, including the system block, so the scorer's cached
system-block KV is keyed by `cache_tag` and never shared across interventions.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from assistant_axis import ActivationSteering, build_capping_steerer


@contextmanager
def _tagged(scorer, tag, steerer):
    old = scorer.cache_tag
    scorer.cache_tag = tag
    try:
        with steerer:
            yield
    finally:
        scorer.cache_tag = old


@contextmanager
def capped(scorer, capping_config: dict, experiment_id: str):
    with _tagged(scorer, ("cap", experiment_id), build_capping_steerer(scorer.model, capping_config, experiment_id)):
        yield


@contextmanager
def steered(scorer, vector: torch.Tensor, layer: int, coeff: float, name: str):
    """h <- h + coeff * vector at `layer`, all positions. Pass a unit vector to make coeff a norm."""
    st = ActivationSteering(scorer.model, steering_vectors=[vector.float()], coefficients=[float(coeff)],
                            layer_indices=[layer], intervention_type="addition", positions="all")
    with _tagged(scorer, ("steer", name, layer, float(coeff)), st):
        yield


def capping_layers(capping_config: dict, experiment_id: str) -> list[tuple[int, torch.Tensor, float]]:
    """(layer, vector, cap) triples of one released experiment, for bind-rate measurement."""
    exp = next(e for e in capping_config["experiments"] if e["id"] == experiment_id)
    out = []
    for iv in exp["interventions"]:
        if "cap" in iv:
            vd = capping_config["vectors"][iv["vector"]]
            out.append((vd["layer"], vd["vector"].float(), float(iv["cap"])))
    return out
