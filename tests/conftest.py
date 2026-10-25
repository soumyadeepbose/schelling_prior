import os

import pytest

DEV_MODEL = os.environ.get("SCHELLING_TEST_MODEL", "Qwen/Qwen3-0.6B")
DTYPE = os.environ.get("SCHELLING_TEST_DTYPE", "float32")


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(DEV_MODEL)


@pytest.fixture(scope="session")
def model():
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(DEV_MODEL, dtype=getattr(torch, DTYPE), attn_implementation="sdpa").to(dev)
    m.eval()
    return m


@pytest.fixture(scope="session")
def lp_tol():
    """Absolute tolerance on log-probabilities when two exact computation paths are compared."""
    return 2e-4 if DTYPE == "float32" else 0.1


def pytest_collection_modifyitems(config, items):
    """Exact path-equivalence tests are fp32-only: bf16 noise is measured by modal_app.py::precision_check."""
    if DTYPE == "float32":
        return
    skip = pytest.mark.skip(reason="exact equivalence is checked in fp32; bf16 noise is quantified separately")
    exact = {"test_cached_matches_uncached", "test_batched_equals_single_padding_and_positions",
             "test_jacobian_matches_finite_difference",
             "test_steered_reuse_matches_reference", "test_uncached_batched_matches_uncached",
             "test_steered_decision_and_replace_modes"}
    for it in items:
        if it.name in exact:
            it.add_marker(skip)
