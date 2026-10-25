"""score_cells: parquet rows, saved residuals and axis projections line up (Phase 2 storage path)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src import run
from src.readout_hf import Scorer

pytestmark = pytest.mark.model


def test_score_cells_storage(model, tok, tmp_path):
    personas = {"assistant": {"prompts": ["You are an AI assistant.", "You are a large language model."], "stratum": "assistant"},
                "contrarian": {"prompts": ["You are a contrarian.", "Act as a contrarian."], "stratum": "theory"}}
    cells = run.phase_cells(personas, ["F1"], ["pick", "coord"], n_orders=2)[:90]
    L = model.config.num_hidden_layers
    axis = torch.randn(L, model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    out = tmp_path / "dist.parquet"
    keep = lambda c: c.order_id == 0 and c.persona_par == 0
    run.score_cells(Scorer(model, tok), cells, {"model": "dev"}, out, axis=axis, act_layer=10,
                    extra_layers=[12, 13], extra_keep=keep, chunk=32)
    df = pd.read_parquet(out)
    A = np.load(str(out).replace(".parquet", "_l10.npy"))
    ex = np.load(str(out).replace(".parquet", "_extra.npz"))
    assert len(df) == len(cells) == len(A)
    a = (axis[10] / axis[10].norm()).numpy()
    assert np.allclose(A.astype(np.float32) @ a, df.proj_l10.values, rtol=2e-2, atol=2e-2 * np.abs(df.proj_l10).max())
    kept = [i for i, c in enumerate(cells) if keep(c)]
    assert ex["rows"].tolist() == kept and ex["l12"].shape == (len(kept), model.config.hidden_size)
    assert all(np.isnan(df.proj_extra[i]).all() == (i not in kept) for i in range(len(df)))
    assert (df.persona.values == np.array([c.persona for c in cells])).all()
