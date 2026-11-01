"""Synthetic evaluation checks for the single-cell worker."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from data_juicer_sandbox.specific_hooks.single_cell.config import Request
from data_juicer_sandbox.specific_hooks.single_cell.evaluation import (
    baseline_predictions,
    metrics,
)


def test_metrics_and_baselines(payload):
    ctrl = np.array([1.0, 2.0, 4.0, 7.0])
    truth = ctrl + [3, 1, 2, 4]
    result = metrics(truth, truth, ctrl, np.array([0, 1, 3]))
    assert result == pytest.approx(dict(r2_all=1, r2_top_degs=1, delta_pearson=1))
    result = metrics(ctrl, truth, ctrl, np.array([0, 1, 3]))
    assert result["r2_all"] == pytest.approx(27**2 / (21 * 38))
    assert result["delta_pearson"] is None
    req = Request.from_dict(payload | {"arithmetic_reference": "train"})
    a = ad.read_h5ad(req.data_path)
    train = a[~((a.obs.type == "held") & (a.obs.condition == "s"))].copy()
    baselines = baseline_predictions(train, req, ctrl)
    np.testing.assert_allclose(baselines["control"], ctrl)
    np.testing.assert_allclose(baselines["linear"], truth)
    with pytest.raises(ValueError, match="paired"):
        baseline_predictions(train[train.obs.condition == "c"], req, ctrl)


def test_metrics_hand_calculation_and_invalid_values():
    ctrl = np.array([1.0, 2.0, 4.0, 7.0])
    truth = np.array([4.0, 3.0, 6.0, 11.0])
    result = metrics(ctrl, truth, ctrl, np.array([0, 1, 3]))
    # Centered cross-product is 27; sums of squares are 21 and 38.
    assert result["r2_all"] == pytest.approx(27**2 / (21 * 38))
    assert result["r2_top_degs"] == pytest.approx(27**2 / ((62 / 3) * 38))
    result = metrics(-truth, truth, ctrl, np.array([0, 1, 3]))
    assert result["r2_all"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite"):
        metrics(np.array([np.nan, 1]), np.ones(2), np.zeros(2), np.array([0, 1]))


def test_linear_baseline_weights_types_equally(payload):
    control = np.array([1.0, 2.0, 4.0, 7.0])
    delta_a, delta_b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 2.0, 1.0, 3.0])
    obs = pd.DataFrame(
        dict(type=["A"] * 2 + ["B"] * 6, condition=["c", "s"] + ["c"] * 3 + ["s"] * 3),
        index=[f"cell{i}" for i in range(8)],
    )
    train = ad.AnnData(np.vstack([control, control + delta_a] + [control] * 3 + [control + delta_b] * 3), obs=obs)
    result = baseline_predictions(train, Request.from_dict(payload | {"arithmetic_reference": "train"}), control)
    np.testing.assert_allclose(result["linear"], control + (delta_a + delta_b) / 2)
