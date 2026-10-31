"""Synthetic sampling checks for the single-cell worker."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from data_juicer_sandbox.specific_hooks.single_cell.config import Request
from data_juicer_sandbox.specific_hooks.single_cell.sampling import sample


def test_sampling_counts_holdout_and_reproducibility(payload):
    result = sample(Request.from_dict(payload))
    candidate = result["candidates"][0]
    assert candidate["composition"] == {"A|c": 3, "A|s": 3, "B|c": 3, "B|s": 3, "held|c": 6}
    names = json.loads(Path(candidate["manifest"]).read_text())
    assert len(set(names)) == 18
    assert not set(names) & {f"cell{i}" for i in range(6, 12)}
    sample(Request.from_dict(payload))
    assert names == json.loads(Path(candidate["manifest"]).read_text())
    payload["candidates"][0]["seed"] = 12
    sample(Request.from_dict(payload))
    assert names != json.loads(Path(candidate["manifest"]).read_text())


def test_sampling_caps_and_odd_budget(payload):
    payload["budget"] = 100
    candidate = sample(Request.from_dict(payload))["candidates"][0]
    assert sum(candidate["composition"].values()) == 30
    payload["budget"] = 5
    payload["candidates"][0]["mixture"] = {"A": 1.0, "B": 0.0}
    counts = sample(Request.from_dict(payload))["candidates"][0]["composition"]
    assert counts["A|c"] == 2 and counts["A|s"] == 3
    assert counts["B|c"] == counts["B|s"] == 0


def test_unknown_cell_type(payload):
    payload["candidates"][0]["mixture"] = {"missing": 1.0}
    with pytest.raises(ValueError, match="mixture"):
        sample(Request.from_dict(payload))


def test_manifest_rejects_leakage_and_missing_holdout(payload):
    from data_juicer_sandbox.specific_hooks.single_cell.sampling import training_subset

    candidate = sample(Request.from_dict(payload))["candidates"][0]
    req = Request.from_dict(payload)
    data = ad.read_h5ad(req.data_path)
    manifest = Path(candidate["manifest"])
    names = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(names + ["cell6"]))
    with pytest.raises(ValueError, match="held-out"):
        training_subset(data, candidate, req)
    manifest.write_text(json.dumps(names[1:]))
    with pytest.raises(ValueError, match="every holdout"):
        training_subset(data, candidate, req)


@pytest.mark.parametrize("damage", ["missing_counts", "nonfinite", "missing_column", "holdout_stim"])
def test_invalid_input_data(payload, damage):
    data = ad.read_h5ad(payload["data_path"])
    if damage == "missing_counts":
        del data.layers["counts"]
    elif damage == "nonfinite":
        data.X.data[0] = np.nan
    elif damage == "missing_column":
        del data.obs["donor"]
    else:
        data = data[~((data.obs.type == "held") & (data.obs.condition == "s"))].copy()
    data.write_h5ad(payload["data_path"])
    with pytest.raises(ValueError):
        sample(Request.from_dict(payload))


def test_actual_mixture_uses_capped_counts_and_warns(payload):
    from loguru import logger

    payload["budget"] = 40
    payload["candidates"][0]["mixture"] = {"A": 0.9, "B": 0.1}
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        candidate = sample(Request.from_dict(payload))["candidates"][0]
    finally:
        logger.remove(sink)
    assert candidate["actual_mixture"] == {"A": 0.75, "B": 0.25}
    assert sum(candidate["actual_mixture"].values()) == pytest.approx(1)
    assert any("A" in message and "18" in message and "6" in message for message in messages)


def test_actual_mixture_preserves_zero_weight(payload):
    payload["candidates"][0]["mixture"] = {"A": 1.0, "B": 0.0}
    candidate = sample(Request.from_dict(payload))["candidates"][0]
    assert candidate["actual_mixture"] == {"A": 1.0, "B": 0.0}
