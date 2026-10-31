"""Synthetic config checks for the single-cell worker."""

import pytest

from data_juicer_sandbox.specific_hooks.single_cell.config import Request


@pytest.mark.parametrize(
    "change",
    [
        {"budget": 0},
        {"budget": True},
        {"max_parallel": 0},
        {"fields": {"cell_type": "type"}},
        {"conditions": {"control": "x", "perturbed": "x"}},
        {"candidates": [{"id": "../bad", "mixture": {"A": 1}, "seed": 0}]},
        {"candidates": [{"id": "r", "mixture": {"A": float("nan")}, "seed": 0}]},
        {"candidates": [{"id": "r", "mixture": {"A": -0.1, "B": 1.1}, "seed": 0}]},
        {"candidates": [{"id": "r", "mixture": {"A": 0.2}, "seed": -1}]},
        {"train": {"early_stopping": "false"}},
    ],
)
def test_invalid_requests(payload, change):
    with pytest.raises(ValueError):
        Request.from_dict(payload | change)


def test_invalid_types_and_duplicate_ids(payload):
    payload["candidates"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        Request.from_dict(payload)


def test_default_holdout(payload):
    del payload["holdout_cell_type"]
    assert Request.from_dict(payload).holdout_cell_type == "CD4 T cells"
