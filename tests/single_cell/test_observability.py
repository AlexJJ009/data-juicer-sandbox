"""Check single-cell metric persistence and watcher payloads with synthetic results."""

import json
from pathlib import Path
from typing import Any

import pytest
import wandb

from data_juicer_sandbox.specific_hooks.single_cell.observability import (
    record_evaluation,
    record_suggestions,
)


class Watcher:
    """Collect watch calls for assertions without opening a WandB run."""

    def __init__(self) -> None:
        """Initialize an empty list of result/namespace calls."""
        self.calls: list[tuple[dict[str, Any], str]] = []

    def watch(self, res: dict[str, Any], meta_name: str = "") -> None:
        """Store the result and namespace; return None."""
        self.calls.append((res, meta_name))


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluation_summary_and_table(tmp_path: Path, enabled: bool) -> None:
    """Persist full details and sample deviations, including undefined baseline values."""
    candidates = [
        dict(
            id=f"r2-m1-s{seed}",
            seed=seed,
            mixture={"A": 1.0},
            actual_mixture={"A": 1.0},
            metrics={"r2_all": score, "r2_top_degs": score, "delta_pearson": -score},
            baselines={"control": {"r2_all": 0.2, "delta_pearson": None}, "linear": {"r2_all": 0.4}},
        )
        for seed, score in zip((11, 23, 37), (0.2, 0.4, 0.6))
    ]
    watcher = Watcher() if enabled else None
    record_evaluation(candidates, tmp_path, 2, "r2_top_degs", watcher)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["candidates"] == candidates
    assert summary["scalars"]["r2/m1/metrics/r2_top_degs/mean"] == pytest.approx(0.4)
    assert summary["scalars"]["r2/m1/metrics/r2_top_degs/std"] == pytest.approx(0.2)
    assert summary["scalars"]["r2/best_score"] == 0.6
    assert summary["scalars"]["r2/best_mean_score"] == pytest.approx(0.4)
    assert summary["scalars"]["r2/m1/s11/baselines/control/r2_all"] == 0.2
    assert summary["scalars"]["r2/m1/s11/baselines/control/delta_pearson"] is None
    if watcher:
        scalars, namespace = watcher.calls[0]
        assert namespace == "single_cell"
        assert scalars == summary["scalars"]
        table = watcher.calls[1][0]["r2/candidates"]
        assert isinstance(table, wandb.Table)
        assert len(table.data) == 3
        assert "requested_mixture" in table.columns
        assert "actual_mixture" in table.columns


def test_suggestions_persist_without_wandb(tmp_path: Path) -> None:
    """Record CLIMB suggestions on disk and expose every weight to the watcher."""
    watcher = Watcher()
    mixtures = [{"A": 0.25, "B": 0.75}]
    record_suggestions(mixtures, tmp_path, 2, watcher)
    assert json.loads((tmp_path / "suggestions.json").read_text())["mixtures"] == mixtures
    assert watcher.calls[0][0]["r2/suggestions/m1/A"] == 0.25
