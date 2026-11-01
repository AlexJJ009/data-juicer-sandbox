"""Persist round summaries and send single-cell scalars/tables through Sandbox's watcher."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Protocol


class ResultWatcher(Protocol):
    """Sandbox watcher interface consumed by single-cell result recording."""

    def watch(self, res: dict[str, Any], meta_name: str = "") -> None:
        """Log res under meta_name; return None (default matches SandboxWatcher)."""
        ...


def record_evaluation(
    candidates: list[dict[str, Any]], directory: Path, iteration: int, metric: str, watcher: ResultWatcher | None
) -> None:
    """Write summary.json and log candidate values, seed statistics and a WandB table.

    Args:
        candidates: Evaluated candidates with requested/actual mixtures and metrics.
        directory: Existing iteration directory in the experiment output.
        iteration: One-based Sandbox round number.
        metric: Metric used by CLIMB and the round's best-score comparison.
        watcher: Existing Sandbox watcher, or None for disk-only recording.

    Returns:
        None. Undefined correlations remain JSON null; statistics use defined seeds
        and include their count. Standard deviations are sample deviations (ddof=1),
        undefined with fewer than two observations: https://docs.python.org/3/library/statistics.html#statistics.stdev
    """
    scalars: dict[str, float | int | None] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    scores = []
    for candidate in candidates:
        identity = re.fullmatch(r"r(\d+)-m(\d+)-s(\d+)", candidate["id"])
        if identity is None or int(identity[1]) != iteration or int(identity[3]) != candidate["seed"]:
            raise ValueError(f"Invalid round/mixture/seed candidate identity: {candidate['id']}")
        mixture = f"m{identity[2]}"
        prefix = f"r{iteration}/{mixture}"
        sections = {"metrics": candidate["metrics"], "baselines": candidate.get("baselines", {})}
        values = {f"metrics/{key}": value for key, value in sections["metrics"].items()}
        for baseline, metrics in sections["baselines"].items():
            values.update({f"baselines/{baseline}/{key}": value for key, value in metrics.items()})
        for key, value in values.items():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"Non-finite/non-numeric metric {candidate['id']}/{key}: {value}")
            scalars[f"{prefix}/s{candidate['seed']}/{key}"] = value
            group = grouped[f"{prefix}/{key}"]
            if value is not None:
                group.append(value)
        score = candidate["metrics"].get(metric)
        if score is not None:
            scores.append(score)
    for key, values in grouped.items():
        scalars[f"{key}/count"] = len(values)
        scalars[f"{key}/mean"] = mean(values) if values else None
        scalars[f"{key}/std"] = stdev(values) if len(values) > 1 else None
    scalars[f"r{iteration}/best_score"] = max(scores) if scores else None
    mixture_means = [mean(values) for key, values in grouped.items() if key.endswith(f"/metrics/{metric}") and values]
    scalars[f"r{iteration}/best_mean_score"] = max(mixture_means) if mixture_means else None
    summary = dict(iteration=iteration, metric=metric, std_ddof=1, scalars=scalars, candidates=candidates)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if watcher is not None:
        import wandb

        watcher.watch(scalars, "single_cell")
        # Keep rich values out of hook outputs: Sandbox serializes those to JSON.
        columns = ["iteration", "candidate", "seed", "requested_mixture", "actual_mixture", "metrics", "baselines"]
        rows = [
            [iteration, c["id"], c["seed"], c["mixture"], c["actual_mixture"], c["metrics"], c.get("baselines", {})]
            for c in candidates
        ]
        watcher.watch({f"r{iteration}/candidates": wandb.Table(columns=columns, data=rows)}, "single_cell")


def record_suggestions(
    mixtures: list[dict[str, float]], directory: Path, iteration: int, watcher: ResultWatcher | None
) -> None:
    """Persist CLIMB mixtures in directory and log each round/mixture/type weight.

    Args:
        mixtures: Suggested probability dictionaries from CLIMB.
        directory: Existing iteration output directory.
        iteration: One-based round producing suggestions for the next round.
        watcher: Existing Sandbox watcher, or None for disk-only recording.

    Returns:
        None; suggestions.json is written even when WandB is disabled.
    """
    (directory / "suggestions.json").write_text(
        json.dumps(dict(iteration=iteration, mixtures=mixtures), indent=2, allow_nan=False), encoding="utf-8"
    )
    if watcher is not None:
        watcher.watch(
            {
                f"r{iteration}/suggestions/m{i}/{kind}": weight
                for i, mix in enumerate(mixtures, 1)
                for kind, weight in mix.items()
            },
            "single_cell",
        )
