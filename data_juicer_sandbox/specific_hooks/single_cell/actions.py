"""Orchestrate sample, train and evaluate, including data/model I/O and DEG caching."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from typing import Any

import anndata as ad
import numpy as np

from .config import Request
from .evaluation import (
    baseline_predictions,
    holdout_groups,
    mean_expression,
    metrics,
    top_degs,
)


def sample(request: Request) -> dict[str, Any]:
    """Materialize manifests for request candidates and return sample protocol results."""
    from .sampling import sample as sample_candidates

    return sample_candidates(request)


def train(request: Request) -> dict[str, Any]:
    """Train request candidates in bounded fresh processes and return ordered results."""
    from .parallel import train_candidates

    return {"candidates": train_candidates(request)}


def cached_top_degs(data: ad.AnnData, request: Request) -> np.ndarray:
    """Rank holdout stim versus ctrl by Wilcoxon score and cache gene names in work_dir/degs.json.

    Cache identity includes input file identity, gene order, configured fields,
    conditions, holdout, n_degs and scanpy version; changed inputs invalidate it.
    """
    stat = request.data_path.stat()
    identity = dict(
        path=str(request.data_path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        genes=data.var_names.tolist(),
        fields=vars(request.fields),
        conditions=vars(request.conditions),
        holdout=request.holdout_cell_type,
        n_degs=request.n_degs,
        scanpy=version("scanpy"),
    )
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = request.work_dir / "degs.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            indices = data.var_names.get_indexer(saved["genes"])
            if len(indices) != min(request.n_degs, data.n_vars) or (indices < 0).any():
                raise ValueError(f"invalid DEG cache: {cache}")
            return indices
    indices = top_degs(data, request)
    genes = data.var_names[indices].tolist()
    request.work_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dict(signature=signature, genes=genes)), encoding="utf-8")
    return indices


def evaluate(request: Request) -> dict[str, Any]:
    """Load each trained candidate, predict holdout stim and return metrics and both baselines."""
    from .sampling import read_data, training_subset
    from .training import predict

    data = read_data(request)
    control, truth = holdout_groups(data, request)
    control_mean, truth_mean = mean_expression(control), mean_expression(truth)
    degs = cached_top_degs(data, request)
    results = []
    for candidate in request.candidates:
        train = training_subset(data, candidate, request)
        predicted = predict(train, control, candidate, request)
        baselines = baseline_predictions(train, request, control_mean)
        results.append(
            dict(
                candidate,
                metrics=metrics(mean_expression(predicted), truth_mean, control_mean, degs),
                baselines={key: metrics(value, truth_mean, control_mean, degs) for key, value in baselines.items()},
            )
        )
    return {"candidates": results}
