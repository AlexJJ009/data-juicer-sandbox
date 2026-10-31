"""Validate input matrices and materialize reproducible candidate training manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from loguru import logger
from scipy import sparse

from .config import Request


def read_data(request: Request) -> ad.AnnData:
    """Load request.data_path and validate identifiers, obs fields and both holdout conditions."""
    data = ad.read_h5ad(request.data_path)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("obs_names and var_names must be unique")
    for column in (request.fields.cell_type, request.fields.condition, request.fields.donor):
        if column not in data.obs or data.obs[column].isna().any():
            raise ValueError(f"missing obs column or values: {column}")
    conditions = {request.conditions.control, request.conditions.perturbed}
    if set(data.obs[request.fields.condition]) != conditions:
        raise ValueError(f"input conditions must equal {conditions}")
    for condition in conditions:
        if not (
            (data.obs[request.fields.cell_type] == request.holdout_cell_type)
            & (data.obs[request.fields.condition] == condition)
        ).any():
            raise ValueError(f"holdout lacks condition {condition}")
    if "counts" not in data.layers:
        raise ValueError('input requires layers["counts"] for the refinement contract')
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X)
    if not np.isfinite(values).all():
        raise ValueError("input X must contain finite log-normalized expression")
    if data.n_vars < 2:
        raise ValueError("at least two genes are required")
    return data


def training_subset(data: ad.AnnData, candidate: dict[str, Any], request: Request) -> ad.AnnData:
    """Read candidate manifest and return a copied subset, rejecting held-out stim or unknown/duplicate cells."""
    with Path(candidate["manifest"]).open(encoding="utf-8") as handle:
        names = json.load(handle)
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError(f"candidate {candidate['id']}: manifest must be a nonempty cell-name list")
    if len(set(names)) != len(names) or not set(names) <= set(data.obs_names):
        raise ValueError(f"candidate {candidate['id']}: duplicate or unknown manifest cells")
    subset = data[names].copy()
    obs = subset.obs
    if (
        (obs[request.fields.cell_type] == request.holdout_cell_type)
        & (obs[request.fields.condition] == request.conditions.perturbed)
    ).any():
        raise ValueError(f"candidate {candidate['id']}: manifest contains held-out perturbed cells")
    required = set(
        data.obs_names[
            (data.obs[request.fields.cell_type] == request.holdout_cell_type)
            & (data.obs[request.fields.condition] == request.conditions.control)
        ]
    )
    if not required <= set(names):
        raise ValueError(f"candidate {candidate['id']}: manifest must retain every holdout control")
    if set(obs[request.fields.condition]) != {request.conditions.control, request.conditions.perturbed}:
        raise ValueError(f"candidate {candidate['id']}: training requires both conditions")
    return subset


def sample(request: Request) -> dict[str, Any]:
    """Sample each candidate without replacement and return protocol artifacts.

    Each type gets round(weight * budget / 2) cells per condition, capped
    independently per condition; unused capacity is not redistributed. Equal
    per-condition targets keep every sampled type paired, which the linear
    baseline requires. All holdout controls are added outside budget.
    Results include actual_mixture, normalized over sampled non-holdout cells,
    with the same keys as the requested mixture. Empty non-holdout samples raise
    ValueError because no actual mixture can be normalized.
    """
    data = read_data(request)
    types = data.obs[request.fields.cell_type]
    conditions = data.obs[request.fields.condition]
    expected = set(types) - {request.holdout_cell_type}
    results = []
    for candidate in request.candidates:
        if set(candidate["mixture"]) != expected:
            raise ValueError(f"candidate {candidate['id']}: mixture keys must equal {sorted(expected)}")
        rng = np.random.default_rng(candidate["seed"])
        indices: list[int] = []
        composition = {}
        actual_counts = {}
        for cell_type, weight in sorted(candidate["mixture"].items()):
            count = round(weight * request.budget / 2)
            actual_counts[cell_type] = 0
            for condition in (request.conditions.control, request.conditions.perturbed):
                available = np.flatnonzero((types == cell_type) & (conditions == condition))
                chosen = rng.choice(available, size=min(count, len(available)), replace=False)
                if len(chosen) < count:
                    logger.warning(
                        "Candidate {} type {} condition {}: requested {}, actual {}",
                        candidate["id"],
                        cell_type,
                        condition,
                        count,
                        len(chosen),
                    )
                actual_counts[cell_type] += len(chosen)
                indices.extend(chosen.tolist())
                composition[f"{cell_type}|{condition}"] = len(chosen)
        sampled_total = sum(actual_counts.values())
        if sampled_total == 0:
            raise ValueError(f"candidate {candidate['id']}: no non-holdout cells sampled")
        actual_mixture = {cell_type: count / sampled_total for cell_type, count in actual_counts.items()}
        holdout = np.flatnonzero((types == request.holdout_cell_type) & (conditions == request.conditions.control))
        indices.extend(holdout.tolist())
        composition[f"{request.holdout_cell_type}|{request.conditions.control}"] = len(holdout)
        directory = request.work_dir / candidate["id"]
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(data.obs_names[sorted(indices)].tolist()), encoding="utf-8")
        results.append(dict(candidate, manifest=str(manifest), composition=composition, actual_mixture=actual_mixture))
        logger.info("Sampled candidate {}: {} cells", candidate["id"], len(indices))
    return {"candidates": results}
