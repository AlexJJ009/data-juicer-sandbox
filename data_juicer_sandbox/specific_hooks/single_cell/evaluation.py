"""Model-independent expression metrics, baselines and holdout DEG ranking."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from .config import Request
from .reference import arithmetic_population


def mean_expression(data: ad.AnnData) -> np.ndarray:
    """Return the per-gene float64 mean of nonempty dense or sparse AnnData.X."""
    if data.n_obs == 0:
        raise ValueError("cannot average an empty cell population")
    return np.asarray(data.X.astype(np.float64).mean(axis=0)).ravel()


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("correlation requires at least two finite expression values")
    left, right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    # A constant vector has undefined correlation; JSON null preserves this fact.
    if denominator == 0:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def metrics(
    predicted: np.ndarray, truth: np.ndarray, control: np.ndarray, deg_indices: np.ndarray
) -> dict[str, float | None]:
    """Return squared mean-expression correlations and signed delta correlation.

    Inputs are matching gene-mean vectors and the selected DEG indices. Undefined
    correlations (including the control baseline's zero delta) return None.
    """
    predicted, truth, control = (np.asarray(v, dtype=np.float64) for v in (predicted, truth, control))
    if predicted.ndim != 1 or predicted.shape != truth.shape or predicted.shape != control.shape:
        raise ValueError("predicted, truth and control must be matching one-dimensional vectors")
    all_corr = _pearson(predicted, truth)
    deg_corr = _pearson(predicted[deg_indices], truth[deg_indices])
    return {
        "r2_all": None if all_corr is None else all_corr**2,
        "r2_top_degs": None if deg_corr is None else deg_corr**2,
        "delta_pearson": _pearson(predicted - control, truth - control),
    }


def baseline_predictions(
    train: ad.AnnData, request: Request, control: np.ndarray, *, data: ad.AnnData | None = None
) -> dict[str, np.ndarray]:
    """Return control and equally type-weighted linear predictions from the configured reference.

    Args:
        train: Candidate training cells, used only by the legacy train policy.
        request: Validated reference policy and obs field configuration.
        control: Holdout control mean expression in matching gene order.
        data: Optional full input data to avoid disk reads in full mode.

    Returns:
        Control and linear gene means. Every included type must have paired
        conditions; missing pairs in legacy mode raise ValueError.
    """
    train = arithmetic_population(train, request, data)
    types = train.obs[request.fields.cell_type]
    conditions = train.obs[request.fields.condition]
    deltas = []
    for cell_type in sorted(set(types) - {request.holdout_cell_type}):
        groups = [
            train[(types == cell_type) & (conditions == condition)]
            for condition in (request.conditions.control, request.conditions.perturbed)
        ]
        if any(group.n_obs == 0 for group in groups):
            raise ValueError(f"linear baseline requires paired conditions for {cell_type}")
        deltas.append(mean_expression(groups[1]) - mean_expression(groups[0]))
    if not deltas:
        raise ValueError("linear baseline requires at least one paired non-holdout cell type")
    return {"control": control.copy(), "linear": control + np.mean(deltas, axis=0)}


def holdout_groups(data: ad.AnnData, request: Request) -> tuple[ad.AnnData, ad.AnnData]:
    """Return holdout control and true perturbed populations using configured obs fields."""
    types, conditions = data.obs[request.fields.cell_type], data.obs[request.fields.condition]
    return tuple(
        data[(types == request.holdout_cell_type) & (conditions == condition)].copy()
        for condition in (request.conditions.control, request.conditions.perturbed)
    )


def top_degs(data: ad.AnnData, request: Request) -> np.ndarray:
    """Return Wilcoxon DEG indices from data and configuration without mutating inputs or doing I/O."""
    held = data[data.obs[request.fields.cell_type] == request.holdout_cell_type].copy()
    sc.tl.rank_genes_groups(
        held,
        groupby=request.fields.condition,
        groups=[request.conditions.perturbed],
        reference=request.conditions.control,
        method="wilcoxon",
        use_raw=False,
        n_genes=min(request.n_degs, data.n_vars),
        rankby_abs=False,
    )
    genes = held.uns["rank_genes_groups"]["names"][request.conditions.perturbed].tolist()
    return data.var_names.get_indexer(genes)
