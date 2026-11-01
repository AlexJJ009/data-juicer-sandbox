"""Select the shared inference and linear-baseline population independently of sampling."""

from __future__ import annotations

import anndata as ad

from .config import Request


def arithmetic_population(train: ad.AnnData, request: Request, data: ad.AnnData | None = None) -> ad.AnnData:
    """Return the configured reference, retaining holdout controls but never holdout stim.

    Args:
        train: Manifest-validated candidate training population for legacy mode.
        request: Validated reference policy and obs field names.
        data: Complete input data, optionally shared by the evaluation orchestrator.
            If omitted in full mode, read request.data_path. Ignored in train mode.

    Returns:
        The original training population in train mode, or a copied full reference
        with paired non-holdout types and holdout controls. Unpaired types cannot
        define a perturbation and are excluded in full mode. No paired types is an error.

    Full is the request default. With train, pertpy balances every type present in
    the candidate, so a mixture with 2,394 B and 6 CD8 cells predicts with B and CD8
    weighted equally: inference, not training composition, then drives the score.
    Scgen.load accepts different cells with identical
    ordered var_names and restores its saved obs column setup:
    https://github.com/scverse/pertpy/blob/1.4.0/pertpy/tools/_scgen/_scgen.py
    """
    if request.arithmetic_reference == "train":
        return train
    if data is None:
        from .sampling import read_data

        data = read_data(request)
    types = data.obs[request.fields.cell_type]
    conditions = data.obs[request.fields.condition]
    paired = set(types[conditions == request.conditions.control]) & set(
        types[conditions == request.conditions.perturbed]
    )
    paired.discard(request.holdout_cell_type)
    if not paired:
        raise ValueError("full arithmetic reference requires a paired non-holdout cell type")
    keep = types.isin(paired) | ((types == request.holdout_cell_type) & (conditions == request.conditions.control))
    return data[keep].copy()
