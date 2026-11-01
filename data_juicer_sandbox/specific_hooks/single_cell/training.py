"""Adapt pertpy's JAX Scgen training, persistence and seeded holdout prediction."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import anndata as ad
import numpy as np
from loguru import logger

from .config import Request
from .reference import arithmetic_population


def train_candidate(request: Request, candidate: dict[str, Any]) -> dict[str, Any]:
    """Train one candidate with protocol parameters and seed; return model path, seconds and actual epochs.

    Args:
        request: Validated shared configuration and training options.
        candidate: Sample result containing id, seed, mixture and manifest.

    Returns:
        Candidate fields plus model_dir, train_seconds and epochs_trained.

    Called in a CPU-limited fresh worker process. Uses Scgen architecture defaults:
    https://pertpy.readthedocs.io/en/stable/usage/tools/pertpy.tools.Scgen.html
    """
    import jax
    from pertpy.tools import Scgen

    from .sampling import read_data, training_subset

    backend = jax.default_backend()
    if request.training_backend == "auto" and backend == "cpu":
        logger.warning("GPU unavailable or disabled; falling back to CPU for {}", candidate["id"])
    logger.info("Candidate {} uses JAX backend {}: {}", candidate["id"], backend, jax.devices())
    train = training_subset(read_data(request), candidate, request)
    seed = candidate["seed"]
    random.seed(seed)
    np.random.seed(seed)
    # Current pertpy uses its own JAX loop, not scvi's trainer. Scgen.train(seed)
    # forwards this seed to JAX PRNGKey as well as split and minibatch generators.
    Scgen.setup_anndata(train, batch_key=request.fields.condition, labels_key=request.fields.cell_type)
    model = Scgen(train)
    start = time.monotonic()
    model.train(**asdict(request.train), seed=seed)
    seconds = time.monotonic() - start
    epochs = len(model.history["train_loss"])
    if epochs < 1:
        raise RuntimeError(f"candidate {candidate['id']}: no training epochs recorded")
    directory = request.work_dir / candidate["id"] / "model"
    model.save(directory, overwrite=True)
    logger.info("Trained {}: {} epochs, {:.3f}s", candidate["id"], epochs, seconds)
    return dict(
        candidate, model_dir=str(directory), train_seconds=seconds, epochs_trained=epochs, training_backend=backend
    )


def predict(
    train: ad.AnnData,
    control: ad.AnnData,
    candidate: dict[str, Any],
    request: Request,
    *,
    data: ad.AnnData | None = None,
) -> ad.AnnData:
    """Load a candidate model and return seeded Scgen.predict output for holdout controls.

    Args:
        train: Candidate training subset, in the order recorded by its manifest.
        control: Held-out type's control cells to predict.
        candidate: Train result containing model_dir and seed.
        request: Validated field, condition, holdout and reference policy.
        data: Optional complete data to avoid repeated disk reads in full mode.

    Returns:
        AnnData containing predicted log-normalized stimulated expression.

    pertpy 1.4.0 predict and balancer create unseeded default_rng instances. A scoped
    replacement supplies reproducible child streams without modifying pertpy source.
    This function runs serially in the worker; the replacement is restored on error.
    """
    from pertpy.tools import Scgen

    reference = arithmetic_population(train, request, data)
    model = Scgen.load(candidate["model_dir"], adata=reference)
    # Exclude the held-out type from latent arithmetic: it has no matched stim.
    types = sorted(set(reference.obs[request.fields.cell_type]) - {request.holdout_cell_type})
    if not types:
        raise ValueError(f"candidate {candidate['id']}: no non-holdout training types")
    with seeded_numpy_default_rng(candidate["seed"]):
        predicted, _ = model.predict(
            ctrl_key=request.conditions.control,
            stim_key=request.conditions.perturbed,
            adata_to_predict=control,
            restrict_arithmetic_to={request.fields.cell_type: types},
        )
    return predicted


@contextmanager
def seeded_numpy_default_rng(seed: int) -> Iterator[None]:
    """Temporarily seed unseeded NumPy generators; yield no value and restore on exit.

    Args:
        seed: SeedSequence root for independent, reproducible child streams.

    pertpy 1.4.0 uses unseeded default_rng in tools/_scgen/_scgen.py:410
    (predict) and tools/_scgen/_utils.py:73 (balancer). Only calls without an
    explicit seed are replaced. This process-global context is not thread safe;
    use it only around serial prediction in an isolated worker process.
    Source: https://github.com/scverse/pertpy/tree/1.4.0/pertpy/tools/_scgen
    """
    factory = np.random.default_rng
    sequence = np.random.SeedSequence(seed)

    def seeded_rng(seed: Any = None) -> np.random.Generator:
        return factory(sequence.spawn(1)[0] if seed is None else seed)

    np.random.default_rng = seeded_rng
    try:
        yield
    finally:
        np.random.default_rng = factory
