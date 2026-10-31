"""Average actual sampled mixtures and search cell-type weights with the CLIMB predictor."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from data_juicer_sandbox.thirdparty.climb.predictor import fit_predictor

from .config import MIXTURE_SUM_TOLERANCE, SEED_MAX, Request

# Search defaults follow the pinned CLIMB source. The uniform prior substitutes
# for token shares when no cell availability shares are supplied.
# https://github.com/NVIDIA/NeMo-Curator/blob/fdee8f8884cd0104772183c7938c56dbaf30841c/tutorials/text/nemotron-climb-data-curation/8_predict.py
DEFAULT_NUM_SAMPLES = 100_000
# The unchanged 90/10 split must leave at least two training rows for LightGBM.
MIN_DISTINCT_MIXTURES = 3
MIN_TOP_POOL = 128
PRIOR_FLOOR = 1e-12
# Suggestion count and seed follow the single-cell task's JSON protocol.
DEFAULT_N_SUGGESTIONS = 5
DEFAULT_SEED = 0


@dataclass(frozen=True)
class _SuggestOptions:
    domains: list[str]
    grouped: dict[tuple[float, ...], list[float]]
    n_suggestions: int
    num_samples: int
    seed: int
    top_k: int
    alphas: np.ndarray


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _weights(value: Any, domains: list[str], name: str) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != set(domains):
        raise ValueError(f"{name} must have the same cell-type keys as history.")
    if any(type(v) not in (int, float) for v in value.values()):
        raise ValueError(f"{name} weights must be finite nonnegative numbers.")
    weights = np.asarray([value[key] for key in domains], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"{name} weights must be finite nonnegative numbers with a positive total.")
    return weights


def _parse_options(request: Request) -> _SuggestOptions:
    payload = request.payload
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("history must be a list of mixture/score objects.")
    grouped: dict[tuple[float, ...], list[float]] = {}
    domains: list[str] = []
    for index, row in enumerate(history):
        if not isinstance(row, dict) or not isinstance(row.get("mixture"), dict) or "score" not in row:
            raise ValueError(f"history[{index}] requires a mixture object and score.")
        if index == 0:
            if not row["mixture"] or any(not isinstance(k, str) or not k.strip() for k in row["mixture"]):
                raise ValueError("History mixture cell-type names must be nonempty strings.")
            domains = sorted(row["mixture"])
            if request.holdout_cell_type in domains:
                raise ValueError("History mixtures must exclude holdout_cell_type.")
        weights = _weights(row["mixture"], domains, f"history[{index}].mixture")
        if not np.isclose(weights.sum(), 1, atol=MIXTURE_SUM_TOLERANCE, rtol=0):
            raise ValueError("History mixture weights must sum to 1.")
        score = row["score"]
        if type(score) not in (int, float) or not np.isfinite(score):
            raise ValueError(f"history[{index}].score must be a finite number.")
        grouped.setdefault(tuple(weights), []).append(float(score))
    if len(grouped) < MIN_DISTINCT_MIXTURES:
        raise ValueError(
            f"CLIMB requires at least {MIN_DISTINCT_MIXTURES} distinct mixtures after averaging seeds; got {len(grouped)}."
        )
    count = _integer(payload.get("n_suggestions", DEFAULT_N_SUGGESTIONS), "n_suggestions")
    samples = _integer(payload.get("num_samples", DEFAULT_NUM_SAMPLES), "num_samples")
    if samples < count:
        raise ValueError("num_samples must be >= n_suggestions.")
    seed = _integer(payload.get("seed", DEFAULT_SEED), "seed", minimum=0)
    if seed > SEED_MAX:
        raise ValueError(f"seed must be <= {SEED_MAX}.")
    top_k = _integer(payload.get("top_k", min(samples, max(count * count, MIN_TOP_POOL))), "top_k")
    if not count <= top_k <= samples:
        raise ValueError("top_k must be between n_suggestions and num_samples.")
    prior = payload.get("prior", dict.fromkeys(domains, 1 / len(domains)))
    alphas = _weights(prior, domains, "prior")
    alphas = np.maximum(alphas / alphas.sum(), PRIOR_FLOOR)
    return _SuggestOptions(domains, grouped, count, samples, seed, top_k, alphas)


def suggest(request: Request) -> dict[str, list[dict[str, float]]]:
    """Return seeded CLIMB suggestions from validated shared fields and payload options.

    Args:
        request: Worker Request; payload supplies history (actual mixture, score),
            n_suggestions, seed, num_samples, top_k and Dirichlet prior shares.
            Search defaults follow the pinned CLIMB URL above.
    Returns:
        A mixtures list of n_suggestions cell-type weight dictionaries, after
        averaging repeated history mixtures before fitting the original regressor.
    Raises:
        ValueError: Invalid search options, nonfinite scores or insufficient history.
    """
    options = _parse_options(request)
    frame = pd.DataFrame(list(options.grouped), columns=options.domains)
    # Avoid collisions with arbitrary cell-type strings used as feature names.
    target = "__score__"
    while target in options.domains:
        target += "_"
    frame[target] = [float(np.mean(scores)) for scores in options.grouped.values()]
    predictor = fit_predictor(frame, options.domains, target)
    rng = np.random.default_rng(options.seed)
    samples = rng.dirichlet(options.alphas, options.num_samples)
    predictions = predictor.predict(samples)
    if not np.isfinite(predictions).all():
        raise ValueError("CLIMB predictor returned nonfinite scores.")
    if np.ptp(predictions) == 0:
        logger.warning("CLIMB predictions are constant; suggestions sample a tied top-score pool.")
    # Adapted from generate_mixtures in the pinned source: randomly choose within
    # the best-scoring pool to avoid returning near-identical maxima.
    if options.n_suggestions == 1:
        chosen = samples[[np.argmax(predictions)]]
    else:
        pool = samples[np.argsort(predictions)[-options.top_k :]]
        chosen = pool[rng.choice(len(pool), options.n_suggestions, replace=False)]
    logger.info(
        "CLIMB in {} fitted {} distinct mixtures; scored {} candidates",
        request.work_dir,
        len(frame),
        options.num_samples,
    )
    return {"mixtures": [dict(zip(options.domains, map(float, weights))) for weights in chosen]}
