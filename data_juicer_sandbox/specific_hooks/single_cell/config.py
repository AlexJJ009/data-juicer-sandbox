"""Parse and validate the shared JSON contract before any worker action runs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TRAINING_BACKEND = "cpu"
DEFAULT_MAX_PARALLEL = 1
DEFAULT_N_DEGS = 100
DEFAULT_HOLDOUT_CELL_TYPE = "CD4 T cells"
SEED_MAX = 2**32 - 1
MIXTURE_SUM_TOLERANCE = 1e-8


REQUIRED_ARTIFACTS = {"train": ("manifest",), "evaluate": ("manifest", "model_dir")}
CANDIDATE_ACTIONS = {"sample", "train", "evaluate"}


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Fields:
    """Configured obs columns; each value names a cell type, condition or donor column."""

    cell_type: str
    condition: str
    donor: str


@dataclass(frozen=True)
class Conditions:
    """Configured control and perturbed condition names in the input obs table."""

    control: str
    perturbed: str


@dataclass(frozen=True)
class TrainOptions:
    """Protocol training defaults, passed unchanged to Scgen.train.

    Defaults follow the task contract; API: https://pertpy.readthedocs.io/en/stable/usage/tools/pertpy.tools.Scgen.html
    """

    max_epochs: int = 100
    batch_size: int = 32
    early_stopping: bool = True
    early_stopping_patience: int = 25


@dataclass(frozen=True)
class Request:
    """Validated worker request; candidates retain artifacts from previous actions.

    training_backend defaults to cpu for compatibility with the original worker;
    auto enables JAX GPU discovery and CPU fallback. An explicit JAX_PLATFORMS
    environment policy still takes precedence in auto mode.
    Use from_dict or from_json for validation. Extra top-level fields are retained
    in payload so an independently registered suggest action can parse its options.
    """

    action: str
    data_path: Path
    work_dir: Path
    fields: Fields
    conditions: Conditions
    holdout_cell_type: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    budget: int | None = None
    train: TrainOptions = field(default_factory=TrainOptions)
    training_backend: str = DEFAULT_TRAINING_BACKEND
    max_parallel: int = DEFAULT_MAX_PARALLEL
    n_degs: int = DEFAULT_N_DEGS
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> Request:
        """Read a UTF-8 JSON request at path and return a validated Request."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Request:
        """Validate JSON-compatible value and return a request, raising ValueError on invalid fields."""
        if not isinstance(value, dict):
            raise ValueError("request must be a JSON object")
        names = _parse_names(value)
        action = _text(value.get("action"), "action")
        holdout = _text(value.get("holdout_cell_type", DEFAULT_HOLDOUT_CELL_TYPE), "holdout_cell_type")
        budget = value.get("budget")
        if budget is not None or action == "sample":
            budget = _integer(budget, "budget")
        backend = value.get("training_backend", DEFAULT_TRAINING_BACKEND)
        if backend not in ("cpu", "auto"):
            raise ValueError("training_backend must be cpu or auto")
        options = _parse_train_options(value)
        candidates = _parse_candidates(value.get("candidates", []), action, holdout)
        return cls(
            action=action,
            data_path=Path(_text(value.get("data_path"), "data_path")).resolve(),
            work_dir=Path(_text(value.get("work_dir"), "work_dir")).resolve(),
            holdout_cell_type=holdout,
            candidates=candidates,
            budget=budget,
            train=options,
            training_backend=backend,
            max_parallel=_integer(value.get("max_parallel", DEFAULT_MAX_PARALLEL), "max_parallel"),
            n_degs=_integer(value.get("n_degs", DEFAULT_N_DEGS), "n_degs", minimum=2),
            payload=dict(value),
            **names,
        )


def _parse_names(value: dict[str, Any]) -> dict[str, Fields | Conditions]:
    names = {}
    for key, schema in (("fields", Fields), ("conditions", Conditions)):
        raw = value.get(key)
        if not isinstance(raw, dict) or set(raw) != set(schema.__dataclass_fields__):
            raise ValueError(f"{key} requires {list(schema.__dataclass_fields__)}")
        names[key] = schema(**{k: _text(v, f"{key}.{k}") for k, v in raw.items()})
    if names["conditions"].control == names["conditions"].perturbed:
        raise ValueError("conditions must be distinct")
    return names


def _parse_train_options(value: dict[str, Any]) -> TrainOptions:
    raw_options = value.get("train", {})
    if not isinstance(raw_options, dict) or set(raw_options) - set(TrainOptions.__dataclass_fields__):
        raise ValueError("train contains unsupported options")
    options = TrainOptions(**raw_options)
    for key in ("max_epochs", "batch_size", "early_stopping_patience"):
        _integer(getattr(options, key), f"train.{key}")
    if type(options.early_stopping) is not bool:
        raise ValueError("train.early_stopping must be boolean")
    return options


def _parse_candidates(raw: Any, action: str, holdout: str) -> list[dict[str, Any]]:
    candidates = raw
    if not isinstance(candidates, list) or (action in CANDIDATE_ACTIONS and not candidates):
        raise ValueError("candidates must be a nonempty list for sample/train/evaluate")
    ids = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        cid = candidate.get("id")
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cid):
            raise ValueError("candidate id must be a safe single path component")
        if cid in ids:
            raise ValueError(f"duplicate candidate id: {cid}")
        ids.add(cid)
        seed = _integer(candidate.get("seed"), f"candidate {cid} seed", minimum=0)
        if seed > SEED_MAX:
            raise ValueError(f"seed must be <= {SEED_MAX}")
        _validate_mixture(candidate, holdout)
        for artifact in REQUIRED_ARTIFACTS.get(action, ()):
            _text(candidate.get(artifact), f"candidate {cid} {artifact}")
    return candidates


def _validate_mixture(candidate: dict[str, Any], holdout: str) -> None:
    cid = candidate["id"]
    mixture = candidate.get("mixture")
    if not isinstance(mixture, dict) or not mixture or holdout in mixture:
        raise ValueError(f"candidate {cid}: mixture must exclude holdout and be nonempty")
    for cell_type, weight in mixture.items():
        _text(cell_type, "mixture cell type")
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("mixture weights must be finite nonnegative numbers")
    if not math.isclose(sum(mixture.values()), 1.0, abs_tol=MIXTURE_SUM_TOLERANCE, rel_tol=0):
        raise ValueError("mixture weights must sum to 1")
