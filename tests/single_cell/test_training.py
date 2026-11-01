"""Synthetic training checks for the single-cell worker."""

import sys
from typing import Any

import anndata as ad
import numpy as np
import pytest

from data_juicer_sandbox.specific_hooks.single_cell.config import Request
from data_juicer_sandbox.specific_hooks.single_cell.sampling import sample


def test_training_parameter_forwarding(payload, monkeypatch):
    import types

    from data_juicer_sandbox.specific_hooks.single_cell.training import train_candidate

    candidate = sample(Request.from_dict(payload))["candidates"][0]
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(default_backend=lambda: "cpu", devices=lambda: []))
    calls = {}

    class FakeScgen:
        @staticmethod
        def setup_anndata(data, **kwargs):
            calls["setup"] = kwargs
            assert data.n_obs == 18

        def __init__(self, data):
            self.history = {}

        def train(self, **kwargs):
            calls["train"] = kwargs
            self.history["train_loss"] = [1.0]

        def save(self, directory, **kwargs):
            calls["save"] = (directory, kwargs)

    monkeypatch.setitem(sys.modules, "pertpy.tools", types.SimpleNamespace(Scgen=FakeScgen))
    req = Request.from_dict(
        payload | {"train": {"max_epochs": 1, "batch_size": 8, "early_stopping": False, "early_stopping_patience": 2}}
    )
    result = train_candidate(req, candidate)
    assert calls["setup"] == dict(batch_key="condition", labels_key="type")
    assert calls["train"] == dict(max_epochs=1, batch_size=8, early_stopping=False, early_stopping_patience=2, seed=11)
    assert calls["save"] == (req.work_dir / "r1" / "model", {"overwrite": True})
    assert result["epochs_trained"] == 1 and result["train_seconds"] >= 0


def test_prediction_randomness_is_seeded_and_restored(payload, monkeypatch):
    import types

    from data_juicer_sandbox.specific_hooks.single_cell.training import predict

    candidate = sample(Request.from_dict(payload))["candidates"][0] | {"model_dir": "unused"}
    req = Request.from_dict(payload)
    data = ad.read_h5ad(req.data_path)
    original_rng = np.random.default_rng
    calls = []

    class FakeScgen:
        @classmethod
        def load(cls, directory, adata):
            assert directory == "unused"
            return cls()

        def predict(self, **kwargs):
            calls.append(kwargs)
            return np.random.default_rng().normal(size=4), None

    monkeypatch.setitem(sys.modules, "pertpy.tools", types.SimpleNamespace(Scgen=FakeScgen))
    first = predict(data, data[:6], candidate, req)
    np.testing.assert_array_equal(first, predict(data, data[:6], candidate, req))
    assert not np.array_equal(first, predict(data, data[:6], candidate | {"seed": 12}, req))
    assert np.random.default_rng is original_rng
    assert calls[0]["restrict_arithmetic_to"] == {"type": ["A", "B"]}


def test_seeded_rng_restores_after_error_and_preserves_explicit_seed():
    import pytest

    from data_juicer_sandbox.specific_hooks.single_cell.training import (
        seeded_numpy_default_rng,
    )

    original = np.random.default_rng
    expected = original(123).normal(size=4)
    with pytest.raises(RuntimeError, match="prediction failed"):
        with seeded_numpy_default_rng(11):
            np.testing.assert_array_equal(np.random.default_rng(123).normal(size=4), expected)
            first, second = np.random.default_rng().normal(size=4), np.random.default_rng().normal(size=4)
            assert not np.array_equal(first, second)
            raise RuntimeError("prediction failed")
    assert np.random.default_rng is original
    with seeded_numpy_default_rng(11):
        np.testing.assert_array_equal(np.random.default_rng().normal(size=4), first)


def test_auto_backend_warns_on_cpu_fallback(payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Report CPU fallback and actual backend without requiring GPU hardware."""
    import types

    from data_juicer_sandbox.specific_hooks.single_cell import training

    candidate = sample(Request.from_dict(payload))["candidates"][0]
    warnings = []

    class FakeScgen:
        """Minimal Scgen replacement for backend reporting."""

        history = {"train_loss": [1.0]}

        def __init__(self, data: ad.AnnData) -> None:
            """Accept synthetic training data."""

        @staticmethod
        def setup_anndata(data: ad.AnnData, **kwargs: Any) -> None:
            """Accept registration parameters."""

        def train(self, **kwargs: Any) -> None:
            """Accept training parameters."""

        def save(self, directory: object, **kwargs: Any) -> None:
            """Accept a model output path."""

    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(default_backend=lambda: "cpu", devices=lambda: []))
    monkeypatch.setitem(sys.modules, "pertpy.tools", types.SimpleNamespace(Scgen=FakeScgen))
    monkeypatch.setattr(training.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))
    result = training.train_candidate(Request.from_dict(payload | {"training_backend": "auto"}), candidate)
    assert result["training_backend"] == "cpu"
    assert len(warnings) == 1 and "falling back to CPU" in warnings[0]
