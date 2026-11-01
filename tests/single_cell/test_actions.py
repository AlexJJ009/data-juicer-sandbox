"""Synthetic actions checks for the single-cell worker."""

import anndata as ad
import numpy as np

from data_juicer_sandbox.specific_hooks.single_cell.config import Request


def test_degs_cache_and_identity(payload, monkeypatch):
    from data_juicer_sandbox.specific_hooks.single_cell import actions, evaluation

    req = Request.from_dict(payload | {"n_degs": 3})
    data = ad.read_h5ad(req.data_path)
    calls = []

    def rank(held, **kwargs):
        calls.append(kwargs)
        assert set(held.obs.type) == {"held"}
        assert kwargs["method"] == "wilcoxon" and kwargs["use_raw"] is False
        assert kwargs["groups"] == ["s"] and kwargs["reference"] == "c"
        held.uns["rank_genes_groups"] = {"names": np.array([("3",), ("0",), ("2",)], dtype=[("s", "O")])}

    monkeypatch.setattr(evaluation.sc.tl, "rank_genes_groups", rank)
    np.testing.assert_array_equal(actions.cached_top_degs(data, req), [3, 0, 2])
    np.testing.assert_array_equal(actions.cached_top_degs(data, req), [3, 0, 2])
    assert len(calls) == 1
    # A modified file identity invalidates cached ranking even at the same path.
    data.write_h5ad(req.data_path)
    actions.cached_top_degs(data, req)
    assert len(calls) == 2


def test_evaluate_keeps_candidate_artifacts_and_uses_prediction(payload, monkeypatch):
    from data_juicer_sandbox.specific_hooks.single_cell import actions, training
    from data_juicer_sandbox.specific_hooks.single_cell.sampling import sample

    sampled = sample(Request.from_dict(payload))["candidates"][0]
    candidate = sampled | {"model_dir": "unused"}
    request = Request.from_dict(payload | {"action": "evaluate", "candidates": [candidate], "n_degs": 3})

    def predict(train, control, candidate, request, *, data):
        assert train.n_obs == 18 and control.n_obs == 6
        assert candidate["actual_mixture"] == {"A": 0.5, "B": 0.5}
        result = control.copy()
        result.X = result.X.toarray() + [3, 1, 2, 4]
        return result

    monkeypatch.setattr(training, "predict", predict)
    result = actions.evaluate(request)["candidates"][0]
    assert all(result[key] == value for key, value in candidate.items())
    assert result["metrics"]["r2_all"] > 0.999999
    assert set(result["baselines"]) == {"control", "linear"}
