"""Exercise worker handoff, CLIMB search, and the two-iteration Sandbox loop."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from data_juicer_sandbox.specific_hooks.single_cell.config import Request
from data_juicer_sandbox.specific_hooks.single_cell.hooks import (
    SingleCellEvaluateHook,
    SingleCellSampleHook,
)
from data_juicer_sandbox.specific_hooks.single_cell.launch import main
from data_juicer_sandbox.specific_hooks.single_cell.mixtures import suggest

CONFIG = Path("configs/single_cell/kang2018_mixing.yaml")
FAKE_WORKER = """\
import argparse
import json
from pathlib import Path
from data_juicer_sandbox.specific_hooks.single_cell.mixtures import suggest
from data_juicer_sandbox.specific_hooks.single_cell.config import Request
p = argparse.ArgumentParser()
p.add_argument('--request')
p.add_argument('--result')
a = p.parse_args()
r = json.loads(Path(a.request).read_text())
action = r['action']
with (Path(r['work_dir']).parent / 'calls.jsonl').open('a') as f:
    f.write(json.dumps(r) + '\\n')
if action == 'suggest':
    result = suggest(Request.from_dict(r))
else:
    candidates = r['candidates']
    for c in candidates:
        if action == 'sample':
            weights = dict(c['mixture'])
            weights[next(iter(weights))] *= 0.5
            total = sum(weights.values())
            c.update(manifest=str(Path(r['work_dir']) / c['id'] / 'manifest.json'), composition={},
                     actual_mixture={k: v / total for k, v in weights.items()})
        elif action == 'train':
            assert 'manifest' in c
            c.update(model_dir=str(Path(r['work_dir']) / c['id'] / 'model'), train_seconds=0, epochs_trained=1)
        elif action == 'evaluate':
            assert 'model_dir' in c
            score = c['actual_mixture'][next(iter(c['actual_mixture']))]
            c.update(metrics={'r2_top_degs': score, 'r2_all': score / 2}, baselines={})
    result = {'candidates': candidates}
Path(a.result).write_text(json.dumps(result))
"""


def search_request(options: dict) -> Request:
    """Build a validated shared request from synthetic search options; return Request."""
    return Request.from_dict(
        {
            "action": "suggest",
            "data_path": "unused.h5ad",
            "work_dir": ".",
            "fields": {"cell_type": "type", "condition": "condition", "donor": "donor"},
            "conditions": {"control": "c", "perturbed": "s"},
            "holdout_cell_type": "held",
            **options,
        }
    )


@pytest.fixture
def fake_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a protocol-compatible worker and expose the source checkout to it."""
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER)
    return path


def sample_hook(tmp_path: Path, script: Path, outputs: list[str] | None = None) -> SingleCellSampleHook:
    """Build a sample hook with small deterministic candidates for handoff tests."""
    return SingleCellSampleHook(
        {
            "hook": "SingleCellSampleHook",
            "output": outputs or ["candidates"],
            "extra_configs": {
                "python": sys.executable,
                "worker_script": str(script),
                "request": {"work_dir": str(tmp_path), "budget": 10, "mixtures": [{"A": 1.0}], "seeds": [11, 23]},
            },
        },
        None,
    )


@pytest.mark.parametrize("mode", ["script", "module"])
def test_handoff_and_output_order(
    tmp_path: Path, fake_worker: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Preserve candidate IDs, request locations, and configured output ordering."""
    hook = sample_hook(tmp_path, fake_worker, ["request_path", "candidates"])
    if mode == "module":
        hook.extra_cfg.pop("worker_script")
        hook.extra_cfg["worker_module"] = fake_worker.stem
        monkeypatch.setenv("PYTHONPATH", f"{fake_worker.parent}:{Path.cwd()}")
    hook.curr_iter = 2
    request_path, candidates = hook.hook()
    request = json.loads(Path(request_path).read_text())
    assert request["action"] == "sample"
    assert [c["id"] for c in request["candidates"]] == ["r2-m1-s11", "r2-m1-s23"]
    assert candidates[0]["manifest"].endswith("r2-m1-s11/manifest.json")
    assert Path(request_path).parent == tmp_path / "iter_2"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_stale_result_and_failure(tmp_path: Path, exit_code: int) -> None:
    """Reject failed or resultless workers even when a stale result exists."""
    script = tmp_path / "broken.py"
    script.write_text(f'import sys; print("worker diagnostic", file=sys.stderr); sys.exit({exit_code})')
    hook = sample_hook(tmp_path, script)
    hook.curr_iter = 1
    directory = tmp_path / "iter_1"
    directory.mkdir()
    result = directory / "sample.result.json"
    result.write_text('{"candidates": ["stale"]}')
    with pytest.raises(RuntimeError, match="worker diagnostic"):
        hook.hook()
    assert not result.exists()


@pytest.mark.parametrize("n_suggestions", [1, 5])
def test_suggest_direction_reproducibility_and_mean(n_suggestions: int) -> None:
    """Favor the high-score direction and average repeated seeds reproducibly."""
    history = []
    for x in np.linspace(0, 1, 120):
        history.extend(
            [{"mixture": {"A": float(x), "B": float(1 - x)}, "score": float(x) + noise} for noise in (-0.05, 0.05)]
        )
    request = {"history": history, "n_suggestions": n_suggestions, "seed": 9, "num_samples": 3000}
    result = suggest(search_request(request))
    assert result == suggest(search_request(request))
    means = history[::2]
    for row in means:
        row["score"] += 0.05
    assert result == suggest(search_request({**request, "history": means}))
    assert np.mean([m["A"] for m in result["mixtures"]]) > 0.75
    assert all(sum(m.values()) == pytest.approx(1) for m in result["mixtures"])


@pytest.mark.parametrize("history", [[], [{"mixture": {"A": 1}, "score": 1}] * 3])
def test_insufficient_distinct_history(history: list[dict]) -> None:
    """Count independent mixtures rather than repeated evaluation seeds."""
    with pytest.raises(ValueError, match="at least 3 distinct mixtures"):
        suggest(search_request({"history": history, "n_suggestions": 5, "seed": 0}))


@pytest.mark.parametrize("metric", ["r2_top_degs", "r2_all"])
def test_two_round_pipeline(tmp_path: Path, fake_worker: Path, monkeypatch: pytest.MonkeyPatch, metric: str) -> None:
    """Run the real Sandbox loop with configurable metric and iteration mappings."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    config = yaml.safe_load(CONFIG.read_text())
    config["work_dir"] = str(tmp_path / "run")
    config["evaluation_job_configs"][0]["extra_configs"]["request"]["metric"] = metric
    for job in config["execution_job_configs"] + config["evaluation_job_configs"]:
        job["extra_configs"]["python"] = sys.executable
        job["extra_configs"]["worker_script"] = str(fake_worker)
        job["extra_configs"]["request"]["work_dir"] = config["work_dir"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["--config", str(path)])
    root = Path(config["work_dir"])
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    assert [r["action"] for r in calls] == ["sample", "train", "evaluate", "suggest"] * 2
    assert len(calls[0]["candidates"]) == len(calls[4]["candidates"]) == 15
    suggestions = json.loads((root / "iter_1/suggest.result.json").read_text())["mixtures"]
    assert [c["mixture"] for c in calls[4]["candidates"][::3]] == suggestions
    assert [c["seed"] for c in calls[4]["candidates"][:3]] == [11, 23, 37]
    assert calls[3]["history"][0]["score"] == pytest.approx(
        json.loads((root / "iter_1/sample.result.json").read_text())["candidates"][0]["actual_mixture"]["B cells"]
        / (2 if metric == "r2_all" else 1)
    )
    for iteration, call in ((1, calls[3]), (2, calls[7])):
        candidates = json.loads((root / f"iter_{iteration}/evaluate.result.json").read_text())["candidates"]
        assert [row["mixture"] for row in call["history"][-15:]] == [c["actual_mixture"] for c in candidates]
        assert any(c["actual_mixture"] != c["mixture"] for c in candidates)
    assert len(calls[3]["history"]) == 15
    assert len(calls[7]["history"]) == 30
    assert len(json.loads((root / "context_infos.json").read_text())) == 2


def test_worker_cli_suggest(payload: dict, tmp_path: Path) -> None:
    """Dispatch suggest through the real worker JSON CLI and validate its result."""
    payload.update(
        action="suggest",
        candidates=[],
        history=[{"mixture": {"A": x, "B": 1 - x}, "score": x} for x in (0.1, 0.5, 0.9)],
        n_suggestions=3,
        seed=7,
        num_samples=100,
    )
    request, result = tmp_path / "suggest.json", tmp_path / "result.json"
    request.write_text(json.dumps(payload))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "data_juicer_sandbox.specific_hooks.single_cell.worker",
            "--request",
            str(request),
            "--result",
            str(result),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    mixtures = json.loads(result.read_text())["mixtures"]
    assert len(mixtures) == 3
    assert all(set(m) == {"A", "B"} and sum(m.values()) == pytest.approx(1) for m in mixtures)


def test_evaluation_history_uses_actual_mixture(tmp_path: Path) -> None:
    """Fit on sampled proportions when cell scarcity changes requested weights."""
    hook = SingleCellEvaluateHook(
        {
            "hook": "SingleCellEvaluateHook",
            "extra_configs": {
                "python": sys.executable,
                "request": {"work_dir": str(tmp_path), "history": [{"mixture": {"A": 1.0, "B": 0.0}, "score": 0.2}]},
            },
        },
        None,
    )
    candidate = {
        "id": "scarce",
        "mixture": {"A": 0.5, "B": 0.5},
        "actual_mixture": {"A": 0.8, "B": 0.2},
        "metrics": {"r2_top_degs": 0.9},
    }
    result = hook.prepare_result({}, {"candidates": [candidate]})
    assert result["history"] == [
        {"mixture": {"A": 1.0, "B": 0.0}, "score": 0.2},
        {"mixture": {"A": 0.8, "B": 0.2}, "score": 0.9},
    ]


@pytest.mark.parametrize(
    "options, message",
    [
        ({"history": {}}, "history must be a list"),
        ({"history": [None]}, "requires a mixture object and score"),
        ({"history": [{"mixture": {"held": 1}, "score": 1}]}, "exclude holdout"),
        ({"history": [{"mixture": {"A": 0.1}, "score": 1}]}, "sum to 1"),
        ({"history": [{"mixture": {"A": 1}, "score": None}]}, "finite number"),
        ({"history": [{"mixture": {"A": "1"}, "score": 1}]}, "finite nonnegative"),
        ({"history": [{"mixture": {"A": 1}, "score": float("nan")}]}, "finite number"),
        ({"history": [{"mixture": {"A": float("inf")}, "score": 1}]}, "finite nonnegative"),
        ({"n_suggestions": True}, "n_suggestions"),
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": 1}, "num_samples must be >="),
        ({"seed": -1}, "seed"),
        ({"seed": "0"}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"top_k": 1}, "top_k must be between"),
        ({"top_k": 100001}, "top_k must be between"),
        ({"prior": {"A": 0, "B": 0}}, "positive total"),
        ({"prior": {"A": 1}}, "same cell-type keys"),
    ],
)
def test_invalid_search_options(options: dict, message: str) -> None:
    """Reject malformed payload options with explanatory ValueError messages."""
    history = [{"mixture": {"A": x, "B": 1 - x}, "score": x} for x in (0.1, 0.5, 0.9)]
    with pytest.raises(ValueError, match=message):
        suggest(search_request({"history": history, **options}))


def test_two_distinct_mixtures_explain_training_minimum() -> None:
    """Explain the LightGBM two-training-row minimum before the CLIMB split."""
    history = [{"mixture": {"A": x, "B": 1 - x}, "score": x} for x in (0.1, 0.9)]
    with pytest.raises(ValueError, match="at least 3 distinct mixtures"):
        suggest(search_request({"history": history}))
