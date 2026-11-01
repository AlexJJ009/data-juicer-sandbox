"""Synthetic parallel checks for the single-cell worker."""

from pathlib import Path

import pytest

from data_juicer_sandbox.specific_hooks.single_cell.config import Request
from data_juicer_sandbox.specific_hooks.single_cell.sampling import sample
from data_juicer_sandbox.specific_hooks.single_cell.worker import dispatch


def test_train_subprocess_failure_names_candidate(payload):
    candidate = sample(Request.from_dict(payload))["candidates"][0]
    candidate["manifest"] = str(Path(payload["work_dir"]) / "missing.json")
    req = Request.from_dict(payload | {"action": "train", "candidates": [candidate], "max_parallel": 4})
    with pytest.raises(RuntimeError, match="candidate r1 failed"):
        dispatch(req)
    log = Path(payload["work_dir"]) / "r1" / "train.log"
    assert "missing.json" in log.read_text()


@pytest.mark.parametrize("fail", [False, True])
def test_pool_bounds_order_environment_and_failure(payload, monkeypatch, fail):
    import os
    from concurrent.futures import Future

    from data_juicer_sandbox.specific_hooks.single_cell import parallel

    candidates = [payload["candidates"][0] | {"id": f"c{i}", "manifest": "unused"} for i in range(4)]
    request = Request.from_dict(payload | {"action": "train", "candidates": candidates, "max_parallel": 2})
    started, stopped = [], []
    original = os.environ.get("OMP_NUM_THREADS")
    active = set()

    class Process:
        def is_alive(self):
            return True

        def kill(self):
            stopped.append("killed")

        def join(self):
            stopped.append("joined")

    class Pool:
        _processes = {1: Process(), 2: Process()}

        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["max_tasks_per_child"] == 1
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert os.environ["OMP_NUM_THREADS"] == "4"
            assert os.environ["JAX_PLATFORMS"] == "cpu"

        def submit(self, function, req, candidate):
            assert function is parallel._train_logged
            assert len(active) < 2
            future = Future()
            started.append(candidate["id"])
            if fail and candidate["id"] == "c0":
                future.set_exception(ValueError("broken"))
            else:
                future.set_result(candidate)
            active.add(future)
            return future

        def shutdown(self, **kwargs):
            stopped.append(kwargs)

    def complete(futures, **kwargs):
        # Complete in reverse order to check result ordering independently.
        done = set(futures)
        active.difference_update(done)
        return done, set()

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda _: set(range(8)))
    monkeypatch.setattr(parallel, "wait", complete)
    if fail:
        with pytest.raises(RuntimeError, match="candidate c0 failed"):
            parallel.train_candidates(request)
        assert started == ["c0", "c1"]
        assert stopped.count("killed") == stopped.count("joined") == 2
        assert {"wait": True, "cancel_futures": True} in stopped
    else:
        assert parallel.train_candidates(request) == candidates
        assert started == ["c0", "c1", "c2", "c3"]
        assert stopped == [{"wait": True}]
    assert os.environ.get("OMP_NUM_THREADS") == original


def _inspect_child(request, candidate):
    import os

    from data_juicer_sandbox.specific_hooks.single_cell import parallel, training

    def train_candidate(request, candidate):
        return dict(
            candidate, pid=os.getpid(), cpus=sorted(os.sched_getaffinity(0)), threads=os.environ["OMP_NUM_THREADS"]
        )

    training.train_candidate = train_candidate
    return parallel._train_logged(request, candidate)


def test_spawn_recycles_processes_and_releases_cpu_groups(payload, monkeypatch):
    from data_juicer_sandbox.specific_hooks.single_cell import parallel

    candidates = [payload["candidates"][0] | {"id": f"c{i}", "manifest": "unused"} for i in range(4)]
    request = Request.from_dict(payload | {"action": "train", "candidates": candidates, "max_parallel": 2})
    monkeypatch.setattr(parallel, "_train_logged", _inspect_child)
    results = parallel.train_candidates(request)
    assert len({candidate["pid"] for candidate in results}) == 4
    assert [candidate["id"] for candidate in results] == [candidate["id"] for candidate in candidates]
    assert all(len(candidate["cpus"]) == int(candidate["threads"]) <= 4 for candidate in results)


@pytest.mark.parametrize("inherited", [None, "cpu"])
def test_auto_backend_child_environment_is_bounded_and_restored(
    monkeypatch: pytest.MonkeyPatch, inherited: str | None
) -> None:
    """Allow JAX discovery without overriding an explicit CPU policy or preallocating the card."""
    import os

    from data_juicer_sandbox.specific_hooks.single_cell.parallel import (
        training_environment,
    )

    if inherited is None:
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    else:
        monkeypatch.setenv("JAX_PLATFORMS", inherited)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    with training_environment(4, backend="auto", parallel=5):
        assert os.environ.get("JAX_PLATFORMS", "") == (inherited or "")
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) == pytest.approx(0.15)
    assert os.environ.get("JAX_PLATFORMS") == inherited
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
