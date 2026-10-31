"""Run candidate training in disposable spawn processes with bounded CPU concurrency."""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from loguru import logger

from .config import Request

# Resource policy from the worker contract, not a model hyperparameter.
MAX_THREADS_PER_CHILD = 4
THREAD_ENVIRONMENTS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
)


@contextmanager
def training_environment(threads: int) -> Iterator[None]:
    """Set child import-time CPU limits to threads; yield None and restore the environment on exit.

    The process pool must be created and shut down inside this context so replacement
    spawn processes inherit the limits too. This process-global context is not thread safe.
    """
    environment = {key: str(threads) for key in THREAD_ENVIRONMENTS}
    environment.update(
        TF_NUM_INTEROP_THREADS="1",
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={threads}",
    )
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_CHILD_CPU_GROUPS: Queue | None = None
_CHILD_CPUS: list[int] = []


def _initialize_child(cpu_groups: Queue) -> None:
    global _CHILD_CPU_GROUPS, _CHILD_CPUS

    # JAX 0.12.0.1 ignores the OMP/XLA thread limits for several runtime pools:
    # measured 869 threads with 208 visible CPUs versus 25 with four CPUs.
    # Lease a CPU group for this disposable process, including replacement workers.
    cpus = cpu_groups.get()
    os.sched_setaffinity(0, cpus)
    _CHILD_CPU_GROUPS, _CHILD_CPUS = cpu_groups, cpus


def _cpu_capacity(cpus: list[int]) -> int:
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    if not quota_path.exists():
        return len(cpus)
    quota, period = quota_path.read_text().split()
    if quota == "max":
        return len(cpus)
    return min(len(cpus), max(1, int(quota) // int(period)))


def _train_logged(request: Request, candidate: dict[str, Any]) -> dict[str, Any]:
    directory = request.work_dir / candidate["id"]
    directory.mkdir(parents=True, exist_ok=True)
    # Redirect file descriptors as well as Python streams: native JAX messages
    # must end up in the same candidate log. Each process runs exactly one task.
    with (directory / "train.log").open("w", encoding="utf-8") as log:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        logger.remove()
        logger.add(sys.stderr)
        try:
            from .training import train_candidate

            return train_candidate(request, candidate)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            # Release before multiprocessing runs its queue finalizers. An atexit
            # callback can start a feeder after its shutdown hook has already run.
            if _CHILD_CPU_GROUPS is not None:
                _CHILD_CPU_GROUPS.put(_CHILD_CPUS)


def _stop_workers(executor: ProcessPoolExecutor) -> None:
    # Python 3.12 has no terminate_workers (added in 3.14). Keep the single
    # compatibility access here: shutdown(cancel_futures=True) cannot stop training
    # already running. Snapshot before shutdown clears the process collection.
    processes = list(executor._processes.values())
    for process in processes:
        if process.is_alive():
            process.kill()
    executor.shutdown(wait=True, cancel_futures=True)
    for process in processes:
        process.join()


def train_candidates(request: Request) -> list[dict[str, Any]]:
    """Train request candidates in fresh processes; return results in request order.

    Concurrency is capped by request.max_parallel, candidate count and available
    CPUs. Each child inherits BLAS/OMP/XLA settings of at most four threads before imports.
    On failure kill active siblings, leave pending candidates unsubmitted, and raise
    RuntimeError naming the candidate and its log. No result is returned on failure.
    """
    cpus = sorted(os.sched_getaffinity(0))
    capacity = _cpu_capacity(cpus)
    parallel = min(request.max_parallel, len(request.candidates), capacity)
    threads = min(MAX_THREADS_PER_CHILD, max(1, capacity // parallel))
    context = multiprocessing.get_context("spawn")
    cpu_groups = context.Queue()
    for slot in range(parallel):
        cpu_groups.put(cpus[slot * threads : (slot + 1) * threads])
    results = {}
    try:
        with training_environment(threads):
            executor = ProcessPoolExecutor(
                max_workers=parallel,
                mp_context=context,
                initializer=_initialize_child,
                initargs=(cpu_groups,),
                max_tasks_per_child=1,
            )
            active = {}
            pending = iter(enumerate(request.candidates))
            try:
                for index, candidate in pending:
                    active[executor.submit(_train_logged, request, candidate)] = index
                    if len(active) == parallel:
                        break
                while active:
                    completed, _ = wait(active, return_when=FIRST_COMPLETED)
                    # Check every completion before submitting another task so a known
                    # failure never starts a pending candidate.
                    for future in completed:
                        index = active.pop(future)
                        candidate = request.candidates[index]
                        try:
                            results[index] = future.result()
                        except Exception as error:
                            raise RuntimeError(
                                f"candidate {candidate['id']} failed; "
                                f"see {request.work_dir / candidate['id'] / 'train.log'}: {error}"
                            ) from error
                    for _ in completed:
                        item = next(pending, None)
                        if item is not None:
                            index, candidate = item
                            active[executor.submit(_train_logged, request, candidate)] = index
            except BaseException:
                _stop_workers(executor)
                raise
            else:
                executor.shutdown(wait=True)
    finally:
        cpu_groups.close()
        cpu_groups.join_thread()
    return [results[index] for index in range(len(request.candidates))]
