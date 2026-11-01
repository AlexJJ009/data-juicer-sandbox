"""Bridge Sandbox stages to isolated single-cell workers through JSON files."""

import json
import subprocess
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

from data_juicer.config import prepare_side_configs
from jsonargparse import Namespace
from loguru import logger

from data_juicer_sandbox.hooks import HOOK_MAPPING, BaseHook

from .observability import record_evaluation, record_suggestions

LOG_TAIL_LINES = 30

WORKER_MODULE = "data_juicer_sandbox.specific_hooks.single_cell.worker"


def _log_diagnostic(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as log:
        tail = "".join(deque(log, maxlen=LOG_TAIL_LINES))
    return f"see {path}; last {LOG_TAIL_LINES} lines:\n{tail}"


class SingleCellWorkerHook(BaseHook):
    """Run one worker action using extra_configs and return configured outputs.

    Args:
        job_cfg: Standard BaseHook settings. extra_configs supplies python,
            request, and optionally worker_module or a worker_script for testing.
        watcher: Sandbox watcher, forwarded to BaseHook.
    """

    action: ClassVar[str]

    def specify_dj_and_extra_configs(self, allow_fail: bool = False) -> None:
        """Load worker settings as dictionaries; return None.

        Args:
            allow_fail: BaseHook compatibility argument. Invalid settings still fail.
        """
        if self.dj_cfg:
            raise ValueError("Single-cell hooks accept extra_configs only.")
        # Cell-type names contain spaces. Namespace conversion rejects these
        # keys on reinitialization after input/updater mappings, so retain JSON
        # dictionaries while using the inherited BaseHook mapping machinery.
        settings = prepare_side_configs(self.extra_cfg)
        self.extra_cfg = settings.as_dict() if isinstance(settings, Namespace) else settings

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Adapt a copied request to this stage and return the worker payload."""
        return request

    def prepare_result(self, request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Adapt a worker result using its request and return available outputs."""
        return result

    def hook(self, **kwargs: Any) -> tuple[Any, ...]:
        """Call the worker with extra_configs; return values in output_keys order.

        Raises RuntimeError with the log path and tail on worker failure or invalid results.
        """
        settings = self.extra_cfg
        if not settings or not settings.get("python"):
            raise ValueError("single-cell extra_configs must specify a Python executable and request.")
        request = deepcopy(settings["request"])
        iteration = self.curr_iter if self.curr_iter is not None else 1
        directory = Path(request["work_dir"]).resolve() / f"iter_{iteration}"
        directory.mkdir(parents=True, exist_ok=True)
        request.update(action=self.action, work_dir=str(directory))
        request = self.prepare_request(request)
        request_path = directory / f"{self.action}.request.json"
        result_path = directory / f"{self.action}.result.json"
        # Delete even when invocation fails so a previous successful run cannot
        # masquerade as the result of the current worker invocation.
        result_path.unlink(missing_ok=True)
        request_path.write_text(json.dumps(request, indent=2, allow_nan=False))
        command = [settings["python"]]
        if settings.get("worker_script"):
            command.append(settings["worker_script"])
        else:
            command.extend(["-m", settings.get("worker_module", WORKER_MODULE)])
        command.extend(["--request", str(request_path), "--result", str(result_path)])
        logger.info("Single-cell {} iteration {}: {}", self.action, iteration, request_path)
        log_path = directory / f"{self.action}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Worker {self.action} exited {completed.returncode}; {_log_diagnostic(log_path)}")
        if not result_path.is_file():
            raise RuntimeError(f"Worker {self.action} returned no result at {result_path}; {_log_diagnostic(log_path)}")
        try:
            result = json.loads(result_path.read_text())
        except (ValueError, OSError) as error:
            raise RuntimeError(
                f"Worker {self.action} result cannot be read: {error}; {_log_diagnostic(log_path)}"
            ) from error
        if not isinstance(result, dict):
            raise ValueError(f"Worker {self.action} result must be a JSON object; {_log_diagnostic(log_path)}")
        result = self.prepare_result(request, result)
        if self.action == "evaluate":
            metric = settings["request"].get("metric", "r2_top_degs")
            record_evaluation(result["candidates"], directory, iteration, metric, self.watcher)
        elif self.action == "suggest":
            record_suggestions(result["mixtures"], directory, iteration, self.watcher)
        result.update(request_path=str(request_path), result_path=str(result_path))
        keys = self.output_keys or list(result)
        missing = [key for key in keys if key not in result]
        if missing:
            raise ValueError(f"Worker {self.action} result missing output keys: {missing}; {_log_diagnostic(log_path)}")
        return tuple(result[key] for key in keys)


class SingleCellSampleHook(SingleCellWorkerHook):
    """Expand configured mixtures and seeds, then return sampled worker candidates."""

    action = "sample"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Expand mixtures × seeds with iteration-specific IDs; return sample payload."""
        if "mixtures" in request:
            mixtures = request.pop("mixtures")
            seeds = request.pop("seeds")
            if not mixtures or not seeds:
                raise ValueError("Sampling requires nonempty mixtures and seeds.")
            if len(set(seeds)) != len(seeds):
                raise ValueError("Sampling seeds must be unique to avoid duplicate candidate IDs.")
            iteration = self.curr_iter if self.curr_iter is not None else 1
            request["candidates"] = [
                {"id": f"r{iteration}-m{i}-s{seed}", "mixture": mixture, "seed": seed}
                for i, mixture in enumerate(mixtures, start=1)
                for seed in seeds
            ]
        return request


class SingleCellTrainHook(SingleCellWorkerHook):
    """Forward sampled candidates to training and return trained worker candidates."""

    action = "train"


class SingleCellEvaluateHook(SingleCellWorkerHook):
    """Evaluate trained candidates and return scored candidates plus search history."""

    action = "evaluate"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Keep metric and accumulated history in the hook; return evaluation payload."""
        request.pop("history", None)
        request.pop("metric", None)
        return request

    def prepare_result(self, request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Append this round's configured metric to prior history; return both outputs."""
        settings = self.extra_cfg
        options = settings["request"]
        metric = options.get("metric", "r2_top_degs")
        history = deepcopy(options.get("history", []))
        for candidate in result["candidates"]:
            if metric not in candidate.get("metrics", {}):
                raise ValueError(f'Evaluation result for {candidate.get("id")} lacks metric {metric!r}.')
            history.append({"mixture": candidate["actual_mixture"], "score": candidate["metrics"][metric]})
        return {**result, "history": history}


class SingleCellSuggestHook(SingleCellWorkerHook):
    """Forward scored history to CLIMB and return suggested mixtures for the updater."""

    action = "suggest"


def register_hooks() -> None:
    """Register the four single-cell hook classes before executor creation; return None."""
    for hook in (SingleCellSampleHook, SingleCellTrainHook, SingleCellEvaluateHook, SingleCellSuggestHook):
        HOOK_MAPPING[hook.__name__] = hook
