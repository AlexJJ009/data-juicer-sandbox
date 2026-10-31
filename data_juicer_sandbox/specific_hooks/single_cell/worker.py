"""Parse worker CLI requests, dispatch registered actions and atomically publish JSON results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from .actions import evaluate, sample, train
from .config import Request
from .mixtures import suggest

Action = Callable[[Request], dict[str, Any]]
ACTIONS: dict[str, Action] = {}


def register_action(name: str, action: Action) -> None:
    """Register an action callable taking Request and returning a JSON object; reject duplicate names.

    Args:
        name: Action name, including suggest for the mixing integration.
        action: Callable receiving a validated Request and returning a JSON object.

    Returns:
        None. The callable is installed in ACTIONS; duplicate names raise ValueError.
    """
    if not isinstance(name, str) or not name or not callable(action):
        raise ValueError("action registration requires a nonempty name and callable")
    if name in ACTIONS:
        raise ValueError(f"action already registered: {name}")
    ACTIONS[name] = action


register_action("sample", sample)
register_action("train", train)
register_action("evaluate", evaluate)
register_action("suggest", suggest)


def dispatch(request: Request) -> dict[str, Any]:
    """Execute the registered request action and return its JSON result, or reject unsupported actions."""
    if request.action not in ACTIONS:
        raise ValueError(f"unsupported action: {request.action}; registered actions: {sorted(ACTIONS)}")
    return ACTIONS[request.action](request)


def main(argv: list[str] | None = None) -> int:
    """Run the JSON worker CLI; atomically write result only on success and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    temporary: str | None = None
    try:
        request = Request.from_json(args.request)
        result = dispatch(request)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.result.parent, prefix=f".{args.result.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(result, handle, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, args.result)
        return 0
    except Exception as error:
        logger.error("worker failed: {}", error)
        return 1
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
