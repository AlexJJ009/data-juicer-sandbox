"""Synthetic worker checks for the single-cell worker."""

import json
import subprocess
import sys

from data_juicer_sandbox.specific_hooks.single_cell.config import Request
from data_juicer_sandbox.specific_hooks.single_cell.worker import (
    ACTIONS,
    dispatch,
    register_action,
)


def test_dispatch_extension_and_cli_error(payload, tmp_path):
    register_action("test_extension", lambda req: {"ok": req.holdout_cell_type})
    try:
        assert dispatch(Request.from_dict(payload | {"action": "test_extension"})) == {"ok": "held"}
    finally:
        ACTIONS.pop("test_extension")
    req_path, res_path = tmp_path / "req.json", tmp_path / "res.json"
    req_path.write_text(json.dumps(payload | {"action": "bad"}))
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "data_juicer_sandbox.specific_hooks.single_cell.worker",
            "--request",
            str(req_path),
            "--result",
            str(res_path),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0
    assert "bad" in done.stderr
    assert not res_path.exists()


def test_cli_success(payload, tmp_path):
    req_path, res_path = tmp_path / "request.json", tmp_path / "result.json"
    req_path.write_text(json.dumps(payload))
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "data_juicer_sandbox.specific_hooks.single_cell.worker",
            "--request",
            str(req_path),
            "--result",
            str(res_path),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(res_path.read_text())
    assert result["candidates"][0]["composition"]["held|c"] == 6
