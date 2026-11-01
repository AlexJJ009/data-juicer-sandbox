"""Register the single-cell hooks and launch a standard Sandbox configuration."""

import os

from data_juicer_sandbox.cli.sandbox_starter import init_sandbox_configs
from data_juicer_sandbox.pipelines import SandBoxExecutor
from data_juicer_sandbox.specific_hooks.single_cell.hooks import register_hooks


def main(args: list[str] | None = None) -> None:
    """Parse optional CLI arguments, register hooks, run Sandbox, and return None."""
    if os.environ.get("WANDB_MODE") == "offline":
        # WandB 0.19.0 otherwise queries max_cli_version when serializing Tables,
        # even offline. Use its public compatibility switch; never require a key.
        # https://github.com/wandb/wandb/blob/v0.19.0/wandb/sdk/data_types/base_types/wb_value.py
        os.environ.setdefault("WANDB_ALLOW_OFFLINE_ARTIFACTS", "false")
    register_hooks()
    config = init_sandbox_configs(args)
    SandBoxExecutor(config).run()


if __name__ == "__main__":
    main()
