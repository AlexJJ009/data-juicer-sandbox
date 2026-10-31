"""Register the single-cell hooks and launch a standard Sandbox configuration."""

from data_juicer_sandbox.cli.sandbox_starter import init_sandbox_configs
from data_juicer_sandbox.pipelines import SandBoxExecutor
from data_juicer_sandbox.specific_hooks.single_cell.hooks import register_hooks


def main(args: list[str] | None = None) -> None:
    """Parse optional CLI arguments, register hooks, run Sandbox, and return None."""
    register_hooks()
    config = init_sandbox_configs(args)
    SandBoxExecutor(config).run()


if __name__ == "__main__":
    main()
