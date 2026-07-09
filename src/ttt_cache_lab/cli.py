from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.experiments.runner import ExperimentRunner

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttt-cache-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an experiment from a YAML config")
    run.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        console.print(f"[bold]Running experiment:[/bold] {config.name}")
        result = ExperimentRunner(config).run()
        console.print(f"Wrote {result.jsonl_path}")
        console.print(f"Wrote {result.csv_path}")


if __name__ == "__main__":
    main()
