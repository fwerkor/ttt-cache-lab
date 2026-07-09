from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.experiments.runner import ExperimentRunner
from ttt_cache_lab.experiments.summarize import summarize_csv, to_markdown, write_summary
from ttt_cache_lab.updates.targets import ModuleKind

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttt-cache-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an experiment from a YAML config")
    run.add_argument("--config", required=True, type=Path)

    summarize = subparsers.add_parser("summarize", help="Summarize an experiment summary.csv")
    summarize.add_argument("--input", required=True, type=Path)
    summarize.add_argument("--output", type=Path, default=None)

    subparsers.add_parser("list-targets", help="List supported update target names")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        console.print(f"[bold]Running experiment:[/bold] {config.name}")
        result = ExperimentRunner(config).run()
        console.print(f"Wrote {result.jsonl_path}")
        console.print(f"Wrote {result.csv_path}")
        return
    if args.command == "summarize":
        rows = summarize_csv(args.input)
        print(to_markdown(rows))
        if args.output:
            write_summary(rows, args.output)
            print(f"Wrote {args.output}")
        return
    if args.command == "list-targets":
        for item in ModuleKind:
            if item.value != "unknown":
                print(item.value)
        return


if __name__ == "__main__":
    main()
