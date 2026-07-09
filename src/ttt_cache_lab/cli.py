from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ttt_cache_lab.configs import ExperimentConfig, SweepConfig, VersionedExperimentConfig
from ttt_cache_lab.experiments.report import generate_report
from ttt_cache_lab.experiments.runner import ExperimentRunner
from ttt_cache_lab.experiments.summarize import (
    first_table_markdown,
    summarize_csv,
    to_markdown,
    write_summary,
)
from ttt_cache_lab.experiments.sweep import run_sweep
from ttt_cache_lab.experiments.versioned import VersionedExperimentRunner, write_version_summary
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

    first_table = subparsers.add_parser("first-table", help="Render the first feasibility table")
    first_table.add_argument("--input", required=True, type=Path)

    sweep = subparsers.add_parser("sweep", help="Run a YAML-defined sweep")
    sweep.add_argument("--config", required=True, type=Path)

    versioned = subparsers.add_parser("versioned-run", help="Run a multi-step versioned adapter experiment")
    versioned.add_argument("--config", required=True, type=Path)
    versioned.add_argument("--version-summary", action="store_true")

    version_summary = subparsers.add_parser("version-summary", help="Summarize a versioned records CSV")
    version_summary.add_argument("--input", required=True, type=Path)
    version_summary.add_argument("--output", required=True, type=Path)

    version_report = subparsers.add_parser(
        "version-report", help="Generate Markdown and SVG report from versioned records"
    )
    version_report.add_argument("--input", required=True, type=Path)
    version_report.add_argument("--output-dir", required=True, type=Path)

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
    if args.command == "first-table":
        print(first_table_markdown(summarize_csv(args.input)))
        return
    if args.command == "sweep":
        sweep_config = SweepConfig.from_yaml(args.config)
        artifacts = run_sweep(sweep_config)
        console.print(f"Wrote {artifacts.merged_records_csv}")
        console.print(f"Wrote {artifacts.grouped_csv}")
        return
    if args.command == "versioned-run":
        versioned_config = VersionedExperimentConfig.from_yaml(args.config)
        versioned_artifacts = VersionedExperimentRunner(versioned_config).run()
        console.print(f"Wrote {versioned_artifacts.jsonl_path}")
        console.print(f"Wrote {versioned_artifacts.csv_path}")
        if args.version_summary:
            output = versioned_config.output_dir / "version_summary.csv"
            write_version_summary(versioned_artifacts.csv_path, output)
            console.print(f"Wrote {output}")
        return
    if args.command == "version-summary":
        write_version_summary(args.input, args.output)
        console.print(f"Wrote {args.output}")
        return
    if args.command == "version-report":
        report = generate_report(args.input, args.output_dir)
        console.print(f"Wrote {report}")
        return
    if args.command == "list-targets":
        for item in ModuleKind:
            if item.value != "unknown":
                print(item.value)
        return


if __name__ == "__main__":
    main()
