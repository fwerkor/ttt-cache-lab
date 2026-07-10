from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ttt_cache_lab.configs import ExperimentConfig, SweepConfig, VersionedExperimentConfig, VersionedSweepConfig
from ttt_cache_lab.experiments.failure_map import FailureThresholds, generate_failure_map
from ttt_cache_lab.experiments.pareto import generate_pareto
from ttt_cache_lab.experiments.report import generate_report
from ttt_cache_lab.experiments.runner import ExperimentRunner
from ttt_cache_lab.experiments.static_adapters import StaticAdapterExperimentRunner
from ttt_cache_lab.experiments.summarize import (
    first_table_markdown,
    summarize_csv,
    to_markdown,
    write_summary,
)
from ttt_cache_lab.experiments.sweep import run_sweep, run_versioned_sweep
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

    versioned_sweep = subparsers.add_parser("versioned-sweep", help="Run a YAML-defined versioned sweep")
    versioned_sweep.add_argument("--config", required=True, type=Path)

    versioned = subparsers.add_parser("versioned-run", help="Run a multi-step versioned adapter experiment")
    versioned.add_argument("--config", required=True, type=Path)
    versioned.add_argument("--version-summary", action="store_true")

    static_run = subparsers.add_parser("static-run", help="Run a fixed multi-adapter cache experiment")
    static_run.add_argument("--config", required=True, type=Path)
    static_run.add_argument("--version-summary", action="store_true")

    version_summary = subparsers.add_parser("version-summary", help="Summarize a versioned records CSV")
    version_summary.add_argument("--input", required=True, type=Path)
    version_summary.add_argument("--output", required=True, type=Path)

    version_report = subparsers.add_parser(
        "version-report", help="Generate Markdown and SVG report from versioned records"
    )
    version_report.add_argument("--input", required=True, type=Path)
    version_report.add_argument("--output-dir", required=True, type=Path)

    failure_map = subparsers.add_parser("failure-map", help="Generate E3 failure-map tables and heatmap")
    failure_map.add_argument("--input", required=True, type=Path)
    failure_map.add_argument("--output-dir", required=True, type=Path)
    failure_map.add_argument("--safe-kl", type=float, default=0.05)
    failure_map.add_argument("--safe-top1", type=float, default=0.99)
    failure_map.add_argument("--safe-task-drop", type=float, default=0.01)

    pareto = subparsers.add_parser("pareto", help="Generate E4 quality-cost Pareto table")
    pareto.add_argument("--input", required=True, type=Path)
    pareto.add_argument("--output-dir", required=True, type=Path)

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
    if args.command == "versioned-sweep":
        versioned_sweep_config = VersionedSweepConfig.from_yaml(args.config)
        artifacts = run_versioned_sweep(versioned_sweep_config)
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
    if args.command == "static-run":
        static_config = VersionedExperimentConfig.from_yaml(args.config)
        static_artifacts = StaticAdapterExperimentRunner(static_config).run()
        console.print(f"Wrote {static_artifacts.jsonl_path}")
        console.print(f"Wrote {static_artifacts.csv_path}")
        if args.version_summary:
            output = static_config.output_dir / "version_summary.csv"
            write_version_summary(static_artifacts.csv_path, output)
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
    if args.command == "failure-map":
        policy = generate_failure_map(
            args.input,
            args.output_dir,
            thresholds=FailureThresholds(
                safe_kl=args.safe_kl,
                safe_top1=args.safe_top1,
                safe_task_drop=args.safe_task_drop,
            ),
        )
        console.print(f"Wrote {policy}")
        return
    if args.command == "pareto":
        pareto_csv = generate_pareto(args.input, args.output_dir)
        console.print(f"Wrote {pareto_csv}")
        return
    if args.command == "list-targets":
        for item in ModuleKind:
            if item.value != "unknown":
                print(item.value)
        return


if __name__ == "__main__":
    main()
