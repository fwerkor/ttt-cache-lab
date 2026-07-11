from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ttt_cache_lab.configs import ExperimentConfig, SweepConfig, VersionedExperimentConfig, VersionedSweepConfig
from ttt_cache_lab.experiments.boundary_analysis import generate_boundary_analysis
from ttt_cache_lab.experiments.failure_map import FailureThresholds, generate_failure_map
from ttt_cache_lab.experiments.failures import capture_run_failure
from ttt_cache_lab.experiments.pareto import generate_pareto
from ttt_cache_lab.experiments.propagation_analysis import generate_propagation_analysis
from ttt_cache_lab.experiments.report import generate_report
from ttt_cache_lab.experiments.results import merge_record_files
from ttt_cache_lab.experiments.runner import ExperimentRunner
from ttt_cache_lab.experiments.static_adapters import StaticAdapterExperimentRunner
from ttt_cache_lab.experiments.statistics import generate_statistical_report
from ttt_cache_lab.experiments.study import run_study_job, select_study_jobs, write_study_plan
from ttt_cache_lab.experiments.study_analysis import StudyThresholds, generate_study_analysis
from ttt_cache_lab.experiments.summarize import (
    first_table_markdown,
    summarize_csv,
    to_markdown,
    write_summary,
)
from ttt_cache_lab.experiments.sweep import run_sweep, run_versioned_sweep
from ttt_cache_lab.experiments.task_probe import run_task_probe
from ttt_cache_lab.experiments.versioned import VersionedExperimentRunner, write_version_summary
from ttt_cache_lab.experiments.window_analysis import WindowThresholds, generate_window_analysis
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

    task_probe = subparsers.add_parser(
        "task-probe",
        help="Run baseline-only task viability checks before an expensive experiment",
    )
    task_probe.add_argument("--config", required=True, type=Path)
    task_probe.add_argument("--output-dir", required=True, type=Path)
    task_probe.add_argument("--max-samples", type=int, default=None)
    task_probe.add_argument("--min-mean-score", type=float, default=None)
    task_probe.add_argument("--max-mean-score", type=float, default=None)

    static_run = subparsers.add_parser("static-run", help="Run a fixed multi-adapter cache experiment")
    static_run.add_argument("--config", required=True, type=Path)
    static_run.add_argument("--version-summary", action="store_true")

    version_summary = subparsers.add_parser("version-summary", help="Summarize a versioned records CSV")
    version_summary.add_argument("--input", required=True, type=Path)
    version_summary.add_argument("--output", required=True, type=Path)

    merge_records = subparsers.add_parser(
        "merge-records", help="Merge one or more records.jsonl files for cross-run analysis"
    )
    merge_records.add_argument("--input", required=True, type=Path, nargs="+")
    merge_records.add_argument("--output-dir", required=True, type=Path)

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

    window_analysis = subparsers.add_parser(
        "window-analysis",
        help="Aggregate finite recompute-window sweeps and select minimum safe windows",
    )
    window_analysis.add_argument("--input", required=True, type=Path)
    window_analysis.add_argument("--output-dir", required=True, type=Path)
    window_analysis.add_argument("--safe-kl", type=float, default=0.05)
    window_analysis.add_argument("--safe-top1", type=float, default=0.99)
    window_analysis.add_argument("--safe-task-drop", type=float, default=0.01)
    window_analysis.add_argument("--min-safe-rate", type=float, default=0.95)

    propagation_analysis = subparsers.add_parser(
        "propagation-analysis",
        help="Aggregate layerwise propagation records into drift profiles",
    )
    propagation_analysis.add_argument("--input", required=True, type=Path)
    propagation_analysis.add_argument("--output-dir", required=True, type=Path)
    propagation_analysis.add_argument("--recovery-ratio", type=float, default=0.1)

    boundary_analysis = subparsers.add_parser(
        "boundary-analysis",
        help="Evaluate rejoin-layer compatibility metrics and a held-out predictor",
    )
    boundary_analysis.add_argument("--boundary-input", required=True, type=Path)
    boundary_analysis.add_argument("--summary-input", required=True, type=Path)
    boundary_analysis.add_argument("--output-dir", required=True, type=Path)
    boundary_analysis.add_argument("--ridge", type=float, default=1e-3)

    statistics = subparsers.add_parser(
        "statistics",
        help="Generate cluster-bootstrap confidence intervals and paired comparisons",
    )
    statistics.add_argument("--input", required=True, type=Path)
    statistics.add_argument("--output-dir", required=True, type=Path)
    statistics.add_argument("--reference-strategy", default="full_recompute")
    statistics.add_argument("--bootstrap-resamples", type=int, default=2000)
    statistics.add_argument("--confidence-level", type=float, default=0.95)
    statistics.add_argument("--seed", type=int, default=2027)

    study_plan = subparsers.add_parser(
        "study-plan",
        help="Expand a paper study manifest into a stable job matrix",
    )
    study_plan.add_argument("--manifest", required=True, type=Path)
    study_plan.add_argument("--output-dir", type=Path, default=None)

    study_run = subparsers.add_parser(
        "study-run",
        help="Run one job or one modulo shard from a paper study manifest",
    )
    study_run.add_argument("--manifest", required=True, type=Path)
    study_run.add_argument("--job-index", type=int, default=None)
    study_run.add_argument("--tag", default=None)
    study_run.add_argument("--shard-index", type=int, default=None)
    study_run.add_argument("--num-shards", type=int, default=None)
    study_run.add_argument("--dry-run", action="store_true")

    study_analysis = subparsers.add_parser(
        "study-analysis",
        help="Generate experiment-specific E1-E8 tables and figures from merged records",
    )
    study_analysis.add_argument("--input", required=True, type=Path)
    study_analysis.add_argument("--output-dir", required=True, type=Path)
    study_analysis.add_argument("--safe-kl", type=float, default=0.05)
    study_analysis.add_argument("--safe-top1", type=float, default=0.99)
    study_analysis.add_argument("--safe-task-drop", type=float, default=0.01)

    subparsers.add_parser("list-targets", help="List supported update target names")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        console.print(f"[bold]Running experiment:[/bold] {config.name}")
        result = capture_run_failure(
            config.output_dir,
            config,
            lambda: ExperimentRunner(config).run(),
        )
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
        artifacts = capture_run_failure(
            sweep_config.output_dir,
            sweep_config,
            lambda: run_sweep(sweep_config),
        )
        console.print(f"Wrote {artifacts.merged_records_csv}")
        console.print(f"Wrote {artifacts.grouped_csv}")
        return
    if args.command == "versioned-sweep":
        versioned_sweep_config = VersionedSweepConfig.from_yaml(args.config)
        artifacts = capture_run_failure(
            versioned_sweep_config.output_dir,
            versioned_sweep_config,
            lambda: run_versioned_sweep(versioned_sweep_config),
        )
        console.print(f"Wrote {artifacts.merged_records_csv}")
        console.print(f"Wrote {artifacts.grouped_csv}")
        return
    if args.command == "versioned-run":
        versioned_config = VersionedExperimentConfig.from_yaml(args.config)
        versioned_artifacts = capture_run_failure(
            versioned_config.output_dir,
            versioned_config,
            lambda: VersionedExperimentRunner(versioned_config).run(),
        )
        console.print(f"Wrote {versioned_artifacts.jsonl_path}")
        console.print(f"Wrote {versioned_artifacts.csv_path}")
        if args.version_summary:
            output = versioned_config.output_dir / "version_summary.csv"
            write_version_summary(versioned_artifacts.csv_path, output)
            console.print(f"Wrote {output}")
        return
    if args.command == "task-probe":
        probe_config = VersionedExperimentConfig.from_yaml(args.config)
        probe_artifacts = run_task_probe(
            probe_config,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            min_mean_score=args.min_mean_score,
            max_mean_score=args.max_mean_score,
        )
        console.print(f"Wrote {probe_artifacts.records_jsonl}")
        console.print(f"Wrote {probe_artifacts.records_csv}")
        console.print(f"Wrote {probe_artifacts.summary_json}")
        console.print(
            f"Mean score: {probe_artifacts.summary.mean_score:.6f} "
            f"({probe_artifacts.summary.sample_count} samples)"
        )
        return
    if args.command == "static-run":
        static_config = VersionedExperimentConfig.from_yaml(args.config)
        static_artifacts = capture_run_failure(
            static_config.output_dir,
            static_config,
            lambda: StaticAdapterExperimentRunner(static_config).run(),
        )
        console.print(f"Wrote {static_artifacts.jsonl_path}")
        console.print(f"Wrote {static_artifacts.csv_path}")
        if args.version_summary:
            output = static_config.output_dir / "version_summary.csv"
            write_version_summary(static_artifacts.csv_path, output)
            console.print(f"Wrote {output}")
        return
    if args.command == "merge-records":
        merged_artifacts = merge_record_files(args.input, args.output_dir)
        console.print(f"Wrote {merged_artifacts.jsonl_path}")
        console.print(f"Wrote {merged_artifacts.csv_path}")
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
    if args.command == "window-analysis":
        cells, minima = generate_window_analysis(
            args.input,
            args.output_dir,
            thresholds=WindowThresholds(
                safe_kl=args.safe_kl,
                safe_top1=args.safe_top1,
                safe_task_drop=args.safe_task_drop,
                min_safe_rate=args.min_safe_rate,
            ),
        )
        console.print(f"Wrote {cells}")
        console.print(f"Wrote {minima}")
        return
    if args.command == "propagation-analysis":
        layers, profiles = generate_propagation_analysis(
            args.input,
            args.output_dir,
            recovery_ratio=args.recovery_ratio,
        )
        console.print(f"Wrote {layers}")
        console.print(f"Wrote {profiles}")
        return
    if args.command == "boundary-analysis":
        boundary_artifacts = generate_boundary_analysis(
            args.boundary_input,
            args.summary_input,
            args.output_dir,
            ridge=args.ridge,
        )
        console.print(f"Wrote {boundary_artifacts.enriched_rows_path}")
        console.print(f"Wrote {boundary_artifacts.metric_evaluation_path}")
        console.print(f"Wrote {boundary_artifacts.group_selections_path}")
        console.print(f"Wrote {boundary_artifacts.predictor_summary_path}")
        return
    if args.command == "statistics":
        outputs = generate_statistical_report(
            args.input,
            args.output_dir,
            reference_strategy=args.reference_strategy,
            bootstrap_resamples=args.bootstrap_resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
        for output in outputs:
            console.print(f"Wrote {output}")
        return
    if args.command == "study-plan":
        for output in write_study_plan(args.manifest, args.output_dir):
            console.print(f"Wrote {output}")
        return
    if args.command == "study-run":
        jobs = select_study_jobs(
            args.manifest,
            job_index=args.job_index,
            tag=args.tag,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        for job in jobs:
            console.print(
                f"[bold]Study job {job.index}:[/bold] {job.name} seed={job.seed} -> {job.output_dir}"
            )
            if not args.dry_run:
                study_artifacts = run_study_job(job)
                console.print(f"Wrote {study_artifacts.csv_path}")
        return
    if args.command == "study-analysis":
        outputs = generate_study_analysis(
            args.input,
            args.output_dir,
            thresholds=StudyThresholds(
                safe_kl=args.safe_kl,
                safe_top1=args.safe_top1,
                safe_task_drop=args.safe_task_drop,
            ),
        )
        for output in outputs:
            console.print(f"Wrote {output}")
        return
    if args.command == "list-targets":
        for item in ModuleKind:
            if item.value != "unknown":
                print(item.value)
        return


if __name__ == "__main__":
    main()
