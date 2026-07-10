from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ttt_cache_lab.configs import (
    ExperimentConfig,
    SweepAxis,
    SweepConfig,
    VersionedExperimentConfig,
    VersionedSweepConfig,
)
from ttt_cache_lab.experiments.runner import ExperimentRunner
from ttt_cache_lab.experiments.summarize import summarize_csv, write_summary
from ttt_cache_lab.experiments.versioned import VersionedExperimentRunner, write_version_summary


@dataclass(frozen=True)
class SweepArtifacts:
    output_dir: Path
    merged_records_csv: Path
    grouped_csv: Path
    run_dirs: list[Path]


def run_sweep(config: SweepConfig) -> SweepArtifacts:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    merged_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for experiment in config.expand():
        if experiment.output_dir.exists() and not experiment.resume:
            shutil.rmtree(experiment.output_dir)
        artifacts = ExperimentRunner(experiment).run()
        run_dirs.append(experiment.output_dir)
        with artifacts.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = [
                    "run_name",
                    *(f"sweep.{axis.path}" for axis in config.axes),
                    *list(reader.fieldnames or []),
                ]
            metadata = _sweep_metadata(experiment, config.axes)
            for row in reader:
                merged_rows.append({"run_name": experiment.name, **metadata, **row})

    merged = config.output_dir / "merged_records.csv"
    with merged.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    grouped = config.output_dir / "grouped.csv"
    rows = summarize_csv(merged)
    write_summary(rows, grouped)

    return SweepArtifacts(
        output_dir=config.output_dir,
        merged_records_csv=merged,
        grouped_csv=grouped,
        run_dirs=run_dirs,
    )



def run_versioned_sweep(config: VersionedSweepConfig) -> SweepArtifacts:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    merged_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for experiment in config.expand():
        if experiment.output_dir.exists() and not experiment.resume:
            shutil.rmtree(experiment.output_dir)
        artifacts = VersionedExperimentRunner(experiment).run()
        run_dirs.append(experiment.output_dir)
        with artifacts.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = [
                    "run_name",
                    *(f"sweep.{axis.path}" for axis in config.axes),
                    *list(reader.fieldnames or []),
                ]
            metadata = _sweep_metadata(experiment, config.axes)
            for row in reader:
                merged_rows.append({"run_name": experiment.name, **metadata, **row})

    merged = config.output_dir / "merged_records.csv"
    with merged.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(merged_rows)

    grouped = config.output_dir / "version_summary.csv"
    write_version_summary(merged, grouped)
    return SweepArtifacts(
        output_dir=config.output_dir,
        merged_records_csv=merged,
        grouped_csv=grouped,
        run_dirs=run_dirs,
    )


def _sweep_metadata(
    experiment: ExperimentConfig | VersionedExperimentConfig,
    axes: Sequence[SweepAxis],
) -> dict[str, str]:
    payload = experiment.model_dump(mode="json")
    metadata: dict[str, str] = {}
    for axis in axes:
        path = axis.path
        value: Any = payload
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"Expanded experiment is missing sweep axis {path!r}")
            value = value[part]
        metadata[f"sweep.{path}"] = _csv_value(value)
    return metadata


def _csv_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
