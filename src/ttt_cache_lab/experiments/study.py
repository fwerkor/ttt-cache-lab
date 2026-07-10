from __future__ import annotations

import csv
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.experiments.results import ExperimentArtifacts
from ttt_cache_lab.experiments.static_adapters import StaticAdapterExperimentRunner
from ttt_cache_lab.experiments.versioned import VersionedExperimentRunner, write_version_summary


class StudyJobConfig(BaseModel):
    name: str
    config: Path
    runner: Literal["versioned", "static"] = "versioned"
    seeds: list[int] = Field(default_factory=lambda: [7, 17, 29])
    tags: list[str] = Field(default_factory=list)
    required_paths: list[Path] = Field(default_factory=list)
    enabled: bool = True


class StudyManifest(BaseModel):
    name: str
    output_dir: Path
    jobs: list[StudyJobConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> StudyManifest:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)


@dataclass(frozen=True)
class ExpandedStudyJob:
    index: int
    name: str
    config_path: Path
    runner: Literal["versioned", "static"]
    seed: int
    tags: tuple[str, ...]
    required_paths: tuple[Path, ...]
    output_dir: Path

    def command(self, manifest_path: Path) -> str:
        return (
            "python -m ttt_cache_lab.cli study-run "
            f"--manifest {shlex.quote(str(manifest_path))} --job-index {self.index}"
        )


def expand_study(manifest_path: Path) -> tuple[StudyManifest, list[ExpandedStudyJob]]:
    manifest = StudyManifest.from_yaml(manifest_path)
    output_root = _resolve_path(manifest.output_dir, manifest_path=manifest_path)
    expanded: list[ExpandedStudyJob] = []
    for job in manifest.jobs:
        if not job.enabled:
            continue
        config_path = _resolve_path(job.config, manifest_path=manifest_path)
        required = tuple(
            _resolve_path(path, manifest_path=manifest_path)
            for path in job.required_paths
        )
        for seed in job.seeds:
            expanded.append(
                ExpandedStudyJob(
                    index=len(expanded),
                    name=job.name,
                    config_path=config_path,
                    runner=job.runner,
                    seed=seed,
                    tags=tuple(job.tags),
                    required_paths=required,
                    output_dir=output_root / job.name / f"seed-{seed}",
                )
            )
    return manifest, expanded


def write_study_plan(manifest_path: Path, output_dir: Path | None = None) -> list[Path]:
    manifest, jobs = expand_study(manifest_path)
    destination = output_dir or _resolve_path(manifest.output_dir, manifest_path=manifest_path) / "plan"
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "job_matrix.csv"
    commands_path = destination / "commands.sh"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "job_index",
            "name",
            "runner",
            "seed",
            "tags",
            "config_path",
            "output_dir",
            "required_paths",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_index": job.index,
                    "name": job.name,
                    "runner": job.runner,
                    "seed": job.seed,
                    "tags": ",".join(job.tags),
                    "config_path": job.config_path,
                    "output_dir": job.output_dir,
                    "required_paths": ",".join(str(path) for path in job.required_paths),
                }
            )
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    commands.extend(job.command(manifest_path) for job in jobs)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    return [matrix_path, commands_path]


def select_study_jobs(
    manifest_path: Path,
    *,
    job_index: int | None = None,
    tag: str | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[ExpandedStudyJob]:
    _, jobs = expand_study(manifest_path)
    if job_index is not None:
        if job_index < 0 or job_index >= len(jobs):
            raise IndexError(f"job_index {job_index} is outside [0, {len(jobs)})")
        jobs = [jobs[job_index]]
    if tag is not None:
        jobs = [job for job in jobs if tag in job.tags]
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("shard_index and num_shards must be provided together")
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
        jobs = [job for job in jobs if job.index % num_shards == shard_index]
    return jobs


def run_study_job(job: ExpandedStudyJob) -> ExperimentArtifacts:
    missing = [path for path in job.required_paths if not path.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Study job {job.name!r} is missing required artifacts: {rendered}")
    config = VersionedExperimentConfig.from_yaml(job.config_path).model_copy(
        update={
            "seed": job.seed,
            "output_dir": job.output_dir,
        },
        deep=True,
    )
    if job.runner == "static":
        artifacts = StaticAdapterExperimentRunner(config).run()
    else:
        artifacts = VersionedExperimentRunner(config).run()
    write_version_summary(artifacts.csv_path, job.output_dir / "version_summary.csv")
    return artifacts


def _resolve_path(path: Path, *, manifest_path: Path) -> Path:
    if path.is_absolute():
        return path
    working_directory_candidate = Path.cwd() / path
    if working_directory_candidate.exists() or path.parts[:1] in {("runs",), ("configs",), ("scripts",)}:
        return working_directory_candidate
    return manifest_path.parent / path
