from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.data.loader import build_task_samples
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.models.interface import ModelBackend


@dataclass(frozen=True)
class TaskProbeRecord:
    sample_id: int
    dataset_sample_id: str
    task_name: str
    task_family: str
    model_name: str
    context_length: int
    neutral_padding_tokens: int
    synthetic_difficulty: str
    prompt_format: str
    answer: str
    generated_text: str
    task_score: float
    latency_seconds: float
    memory_allocated: int
    peak_memory_allocated: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskProbeSummary:
    sample_count: int
    mean_score: float
    minimum_score: float
    maximum_score: float
    nonzero_fraction: float
    perfect_fraction: float
    mean_latency_seconds: float
    degenerate_all_zero: bool
    degenerate_all_one: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskProbeArtifacts:
    records_jsonl: Path
    records_csv: Path
    summary_json: Path
    records: tuple[TaskProbeRecord, ...]
    summary: TaskProbeSummary


def run_task_probe(
    config: VersionedExperimentConfig,
    *,
    output_dir: Path,
    max_samples: int | None = None,
    min_mean_score: float | None = None,
    max_mean_score: float | None = None,
    min_nonzero_fraction: float | None = None,
    max_perfect_fraction: float | None = None,
) -> TaskProbeArtifacts:
    backend = build_backend(config.model, seed=config.seed)
    backend.configure_metrics(capture_attention=False, capture_hidden_states=False)
    samples = build_task_samples(config.data, seed=config.seed)
    return run_task_probe_with_backend(
        config,
        backend=backend,
        samples=samples,
        output_dir=output_dir,
        max_samples=max_samples,
        min_mean_score=min_mean_score,
        max_mean_score=max_mean_score,
        min_nonzero_fraction=min_nonzero_fraction,
        max_perfect_fraction=max_perfect_fraction,
    )


def run_task_probe_with_backend(
    config: VersionedExperimentConfig,
    *,
    backend: ModelBackend,
    samples: Sequence[TaskSample],
    output_dir: Path,
    max_samples: int | None = None,
    min_mean_score: float | None = None,
    max_mean_score: float | None = None,
    min_nonzero_fraction: float | None = None,
    max_perfect_fraction: float | None = None,
) -> TaskProbeArtifacts:
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    for name, value in (
        ("min_mean_score", min_mean_score),
        ("max_mean_score", max_mean_score),
        ("min_nonzero_fraction", min_nonzero_fraction),
        ("max_perfect_fraction", max_perfect_fraction),
    ):
        _validate_threshold(name, value)
    if min_mean_score is not None and max_mean_score is not None and min_mean_score > max_mean_score:
        raise ValueError("min_mean_score cannot exceed max_mean_score")

    selected_samples = list(samples[:max_samples] if max_samples is not None else samples)
    if not selected_samples:
        raise ValueError("Task probe selected no samples")

    records: list[TaskProbeRecord] = []
    for sample_id, raw_sample in enumerate(selected_samples):
        sample = backend.prepare_sample(raw_sample, context_length=config.data.context_length)
        started = time.perf_counter()
        output = backend.prefill(sample.prompt)
        latency = time.perf_counter() - started
        extras = output.extras or {}
        score = float(backend.score_answer(sample, output))
        if not math.isfinite(score):
            raise ValueError(f"Task probe produced a non-finite score for sample {sample_id}")
        records.append(
            TaskProbeRecord(
                sample_id=sample_id,
                dataset_sample_id=str(sample.metadata.get("dataset_sample_id", sample_id)),
                task_name=str(sample.metadata.get("task", config.data.task)),
                task_family=str(sample.metadata.get("task_family", config.data.task_family)),
                model_name=str(config.model.model_name_or_path or config.model.backend),
                context_length=config.data.context_length,
                neutral_padding_tokens=int(sample.metadata.get("neutral_padding_tokens", 0)),
                synthetic_difficulty=str(sample.metadata.get("synthetic_difficulty", config.data.synthetic_difficulty)),
                prompt_format="chat_template" if config.model.use_chat_template else "plain",
                answer=sample.answer,
                generated_text=str(extras.get("generated_text", "")),
                task_score=score,
                latency_seconds=latency,
                memory_allocated=int(extras.get("memory_allocated", 0)),
                peak_memory_allocated=int(extras.get("peak_memory_allocated", 0)),
            )
        )

    scores = [record.task_score for record in records]
    summary = TaskProbeSummary(
        sample_count=len(records),
        mean_score=mean(scores),
        minimum_score=min(scores),
        maximum_score=max(scores),
        nonzero_fraction=sum(score > 0.0 for score in scores) / len(scores),
        perfect_fraction=sum(score >= 1.0 for score in scores) / len(scores),
        mean_latency_seconds=mean(record.latency_seconds for record in records),
        degenerate_all_zero=all(score == 0.0 for score in scores),
        degenerate_all_one=all(score >= 1.0 for score in scores),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records_jsonl = output_dir / "task_probe.jsonl"
    records_csv = output_dir / "task_probe.csv"
    summary_json = output_dir / "task_probe_summary.json"
    _atomic_write_jsonl(records_jsonl, records)
    _atomic_write_csv(records_csv, records)
    _atomic_write_json(summary_json, summary.to_dict())

    if min_mean_score is not None and summary.mean_score < min_mean_score:
        raise RuntimeError(
            f"Task probe mean score {summary.mean_score:.6f} is below the required minimum "
            f"{min_mean_score:.6f}; artifacts were written to {output_dir}"
        )
    if max_mean_score is not None and summary.mean_score > max_mean_score:
        raise RuntimeError(
            f"Task probe mean score {summary.mean_score:.6f} exceeds the required maximum "
            f"{max_mean_score:.6f}; artifacts were written to {output_dir}"
        )
    if min_nonzero_fraction is not None and summary.nonzero_fraction < min_nonzero_fraction:
        raise RuntimeError(
            f"Task probe nonzero fraction {summary.nonzero_fraction:.6f} is below the required "
            f"minimum {min_nonzero_fraction:.6f}; artifacts were written to {output_dir}"
        )
    if max_perfect_fraction is not None and summary.perfect_fraction > max_perfect_fraction:
        raise RuntimeError(
            f"Task probe perfect fraction {summary.perfect_fraction:.6f} exceeds the required "
            f"maximum {max_perfect_fraction:.6f}; artifacts were written to {output_dir}"
        )
    return TaskProbeArtifacts(
        records_jsonl=records_jsonl,
        records_csv=records_csv,
        summary_json=summary_json,
        records=tuple(records),
        summary=summary,
    )


def run_configured_task_probe(
    config: VersionedExperimentConfig,
    *,
    backend: ModelBackend,
    samples: Sequence[TaskSample],
) -> TaskProbeArtifacts | None:
    viability = config.task_viability
    if not viability.enabled or not config.metrics.compute_task_metrics:
        return None
    return run_task_probe_with_backend(
        config,
        backend=backend,
        samples=samples,
        output_dir=config.output_dir / "task_probe",
        max_samples=min(viability.probe_samples, len(samples)),
        min_mean_score=viability.min_mean_score,
        max_mean_score=viability.max_mean_score,
        min_nonzero_fraction=viability.min_nonzero_fraction,
        max_perfect_fraction=viability.max_perfect_fraction,
    )


def _validate_threshold(name: str, value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _atomic_write_jsonl(path: Path, records: list[TaskProbeRecord]) -> None:
    with _temporary_text_file(path) as (handle, temporary):
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, records: list[TaskProbeRecord]) -> None:
    with _temporary_text_file(path, newline="") as (handle, temporary):
        fieldnames = list(records[0].to_dict())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    with _temporary_text_file(path) as (handle, temporary):
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _temporary_text_file:
    def __init__(self, destination: Path, *, newline: str | None = None) -> None:
        self.destination = destination
        self.newline = newline
        self.handle: Any | None = None
        self.path: Path | None = None

    def __enter__(self) -> tuple[Any, Path]:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=self.newline,
            dir=self.destination.parent,
            prefix=f".{self.destination.name}.",
            suffix=".tmp",
            delete=False,
        )
        self.handle = handle
        self.path = Path(handle.name)
        return handle, self.path

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle is not None:
            self.handle.close()
        if exc_type is not None and self.path is not None:
            self.path.unlink(missing_ok=True)
