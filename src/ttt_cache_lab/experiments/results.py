from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    sample_id: int
    update_target: str
    cache_strategy: str
    action: str
    cache_state: str
    first_invalid_layer: int | None
    task_score: float
    logits_kl: float
    top1_agreement: float
    relative_error: float
    latency_units: float
    reason: str
    experiment_id: str = "single_step"
    adapter_id: str = "adapter"
    adapter_version: int = 1
    cached_version: int = 0
    version_gap: int = 1
    update_step: int = 1
    accumulated_update_norm: float = 0.0
    accumulated_raw_update_norm: float = 0.0
    update_norm_since_cache: float = 0.0
    raw_update_norm_since_cache: float = 0.0
    update_scale: float = 0.0
    lora_rank: int = 0
    update_mode: str = "random"
    norm_control: str = ""
    hidden_relative_error: float = 0.0
    cache_bytes: int = 0
    physical_cache_bytes: int = 0
    memory_allocated: int = 0
    peak_memory_allocated: int = 0
    adaptation_latency: float = 0.0
    cache_maintenance_latency: float = 0.0
    decode_latency: float = 0.0
    end_to_end_latency: float = 0.0
    throughput_tokens_per_s: float = 0.0
    recompute_fraction: float = 0.0
    cache_hit: bool = False
    refresh_count: int = 0
    rejected_reuse: bool = False
    false_safe: bool = False
    strategy_mode: str = ""
    strategy_available: bool = True
    strategy_fallback: str = ""
    baseline_fidelity: str = ""
    baseline_source: str = ""
    baseline_reference: str = ""
    cache_block_count: int = 0
    cache_entry_count: int = 0
    total_cache_bytes: int = 0
    evicted_cache_entries: int = 0
    context_length: int = 0
    model_name: str = ""
    model_num_layers: int = 0
    model_hidden_size: int = 0
    configured_update_norm: float = 0.0
    baseline_task_score: float = 0.0
    full_task_score: float = 0.0
    adaptation_gain_vs_base: float = 0.0
    attention_shift: float | None = None
    attention_metric_available: bool = False
    strategy_flops: float = 0.0
    full_recompute_flops: float = 0.0
    flops_fraction: float = 0.0
    planner_source: str = ""
    failure_map_path: str = ""
    failure_map_sha256: str = ""
    cache_manager_scope: str = ""
    seed: int = 0
    task_name: str = ""
    task_family: str = ""
    benchmark_name: str = ""
    evaluation_partition: str = "all"
    dataset_split: str = ""
    dataset_sample_id: str = ""
    dataset_category: str = ""
    selection_seed: int = 0
    backend_name: str = ""
    torch_dtype: str = ""
    attention_implementation: str = ""
    git_commit: str = ""
    run_config_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> tuple[Any, ...]:
        return (
            self.experiment_id,
            self.dataset_sample_id or self.sample_id,
            self.evaluation_partition,
            self.update_target,
            self.cache_strategy,
            self.adapter_id,
            self.adapter_version,
            self.cached_version,
            self.update_step,
            self.lora_rank,
            self.configured_update_norm,
            self.context_length,
            self.model_name,
            self.seed,
        )


@dataclass(frozen=True)
class ExperimentArtifacts:
    jsonl_path: Path
    csv_path: Path
    records: list[ExperimentRecord]
    metadata_path: Path | None = None


def write_records(
    records: list[ExperimentRecord],
    output_dir: Path,
    *,
    merge_existing: bool = False,
    metadata_path: Path | None = None,
) -> ExperimentArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "records.jsonl"
    csv_path = output_dir / "summary.csv"

    merged: dict[tuple[Any, ...], ExperimentRecord] = {}
    if merge_existing and jsonl_path.exists():
        for record in read_records(jsonl_path):
            merged[record.identity()] = record
    for record in records:
        merged[record.identity()] = record
    final_records = list(merged.values())

    _atomic_write_jsonl(jsonl_path, final_records)
    _atomic_write_csv(csv_path, final_records)
    return ExperimentArtifacts(
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        records=final_records,
        metadata_path=metadata_path,
    )


def merge_record_files(
    inputs: list[Path],
    output_dir: Path,
) -> ExperimentArtifacts:
    records: list[ExperimentRecord] = []
    for path in inputs:
        records.extend(read_records(path))
    return write_records(records, output_dir, merge_existing=False)


def read_records(path: Path) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Record {line_number} in {path} is not a JSON object")
            records.append(ExperimentRecord(**payload))
    return records


def _atomic_write_jsonl(path: Path, records: list[ExperimentRecord]) -> None:
    with _temporary_text_file(path) as (handle, temporary):
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, records: list[ExperimentRecord]) -> None:
    with _temporary_text_file(path, newline="") as (handle, temporary):
        fieldnames = list(records[0].to_dict().keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
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
