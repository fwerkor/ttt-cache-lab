from __future__ import annotations

import csv
import json
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
    adapter_version: int = 1
    cached_version: int = 0
    version_gap: int = 1
    update_step: int = 1
    accumulated_update_norm: float = 0.0
    lora_rank: int = 0
    update_mode: str = "random"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentArtifacts:
    jsonl_path: Path
    csv_path: Path
    records: list[ExperimentRecord]


def write_records(records: list[ExperimentRecord], output_dir: Path) -> ExperimentArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "records.jsonl"
    csv_path = output_dir / "summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0].to_dict().keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())

    return ExperimentArtifacts(jsonl_path=jsonl_path, csv_path=csv_path, records=records)
