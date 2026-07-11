from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row.get(field, 0.0)) for row in rows]
    return fmean(values) if values else 0.0


def _load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/records.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["records_path"] = str(path)
                    records.append(row)
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if int(row.get("adapter_version", 0)) <= 0:
            continue
        key = (
            row.get("model_name", ""),
            row.get("update_target", ""),
            float(row.get("configured_update_norm", 0.0)),
            int(row.get("adapter_version", 0)),
            int(row.get("context_length", 0)),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        model, target, norm, version, context = key
        stale = [row for row in rows if row.get("cache_strategy") == "stale_reuse"]
        delta = [row for row in rows if row.get("cache_strategy") == "delta_correction"]
        if not stale or not delta:
            continue
        stale_by_sample = {int(row["sample_id"]): row for row in stale}
        delta_by_sample = {int(row["sample_id"]): row for row in delta}
        common = sorted(stale_by_sample.keys() & delta_by_sample.keys())
        paired_improvements = [
            float(stale_by_sample[sample]["logits_kl"])
            - float(delta_by_sample[sample]["logits_kl"])
            for sample in common
        ]
        paired_relative_improvements = []
        for sample in common:
            stale_kl = float(stale_by_sample[sample]["logits_kl"])
            delta_kl = float(delta_by_sample[sample]["logits_kl"])
            paired_relative_improvements.append(
                (stale_kl - delta_kl) / stale_kl if stale_kl > 0.0 else 0.0
            )
        output.append(
            {
                "model_name": model,
                "update_target": target,
                "configured_update_norm": norm,
                "adapter_version": version,
                "context_length": context,
                "paired_samples": len(common),
                "stale_logits_kl_mean": _mean(stale, "logits_kl"),
                "delta_logits_kl_mean": _mean(delta, "logits_kl"),
                "delta_kl_improvement_mean": fmean(paired_improvements) if paired_improvements else 0.0,
                "delta_kl_relative_improvement_mean": (
                    fmean(paired_relative_improvements) if paired_relative_improvements else 0.0
                ),
                "stale_relative_error_mean": _mean(stale, "relative_error"),
                "delta_relative_error_mean": _mean(delta, "relative_error"),
                "delta_raw_l2_mean": _mean(delta, "delta_raw_l2"),
                "delta_stored_l2_mean": _mean(delta, "delta_stored_l2"),
                "delta_raw_max_abs_mean": _mean(delta, "delta_raw_max_abs"),
                "delta_stored_max_abs_mean": _mean(delta, "delta_stored_max_abs"),
                "delta_changed_fraction_mean": _mean(delta, "delta_changed_fraction"),
                "delta_quantization_retention_mean": _mean(delta, "delta_quantization_retention"),
                "stale_maintenance_latency_mean": _mean(stale, "cache_maintenance_latency"),
                "delta_maintenance_latency_mean": _mean(delta, "cache_maintenance_latency"),
                "delta_maintenance_overhead_mean": (
                    _mean(delta, "cache_maintenance_latency")
                    - _mean(stale, "cache_maintenance_latency")
                ),
                "delta_better_rate": (
                    sum(value > 0.0 for value in paired_improvements) / len(paired_improvements)
                    if paired_improvements
                    else 0.0
                ),
                "delta_equal_rate": (
                    sum(value == 0.0 for value in paired_improvements) / len(paired_improvements)
                    if paired_improvements
                    else 0.0
                ),
                "records_path": delta[0].get("records_path", ""),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired stale/delta quantization sweeps.")
    parser.add_argument("--root", type=Path, default=Path("runs/delta_quantization"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/delta_quantization/summary.csv"),
    )
    args = parser.parse_args()
    rows = summarize(_load_records(args.root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No paired stale/delta records found below {args.root}")
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
