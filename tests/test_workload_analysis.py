from __future__ import annotations

import csv
from pathlib import Path

from ttt_cache_lab.experiments.study_analysis import generate_study_analysis


def test_e8_analysis_reports_tail_latency_and_evictions(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    rows = [
        {
            "experiment_id": "e8_cache_pressure",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "model_num_layers": "28",
            "model_hidden_size": "3584",
            "model_parameter_count": "7000000000",
            "context_length": "16384",
            "task_name": "variable_tracking",
            "task_family": "state_tracking",
            "benchmark_name": "controlled-request-trace",
            "evaluation_partition": "test",
            "dataset_split": "generated",
            "dataset_category": "state_tracking",
            "lora_rank": "8",
            "configured_update_norm": "0.001",
            "update_mode": "lora_train",
            "norm_control": "target_l2",
            "seed": "7",
            "update_target": "lora.k",
            "cache_strategy": "adaptive",
            "task_score": "1.0",
            "latency_p50": str(latency),
            "end_to_end_latency": str(latency),
            "throughput_tokens_per_s": "20",
            "cache_hit": "True",
            "false_safe": "False",
            "cache_entry_count": str(entries),
            "total_cache_bytes": str(entries * 100),
            "evicted_cache_entries": str(max(0, entries - 2)),
            "refresh_count": "1",
        }
        for latency, entries in ((1.0, 1), (2.0, 2), (10.0, 3))
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs = generate_study_analysis(source, tmp_path / "analysis")
    assert tmp_path / "analysis" / "e8_cache_pressure.csv" in outputs
    with (tmp_path / "analysis" / "e8_cache_pressure.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        summary = next(csv.DictReader(handle))
    assert float(summary["latency_p95"]) > float(summary["latency_p50"])
    assert float(summary["evicted_cache_entries"]) == 1.0
    assert (tmp_path / "analysis" / "e8_latency_cache_pressure.svg").exists()
