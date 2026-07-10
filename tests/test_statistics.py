from __future__ import annotations

import csv
from pathlib import Path

from ttt_cache_lab.experiments.statistics import generate_statistical_report


def test_generate_statistical_report(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    rows: list[dict[str, object]] = []
    for seed in (1, 2):
        for sample_id in (0, 1):
            common = {
                "experiment_id": "e4",
                "sample_id": sample_id,
                "dataset_sample_id": f"sample-{sample_id}",
                "seed": seed,
                "task_name": "qa",
                "task_family": "real_qa",
                "benchmark_name": "benchmark",
                "evaluation_partition": "test",
                "dataset_split": "test",
                "dataset_category": "qa",
                "model_name": "model-7b",
                "model_num_layers": 32,
                "model_hidden_size": 4096,
                "context_length": 8192,
                "update_target": "lora.k",
                "adapter_id": f"adapter-{sample_id}",
                "adapter_version": 1,
                "cached_version": 0,
                "version_gap": 1,
                "lora_rank": 8,
                "configured_update_norm": 0.001,
                "update_mode": "lora_train",
                "norm_control": "target_l2",
                "top1_agreement": 1.0,
                "relative_error": 0.0,
                "throughput_tokens_per_s": 10.0,
                "total_cache_bytes": 100,
                "flops_fraction": 1.0,
                "cache_hit": False,
                "false_safe": False,
            }
            rows.append(
                {
                    **common,
                    "cache_strategy": "full_recompute",
                    "task_score": 1.0,
                    "logits_kl": 0.0,
                    "end_to_end_latency": 10.0 + sample_id,
                }
            )
            rows.append(
                {
                    **common,
                    "cache_strategy": "adaptive",
                    "task_score": 0.99,
                    "logits_kl": 0.01,
                    "end_to_end_latency": 5.0 + sample_id,
                    "flops_fraction": 0.5,
                    "cache_hit": True,
                }
            )
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    outputs = generate_statistical_report(
        source,
        tmp_path / "statistics",
        bootstrap_resamples=200,
        confidence_level=0.95,
    )
    assert len(outputs) == 5
    paired_path = tmp_path / "statistics" / "paired_comparisons.csv"
    with paired_path.open("r", encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    speedup = next(row for row in paired if row["metric"] == "speedup_vs_reference")
    assert float(speedup["mean"]) > 1.0
    safety_path = tmp_path / "statistics" / "safety_intervals.csv"
    with safety_path.open("r", encoding="utf-8", newline="") as handle:
        safety = list(csv.DictReader(handle))
    adaptive = next(row for row in safety if row["cache_strategy"] == "adaptive")
    assert float(adaptive["false_safe_rate"]) == 0.0
    assert float(adaptive["wilson_high"]) > 0.0
    assert "cluster bootstrap" in (tmp_path / "statistics" / "statistical_summary.md").read_text(
        encoding="utf-8"
    )
