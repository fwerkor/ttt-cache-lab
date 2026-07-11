from __future__ import annotations

import csv
from pathlib import Path

from ttt_cache_lab.experiments.window_analysis import WindowThresholds, generate_window_analysis


def test_window_analysis_selects_smallest_safe_window(tmp_path: Path) -> None:
    input_csv = tmp_path / "records.csv"
    fieldnames = [
        "sample_id",
        "experiment_id",
        "model_name",
        "model_num_layers",
        "context_length",
        "synthetic_difficulty",
        "prompt_format",
        "task_name",
        "task_family",
        "update_target",
        "adapter_id",
        "adapter_version",
        "cached_version",
        "version_gap",
        "lora_rank",
        "configured_update_norm",
        "update_mode",
        "norm_control",
        "cache_strategy",
        "recompute_window_size",
        "task_score",
        "logits_kl",
        "top1_agreement",
        "false_safe",
        "recompute_fraction",
        "flops_fraction",
        "end_to_end_latency",
    ]
    rows: list[dict[str, object]] = []
    for sample_id in (0, 1):
        common = {
            "sample_id": sample_id,
            "experiment_id": "w1",
            "model_name": "toy",
            "model_num_layers": 4,
            "context_length": 128,
            "synthetic_difficulty": "easy",
            "prompt_format": "plain",
            "task_name": "passkey",
            "task_family": "retrieval",
            "update_target": "lora.k:1",
            "adapter_id": f"sample-{sample_id}:lora.k:1",
            "adapter_version": 1,
            "cached_version": 0,
            "version_gap": 1,
            "lora_rank": 4,
            "configured_update_norm": 0.01,
            "update_mode": "random",
            "norm_control": "target_l2",
        }
        rows.append(
            {
                **common,
                "cache_strategy": "full_recompute",
                "recompute_window_size": 0,
                "task_score": 1.0,
                "logits_kl": 0.0,
                "top1_agreement": 1.0,
                "false_safe": False,
                "recompute_fraction": 1.0,
                "flops_fraction": 1.0,
                "end_to_end_latency": 10.0,
            }
        )
        rows.append(
            {
                **common,
                "cache_strategy": "stale_reuse",
                "recompute_window_size": 0,
                "task_score": 0.5,
                "logits_kl": 0.1,
                "top1_agreement": 0.0,
                "false_safe": True,
                "recompute_fraction": 0.0,
                "flops_fraction": 0.0,
                "end_to_end_latency": 1.0,
            }
        )
        rows.append(
            {
                **common,
                "cache_strategy": "windowed_recompute_1",
                "recompute_window_size": 1,
                "task_score": 0.5,
                "logits_kl": 0.2,
                "top1_agreement": 0.0,
                "false_safe": True,
                "recompute_fraction": 0.25,
                "flops_fraction": 0.25,
                "end_to_end_latency": 3.0,
            }
        )
        rows.append(
            {
                **common,
                "cache_strategy": "windowed_recompute_2",
                "recompute_window_size": 2,
                "task_score": 1.0,
                "logits_kl": 0.01,
                "top1_agreement": 1.0,
                "false_safe": False,
                "recompute_fraction": 0.5,
                "flops_fraction": 0.5,
                "end_to_end_latency": 5.0,
            }
        )
        rows.append(
            {
                **common,
                "cache_strategy": "windowed_recompute_3",
                "recompute_window_size": 3,
                "task_score": 1.0,
                "logits_kl": 0.02,
                "top1_agreement": 1.0,
                "false_safe": False,
                "recompute_fraction": 0.75,
                "flops_fraction": 0.75,
                "end_to_end_latency": 7.0,
            }
        )
    zero_gap = {
        **rows[0],
        "adapter_version": 0,
        "version_gap": 0,
        "cache_strategy": "full_recompute",
        "recompute_window_size": 0,
        "task_score": 1.0,
        "logits_kl": 0.0,
        "top1_agreement": 1.0,
        "false_safe": False,
        "recompute_fraction": 1.0,
        "flops_fraction": 1.0,
        "end_to_end_latency": 10.0,
    }
    rows.extend(
        [
            zero_gap,
            {
                **zero_gap,
                "cache_strategy": "windowed_recompute",
                "recompute_fraction": 0.0,
                "flops_fraction": 0.0,
                "end_to_end_latency": 1.0,
            },
        ]
    )
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cells_path, minima_path = generate_window_analysis(
        input_csv,
        tmp_path / "analysis",
        thresholds=WindowThresholds(min_safe_rate=1.0),
    )
    assert cells_path.exists()
    assert minima_path.exists()
    with minima_path.open("r", encoding="utf-8", newline="") as handle:
        minima = list(csv.DictReader(handle))
    assert len(minima) == 1
    assert minima[0]["minimum_safe_window"] == "2"
    assert minima[0]["safe_window_found"] == "True"
    assert minima[0]["minimum_beneficial_window"] == "2"
    assert minima[0]["beneficial_window_found"] == "True"
    assert minima[0]["best_kl_window"] == "2"
    assert minima[0]["kl_nonincreasing_with_window"] == "False"
    assert minima[0]["kl_monotonicity_violations"] == "1"
    assert float(minima[0]["safe_rate"]) == 1.0
    assert float(minima[0]["beneficial_vs_stale_rate"]) == 1.0
