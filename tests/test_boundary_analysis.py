from __future__ import annotations

import csv
from pathlib import Path

from ttt_cache_lab.experiments.boundary_analysis import generate_boundary_analysis


def test_boundary_analysis_ranks_rejoin_windows_and_cross_validates(tmp_path: Path) -> None:
    boundary_path = tmp_path / "boundary.csv"
    summary_path = tmp_path / "summary.csv"
    boundary_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    curves = {
        0: {1: 0.30, 2: 0.10, 4: 0.20},
        1: {1: 0.10, 2: 0.30, 4: 0.20},
        2: {1: 0.20, 2: 0.30, 4: 0.10},
    }
    for sample_id, curve in curves.items():
        common = {
            "sample_id": sample_id,
            "dataset_sample_id": f"sample-{sample_id}",
            "task_name": "passkey",
            "task_family": "retrieval",
            "model_name": "toy",
            "model_num_layers": 6,
            "update_target": "lora.k:1",
            "target_layer": 1,
            "adapter_version": 1,
            "cached_version": 0,
            "version_gap": 1,
            "context_length": 64,
            "synthetic_difficulty": "easy",
            "seed": 7,
            "configured_update_norm": 0.1,
            "accumulated_update_norm": 0.1,
            "accumulated_raw_update_norm": 0.1,
        }
        summary_rows.append(
            {
                "sample_id": sample_id,
                "dataset_sample_id": f"sample-{sample_id}",
                "task_name": "passkey",
                "update_target": "lora.k:1",
                "adapter_version": 1,
                "cached_version": 0,
                "context_length": 64,
                "model_name": "toy",
                "seed": 7,
                "cache_strategy": "stale_reuse",
                "logits_kl": 0.25,
                "top1_agreement": 1.0,
                "full_task_score": 1.0,
                "task_score": 1.0,
            }
        )
        for window, logits_kl in curve.items():
            boundary_rows.append(
                {
                    **common,
                    "cache_strategy": f"windowed_recompute_{window}",
                    "window_size": window,
                    "boundary_layer": 1 + window,
                    "has_stale_rejoin": True,
                    "logits_kl": logits_kl,
                    "top1_agreement": 1.0,
                    "task_drop_vs_full": 0.0,
                    "recompute_fraction": window / 6,
                    "attention_kl": logits_kl * 1.2,
                    "attention_js": logits_kl,
                    "attention_l1": logits_kl * 2.0,
                    "attention_topk_overlap": 1.0 - logits_kl,
                    "attention_output_relative_error": logits_kl,
                    "attention_output_cosine_distance": logits_kl / 2,
                    "boundary_input_hidden_relative_error": logits_kl * 1.1,
                    "boundary_next_hidden_relative_error": logits_kl * 0.9,
                    "key_relative_error": logits_kl * 1.3,
                    "value_relative_error": logits_kl * 1.4,
                    "attention_weighted_key_relative_error": logits_kl * 0.8,
                    "attention_weighted_value_relative_error": logits_kl * 0.7,
                    "metric_available": True,
                }
            )

    _write(boundary_path, boundary_rows)
    _write(summary_path, summary_rows)
    artifacts = generate_boundary_analysis(
        boundary_path,
        summary_path,
        tmp_path / "analysis",
        ridge=1e-6,
    )
    assert artifacts.enriched_rows_path.exists()
    assert artifacts.metric_evaluation_path.exists()
    assert artifacts.group_selections_path.exists()
    assert artifacts.predictor_summary_path.exists()

    metrics = _read(artifacts.metric_evaluation_path)
    attention_output = next(
        row for row in metrics if row["selector"] == "attention_output_relative_error"
    )
    assert float(attention_output["oracle_window_hit_rate"]) == 1.0
    assert float(attention_output["beneficial_selection_rate"]) == 1.0
    assert float(attention_output["mean_kl_regret"]) == 0.0
    assert float(attention_output["global_spearman_vs_kl"]) == 1.0

    shortest = next(row for row in metrics if row["selector"] == "shortest_window")
    assert float(shortest["oracle_window_hit_rate"]) < 1.0
    assert float(shortest["mean_kl_regret"]) > 0.0

    predictor = _read(artifacts.predictor_summary_path)[0]
    assert predictor["status"] == "ok"
    assert int(predictor["held_out_sample_count"]) == 3
    assert float(predictor["oracle_window_hit_rate"]) == 1.0
    assert float(predictor["mean_kl_regret"]) == 0.0


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
