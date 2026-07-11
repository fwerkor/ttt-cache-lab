from __future__ import annotations

import csv
from pathlib import Path

from ttt_cache_lab.experiments.propagation_analysis import generate_propagation_analysis


def test_propagation_analysis_detects_strong_decay(tmp_path: Path) -> None:
    input_csv = tmp_path / "propagation.csv"
    fieldnames = [
        "sample_id",
        "model_name",
        "context_length",
        "synthetic_difficulty",
        "task_name",
        "update_target",
        "target_layer",
        "adapter_version",
        "cached_version",
        "version_gap",
        "configured_update_norm",
        "layer_id",
        "hidden_relative_error",
        "hidden_cosine_distance",
        "hidden_norm_ratio",
        "key_relative_error",
        "key_cosine_distance",
        "value_relative_error",
        "value_cosine_distance",
    ]
    rows = []
    for sample_id in (0, 1):
        for layer_id, error in enumerate((1.0, 0.5, 0.05)):
            rows.append(
                {
                    "sample_id": sample_id,
                    "model_name": "toy",
                    "context_length": 128,
                    "synthetic_difficulty": "easy",
                    "task_name": "passkey",
                    "update_target": "lora.k:0",
                    "target_layer": 0,
                    "adapter_version": 4,
                    "cached_version": 0,
                    "version_gap": 4,
                    "configured_update_norm": 0.01,
                    "layer_id": layer_id,
                    "hidden_relative_error": error,
                    "hidden_cosine_distance": error / 10,
                    "hidden_norm_ratio": 1.0,
                    "key_relative_error": error,
                    "key_cosine_distance": error / 10,
                    "value_relative_error": error,
                    "value_cosine_distance": error / 10,
                }
            )
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    layers_path, profiles_path = generate_propagation_analysis(
        input_csv,
        tmp_path / "analysis",
        recovery_ratio=0.1,
    )
    assert layers_path.exists()
    assert profiles_path.exists()
    with profiles_path.open("r", encoding="utf-8", newline="") as handle:
        profiles = list(csv.DictReader(handle))
    assert len(profiles) == 1
    assert profiles[0]["profile"] == "strong_decay"
    assert profiles[0]["recovery_layer"] == "2"
    assert profiles[0]["recovered_before_end"] == "True"
    assert float(profiles[0]["tail_to_peak_ratio"]) == 0.05
