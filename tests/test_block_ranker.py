from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ttt_cache_lab.experiments.block_ranker import (
    FEATURE_NAMES,
    fit_block_ranker,
    load_block_ranker,
    score_block_features,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _condition(target: str, sample: int) -> dict[str, Any]:
    return {
        "sample_id": sample,
        "dataset_sample_id": f"sample-{sample}",
        "update_target": target,
        "block_size": 64,
        "final_adapter_fingerprint": f"{target}-{sample}",
    }


def _feature_row(target: str, sample: int, block: int) -> dict[str, Any]:
    preferred = block + 1 if target == "lora.k_middle" else 4 - block
    return {
        **_condition(target, sample),
        "layer": 14,
        "token_block": block,
        "stale_attention_mass": preferred * 0.1,
        "input_weight_bound": preferred * 0.2,
        "attention_input_bound": preferred * 0.3,
        "predicted_delta_norm": preferred * 0.4,
        "attention_predicted_delta": preferred * 0.5,
        "token_center_fraction": 0.5,
        "token_length_fraction": 0.25,
        "layer_fraction": 0.5,
    }


def _build_input(directory: Path) -> None:
    directory.mkdir(parents=True)
    features: list[dict[str, Any]] = []
    masks: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for target in ("lora.k_middle", "lora.v_middle"):
        preferred = [3, 2] if target == "lora.k_middle" else [0, 1]
        for sample in (0, 1):
            condition = _condition(target, sample)
            features.extend(_feature_row(target, sample, block) for block in range(4))
            records.append(
                {
                    **condition,
                    "selector": "stale",
                    "selected_cells": 0,
                    "logits_kl": 0.10,
                }
            )
            for count in (1, 2):
                budget = count / 4
                for block in preferred[:count]:
                    masks.append(
                        {
                            **condition,
                            "selector": "sparse_delta_oracle",
                            "requested_budget_fraction": budget,
                            "layer": 14,
                            "token_block": block,
                        }
                    )
                best_count = 1 if target == "lora.k_middle" and sample == 0 else 2
                records.append(
                    {
                        **condition,
                        "selector": "sparse_delta_oracle",
                        "requested_budget_fraction": budget,
                        "selected_cells": count,
                        "logits_kl": 0.01 if count == best_count else 0.03,
                    }
                )
    _write_csv(directory / "block_features.csv", features)
    _write_csv(directory / "block_masks.csv", masks)
    _write_csv(directory / "blockwise_records.csv", records)


def test_fit_and_score_target_specific_block_ranker(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output = tmp_path / "ranker.json"
    _build_input(input_dir)

    fit_block_ranker([input_dir], output_path=output, ridge_values=(1e-3, 1e-1))
    ranker = load_block_ranker(output)

    assert ranker["feature_names"] == list(FEATURE_NAMES)
    assert set(ranker["models"]) == {"lora.k_middle", "lora.v_middle"}
    assert ranker["models"]["lora.k_middle"]["default_count"] == 1
    assert ranker["models"]["lora.v_middle"]["default_count"] == 2
    assert ranker["models"]["lora.k_middle"]["training_conditions"] == 2

    k_rows = [_feature_row("lora.k_middle", 9, block) for block in range(4)]
    k_scores, k_count = score_block_features(
        ranker,
        update_target="lora.k_middle",
        feature_rows=k_rows,
    )
    assert int(k_scores.argmax()) == 3
    assert k_count == 1

    v_rows = [_feature_row("lora.v_middle", 9, block) for block in range(4)]
    v_scores, v_count = score_block_features(
        ranker,
        update_target="lora.v_middle",
        feature_rows=v_rows,
    )
    assert int(v_scores.argmax()) == 0
    assert v_count == 2


def test_load_block_ranker_rejects_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "ranker.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "feature_names": ["wrong"],
                "models": {"lora.k_middle": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature schema"):
        load_block_ranker(path)
