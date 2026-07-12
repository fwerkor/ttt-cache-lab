from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ttt_cache_lab.experiments.block_ranker import (
    FEATURE_NAMES,
    _baseline_reference_router_feature_vector,
    _confidence_probe_policies,
    _one_probe_policies,
    _reference_candidate_pool_policies,
    _reference_candidate_router_policies,
    fit_block_ranker,
    load_block_ranker,
    route_committed_candidate,
    route_reference_candidate,
    route_zero_probe_recompute,
    score_block_features,
)


def test_zero_probe_recompute_router_uses_only_frozen_risk_threshold() -> None:
    ranker = {
        "models": {
            "lora.k_middle": {
                "zero_probe_recompute_policy": {
                    "risk_feature": "router_baseline_stale_kl",
                    "risk_threshold": 0.01,
                    "trigger_quantile": 0.875,
                    "runtime_forward_count": 0,
                    "runtime_uses_full_reference": False,
                }
            }
        }
    }

    trigger, score, policy = route_zero_probe_recompute(
        ranker,
        update_target="lora.k_middle",
        condition={"router_baseline_stale_kl": 0.0101},
    )
    assert trigger is True
    assert score == pytest.approx(0.0101)
    assert policy["runtime_forward_count"] == 0
    assert policy["runtime_uses_full_reference"] is False

    trigger, score, _ = route_zero_probe_recompute(
        ranker,
        update_target="lora.k_middle",
        condition={"router_baseline_stale_kl": 0.0099},
    )
    assert trigger is False
    assert score == pytest.approx(0.0099)


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


def test_one_probe_policy_prefers_safe_candidate_and_conservative_margin() -> None:
    rows: list[dict[str, str]] = []
    for sample, nll_gain in ((0, 0.30), (1, 0.25)):
        condition = {
            key: str(value)
            for key, value in _condition("lora.k_middle", sample).items()
        }
        rows.append(
            {
                **condition,
                "selector": "stale",
                "selected_cells": "0",
                "logits_kl": "0.10",
                "reference_token_nll": "1.0",
                "cache_maintenance_latency": "0.0",
                "decode_latency": "0.02",
            }
        )
        rows.append(
            {
                **condition,
                "selector": "sparse_input_bound",
                "selected_cells": "2",
                "logits_kl": "0.04",
                "reference_token_nll": str(1.0 - nll_gain),
                "cache_maintenance_latency": "0.01",
                "decode_latency": "0.02",
            }
        )

    policy = _one_probe_policies(rows)["lora.k_middle"]

    assert policy["candidate_selector"] == "sparse_input_bound"
    assert policy["candidate_count"] == 2
    assert policy["reference_nll_margin"] == 0.2
    assert policy["calibration_harmful"] == 0
    assert policy["calibration_accepted"] == 2




def test_reference_candidate_pool_prefers_safe_high_recovery_candidate() -> None:
    rows: list[dict[str, str]] = []
    for sample, candidate_kl in ((0, 0.04), (1, 0.05)):
        condition = {
            key: str(value)
            for key, value in _condition("lora.k_middle", sample).items()
        }
        rows.extend(
            [
                {
                    **condition,
                    "selector": "stale",
                    "selected_cells": "0",
                    "logits_kl": "0.10",
                    "reference_token_nll": "1.0",
                },
                {
                    **condition,
                    "selector": "sparse_input_bound",
                    "selected_cells": "2",
                    "logits_kl": str(candidate_kl),
                    "reference_token_nll": "0.7",
                },
                {
                    **condition,
                    "selector": "sparse_attention_mass",
                    "selected_cells": "1",
                    "logits_kl": "0.12",
                    "reference_token_nll": "0.8",
                },
            ]
        )

    policy = _reference_candidate_pool_policies(
        rows,
        pool_size=1,
        min_stale_kl=0.01,
    )["lora.k_middle"]

    assert policy["candidates"] == [
        {"candidate_selector": "sparse_input_bound", "candidate_count": 2}
    ]
    assert policy["calibration_harmful"] == 0
    assert policy["calibration_beneficial"] == 2
    assert policy["calibration_weighted_recovery"] == pytest.approx(0.55)



def test_reference_candidate_router_learns_one_probe_choice() -> None:
    feature_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, str]] = []
    for sample in range(4):
        target = "lora.k_middle"
        feature_rows.extend(_feature_row(target, sample, block) for block in range(4))
        condition = {
            key: str(value)
            for key, value in _condition(target, sample).items()
        }
        record_rows.extend(
            [
                {
                    **condition,
                    "selector": "stale",
                    "selected_cells": "0",
                    "logits_kl": "0.10",
                    "reference_token_nll": "1.0",
                },
                {
                    **condition,
                    "selector": "sparse_input_bound",
                    "selected_cells": "2",
                    "logits_kl": "0.03",
                    "reference_token_nll": "0.6",
                },
                {
                    **condition,
                    "selector": "sparse_attention_mass",
                    "selected_cells": "1",
                    "logits_kl": "0.12",
                    "reference_token_nll": "0.9",
                },
            ]
        )
    pool = {
        "lora.k_middle": {
            "candidates": [
                {
                    "candidate_selector": "sparse_input_bound",
                    "candidate_count": 2,
                },
                {
                    "candidate_selector": "sparse_attention_mass",
                    "candidate_count": 1,
                },
            ]
        }
    }
    policies = _reference_candidate_router_policies(
        feature_rows,
        record_rows,
        pool,
        ridge_values=(1.0, 100.0),
        min_stale_kl=0.01,
    )
    policy = policies["lora.k_middle"]
    ranker = {
        "models": {
            "lora.k_middle": {
                "reference_candidate_router_policy": policy,
            }
        }
    }

    selected, scores = route_reference_candidate(
        ranker,
        update_target="lora.k_middle",
        feature_rows=[_feature_row("lora.k_middle", 9, block) for block in range(4)],
    )

    assert selected == {
        "candidate_selector": "sparse_input_bound",
        "candidate_count": 2,
    }
    assert scores.shape == (2,)
    assert policy["material_harmful"] == 0
    assert policy["material_weighted_recovery"] == pytest.approx(0.7)



def test_baseline_reference_router_uses_non_attention_features() -> None:
    feature_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, str]] = []
    for sample in range(4):
        target = "lora.v_middle"
        rows = [_feature_row(target, sample, block) for block in range(4)]
        for row in rows:
            row["stale_attention_mass"] = str(1000.0 * (sample + 1))
            row["attention_input_bound"] = str(2000.0 * (sample + 1))
            row["attention_predicted_delta"] = str(3000.0 * (sample + 1))
        feature_rows.extend(rows)
        condition = {
            key: str(value)
            for key, value in _condition(target, sample).items()
        }
        record_rows.extend(
            [
                {
                    **condition,
                    "selector": "stale",
                    "selected_cells": "0",
                    "logits_kl": "0.10",
                    "reference_token_nll": "1.0",
                },
                {
                    **condition,
                    "selector": "sparse_predicted_delta_norm",
                    "selected_cells": "2",
                    "logits_kl": "0.02",
                    "reference_token_nll": "0.5",
                },
                {
                    **condition,
                    "selector": "sparse_input_bound",
                    "selected_cells": "1",
                    "logits_kl": "0.08",
                    "reference_token_nll": "0.8",
                },
            ]
        )
    pool = {
        "lora.v_middle": {
            "candidates": [
                {
                    "candidate_selector": "sparse_predicted_delta_norm",
                    "candidate_count": 2,
                },
                {
                    "candidate_selector": "sparse_input_bound",
                    "candidate_count": 1,
                },
            ]
        }
    }
    policies = _reference_candidate_router_policies(
        feature_rows,
        record_rows,
        pool,
        ridge_values=(1.0,),
        min_stale_kl=0.01,
        feature_mode="baseline_only",
    )
    policy = policies["lora.v_middle"]
    ranker = {
        "models": {
            "lora.v_middle": {
                "baseline_reference_candidate_router_policy": policy,
            }
        }
    }
    query = [_feature_row("lora.v_middle", 9, block) for block in range(4)]
    selected, scores = route_reference_candidate(
        ranker,
        update_target="lora.v_middle",
        feature_rows=query,
        policy_name="baseline_reference_candidate_router_policy",
    )

    assert policy["feature_mode"] == "baseline_only"
    assert selected == {
        "candidate_selector": "sparse_predicted_delta_norm",
        "candidate_count": 2,
    }
    assert scores.shape == (2,)



def test_committed_router_selects_positive_lower_bound_and_can_abstain() -> None:
    rows = [_feature_row("lora.k_middle", 9, block) for block in range(4)]
    dimension = len(
        _baseline_reference_router_feature_vector(rows, expected_cells=4)
    )
    candidates = [
        {"candidate_selector": "sparse_input_bound", "candidate_count": 1},
        {
            "candidate_selector": "sparse_predicted_delta_norm",
            "candidate_count": 2,
        },
    ]

    def ranker(errors: list[float]) -> dict[str, Any]:
        return {
            "models": {
                "lora.k_middle": {
                    "baseline_committed_candidate_router_policy": {
                        "candidates": candidates,
                        "expected_direct_cells": 4,
                        "mean": [0.0] * dimension,
                        "scale": [1.0] * dimension,
                        "intercept": [0.20, 0.10],
                        "weights": [[0.0, 0.0] for _ in range(dimension)],
                        "overprediction_error": errors,
                        "minimum_lower_bound": 0.0,
                    }
                }
            }
        }

    selected, predicted, lower = route_committed_candidate(
        ranker([0.05, 0.20]),
        update_target="lora.k_middle",
        feature_rows=rows,
    )
    abstained, _, rejected_lower = route_committed_candidate(
        ranker([0.30, 0.20]),
        update_target="lora.k_middle",
        feature_rows=rows,
    )

    assert selected == candidates[0]
    assert predicted.tolist() == pytest.approx([0.20, 0.10])
    assert lower.tolist() == pytest.approx([0.15, -0.10])
    assert abstained is None
    assert max(rejected_lower) <= 0.0

    guarded = ranker([0.05, 0.20])
    guarded["models"]["lora.k_middle"][
        "baseline_committed_candidate_router_policy"
    ]["runtime_guard"] = {
        "rules": [
            {
                "candidate_selector": "sparse_input_bound",
                "entropy_min": 0.10,
                "entropy_max": 0.20,
            }
        ]
    }
    trusted, _, _ = route_committed_candidate(
        guarded,
        update_target="lora.k_middle",
        feature_rows=rows,
        stale_output_entropy=0.15,
    )
    out_of_band, _, _ = route_committed_candidate(
        guarded,
        update_target="lora.k_middle",
        feature_rows=rows,
        stale_output_entropy=0.05,
    )
    assert trusted == candidates[0]
    assert out_of_band is None

def test_confidence_probe_policy_rejects_low_confidence_harm() -> None:
    rows: list[dict[str, str]] = []
    for sample, confidence_gain, candidate_kl in (
        (0, 0.03, 0.04),
        (1, 0.0005, 0.12),
    ):
        condition = {
            key: str(value)
            for key, value in _condition("lora.k_middle", sample).items()
        }
        rows.append(
            {
                **condition,
                "selector": "stale",
                "selected_cells": "0",
                "logits_kl": "0.10",
                "output_max_probability": "0.50",
                "cache_maintenance_latency": "0.0",
                "decode_latency": "0.02",
            }
        )
        rows.append(
            {
                **condition,
                "selector": "sparse_input_bound",
                "selected_cells": "2",
                "logits_kl": str(candidate_kl),
                "output_max_probability": str(0.50 + confidence_gain),
                "cache_maintenance_latency": "0.01",
                "decode_latency": "0.02",
            }
        )

    policy = _confidence_probe_policies(rows)["lora.k_middle"]

    assert policy["candidate_selector"] == "sparse_input_bound"
    assert policy["candidate_count"] == 2
    assert policy["confidence_margin"] == 0.02
    assert policy["calibration_harmful"] == 0
    assert policy["calibration_accepted"] == 1


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
