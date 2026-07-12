from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ttt_cache_lab.cache.dynamic_probe_model import (
    load_dynamic_probe_model,
    prompt_anchor_feature_values,
    score_prompt_anchor_point,
)


def _point_trace() -> dict[str, object]:
    return {
        "count": 2,
        "selected_cells": [[3, 1], [4, 2]],
        "selected_token_blocks": [1, 2],
        "objectives": [
            {
                "name": "prompt_anchor_b1_nll_1",
                "probe_length": 1,
                "normalized_score": 0.98,
            },
            {
                "name": "prompt_anchor_b1_nll_2",
                "probe_length": 2,
                "normalized_score": 0.99,
            },
            {
                "name": "prompt_anchor_b2_nll_1",
                "probe_length": 1,
                "normalized_score": 0.97,
            },
        ],
        "evaluation_only_logits_kl": 999.0,
        "evaluation_only_kl_gain_vs_stale": -999.0,
    }


def test_prompt_anchor_features_ignore_evaluation_only_kl() -> None:
    first = _point_trace()
    second = _point_trace()
    second["evaluation_only_logits_kl"] = 0.0
    second["evaluation_only_kl_gain_vs_stale"] = 1000.0

    first_features = prompt_anchor_feature_values(
        first,
        risk_score=0.7,
        control_score=0.75,
        activated_risk=0.2,
        budget_cap=4,
        token_block_count=8,
    )
    second_features = prompt_anchor_feature_values(
        second,
        risk_score=0.7,
        control_score=0.75,
        activated_risk=0.2,
        budget_cap=4,
        token_block_count=8,
    )

    assert first_features == second_features
    assert first_features["worst"] == pytest.approx(0.99)
    assert first_features["anchor_count"] == 2.0
    assert first_features["block_mean"] == pytest.approx(1.5 / 7.0)


def test_loaded_dynamic_probe_model_scores_only_declared_features(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.json"
    payload = {
        "format_version": 1,
        "probe_source": "prompt_anchor",
        "targets": {
            "lora.k_middle": {
                "model_type": "logistic",
                "feature_names": ["worst", "improved_frac"],
                "feature_mean": [1.0, 0.5],
                "feature_scale": [0.1, 0.5],
                "coefficients": [0.0, -1.0, 1.0],
                "threshold": 0.5,
                "metadata": {},
            }
        },
        "metadata": {},
    }
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    model = load_dynamic_probe_model(model_path)

    probability = score_prompt_anchor_point(
        model,
        update_target="lora.k_middle",
        point_trace=_point_trace(),
        risk_score=0.7,
        control_score=0.75,
        activated_risk=0.2,
        budget_cap=4,
        token_block_count=8,
    )

    assert probability is not None
    features = prompt_anchor_feature_values(
        _point_trace(),
        risk_score=0.7,
        control_score=0.75,
        activated_risk=0.2,
        budget_cap=4,
        token_block_count=8,
    )
    expected_logit = -((features["worst"] - 1.0) / 0.1) + (
        (features["improved_frac"] - 0.5) / 0.5
    )
    assert probability == pytest.approx(1.0 / (1.0 + np.exp(-expected_logit)))
