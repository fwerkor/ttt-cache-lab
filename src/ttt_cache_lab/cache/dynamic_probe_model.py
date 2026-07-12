from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ANCHOR_PATTERN = re.compile(r"prompt_anchor_(b\d+)_nll_(\d+)$")


@dataclass(frozen=True)
class DynamicProbeTargetModel:
    model_type: str
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    threshold: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.model_type != "logistic":
            raise ValueError(f"Unsupported dynamic probe model type: {self.model_type}")
        feature_count = len(self.feature_names)
        if self.feature_mean.shape != (feature_count,):
            raise ValueError("feature_mean must match feature_names")
        if self.feature_scale.shape != (feature_count,):
            raise ValueError("feature_scale must match feature_names")
        if self.coefficients.shape != (feature_count + 1,):
            raise ValueError("coefficients must contain intercept plus one value per feature")
        if np.any(self.feature_scale <= 0.0):
            raise ValueError("feature_scale values must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    def probability(self, feature_values: dict[str, float]) -> float:
        missing = [name for name in self.feature_names if name not in feature_values]
        if missing:
            raise ValueError(f"Missing dynamic probe features: {missing}")
        vector = np.asarray(
            [float(feature_values[name]) for name in self.feature_names],
            dtype=np.float64,
        )
        standardized = (vector - self.feature_mean) / self.feature_scale
        logit = float(self.coefficients[0] + standardized @ self.coefficients[1:])
        if logit >= 0.0:
            return float(1.0 / (1.0 + math.exp(-min(logit, 40.0))))
        exp_logit = math.exp(max(logit, -40.0))
        return float(exp_logit / (1.0 + exp_logit))


@dataclass(frozen=True)
class DynamicProbeModel:
    format_version: int
    probe_source: str
    targets: dict[str, DynamicProbeTargetModel]
    metadata: dict[str, Any]

    def target_model(self, update_target: str) -> DynamicProbeTargetModel | None:
        direct = self.targets.get(update_target)
        if direct is not None:
            return direct
        if ".k" in update_target:
            return self.targets.get("lora.k_middle")
        if ".v" in update_target:
            return self.targets.get("lora.v_middle")
        return None


def load_dynamic_probe_model(path: Path) -> DynamicProbeModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("Unsupported dynamic probe model format")
    targets: dict[str, DynamicProbeTargetModel] = {}
    raw_targets = payload.get("targets", {})
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError("Dynamic probe model must contain target models")
    for target, raw in raw_targets.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid target model for {target}")
        targets[str(target)] = DynamicProbeTargetModel(
            model_type=str(raw.get("model_type", "")),
            feature_names=tuple(str(value) for value in raw.get("feature_names", [])),
            feature_mean=np.asarray(raw.get("feature_mean", []), dtype=np.float64),
            feature_scale=np.asarray(raw.get("feature_scale", []), dtype=np.float64),
            coefficients=np.asarray(raw.get("coefficients", []), dtype=np.float64),
            threshold=float(raw.get("threshold", math.nan)),
            metadata=dict(raw.get("metadata", {})),
        )
    return DynamicProbeModel(
        format_version=1,
        probe_source=str(payload.get("probe_source", "")),
        targets=targets,
        metadata=dict(payload.get("metadata", {})),
    )


def prompt_anchor_feature_values(
    point_trace: dict[str, Any],
    *,
    risk_score: float,
    control_score: float,
    activated_risk: float,
    budget_cap: int,
    token_block_count: int,
) -> dict[str, float]:
    """Build inference-time features from prompt-anchor probe observations.

    Evaluation-only labels stored in calibration traces are intentionally ignored.
    """

    objectives = point_trace.get("objectives", [])
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("Prompt-anchor point must contain probe objectives")
    normalized = np.asarray(
        [float(item["normalized_score"]) for item in objectives],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Prompt-anchor normalized scores must be finite")

    by_anchor: dict[str, list[float]] = {}
    by_horizon: dict[int, list[float]] = {}
    for item in objectives:
        name = str(item["name"])
        match = _ANCHOR_PATTERN.match(name)
        anchor = match.group(1) if match else name
        horizon = int(item.get("probe_length", 1))
        value = float(item["normalized_score"])
        by_anchor.setdefault(anchor, []).append(value)
        by_horizon.setdefault(horizon, []).append(value)

    anchor_max = np.asarray([max(values) for values in by_anchor.values()])
    anchor_mean = np.asarray([np.mean(values) for values in by_anchor.values()])
    horizon_max = np.asarray([max(values) for values in by_horizon.values()])
    horizon_mean = np.asarray([np.mean(values) for values in by_horizon.values()])
    raw_blocks = point_trace.get("selected_token_blocks", [])
    blocks = np.asarray([int(value) for value in raw_blocks], dtype=np.float64)
    denominator = float(max(token_block_count - 1, 1))
    if blocks.size:
        block_mean = float(np.mean(blocks) / denominator)
        block_min = float(np.min(blocks) / denominator)
        block_max = float(np.max(blocks) / denominator)
        block_span = float((np.max(blocks) - np.min(blocks)) / denominator)
    else:
        block_mean = block_min = block_max = block_span = 0.0

    return {
        "risk": float(risk_score),
        "control": float(control_score),
        "activated": float(activated_risk),
        "count": float(point_trace.get("count", 0)),
        "cap": float(budget_cap),
        "worst": float(np.max(normalized)),
        "mean": float(np.mean(normalized)),
        "median": float(np.median(normalized)),
        "q10": float(np.quantile(normalized, 0.10)),
        "q25": float(np.quantile(normalized, 0.25)),
        "q75": float(np.quantile(normalized, 0.75)),
        "q90": float(np.quantile(normalized, 0.90)),
        "best": float(np.min(normalized)),
        "std": float(np.std(normalized)),
        "range": float(np.ptp(normalized)),
        "improved_frac": float(np.mean(normalized < 1.0)),
        "improved_05pct": float(np.mean(normalized < 0.995)),
        "improved_1pct": float(np.mean(normalized < 0.99)),
        "improved_2pct": float(np.mean(normalized < 0.98)),
        "regressed_frac": float(np.mean(normalized > 1.0)),
        "anchor_count": float(len(by_anchor)),
        "objective_count": float(len(normalized)),
        "anchor_max_mean": float(np.mean(anchor_max)),
        "anchor_max_median": float(np.median(anchor_max)),
        "anchor_max_max": float(np.max(anchor_max)),
        "anchor_mean_mean": float(np.mean(anchor_mean)),
        "anchor_mean_median": float(np.median(anchor_mean)),
        "horizon_max_mean": float(np.mean(horizon_max)),
        "horizon_max_max": float(np.max(horizon_max)),
        "horizon_mean_mean": float(np.mean(horizon_mean)),
        "block_mean": block_mean,
        "block_min": block_min,
        "block_max": block_max,
        "block_span": block_span,
    }


def score_prompt_anchor_point(
    model: DynamicProbeModel,
    *,
    update_target: str,
    point_trace: dict[str, Any],
    risk_score: float,
    control_score: float,
    activated_risk: float,
    budget_cap: int,
    token_block_count: int,
) -> float | None:
    target_model = model.target_model(update_target)
    if target_model is None:
        return None
    features = prompt_anchor_feature_values(
        point_trace,
        risk_score=risk_score,
        control_score=control_score,
        activated_risk=activated_risk,
        budget_cap=budget_cap,
        token_block_count=token_block_count,
    )
    return target_model.probability(features)
