from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

FEATURE_NAMES = (
    "log_stale_attention_mass",
    "log_input_weight_bound",
    "log_attention_input_bound",
    "log_predicted_delta_norm",
    "log_attention_predicted_delta",
    "token_center_fraction",
    "token_center_squared",
    "token_length_fraction",
    "layer_fraction",
)
RANK_FEATURE_NAMES = FEATURE_NAMES[:5]
RANK_FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
STATIC_SELECTOR_NAMES = (
    "sparse_attention_mass",
    "sparse_input_bound",
    "sparse_attention_input_bound",
    "sparse_predicted_delta_norm",
    "sparse_attention_predicted_delta",
)
ONE_PROBE_MARGINS = (
    0.0,
    1e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
    2e-1,
)
CONFIDENCE_PROBE_MARGINS = (
    0.0,
    1e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
)
REFERENCE_POOL_SIZE = 4
REFERENCE_POOL_MIN_STALE_KL = 1e-2
REFERENCE_ROUTER_RIDGE_VALUES = (
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
)
REFERENCE_ROUTER_OBJECTIVES = (
    "relative_gain",
    "absolute_gain",
    "reference_nll_gain",
    "best_candidate",
)
ROUTER_BLOCK_FEATURE_NAMES = (
    "stale_attention_mass",
    "input_weight_bound",
    "attention_input_bound",
    "predicted_delta_norm",
    "attention_predicted_delta",
)
BASELINE_REFERENCE_SELECTOR_NAMES = (
    "sparse_input_bound",
    "sparse_predicted_delta_norm",
)


def fit_block_ranker(
    input_dirs: list[Path],
    *,
    output_path: Path,
    ridge_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> Path:
    if not input_dirs:
        raise ValueError("At least one blockwise input directory is required")
    if not ridge_values or any(value < 0.0 for value in ridge_values):
        raise ValueError("ridge values must be nonnegative")

    feature_rows: list[dict[str, str]] = []
    mask_rows: list[dict[str, str]] = []
    record_rows: list[dict[str, str]] = []
    for directory in input_dirs:
        feature_rows.extend(_read_rows(directory / "block_features.csv"))
        mask_rows.extend(_read_rows(directory / "block_masks.csv"))
        record_rows.extend(_read_rows(directory / "blockwise_records.csv"))
    if not feature_rows:
        raise ValueError("No block feature rows were found")

    oracle_masks = _oracle_masks(mask_rows)
    labels = _oracle_frequency_labels(feature_rows, oracle_masks)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        condition = _condition_key(row)
        cell = (int(row["layer"]), int(row["token_block"]))
        grouped[row["update_target"]].append(
            {
                "condition": condition,
                "cell": cell,
                "center": float(row["token_center_fraction"]),
                "x": _feature_vector(row),
                "y": labels.get((condition, cell), 0.0),
            }
        )

    default_budgets = _default_budgets(record_rows)
    one_probe_policies = _one_probe_policies(record_rows)
    confidence_probe_policies = _confidence_probe_policies(record_rows)
    reference_candidate_pool_policies = _reference_candidate_pool_policies(
        record_rows,
        pool_size=REFERENCE_POOL_SIZE,
        min_stale_kl=REFERENCE_POOL_MIN_STALE_KL,
    )
    reference_candidate_router_policies = _reference_candidate_router_policies(
        feature_rows,
        record_rows,
        reference_candidate_pool_policies,
        ridge_values=REFERENCE_ROUTER_RIDGE_VALUES,
        min_stale_kl=REFERENCE_POOL_MIN_STALE_KL,
    )
    baseline_reference_candidate_pool_policies = _reference_candidate_pool_policies(
        record_rows,
        pool_size=REFERENCE_POOL_SIZE,
        min_stale_kl=REFERENCE_POOL_MIN_STALE_KL,
        selector_names=BASELINE_REFERENCE_SELECTOR_NAMES,
    )
    baseline_reference_candidate_router_policies = (
        _reference_candidate_router_policies(
            feature_rows,
            record_rows,
            baseline_reference_candidate_pool_policies,
            ridge_values=REFERENCE_ROUTER_RIDGE_VALUES,
            min_stale_kl=REFERENCE_POOL_MIN_STALE_KL,
            feature_mode="baseline_only",
        )
    )
    models: dict[str, Any] = {}
    for target, items in sorted(grouped.items()):
        x = np.stack([item["x"] for item in items], axis=0)
        y = np.asarray([item["y"] for item in items], dtype=np.float64)
        groups = [str(item["condition"][0]) for item in items]
        ridge, cv_mse = _choose_ridge(x, y, groups, ridge_values)
        mean, scale, intercept, weights = _fit_standardized_ridge(x, y, ridge)
        ridge_overlap = _ridge_cv_overlap(items, oracle_masks, ridge)
        fusion_feature, fusion_weight, fusion_overlap = _choose_rank_fusion(
            items, oracle_masks
        )
        position_prior = _position_prior(items)
        selected_mode = (
            "rank_fusion"
            if fusion_overlap >= ridge_overlap - 1e-15
            else "ridge"
        )
        models[target] = {
            "feature_names": list(FEATURE_NAMES),
            "selected_mode": selected_mode,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "intercept": float(intercept),
            "weights": weights.tolist(),
            "ridge": float(ridge),
            "grouped_cv_mse": float(cv_mse),
            "ridge_cv_oracle_overlap": float(ridge_overlap),
            "rank_fusion_feature": fusion_feature,
            "rank_fusion_prior_weight": float(fusion_weight),
            "rank_fusion_cv_oracle_overlap": float(fusion_overlap),
            "position_prior": {
                _center_key(center): float(value)
                for center, value in sorted(position_prior.items())
            },
            "training_cells": int(len(items)),
            "training_conditions": int(len(set(item["condition"] for item in items))),
            "default_count": int(default_budgets.get(target, 0)),
            "one_probe_policy": one_probe_policies.get(target),
            "confidence_probe_policy": confidence_probe_policies.get(target),
            "reference_candidate_pool_policy": reference_candidate_pool_policies.get(
                target
            ),
            "reference_candidate_router_policy": (
                reference_candidate_router_policies.get(target)
            ),
            "baseline_reference_candidate_pool_policy": (
                baseline_reference_candidate_pool_policies.get(target)
            ),
            "baseline_reference_candidate_router_policy": (
                baseline_reference_candidate_router_policies.get(target)
            ),
        }

    payload = {
        "format_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "models": models,
        "training_inputs": [str(path) for path in input_dirs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def load_block_ranker(path: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError(f"Unsupported block ranker format in {path}")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Block ranker feature schema does not match this implementation")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Block ranker contains no target models")
    return payload


def score_block_features(
    ranker: dict[str, Any],
    *,
    update_target: str,
    feature_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, int]:
    models = ranker.get("models", {})
    model = models.get(update_target)
    if not isinstance(model, dict):
        raise ValueError(f"Block ranker has no model for update target {update_target!r}")
    if not feature_rows:
        raise ValueError("At least one block feature row is required")
    x = np.stack([_feature_vector(row) for row in feature_rows], axis=0)
    selected_mode = str(model.get("selected_mode", "ridge"))
    if selected_mode == "rank_fusion":
        feature_name = str(model["rank_fusion_feature"])
        try:
            feature_index = FEATURE_NAMES.index(feature_name)
        except ValueError as error:
            raise ValueError(f"Unknown rank-fusion feature {feature_name!r}") from error
        feature_scores = x[:, feature_index]
        prior_scores = _lookup_position_prior(
            model.get("position_prior", {}),
            [float(row.get("token_center_fraction", 0.0)) for row in feature_rows],
        )
        prior_weight = float(model.get("rank_fusion_prior_weight", 0.0))
        scores = _rank_fusion_scores(
            prior_scores,
            feature_scores,
            prior_weight=prior_weight,
        )
    elif selected_mode == "ridge":
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        standardized = (x - mean) / scale
        scores = float(model["intercept"]) + standardized @ weights
    else:
        raise ValueError(f"Unsupported block ranker mode {selected_mode!r}")
    return np.asarray(scores, dtype=np.float64), int(model.get("default_count", 0))



def route_reference_candidate(
    ranker: dict[str, Any],
    *,
    update_target: str,
    feature_rows: list[dict[str, Any]],
    policy_name: str = "reference_candidate_router_policy",
) -> tuple[dict[str, Any], np.ndarray]:
    """Route one condition to a single distilled sparse-repair candidate."""
    model = ranker.get("models", {}).get(update_target)
    if not isinstance(model, dict):
        raise ValueError(f"Block ranker has no model for update target {update_target!r}")
    policy = model.get(policy_name)
    if not isinstance(policy, dict):
        raise ValueError(
            f"Block ranker has no {policy_name!r} for {update_target!r}"
        )
    expected_cells = int(policy.get("expected_direct_cells", 0))
    feature_mode = str(policy.get("feature_mode", "full"))
    if feature_mode == "full":
        vector = _reference_router_feature_vector(
            feature_rows,
            expected_cells=expected_cells,
        )
    elif feature_mode == "baseline_only":
        vector = _baseline_reference_router_feature_vector(
            feature_rows,
            expected_cells=expected_cells,
        )
    else:
        raise ValueError(f"Unsupported reference router feature mode {feature_mode!r}")
    mean = np.asarray(policy["mean"], dtype=np.float64)
    scale = np.asarray(policy["scale"], dtype=np.float64)
    intercept = np.asarray(policy["intercept"], dtype=np.float64)
    weights = np.asarray(policy["weights"], dtype=np.float64)
    if vector.shape != mean.shape or scale.shape != mean.shape:
        raise ValueError("Reference router feature schema does not match fitted model")
    scores = intercept + ((vector - mean) / scale) @ weights
    candidates = policy.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) != len(scores):
        raise ValueError("Reference router candidate schema is invalid")
    selected = candidates[int(np.argmax(scores))]
    if not isinstance(selected, dict):
        raise ValueError("Reference router selected an invalid candidate")
    return cast(dict[str, Any], selected), np.asarray(scores, dtype=np.float64)


def _feature_vector(row: dict[str, Any]) -> np.ndarray:
    def positive(name: str) -> float:
        return max(float(row.get(name, 0.0)), 0.0)

    center = float(row.get("token_center_fraction", 0.0))
    return np.asarray(
        [
            math.log1p(positive("stale_attention_mass")),
            math.log1p(positive("input_weight_bound")),
            math.log1p(positive("attention_input_bound")),
            math.log1p(positive("predicted_delta_norm")),
            math.log1p(positive("attention_predicted_delta")),
            center,
            center * center,
            float(row.get("token_length_fraction", 0.0)),
            float(row.get("layer_fraction", 0.0)),
        ],
        dtype=np.float64,
    )


def _choose_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    ridge_values: tuple[float, ...],
) -> tuple[float, float]:
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return float(ridge_values[0]), math.nan
    best_ridge = float(ridge_values[0])
    best_mse = math.inf
    groups_array = np.asarray(groups)
    for ridge in ridge_values:
        errors: list[float] = []
        for held_out in unique_groups:
            train = groups_array != held_out
            test = ~train
            if not np.any(train) or not np.any(test):
                continue
            mean, scale, intercept, weights = _fit_standardized_ridge(
                x[train], y[train], float(ridge)
            )
            prediction = intercept + ((x[test] - mean) / scale) @ weights
            errors.extend(((prediction - y[test]) ** 2).tolist())
        mse = float(np.mean(errors)) if errors else math.inf
        if mse < best_mse - 1e-15:
            best_mse = mse
            best_ridge = float(ridge)
    return best_ridge, best_mse


def _fit_standardized_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    design = np.concatenate([np.ones((len(z), 1), dtype=np.float64), z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return mean, scale, float(coefficients[0]), coefficients[1:]


def _ridge_cv_overlap(
    items: list[dict[str, Any]],
    oracle_masks: dict[tuple[tuple[str, ...], int], set[tuple[int, int]]],
    ridge: float,
) -> float:
    conditions = sorted(set(item["condition"] for item in items))
    if len(conditions) < 2:
        return math.nan
    overlaps: list[float] = []
    for held_out in conditions:
        train_items = [item for item in items if item["condition"] != held_out]
        test_items = [item for item in items if item["condition"] == held_out]
        train_x = np.stack([item["x"] for item in train_items], axis=0)
        train_y = np.asarray([item["y"] for item in train_items], dtype=np.float64)
        mean, scale, intercept, weights = _fit_standardized_ridge(
            train_x, train_y, ridge
        )
        test_x = np.stack([item["x"] for item in test_items], axis=0)
        scores = intercept + ((test_x - mean) / scale) @ weights
        overlaps.extend(
            _condition_oracle_overlaps(test_items, scores, oracle_masks)
        )
    return float(np.mean(overlaps)) if overlaps else math.nan


def _choose_rank_fusion(
    items: list[dict[str, Any]],
    oracle_masks: dict[tuple[tuple[str, ...], int], set[tuple[int, int]]],
) -> tuple[str, float, float]:
    conditions = sorted(set(item["condition"] for item in items))
    if len(conditions) < 2:
        return "log_input_weight_bound", 0.0, math.nan
    best_feature = "log_input_weight_bound"
    best_weight = 0.0
    best_overlap = -math.inf
    for feature_name in RANK_FEATURE_NAMES:
        feature_index = FEATURE_NAMES.index(feature_name)
        for prior_weight in RANK_FUSION_WEIGHTS:
            overlaps: list[float] = []
            for held_out in conditions:
                train_items = [item for item in items if item["condition"] != held_out]
                test_items = [item for item in items if item["condition"] == held_out]
                prior = _position_prior(train_items)
                prior_scores = np.asarray(
                    [
                        _nearest_prior(prior, float(item["center"]))
                        for item in test_items
                    ],
                    dtype=np.float64,
                )
                feature_scores = np.asarray(
                    [float(item["x"][feature_index]) for item in test_items],
                    dtype=np.float64,
                )
                scores = _rank_fusion_scores(
                    prior_scores,
                    feature_scores,
                    prior_weight=prior_weight,
                )
                overlaps.extend(
                    _condition_oracle_overlaps(test_items, scores, oracle_masks)
                )
            overlap = float(np.mean(overlaps)) if overlaps else -math.inf
            candidate = (overlap, -prior_weight, feature_name)
            incumbent = (best_overlap, -best_weight, best_feature)
            if candidate > incumbent:
                best_feature = feature_name
                best_weight = prior_weight
                best_overlap = overlap
    return best_feature, best_weight, best_overlap


def _condition_oracle_overlaps(
    items: list[dict[str, Any]],
    scores: np.ndarray,
    oracle_masks: dict[tuple[tuple[str, ...], int], set[tuple[int, int]]],
) -> list[float]:
    if not items:
        return []
    condition = cast(tuple[str, ...], items[0]["condition"])
    cells = [cast(tuple[int, int], item["cell"]) for item in items]
    order = np.argsort(-scores, kind="mergesort")
    counts = sorted(count for key, count in oracle_masks if key == condition)
    overlaps: list[float] = []
    for count in counts:
        oracle = oracle_masks[(condition, count)]
        selected = {cells[index] for index in order[:count]}
        overlaps.append(len(selected & oracle) / max(count, 1))
    return overlaps


def _position_prior(items: list[dict[str, Any]]) -> dict[float, float]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for item in items:
        grouped[round(float(item["center"]), 12)].append(float(item["y"]))
    return {center: float(np.mean(values)) for center, values in grouped.items()}


def _center_key(center: float) -> str:
    return f"{round(center, 12):.12g}"


def _lookup_position_prior(
    raw_prior: Any,
    centers: list[float],
) -> np.ndarray:
    if not isinstance(raw_prior, dict) or not raw_prior:
        return np.zeros(len(centers), dtype=np.float64)
    prior = {float(key): float(value) for key, value in raw_prior.items()}
    return np.asarray(
        [_nearest_prior(prior, center) for center in centers], dtype=np.float64
    )


def _nearest_prior(prior: dict[float, float], center: float) -> float:
    if not prior:
        return 0.0
    nearest = min(prior, key=lambda candidate: abs(candidate - center))
    return prior[nearest]


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _rank_fusion_scores(
    prior_scores: np.ndarray,
    feature_scores: np.ndarray,
    *,
    prior_weight: float,
) -> np.ndarray:
    prior_ranks = _descending_ranks(prior_scores)
    feature_ranks = _descending_ranks(feature_scores)
    return -(
        prior_weight * prior_ranks + (1.0 - prior_weight) * feature_ranks
    )


def _oracle_masks(
    rows: list[dict[str, str]],
) -> dict[tuple[tuple[str, ...], int], set[tuple[int, int]]]:
    by_budget: dict[
        tuple[tuple[str, ...], float], set[tuple[int, int]]
    ] = defaultdict(set)
    for row in rows:
        if row.get("selector") != "sparse_delta_oracle":
            continue
        budget = round(float(row["requested_budget_fraction"]), 12)
        by_budget[(_condition_key(row), budget)].add(
            (int(row["layer"]), int(row["token_block"]))
        )

    masks: dict[tuple[tuple[str, ...], int], set[tuple[int, int]]] = {}
    for (condition, _), cells in by_budget.items():
        count = len(cells)
        key = (condition, count)
        previous = masks.get(key)
        if previous is not None and previous != cells:
            raise ValueError(
                "Multiple sparse oracle masks with the same selected-cell count "
                f"for condition {condition!r}"
            )
        masks[key] = cells
    return masks


def _oracle_frequency_labels(
    feature_rows: list[dict[str, str]],
    masks: dict[tuple[tuple[str, ...], int], set[tuple[int, int]]],
) -> dict[tuple[tuple[str, ...], tuple[int, int]], float]:
    counts_by_condition: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for condition, count in masks:
        counts_by_condition[condition].append(count)
    labels: dict[tuple[tuple[str, ...], tuple[int, int]], float] = {}
    for row in feature_rows:
        condition = _condition_key(row)
        cell = (int(row["layer"]), int(row["token_block"]))
        counts = counts_by_condition.get(condition, [])
        if not counts:
            labels[(condition, cell)] = 0.0
            continue
        labels[(condition, cell)] = float(
            np.mean([cell in masks[(condition, count)] for count in counts])
        )
    return labels


def _default_budgets(rows: list[dict[str, str]]) -> dict[str, int]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("selector") in {"stale", "sparse_delta_oracle"}:
            grouped[_condition_key(row)].append(row)

    gains_by_target: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for condition, candidates in grouped.items():
        stale_rows = [row for row in candidates if row.get("selector") == "stale"]
        if not stale_rows:
            continue
        stale_kl = float(stale_rows[0]["logits_kl"])
        denominator = max(abs(stale_kl), 1e-12)
        gains_by_target[condition[1]][0].append(0.0)
        for row in candidates:
            if row.get("selector") != "sparse_delta_oracle":
                continue
            count = int(row.get("selected_cells", 0))
            gain = (stale_kl - float(row["logits_kl"])) / denominator
            gains_by_target[condition[1]][count].append(gain)

    defaults: dict[str, int] = {}
    for target, by_count in gains_by_target.items():
        defaults[target] = min(
            by_count,
            key=lambda count: (
                -float(np.median(by_count[count])),
                -float(np.mean(by_count[count])),
                count,
            ),
        )
    return defaults


def _one_probe_policies(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("selector") == "stale" or row.get("selector") in STATIC_SELECTOR_NAMES:
            grouped[_condition_key(row)].append(row)

    targets = sorted({condition[1] for condition in grouped})
    policies: dict[str, dict[str, Any]] = {}
    for target in targets:
        conditions = [condition for condition in grouped if condition[1] == target]
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for selector in STATIC_SELECTOR_NAMES:
            available_counts = sorted(
                {
                    int(row.get("selected_cells", 0))
                    for condition in conditions
                    for row in grouped[condition]
                    if row.get("selector") == selector
                }
            )
            for count in available_counts:
                for margin in ONE_PROBE_MARGINS:
                    gains: list[float] = []
                    relative_gains: list[float] = []
                    harms: list[float] = []
                    accepted = 0
                    latencies: list[float] = []
                    complete = True
                    for condition in conditions:
                        stale_rows = [
                            row
                            for row in grouped[condition]
                            if row.get("selector") == "stale"
                        ]
                        candidate_rows = [
                            row
                            for row in grouped[condition]
                            if row.get("selector") == selector
                            and int(row.get("selected_cells", 0)) == count
                        ]
                        if not stale_rows or not candidate_rows:
                            complete = False
                            break
                        stale = stale_rows[0]
                        candidate = candidate_rows[0]
                        stale_nll = float(stale.get("reference_token_nll", "nan"))
                        candidate_nll = float(
                            candidate.get("reference_token_nll", "nan")
                        )
                        if not np.isfinite(stale_nll) or not np.isfinite(candidate_nll):
                            complete = False
                            break
                        use_candidate = stale_nll - candidate_nll > margin
                        stale_kl = float(stale["logits_kl"])
                        selected_kl = (
                            float(candidate["logits_kl"]) if use_candidate else stale_kl
                        )
                        gain = stale_kl - selected_kl
                        gains.append(gain)
                        relative_gains.append(gain / max(abs(stale_kl), 1e-12))
                        harms.append(selected_kl - stale_kl)
                        accepted += int(use_candidate)
                        latencies.append(
                            float(candidate.get("cache_maintenance_latency", 0.0))
                            + float(candidate.get("decode_latency", 0.0))
                        )
                    if not complete or not gains:
                        continue
                    harmful = sum(harm > 1e-15 for harm in harms)
                    worst_harm = max(harms)
                    mean_gain = float(np.mean(gains))
                    mean_relative_gain = float(np.mean(relative_gains))
                    mean_latency = float(np.mean(latencies))
                    # Safety first. On equivalent calibration behavior prefer a larger
                    # margin, then fewer repaired blocks and lower probe latency.
                    key = (
                        -harmful,
                        -worst_harm,
                        mean_gain,
                        mean_relative_gain,
                        margin,
                        -count,
                        -mean_latency,
                        selector,
                    )
                    candidates.append(
                        (
                            key,
                            {
                                "candidate_selector": selector,
                                "candidate_count": count,
                                "reference_nll_margin": margin,
                                "calibration_conditions": len(conditions),
                                "calibration_harmful": harmful,
                                "calibration_worst_harm": worst_harm,
                                "calibration_mean_gain": mean_gain,
                                "calibration_mean_relative_gain": mean_relative_gain,
                                "calibration_accepted": accepted,
                                "calibration_mean_candidate_latency": mean_latency,
                            },
                        )
                    )
        if candidates:
            policies[target] = max(candidates, key=lambda item: item[0])[1]
    return policies


def _confidence_probe_policies(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Fit a one-candidate gate that needs no reference answer at runtime."""
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("selector") == "stale" or row.get("selector") in STATIC_SELECTOR_NAMES:
            grouped[_condition_key(row)].append(row)

    targets = sorted({condition[1] for condition in grouped})
    policies: dict[str, dict[str, Any]] = {}
    for target in targets:
        conditions = [condition for condition in grouped if condition[1] == target]
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for selector in STATIC_SELECTOR_NAMES:
            available_counts = sorted(
                {
                    int(row.get("selected_cells", 0))
                    for condition in conditions
                    for row in grouped[condition]
                    if row.get("selector") == selector
                }
            )
            for count in available_counts:
                for margin in CONFIDENCE_PROBE_MARGINS:
                    gains: list[float] = []
                    relative_gains: list[float] = []
                    harms: list[float] = []
                    accepted = 0
                    latencies: list[float] = []
                    complete = True
                    for condition in conditions:
                        stale_rows = [
                            row
                            for row in grouped[condition]
                            if row.get("selector") == "stale"
                        ]
                        candidate_rows = [
                            row
                            for row in grouped[condition]
                            if row.get("selector") == selector
                            and int(row.get("selected_cells", 0)) == count
                        ]
                        if not stale_rows or not candidate_rows:
                            complete = False
                            break
                        stale = stale_rows[0]
                        candidate = candidate_rows[0]
                        stale_confidence = float(
                            stale.get("output_max_probability", "nan")
                        )
                        candidate_confidence = float(
                            candidate.get("output_max_probability", "nan")
                        )
                        if not np.isfinite(stale_confidence) or not np.isfinite(
                            candidate_confidence
                        ):
                            complete = False
                            break
                        use_candidate = candidate_confidence - stale_confidence > margin
                        stale_kl = float(stale["logits_kl"])
                        selected_kl = (
                            float(candidate["logits_kl"]) if use_candidate else stale_kl
                        )
                        gain = stale_kl - selected_kl
                        gains.append(gain)
                        relative_gains.append(gain / max(abs(stale_kl), 1e-12))
                        harms.append(selected_kl - stale_kl)
                        accepted += int(use_candidate)
                        latencies.append(
                            float(candidate.get("cache_maintenance_latency", 0.0))
                            + float(candidate.get("decode_latency", 0.0))
                        )
                    if not complete or not gains:
                        continue
                    harmful = sum(harm > 1e-15 for harm in harms)
                    worst_harm = max(harms)
                    mean_gain = float(np.mean(gains))
                    mean_relative_gain = float(np.mean(relative_gains))
                    mean_latency = float(np.mean(latencies))
                    key = (
                        -harmful,
                        -worst_harm,
                        mean_gain,
                        mean_relative_gain,
                        margin,
                        -count,
                        -mean_latency,
                        selector,
                    )
                    candidates.append(
                        (
                            key,
                            {
                                "candidate_selector": selector,
                                "candidate_count": count,
                                "confidence_margin": margin,
                                "calibration_conditions": len(conditions),
                                "calibration_harmful": harmful,
                                "calibration_worst_harm": worst_harm,
                                "calibration_mean_gain": mean_gain,
                                "calibration_mean_relative_gain": mean_relative_gain,
                                "calibration_accepted": accepted,
                                "calibration_mean_candidate_latency": mean_latency,
                            },
                        )
                    )
        if candidates:
            policies[target] = max(candidates, key=lambda item: item[0])[1]
    return policies




def _reference_candidate_pool_policies(
    rows: list[dict[str, str]],
    *,
    pool_size: int = REFERENCE_POOL_SIZE,
    min_stale_kl: float = REFERENCE_POOL_MIN_STALE_KL,
    selector_names: tuple[str, ...] = STATIC_SELECTOR_NAMES,
) -> dict[str, dict[str, Any]]:
    """Distill a small static candidate set for reference-guided selection."""
    if pool_size < 1:
        raise ValueError("reference candidate pool size must be positive")
    if min_stale_kl < 0.0:
        raise ValueError("minimum stale KL must be nonnegative")
    if not selector_names:
        raise ValueError("reference candidate selectors must not be empty")

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("selector") == "stale" or row.get("selector") in selector_names:
            grouped[_condition_key(row)].append(row)

    policies: dict[str, dict[str, Any]] = {}
    targets = sorted({condition[1] for condition in grouped})
    for target in targets:
        conditions: list[tuple[str, ...]] = []
        for condition, candidates in grouped.items():
            if condition[1] != target:
                continue
            stale_rows = [row for row in candidates if row.get("selector") == "stale"]
            if not stale_rows:
                continue
            stale_kl = float(stale_rows[0].get("logits_kl", "nan"))
            stale_nll = float(stale_rows[0].get("reference_token_nll", "nan"))
            if (
                not np.isfinite(stale_kl)
                or not np.isfinite(stale_nll)
                or stale_kl < min_stale_kl
            ):
                continue
            conditions.append(condition)
        if not conditions:
            continue

        available: set[tuple[str, int]] | None = None
        rows_by_condition: dict[
            tuple[str, ...], dict[tuple[str, int], dict[str, str]]
        ] = {}
        stale_by_condition: dict[tuple[str, ...], dict[str, str]] = {}
        for condition in conditions:
            candidates = grouped[condition]
            stale_by_condition[condition] = next(
                row for row in candidates if row.get("selector") == "stale"
            )
            keyed = {
                (str(row["selector"]), int(row.get("selected_cells", 0))): row
                for row in candidates
                if row.get("selector") in selector_names
                and int(row.get("selected_cells", 0)) > 0
                and np.isfinite(float(row.get("reference_token_nll", "nan")))
            }
            rows_by_condition[condition] = keyed
            keys = set(keyed)
            available = keys if available is None else available & keys
        if not available:
            continue

        def evaluate_pool(
            pool: list[tuple[str, int]],
            *,
            condition_list: tuple[tuple[str, ...], ...] = tuple(conditions),
            stale_rows: dict[tuple[str, ...], dict[str, str]] = stale_by_condition,
            candidate_rows: dict[
                tuple[str, ...], dict[tuple[str, int], dict[str, str]]
            ] = rows_by_condition,
        ) -> dict[str, float | int]:
            gain_sum = 0.0
            stale_sum = 0.0
            harmful = 0
            beneficial = 0
            worst_relative_gain = math.inf
            for condition in condition_list:
                stale = stale_rows[condition]
                stale_kl = float(stale["logits_kl"])
                best_kl = stale_kl
                best_nll = float(stale["reference_token_nll"])
                for key in pool:
                    candidate = candidate_rows[condition][key]
                    candidate_nll = float(candidate["reference_token_nll"])
                    if candidate_nll < best_nll - 1e-15:
                        best_nll = candidate_nll
                        best_kl = float(candidate["logits_kl"])
                gain = stale_kl - best_kl
                relative_gain = gain / max(abs(stale_kl), 1e-12)
                gain_sum += gain
                stale_sum += stale_kl
                harmful += int(gain < -1e-15)
                beneficial += int(gain > 1e-15)
                worst_relative_gain = min(worst_relative_gain, relative_gain)
            return {
                "weighted_recovery": gain_sum / max(stale_sum, 1e-12),
                "harmful": harmful,
                "beneficial": beneficial,
                "worst_relative_gain": worst_relative_gain,
            }

        selected: list[tuple[str, int]] = []
        calibration: dict[str, float | int] = {}
        for _ in range(min(pool_size, len(available))):
            best: tuple[
                tuple[Any, ...], tuple[str, int], dict[str, float | int]
            ] | None = None
            for candidate in sorted(available - set(selected)):
                metrics = evaluate_pool([*selected, candidate])
                key = (
                    -int(metrics["harmful"]),
                    float(metrics["weighted_recovery"]),
                    float(metrics["worst_relative_gain"]),
                    int(metrics["beneficial"]),
                    -candidate[1],
                    candidate[0],
                )
                if best is None or key > best[0]:
                    best = (key, candidate, metrics)
            if best is None:
                break
            selected.append(best[1])
            calibration = best[2]

        if selected:
            policies[target] = {
                "candidates": [
                    {"candidate_selector": selector, "candidate_count": count}
                    for selector, count in selected
                ],
                "reference_nll_margin": 0.0,
                "calibration_conditions": len(conditions),
                "calibration_min_stale_kl": float(min_stale_kl),
                "calibration_harmful": int(calibration.get("harmful", 0)),
                "calibration_beneficial": int(calibration.get("beneficial", 0)),
                "calibration_weighted_recovery": float(
                    calibration.get("weighted_recovery", 0.0)
                ),
                "calibration_worst_relative_gain": float(
                    calibration.get("worst_relative_gain", 0.0)
                ),
            }
    return policies



def _reference_candidate_router_policies(
    feature_rows: list[dict[str, str]],
    record_rows: list[dict[str, str]],
    pool_policies: dict[str, dict[str, Any]],
    *,
    ridge_values: tuple[float, ...] = REFERENCE_ROUTER_RIDGE_VALUES,
    min_stale_kl: float = REFERENCE_POOL_MIN_STALE_KL,
    feature_mode: str = "full",
) -> dict[str, dict[str, Any]]:
    """Fit a one-probe router over the distilled candidate pool."""
    if not ridge_values or any(value < 0.0 for value in ridge_values):
        raise ValueError("reference router ridge values must be nonnegative")
    if feature_mode not in {"full", "baseline_only"}:
        raise ValueError(f"Unsupported reference router feature mode {feature_mode!r}")
    features_by_condition: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    records_by_condition: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        features_by_condition[_condition_key(row)].append(row)
    for row in record_rows:
        records_by_condition[_condition_key(row)].append(row)

    policies: dict[str, dict[str, Any]] = {}
    for target, pool_policy in sorted(pool_policies.items()):
        raw_candidates = pool_policy.get("candidates", [])
        candidates = [
            {
                "candidate_selector": str(item["candidate_selector"]),
                "candidate_count": int(item["candidate_count"]),
            }
            for item in raw_candidates
            if isinstance(item, dict)
        ]
        if not candidates:
            continue
        items: list[dict[str, Any]] = []
        direct_counts: list[int] = []
        for condition, rows in features_by_condition.items():
            if condition[1] != target:
                continue
            records = records_by_condition.get(condition, [])
            stale_rows = [row for row in records if row.get("selector") == "stale"]
            if not stale_rows:
                continue
            stale = stale_rows[0]
            stale_kl = float(stale.get("logits_kl", "nan"))
            stale_nll = float(stale.get("reference_token_nll", "nan"))
            if not np.isfinite(stale_kl) or not np.isfinite(stale_nll):
                continue
            candidate_rows: list[dict[str, str]] = []
            complete = True
            for candidate in candidates:
                matches = [
                    row
                    for row in records
                    if row.get("selector") == candidate["candidate_selector"]
                    and int(row.get("selected_cells", 0))
                    == candidate["candidate_count"]
                ]
                if not matches:
                    complete = False
                    break
                row = matches[0]
                if not np.isfinite(float(row.get("reference_token_nll", "nan"))):
                    complete = False
                    break
                candidate_rows.append(row)
            if not complete:
                continue
            expected_cells = len(rows)
            direct_counts.append(expected_cells)
            items.append(
                {
                    "condition": condition,
                    "sample_group": str(rows[0].get("sample_id", condition[0])),
                    "feature_rows": rows,
                    "expected_cells": expected_cells,
                    "stale_kl": stale_kl,
                    "stale_nll": stale_nll,
                    "candidate_kl": np.asarray(
                        [float(row["logits_kl"]) for row in candidate_rows],
                        dtype=np.float64,
                    ),
                    "candidate_nll": np.asarray(
                        [float(row["reference_token_nll"]) for row in candidate_rows],
                        dtype=np.float64,
                    ),
                }
            )
        if not items:
            continue
        expected_cells = max(set(direct_counts), key=direct_counts.count)
        items = [item for item in items if item["expected_cells"] == expected_cells]
        if len(items) < 2:
            continue
        vector_fn = (
            _reference_router_feature_vector
            if feature_mode == "full"
            else _baseline_reference_router_feature_vector
        )
        x = np.stack(
            [
                vector_fn(item["feature_rows"], expected_cells=expected_cells)
                for item in items
            ],
            axis=0,
        )
        groups = np.asarray([item["sample_group"] for item in items])
        objectives = {
            objective: _reference_router_targets(items, objective=objective)
            for objective in REFERENCE_ROUTER_OBJECTIVES
        }
        best: tuple[tuple[Any, ...], str, float, np.ndarray] | None = None
        for objective, y in objectives.items():
            for ridge in ridge_values:
                predictions = np.zeros_like(y)
                unique_groups = sorted(set(groups.tolist()))
                if len(unique_groups) < 2:
                    mean, scale, intercept, weights = (
                        _fit_standardized_multioutput_ridge(x, y, float(ridge))
                    )
                    predictions = intercept + ((x - mean) / scale) @ weights
                else:
                    for held_out in unique_groups:
                        train = groups != held_out
                        test = ~train
                        mean, scale, intercept, weights = (
                            _fit_standardized_multioutput_ridge(
                                x[train], y[train], float(ridge)
                            )
                        )
                        predictions[test] = (
                            intercept + ((x[test] - mean) / scale) @ weights
                        )
                choices = np.argmax(predictions, axis=1)
                metrics = _reference_router_metrics(
                    items,
                    choices,
                    min_stale_kl=min_stale_kl,
                )
                key = (
                    int(metrics["material_harmful"]),
                    -float(metrics["material_weighted_recovery"]),
                    int(metrics["all_harmful"]),
                    -float(metrics["all_weighted_recovery"]),
                    REFERENCE_ROUTER_OBJECTIVES.index(objective),
                    float(ridge),
                )
                if best is None or key < best[0]:
                    best = (key, objective, float(ridge), predictions.copy())
        if best is None:
            continue
        _, objective, ridge, cv_predictions = best
        cv_choices = np.argmax(cv_predictions, axis=1)
        y = objectives[objective]
        mean, scale, intercept, weights = _fit_standardized_multioutput_ridge(
            x, y, ridge
        )
        metrics = _reference_router_metrics(
            items,
            cv_choices,
            min_stale_kl=min_stale_kl,
        )
        second_probe_margin, second_probe_metrics = (
            _choose_reference_router_second_probe_margin(
                items,
                cv_predictions,
                min_stale_kl=min_stale_kl,
                max_average_probes=1.5,
            )
        )
        policies[target] = {
            "candidates": candidates,
            "feature_mode": feature_mode,
            "objective": objective,
            "ridge": ridge,
            "expected_direct_cells": expected_cells,
            "feature_dimension": int(x.shape[1]),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "intercept": intercept.tolist(),
            "weights": weights.tolist(),
            "second_probe_score_margin": second_probe_margin,
            "second_probe_max_average_probes": 1.5,
            **second_probe_metrics,
            "calibration_conditions": len(items),
            "calibration_min_stale_kl": float(min_stale_kl),
            **metrics,
        }
    return policies


def _reference_router_targets(
    items: list[dict[str, Any]],
    *,
    objective: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for item in items:
        stale_kl = float(item["stale_kl"])
        candidate_kl = np.asarray(item["candidate_kl"], dtype=np.float64)
        candidate_nll = np.asarray(item["candidate_nll"], dtype=np.float64)
        relative_gain = (stale_kl - candidate_kl) / max(stale_kl, 1e-6)
        if objective == "relative_gain":
            rows.append(relative_gain)
        elif objective == "absolute_gain":
            rows.append(stale_kl - candidate_kl)
        elif objective == "reference_nll_gain":
            rows.append(float(item["stale_nll"]) - candidate_nll)
        elif objective == "best_candidate":
            target = np.zeros(len(candidate_kl), dtype=np.float64)
            target[int(np.argmax(relative_gain))] = 1.0
            rows.append(target)
        else:
            raise ValueError(f"Unsupported reference router objective {objective!r}")
    return np.stack(rows, axis=0)


def _reference_router_metrics(
    items: list[dict[str, Any]],
    choices: np.ndarray,
    *,
    min_stale_kl: float,
) -> dict[str, float | int]:
    stale_values: list[float] = []
    gains: list[float] = []
    for item, choice in zip(items, choices, strict=True):
        index = int(choice)
        stale_kl = float(item["stale_kl"])
        stale_nll = float(item["stale_nll"])
        candidate_nll = float(item["candidate_nll"][index])
        selected_kl = (
            float(item["candidate_kl"][index])
            if candidate_nll < stale_nll - 1e-15
            else stale_kl
        )
        stale_values.append(stale_kl)
        gains.append(stale_kl - selected_kl)
    stale = np.asarray(stale_values, dtype=np.float64)
    gain = np.asarray(gains, dtype=np.float64)
    material = stale >= min_stale_kl

    def weighted(mask: np.ndarray) -> float:
        return float(gain[mask].sum() / max(stale[mask].sum(), 1e-12))

    return {
        "all_harmful": int(np.sum(gain < -1e-15)),
        "all_beneficial": int(np.sum(gain > 1e-15)),
        "all_weighted_recovery": weighted(np.ones(len(stale), dtype=bool)),
        "material_conditions": int(np.sum(material)),
        "material_harmful": int(np.sum((gain < -1e-15) & material)),
        "material_beneficial": int(np.sum((gain > 1e-15) & material)),
        "material_weighted_recovery": weighted(material) if np.any(material) else 0.0,
    }



def _choose_reference_router_second_probe_margin(
    items: list[dict[str, Any]],
    predictions: np.ndarray,
    *,
    min_stale_kl: float,
    max_average_probes: float,
) -> tuple[float, dict[str, float | int]]:
    """Choose a safety-first confidence threshold for an optional second probe."""
    if predictions.ndim != 2 or predictions.shape[0] != len(items):
        raise ValueError("Reference router predictions have an invalid shape")
    order = np.argsort(-predictions, axis=1)
    margins = predictions[np.arange(len(items)), order[:, 0]] - predictions[
        np.arange(len(items)), order[:, 1]
    ]
    thresholds = sorted(
        {
            0.0,
            *np.quantile(margins, np.linspace(0.05, 0.95, 19)).tolist(),
            float(np.max(margins) + 1e-12),
        }
    )
    best: tuple[tuple[Any, ...], float, dict[str, float | int]] | None = None
    for threshold in thresholds:
        metrics = _reference_router_second_probe_metrics(
            items,
            predictions,
            score_margin=threshold,
            min_stale_kl=min_stale_kl,
        )
        if float(metrics["second_probe_average_probes"]) > max_average_probes + 1e-12:
            continue
        key = (
            int(metrics["second_probe_material_harmful"]),
            -float(metrics["second_probe_material_weighted_recovery"]),
            float(metrics["second_probe_average_probes"]),
            int(metrics["second_probe_all_harmful"]),
            -float(metrics["second_probe_all_weighted_recovery"]),
            threshold,
        )
        if best is None or key < best[0]:
            best = (key, threshold, metrics)
    if best is None:
        metrics = _reference_router_second_probe_metrics(
            items,
            predictions,
            score_margin=0.0,
            min_stale_kl=min_stale_kl,
        )
        return 0.0, metrics
    return float(best[1]), best[2]


def _reference_router_second_probe_metrics(
    items: list[dict[str, Any]],
    predictions: np.ndarray,
    *,
    score_margin: float,
    min_stale_kl: float,
) -> dict[str, float | int]:
    order = np.argsort(-predictions, axis=1)
    stale_values: list[float] = []
    gains: list[float] = []
    probe_counts: list[int] = []
    for row_index, item in enumerate(items):
        first = int(order[row_index, 0])
        second = int(order[row_index, 1])
        margin = float(predictions[row_index, first] - predictions[row_index, second])
        trigger_second = margin < score_margin
        stale_kl = float(item["stale_kl"])
        best_kl = stale_kl
        best_nll = float(item["stale_nll"])
        candidate_nll = np.asarray(item["candidate_nll"], dtype=np.float64)
        candidate_kl = np.asarray(item["candidate_kl"], dtype=np.float64)
        for candidate_index in (first, second) if trigger_second else (first,):
            if candidate_nll[candidate_index] < best_nll - 1e-15:
                best_nll = float(candidate_nll[candidate_index])
                best_kl = float(candidate_kl[candidate_index])
        stale_values.append(stale_kl)
        gains.append(stale_kl - best_kl)
        probe_counts.append(2 if trigger_second else 1)
    stale = np.asarray(stale_values, dtype=np.float64)
    gain = np.asarray(gains, dtype=np.float64)
    material = stale >= min_stale_kl

    def weighted(mask: np.ndarray) -> float:
        return float(gain[mask].sum() / max(stale[mask].sum(), 1e-12))

    return {
        "second_probe_average_probes": float(np.mean(probe_counts)),
        "second_probe_trigger_rate": float(np.mean(np.asarray(probe_counts) == 2)),
        "second_probe_all_harmful": int(np.sum(gain < -1e-15)),
        "second_probe_all_beneficial": int(np.sum(gain > 1e-15)),
        "second_probe_all_weighted_recovery": weighted(
            np.ones(len(stale), dtype=bool)
        ),
        "second_probe_material_conditions": int(np.sum(material)),
        "second_probe_material_harmful": int(
            np.sum((gain < -1e-15) & material)
        ),
        "second_probe_material_beneficial": int(
            np.sum((gain > 1e-15) & material)
        ),
        "second_probe_material_weighted_recovery": (
            weighted(material) if np.any(material) else 0.0
        ),
    }


def _fit_standardized_multioutput_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    design = np.concatenate([np.ones((len(z), 1), dtype=np.float64), z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return mean, scale, coefficients[0], coefficients[1:]


def _baseline_reference_router_feature_vector(
    feature_rows: list[dict[str, Any]],
    *,
    expected_cells: int,
) -> np.ndarray:
    """Build a router vector without post-update attention or stale logits."""
    if expected_cells <= 0 or len(feature_rows) != expected_cells:
        raise ValueError(
            "Baseline reference router requires exactly "
            f"{expected_cells} direct cells, received {len(feature_rows)}"
        )
    ordered = sorted(
        feature_rows,
        key=lambda row: (int(row.get("layer", 0)), int(row.get("token_block", 0))),
    )
    names = ("input_weight_bound", "predicted_delta_norm")
    raw = np.stack(
        [
            np.asarray(
                [math.log1p(max(float(row.get(name, 0.0)), 0.0)) for name in names],
                dtype=np.float64,
            )
            for row in ordered
        ],
        axis=0,
    )
    ranks = np.empty_like(raw)
    denominator = max(expected_cells - 1, 1)
    for feature_index in range(raw.shape[1]):
        order = np.argsort(-raw[:, feature_index], kind="mergesort")
        feature_ranks = np.empty(expected_cells, dtype=np.float64)
        feature_ranks[order] = np.arange(expected_cells, dtype=np.float64)
        ranks[:, feature_index] = 1.0 - feature_ranks / denominator
    positions = np.asarray(
        [float(row.get("token_center_fraction", 0.0)) for row in ordered],
        dtype=np.float64,
    )
    aggregates: list[float] = []
    for feature_index in range(raw.shape[1]):
        values = raw[:, feature_index]
        aggregates.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.max(values)),
                float(np.median(values)),
                float(np.mean(np.sort(values)[-min(2, len(values)) :])),
                float(np.max(values) - np.median(values)),
            ]
        )
    context_length = max(float(ordered[0].get("context_length", 1.0)), 1.0)
    return np.concatenate(
        [
            raw.reshape(-1),
            ranks.reshape(-1),
            positions,
            positions * positions,
            np.asarray(aggregates, dtype=np.float64),
            np.asarray([math.log2(context_length)], dtype=np.float64),
        ]
    )


def _reference_router_feature_vector(
    feature_rows: list[dict[str, Any]],
    *,
    expected_cells: int,
) -> np.ndarray:
    if expected_cells <= 0 or len(feature_rows) != expected_cells:
        raise ValueError(
            "Reference router requires exactly "
            f"{expected_cells} direct cells, received {len(feature_rows)}"
        )
    ordered = sorted(
        feature_rows,
        key=lambda row: (int(row.get("layer", 0)), int(row.get("token_block", 0))),
    )
    raw = np.stack(
        [
            np.asarray(
                [
                    math.log1p(max(float(row.get(name, 0.0)), 0.0))
                    for name in ROUTER_BLOCK_FEATURE_NAMES
                ],
                dtype=np.float64,
            )
            for row in ordered
        ],
        axis=0,
    )
    ranks = np.empty_like(raw)
    denominator = max(expected_cells - 1, 1)
    for feature_index in range(raw.shape[1]):
        order = np.argsort(-raw[:, feature_index], kind="mergesort")
        feature_ranks = np.empty(expected_cells, dtype=np.float64)
        feature_ranks[order] = np.arange(expected_cells, dtype=np.float64)
        ranks[:, feature_index] = 1.0 - feature_ranks / denominator
    aggregates: list[float] = []
    for feature_index in range(raw.shape[1]):
        values = raw[:, feature_index]
        aggregates.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.max(values)),
                float(np.median(values)),
                float(np.mean(np.sort(values)[-min(2, len(values)) :])),
                float(np.max(values) - np.median(values)),
            ]
        )
    context_length = max(float(ordered[0].get("context_length", 1.0)), 1.0)
    return np.concatenate(
        [
            raw.reshape(-1),
            ranks.reshape(-1),
            np.asarray(aggregates, dtype=np.float64),
            np.asarray([math.log2(context_length)], dtype=np.float64),
        ]
    )


def _condition_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("dataset_sample_id", row.get("sample_id", ""))),
        str(row.get("update_target", "")),
        str(row.get("block_size", "")),
        str(row.get("final_adapter_fingerprint", "")),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
