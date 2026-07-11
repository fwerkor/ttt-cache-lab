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
