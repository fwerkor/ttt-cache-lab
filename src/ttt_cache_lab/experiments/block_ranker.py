from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
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
        key = _condition_key(row)
        cell = (int(row["layer"]), int(row["token_block"]))
        grouped[row["update_target"]].append(
            {
                "condition": key,
                "cell": cell,
                "x": _feature_vector(row),
                "y": labels.get((key, cell), 0.0),
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
        models[target] = {
            "feature_names": list(FEATURE_NAMES),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "intercept": float(intercept),
            "weights": weights.tolist(),
            "ridge": float(ridge),
            "grouped_cv_mse": float(cv_mse),
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
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    x = np.stack([_feature_vector(row) for row in feature_rows], axis=0)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    standardized = (x - mean) / scale
    scores = float(model["intercept"]) + standardized @ weights
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
    by_target: dict[str, list[int]] = defaultdict(list)
    for condition, candidates in grouped.items():
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda row: (float(row["logits_kl"]), int(row.get("selected_cells", 0))),
        )
        by_target[condition[1]].append(int(best.get("selected_cells", 0)))
    defaults: dict[str, int] = {}
    for target, counts in by_target.items():
        frequencies = Counter(counts)
        defaults[target] = min(
            frequencies,
            key=lambda count: (-frequencies[count], count),
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
