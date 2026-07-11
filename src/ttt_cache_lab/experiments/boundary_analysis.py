from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

METRIC_FIELDS = (
    "attention_kl",
    "attention_js",
    "attention_l1",
    "attention_argmax_disagreement",
    "attention_top4_distance",
    "attention_top8_distance",
    "attention_topk_distance",
    "attention_output_relative_error",
    "attention_output_cosine_distance",
    "boundary_input_hidden_relative_error",
    "boundary_next_hidden_relative_error",
    "key_relative_error",
    "value_relative_error",
    "attention_weighted_key_relative_error",
    "attention_weighted_value_relative_error",
    "suffix_attention_js_mean",
    "suffix_attention_js_max",
    "suffix_attention_output_relative_error_mean",
    "suffix_attention_output_relative_error_max",
    "suffix_attention_input_relative_error_mean",
    "suffix_attention_input_relative_error_max",
    "suffix_attention_input_relative_error_last",
    "suffix_amplification_ratio",
)

PREDICTOR_FEATURES = (
    "attention_js",
    "attention_argmax_disagreement",
    "attention_top4_distance",
    "attention_top8_distance",
    "attention_topk_distance",
    "attention_output_relative_error",
    "attention_output_cosine_distance",
    "boundary_next_hidden_relative_error",
    "attention_weighted_key_relative_error",
    "attention_weighted_value_relative_error",
    "suffix_attention_js_mean",
    "suffix_attention_js_max",
    "suffix_attention_output_relative_error_mean",
    "suffix_attention_output_relative_error_max",
    "suffix_attention_input_relative_error_mean",
    "suffix_attention_input_relative_error_max",
    "suffix_attention_input_relative_error_last",
    "suffix_amplification_ratio",
    "recompute_fraction",
)


@dataclass(frozen=True)
class BoundaryAnalysisArtifacts:
    enriched_rows_path: Path
    metric_evaluation_path: Path
    group_selections_path: Path
    predictor_summary_path: Path


def generate_boundary_analysis(
    boundary_input: Path,
    summary_input: Path,
    output_dir: Path,
    *,
    ridge: float = 1e-3,
) -> BoundaryAnalysisArtifacts:
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_rows = _read_rows(boundary_input)
    summary_rows = _read_rows(summary_input)
    enriched = _enrich(boundary_rows, summary_rows)
    enriched_rows_path = output_dir / "boundary_rows_enriched.csv"
    _write_rows(enriched_rows_path, enriched)

    metric_rows, selection_rows = _evaluate_metrics(enriched)
    metric_evaluation_path = output_dir / "boundary_metric_evaluation.csv"
    group_selections_path = output_dir / "boundary_group_selections.csv"
    _write_rows(metric_evaluation_path, metric_rows)
    _write_rows(group_selections_path, selection_rows)

    predictor_rows = _leave_one_sample_out_predictor(enriched, ridge=ridge)
    predictor_summary_path = output_dir / "boundary_predictor_summary.csv"
    _write_rows(predictor_summary_path, predictor_rows)
    (output_dir / "boundary_analysis.md").write_text(
        _to_markdown(metric_rows, predictor_rows), encoding="utf-8"
    )
    return BoundaryAnalysisArtifacts(
        enriched_rows_path=enriched_rows_path,
        metric_evaluation_path=metric_evaluation_path,
        group_selections_path=group_selections_path,
        predictor_summary_path=predictor_summary_path,
    )


def _enrich(
    boundary_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> list[dict[str, str | int | float | bool]]:
    stale: dict[tuple[str, ...], dict[str, str]] = {}
    for row in summary_rows:
        if row.get("cache_strategy") != "stale_reuse":
            continue
        stale[_summary_key(row)] = row

    enriched: list[dict[str, str | int | float | bool]] = []
    for row in boundary_rows:
        if not _boolean(row, "has_stale_rejoin") or not _boolean(row, "metric_available"):
            continue
        item: dict[str, str | int | float | bool] = dict(row)
        item["attention_argmax_disagreement"] = 1.0 - _number(
            row, "attention_argmax_agreement"
        )
        item["attention_top4_distance"] = 1.0 - _number(
            row, "attention_top4_overlap"
        )
        item["attention_top8_distance"] = 1.0 - _number(
            row, "attention_top8_overlap"
        )
        item["attention_topk_distance"] = 1.0 - _number(row, "attention_topk_overlap")
        reference = stale.get(_boundary_key(row))
        stale_kl = _number(reference, "logits_kl") if reference is not None else math.nan
        stale_top1 = (
            _number(reference, "top1_agreement") if reference is not None else math.nan
        )
        stale_task_drop = (
            _number(reference, "full_task_score") - _number(reference, "task_score")
            if reference is not None
            else math.nan
        )
        item["stale_logits_kl"] = stale_kl
        item["stale_top1_agreement"] = stale_top1
        item["stale_task_drop_vs_full"] = stale_task_drop
        item["kl_gain_vs_stale"] = (
            stale_kl - _number(row, "logits_kl") if math.isfinite(stale_kl) else math.nan
        )
        item["beneficial_vs_stale"] = (
            math.isfinite(stale_kl)
            and _number(row, "logits_kl") <= stale_kl + 1e-12
            and _number(row, "top1_agreement") >= stale_top1 - 1e-12
            and _number(row, "task_drop_vs_full") <= stale_task_drop + 1e-12
        )
        enriched.append(item)
    return enriched


def _evaluate_metrics(
    rows: list[dict[str, str | int | float | bool]],
) -> tuple[
    list[dict[str, str | int | float | bool]],
    list[dict[str, str | int | float | bool]],
]:
    groups = _groups(rows)
    selectors: dict[str, Callable[[dict[str, str | int | float | bool]], float]] = {
        field: _metric_selector(field) for field in METRIC_FIELDS
    }
    selectors["shortest_window"] = lambda row: _numeric(row, "window_size")
    selectors["largest_window"] = lambda row: -_numeric(row, "window_size")

    evaluation: list[dict[str, str | int | float | bool]] = []
    selections: list[dict[str, str | int | float | bool]] = []
    all_kl = np.asarray([_numeric(row, "logits_kl") for row in rows], dtype=np.float64)
    for name, selector in selectors.items():
        selected_rows: list[dict[str, str | int | float | bool]] = []
        regrets: list[float] = []
        normalized_regrets: list[float] = []
        oracle_hits = 0
        beneficial_hits = 0
        within_correlations: list[float] = []
        for group_key, group in sorted(groups.items()):
            ordered = sorted(group, key=lambda row: (_numeric(row, "window_size"), str(row)))
            chosen = min(ordered, key=lambda row: (selector(row), _numeric(row, "window_size")))
            oracle = min(
                ordered,
                key=lambda row: (_numeric(row, "logits_kl"), _numeric(row, "window_size")),
            )
            chosen_kl = _numeric(chosen, "logits_kl")
            oracle_kl = _numeric(oracle, "logits_kl")
            regret = chosen_kl - oracle_kl
            scale = max(max(_numeric(row, "logits_kl") for row in ordered) - oracle_kl, 1e-12)
            normalized_regret = regret / scale
            regrets.append(regret)
            normalized_regrets.append(normalized_regret)
            oracle_hit = int(_numeric(chosen, "window_size") == _numeric(oracle, "window_size"))
            oracle_hits += oracle_hit
            beneficial = bool(chosen.get("beneficial_vs_stale", False))
            beneficial_hits += int(beneficial)
            if name in METRIC_FIELDS and len(ordered) >= 3:
                correlation = _spearman(
                    np.asarray([selector(row) for row in ordered], dtype=np.float64),
                    np.asarray([_numeric(row, "logits_kl") for row in ordered], dtype=np.float64),
                )
                if math.isfinite(correlation):
                    within_correlations.append(correlation)
            selection: dict[str, str | int | float | bool] = {
                **dict(group_key),
                "selector": name,
                "selected_window": int(_numeric(chosen, "window_size")),
                "oracle_window": int(_numeric(oracle, "window_size")),
                "selected_logits_kl": chosen_kl,
                "oracle_logits_kl": oracle_kl,
                "kl_regret": regret,
                "normalized_kl_regret": normalized_regret,
                "selected_beneficial_vs_stale": beneficial,
                "selected_recompute_fraction": _numeric(chosen, "recompute_fraction"),
                "selected_boundary_layer": int(_numeric(chosen, "boundary_layer")),
                "oracle_hit": bool(oracle_hit),
            }
            selections.append(selection)
            selected_rows.append(chosen)

        metric_values = np.asarray([selector(row) for row in rows], dtype=np.float64)
        global_correlation = (
            _spearman(metric_values, all_kl) if name in METRIC_FIELDS else math.nan
        )
        count = len(selected_rows)
        evaluation.append(
            {
                "selector": name,
                "group_count": count,
                "global_spearman_vs_kl": global_correlation,
                "mean_within_group_spearman_vs_kl": (
                    float(np.mean(within_correlations)) if within_correlations else math.nan
                ),
                "valid_within_group_correlations": len(within_correlations),
                "oracle_window_hit_rate": oracle_hits / count if count else 0.0,
                "beneficial_selection_rate": beneficial_hits / count if count else 0.0,
                "mean_kl_regret": float(np.mean(regrets)) if regrets else 0.0,
                "median_kl_regret": float(np.median(regrets)) if regrets else 0.0,
                "mean_normalized_kl_regret": (
                    float(np.mean(normalized_regrets)) if normalized_regrets else 0.0
                ),
                "mean_selected_recompute_fraction": (
                    float(
                        np.mean(
                            [_numeric(row, "recompute_fraction") for row in selected_rows]
                        )
                    )
                    if selected_rows
                    else 0.0
                ),
            }
        )
    evaluation.sort(
        key=lambda row: (
            _sort_number(row["mean_normalized_kl_regret"]),
            -_sort_number(row["beneficial_selection_rate"]),
            str(row["selector"]),
        )
    )
    return evaluation, selections


def _leave_one_sample_out_predictor(
    rows: list[dict[str, str | int | float | bool]],
    *,
    ridge: float,
) -> list[dict[str, str | int | float | bool]]:
    sample_ids = sorted({_sample_key(row) for row in rows})
    if len(sample_ids) < 2:
        return [
            {
                "predictor": "ridge_log_kl_leave_one_sample_out",
                "status": "insufficient_samples",
                "held_out_sample_count": len(sample_ids),
                "group_count": 0,
                "oracle_window_hit_rate": math.nan,
                "beneficial_selection_rate": math.nan,
                "mean_kl_regret": math.nan,
                "median_kl_regret": math.nan,
                "mean_normalized_kl_regret": math.nan,
                "mean_selected_recompute_fraction": math.nan,
            }
        ]

    predictions: list[dict[str, str | int | float | bool]] = []
    for held_out in sample_ids:
        train = [row for row in rows if _sample_key(row) != held_out]
        test = [row for row in rows if _sample_key(row) == held_out]
        if not train or not test:
            continue
        x_train = _feature_matrix(train)
        y_train = np.log(np.asarray([_numeric(row, "logits_kl") for row in train]) + 1e-8)
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1e-12] = 1.0
        standardized = (x_train - mean) / std
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
        x_test = (_feature_matrix(test) - mean) / std
        predicted = np.column_stack([np.ones(len(x_test)), x_test]) @ coefficients
        for row, value in zip(test, predicted, strict=True):
            predictions.append({**row, "predicted_log_kl": float(value)})

    groups = _groups(predictions)
    regrets: list[float] = []
    normalized_regrets: list[float] = []
    selected_recompute: list[float] = []
    oracle_hits = 0
    beneficial_hits = 0
    for group in groups.values():
        chosen = min(
            group,
            key=lambda row: (
                _numeric(row, "predicted_log_kl"),
                _numeric(row, "window_size"),
            ),
        )
        oracle = min(
            group,
            key=lambda row: (_numeric(row, "logits_kl"), _numeric(row, "window_size")),
        )
        chosen_kl = _numeric(chosen, "logits_kl")
        oracle_kl = _numeric(oracle, "logits_kl")
        regret = chosen_kl - oracle_kl
        scale = max(max(_numeric(row, "logits_kl") for row in group) - oracle_kl, 1e-12)
        regrets.append(regret)
        normalized_regrets.append(regret / scale)
        selected_recompute.append(_numeric(chosen, "recompute_fraction"))
        oracle_hits += int(_numeric(chosen, "window_size") == _numeric(oracle, "window_size"))
        beneficial_hits += int(bool(chosen.get("beneficial_vs_stale", False)))

    count = len(groups)
    return [
        {
            "predictor": "ridge_log_kl_leave_one_sample_out",
            "status": "ok" if count else "no_groups",
            "held_out_sample_count": len(sample_ids),
            "group_count": count,
            "feature_count": len(PREDICTOR_FEATURES),
            "features": ";".join(PREDICTOR_FEATURES),
            "ridge": ridge,
            "oracle_window_hit_rate": oracle_hits / count if count else math.nan,
            "beneficial_selection_rate": beneficial_hits / count if count else math.nan,
            "mean_kl_regret": float(np.mean(regrets)) if regrets else math.nan,
            "median_kl_regret": float(np.median(regrets)) if regrets else math.nan,
            "mean_normalized_kl_regret": (
                float(np.mean(normalized_regrets)) if normalized_regrets else math.nan
            ),
            "mean_selected_recompute_fraction": (
                float(np.mean(selected_recompute)) if selected_recompute else math.nan
            ),
        }
    ]


def _groups(
    rows: list[dict[str, str | int | float | bool]],
) -> dict[
    tuple[tuple[str, str], ...],
    list[dict[str, str | int | float | bool]],
]:
    fields = (
        "model_name",
        "task_name",
        "sample_id",
        "dataset_sample_id",
        "update_target",
        "adapter_version",
        "cached_version",
        "version_gap",
        "context_length",
        "synthetic_difficulty",
        "configured_update_norm",
        "seed",
    )
    groups: dict[
        tuple[tuple[str, str], ...],
        list[dict[str, str | int | float | bool]],
    ] = defaultdict(list)
    for row in rows:
        key = tuple((field, str(row.get(field, ""))) for field in fields)
        groups[key].append(row)
    return groups


def _metric_selector(
    field: str,
) -> Callable[[dict[str, str | int | float | bool]], float]:
    def select(row: dict[str, str | int | float | bool]) -> float:
        return _numeric(row, field)

    return select


def _feature_matrix(rows: list[dict[str, str | int | float | bool]]) -> np.ndarray:
    return np.asarray(
        [[_numeric(row, field) for field in PREDICTOR_FEATURES] for row in rows],
        dtype=np.float64,
    )


def _summary_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("sample_id", ""),
        row.get("dataset_sample_id", ""),
        row.get("task_name", ""),
        row.get("update_target", ""),
        row.get("adapter_version", ""),
        row.get("cached_version", ""),
        row.get("context_length", ""),
        row.get("model_name", ""),
        row.get("seed", ""),
    )


def _boundary_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("sample_id", ""),
        row.get("dataset_sample_id", ""),
        row.get("task_name", ""),
        row.get("update_target", ""),
        row.get("adapter_version", ""),
        row.get("cached_version", ""),
        row.get("context_length", ""),
        row.get("model_name", ""),
        row.get("seed", ""),
    )


def _sample_key(row: dict[str, str | int | float | bool]) -> str:
    return f"{row.get('model_name', '')}:{row.get('dataset_sample_id') or row.get('sample_id', '')}"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average = 0.5 * (index + end - 1) + 1.0
        ranks[order[index:end]] = average
        index = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 2:
        return math.nan
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(
    path: Path,
    rows: list[dict[str, str | int | float | bool]],
) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str] | None, field: str) -> float:
    if row is None:
        return 0.0
    raw = row.get(field)
    return float(raw) if raw not in {None, ""} else 0.0


def _numeric(row: dict[str, str | int | float | bool], field: str) -> float:
    raw = row.get(field)
    if raw in {None, ""}:
        return 0.0
    if isinstance(raw, bool):
        return float(raw)
    return float(raw)


def _boolean(row: dict[str, str], field: str) -> bool:
    return row.get(field, "").strip().lower() in {"1", "true", "yes"}


def _sort_number(value: str | int | float | bool) -> float:
    number = float(value)
    return number if math.isfinite(number) else math.inf


def _to_markdown(
    metric_rows: list[dict[str, str | int | float | bool]],
    predictor_rows: list[dict[str, str | int | float | bool]],
) -> str:
    lines = ["# Boundary compatibility analysis", ""]
    if metric_rows:
        fields = (
            "selector",
            "global_spearman_vs_kl",
            "mean_within_group_spearman_vs_kl",
            "oracle_window_hit_rate",
            "beneficial_selection_rate",
            "mean_normalized_kl_regret",
            "mean_selected_recompute_fraction",
        )
        lines.extend(
            [
                "## Single-metric selectors",
                "",
                "| " + " | ".join(fields) + " |",
                "|" + "|".join("---" for _ in fields) + "|",
            ]
        )
        for row in metric_rows:
            lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    if predictor_rows:
        lines.extend(["", "## Held-out predictor", ""])
        for key, value in predictor_rows[0].items():
            lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"
