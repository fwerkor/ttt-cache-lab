from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ttt_cache_lab.experiments.conditions import with_full_reference_metrics


@dataclass(frozen=True)
class WindowThresholds:
    safe_kl: float = 0.05
    safe_top1: float = 0.99
    safe_task_drop: float = 0.01
    min_safe_rate: float = 0.95


def generate_window_analysis(
    input_csv: Path,
    output_dir: Path,
    *,
    thresholds: WindowThresholds | None = None,
) -> tuple[Path, Path]:
    thresholds = thresholds or WindowThresholds()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    enriched = with_full_reference_metrics(rows)
    window_rows = [
        row
        for row in enriched
        if row.get("cache_strategy") == "windowed_recompute"
        and _number(row, "version_gap") > 0.0
        and _window_size(row) > 0
    ]
    cells = _aggregate(window_rows, thresholds=thresholds)
    cells_path = output_dir / "window_cells.csv"
    _write_rows(cells_path, cells)
    minima = _select_minima(cells, thresholds=thresholds)
    minima_path = output_dir / "minimal_safe_windows.csv"
    _write_rows(minima_path, minima)
    markdown_path = output_dir / "minimal_safe_windows.md"
    markdown_path.write_text(_to_markdown(minima), encoding="utf-8")
    return cells_path, minima_path


def _aggregate(
    rows: list[dict[str, str]],
    *,
    thresholds: WindowThresholds,
) -> list[dict[str, str | int | float | bool]]:
    dimensions = tuple(
        field
        for field in (
            "model_name",
            "model_num_layers",
            "context_length",
            "synthetic_difficulty",
            "prompt_format",
            "task_name",
            "task_family",
            "update_target",
            "adapter_version",
            "cached_version",
            "version_gap",
            "lora_rank",
            "configured_update_norm",
            "update_mode",
            "norm_control",
        )
        if any(field in row for row in rows)
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        window_size = _window_size(row)
        key = tuple(row.get(field, "") for field in dimensions) + (str(window_size),)
        groups[key].append(row)

    cells: list[dict[str, str | int | float | bool]] = []
    for key, group in sorted(groups.items()):
        values = dict(zip((*dimensions, "recompute_window_size"), key, strict=True))
        safe_flags = [_safe(row, thresholds=thresholds) for row in group]
        count = len(group)
        cells.append(
            {
                **values,
                "recompute_window_size": int(values["recompute_window_size"]),
                "count": count,
                "safe_rate": sum(safe_flags) / count if count else 0.0,
                "logits_kl_mean": _mean(group, "logits_kl"),
                "top1_agreement_mean": _mean(group, "top1_agreement"),
                "task_drop_vs_full_mean": _mean(group, "task_drop_vs_full"),
                "false_safe_rate": _mean_bool(group, "false_safe"),
                "recompute_fraction_mean": _mean(group, "recompute_fraction"),
                "flops_fraction_mean": _mean(group, "flops_fraction"),
                "end_to_end_latency_mean": _mean(group, "end_to_end_latency"),
            }
        )
    return cells


def _select_minima(
    cells: list[dict[str, str | int | float | bool]],
    *,
    thresholds: WindowThresholds,
) -> list[dict[str, str | int | float | bool]]:
    groups: dict[
        tuple[tuple[str, str], ...], list[dict[str, str | int | float | bool]]
    ] = defaultdict(list)
    excluded = {
        "recompute_window_size",
        "count",
        "safe_rate",
        "logits_kl_mean",
        "top1_agreement_mean",
        "task_drop_vs_full_mean",
        "false_safe_rate",
        "recompute_fraction_mean",
        "flops_fraction_mean",
        "end_to_end_latency_mean",
    }
    for cell in cells:
        key = tuple(sorted((field, str(value)) for field, value in cell.items() if field not in excluded))
        groups[key].append(cell)

    minima: list[dict[str, str | int | float | bool]] = []
    for key, candidates in sorted(groups.items()):
        ordered = sorted(candidates, key=lambda item: int(item["recompute_window_size"]))
        safe = [
            item
            for item in ordered
            if float(item["safe_rate"]) >= thresholds.min_safe_rate
            and float(item["logits_kl_mean"]) <= thresholds.safe_kl
            and float(item["top1_agreement_mean"]) >= thresholds.safe_top1
            and float(item["task_drop_vs_full_mean"]) <= thresholds.safe_task_drop
            and float(item["false_safe_rate"]) == 0.0
        ]
        chosen = safe[0] if safe else ordered[-1]
        minima.append(
            {
                **dict(key),
                "minimum_safe_window": int(chosen["recompute_window_size"]) if safe else -1,
                "safe_window_found": bool(safe),
                "tested_max_window": int(ordered[-1]["recompute_window_size"]),
                "safe_rate": float(chosen["safe_rate"]),
                "logits_kl_mean": float(chosen["logits_kl_mean"]),
                "top1_agreement_mean": float(chosen["top1_agreement_mean"]),
                "task_drop_vs_full_mean": float(chosen["task_drop_vs_full_mean"]),
                "recompute_fraction_mean": float(chosen["recompute_fraction_mean"]),
                "flops_fraction_mean": float(chosen["flops_fraction_mean"]),
                "end_to_end_latency_mean": float(chosen["end_to_end_latency_mean"]),
            }
        )
    return minima


def _safe(row: dict[str, str], *, thresholds: WindowThresholds) -> bool:
    return (
        _number(row, "logits_kl") <= thresholds.safe_kl
        and _number(row, "top1_agreement") >= thresholds.safe_top1
        and _number(row, "task_drop_vs_full") <= thresholds.safe_task_drop
        and not _boolean(row, "false_safe")
    )


def _window_size(row: dict[str, str]) -> int:
    raw = row.get("recompute_window_size") or row.get("sweep.cache.recompute_window_size") or "0"
    return int(float(raw))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = [_number(row, field) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _mean_bool(rows: list[dict[str, str]], field: str) -> float:
    values = [_boolean(row, field) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _number(row: dict[str, str], field: str) -> float:
    raw = row.get(field)
    return float(raw) if raw not in {None, ""} else 0.0


def _boolean(row: dict[str, str], field: str) -> bool:
    return row.get(field, "").strip().lower() in {"1", "true", "yes"}


def _to_markdown(rows: list[dict[str, str | int | float | bool]]) -> str:
    if not rows:
        return "# Minimal safe recompute windows\n\nNo windowed-recompute records were found.\n"
    fields = [
        field
        for field in (
            "model_name",
            "task_name",
            "update_target",
            "version_gap",
            "minimum_safe_window",
            "safe_window_found",
            "safe_rate",
            "logits_kl_mean",
            "task_drop_vs_full_mean",
            "recompute_fraction_mean",
        )
        if field in rows[0]
    ]
    lines = [
        "# Minimal safe recompute windows",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines) + "\n"
