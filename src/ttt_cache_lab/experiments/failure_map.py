from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ttt_cache_lab.experiments.conditions import condition_fields, with_full_reference_metrics

FAILURE_MAP_CORE_STRATEGIES = frozenset(
    {"full_recompute", "stale_reuse", "delta_correction", "layerwise_recompute"}
)


@dataclass(frozen=True)
class FailureThresholds:
    safe_kl: float = 0.05
    safe_top1: float = 0.99
    safe_task_drop: float = 0.01


@dataclass(frozen=True)
class FailureCell:
    condition: tuple[tuple[str, str], ...]
    update_target: str
    version_gap: int
    cache_strategy: str
    count: int
    task_score_mean: float
    task_drop_vs_full: float
    logits_kl_mean: float
    top1_agreement_mean: float
    relative_error_mean: float
    false_safe_rate: float
    attention_shift_mean: float
    attention_metric_available_rate: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            **dict(self.condition),
            "update_target": self.update_target,
            "version_gap": self.version_gap,
            "cache_strategy": self.cache_strategy,
            "count": self.count,
            "task_score_mean": self.task_score_mean,
            "task_drop_vs_full": self.task_drop_vs_full,
            "logits_kl_mean": self.logits_kl_mean,
            "top1_agreement_mean": self.top1_agreement_mean,
            "relative_error_mean": self.relative_error_mean,
            "false_safe_rate": self.false_safe_rate,
            "attention_shift_mean": self.attention_shift_mean,
            "attention_metric_available_rate": self.attention_metric_available_rate,
        }

    def condition_label(self) -> str:
        return ", ".join(f"{field}={value}" for field, value in self.condition if value)


def generate_failure_map(
    input_csv: Path,
    output_dir: Path,
    *,
    thresholds: FailureThresholds | None = None,
) -> Path:
    thresholds = thresholds or FailureThresholds()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in _read_rows(input_csv)
        if row.get("cache_strategy", "") in FAILURE_MAP_CORE_STRATEGIES
    ]
    cells = _aggregate_cells(rows)
    csv_path = output_dir / "failure_map.csv"
    _write_cells(cells, csv_path)
    policy_path = output_dir / "policy_table.md"
    policy_path.write_text(_policy_table(cells, thresholds=thresholds), encoding="utf-8")
    heatmaps = {
        "logits_kl_mean": "logits_kl_heatmap.svg",
        "task_drop_vs_full": "task_drop_heatmap.svg",
        "top1_agreement_mean": "top1_agreement_heatmap.svg",
        "false_safe_rate": "false_safe_heatmap.svg",
        "attention_shift_mean": "attention_shift_heatmap.svg",
    }
    for metric, filename in heatmaps.items():
        heatmap_path = output_dir / filename
        heatmap_path.write_text(_heatmap_svg(cells, metric=metric), encoding="utf-8")
    return policy_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _aggregate_cells(rows: list[dict[str, str]]) -> list[FailureCell]:
    enriched = with_full_reference_metrics(rows)
    dimensions = condition_fields(
        enriched,
        "update_target",
        "version_gap",
        "cache_strategy",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in enriched:
        groups[tuple(row.get(field, "") for field in dimensions)].append(row)

    cells: list[FailureCell] = []
    condition_names = tuple(
        field for field in dimensions if field not in {"update_target", "version_gap", "cache_strategy"}
    )
    for key, records in sorted(groups.items()):
        values = dict(zip(dimensions, key, strict=True))
        cells.append(
            FailureCell(
                condition=tuple((field, values.get(field, "")) for field in condition_names),
                update_target=values.get("update_target", ""),
                version_gap=int(float(values.get("version_gap", "0") or 0)),
                cache_strategy=values.get("cache_strategy", ""),
                count=len(records),
                task_score_mean=_mean(records, "task_score"),
                task_drop_vs_full=_mean(records, "task_drop_vs_full"),
                logits_kl_mean=_mean(records, "logits_kl"),
                top1_agreement_mean=_mean(records, "top1_agreement"),
                relative_error_mean=_mean(records, "relative_error"),
                false_safe_rate=_mean_bool(records, "false_safe"),
                attention_shift_mean=_mean(records, "attention_shift"),
                attention_metric_available_rate=_mean_bool(
                    records, "attention_metric_available"
                ),
            )
        )
    return cells


def _write_cells(cells: list[FailureCell], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(cells[0].to_dict().keys()) if cells else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow(cell.to_dict())


def _policy_table(cells: list[FailureCell], *, thresholds: FailureThresholds) -> str:
    by_condition_target_gap: dict[
        tuple[tuple[tuple[str, str], ...], str, int], list[FailureCell]
    ] = defaultdict(list)
    for cell in cells:
        by_condition_target_gap[(cell.condition, cell.update_target, cell.version_gap)].append(cell)
    lines = [
        "# Failure map policy table",
        "",
        "| condition | update target | version gap | recommended policy | reason |",
        "|---|---|---:|---|---|",
    ]
    for (condition, target, gap), group in sorted(by_condition_target_gap.items()):
        recommendation, reason = _recommend(group, thresholds=thresholds)
        condition_label = ", ".join(f"{field}={value}" for field, value in condition if value) or "default"
        lines.append(f"| {_escape(condition_label)} | {target} | {gap} | {recommendation} | {reason} |")
    return "\n".join(lines) + "\n"


def _recommend(group: list[FailureCell], *, thresholds: FailureThresholds) -> tuple[str, str]:
    strategies = {cell.cache_strategy: cell for cell in group}
    stale = strategies.get("stale_reuse") or strategies.get("frozen_reuse")
    delta = (
        strategies.get("delta_correction")
        or strategies.get("static_base_delta")
        or strategies.get("adaptive")
    )
    partial = strategies.get("layerwise_recompute") or strategies.get("oracle_planner")

    if stale and _safe(cell=stale, thresholds=thresholds):
        return "reuse", "stale/frozen reuse stays under KL, top1, task-drop, and false-safe thresholds"
    if delta and _safe(cell=delta, thresholds=thresholds):
        return "delta", "delta correction is the cheapest safe non-reuse strategy in this cell"
    if partial and partial.top1_agreement_mean >= thresholds.safe_top1 and partial.false_safe_rate == 0.0:
        return "refresh", "partial/layer refresh avoids top-1 disagreement without full recompute"
    return "full_recompute", "reuse/correction are unsafe or unavailable for this cell"


def _safe(*, cell: FailureCell, thresholds: FailureThresholds) -> bool:
    return (
        cell.logits_kl_mean <= thresholds.safe_kl
        and cell.top1_agreement_mean >= thresholds.safe_top1
        and cell.task_drop_vs_full <= thresholds.safe_task_drop
        and cell.false_safe_rate == 0.0
    )


def _heatmap_svg(cells: list[FailureCell], *, metric: str) -> str:
    filtered = [cell for cell in cells if cell.cache_strategy != "full_recompute"]
    rows = sorted(
        {(cell.condition_label(), cell.cache_strategy, cell.update_target) for cell in filtered}
    )
    gaps = sorted({cell.version_gap for cell in filtered})
    if not rows or not gaps:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    values: dict[tuple[str, str, str, int], float | None] = {}
    for cell in filtered:
        value: float | None = float(getattr(cell, metric))
        if metric == "attention_shift_mean" and cell.attention_metric_available_rate == 0.0:
            value = None
        values[(cell.condition_label(), cell.cache_strategy, cell.update_target, cell.version_gap)] = value
    numeric_values = [value for value in values.values() if value is not None]
    max_value = max(numeric_values or [1.0]) or 1.0
    cell_w, cell_h = 96, 32
    left, top = 520, 50
    width = left + cell_w * len(gaps) + 30
    height = top + cell_h * len(rows) + 40
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        (
            f"<text x='{width / 2}' y='25' text-anchor='middle' font-size='16'>"
            f"{metric} by condition, strategy, target, and version gap</text>"
        ),
    ]
    for col, gap in enumerate(gaps):
        x = left + col * cell_w + cell_w / 2
        lines.append(f"<text x='{x}' y='{top - 12}' text-anchor='middle' font-size='12'>gap {gap}</text>")
    for row_index, (condition, strategy, target) in enumerate(rows):
        y = top + row_index * cell_h
        label = _escape(" / ".join(value for value in (condition, strategy, target) if value))
        lines.append(f"<text x='{left - 8}' y='{y + 21}' text-anchor='end' font-size='11'>{label}</text>")
        for col, gap in enumerate(gaps):
            value = values.get((condition, strategy, target, gap))
            x = left + col * cell_w
            if value is None:
                lines.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' "
                    "fill='#eeeeee' stroke='#ddd'/>"
                )
                lines.append(
                    f"<text x='{x + cell_w / 2}' y='{y + 21}' text-anchor='middle' "
                    "font-size='11'>N/A</text>"
                )
                continue
            shade = int(255 - min(220, 220 * value / max_value))
            lines.append(
                f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' "
                f"fill='rgb(255,{shade},{shade})' stroke='#ddd'/>"
            )
            lines.append(
                f"<text x='{x + cell_w / 2}' y='{y + 21}' text-anchor='middle' font-size='11'>{value:.3g}</text>"
            )
    lines.append("</svg>")
    return "\n".join(lines)


def _mean(records: list[dict[str, str]], field: str) -> float:
    values = [float(record[field]) for record in records if record.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _mean_bool(records: list[dict[str, str]], field: str) -> float:
    values = [record.get(field, "False").lower() == "true" for record in records]
    return sum(1.0 for value in values if value) / len(values) if values else 0.0


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
