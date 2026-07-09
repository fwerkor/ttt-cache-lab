from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SAFE_KL = 0.05
SAFE_TOP1 = 0.99
SAFE_TASK_DROP = 0.01


@dataclass(frozen=True)
class FailureCell:
    experiment_id: str
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

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "experiment_id": self.experiment_id,
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
        }


def generate_failure_map(input_csv: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    cells = _aggregate_cells(rows)
    csv_path = output_dir / "failure_map.csv"
    _write_cells(cells, csv_path)
    policy_path = output_dir / "policy_table.md"
    policy_path.write_text(_policy_table(cells), encoding="utf-8")
    heatmap_path = output_dir / "logits_kl_heatmap.svg"
    heatmap_path.write_text(_heatmap_svg(cells, metric="logits_kl_mean"), encoding="utf-8")
    return policy_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _aggregate_cells(rows: list[dict[str, str]]) -> list[FailureCell]:
    groups: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    full_scores: dict[tuple[str, str, int], float] = {}
    for row in rows:
        experiment = row.get("experiment_id", "")
        target = row.get("update_target", "")
        gap = int(float(row.get("version_gap", row.get("adapter_version", "0")) or 0))
        strategy = row.get("cache_strategy", "")
        groups[(experiment, target, gap, strategy)].append(row)

    for (experiment, target, gap, strategy), records in groups.items():
        if strategy == "full_recompute":
            full_scores[(experiment, target, gap)] = _mean(records, "task_score")

    cells = []
    for (experiment, target, gap, strategy), records in sorted(groups.items()):
        full_score = full_scores.get((experiment, target, gap), _mean(records, "task_score"))
        task_score = _mean(records, "task_score")
        cells.append(
            FailureCell(
                experiment_id=experiment,
                update_target=target,
                version_gap=gap,
                cache_strategy=strategy,
                count=len(records),
                task_score_mean=task_score,
                task_drop_vs_full=full_score - task_score,
                logits_kl_mean=_mean(records, "logits_kl"),
                top1_agreement_mean=_mean(records, "top1_agreement"),
                relative_error_mean=_mean(records, "relative_error"),
                false_safe_rate=_mean_bool(records, "false_safe"),
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


def _policy_table(cells: list[FailureCell]) -> str:
    by_target_gap: dict[tuple[str, int], list[FailureCell]] = defaultdict(list)
    for cell in cells:
        by_target_gap[(cell.update_target, cell.version_gap)].append(cell)
    lines = [
        "# Failure map policy table",
        "",
        "| update target | version gap | recommended policy | reason |",
        "|---|---:|---|---|",
    ]
    for (target, gap), group in sorted(by_target_gap.items()):
        recommendation, reason = _recommend(group)
        lines.append(f"| {target} | {gap} | {recommendation} | {reason} |")
    return "\n".join(lines) + "\n"


def _recommend(group: list[FailureCell]) -> tuple[str, str]:
    strategies = {cell.cache_strategy: cell for cell in group}
    stale = strategies.get("stale_reuse") or strategies.get("frozen_reuse")
    delta = strategies.get("delta_correction") or strategies.get("static_base_delta") or strategies.get("adaptive")
    partial = strategies.get("layerwise_recompute") or strategies.get("oracle_planner")

    if stale and _safe(cell=stale):
        return "reuse", "stale/frozen reuse stays under KL, top1, task-drop, and false-safe thresholds"
    if delta and _safe(cell=delta):
        return "delta", "delta correction is the cheapest safe non-reuse strategy in this cell"
    if partial and partial.top1_agreement_mean >= SAFE_TOP1 and partial.false_safe_rate == 0.0:
        return "refresh", "partial/layer refresh avoids top-1 disagreement without full recompute"
    return "full_recompute", "reuse/correction are unsafe or unavailable for this cell"


def _safe(*, cell: FailureCell) -> bool:
    return (
        cell.logits_kl_mean <= SAFE_KL
        and cell.top1_agreement_mean >= SAFE_TOP1
        and cell.task_drop_vs_full <= SAFE_TASK_DROP
        and cell.false_safe_rate == 0.0
    )


def _heatmap_svg(cells: list[FailureCell], *, metric: str) -> str:
    filtered = [cell for cell in cells if cell.cache_strategy in {"stale_reuse", "adaptive", "delta_correction"}]
    targets = sorted({cell.update_target for cell in filtered})
    gaps = sorted({cell.version_gap for cell in filtered})
    if not targets or not gaps:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    values = {(cell.update_target, cell.version_gap): float(getattr(cell, metric)) for cell in filtered}
    max_value = max(values.values() or [1.0]) or 1.0
    cell_w, cell_h = 96, 32
    left, top = 180, 50
    width = left + cell_w * len(gaps) + 30
    height = top + cell_h * len(targets) + 40
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width / 2}' y='25' text-anchor='middle' font-size='16'>{metric} by target and version gap</text>",
    ]
    for col, gap in enumerate(gaps):
        x = left + col * cell_w + cell_w / 2
        lines.append(f"<text x='{x}' y='{top - 12}' text-anchor='middle' font-size='12'>gap {gap}</text>")
    for row, target in enumerate(targets):
        y = top + row * cell_h
        lines.append(f"<text x='{left - 8}' y='{y + 21}' text-anchor='end' font-size='12'>{_escape(target)}</text>")
        for col, gap in enumerate(gaps):
            value = values.get((target, gap), 0.0)
            shade = int(255 - min(220, 220 * value / max_value))
            x = left + col * cell_w
            rect = (
                f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' "
                f"fill='rgb(255,{shade},{shade})' stroke='#ddd'/>"
            )
            lines.append(rect)
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
