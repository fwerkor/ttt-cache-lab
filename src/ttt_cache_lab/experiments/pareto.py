from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParetoPoint:
    experiment_id: str
    cache_strategy: str
    update_target: str
    task_score_mean: float
    latency_units_mean: float
    recompute_fraction_mean: float
    refresh_count_mean: float
    false_safe_rate: float
    dominated: bool

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "experiment_id": self.experiment_id,
            "cache_strategy": self.cache_strategy,
            "update_target": self.update_target,
            "task_score_mean": self.task_score_mean,
            "latency_units_mean": self.latency_units_mean,
            "recompute_fraction_mean": self.recompute_fraction_mean,
            "refresh_count_mean": self.refresh_count_mean,
            "false_safe_rate": self.false_safe_rate,
            "dominated": self.dominated,
        }


def generate_pareto(input_csv: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    points = _aggregate(rows)
    output = output_dir / "pareto.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(points[0].to_dict().keys()) if points else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point.to_dict())
    (output_dir / "pareto.md").write_text(_markdown(points), encoding="utf-8")
    (output_dir / "pareto.svg").write_text(_pareto_svg(points), encoding="utf-8")
    return output


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _aggregate(rows: list[dict[str, str]]) -> list[ParetoPoint]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("experiment_id", ""), row.get("cache_strategy", ""), row.get("update_target", ""))
        groups[key].append(row)
    raw_points = []
    for (experiment, strategy, target), records in sorted(groups.items()):
        raw_points.append(
            ParetoPoint(
                experiment_id=experiment,
                cache_strategy=strategy,
                update_target=target,
                task_score_mean=_mean(records, "task_score"),
                latency_units_mean=_mean_preferred(records, ("end_to_end_latency", "latency_units")),
                recompute_fraction_mean=_mean(records, "recompute_fraction"),
                refresh_count_mean=_mean(records, "refresh_count"),
                false_safe_rate=_mean_bool(records, "false_safe"),
                dominated=False,
            )
        )
    return [
        ParetoPoint(
            experiment_id=point.experiment_id,
            cache_strategy=point.cache_strategy,
            update_target=point.update_target,
            task_score_mean=point.task_score_mean,
            latency_units_mean=point.latency_units_mean,
            recompute_fraction_mean=point.recompute_fraction_mean,
            refresh_count_mean=point.refresh_count_mean,
            false_safe_rate=point.false_safe_rate,
            dominated=_is_dominated(point, raw_points),
        )
        for point in raw_points
    ]


def _is_dominated(point: ParetoPoint, all_points: list[ParetoPoint]) -> bool:
    candidates = [
        other
        for other in all_points
        if other.experiment_id == point.experiment_id and other.update_target == point.update_target and other != point
    ]
    for other in candidates:
        no_worse_quality = other.task_score_mean >= point.task_score_mean
        no_worse_latency = other.latency_units_mean <= point.latency_units_mean
        strictly_better = (
            other.task_score_mean > point.task_score_mean or other.latency_units_mean < point.latency_units_mean
        )
        no_worse_safety = other.false_safe_rate <= point.false_safe_rate
        if no_worse_quality and no_worse_latency and no_worse_safety and strictly_better:
            return True
    return False


def _markdown(points: list[ParetoPoint]) -> str:
    lines = [
        "# Planner Pareto table",
        "",
        "| experiment | target | strategy | task | latency | recompute | refreshes | false-safe | dominated |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for point in points:
        lines.append(
            "| "
            + " | ".join(
                [
                    point.experiment_id,
                    point.update_target,
                    point.cache_strategy,
                    f"{point.task_score_mean:.4f}",
                    f"{point.latency_units_mean:.4f}",
                    f"{point.recompute_fraction_mean:.4f}",
                    f"{point.refresh_count_mean:.4f}",
                    f"{point.false_safe_rate:.4f}",
                    "yes" if point.dominated else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _pareto_svg(points: list[ParetoPoint]) -> str:
    if not points:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    width, height = 1000, 640
    left, right, top, bottom = 90, 320, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    latencies = [point.latency_units_mean for point in points]
    qualities = [point.task_score_mean for point in points]
    min_x, max_x = min(latencies), max(latencies)
    min_y, max_y = min(qualities), max(qualities)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0

    def x_coord(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def y_coord(value: float) -> float:
        return top + plot_h - (value - min_y) / (max_y - min_y) * plot_h

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='black'/>",
        (
            f"<text x='{left + plot_w / 2}' y='{height - 25}' text-anchor='middle' font-size='14'>"
            "end-to-end latency (lower is better)</text>"
        ),
        (
            f"<text x='22' y='{top + plot_h / 2}' text-anchor='middle' font-size='14' "
            f"transform='rotate(-90 22 {top + plot_h / 2})'>"
            "task score (higher is better)</text>"
        ),
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='17'>Planner quality-cost Pareto frontier</text>",
        f"<text x='{left}' y='{top + plot_h + 22}' text-anchor='middle' font-size='11'>{min_x:.4g}</text>",
        f"<text x='{left + plot_w}' y='{top + plot_h + 22}' text-anchor='middle' font-size='11'>{max_x:.4g}</text>",
        f"<text x='{left - 10}' y='{top + plot_h}' text-anchor='end' font-size='11'>{min_y:.4g}</text>",
        f"<text x='{left - 10}' y='{top + 4}' text-anchor='end' font-size='11'>{max_y:.4g}</text>",
    ]
    legend_y = top + 10
    for index, point in enumerate(points):
        x = x_coord(point.latency_units_mean)
        y = y_coord(point.task_score_mean)
        opacity = "0.35" if point.dominated else "1.0"
        radius = 4 if point.dominated else 6
        lines.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius}' fill='black' opacity='{opacity}' "
            f"stroke='black' stroke-width='1'/>"
        )
        label = _escape(f"{point.update_target} / {point.cache_strategy}")
        lines.append(
            f"<text x='{left + plot_w + 18}' y='{legend_y + index * 17}' font-size='11' "
            f"opacity='{opacity}'>{label}</text>"
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _mean_preferred(records: list[dict[str, str]], fields: tuple[str, ...]) -> float:
    for field in fields:
        values = [float(record[field]) for record in records if record.get(field) not in {None, ""}]
        if values:
            return sum(values) / len(values)
    return 0.0


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mean(records: list[dict[str, str]], field: str) -> float:
    values = [float(record[field]) for record in records if record.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _mean_bool(records: list[dict[str, str]], field: str) -> float:
    values = [record.get(field, "False").lower() == "true" for record in records]
    return sum(1.0 for value in values if value) / len(values) if values else 0.0
