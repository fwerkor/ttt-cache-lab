from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import cast

from ttt_cache_lab.experiments.conditions import condition_fields
from ttt_cache_lab.experiments.study_analysis import generate_study_analysis

NUMERIC_FIELDS = [
    "task_score",
    "logits_kl",
    "top1_agreement",
    "relative_error",
    "hidden_relative_error",
    "latency_units",
    "recompute_fraction",
    "refresh_count",
]


def generate_report(input_csv: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    grouped = _group(
        rows,
        list(
            condition_fields(
                rows,
                "update_target",
                "cache_strategy",
                "adapter_version",
                "cached_version",
                "version_gap",
            )
        ),
    )
    series_fields = condition_fields(rows, "update_target", "cache_strategy")
    markdown = _markdown_report(rows, grouped)
    report_path = output_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    for metric in ["logits_kl", "relative_error", "task_score", "latency_units"]:
        _write_svg_line_plot(
            grouped,
            metric=metric,
            output=output_dir / f"{metric}_by_version.svg",
            series_fields=series_fields,
        )
    generate_study_analysis(input_csv, output_dir)
    return report_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, str | float | int]]:
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in keys)].append(row)
    out: list[dict[str, str | float | int]] = []
    for key, records in sorted(buckets.items()):
        item: dict[str, str | float | int] = {name: value for name, value in zip(keys, key, strict=True)}
        item["count"] = len(records)
        for field in NUMERIC_FIELDS:
            item[f"{field}_mean"] = _mean(records, field)
        if "version_gap" in records[0]:
            item["version_gap_mean"] = _mean(records, "version_gap")
        if "accumulated_update_norm" in records[0]:
            item["accumulated_update_norm_mean"] = _mean(records, "accumulated_update_norm")
        out.append(item)
    return out


def _markdown_report(rows: list[dict[str, str]], grouped: list[dict[str, str | float | int]]) -> str:
    experiments = sorted({row.get("experiment_id", "") for row in rows})
    targets = sorted({row.get("update_target", "") for row in rows})
    strategies = sorted({row.get("cache_strategy", "") for row in rows})
    versions = sorted({int(row.get("adapter_version", "0")) for row in rows})
    lines = [
        "# Experiment report",
        "",
        "## Overview",
        "",
        f"- records: {len(rows)}",
        f"- experiments: {', '.join(experiments)}",
        f"- update targets: {', '.join(targets)}",
        f"- cache strategies: {', '.join(strategies)}",
        f"- adapter versions: {', '.join(str(v) for v in versions)}",
        "",
        "## Grouped means",
        "",
        "| experiment | target | strategy | version | n | task | KL | top1 | rel_err | latency |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in grouped[:400]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("experiment_id", "")),
                    str(item.get("update_target", "")),
                    str(item.get("cache_strategy", "")),
                    str(item.get("adapter_version", "")),
                    str(item.get("count", "")),
                    _fmt(item.get("task_score_mean")),
                    _fmt(item.get("logits_kl_mean")),
                    _fmt(item.get("top1_agreement_mean")),
                    _fmt(item.get("relative_error_mean")),
                    _fmt(item.get("latency_units_mean")),
                ]
            )
            + " |"
        )
    if len(grouped) > 400:
        lines.append(f"\n_Truncated grouped table to 400 rows out of {len(grouped)}._")
    lines += [
        "",
        "## Generated figures",
        "",
        "- `logits_kl_by_version.svg`",
        "- `relative_error_by_version.svg`",
        "- `task_score_by_version.svg`",
        "- `latency_units_by_version.svg`",
        "",
    ]
    return "\n".join(lines)


def _write_svg_line_plot(
    grouped: list[dict[str, str | float | int]],
    *,
    metric: str,
    output: Path,
    series_fields: tuple[str, ...],
) -> None:
    series: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    field = f"{metric}_mean"
    for item in grouped:
        version_raw = item.get("adapter_version", 0)
        value_raw = item.get(field)
        if value_raw is None:
            continue
        key = tuple(str(item.get(field, "")) for field in series_fields)
        series[key].append((float(version_raw), float(value_raw)))
    width, height = 900, 520
    margin = 60
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        output.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(xmin, xmax):
        xmax = xmin + 1
    if math.isclose(ymin, ymax):
        ymax = ymin + 1

    def sx(x: float) -> float:
        return margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='18'>{metric} by adapter version</text>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black'/>",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black'/>",
        f"<text x='{width / 2}' y='{height - 15}' text-anchor='middle' font-size='13'>adapter_version</text>",
        (
            f"<text x='18' y='{height / 2}' transform='rotate(-90 18 {height / 2})' "
            f"text-anchor='middle' font-size='13'>{metric}</text>"
        ),
    ]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for idx, (key, points) in enumerate(sorted(series.items())[:24]):
        points = sorted(points)
        color = palette[idx % len(palette)]
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        lines.append(f"<polyline points='{polyline}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in points:
            lines.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='2.5' fill='{color}'/>")
        lx = width - margin + 10
        ly = margin + 18 * idx
        label = " / ".join(value for value in key if value)
        lines.append(f"<text x='{lx}' y='{ly}' font-size='10' fill='{color}'>{_escape(label[:48])}</text>")
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _fmt(value: object) -> str:
    if value is None:
        return ""
    number = float(cast(float | int | str, value))
    if abs(number) >= 1000 or (abs(number) < 0.001 and number != 0):
        return f"{number:.4g}"
    return f"{number:.4f}"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
