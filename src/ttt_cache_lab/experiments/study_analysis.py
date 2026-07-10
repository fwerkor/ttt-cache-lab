from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyThresholds:
    safe_kl: float = 0.05
    safe_top1: float = 0.99
    safe_task_drop: float = 0.01


def generate_study_analysis(
    input_csv: Path,
    output_dir: Path,
    *,
    thresholds: StudyThresholds | None = None,
) -> list[Path]:
    thresholds = thresholds or StudyThresholds()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    artifacts: list[Path] = []
    experiment_ids = {row.get("experiment_id", "").lower() for row in rows}
    if any("e1" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e1(rows, output_dir))
    if any("e2" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e2(rows, output_dir, thresholds=thresholds))
    if any("e3" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e3(rows, output_dir, thresholds=thresholds))
    if any("e4" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e4(rows, output_dir, thresholds=thresholds))
    if any("e5" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e5(rows, output_dir, thresholds=thresholds))
    if any("e6" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e6(rows, output_dir))
    if any("e7" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e7(rows, output_dir, thresholds=thresholds))
    return artifacts


def _analyze_e1(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    e1_rows = [row for row in rows if "e1" in row.get("experiment_id", "").lower()]
    summary = _aggregate(
        e1_rows,
        keys=("update_target", "cache_strategy"),
        metrics=(
            "task_score",
            "end_to_end_latency",
            "cache_bytes",
            "total_cache_bytes",
            "cache_hit",
            "evicted_cache_entries",
        ),
    )
    csv_path = output_dir / "e1_cache_cost.csv"
    _write_dicts(csv_path, summary)
    md_path = output_dir / "e1_cache_cost.md"
    md_path.write_text(
        _markdown_table(
            "E1 static-adapter cache cost",
            summary,
            columns=(
                "update_target",
                "cache_strategy",
                "task_score_mean",
                "end_to_end_latency_mean",
                "cache_bytes_mean",
                "total_cache_bytes_mean",
                "cache_hit_mean",
                "evicted_cache_entries_mean",
            ),
        ),
        encoding="utf-8",
    )
    svg_path = output_dir / "e1_memory_latency.svg"
    svg_path.write_text(
        _scatter_svg(
            summary,
            x_field="total_cache_bytes_mean",
            y_field="end_to_end_latency_mean",
            label_fields=("cache_strategy", "update_target"),
            title="E1 total cache memory versus end-to-end latency",
            x_label="total cache bytes",
            y_label="end-to-end latency",
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path, svg_path]


def _analyze_e2(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    thresholds: StudyThresholds,
) -> list[Path]:
    e2_rows = [row for row in rows if "e2" in row.get("experiment_id", "").lower()]
    enriched = _with_full_task_drop(e2_rows)
    summary = _aggregate(
        enriched,
        keys=("update_target", "cache_strategy", "version_gap"),
        metrics=(
            "task_score",
            "task_drop_vs_full",
            "logits_kl",
            "top1_agreement",
            "attention_shift",
            "end_to_end_latency",
            "strategy_flops",
            "flops_fraction",
        ),
    )
    csv_path = output_dir / "e2_version_drift.csv"
    _write_dicts(csv_path, summary)

    boundaries: list[dict[str, object]] = []
    groups = _group(enriched, ("update_target", "cache_strategy"))
    for (target, strategy), records in sorted(groups.items()):
        candidates = sorted(records, key=lambda row: _number(row, "version_gap"))
        failed = [row for row in candidates if not _safe(row, thresholds=thresholds)]
        first = failed[0] if failed else None
        boundaries.append(
            {
                "update_target": target,
                "cache_strategy": strategy,
                "first_unsafe_version_gap": int(_number(first, "version_gap")) if first else -1,
                "first_unsafe_update_norm": _number(first, "update_norm_since_cache") if first else 0.0,
                "task_drop_vs_full": _number(first, "task_drop_vs_full") if first else 0.0,
                "logits_kl": _number(first, "logits_kl") if first else 0.0,
                "top1_agreement": _number(first, "top1_agreement") if first else 1.0,
            }
        )
    boundary_csv = output_dir / "e2_first_boundary.csv"
    _write_dicts(boundary_csv, boundaries)
    boundary_md = output_dir / "e2_first_boundary.md"
    boundary_md.write_text(
        _markdown_table(
            "E2 first safety-boundary crossing",
            boundaries,
            columns=(
                "update_target",
                "cache_strategy",
                "first_unsafe_version_gap",
                "first_unsafe_update_norm",
                "task_drop_vs_full",
                "logits_kl",
                "top1_agreement",
            ),
        ),
        encoding="utf-8",
    )
    svg_path = output_dir / "e2_task_drop_by_gap.svg"
    svg_path.write_text(
        _line_svg(
            summary,
            x_field="version_gap",
            y_field="task_drop_vs_full_mean",
            series_fields=("cache_strategy", "update_target"),
            title="E2 task drop versus cache-version gap",
            x_label="version gap",
            y_label="task drop versus full recompute",
        ),
        encoding="utf-8",
    )
    return [csv_path, boundary_csv, boundary_md, svg_path]


def _analyze_e3(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    thresholds: StudyThresholds,
) -> list[Path]:
    from ttt_cache_lab.experiments.failure_map import FailureThresholds, generate_failure_map

    e3_rows = [row for row in rows if "e3" in row.get("experiment_id", "").lower()]
    source = output_dir / "e3_records.csv"
    _write_raw_rows(source, e3_rows)
    failure_dir = output_dir / "e3_failure_map"
    generate_failure_map(
        source,
        failure_dir,
        thresholds=FailureThresholds(
            safe_kl=thresholds.safe_kl,
            safe_top1=thresholds.safe_top1,
            safe_task_drop=thresholds.safe_task_drop,
        ),
    )
    return [source, *sorted(failure_dir.iterdir())]


def _analyze_e4(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    thresholds: StudyThresholds,
) -> list[Path]:
    from ttt_cache_lab.experiments.pareto import generate_pareto

    e4_rows = _with_full_task_drop(
        [row for row in rows if "e4" in row.get("experiment_id", "").lower()]
    )
    groups = _group(e4_rows, ("update_target", "cache_strategy"))
    summary: list[dict[str, object]] = []
    for (target, strategy), records in sorted(groups.items()):
        full_latency = _reference_mean(records, e4_rows, "end_to_end_latency")
        latency = _mean(records, "end_to_end_latency")
        summary.append(
            {
                "update_target": target,
                "cache_strategy": strategy,
                "count": len(records),
                "task_score_mean": _mean(records, "task_score"),
                "task_drop_vs_full_mean": _mean(records, "task_drop_vs_full"),
                "logits_kl_mean": _mean(records, "logits_kl"),
                "top1_agreement_mean": _mean(records, "top1_agreement"),
                "attention_shift_mean": _mean(records, "attention_shift"),
                "end_to_end_latency_mean": latency,
                "speedup_vs_full": full_latency / latency if latency > 0.0 else 0.0,
                "total_cache_bytes_mean": _mean(records, "total_cache_bytes"),
                "flops_fraction_mean": _mean(records, "flops_fraction"),
                "safe_rate": sum(
                    1.0 for row in records if _safe(row, thresholds=thresholds)
                )
                / len(records),
                "false_safe_rate": sum(_bool(row, "false_safe") for row in records)
                / len(records),
            }
        )
    csv_path = output_dir / "e4_planner_comparison.csv"
    _write_dicts(csv_path, summary)
    md_path = output_dir / "e4_planner_comparison.md"
    md_path.write_text(
        _markdown_table(
            "E4 planner quality-cost comparison",
            summary,
            columns=(
                "update_target",
                "cache_strategy",
                "task_score_mean",
                "task_drop_vs_full_mean",
                "logits_kl_mean",
                "top1_agreement_mean",
                "attention_shift_mean",
                "end_to_end_latency_mean",
                "speedup_vs_full",
                "total_cache_bytes_mean",
                "flops_fraction_mean",
                "safe_rate",
                "false_safe_rate",
            ),
        ),
        encoding="utf-8",
    )
    svg_path = output_dir / "e4_quality_cost.svg"
    svg_path.write_text(
        _scatter_svg(
            summary,
            x_field="end_to_end_latency_mean",
            y_field="task_score_mean",
            label_fields=("cache_strategy", "update_target"),
            title="E4 planner quality versus end-to-end latency",
            x_label="end-to-end latency",
            y_label="task score",
        ),
        encoding="utf-8",
    )
    pareto_source = output_dir / "e4_records.csv"
    _write_raw_rows(pareto_source, e4_rows)
    generate_pareto(pareto_source, output_dir / "e4_pareto")
    return [
        csv_path,
        md_path,
        svg_path,
        pareto_source,
        *sorted((output_dir / "e4_pareto").iterdir()),
    ]


def _reference_mean(
    records: list[dict[str, str]],
    all_rows: list[dict[str, str]],
    field: str,
) -> float:
    targets = {row.get("update_target", "") for row in records}
    references = [
        row
        for row in all_rows
        if row.get("update_target", "") in targets
        and row.get("cache_strategy", "") == "full_recompute"
    ]
    return _mean(references, field)


def _analyze_e5(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    thresholds: StudyThresholds,
) -> list[Path]:
    e5_rows = _with_full_task_drop(
        [row for row in rows if "e5" in row.get("experiment_id", "").lower()]
    )
    groups = _group(
        e5_rows,
        ("cache_strategy", "lora_rank", "configured_update_norm", "version_gap"),
    )
    safe_rows: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        strategy, rank, update_norm, gap = key
        safe_rate = sum(1.0 for row in records if _safe(row, thresholds=thresholds)) / len(records)
        safe_rows.append(
            {
                "cache_strategy": strategy,
                "lora_rank": rank,
                "update_norm": update_norm,
                "version_gap": gap,
                "count": len(records),
                "safe_rate": safe_rate,
                "task_drop_vs_full_mean": _mean(records, "task_drop_vs_full"),
                "logits_kl_mean": _mean(records, "logits_kl"),
                "top1_agreement_mean": _mean(records, "top1_agreement"),
                "attention_shift_mean": _mean(records, "attention_shift"),
                "flops_fraction_mean": _mean(records, "flops_fraction"),
            }
        )
    csv_path = output_dir / "e5_safe_region.csv"
    _write_dicts(csv_path, safe_rows)
    md_path = output_dir / "e5_safe_region.md"
    md_path.write_text(
        _markdown_table(
            "E5 rank and update-norm safe region",
            safe_rows,
            columns=(
                "cache_strategy",
                "lora_rank",
                "update_norm",
                "version_gap",
                "safe_rate",
                "task_drop_vs_full_mean",
                "logits_kl_mean",
                "top1_agreement_mean",
                "attention_shift_mean",
                "flops_fraction_mean",
            ),
        ),
        encoding="utf-8",
    )
    svg_path = output_dir / "e5_safe_region_heatmap.svg"
    svg_path.write_text(
        _matrix_svg(
            safe_rows,
            row_field="lora_rank",
            column_field="update_norm",
            value_field="safe_rate",
            facet_field="cache_strategy",
            title="E5 safe rate by LoRA rank and update norm",
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path, svg_path]


def _analyze_e6(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    e6_rows = [row for row in rows if "e6" in row.get("experiment_id", "").lower()]
    for row in e6_rows:
        if not row.get("context_length"):
            row["context_length"] = row.get("sweep.data.context_length", "0")
        if not row.get("model_name"):
            row["model_name"] = row.get("sweep.model.model_name_or_path", "") or "unknown"
    summary = _aggregate(
        e6_rows,
        keys=("model_name", "model_num_layers", "context_length", "cache_strategy"),
        metrics=(
            "task_score",
            "logits_kl",
            "attention_shift",
            "end_to_end_latency",
            "throughput_tokens_per_s",
            "total_cache_bytes",
            "strategy_flops",
            "flops_fraction",
        ),
    )
    csv_path = output_dir / "e6_context_model_scaling.csv"
    _write_dicts(csv_path, summary)
    md_path = output_dir / "e6_context_model_scaling.md"
    md_path.write_text(
        _markdown_table(
            "E6 context and model-scale results",
            summary,
            columns=(
                "model_name",
                "model_num_layers",
                "context_length",
                "cache_strategy",
                "task_score_mean",
                "logits_kl_mean",
                "attention_shift_mean",
                "end_to_end_latency_mean",
                "throughput_tokens_per_s_mean",
                "total_cache_bytes_mean",
                "strategy_flops_mean",
                "flops_fraction_mean",
            ),
        ),
        encoding="utf-8",
    )
    svg_path = output_dir / "e6_latency_by_context.svg"
    svg_path.write_text(
        _line_svg(
            summary,
            x_field="context_length",
            y_field="end_to_end_latency_mean",
            series_fields=("model_name", "cache_strategy"),
            title="E6 end-to-end latency by context and model",
            x_label="context length",
            y_label="end-to-end latency",
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path, svg_path]


def _analyze_e7(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    thresholds: StudyThresholds,
) -> list[Path]:
    e7_rows = _with_full_task_drop(
        [row for row in rows if "e7" in row.get("experiment_id", "").lower()]
    )
    groups = _group(e7_rows, ("update_target", "cache_strategy"))
    boundaries: list[dict[str, object]] = []
    for (target, strategy), records in sorted(groups.items()):
        points = sorted(
            records,
            key=lambda row: (
                _number(row, "version_gap"),
                _number(row, "update_norm_since_cache"),
            ),
        )
        unsafe = [row for row in points if not _safe(row, thresholds=thresholds)]
        first = unsafe[0] if unsafe else None
        boundaries.append(
            {
                "update_target": target,
                "cache_strategy": strategy,
                "tested_points": len(points),
                "unsafe_points": len(unsafe),
                "first_unsafe_version_gap": int(_number(first, "version_gap")) if first else -1,
                "first_unsafe_update_norm": _number(first, "update_norm_since_cache") if first else 0.0,
                "first_unsafe_kl": _number(first, "logits_kl") if first else 0.0,
                "first_unsafe_task_drop": _number(first, "task_drop_vs_full") if first else 0.0,
                "first_unsafe_attention_shift": _number(first, "attention_shift") if first else 0.0,
                "first_unsafe_flops_fraction": _number(first, "flops_fraction") if first else 0.0,
                "false_safe_rate": sum(_bool(row, "false_safe") for row in points) / len(points),
            }
        )
    csv_path = output_dir / "e7_failure_boundary.csv"
    _write_dicts(csv_path, boundaries)
    md_path = output_dir / "e7_failure_boundary.md"
    md_path.write_text(
        _markdown_table(
            "E7 failure-boundary sweep",
            boundaries,
            columns=(
                "update_target",
                "cache_strategy",
                "tested_points",
                "unsafe_points",
                "first_unsafe_version_gap",
                "first_unsafe_update_norm",
                "first_unsafe_kl",
                "first_unsafe_task_drop",
                "first_unsafe_attention_shift",
                "first_unsafe_flops_fraction",
                "false_safe_rate",
            ),
        ),
        encoding="utf-8",
    )
    svg_path = output_dir / "e7_false_safe_rate.svg"
    svg_path.write_text(
        _bar_svg(
            boundaries,
            value_field="false_safe_rate",
            label_fields=("cache_strategy", "update_target"),
            title="E7 false-safe rate by ablation and target",
            y_label="false-safe rate",
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path, svg_path]


def _with_full_task_drop(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    full_scores: dict[tuple[tuple[str, str], ...], float] = {}
    for row in rows:
        if row.get("cache_strategy") == "full_recompute":
            full_scores[_reference_key(row)] = _number(row, "task_score")
    enriched: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        full_score = full_scores.get(_reference_key(row))
        if full_score is not None:
            item["task_drop_vs_full"] = str(full_score - _number(row, "task_score"))
        else:
            item["task_drop_vs_full"] = row.get("task_drop_vs_full", "0")
        enriched.append(item)
    return enriched


def _reference_key(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    fields: list[str] = [
        field
        for field in (
            "run_name",
            "experiment_id",
            "sample_id",
            "update_target",
            "adapter_id",
            "adapter_version",
            "lora_rank",
            "update_mode",
            "context_length",
            "model_name",
        )
        if field in row
    ]
    fields.extend(sorted(field for field in row if field.startswith("sweep.")))
    return tuple((field, row.get(field, "")) for field in fields)


def _safe(row: dict[str, str], *, thresholds: StudyThresholds) -> bool:
    return (
        _number(row, "logits_kl") <= thresholds.safe_kl
        and _number(row, "top1_agreement", default=1.0) >= thresholds.safe_top1
        and _number(row, "task_drop_vs_full") <= thresholds.safe_task_drop
        and not _bool(row, "false_safe")
    )


def _aggregate(
    rows: list[dict[str, str]],
    *,
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    groups = _group(rows, keys)
    output: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        item: dict[str, object] = dict(zip(keys, key, strict=True))
        item["count"] = len(records)
        for metric in metrics:
            item[f"{metric}_mean"] = _mean(records, metric)
        output.append(item)
    return output


def _group(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_raw_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_dicts(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(
    title: str,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: tuple[str, ...],
) -> str:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = [_number(row, field) for row in rows if row.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _number(row: dict[str, str] | None, field: str, *, default: float = 0.0) -> float:
    if row is None:
        return default
    raw = row.get(field)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool(row: dict[str, str], field: str) -> bool:
    return row.get(field, "false").lower() in {"true", "1", "yes"}


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _line_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    x_field: str,
    y_field: str,
    series_fields: tuple[str, ...],
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    series: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        try:
            x = _value_number(row.get(x_field, 0))
            y = _value_number(row.get(y_field, 0))
        except (TypeError, ValueError):
            continue
        series[tuple(str(row.get(field, "")) for field in series_fields)].append((x, y))
    return _xy_svg(series, title=title, x_label=x_label, y_label=y_label, connect=True)


def _scatter_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    x_field: str,
    y_field: str,
    label_fields: tuple[str, ...],
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    series = {
        tuple(str(row.get(field, "")) for field in label_fields): [
            (_value_number(row.get(x_field, 0)), _value_number(row.get(y_field, 0)))
        ]
        for row in rows
    }
    return _xy_svg(series, title=title, x_label=x_label, y_label=y_label, connect=False)


def _xy_svg(
    series: dict[tuple[str, ...], list[tuple[float, float]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    connect: bool,
) -> str:
    width, height, margin = 960, 560, 70
    points = [point for values in series.values() for point in values]
    if not points:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymax = ymin + 1.0

    def sx(value: float) -> float:
        return margin + (value - xmin) / (xmax - xmin) * (width - 2 * margin)

    def sy(value: float) -> float:
        return height - margin - (value - ymin) / (ymax - ymin) * (height - 2 * margin)

    lines = _svg_axes(width, height, margin, title, x_label, y_label)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for index, (label, values) in enumerate(sorted(series.items())):
        values = sorted(values)
        color = palette[index % len(palette)]
        if connect and len(values) > 1:
            polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
            lines.append(f"<polyline points='{polyline}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in values:
            lines.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='{color}'/>")
        lines.append(
            f"<text x='{width - margin + 8}' y='{margin + 16 * index}' font-size='10' fill='{color}'>"
            f"{_escape(' / '.join(label))}</text>"
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _bar_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    value_field: str,
    label_fields: tuple[str, ...],
    title: str,
    y_label: str,
) -> str:
    width, margin = 1000, 70
    height = max(300, margin * 2 + 28 * len(rows))
    values = [_value_number(row.get(value_field, 0)) for row in rows]
    maximum = max(values or [1.0]) or 1.0
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='18'>{_escape(title)}</text>",
        (
            f"<text x='18' y='{height / 2}' transform='rotate(-90 18 {height / 2})' "
            f"font-size='12'>{_escape(y_label)}</text>"
        ),
    ]
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        y = margin + index * 28
        label = " / ".join(str(row.get(field, "")) for field in label_fields)
        bar_width = (width - 420) * value / maximum
        lines.append(f"<text x='390' y='{y + 16}' text-anchor='end' font-size='11'>{_escape(label)}</text>")
        lines.append(f"<rect x='400' y='{y}' width='{bar_width:.2f}' height='20' fill='#4c78a8'/>")
        lines.append(f"<text x='{405 + bar_width:.2f}' y='{y + 15}' font-size='11'>{value:.4g}</text>")
    lines.append("</svg>")
    return "\n".join(lines)


def _matrix_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    row_field: str,
    column_field: str,
    value_field: str,
    facet_field: str,
    title: str,
) -> str:
    facets = sorted({str(row.get(facet_field, "")) for row in rows})
    row_values = sorted({str(row.get(row_field, "")) for row in rows}, key=_sortable_number)
    columns = sorted({str(row.get(column_field, "")) for row in rows}, key=_sortable_number)
    cell_w, cell_h = 92, 32
    left, top = 220, 60
    facet_h = cell_h * max(1, len(row_values)) + 45
    width = left + cell_w * max(1, len(columns)) + 40
    height = top + facet_h * max(1, len(facets))
    values = {
        (
            str(row.get(facet_field, "")),
            str(row.get(row_field, "")),
            str(row.get(column_field, "")),
        ): _value_number(row.get(value_field, 0))
        for row in rows
    }
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width / 2}' y='25' text-anchor='middle' font-size='17'>{_escape(title)}</text>",
    ]
    for facet_index, facet in enumerate(facets):
        base_y = top + facet_index * facet_h
        lines.append(f"<text x='10' y='{base_y + 18}' font-size='13'>{_escape(facet)}</text>")
        for column_index, column in enumerate(columns):
            x = left + column_index * cell_w
            lines.append(
                f"<text x='{x + cell_w / 2}' y='{base_y + 18}' text-anchor='middle' "
                f"font-size='11'>{_escape(column)}</text>"
            )
        for row_index, row_value in enumerate(row_values):
            y = base_y + 25 + row_index * cell_h
            lines.append(f"<text x='{left - 8}' y='{y + 21}' text-anchor='end' font-size='11'>rank {row_value}</text>")
            for column_index, column in enumerate(columns):
                value = values.get((facet, row_value, column), 0.0)
                shade = int(255 - 170 * max(0.0, min(1.0, value)))
                x = left + column_index * cell_w
                lines.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' "
                    f"fill='rgb({shade},255,{shade})' stroke='#ddd'/>"
                )
                lines.append(
                    f"<text x='{x + cell_w / 2}' y='{y + 21}' text-anchor='middle' "
                    f"font-size='11'>{value:.2f}</text>"
                )
    lines.append("</svg>")
    return "\n".join(lines)


def _svg_axes(width: int, height: int, margin: int, title: str, x_label: str, y_label: str) -> list[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='18'>{_escape(title)}</text>",
        (
            f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' "
            f"y2='{height - margin}' stroke='black'/>"
        ),
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black'/>",
        (
            f"<text x='{width / 2}' y='{height - 16}' text-anchor='middle' "
            f"font-size='12'>{_escape(x_label)}</text>"
        ),
        (
            f"<text x='18' y='{height / 2}' transform='rotate(-90 18 {height / 2})' "
            f"text-anchor='middle' font-size='12'>{_escape(y_label)}</text>"
        ),
    ]


def _value_number(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _sortable_number(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
