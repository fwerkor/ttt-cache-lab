from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ttt_cache_lab.experiments.conditions import (
    condition_fields,
    reference_key,
    with_full_reference_metrics,
)


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
    if any("e8" in experiment for experiment in experiment_ids):
        artifacts.extend(_analyze_e8(rows, output_dir))
    if any(any(tag in experiment for tag in ("e2", "e3", "e4", "e5", "e6", "e7")) for experiment in experiment_ids):
        artifacts.extend(_analyze_adaptation_effect(rows, output_dir))
    return artifacts


def _analyze_e1(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    e1_rows = [row for row in rows if "e1" in row.get("experiment_id", "").lower()]
    group_fields = condition_fields(e1_rows, "update_target", "cache_strategy")
    summary = _aggregate(
        e1_rows,
        keys=group_fields,
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
                *group_fields,
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
    enriched = with_full_reference_metrics(e2_rows)
    summary_fields = condition_fields(
        enriched,
        "update_target",
        "cache_strategy",
        "adapter_version",
        "cached_version",
        "version_gap",
    )
    summary = _aggregate(
        enriched,
        keys=summary_fields,
        metrics=(
            "task_score",
            "task_drop_vs_full",
            "task_delta_vs_base",
            "task_regression_vs_base",
            "below_base",
            "adaptation_gain_available",
            "adaptation_gain_retention",
            "positive_adaptation_gain_available",
            "positive_adaptation_gain_retention",
            "positive_adaptation_gain_reference",
            "positive_adaptation_gain_retained",
            "lost_positive_adaptation_gain",
            "logits_kl",
            "top1_agreement",
            "attention_shift",
            "attention_metric_available",
            "end_to_end_latency",
            "strategy_flops",
            "flops_fraction",
        ),
    )
    for row in summary:
        available = float(
            str(row.get("positive_adaptation_gain_available_mean", 0.0))
        )
        unconditional = float(
            str(row.get("positive_adaptation_gain_retention_mean", 0.0))
        )
        reference = float(
            str(row.get("positive_adaptation_gain_reference_mean", 0.0))
        )
        retained = float(
            str(row.get("positive_adaptation_gain_retained_mean", 0.0))
        )
        row["positive_adaptation_gain_retention_conditional_mean"] = (
            unconditional / available if available > 0.0 else 0.0
        )
        row["positive_adaptation_gain_retention_weighted"] = (
            retained / reference if reference > 0.0 else 0.0
        )
    csv_path = output_dir / "e2_version_drift.csv"
    _write_dicts(csv_path, summary)

    boundaries: list[dict[str, object]] = []
    boundary_fields = condition_fields(enriched, "update_target", "cache_strategy")
    groups = _group(enriched, boundary_fields)
    for key, records in sorted(groups.items()):
        dimensions = dict(zip(boundary_fields, key, strict=True))
        candidates = sorted(records, key=lambda row: _number(row, "version_gap"))
        failed = [row for row in candidates if not _safe(row, thresholds=thresholds)]
        first = failed[0] if failed else None
        boundaries.append(
            {
                **dimensions,
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
                *boundary_fields,
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
            series_fields=condition_fields(enriched, "cache_strategy", "update_target"),
            title="E2 task drop versus cache-version gap",
            x_label="version gap",
            y_label="task drop versus full recompute",
        ),
        encoding="utf-8",
    )
    base_delta_svg = output_dir / "e2_task_delta_vs_base_by_gap.svg"
    base_delta_svg.write_text(
        _line_svg(
            summary,
            x_field="version_gap",
            y_field="task_delta_vs_base_mean",
            series_fields=condition_fields(enriched, "cache_strategy", "update_target"),
            title="E2 task change versus pre-update model",
            x_label="version gap",
            y_label="task score minus pre-update baseline",
        ),
        encoding="utf-8",
    )
    gain_retention_svg = output_dir / "e2_adaptation_gain_retention_by_gap.svg"
    gain_retention_svg.write_text(
        _line_svg(
            summary,
            x_field="version_gap",
            y_field="positive_adaptation_gain_retention_weighted",
            series_fields=condition_fields(enriched, "cache_strategy", "update_target"),
            title="E2 retained positive adaptation gain versus cache-version gap",
            x_label="version gap",
            y_label="retained fraction of fresh-cache adaptation gain",
        ),
        encoding="utf-8",
    )
    return [
        csv_path,
        boundary_csv,
        boundary_md,
        svg_path,
        base_delta_svg,
        gain_retention_svg,
    ]


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

    e4_rows = with_full_reference_metrics(
        [row for row in rows if "e4" in row.get("experiment_id", "").lower()]
    )
    group_fields = condition_fields(e4_rows, "update_target", "cache_strategy")
    groups = _group(e4_rows, group_fields)
    summary: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        dimensions = dict(zip(group_fields, key, strict=True))
        latency = _mean(records, "end_to_end_latency")
        summary.append(
            {
                **dimensions,
                "count": len(records),
                "task_score_mean": _mean(records, "task_score"),
                "task_drop_vs_full_mean": _mean(records, "task_drop_vs_full"),
                "logits_kl_mean": _mean(records, "logits_kl"),
                "top1_agreement_mean": _mean(records, "top1_agreement"),
                "attention_shift_mean": _mean(records, "attention_shift"),
                "attention_metric_available_rate": _mean(
                    records, "attention_metric_available"
                ),
                "end_to_end_latency_mean": latency,
                "speedup_vs_full": _mean(records, "speedup_vs_full"),
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
                *group_fields,
                "task_score_mean",
                "task_drop_vs_full_mean",
                "logits_kl_mean",
                "top1_agreement_mean",
                "attention_shift_mean",
                "attention_metric_available_rate",
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


def _analyze_e5(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    thresholds: StudyThresholds,
) -> list[Path]:
    e5_rows = with_full_reference_metrics(
        [row for row in rows if "e5" in row.get("experiment_id", "").lower()]
    )
    group_fields = condition_fields(
        e5_rows,
        "update_target",
        "cache_strategy",
        "strategy_mode",
        "adapter_version",
        "version_gap",
    )
    groups = _group(e5_rows, group_fields)
    safe_rows: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        dimensions = dict(zip(group_fields, key, strict=True))
        safe_rate = sum(1.0 for row in records if _safe(row, thresholds=thresholds)) / len(records)
        facet = " / ".join(
            str(dimensions.get(field, ""))
            for field in ("update_target", "cache_strategy", "strategy_mode", "version_gap")
        )
        safe_rows.append(
            {
                **dimensions,
                "facet": facet,
                "count": len(records),
                "safe_rate": safe_rate,
                "task_drop_vs_full_mean": _mean(records, "task_drop_vs_full"),
                "relative_error_mean": _mean(records, "relative_error"),
                "hidden_relative_error_mean": _mean(records, "hidden_relative_error"),
                "logits_kl_mean": _mean(records, "logits_kl"),
                "top1_agreement_mean": _mean(records, "top1_agreement"),
                "cache_maintenance_latency_mean": _mean(
                    records, "cache_maintenance_latency"
                ),
                "strategy_flops_mean": _mean(records, "strategy_flops"),
                "cache_bytes_mean": _mean(records, "cache_bytes"),
                "physical_cache_bytes_mean": _mean(records, "physical_cache_bytes"),
                "strategy_available_rate": _mean(records, "strategy_available"),
                "fallback_rate": 1.0 - _mean(records, "strategy_available"),
                "attention_shift_mean": _mean(records, "attention_shift"),
                "attention_metric_available_rate": _mean(
                    records, "attention_metric_available"
                ),
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
                *group_fields,
                "safe_rate",
                "task_drop_vs_full_mean",
                "relative_error_mean",
                "hidden_relative_error_mean",
                "logits_kl_mean",
                "top1_agreement_mean",
                "cache_maintenance_latency_mean",
                "strategy_flops_mean",
                "cache_bytes_mean",
                "physical_cache_bytes_mean",
                "strategy_available_rate",
                "fallback_rate",
                "attention_shift_mean",
                "attention_metric_available_rate",
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
            column_field="configured_update_norm",
            value_field="safe_rate",
            facet_field="facet",
            title="E5 safe rate by LoRA rank and update norm",
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path, svg_path]


def _analyze_e6(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    e6_rows = with_full_reference_metrics(
        [row for row in rows if "e6" in row.get("experiment_id", "").lower()]
    )
    for row in e6_rows:
        if not row.get("context_length"):
            row["context_length"] = row.get("sweep.data.context_length", "0")
        if not row.get("model_name"):
            row["model_name"] = row.get("sweep.model.model_name_or_path", "") or "unknown"
    group_fields = condition_fields(
        e6_rows,
        "update_target",
        "cache_strategy",
        "adapter_version",
        "version_gap",
    )
    summary = _aggregate(
        e6_rows,
        keys=group_fields,
        metrics=(
            "task_score",
            "task_drop_vs_full",
            "speedup_vs_full",
            "relative_error",
            "logits_kl",
            "attention_shift",
            "attention_metric_available",
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
                *group_fields,
                "task_score_mean",
                "task_drop_vs_full_mean",
                "speedup_vs_full_mean",
                "relative_error_mean",
                "logits_kl_mean",
                "attention_shift_mean",
                "attention_metric_available_mean",
                "end_to_end_latency_mean",
                "throughput_tokens_per_s_mean",
                "total_cache_bytes_mean",
                "strategy_flops_mean",
                "flops_fraction_mean",
            ),
        ),
        encoding="utf-8",
    )
    series_fields = condition_fields(
        e6_rows, "model_name", "cache_strategy", "update_target"
    )
    svg_path = output_dir / "e6_latency_by_context.svg"
    svg_path.write_text(
        _line_svg(
            summary,
            x_field="context_length",
            y_field="end_to_end_latency_mean",
            series_fields=series_fields,
            title="E6 end-to-end latency by context and model",
            x_label="context length",
            y_label="end-to-end latency",
        ),
        encoding="utf-8",
    )
    speedup_svg = output_dir / "e6_speedup_by_context.svg"
    speedup_svg.write_text(
        _line_svg(
            summary,
            x_field="context_length",
            y_field="speedup_vs_full_mean",
            series_fields=series_fields,
            title="E6 speedup versus full recompute by context",
            x_label="context length",
            y_label="speedup versus full recompute",
        ),
        encoding="utf-8",
    )
    task_drop_svg = output_dir / "e6_task_drop_by_context.svg"
    task_drop_svg.write_text(
        _line_svg(
            summary,
            x_field="context_length",
            y_field="task_drop_vs_full_mean",
            series_fields=series_fields,
            title="E6 task drop versus full recompute by context",
            x_label="context length",
            y_label="task drop versus full recompute",
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path, svg_path, speedup_svg, task_drop_svg]


def _analyze_e7(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    thresholds: StudyThresholds,
) -> list[Path]:
    e7_rows = with_full_reference_metrics(
        [row for row in rows if "e7" in row.get("experiment_id", "").lower()]
    )
    group_fields = condition_fields(e7_rows, "update_target", "cache_strategy")
    groups = _group(e7_rows, group_fields)
    boundaries: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        dimensions = dict(zip(group_fields, key, strict=True))
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
                **dimensions,
                "tested_points": len(points),
                "unsafe_points": len(unsafe),
                "first_unsafe_version_gap": int(_number(first, "version_gap")) if first else -1,
                "first_unsafe_update_norm": _number(first, "update_norm_since_cache") if first else 0.0,
                "first_unsafe_kl": _number(first, "logits_kl") if first else 0.0,
                "first_unsafe_task_drop": _number(first, "task_drop_vs_full") if first else 0.0,
                "first_unsafe_attention_shift": _optional_number(first, "attention_shift"),
                "attention_metric_available_rate": _mean(
                    points, "attention_metric_available"
                ),
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
                *group_fields,
                "tested_points",
                "unsafe_points",
                "first_unsafe_version_gap",
                "first_unsafe_update_norm",
                "first_unsafe_kl",
                "first_unsafe_task_drop",
                "first_unsafe_attention_shift",
                "attention_metric_available_rate",
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
    effect_csv, effect_md = _write_e7_ablation_effect(e7_rows, output_dir)
    return [csv_path, md_path, svg_path, effect_csv, effect_md]


def _write_e7_ablation_effect(
    rows: list[dict[str, str]], output_dir: Path
) -> tuple[Path, Path]:
    adaptive = {
        reference_key(row): row
        for row in rows
        if row.get("cache_strategy") == "adaptive"
    }
    paired: list[dict[str, str]] = []
    for row in rows:
        strategy = row.get("cache_strategy", "")
        if not strategy.startswith("adaptive_no_"):
            continue
        baseline = adaptive.get(reference_key(row))
        if baseline is None:
            raise ValueError(
                f"Missing adaptive reference for E7 ablation {strategy!r}: "
                f"{reference_key(row)!r}"
            )
        item = dict(row)
        item["task_score_delta_vs_adaptive"] = str(
            _number(row, "task_score") - _number(baseline, "task_score")
        )
        item["latency_delta_vs_adaptive"] = str(
            _number(row, "end_to_end_latency")
            - _number(baseline, "end_to_end_latency")
        )
        item["false_safe_delta_vs_adaptive"] = str(
            float(_bool(row, "false_safe")) - float(_bool(baseline, "false_safe"))
        )
        item["refresh_count_delta_vs_adaptive"] = str(
            _number(row, "refresh_count") - _number(baseline, "refresh_count")
        )
        item["flops_fraction_delta_vs_adaptive"] = str(
            _number(row, "flops_fraction") - _number(baseline, "flops_fraction")
        )
        paired.append(item)
    fields = condition_fields(paired, "update_target", "cache_strategy")
    summary = _aggregate(
        paired,
        keys=fields,
        metrics=(
            "task_score_delta_vs_adaptive",
            "latency_delta_vs_adaptive",
            "false_safe_delta_vs_adaptive",
            "refresh_count_delta_vs_adaptive",
            "flops_fraction_delta_vs_adaptive",
        ),
    )
    csv_path = output_dir / "e7_ablation_effect.csv"
    _write_dicts(csv_path, summary)
    md_path = output_dir / "e7_ablation_effect.md"
    md_path.write_text(
        _markdown_table(
            "E7 paired ablation effects versus the full adaptive planner",
            summary,
            columns=(
                *fields,
                "count",
                "task_score_delta_vs_adaptive_mean",
                "latency_delta_vs_adaptive_mean",
                "false_safe_delta_vs_adaptive_mean",
                "refresh_count_delta_vs_adaptive_mean",
                "flops_fraction_delta_vs_adaptive_mean",
            ),
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def _analyze_e8(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    e8_rows = [row for row in rows if "e8" in row.get("experiment_id", "").lower()]
    group_fields = condition_fields(e8_rows, "update_target", "cache_strategy")
    groups = _group(e8_rows, group_fields)
    summary: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        dimensions = dict(zip(group_fields, key, strict=True))
        latencies = sorted(
            _number(row, "latency_p50", default=_number(row, "end_to_end_latency"))
            for row in records
        )
        summary.append(
            {
                **dimensions,
                "count": len(records),
                "task_score_mean": _mean(records, "task_score"),
                "latency_p50": _quantile(latencies, 0.50),
                "latency_p95": _quantile(latencies, 0.95),
                "throughput_mean": _mean(records, "throughput_tokens_per_s"),
                "cache_hit_rate": _mean(records, "cache_hit"),
                "false_safe_rate": _mean(records, "false_safe"),
                "peak_cache_entries": max((_number(row, "cache_entry_count") for row in records), default=0.0),
                "peak_cache_bytes": max((_number(row, "total_cache_bytes") for row in records), default=0.0),
                "evicted_cache_entries": max(
                    (_number(row, "evicted_cache_entries") for row in records), default=0.0
                ),
                "refresh_count_mean": _mean(records, "refresh_count"),
            }
        )
    csv_path = output_dir / "e8_cache_pressure.csv"
    _write_dicts(csv_path, summary)
    md_path = output_dir / "e8_cache_pressure.md"
    md_path.write_text(
        _markdown_table(
            "E8 sustained cache-pressure workload",
            summary,
            columns=(
                *group_fields,
                "count",
                "task_score_mean",
                "latency_p50",
                "latency_p95",
                "throughput_mean",
                "cache_hit_rate",
                "false_safe_rate",
                "peak_cache_entries",
                "peak_cache_bytes",
                "evicted_cache_entries",
                "refresh_count_mean",
            ),
        ),
        encoding="utf-8",
    )
    svg_path = output_dir / "e8_latency_cache_pressure.svg"
    svg_path.write_text(
        _scatter_svg(
            summary,
            x_field="peak_cache_bytes",
            y_field="latency_p95",
            label_fields=("model_name", "cache_strategy", "update_target"),
            title="E8 p95 latency under cache pressure",
            x_label="peak cache bytes",
            y_label="p95 latency",
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path, svg_path]


def _analyze_adaptation_effect(
    rows: list[dict[str, str]], output_dir: Path
) -> list[Path]:
    full_rows = [
        row
        for row in rows
        if row.get("cache_strategy") == "full_recompute"
        and any(tag in row.get("experiment_id", "").lower() for tag in ("e2", "e3", "e4", "e5", "e6", "e7"))
    ]
    fields = condition_fields(
        full_rows, "update_target", "adapter_version", "version_gap"
    )
    summary = _aggregate(
        full_rows,
        keys=fields,
        metrics=(
            "baseline_task_score",
            "full_task_score",
            "adaptation_gain_vs_base",
            "accumulated_update_norm",
            "accumulated_raw_update_norm",
            "update_scale",
            "adaptation_latency",
        ),
    )
    csv_path = output_dir / "adaptation_effect.csv"
    _write_dicts(csv_path, summary)
    md_path = output_dir / "adaptation_effect.md"
    md_path.write_text(
        _markdown_table(
            "Adaptation effectiveness and update diagnostics",
            summary,
            columns=(
                *fields,
                "count",
                "baseline_task_score_mean",
                "full_task_score_mean",
                "adaptation_gain_vs_base_mean",
                "accumulated_update_norm_mean",
                "accumulated_raw_update_norm_mean",
                "update_scale_mean",
                "adaptation_latency_mean",
            ),
        ),
        encoding="utf-8",
    )
    return [csv_path, md_path]


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


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = [_number(row, field) for row in rows if row.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _number(row: dict[str, str] | None, field: str, *, default: float = 0.0) -> float:
    if row is None:
        return default
    raw = row.get(field)
    if raw is None or raw == "":
        return default
    lowered = raw.lower()
    if lowered in {"true", "yes"}:
        return 1.0
    if lowered in {"false", "no"}:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return default


def _optional_number(row: dict[str, str] | None, field: str) -> float | None:
    if row is None:
        return None
    raw = row.get(field)
    if raw is None or raw == "":
        return None
    return _number(row, field)


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
