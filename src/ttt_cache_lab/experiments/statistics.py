from __future__ import annotations

import csv
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from ttt_cache_lab.experiments.conditions import available_fields, sweep_fields, with_full_reference_metrics

_DEFAULT_METRICS = (
    "task_score",
    "task_drop_vs_full",
    "logits_kl",
    "top1_agreement",
    "relative_error",
    "end_to_end_latency",
    "throughput_tokens_per_s",
    "total_cache_bytes",
    "flops_fraction",
    "cache_hit",
    "false_safe",
)
_GROUP_CANDIDATES = (
    "experiment_id",
    "model_name",
    "model_num_layers",
    "model_hidden_size",
    "context_length",
    "synthetic_difficulty",
    "prompt_format",
    "task_name",
    "task_family",
    "benchmark_name",
    "evaluation_partition",
    "dataset_split",
    "dataset_category",
    "update_target",
    "cache_strategy",
    "adapter_version",
    "cached_version",
    "version_gap",
    "lora_rank",
    "configured_update_norm",
    "update_mode",
    "norm_control",
)
_PAIR_CANDIDATES = tuple(field for field in _GROUP_CANDIDATES if field != "cache_strategy")


def generate_statistical_report(
    input_csv: Path,
    output_dir: Path,
    *,
    reference_strategy: str = "full_recompute",
    bootstrap_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2027,
) -> list[Path]:
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    enriched = with_full_reference_metrics(rows)
    aggregate_path = output_dir / "aggregate_confidence_intervals.csv"
    paired_path = output_dir / "paired_comparisons.csv"
    latency_path = output_dir / "latency_percentiles.csv"
    safety_path = output_dir / "safety_intervals.csv"
    markdown_path = output_dir / "statistical_summary.md"

    aggregate = _aggregate_intervals(
        enriched,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    paired = _paired_comparisons(
        enriched,
        reference_strategy=reference_strategy,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed + 1,
    )
    latency = _latency_percentiles(enriched)
    safety = _safety_intervals(enriched, confidence_level=confidence_level)
    _write_rows(aggregate_path, aggregate)
    _write_rows(paired_path, paired)
    _write_rows(latency_path, latency)
    _write_rows(safety_path, safety)
    markdown_path.write_text(
        _summary_markdown(
            aggregate,
            paired,
            confidence_level=confidence_level,
            reference_strategy=reference_strategy,
        ),
        encoding="utf-8",
    )
    return [aggregate_path, paired_path, latency_path, safety_path, markdown_path]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_fields(rows: Sequence[dict[str, str]], *, paired: bool = False) -> tuple[str, ...]:
    candidates = _PAIR_CANDIDATES if paired else _GROUP_CANDIDATES
    fields = list(available_fields(rows, candidates, include_sweeps=False))
    fields.extend(
        field
        for field in sweep_fields(rows)
        if field not in fields and field not in {"sweep.seed", "sweep.data.selection_seed"}
    )
    return tuple(fields)


def _aggregate_intervals(
    rows: Sequence[dict[str, str]],
    *,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    fields = _group_fields(rows)
    groups = _group(rows, fields)
    result: list[dict[str, object]] = []
    for group_index, (key, records) in enumerate(sorted(groups.items())):
        dimensions = dict(zip(fields, key, strict=True))
        clusters = _clusters(records)
        count_seeds = len({row.get("seed", "") for row in records})
        for metric in _DEFAULT_METRICS:
            values = [_number(row, metric) for row in records if _present(row, metric)]
            if not values:
                continue
            low, high = _cluster_bootstrap_interval(
                clusters,
                metric=metric,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed + group_index * 101 + len(result),
            )
            result.append(
                {
                    **dimensions,
                    "metric": metric,
                    "count_records": len(values),
                    "count_samples": len(clusters),
                    "count_seeds": count_seeds,
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return result


def _paired_comparisons(
    rows: Sequence[dict[str, str]],
    *,
    reference_strategy: str,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    reference = {
        _pair_identity(row): row
        for row in rows
        if row.get("cache_strategy") == reference_strategy
    }
    fields = _group_fields(rows, paired=True)
    candidate_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strategy = row.get("cache_strategy", "")
        if strategy == reference_strategy:
            continue
        key = tuple(row.get(field, "") for field in (*fields, "cache_strategy"))
        candidate_groups[key].append(row)

    result: list[dict[str, object]] = []
    for group_index, (key, records) in enumerate(sorted(candidate_groups.items())):
        dimensions = dict(zip((*fields, "cache_strategy"), key, strict=True))
        pairs = [(row, reference.get(_pair_identity(row))) for row in records]
        complete = [(row, ref) for row, ref in pairs if ref is not None]
        if not complete:
            continue
        derived = [
            {
                "cluster": _cluster_id(row),
                "task_score_delta": _number(row, "task_score") - _number(ref, "task_score"),
                "latency_delta": _latency(row) - _latency(ref),
                "speedup_vs_reference": _latency(ref) / _latency(row) if _latency(row) > 0.0 else 0.0,
                "false_safe": _number(row, "false_safe"),
            }
            for row, ref in complete
        ]
        for metric in ("task_score_delta", "latency_delta", "speedup_vs_reference", "false_safe"):
            values: list[float] = []
            clusters: dict[str, list[dict[str, str]]] = defaultdict(list)
            for item in derived:
                value = item[metric]
                if not isinstance(value, int | float):
                    raise TypeError(f"Derived metric {metric!r} is not numeric")
                numeric = float(value)
                values.append(numeric)
                clusters[str(item["cluster"])].append({metric: str(numeric)})
            low, high = _cluster_bootstrap_interval(
                clusters,
                metric=metric,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed + group_index * 101 + len(result),
            )
            result.append(
                {
                    **dimensions,
                    "reference_strategy": reference_strategy,
                    "metric": metric,
                    "pair_count": len(values),
                    "cluster_count": len(clusters),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return result


def _latency_percentiles(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    fields = _group_fields(rows)
    groups = _group(rows, fields)
    result: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        values = sorted(_latency(row) for row in records if _latency(row) > 0.0)
        if not values:
            continue
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "count": len(values),
                "p50": _quantile(values, 0.50),
                "p90": _quantile(values, 0.90),
                "p95": _quantile(values, 0.95),
                "p99": _quantile(values, 0.99),
                "maximum": values[-1],
            }
        )
    return result


def _safety_intervals(
    rows: Sequence[dict[str, str]],
    *,
    confidence_level: float,
) -> list[dict[str, object]]:
    fields = _group_fields(rows)
    groups = _group(rows, fields)
    result: list[dict[str, object]] = []
    for key, records in sorted(groups.items()):
        unsafe = sum(1 for row in records if _truthy(row.get("false_safe", "")))
        total = len(records)
        low, high = _wilson_interval(unsafe, total, confidence_level=confidence_level)
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "count": total,
                "false_safe_count": unsafe,
                "false_safe_rate": unsafe / total if total else 0.0,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return result


def _cluster_bootstrap_interval(
    clusters: dict[str, list[dict[str, str]]],
    *,
    metric: str,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    usable = {
        key: [_number(row, metric) for row in records if _present(row, metric)]
        for key, records in clusters.items()
    }
    usable = {key: values for key, values in usable.items() if values}
    if not usable:
        return 0.0, 0.0
    keys = sorted(usable)
    if len(keys) == 1:
        value = statistics.fmean(usable[keys[0]])
        return value, value
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(keys) for _ in keys]
        values = [value for key in sampled for value in usable[key]]
        estimates.append(statistics.fmean(values))
    alpha = (1.0 - confidence_level) / 2.0
    estimates.sort()
    return _quantile(estimates, alpha), _quantile(estimates, 1.0 - alpha)


def _clusters(records: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        result[_cluster_id(row)].append(row)
    return result


def _cluster_id(row: dict[str, str]) -> str:
    sample = row.get("dataset_sample_id") or row.get("sample_id", "")
    return f"{row.get('seed', '')}:{sample}"


def _pair_identity(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    fields = (
        "experiment_id",
        "dataset_sample_id",
        "sample_id",
        "seed",
        "task_name",
        "evaluation_partition",
        "model_name",
        "context_length",
        "synthetic_difficulty",
        "prompt_format",
        "update_target",
        "adapter_id",
        "adapter_version",
        "lora_rank",
        "configured_update_norm",
        "update_mode",
    )
    result = [(field, row.get(field, "")) for field in fields if field in row]
    result.extend((field, row.get(field, "")) for field in sweep_fields([row]))
    return tuple(result)


def _group(
    rows: Sequence[dict[str, str]],
    fields: Sequence[str],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in fields)].append(row)
    return groups


def _latency(row: dict[str, str]) -> float:
    for field in ("latency_p50", "end_to_end_latency", "latency_units"):
        if _present(row, field):
            return _number(row, field)
    return 0.0


def _present(row: dict[str, str], field: str) -> bool:
    return row.get(field) not in {None, ""}


def _number(row: dict[str, str], field: str) -> float:
    raw = row.get(field)
    if raw is None or raw == "":
        return 0.0
    if raw.casefold() in {"true", "false"}:
        return float(raw.casefold() == "true")
    return float(raw)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _wilson_interval(successes: int, total: int, *, confidence_level: float) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _truthy(value: str) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(
    aggregate: Sequence[dict[str, object]],
    paired: Sequence[dict[str, object]],
    *,
    confidence_level: float,
    reference_strategy: str,
) -> str:
    percentage = confidence_level * 100.0
    lines = [
        "# Statistical analysis",
        "",
        f"Intervals use a {percentage:.1f}% cluster bootstrap over `(seed, dataset_sample_id)`.",
        f"Paired comparisons use `{reference_strategy}` on the same sample, seed, target, and adapter version.",
        "",
        f"- Aggregate metric rows: {len(aggregate)}",
        f"- Paired comparison rows: {len(paired)}",
        "- `safety_intervals.csv` reports Wilson intervals, including an upper bound when zero failures are observed.",
        "- `latency_percentiles.csv` reports p50/p90/p95/p99 instead of relying only on arithmetic means.",
        "",
    ]
    return "\n".join(lines)
