from __future__ import annotations

from collections.abc import Iterable, Sequence

REFERENCE_FIELDS = (
    "run_name",
    "experiment_id",
    "sample_id",
    "dataset_sample_id",
    "evaluation_partition",
    "task_name",
    "update_target",
    "adapter_id",
    "adapter_version",
    "lora_rank",
    "update_mode",
    "context_length",
    "synthetic_difficulty",
    "model_name",
    "model_num_layers",
    "model_hidden_size",
    "model_parameter_count",
    "configured_update_norm",
    "seed",
)

CONDITION_FIELDS = (
    "run_name",
    "experiment_id",
    "model_name",
    "model_num_layers",
    "model_hidden_size",
    "model_parameter_count",
    "context_length",
    "synthetic_difficulty",
    "task_name",
    "task_family",
    "benchmark_name",
    "evaluation_partition",
    "dataset_split",
    "dataset_category",
    "lora_rank",
    "configured_update_norm",
    "update_mode",
    "norm_control",
    "seed",
)


def sweep_fields(rows: Iterable[dict[str, str]]) -> tuple[str, ...]:
    return tuple(sorted({field for row in rows for field in row if field.startswith("sweep.")}))


def available_fields(
    rows: Sequence[dict[str, str]],
    candidates: Iterable[str],
    *,
    include_sweeps: bool = True,
) -> tuple[str, ...]:
    present = {field for row in rows for field in row}
    fields = [field for field in candidates if field in present]
    if include_sweeps:
        for field in sweep_fields(rows):
            if field not in fields:
                fields.append(field)
    return tuple(fields)


def condition_fields(
    rows: Sequence[dict[str, str]],
    *extra: str,
) -> tuple[str, ...]:
    return available_fields(rows, (*CONDITION_FIELDS, *extra))


def reference_key(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    fields = [field for field in REFERENCE_FIELDS if field in row]
    fields.extend(sorted(field for field in row if field.startswith("sweep.")))
    return tuple((field, row.get(field, "")) for field in fields)


def with_full_reference_metrics(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    full_rows: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for row in rows:
        if row.get("cache_strategy") != "full_recompute":
            continue
        key = reference_key(row)
        if key in full_rows:
            raise ValueError(f"Duplicate full_recompute reference for condition {key!r}")
        full_rows[key] = row

    enriched: list[dict[str, str]] = []
    for row in rows:
        key = reference_key(row)
        reference = full_rows.get(key)
        if reference is None:
            raise ValueError(
                "Missing full_recompute reference for "
                f"experiment={row.get('experiment_id', '')!r}, "
                f"target={row.get('update_target', '')!r}, "
                f"adapter_version={row.get('adapter_version', '')!r}, "
                f"sample_id={row.get('sample_id', '')!r}"
            )
        item = dict(row)
        task_score = _number(row, "task_score")
        full_task_score = _number(reference, "task_score")
        latency = _preferred_number(row, ("end_to_end_latency", "latency_units"))
        full_latency = _preferred_number(reference, ("end_to_end_latency", "latency_units"))
        item["full_task_score"] = str(full_task_score)
        item["task_drop_vs_full"] = str(full_task_score - task_score)
        item["full_end_to_end_latency"] = str(full_latency)
        item["speedup_vs_full"] = str(full_latency / latency if latency > 0.0 else 0.0)
        enriched.append(item)
    return enriched


def _preferred_number(row: dict[str, str], fields: tuple[str, ...]) -> float:
    for field in fields:
        raw = row.get(field)
        if raw not in {None, ""}:
            return _number(row, field)
    return 0.0


def _number(row: dict[str, str], field: str) -> float:
    raw = row.get(field)
    if raw is None or raw == "":
        return 0.0
    return float(raw)
