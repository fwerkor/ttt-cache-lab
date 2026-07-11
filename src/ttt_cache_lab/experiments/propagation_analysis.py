from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def generate_propagation_analysis(
    input_csv: Path,
    output_dir: Path,
    *,
    recovery_ratio: float = 0.1,
) -> tuple[Path, Path]:
    if not 0.0 < recovery_ratio <= 1.0:
        raise ValueError("recovery_ratio must be in (0, 1]")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_csv)
    layers = _aggregate_layers(rows)
    layers_path = output_dir / "layerwise_propagation.csv"
    _write_rows(layers_path, layers)
    profiles = _build_profiles(layers, recovery_ratio=recovery_ratio)
    profiles_path = output_dir / "propagation_profiles.csv"
    _write_rows(profiles_path, profiles)
    (output_dir / "propagation_profiles.md").write_text(
        _to_markdown(profiles), encoding="utf-8"
    )
    return layers_path, profiles_path


def _aggregate_layers(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    dimensions = (
        "model_name",
        "context_length",
        "synthetic_difficulty",
        "task_name",
        "update_target",
        "target_layer",
        "adapter_version",
        "cached_version",
        "version_gap",
        "configured_update_norm",
        "layer_id",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in dimensions)].append(row)
    output: list[dict[str, str | int | float]] = []
    for key, group in sorted(groups.items()):
        values = dict(zip(dimensions, key, strict=True))
        output.append(
            {
                **values,
                "layer_id": int(values["layer_id"]),
                "count": len(group),
                "hidden_relative_error_mean": _mean(group, "hidden_relative_error"),
                "hidden_cosine_distance_mean": _mean(group, "hidden_cosine_distance"),
                "hidden_norm_ratio_mean": _mean(group, "hidden_norm_ratio"),
                "key_relative_error_mean": _mean(group, "key_relative_error"),
                "key_cosine_distance_mean": _mean(group, "key_cosine_distance"),
                "value_relative_error_mean": _mean(group, "value_relative_error"),
                "value_cosine_distance_mean": _mean(group, "value_cosine_distance"),
            }
        )
    return output


def _build_profiles(
    layers: list[dict[str, str | int | float]],
    *,
    recovery_ratio: float,
) -> list[dict[str, str | int | float | bool]]:
    excluded = {
        "layer_id",
        "count",
        "hidden_relative_error_mean",
        "hidden_cosine_distance_mean",
        "hidden_norm_ratio_mean",
        "key_relative_error_mean",
        "key_cosine_distance_mean",
        "value_relative_error_mean",
        "value_cosine_distance_mean",
    }
    groups: dict[
        tuple[tuple[str, str], ...], list[dict[str, str | int | float]]
    ] = defaultdict(list)
    for row in layers:
        key = tuple(sorted((field, str(value)) for field, value in row.items() if field not in excluded))
        groups[key].append(row)

    profiles: list[dict[str, str | int | float | bool]] = []
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["layer_id"]))
        target_raw = dict(key).get("target_layer", "")
        target_layer = int(target_raw) if target_raw not in {"", "None"} else 0
        suffix = [row for row in ordered if int(row["layer_id"]) >= target_layer]
        if not suffix:
            continue
        errors = [float(row["hidden_relative_error_mean"]) for row in suffix]
        peak = max(errors)
        peak_index = errors.index(peak)
        peak_layer = int(suffix[peak_index]["layer_id"])
        tail = errors[-1]
        first = errors[0]
        threshold = peak * recovery_ratio
        recovery_layer = target_layer if peak == 0.0 else -1
        if peak > 0.0:
            for index, _value in enumerate(errors):
                if all(later <= threshold for later in errors[index:]):
                    recovery_layer = int(suffix[index]["layer_id"])
                    break
        decreasing_steps = sum(
            later <= earlier for earlier, later in zip(errors, errors[1:], strict=False)
        )
        transition_count = max(0, len(errors) - 1)
        decay_fraction = decreasing_steps / transition_count if transition_count else 1.0
        tail_to_peak = tail / peak if peak > 0.0 else 0.0
        profiles.append(
            {
                **dict(key),
                "first_hidden_error": first,
                "peak_hidden_error": peak,
                "peak_layer": peak_layer,
                "tail_hidden_error": tail,
                "tail_to_peak_ratio": tail_to_peak,
                "nonincreasing_step_fraction": decay_fraction,
                "recovery_ratio": recovery_ratio,
                "recovery_layer": recovery_layer,
                "recovered_before_end": recovery_layer >= 0,
                "profile": _profile_label(
                    tail_to_peak,
                    peak_index,
                    len(errors),
                    peak=peak,
                ),
            }
        )
    return profiles


def _profile_label(
    tail_to_peak: float,
    peak_index: int,
    length: int,
    *,
    peak: float,
) -> str:
    if peak == 0.0:
        return "no_drift"
    if tail_to_peak <= 0.1:
        return "strong_decay"
    if tail_to_peak <= 0.5:
        return "partial_decay"
    if peak_index >= max(0, length - 2):
        return "late_amplification"
    return "persistent"


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
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _to_markdown(rows: list[dict[str, str | int | float | bool]]) -> str:
    if not rows:
        return "# Propagation profiles\n\nNo propagation records were found.\n"
    fields = [
        field
        for field in (
            "model_name",
            "task_name",
            "update_target",
            "version_gap",
            "peak_layer",
            "tail_to_peak_ratio",
            "recovery_layer",
            "profile",
        )
        if field in rows[0]
    ]
    lines = [
        "# Propagation profiles",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines) + "\n"
