from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ttt_cache_lab.models.interface import BackendOutput


@dataclass(frozen=True)
class BoundaryRecord:
    sample_id: int
    dataset_sample_id: str
    task_name: str
    task_family: str
    model_name: str
    model_num_layers: int
    update_target: str
    target_layer: int | None
    adapter_version: int
    cached_version: int
    version_gap: int
    cache_strategy: str
    window_size: int
    boundary_layer: int
    has_stale_rejoin: bool
    context_length: int
    synthetic_difficulty: str
    seed: int
    configured_update_norm: float
    accumulated_update_norm: float
    accumulated_raw_update_norm: float
    logits_kl: float
    top1_agreement: float
    task_drop_vs_full: float
    recompute_fraction: float
    attention_kl: float
    attention_js: float
    attention_l1: float
    attention_argmax_agreement: float
    attention_top4_overlap: float
    attention_top8_overlap: float
    attention_topk_overlap: float
    attention_output_relative_error: float
    attention_output_cosine_distance: float
    boundary_input_hidden_relative_error: float
    boundary_next_hidden_relative_error: float
    key_relative_error: float
    value_relative_error: float
    attention_weighted_key_relative_error: float
    attention_weighted_value_relative_error: float
    stale_suffix_layers: int
    suffix_attention_js_mean: float
    suffix_attention_js_max: float
    suffix_attention_output_relative_error_mean: float
    suffix_attention_output_relative_error_max: float
    suffix_attention_input_relative_error_mean: float
    suffix_attention_input_relative_error_max: float
    suffix_attention_input_relative_error_last: float
    suffix_amplification_ratio: float
    online_stale_probe_logits_kl: float
    online_stale_probe_top1_agreement: float
    online_stale_attention_output_boundary_relative_error: float
    online_stale_suffix_attention_js_mean: float
    online_stale_suffix_attention_js_max: float
    online_stale_suffix_attention_output_relative_error_mean: float
    online_stale_suffix_attention_output_relative_error_max: float
    online_stale_suffix_attention_output_relative_error_last: float
    online_stale_suffix_attention_input_relative_error_mean: float
    online_stale_suffix_attention_input_relative_error_max: float
    online_stale_suffix_attention_input_relative_error_last: float
    online_metric_available: bool
    metric_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> tuple[Any, ...]:
        return (
            self.dataset_sample_id or self.sample_id,
            self.model_name,
            self.task_name,
            self.update_target,
            self.adapter_version,
            self.cached_version,
            self.cache_strategy,
            self.window_size,
            self.context_length,
            self.synthetic_difficulty,
            self.seed,
        )


@dataclass(frozen=True)
class BoundaryArtifacts:
    jsonl_path: Path
    csv_path: Path
    records: list[BoundaryRecord]


def collect_boundary_record(
    baseline: BackendOutput,
    full: BackendOutput,
    approx: BackendOutput,
    stale: BackendOutput | None,
    *,
    sample_id: int,
    dataset_sample_id: str,
    task_name: str,
    task_family: str,
    model_name: str,
    model_num_layers: int,
    update_target: str,
    target_layer: int | None,
    adapter_version: int,
    cached_version: int,
    cache_strategy: str,
    window_size: int,
    boundary_layer: int,
    context_length: int,
    synthetic_difficulty: str,
    seed: int,
    configured_update_norm: float,
    accumulated_update_norm: float,
    accumulated_raw_update_norm: float,
    logits_kl: float,
    top1_agreement: float,
    task_drop_vs_full: float,
    recompute_fraction: float,
    topk: int,
) -> BoundaryRecord:
    clamped_boundary = min(max(0, boundary_layer), model_num_layers)
    has_stale_rejoin = clamped_boundary < model_num_layers
    attention_kl = 0.0
    attention_js = 0.0
    attention_l1 = 0.0
    attention_argmax_agreement = 0.0
    attention_top4_overlap = 0.0
    attention_top8_overlap = 0.0
    attention_topk_overlap = 0.0
    attention_output_relative_error = 0.0
    attention_output_cosine_distance = 0.0
    boundary_input_hidden_relative_error = 0.0
    boundary_next_hidden_relative_error = 0.0
    key_relative_error = 0.0
    value_relative_error = 0.0
    weighted_key_relative_error = 0.0
    weighted_value_relative_error = 0.0
    stale_suffix_layers = max(0, model_num_layers - clamped_boundary)
    suffix_attention_js_mean = 0.0
    suffix_attention_js_max = 0.0
    suffix_attention_output_relative_error_mean = 0.0
    suffix_attention_output_relative_error_max = 0.0
    suffix_attention_input_relative_error_mean = 0.0
    suffix_attention_input_relative_error_max = 0.0
    suffix_attention_input_relative_error_last = 0.0
    suffix_amplification_ratio = 0.0
    online_stale_probe_logits_kl = 0.0
    online_stale_probe_top1_agreement = 0.0
    online_stale_attention_output_boundary_relative_error = 0.0
    online_stale_suffix_attention_js_mean = 0.0
    online_stale_suffix_attention_js_max = 0.0
    online_stale_suffix_attention_output_relative_error_mean = 0.0
    online_stale_suffix_attention_output_relative_error_max = 0.0
    online_stale_suffix_attention_output_relative_error_last = 0.0
    online_stale_suffix_attention_input_relative_error_mean = 0.0
    online_stale_suffix_attention_input_relative_error_max = 0.0
    online_stale_suffix_attention_input_relative_error_last = 0.0
    online_metric_available = False
    metric_available = False

    if has_stale_rejoin:
        full_attention = _layer_array(full, "attention_summary", clamped_boundary)
        approx_attention = _layer_array(approx, "attention_summary", clamped_boundary)
        if full_attention is not None and approx_attention is not None:
            (
                attention_kl,
                attention_js,
                attention_l1,
                attention_argmax_agreement,
                attention_top4_overlap,
                attention_top8_overlap,
                attention_topk_overlap,
            ) = _distribution_metrics(full_attention, approx_attention, topk=topk)
            metric_available = True

        full_attention_output = _layer_array(
            full, "attention_output_summary", clamped_boundary
        )
        approx_attention_output = _layer_array(
            approx, "attention_output_summary", clamped_boundary
        )
        if full_attention_output is not None and approx_attention_output is not None:
            (
                attention_output_relative_error,
                attention_output_cosine_distance,
            ) = _vector_metrics(full_attention_output, approx_attention_output)
            metric_available = True

        full_attention_input = _layer_array(
            full, "attention_input_summary", clamped_boundary
        )
        approx_attention_input = _layer_array(
            approx, "attention_input_summary", clamped_boundary
        )
        if full_attention_input is not None and approx_attention_input is not None:
            boundary_input_hidden_relative_error, _ = _vector_metrics(
                full_attention_input, approx_attention_input
            )
            metric_available = True
        next_index = clamped_boundary + 1
        full_next_attention_input = _layer_array(
            full, "attention_input_summary", next_index
        )
        approx_next_attention_input = _layer_array(
            approx, "attention_input_summary", next_index
        )
        if (
            full_next_attention_input is not None
            and approx_next_attention_input is not None
        ):
            boundary_next_hidden_relative_error, _ = _vector_metrics(
                full_next_attention_input, approx_next_attention_input
            )
            metric_available = True

        suffix_attention_js = _suffix_distribution_metric(
            full,
            approx,
            name="attention_summary",
            start_layer=clamped_boundary,
            metric="js",
        )
        suffix_attention_outputs = _suffix_vector_relative_errors(
            full,
            approx,
            name="attention_output_summary",
            start_layer=clamped_boundary,
        )
        suffix_attention_inputs = _suffix_vector_relative_errors(
            full,
            approx,
            name="attention_input_summary",
            start_layer=clamped_boundary,
        )
        suffix_attention_js_mean, suffix_attention_js_max, _ = _summary_metrics(
            suffix_attention_js
        )
        (
            suffix_attention_output_relative_error_mean,
            suffix_attention_output_relative_error_max,
            _,
        ) = _summary_metrics(suffix_attention_outputs)
        (
            suffix_attention_input_relative_error_mean,
            suffix_attention_input_relative_error_max,
            suffix_attention_input_relative_error_last,
        ) = _summary_metrics(suffix_attention_inputs)
        suffix_amplification_ratio = (
            suffix_attention_input_relative_error_last
            / max(attention_output_relative_error, 1e-12)
        )

        if stale is not None:
            online_stale_probe_logits_kl = _logits_kl(stale.logits, approx.logits)
            online_stale_probe_top1_agreement = float(
                int(np.argmax(stale.logits) == np.argmax(approx.logits))
            )
            stale_boundary_output = _layer_array(
                stale, "attention_output_summary", clamped_boundary
            )
            if (
                stale_boundary_output is not None
                and approx_attention_output is not None
            ):
                online_stale_attention_output_boundary_relative_error, _ = (
                    _vector_metrics(stale_boundary_output, approx_attention_output)
                )
                online_metric_available = True
            online_stale_suffix_js = _suffix_distribution_metric(
                stale,
                approx,
                name="attention_summary",
                start_layer=clamped_boundary,
                metric="js",
            )
            online_stale_suffix_outputs = _suffix_vector_relative_errors(
                stale,
                approx,
                name="attention_output_summary",
                start_layer=clamped_boundary,
            )
            online_stale_suffix_inputs = _suffix_vector_relative_errors(
                stale,
                approx,
                name="attention_input_summary",
                start_layer=clamped_boundary,
            )
            (
                online_stale_suffix_attention_js_mean,
                online_stale_suffix_attention_js_max,
                _,
            ) = _summary_metrics(online_stale_suffix_js)
            (
                online_stale_suffix_attention_output_relative_error_mean,
                online_stale_suffix_attention_output_relative_error_max,
                online_stale_suffix_attention_output_relative_error_last,
            ) = _summary_metrics(online_stale_suffix_outputs)
            (
                online_stale_suffix_attention_input_relative_error_mean,
                online_stale_suffix_attention_input_relative_error_max,
                online_stale_suffix_attention_input_relative_error_last,
            ) = _summary_metrics(online_stale_suffix_inputs)
            online_metric_available = online_metric_available or bool(
                online_stale_suffix_js
                or online_stale_suffix_outputs
                or online_stale_suffix_inputs
            )

        baseline_cache = _cache_layers(baseline)
        full_cache = _cache_layers(full)
        if clamped_boundary < min(len(baseline_cache), len(full_cache)):
            old_key, old_value = baseline_cache[clamped_boundary]
            new_key, new_value = full_cache[clamped_boundary]
            key_relative_error, _ = _vector_metrics(new_key, old_key)
            value_relative_error, _ = _vector_metrics(new_value, old_value)
            if full_attention is not None:
                weighted_key_relative_error = _weighted_cache_relative_error(
                    new_key, old_key, full_attention
                )
                weighted_value_relative_error = _weighted_cache_relative_error(
                    new_value, old_value, full_attention
                )
            metric_available = True

    return BoundaryRecord(
        sample_id=sample_id,
        dataset_sample_id=dataset_sample_id,
        task_name=task_name,
        task_family=task_family,
        model_name=model_name,
        model_num_layers=model_num_layers,
        update_target=update_target,
        target_layer=target_layer,
        adapter_version=adapter_version,
        cached_version=cached_version,
        version_gap=adapter_version - cached_version,
        cache_strategy=cache_strategy,
        window_size=window_size,
        boundary_layer=clamped_boundary,
        has_stale_rejoin=has_stale_rejoin,
        context_length=context_length,
        synthetic_difficulty=synthetic_difficulty,
        seed=seed,
        configured_update_norm=configured_update_norm,
        accumulated_update_norm=accumulated_update_norm,
        accumulated_raw_update_norm=accumulated_raw_update_norm,
        logits_kl=logits_kl,
        top1_agreement=top1_agreement,
        task_drop_vs_full=task_drop_vs_full,
        recompute_fraction=recompute_fraction,
        attention_kl=attention_kl,
        attention_js=attention_js,
        attention_l1=attention_l1,
        attention_argmax_agreement=attention_argmax_agreement,
        attention_top4_overlap=attention_top4_overlap,
        attention_top8_overlap=attention_top8_overlap,
        attention_topk_overlap=attention_topk_overlap,
        attention_output_relative_error=attention_output_relative_error,
        attention_output_cosine_distance=attention_output_cosine_distance,
        boundary_input_hidden_relative_error=boundary_input_hidden_relative_error,
        boundary_next_hidden_relative_error=boundary_next_hidden_relative_error,
        key_relative_error=key_relative_error,
        value_relative_error=value_relative_error,
        attention_weighted_key_relative_error=weighted_key_relative_error,
        attention_weighted_value_relative_error=weighted_value_relative_error,
        stale_suffix_layers=stale_suffix_layers,
        suffix_attention_js_mean=suffix_attention_js_mean,
        suffix_attention_js_max=suffix_attention_js_max,
        suffix_attention_output_relative_error_mean=(
            suffix_attention_output_relative_error_mean
        ),
        suffix_attention_output_relative_error_max=(
            suffix_attention_output_relative_error_max
        ),
        suffix_attention_input_relative_error_mean=(
            suffix_attention_input_relative_error_mean
        ),
        suffix_attention_input_relative_error_max=(
            suffix_attention_input_relative_error_max
        ),
        suffix_attention_input_relative_error_last=(
            suffix_attention_input_relative_error_last
        ),
        suffix_amplification_ratio=suffix_amplification_ratio,
        online_stale_probe_logits_kl=online_stale_probe_logits_kl,
        online_stale_probe_top1_agreement=online_stale_probe_top1_agreement,
        online_stale_attention_output_boundary_relative_error=(
            online_stale_attention_output_boundary_relative_error
        ),
        online_stale_suffix_attention_js_mean=(
            online_stale_suffix_attention_js_mean
        ),
        online_stale_suffix_attention_js_max=(
            online_stale_suffix_attention_js_max
        ),
        online_stale_suffix_attention_output_relative_error_mean=(
            online_stale_suffix_attention_output_relative_error_mean
        ),
        online_stale_suffix_attention_output_relative_error_max=(
            online_stale_suffix_attention_output_relative_error_max
        ),
        online_stale_suffix_attention_output_relative_error_last=(
            online_stale_suffix_attention_output_relative_error_last
        ),
        online_stale_suffix_attention_input_relative_error_mean=(
            online_stale_suffix_attention_input_relative_error_mean
        ),
        online_stale_suffix_attention_input_relative_error_max=(
            online_stale_suffix_attention_input_relative_error_max
        ),
        online_stale_suffix_attention_input_relative_error_last=(
            online_stale_suffix_attention_input_relative_error_last
        ),
        online_metric_available=online_metric_available,
        metric_available=metric_available,
    )


def write_boundary_records(
    records: list[BoundaryRecord],
    output_dir: Path,
    *,
    merge_existing: bool = False,
) -> BoundaryArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "boundary_records.jsonl"
    csv_path = output_dir / "boundary_records.csv"
    merged: dict[tuple[Any, ...], BoundaryRecord] = {}
    if merge_existing and jsonl_path.exists():
        for record in read_boundary_records(jsonl_path):
            merged[record.identity()] = record
    for record in records:
        merged[record.identity()] = record
    final_records = list(merged.values())
    _atomic_write_jsonl(jsonl_path, final_records)
    _atomic_write_csv(csv_path, final_records)
    return BoundaryArtifacts(jsonl_path=jsonl_path, csv_path=csv_path, records=final_records)


def read_boundary_records(path: Path) -> list[BoundaryRecord]:
    records: list[BoundaryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Boundary record {line_number} in {path} is not an object")
            payload.setdefault("attention_argmax_agreement", 0.0)
            payload.setdefault("attention_top4_overlap", 0.0)
            payload.setdefault("attention_top8_overlap", 0.0)
            payload.setdefault("stale_suffix_layers", 0)
            payload.setdefault("suffix_attention_js_mean", 0.0)
            payload.setdefault("suffix_attention_js_max", 0.0)
            payload.setdefault("suffix_attention_output_relative_error_mean", 0.0)
            payload.setdefault("suffix_attention_output_relative_error_max", 0.0)
            payload.setdefault("suffix_attention_input_relative_error_mean", 0.0)
            payload.setdefault("suffix_attention_input_relative_error_max", 0.0)
            payload.setdefault("suffix_attention_input_relative_error_last", 0.0)
            payload.setdefault("suffix_amplification_ratio", 0.0)
            payload.setdefault("online_stale_probe_logits_kl", 0.0)
            payload.setdefault("online_stale_probe_top1_agreement", 0.0)
            payload.setdefault(
                "online_stale_attention_output_boundary_relative_error", 0.0
            )
            payload.setdefault("online_stale_suffix_attention_js_mean", 0.0)
            payload.setdefault("online_stale_suffix_attention_js_max", 0.0)
            payload.setdefault(
                "online_stale_suffix_attention_output_relative_error_mean", 0.0
            )
            payload.setdefault(
                "online_stale_suffix_attention_output_relative_error_max", 0.0
            )
            payload.setdefault(
                "online_stale_suffix_attention_output_relative_error_last", 0.0
            )
            payload.setdefault(
                "online_stale_suffix_attention_input_relative_error_mean", 0.0
            )
            payload.setdefault(
                "online_stale_suffix_attention_input_relative_error_max", 0.0
            )
            payload.setdefault(
                "online_stale_suffix_attention_input_relative_error_last", 0.0
            )
            payload.setdefault("online_metric_available", False)
            records.append(BoundaryRecord(**payload))
    return records


def _layer_array(output: BackendOutput, name: str, layer: int) -> np.ndarray | None:
    extras = output.extras or {}
    value = extras.get(name)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim < 2 or layer < 0 or layer >= array.shape[0]:
        return None
    return np.asarray(array[layer], dtype=np.float64)


def _suffix_distribution_metric(
    reference: BackendOutput,
    candidate: BackendOutput,
    *,
    name: str,
    start_layer: int,
    metric: str,
) -> list[float]:
    reference_array = _extra_array(reference, name)
    candidate_array = _extra_array(candidate, name)
    if reference_array is None or candidate_array is None:
        return []
    layer_count = min(reference_array.shape[0], candidate_array.shape[0])
    values: list[float] = []
    for layer in range(start_layer, layer_count):
        kl, js, l1, *_ = _distribution_metrics(
            reference_array[layer], candidate_array[layer], topk=8
        )
        values.append({"kl": kl, "js": js, "l1": l1}[metric])
    return values


def _suffix_vector_relative_errors(
    reference: BackendOutput,
    candidate: BackendOutput,
    *,
    name: str,
    start_layer: int,
) -> list[float]:
    reference_array = _extra_array(reference, name)
    candidate_array = _extra_array(candidate, name)
    if reference_array is None or candidate_array is None:
        return []
    layer_count = min(reference_array.shape[0], candidate_array.shape[0])
    return [
        _vector_metrics(reference_array[layer], candidate_array[layer])[0]
        for layer in range(start_layer, layer_count)
    ]


def _summary_metrics(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.max(values)), float(values[-1])


def _extra_array(output: BackendOutput, name: str) -> np.ndarray | None:
    extras = output.extras or {}
    value = extras.get(name)
    if value is None:
        return None
    array = np.asarray(value)
    return array if array.ndim >= 2 else None


def _cache_layers(output: BackendOutput) -> list[tuple[Any, Any]]:
    extras = output.extras or {}
    past = extras.get("past_key_values")
    if past is not None:
        to_legacy = getattr(past, "to_legacy_cache", None)
        layers = list(to_legacy() if callable(to_legacy) else past)
        result: list[tuple[Any, Any]] = []
        for layer in layers:
            if isinstance(layer, tuple | list) and len(layer) >= 2:
                result.append((layer[0], layer[1]))
        if result:
            return result
    array = output.cache_tensor
    if array.ndim >= 3 and array.shape[1] >= 2:
        return [(array[index, 0], array[index, 1]) for index in range(len(array))]
    return [(array[index], array[index]) for index in range(len(array))]


def _distribution_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    topk: int,
) -> tuple[float, float, float, float, float, float, float]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    size = min(left.size, right.size)
    if size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    left = np.clip(left[:size], 0.0, None)
    right = np.clip(right[:size], 0.0, None)
    epsilon = 1e-12
    left /= max(float(left.sum()), epsilon)
    right /= max(float(right.sum()), epsilon)
    left = np.clip(left, epsilon, None)
    right = np.clip(right, epsilon, None)
    left /= left.sum()
    right /= right.sum()
    kl = float(np.sum(left * np.log(left / right)))
    midpoint = 0.5 * (left + right)
    js = float(
        0.5 * np.sum(left * np.log(left / midpoint))
        + 0.5 * np.sum(right * np.log(right / midpoint))
    )
    l1 = float(np.sum(np.abs(left - right)))
    argmax_agreement = float(int(np.argmax(left) == np.argmax(right)))
    top4_overlap = _topk_overlap(left, right, 4)
    top8_overlap = _topk_overlap(left, right, 8)
    configured_overlap = _topk_overlap(left, right, topk)
    return (
        kl,
        js,
        l1,
        argmax_agreement,
        top4_overlap,
        top8_overlap,
        configured_overlap,
    )


def _topk_overlap(left: np.ndarray, right: np.ndarray, topk: int) -> float:
    k = max(1, min(topk, left.size, right.size))
    left_top = set(np.argpartition(left, -k)[-k:].tolist())
    right_top = set(np.argpartition(right, -k)[-k:].tolist())
    return len(left_top & right_top) / k


def _logits_kl(reference: Any, candidate: Any) -> float:
    left = np.asarray(_to_numpy(reference), dtype=np.float64).reshape(-1)
    right = np.asarray(_to_numpy(candidate), dtype=np.float64).reshape(-1)
    size = min(left.size, right.size)
    if size == 0:
        return 0.0
    left = left[:size] - np.max(left[:size])
    right = right[:size] - np.max(right[:size])
    left_probability = np.exp(left)
    right_probability = np.exp(right)
    left_probability /= max(float(left_probability.sum()), 1e-12)
    right_probability /= max(float(right_probability.sum()), 1e-12)
    left_probability = np.clip(left_probability, 1e-12, None)
    right_probability = np.clip(right_probability, 1e-12, None)
    return float(
        np.sum(left_probability * np.log(left_probability / right_probability))
    )


def _vector_metrics(reference: Any, candidate: Any) -> tuple[float, float]:
    if _is_torch_tensor(reference) and _is_torch_tensor(candidate):
        left = reference.detach().float()
        right = candidate.detach().to(left.device).float()
        size = min(left.numel(), right.numel())
        left = left.reshape(-1)[:size]
        right = right.reshape(-1)[:size]
        difference = float((right - left).norm().item())
        left_norm = float(left.norm().item())
        right_norm = float(right.norm().item())
        dot = float((left * right).sum().item())
    else:
        left_array = np.asarray(_to_numpy(reference), dtype=np.float64).reshape(-1)
        right_array = np.asarray(_to_numpy(candidate), dtype=np.float64).reshape(-1)
        size = min(left_array.size, right_array.size)
        left_array = left_array[:size]
        right_array = right_array[:size]
        difference = float(np.linalg.norm(right_array - left_array))
        left_norm = float(np.linalg.norm(left_array))
        right_norm = float(np.linalg.norm(right_array))
        dot = float(np.dot(left_array, right_array))
    epsilon = 1e-12
    relative = difference / max(left_norm, epsilon)
    cosine = dot / max(left_norm * right_norm, epsilon)
    cosine = max(-1.0, min(1.0, cosine))
    return float(relative), float(1.0 - cosine)


def _weighted_cache_relative_error(reference: Any, candidate: Any, weights: np.ndarray) -> float:
    if _is_torch_tensor(reference) and _is_torch_tensor(candidate):
        left = reference.detach().float()
        right = candidate.detach().to(left.device).float()
        if left.ndim < 2:
            return _vector_metrics(left, right)[0]
        sequence_axis = left.ndim - 2
        sequence_length = min(left.shape[sequence_axis], right.shape[sequence_axis], len(weights))
        if sequence_length <= 0:
            return 0.0
        index = [slice(None)] * left.ndim
        index[sequence_axis] = slice(0, sequence_length)
        left = left[tuple(index)]
        right = right[tuple(index)]
        reduce_axes = tuple(axis for axis in range(left.ndim) if axis != sequence_axis)
        difference = (right - left).pow(2).sum(dim=reduce_axes).sqrt()
        magnitude = left.pow(2).sum(dim=reduce_axes).sqrt()
        torch_weights = left.new_tensor(np.asarray(weights[:sequence_length], dtype=np.float32))
        torch_weights = torch_weights.clamp_min(0)
        torch_weights /= torch_weights.sum().clamp_min(1e-12)
        numerator = float((torch_weights * difference).sum().item())
        denominator = float((torch_weights * magnitude).sum().item())
        return numerator / max(denominator, 1e-12)

    left = np.asarray(_to_numpy(reference), dtype=np.float64)
    right = np.asarray(_to_numpy(candidate), dtype=np.float64)
    if left.ndim < 2:
        return _vector_metrics(left, right)[0]
    sequence_axis = left.ndim - 2
    sequence_length = min(left.shape[sequence_axis], right.shape[sequence_axis], len(weights))
    if sequence_length <= 0:
        return 0.0
    index = [slice(None)] * left.ndim
    index[sequence_axis] = slice(0, sequence_length)
    left = left[tuple(index)]
    right = right[tuple(index)]
    reduce_axes = tuple(axis for axis in range(left.ndim) if axis != sequence_axis)
    difference = np.sqrt(np.sum((right - left) ** 2, axis=reduce_axes))
    magnitude = np.sqrt(np.sum(left**2, axis=reduce_axes))
    normalized_weights = np.clip(np.asarray(weights[:sequence_length], dtype=np.float64), 0.0, None)
    normalized_weights /= max(float(normalized_weights.sum()), 1e-12)
    numerator = float(np.sum(normalized_weights * difference))
    denominator = float(np.sum(normalized_weights * magnitude))
    return numerator / max(denominator, 1e-12)


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.startswith("torch") and hasattr(value, "detach")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _atomic_write_jsonl(path: Path, records: list[BoundaryRecord]) -> None:
    with _temporary_text_file(path) as (handle, temporary):
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, records: list[BoundaryRecord]) -> None:
    with _temporary_text_file(path, newline="") as (handle, temporary):
        fieldnames = list(records[0].to_dict().keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _temporary_text_file:
    def __init__(self, destination: Path, *, newline: str | None = None) -> None:
        self.destination = destination
        self.newline = newline
        self.handle: Any | None = None
        self.path: Path | None = None

    def __enter__(self) -> tuple[Any, Path]:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=self.newline,
            dir=self.destination.parent,
            prefix=f".{self.destination.name}.",
            suffix=".tmp",
            delete=False,
        )
        self.handle = handle
        self.path = Path(handle.name)
        return handle, self.path

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle is not None:
            self.handle.close()
        if exc_type is not None and self.path is not None:
            self.path.unlink(missing_ok=True)
