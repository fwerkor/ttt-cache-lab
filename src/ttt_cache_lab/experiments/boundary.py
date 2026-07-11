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
    attention_topk_overlap: float
    attention_output_relative_error: float
    attention_output_cosine_distance: float
    boundary_input_hidden_relative_error: float
    boundary_next_hidden_relative_error: float
    key_relative_error: float
    value_relative_error: float
    attention_weighted_key_relative_error: float
    attention_weighted_value_relative_error: float
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
    attention_topk_overlap = 0.0
    attention_output_relative_error = 0.0
    attention_output_cosine_distance = 0.0
    boundary_input_hidden_relative_error = 0.0
    boundary_next_hidden_relative_error = 0.0
    key_relative_error = 0.0
    value_relative_error = 0.0
    weighted_key_relative_error = 0.0
    weighted_value_relative_error = 0.0
    metric_available = False

    if has_stale_rejoin:
        full_attention = _layer_array(full, "attention_summary", clamped_boundary)
        approx_attention = _layer_array(approx, "attention_summary", clamped_boundary)
        if full_attention is not None and approx_attention is not None:
            attention_kl, attention_js, attention_l1, attention_topk_overlap = (
                _distribution_metrics(full_attention, approx_attention, topk=topk)
            )
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

        full_hidden = _hidden_layers(full)
        approx_hidden = _hidden_layers(approx)
        if clamped_boundary < min(len(full_hidden), len(approx_hidden)):
            boundary_input_hidden_relative_error, _ = _vector_metrics(
                full_hidden[clamped_boundary], approx_hidden[clamped_boundary]
            )
            metric_available = True
        next_index = clamped_boundary + 1
        if next_index < min(len(full_hidden), len(approx_hidden)):
            boundary_next_hidden_relative_error, _ = _vector_metrics(
                full_hidden[next_index], approx_hidden[next_index]
            )
            metric_available = True

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
        attention_topk_overlap=attention_topk_overlap,
        attention_output_relative_error=attention_output_relative_error,
        attention_output_cosine_distance=attention_output_cosine_distance,
        boundary_input_hidden_relative_error=boundary_input_hidden_relative_error,
        boundary_next_hidden_relative_error=boundary_next_hidden_relative_error,
        key_relative_error=key_relative_error,
        value_relative_error=value_relative_error,
        attention_weighted_key_relative_error=weighted_key_relative_error,
        attention_weighted_value_relative_error=weighted_value_relative_error,
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


def _hidden_layers(output: BackendOutput) -> list[Any]:
    extras = output.extras or {}
    hidden_states = extras.get("hidden_states")
    if isinstance(hidden_states, tuple) and hidden_states:
        return list(hidden_states)
    array = output.hidden_tensor
    return [array[index] for index in range(len(array))]


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
) -> tuple[float, float, float, float]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    size = min(left.size, right.size)
    if size == 0:
        return 0.0, 0.0, 0.0, 0.0
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
    k = max(1, min(topk, size))
    left_top = set(np.argpartition(left, -k)[-k:].tolist())
    right_top = set(np.argpartition(right, -k)[-k:].tolist())
    overlap = len(left_top & right_top) / k
    return kl, js, l1, overlap


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
