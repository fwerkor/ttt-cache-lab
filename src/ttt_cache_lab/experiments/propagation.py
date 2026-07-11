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
class PropagationRecord:
    sample_id: int
    dataset_sample_id: str
    task_name: str
    model_name: str
    update_target: str
    target_layer: int | None
    adapter_version: int
    cached_version: int
    version_gap: int
    layer_id: int
    context_length: int
    synthetic_difficulty: str
    seed: int
    configured_update_norm: float
    accumulated_update_norm: float
    probe_token_count: int
    hidden_relative_error: float
    hidden_cosine_distance: float
    hidden_norm_ratio: float
    key_relative_error: float
    key_cosine_distance: float
    key_norm_ratio: float
    value_relative_error: float
    value_cosine_distance: float
    value_norm_ratio: float

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
            self.layer_id,
            self.context_length,
            self.synthetic_difficulty,
            self.seed,
        )


@dataclass(frozen=True)
class PropagationArtifacts:
    jsonl_path: Path
    csv_path: Path
    records: list[PropagationRecord]


def collect_propagation_records(
    reference: BackendOutput,
    current: BackendOutput,
    *,
    sample_id: int,
    dataset_sample_id: str,
    task_name: str,
    model_name: str,
    update_target: str,
    target_layer: int | None,
    adapter_version: int,
    cached_version: int,
    context_length: int,
    synthetic_difficulty: str,
    seed: int,
    configured_update_norm: float,
    accumulated_update_norm: float,
    probe_tokens: int,
) -> list[PropagationRecord]:
    reference_hidden = _hidden_layers(reference)
    current_hidden = _hidden_layers(current)
    reference_cache = _cache_layers(reference)
    current_cache = _cache_layers(current)
    layer_count = min(
        len(reference_hidden),
        len(current_hidden),
        len(reference_cache),
        len(current_cache),
    )
    records: list[PropagationRecord] = []
    for layer_id in range(layer_count):
        hidden_metrics = _tensor_metrics(
            reference_hidden[layer_id], current_hidden[layer_id], probe_tokens=probe_tokens
        )
        reference_key, reference_value = reference_cache[layer_id]
        current_key, current_value = current_cache[layer_id]
        key_metrics = _tensor_metrics(reference_key, current_key, probe_tokens=probe_tokens)
        value_metrics = _tensor_metrics(reference_value, current_value, probe_tokens=probe_tokens)
        records.append(
            PropagationRecord(
                sample_id=sample_id,
                dataset_sample_id=dataset_sample_id,
                task_name=task_name,
                model_name=model_name,
                update_target=update_target,
                target_layer=target_layer,
                adapter_version=adapter_version,
                cached_version=cached_version,
                version_gap=adapter_version - cached_version,
                layer_id=layer_id,
                context_length=context_length,
                synthetic_difficulty=synthetic_difficulty,
                seed=seed,
                configured_update_norm=configured_update_norm,
                accumulated_update_norm=accumulated_update_norm,
                probe_token_count=probe_tokens,
                hidden_relative_error=hidden_metrics.relative_error,
                hidden_cosine_distance=hidden_metrics.cosine_distance,
                hidden_norm_ratio=hidden_metrics.norm_ratio,
                key_relative_error=key_metrics.relative_error,
                key_cosine_distance=key_metrics.cosine_distance,
                key_norm_ratio=key_metrics.norm_ratio,
                value_relative_error=value_metrics.relative_error,
                value_cosine_distance=value_metrics.cosine_distance,
                value_norm_ratio=value_metrics.norm_ratio,
            )
        )
    return records


def write_propagation_records(
    records: list[PropagationRecord],
    output_dir: Path,
    *,
    merge_existing: bool = False,
) -> PropagationArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "propagation_records.jsonl"
    csv_path = output_dir / "propagation_records.csv"
    merged: dict[tuple[Any, ...], PropagationRecord] = {}
    if merge_existing and jsonl_path.exists():
        for record in read_propagation_records(jsonl_path):
            merged[record.identity()] = record
    for record in records:
        merged[record.identity()] = record
    final_records = list(merged.values())
    _atomic_write_jsonl(jsonl_path, final_records)
    _atomic_write_csv(csv_path, final_records)
    return PropagationArtifacts(jsonl_path=jsonl_path, csv_path=csv_path, records=final_records)


def read_propagation_records(path: Path) -> list[PropagationRecord]:
    records: list[PropagationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Propagation record {line_number} in {path} is not an object")
            records.append(PropagationRecord(**payload))
    return records


@dataclass(frozen=True)
class _TensorMetrics:
    relative_error: float
    cosine_distance: float
    norm_ratio: float


def _hidden_layers(output: BackendOutput) -> list[Any]:
    extras = output.extras or {}
    hidden_states = extras.get("hidden_states")
    if isinstance(hidden_states, tuple) and len(hidden_states) >= 2:
        return list(hidden_states[1:])
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
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                result.append((layer[0], layer[1]))
        if result:
            return result
    array = output.cache_tensor
    return [(array[index], array[index]) for index in range(len(array))]


def _tensor_metrics(reference: Any, current: Any, *, probe_tokens: int) -> _TensorMetrics:
    left = _sample_tensor(reference, probe_tokens=probe_tokens)
    right = _sample_tensor(current, probe_tokens=probe_tokens)
    if _is_torch_tensor(left) and _is_torch_tensor(right):
        left_float = left.detach().float()
        right_float = right.detach().to(left_float.device).float()
        difference_norm = float((right_float - left_float).norm().item())
        left_norm = float(left_float.norm().item())
        right_norm = float(right_float.norm().item())
        dot = float((left_float.reshape(-1) * right_float.reshape(-1)).sum().item())
    else:
        left_array = np.asarray(_to_numpy(left), dtype=np.float64)
        right_array = np.asarray(_to_numpy(right), dtype=np.float64)
        difference_norm = float(np.linalg.norm(right_array - left_array))
        left_norm = float(np.linalg.norm(left_array))
        right_norm = float(np.linalg.norm(right_array))
        dot = float(np.dot(left_array.reshape(-1), right_array.reshape(-1)))
    epsilon = 1e-12
    relative_error = difference_norm / max(left_norm, epsilon)
    cosine = dot / max(left_norm * right_norm, epsilon)
    cosine = max(-1.0, min(1.0, cosine))
    norm_ratio = right_norm / max(left_norm, epsilon)
    return _TensorMetrics(
        relative_error=float(relative_error),
        cosine_distance=float(1.0 - cosine),
        norm_ratio=float(norm_ratio),
    )


def _sample_tensor(value: Any, *, probe_tokens: int) -> Any:
    shape = tuple(int(size) for size in getattr(value, "shape", ()))
    if len(shape) < 2 or probe_tokens <= 0:
        return value
    candidate_axes = list(range(1, max(2, len(shape) - 1)))
    if not candidate_axes:
        return value
    sequence_axis = max(candidate_axes, key=lambda axis: shape[axis])
    sequence_length = shape[sequence_axis]
    if sequence_length <= probe_tokens:
        return value
    positions = np.linspace(0, sequence_length - 1, num=probe_tokens, dtype=np.int64)
    if _is_torch_tensor(value):
        import torch

        index = torch.as_tensor(positions, device=value.device)
        return value.index_select(sequence_axis, index)
    return np.take(np.asarray(value), positions, axis=sequence_axis)


def _is_torch_tensor(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("torch") and hasattr(value, "detach")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _atomic_write_jsonl(path: Path, records: list[PropagationRecord]) -> None:
    with _temporary_text_file(path) as (handle, temporary):
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, records: list[PropagationRecord]) -> None:
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
