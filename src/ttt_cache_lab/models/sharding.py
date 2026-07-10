from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelShardPlan:
    device_map: dict[str, str]
    input_device: str
    devices: tuple[str, ...]
    layer_to_device: tuple[str, ...]


def build_model_shard_plan(config: Any, *, device_type: str, device_ids: list[int]) -> ModelShardPlan:
    if device_type not in {"cuda", "npu"}:
        raise ValueError(f"Model sharding requires CUDA or NPU devices, got {device_type!r}")
    if len(device_ids) < 2:
        raise ValueError("Model sharding requires at least two device ids")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("model.device_ids must not contain duplicates")

    num_layers = _num_hidden_layers(config)
    devices = tuple(f"{device_type}:{device_id}" for device_id in device_ids)
    layer_devices = tuple(
        devices[min(len(devices) - 1, layer * len(devices) // num_layers)] for layer in range(num_layers)
    )
    model_type = str(getattr(config, "model_type", "")).lower()
    tied = bool(getattr(config, "tie_word_embeddings", False))

    if model_type in {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2"}:
        device_map = {
            "model.embed_tokens": devices[0],
            **{f"model.layers.{layer}": layer_devices[layer] for layer in range(num_layers)},
            "model.norm": devices[-1],
            "lm_head": devices[0] if tied else devices[-1],
        }
    elif model_type in {"gpt2", "gpt_bigcode"}:
        device_map = {
            "transformer.wte": devices[0],
            "transformer.wpe": devices[0],
            **{f"transformer.h.{layer}": layer_devices[layer] for layer in range(num_layers)},
            "transformer.ln_f": devices[-1],
            "lm_head": devices[0] if tied else devices[-1],
        }
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r} for explicit model sharding; "
            "supported families are Llama/Mistral/Qwen/Gemma and GPT-2/BigCode."
        )

    return ModelShardPlan(
        device_map=device_map,
        input_device=devices[0],
        devices=devices,
        layer_to_device=layer_devices,
    )


def resolve_shard_device_ids(torch: Any, *, device_type: str, configured: list[int]) -> list[int]:
    if configured:
        return list(configured)
    runtime = getattr(torch, device_type, None)
    if runtime is None or not hasattr(runtime, "device_count"):
        raise RuntimeError(f"torch.{device_type}.device_count is unavailable")
    count = int(runtime.device_count())
    if count < 2:
        raise RuntimeError(f"model_shard requested but only {count} {device_type.upper()} device(s) are visible")
    return list(range(count))


def _num_hidden_layers(config: Any) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(f"Cannot infer transformer layer count from {type(config).__name__}")
