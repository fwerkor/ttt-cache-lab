from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModuleKind(StrEnum):
    ATTENTION_Q = "attention.q"
    ATTENTION_K = "attention.k"
    ATTENTION_V = "attention.v"
    ATTENTION_O = "attention.o"
    ATTENTION_QV = "attention.qv"
    ATTENTION_ATTN = "attention.attn"
    MLP = "mlp"
    NORM = "norm"
    OUTPUT_HEAD = "output_head"
    LORA_Q = "lora.q"
    LORA_K = "lora.k"
    LORA_V = "lora.v"
    LORA_O = "lora.o"
    LORA_QV = "lora.qv"
    LORA_ATTN = "lora.attn"
    LORA_ALL_LATE = "lora.all_late"
    LORA_MLP = "lora.mlp"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UpdateTarget:
    kind: ModuleKind
    layer: int | None = None
    raw: str = ""

    @property
    def is_lora(self) -> bool:
        return self.kind.value.startswith("lora.")

    @property
    def is_multi_module(self) -> bool:
        return self.kind in {
            ModuleKind.ATTENTION_QV,
            ModuleKind.ATTENTION_ATTN,
            ModuleKind.LORA_QV,
            ModuleKind.LORA_ATTN,
            ModuleKind.LORA_ALL_LATE,
        }


def parse_update_target(raw: str, *, num_layers: int | None = None) -> UpdateTarget:
    normalized = raw.strip().lower().replace("-", "_")
    layer: int | None = None

    normalized, layer = _strip_layer_position(normalized, num_layers=num_layers)

    if ":" in normalized:
        name, layer_part = normalized.split(":", 1)
        normalized = name
        layer = _parse_layer_id(layer_part)

    aliases = {
        "q": ModuleKind.ATTENTION_Q,
        "k": ModuleKind.ATTENTION_K,
        "v": ModuleKind.ATTENTION_V,
        "o": ModuleKind.ATTENTION_O,
        "qv": ModuleKind.ATTENTION_QV,
        "attn": ModuleKind.ATTENTION_ATTN,
        "attention.q": ModuleKind.ATTENTION_Q,
        "attention.k": ModuleKind.ATTENTION_K,
        "attention.v": ModuleKind.ATTENTION_V,
        "attention.o": ModuleKind.ATTENTION_O,
        "attention.qv": ModuleKind.ATTENTION_QV,
        "attention.attn": ModuleKind.ATTENTION_ATTN,
        "mlp": ModuleKind.MLP,
        "norm": ModuleKind.NORM,
        "output_head": ModuleKind.OUTPUT_HEAD,
        "head": ModuleKind.OUTPUT_HEAD,
        "lora.q": ModuleKind.LORA_Q,
        "lora.k": ModuleKind.LORA_K,
        "lora.v": ModuleKind.LORA_V,
        "lora.o": ModuleKind.LORA_O,
        "lora.qv": ModuleKind.LORA_QV,
        "lora.attn": ModuleKind.LORA_ATTN,
        "lora.all": ModuleKind.LORA_ALL_LATE,
        "lora.all_late": ModuleKind.LORA_ALL_LATE,
        "lora.mlp": ModuleKind.LORA_MLP,
    }
    kind = aliases.get(normalized, ModuleKind.UNKNOWN)
    return UpdateTarget(kind=kind, layer=layer, raw=raw)


def _strip_layer_position(normalized: str, *, num_layers: int | None) -> tuple[str, int | None]:
    suffix_to_fraction = {
        ".early": 0.0,
        "_early": 0.0,
        ".middle": 0.5,
        "_middle": 0.5,
        ".mid": 0.5,
        "_mid": 0.5,
        ".late": 1.0,
        "_late": 1.0,
    }
    for suffix, fraction in suffix_to_fraction.items():
        if normalized.endswith(suffix):
            base = normalized[: -len(suffix)]
            if num_layers is None:
                return base, None
            if fraction == 0.0:
                return base, 0
            if fraction == 1.0:
                return base, max(0, num_layers - 1)
            return base, max(0, min(num_layers - 1, num_layers // 2))
    return normalized, None


def _parse_layer_id(raw: str) -> int:
    layer_part = raw.strip()
    if layer_part.startswith("layer"):
        layer_part = layer_part.removeprefix("layer")
    return int(layer_part)
