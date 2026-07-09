from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModuleKind(StrEnum):
    ATTENTION_Q = "attention.q"
    ATTENTION_K = "attention.k"
    ATTENTION_V = "attention.v"
    ATTENTION_O = "attention.o"
    MLP = "mlp"
    NORM = "norm"
    OUTPUT_HEAD = "output_head"
    LORA_Q = "lora.q"
    LORA_K = "lora.k"
    LORA_V = "lora.v"
    LORA_O = "lora.o"
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


def parse_update_target(raw: str, *, num_layers: int | None = None) -> UpdateTarget:
    normalized = raw.strip().lower().replace("-", "_")
    layer: int | None = None

    if normalized.endswith(".late") or normalized.endswith("_late"):
        layer = None if num_layers is None else max(0, num_layers - 1)
        normalized = normalized.removesuffix(".late").removesuffix("_late")

    if ":" in normalized:
        name, layer_part = normalized.split(":", 1)
        normalized = name
        if layer_part.startswith("layer"):
            layer_part = layer_part.removeprefix("layer")
        layer = int(layer_part)

    aliases = {
        "q": ModuleKind.ATTENTION_Q,
        "k": ModuleKind.ATTENTION_K,
        "v": ModuleKind.ATTENTION_V,
        "o": ModuleKind.ATTENTION_O,
        "attention.q": ModuleKind.ATTENTION_Q,
        "attention.k": ModuleKind.ATTENTION_K,
        "attention.v": ModuleKind.ATTENTION_V,
        "attention.o": ModuleKind.ATTENTION_O,
        "mlp": ModuleKind.MLP,
        "norm": ModuleKind.NORM,
        "output_head": ModuleKind.OUTPUT_HEAD,
        "head": ModuleKind.OUTPUT_HEAD,
        "lora.q": ModuleKind.LORA_Q,
        "lora.k": ModuleKind.LORA_K,
        "lora.v": ModuleKind.LORA_V,
        "lora.o": ModuleKind.LORA_O,
        "lora.mlp": ModuleKind.LORA_MLP,
        "lora.mlp_late": ModuleKind.LORA_MLP,
    }
    kind = aliases.get(normalized, ModuleKind.UNKNOWN)
    return UpdateTarget(kind=kind, layer=layer, raw=raw)
