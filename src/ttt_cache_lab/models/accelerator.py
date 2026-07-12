from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def import_torch_npu_if_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return True


def has_npu(torch: Any) -> bool:
    return hasattr(torch, "npu") and bool(torch.npu.is_available())


def resolve_device(torch: Any, device: str) -> str:
    if device != "auto":
        if device == "npu":
            return "npu:0"
        if device == "cuda":
            return "cuda:0"
        return device
    if has_npu(torch):
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def synchronize(torch: Any, device: str | Sequence[str]) -> None:
    for item in _devices(device):
        if item.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(item)
        elif item.startswith("npu") and has_npu(torch):
            torch.npu.synchronize(item)


def reset_peak_memory(torch: Any, device: str | Sequence[str]) -> None:
    for item in _devices(device):
        if item.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(item)
        elif item.startswith("npu") and has_npu(torch) and hasattr(torch.npu, "reset_peak_memory_stats"):
            torch.npu.reset_peak_memory_stats(item)


def memory_allocated(torch: Any, device: str | Sequence[str]) -> int:
    total = 0
    for item in _devices(device):
        if item.startswith("cuda") and torch.cuda.is_available():
            total += int(torch.cuda.memory_allocated(item))
        elif item.startswith("npu") and has_npu(torch) and hasattr(torch.npu, "memory_allocated"):
            total += int(torch.npu.memory_allocated(item))
    return total


def max_memory_allocated(torch: Any, device: str | Sequence[str]) -> int:
    total = 0
    for item in _devices(device):
        if item.startswith("cuda") and torch.cuda.is_available():
            total += int(torch.cuda.max_memory_allocated(item))
        elif item.startswith("npu") and has_npu(torch) and hasattr(torch.npu, "max_memory_allocated"):
            total += int(torch.npu.max_memory_allocated(item))
    return total


def _devices(device: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(device, str):
        return (device,)
    return tuple(dict.fromkeys(device))
