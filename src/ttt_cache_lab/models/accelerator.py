from __future__ import annotations

from typing import Any


def import_torch_npu_if_available() -> bool:
    try:
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def has_npu(torch: Any) -> bool:
    return hasattr(torch, "npu") and bool(torch.npu.is_available())


def resolve_device(torch: Any, device: str) -> str:
    if device != "auto":
        if device == "npu":
            return "npu:0"
        return device
    if has_npu(torch):
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    if device.startswith("npu") and has_npu(torch):
        torch.npu.synchronize()


def reset_peak_memory(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return
    if device.startswith("npu") and has_npu(torch) and hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()


def memory_allocated(torch: Any, device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    if device.startswith("npu") and has_npu(torch) and hasattr(torch.npu, "memory_allocated"):
        return int(torch.npu.memory_allocated())
    return 0


def max_memory_allocated(torch: Any, device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    if device.startswith("npu") and has_npu(torch) and hasattr(torch.npu, "max_memory_allocated"):
        return int(torch.npu.max_memory_allocated())
    return 0
