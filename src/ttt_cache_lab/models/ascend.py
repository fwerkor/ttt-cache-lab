from __future__ import annotations

from ttt_cache_lab.models.accelerator import import_torch_npu_if_available
from ttt_cache_lab.models.hf import HuggingFaceBackend


class AscendHuggingFaceBackend(HuggingFaceBackend):
    """HuggingFace backend configured for Ascend NPU through torch-npu.

    The implementation intentionally reuses HuggingFaceBackend so that CUDA and
    Ascend experiments measure the same logic: prefix prefill, stale/frozen cache
    reuse, LoRA training, versioned runner records, and report generation.
    """

    def __init__(self, **kwargs: object) -> None:
        if not import_torch_npu_if_available():
            raise RuntimeError(
                "ascend_hf requires torch-npu. Install a torch/torch-npu/CANN "
                "combination matching the Ascend server, then rerun with "
                "model.backend=ascend_hf."
            )
        if kwargs.get("device") == "auto" or kwargs.get("device") == "npu":
            kwargs["device"] = "npu:0"
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if not self.device.startswith("npu"):
            raise RuntimeError(f"ascend_hf resolved to non-NPU device: {self.device}")
