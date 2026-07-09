from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.models.accelerator import resolve_device


def test_ascend_config_loads() -> None:
    config = VersionedExperimentConfig.from_yaml(Path("configs/experiments/ascend_smoke_qwen_0_5b.yaml"))
    assert config.model.backend == "ascend_hf"
    assert config.model.device == "npu:0"


class _FakeNpu:
    @staticmethod
    def is_available() -> bool:
        return True


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    npu = _FakeNpu()
    cuda = _FakeCuda()


def test_resolve_device_prefers_npu() -> None:
    assert resolve_device(_FakeTorch(), "auto") == "npu:0"
    assert resolve_device(_FakeTorch(), "npu") == "npu:0"
