import numpy as np

from ttt_cache_lab.experiments.metrics import (
    output_cache_bytes,
    output_cache_maintenance_latency,
    output_decode_latency,
    output_peak_memory_allocated,
    output_strategy_latency,
    output_throughput,
)
from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement
from ttt_cache_lab.models.interface import BackendOutput


def test_relative_error_zero_for_same_tensor() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert relative_error(x, x) == 0.0


def test_top1_agreement() -> None:
    a = np.array([[0.1, 0.9], [0.8, 0.2]])
    b = np.array([[0.2, 0.7], [0.1, 0.9]])
    assert top1_agreement(a, b) == 0.5


def test_kl_nonnegative() -> None:
    a = np.array([[0.1, 0.9]])
    b = np.array([[0.2, 0.8]])
    assert kl_divergence(a, b) >= 0.0


def test_output_cost_metrics_prefer_backend_measurements() -> None:
    output = BackendOutput(
        logits=np.zeros((1, 2)),
        cache_tensor=np.zeros((1, 2, 2)),
        hidden_tensor=np.zeros((1, 2)),
        parameter_version=1,
        extras={
            "cache_bytes": 4096,
            "peak_memory_allocated": 8192,
            "strategy_latency": 2.0,
            "cache_maintenance_latency": 1.5,
            "decode_latency": 0.5,
            "generated_tokens": 4,
        },
    )
    assert output_cache_bytes(output) == 4096
    assert output_peak_memory_allocated(output) == 8192
    assert output_strategy_latency(output, fallback=99.0) == 2.0
    assert output_cache_maintenance_latency(output) == 1.5
    assert output_decode_latency(output) == 0.5
    assert output_throughput(output, latency=2.0) == 2.0
