import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName
from ttt_cache_lab.experiments.metrics import (
    attention_distribution_shift,
    is_false_safe,
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


def test_false_safe_uses_kl_top1_and_task_drop() -> None:
    decision = StrategyDecision(
        StrategyName.STALE_REUSE,
        CacheAction.REUSE_STALE,
        CacheBlockState.VALID_APPROX,
        None,
        "test",
    )
    full = BackendOutput(
        logits=np.array([[4.0, 1.0]]),
        cache_tensor=np.zeros((1, 1)),
        hidden_tensor=np.zeros((1, 1)),
        parameter_version=1,
    )
    same_top1_but_shifted = BackendOutput(
        logits=np.array([[1.1, 1.0]]),
        cache_tensor=np.zeros((1, 1)),
        hidden_tensor=np.zeros((1, 1)),
        parameter_version=1,
    )
    assert is_false_safe(
        decision,
        full=full,
        approx=same_top1_but_shifted,
        full_task_score=1.0,
        approx_task_score=1.0,
        kl_threshold=0.01,
        top1_threshold=0.99,
        task_drop_threshold=0.01,
    )
    assert is_false_safe(
        decision,
        full=full,
        approx=full,
        full_task_score=1.0,
        approx_task_score=0.5,
        kl_threshold=0.05,
        top1_threshold=0.99,
        task_drop_threshold=0.01,
    )
    assert not is_false_safe(
        decision,
        full=full,
        approx=full,
        full_task_score=1.0,
        approx_task_score=1.0,
        kl_threshold=0.05,
        top1_threshold=0.99,
        task_drop_threshold=0.01,
    )



def test_attention_distribution_shift_uses_jensen_shannon_divergence() -> None:
    full = BackendOutput(
        logits=np.zeros((1, 2)),
        cache_tensor=np.zeros((1, 1)),
        hidden_tensor=np.zeros((1, 1)),
        parameter_version=1,
        extras={"attention_summary": np.array([[0.9, 0.1]])},
    )
    same = BackendOutput(
        logits=np.zeros((1, 2)),
        cache_tensor=np.zeros((1, 1)),
        hidden_tensor=np.zeros((1, 1)),
        parameter_version=1,
        extras={"attention_summary": np.array([[0.9, 0.1]])},
    )
    shifted = BackendOutput(
        logits=np.zeros((1, 2)),
        cache_tensor=np.zeros((1, 1)),
        hidden_tensor=np.zeros((1, 1)),
        parameter_version=1,
        extras={"attention_summary": np.array([[0.1, 0.9]])},
    )
    assert attention_distribution_shift(full, same) == 0.0
    assert attention_distribution_shift(full, shifted) > 0.0
