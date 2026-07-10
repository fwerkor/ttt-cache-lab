from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import StrategyDecision
from ttt_cache_lab.experiments.metrics import (
    output_cache_maintenance_latency,
    output_decode_latency,
    output_strategy_latency,
)
from ttt_cache_lab.models.interface import BackendOutput, ModelBackend


@dataclass(frozen=True)
class StrategyMeasurement:
    output: BackendOutput
    timed_runs: int
    warmup_runs: int
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_std: float
    decode_latency_p50: float
    cache_maintenance_latency_p50: float



def execute_strategy(
    backend: ModelBackend,
    *,
    prompt: str,
    baseline: BackendOutput,
    full: BackendOutput,
    updated: BackendOutput,
    decision: StrategyDecision,
) -> BackendOutput:
    if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
        return backend.full_recompute(prompt, updated)
    return backend.apply_cache_strategy(
        baseline=baseline,
        full=full,
        updated=updated,
        decision=decision,
    )

def measure_backend_call(
    execute: Callable[[], BackendOutput],
    *,
    warmup_runs: int,
    timed_runs: int,
    fallback_latency: float,
) -> StrategyMeasurement:
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if timed_runs < 1:
        raise ValueError("timed_runs must be positive")
    for _ in range(warmup_runs):
        execute()
    output: BackendOutput | None = None
    latencies: list[float] = []
    decode_latencies: list[float] = []
    maintenance_latencies: list[float] = []
    for _ in range(timed_runs):
        output = execute()
        latencies.append(output_strategy_latency(output, fallback=fallback_latency))
        decode_latencies.append(output_decode_latency(output))
        maintenance_latencies.append(output_cache_maintenance_latency(output))
    if output is None:
        raise RuntimeError("timed measurement produced no output")
    return StrategyMeasurement(
        output=output,
        timed_runs=timed_runs,
        warmup_runs=warmup_runs,
        latency_mean=statistics.fmean(latencies),
        latency_p50=_quantile(latencies, 0.50),
        latency_p95=_quantile(latencies, 0.95),
        latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        decode_latency_p50=_quantile(decode_latencies, 0.50),
        cache_maintenance_latency_p50=_quantile(maintenance_latencies, 0.50),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
