from __future__ import annotations

from typing import Any

import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import StrategyDecision
from ttt_cache_lab.metrics.tensor import kl_divergence, top1_agreement
from ttt_cache_lab.models.interface import BackendOutput


def output_cache_bytes(output: BackendOutput) -> int:
    extras = output.extras or {}
    value = extras.get("cache_bytes")
    if isinstance(value, int | float):
        return int(value)
    return int(output.cache_tensor.nbytes + output.hidden_tensor.nbytes)


def output_memory_allocated(output: BackendOutput) -> int:
    extras = output.extras or {}
    value = extras.get("memory_allocated", 0)
    return int(value) if isinstance(value, int | float) else 0



def output_peak_memory_allocated(output: BackendOutput) -> int:
    extras = output.extras or {}
    value = extras.get("peak_memory_allocated", 0)
    return int(value) if isinstance(value, int | float) else 0


def output_strategy_latency(output: BackendOutput, *, fallback: float) -> float:
    extras = output.extras or {}
    value = extras.get("strategy_latency")
    return float(value) if isinstance(value, int | float) else float(fallback)


def output_decode_latency(output: BackendOutput) -> float:
    extras = output.extras or {}
    value = extras.get("decode_latency", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def output_cache_maintenance_latency(output: BackendOutput) -> float:
    extras = output.extras or {}
    value = extras.get("cache_maintenance_latency", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def output_throughput(output: BackendOutput, *, latency: float) -> float:
    extras = output.extras or {}
    value = extras.get("generated_tokens", 0)
    tokens = float(value) if isinstance(value, int | float) else 0.0
    return tokens / latency if tokens > 0.0 and latency > 0.0 else 0.0

def estimate_recompute_fraction(decision: StrategyDecision, *, num_layers: int) -> float:
    if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
        return 1.0
    if decision.action is CacheAction.PARTIAL_RECOMPUTE:
        if decision.first_invalid_layer is None:
            return 1.0
        remaining = max(0, num_layers - decision.first_invalid_layer)
        return max(0.0, min(1.0, remaining / max(1, num_layers)))
    if decision.action is CacheAction.DELTA_CORRECT:
        return decision.recompute_fraction or 0.15
    if decision.action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
        return decision.recompute_fraction or 0.25
    return decision.recompute_fraction


def is_cache_hit(decision: StrategyDecision) -> bool:
    return decision.action in {
        CacheAction.REUSE_EXACT,
        CacheAction.REUSE_FROZEN,
        CacheAction.REUSE_STALE,
        CacheAction.DELTA_CORRECT,
        CacheAction.ALORA_SUFFIX_RECOMPUTE,
    }


def is_refresh_action(decision: StrategyDecision) -> bool:
    return decision.action in {
        CacheAction.FULL_RECOMPUTE,
        CacheAction.PARTIAL_RECOMPUTE,
        CacheAction.REJECT_UPDATE,
        CacheAction.ALORA_SUFFIX_RECOMPUTE,
    }


def is_false_safe(
    decision: StrategyDecision,
    *,
    full: BackendOutput,
    approx: BackendOutput,
    full_task_score: float,
    approx_task_score: float,
    kl_threshold: float,
    top1_threshold: float,
    task_drop_threshold: float,
) -> bool:
    if decision.action not in {
        CacheAction.REUSE_STALE,
        CacheAction.REUSE_FROZEN,
        CacheAction.DELTA_CORRECT,
        CacheAction.ALORA_SUFFIX_RECOMPUTE,
    }:
        return False
    logits_kl = kl_divergence(full.logits, approx.logits)
    top1 = top1_agreement(full.logits, approx.logits)
    task_drop = full_task_score - approx_task_score
    return bool(
        logits_kl > kl_threshold
        or top1 < top1_threshold
        or task_drop > task_drop_threshold
    )


def output_strategy_mode(output: BackendOutput) -> str:
    extras = output.extras or {}
    for key in ("delta_mode", "partial_mode", "cache_mode"):
        value = extras.get(key)
        if isinstance(value, str):
            return value
    return ""


def as_float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0
