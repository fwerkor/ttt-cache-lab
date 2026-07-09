from __future__ import annotations

from typing import Any

import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import StrategyDecision
from ttt_cache_lab.metrics.tensor import top1_agreement
from ttt_cache_lab.models.interface import BackendOutput


def output_cache_bytes(output: BackendOutput) -> int:
    return int(output.cache_tensor.nbytes + output.hidden_tensor.nbytes)


def output_memory_allocated(output: BackendOutput) -> int:
    extras = output.extras or {}
    value = extras.get("memory_allocated", 0)
    return int(value) if isinstance(value, int | float) else 0


def estimate_recompute_fraction(decision: StrategyDecision, *, num_layers: int) -> float:
    if decision.action is CacheAction.FULL_RECOMPUTE:
        return 1.0
    if decision.action is CacheAction.PARTIAL_RECOMPUTE:
        if decision.first_invalid_layer is None:
            return decision.recompute_fraction or 0.5
        remaining = max(0, num_layers - decision.first_invalid_layer)
        return max(0.0, min(1.0, remaining / max(1, num_layers)))
    if decision.action is CacheAction.DELTA_CORRECT:
        return decision.recompute_fraction or 0.15
    return decision.recompute_fraction


def is_cache_hit(decision: StrategyDecision) -> bool:
    return decision.action in {
        CacheAction.REUSE_EXACT,
        CacheAction.REUSE_FROZEN,
        CacheAction.REUSE_STALE,
        CacheAction.DELTA_CORRECT,
    }


def is_refresh_action(decision: StrategyDecision) -> bool:
    return decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.PARTIAL_RECOMPUTE}


def is_false_safe(decision: StrategyDecision, *, full: BackendOutput, approx: BackendOutput) -> bool:
    if decision.action not in {CacheAction.REUSE_STALE, CacheAction.REUSE_FROZEN, CacheAction.DELTA_CORRECT}:
        return False
    return bool(top1_agreement(full.logits, approx.logits) < 1.0)


def as_float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0
