from __future__ import annotations

from ttt_cache_lab.cache.planner import PlannerRuntime
from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName
from ttt_cache_lab.models.interface import ModelBackend
from ttt_cache_lab.updates.targets import UpdateTarget


def build_planner_runtime(
    backend: ModelBackend,
    target: UpdateTarget,
    *,
    context_length: int,
    total_cache_bytes: int,
    candidate_cache_bytes: int,
    model_name: str,
    lora_rank: int,
    configured_update_norm: float,
    update_mode: str,
) -> PlannerRuntime:
    full = _latency(
        backend,
        StrategyDecision(
            StrategyName.FULL_RECOMPUTE,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            None,
            "Planner runtime calibration: full recompute.",
            recompute_fraction=1.0,
        ),
        context_length=context_length,
    )
    reuse = _latency(
        backend,
        StrategyDecision(
            StrategyName.STALE_REUSE,
            CacheAction.REUSE_STALE,
            CacheBlockState.VALID_APPROX,
            None,
            "Planner runtime calibration: stale reuse.",
        ),
        context_length=context_length,
    )
    delta = _latency(
        backend,
        StrategyDecision(
            StrategyName.DELTA_CORRECTION,
            CacheAction.DELTA_CORRECT,
            CacheBlockState.VALID_APPROX,
            target.layer,
            "Planner runtime calibration: delta correction.",
            recompute_fraction=0.15,
        ),
        context_length=context_length,
    )
    partial = None
    if target.layer is not None:
        partial = _latency(
            backend,
            StrategyDecision(
                StrategyName.LAYERWISE_RECOMPUTE,
                CacheAction.PARTIAL_RECOMPUTE,
                CacheBlockState.INVALID,
                target.layer,
                "Planner runtime calibration: partial recompute.",
            ),
            context_length=context_length,
        )
    return PlannerRuntime(
        total_cache_bytes=total_cache_bytes,
        candidate_cache_bytes=candidate_cache_bytes,
        full_recompute_latency=full,
        reuse_latency=reuse,
        delta_correction_latency=delta,
        partial_recompute_latency=partial,
        model_name=model_name,
        context_length=context_length,
        lora_rank=lora_rank,
        configured_update_norm=configured_update_norm,
        update_mode=update_mode,
    )


def _latency(
    backend: ModelBackend,
    decision: StrategyDecision,
    *,
    context_length: int,
) -> float:
    return max(1e-9, backend.estimate_latency(decision, context_length=context_length))
