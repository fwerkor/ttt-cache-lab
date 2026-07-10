from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ttt_cache_lab.cache.planner import (
    CachePlanner,
    PlannerDecision,
    PlannerPolicy,
    PlannerRuntime,
)
from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.updates.targets import ModuleKind, UpdateTarget


class StrategyName(StrEnum):
    FULL_RECOMPUTE = "full_recompute"
    NO_ADAPTATION = "no_adaptation"
    STALE_REUSE = "stale_reuse"
    FROZEN_REUSE = "frozen_reuse"
    PERIODIC_REFRESH = "periodic_refresh"
    THRESHOLD_REFRESH = "threshold_refresh"
    LAYERWISE_RECOMPUTE = "layerwise_recompute"
    DELTA_CORRECTION = "delta_correction"
    BASE_CACHE_REUSE = "base_cache_reuse"
    ADAPTER_SPECIFIC_CACHE = "adapter_specific_cache"
    STATIC_BASE_DELTA = "static_base_delta"
    ALORA_PREFIX_REUSE = "alora_prefix_reuse"
    LRAGENT_ADAPTER_CACHE = "lragent_adapter_cache"
    FORKKV_BASE_DELTA = "forkkv_base_delta"
    ORACLE_PLANNER = "oracle_planner"
    ADAPTIVE = "adaptive"
    ADAPTIVE_NO_VERSION = "adaptive_no_version"
    ADAPTIVE_NO_TARGET = "adaptive_no_target"
    ADAPTIVE_NO_NORM = "adaptive_no_norm"
    ADAPTIVE_NO_DELTA = "adaptive_no_delta"
    ADAPTIVE_NO_PARTIAL = "adaptive_no_partial"
    ADAPTIVE_NO_PERIODIC = "adaptive_no_periodic"


@dataclass(frozen=True)
class StrategyDecision:
    strategy: StrategyName
    action: CacheAction
    state: CacheBlockState
    first_invalid_layer: int | None
    reason: str
    recompute_fraction: float = 0.0
    reject_reuse: bool = False


class CacheStrategy:
    name: StrategyName

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        raise NotImplementedError

    def decide_with_runtime(
        self,
        target: UpdateTarget,
        *,
        step: int,
        update_norm: float,
        runtime: PlannerRuntime,
    ) -> StrategyDecision:
        del runtime
        return self.decide(target, step=step, update_norm=update_norm)


class FullRecomputeStrategy(CacheStrategy):
    name = StrategyName.FULL_RECOMPUTE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        return StrategyDecision(
            self.name,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            None,
            "Baseline: always recompute full prefix cache.",
            recompute_fraction=1.0,
        )


class NoAdaptationStrategy(CacheStrategy):
    name = StrategyName.NO_ADAPTATION

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        return StrategyDecision(
            self.name,
            CacheAction.REUSE_EXACT,
            CacheBlockState.VALID_EXACT,
            None,
            "Baseline: skip adaptation and reuse the original model output exactly.",
        )


class StaleReuseStrategy(CacheStrategy):
    name = StrategyName.STALE_REUSE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        return StrategyDecision(
            self.name,
            CacheAction.REUSE_STALE,
            CacheBlockState.VALID_APPROX,
            None,
            "Baseline: reuse old cache without correction.",
        )


class FrozenReuseStrategy(CacheStrategy):
    name = StrategyName.FROZEN_REUSE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        return StrategyDecision(
            self.name,
            CacheAction.REUSE_FROZEN,
            CacheBlockState.VALID_FROZEN,
            None,
            "Treat old K/V as frozen evidence.",
        )


class PeriodicRefreshStrategy(CacheStrategy):
    name = StrategyName.PERIODIC_REFRESH

    def __init__(self, period: int = 4) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = period

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        if step % self.period == 0:
            return StrategyDecision(
                self.name,
                CacheAction.FULL_RECOMPUTE,
                CacheBlockState.INVALID,
                None,
                f"Refresh period {self.period} reached.",
                recompute_fraction=1.0,
            )
        return StrategyDecision(
            self.name,
            CacheAction.REUSE_STALE,
            CacheBlockState.VALID_APPROX,
            None,
            f"Reuse until refresh period {self.period} is reached.",
        )


class ThresholdRefreshStrategy(CacheStrategy):
    name = StrategyName.THRESHOLD_REFRESH

    def __init__(self, update_norm_threshold: float = 0.05) -> None:
        self.update_norm_threshold = update_norm_threshold

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        if update_norm > self.update_norm_threshold:
            return StrategyDecision(
                self.name,
                CacheAction.FULL_RECOMPUTE,
                CacheBlockState.INVALID,
                None,
                f"Accumulated update norm {update_norm:.6g} exceeds threshold {self.update_norm_threshold:.6g}.",
                recompute_fraction=1.0,
            )
        return StrategyDecision(
            self.name,
            CacheAction.REUSE_STALE,
            CacheBlockState.VALID_APPROX,
            None,
            f"Accumulated update norm {update_norm:.6g} remains below threshold {self.update_norm_threshold:.6g}.",
        )


class LayerwiseRecomputeStrategy(CacheStrategy):
    name = StrategyName.LAYERWISE_RECOMPUTE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        del step, update_norm
        if target.layer is None:
            return StrategyDecision(
                self.name,
                CacheAction.FULL_RECOMPUTE,
                CacheBlockState.INVALID,
                None,
                "No affected-layer boundary is available; layerwise recompute becomes full recompute.",
                recompute_fraction=1.0,
            )
        return StrategyDecision(
            self.name,
            CacheAction.PARTIAL_RECOMPUTE,
            CacheBlockState.INVALID,
            target.layer,
            "Recompute from the first affected layer onward.",
        )


class AdaptiveStrategy(CacheStrategy):
    name = StrategyName.ADAPTIVE

    def __init__(
        self,
        planner: CachePlanner | None = None,
        *,
        name: StrategyName = StrategyName.ADAPTIVE,
    ) -> None:
        self.planner = planner or CachePlanner()
        self.name = name

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        return self.decide_with_runtime(
            target,
            step=step,
            update_norm=update_norm,
            runtime=PlannerRuntime(),
        )

    def decide_with_runtime(
        self,
        target: UpdateTarget,
        *,
        step: int,
        update_norm: float,
        runtime: PlannerRuntime,
    ) -> StrategyDecision:
        decision: PlannerDecision = self.planner.plan(
            target,
            update_norm=update_norm,
            version_gap=step,
            runtime=runtime,
        )
        return StrategyDecision(
            self.name,
            decision.action,
            decision.state,
            decision.first_invalid_layer,
            decision.reason,
            recompute_fraction=decision.recompute_fraction,
            reject_reuse=decision.reject_reuse,
        )


class DeltaCorrectionStrategy(CacheStrategy):
    name = StrategyName.DELTA_CORRECTION

    def __init__(self, update_norm_threshold: float = 0.05) -> None:
        self.planner = CachePlanner(
            PlannerPolicy(
                update_norm_threshold=update_norm_threshold,
                periodic_refresh_interval=None,
            )
        )

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        decision: PlannerDecision = self.planner.plan(target, update_norm=update_norm, version_gap=step)
        return StrategyDecision(
            self.name,
            decision.action,
            decision.state,
            decision.first_invalid_layer,
            f"Delta-correction baseline: {decision.reason}",
            recompute_fraction=decision.recompute_fraction,
            reject_reuse=decision.reject_reuse,
        )


class BaseCacheReuseStrategy(CacheStrategy):
    name = StrategyName.BASE_CACHE_REUSE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        if target.kind in {ModuleKind.ATTENTION_Q, ModuleKind.LORA_Q, ModuleKind.OUTPUT_HEAD}:
            return StrategyDecision(
                self.name,
                CacheAction.REUSE_FROZEN,
                CacheBlockState.VALID_FROZEN,
                None,
                "Static baseline: base cache is reusable for Q-only or output-head changes.",
            )
        return StrategyDecision(
            self.name,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            target.layer,
            "Static baseline: base cache cannot cover state-changing adapter components.",
            recompute_fraction=1.0,
        )


class AdapterSpecificCacheStrategy(CacheStrategy):
    name = StrategyName.ADAPTER_SPECIFIC_CACHE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        return StrategyDecision(
            self.name,
            CacheAction.REUSE_EXACT if step == 0 else CacheAction.FULL_RECOMPUTE,
            CacheBlockState.VALID_EXACT if step == 0 else CacheBlockState.INVALID,
            None,
            "Static baseline: reuse only when this adapter version already has a cache entry.",
            recompute_fraction=0.0 if step == 0 else 1.0,
        )


class StaticBaseDeltaStrategy(CacheStrategy):
    name = StrategyName.STATIC_BASE_DELTA

    def __init__(self, update_norm_threshold: float = 0.05) -> None:
        self.update_norm_threshold = update_norm_threshold

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        if target.kind in {
            ModuleKind.ATTENTION_K,
            ModuleKind.ATTENTION_V,
            ModuleKind.ATTENTION_QV,
            ModuleKind.LORA_K,
            ModuleKind.LORA_V,
            ModuleKind.LORA_QV,
        }:
            return StrategyDecision(
                self.name,
                CacheAction.DELTA_CORRECT,
                CacheBlockState.VALID_APPROX,
                target.layer,
                "Static baseline: approximate adapter effect with a base-plus-delta cache component.",
            )
        if update_norm <= self.update_norm_threshold:
            return StrategyDecision(
                self.name,
                CacheAction.REUSE_STALE,
                CacheBlockState.VALID_APPROX,
                target.layer,
                "Static baseline: small non-K/V adapter change uses stale base cache.",
            )
        return StrategyDecision(
            self.name,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            target.layer,
            "Static baseline: non-K/V adapter delta is not decomposed safely.",
            recompute_fraction=1.0,
        )


class AloraPrefixReuseStrategy(CacheStrategy):
    name = StrategyName.ALORA_PREFIX_REUSE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        del step, update_norm
        if target.is_lora:
            return StrategyDecision(
                self.name,
                CacheAction.ALORA_SUFFIX_RECOMPUTE,
                CacheBlockState.VALID_EXACT,
                target.layer,
                "aLoRA baseline: reuse the base-model prefix before the invocation marker and recompute the suffix.",
                recompute_fraction=0.25,
            )
        return StrategyDecision(
            self.name,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            target.layer,
            "aLoRA activation semantics apply only to adapter updates.",
            recompute_fraction=1.0,
        )


class LrAgentAdapterCacheStrategy(AdapterSpecificCacheStrategy):
    name = StrategyName.LRAGENT_ADAPTER_CACHE

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        decision = super().decide(target, step=step, update_norm=update_norm)
        return StrategyDecision(
            self.name,
            decision.action,
            decision.state,
            decision.first_invalid_layer,
            "LRAgent-style baseline: keep a dedicated complete cache entry per fixed adapter identity.",
            recompute_fraction=decision.recompute_fraction,
        )


class ForkKvBaseDeltaStrategy(StaticBaseDeltaStrategy):
    name = StrategyName.FORKKV_BASE_DELTA

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        decision = super().decide(target, step=step, update_norm=update_norm)
        return StrategyDecision(
            self.name,
            decision.action,
            decision.state,
            decision.first_invalid_layer,
            f"ForkKV-style base/delta decomposition: {decision.reason}",
            recompute_fraction=decision.recompute_fraction,
            reject_reuse=decision.reject_reuse,
        )


class OraclePlannerStrategy(CacheStrategy):
    name = StrategyName.ORACLE_PLANNER

    def __init__(self, update_norm_threshold: float = 0.05) -> None:
        self.planner = CachePlanner(
            PlannerPolicy(
                update_norm_threshold=update_norm_threshold,
                allow_delta_correction=True,
                periodic_refresh_interval=None,
            )
        )

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        del step, update_norm
        return StrategyDecision(
            self.name,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            target.layer,
            "Measured oracle selection is deferred to the experiment runner.",
            recompute_fraction=1.0,
        )


def build_strategy(
    name: str,
    *,
    refresh_period: int = 4,
    update_norm_threshold: float = 0.05,
    version_gap_threshold: int = 8,
    error_proxy_threshold: float = 0.25,
    latency_budget_fraction: float = 1.0,
    memory_budget_bytes: int | None = None,
    failure_map_path: Path | None = None,
    safe_kl_threshold: float = 0.05,
    safe_top1_threshold: float = 0.99,
    safe_task_drop_threshold: float = 0.01,
) -> CacheStrategy:
    parsed = StrategyName(name)
    if parsed is StrategyName.FULL_RECOMPUTE:
        return FullRecomputeStrategy()
    if parsed is StrategyName.NO_ADAPTATION:
        return NoAdaptationStrategy()
    if parsed is StrategyName.STALE_REUSE:
        return StaleReuseStrategy()
    if parsed is StrategyName.FROZEN_REUSE:
        return FrozenReuseStrategy()
    if parsed is StrategyName.PERIODIC_REFRESH:
        return PeriodicRefreshStrategy(period=refresh_period)
    if parsed is StrategyName.THRESHOLD_REFRESH:
        return ThresholdRefreshStrategy(update_norm_threshold=update_norm_threshold)
    if parsed is StrategyName.LAYERWISE_RECOMPUTE:
        return LayerwiseRecomputeStrategy()
    if parsed in {
        StrategyName.ADAPTIVE,
        StrategyName.ADAPTIVE_NO_VERSION,
        StrategyName.ADAPTIVE_NO_TARGET,
        StrategyName.ADAPTIVE_NO_NORM,
        StrategyName.ADAPTIVE_NO_DELTA,
        StrategyName.ADAPTIVE_NO_PARTIAL,
        StrategyName.ADAPTIVE_NO_PERIODIC,
    }:
        policy = PlannerPolicy(
            update_norm_threshold=update_norm_threshold,
            version_gap_threshold=version_gap_threshold,
            error_proxy_threshold=error_proxy_threshold,
            latency_budget_fraction=latency_budget_fraction,
            memory_budget_bytes=memory_budget_bytes,
            failure_map_path=failure_map_path,
            safe_kl_threshold=safe_kl_threshold,
            safe_top1_threshold=safe_top1_threshold,
            safe_task_drop_threshold=safe_task_drop_threshold,
            allow_delta_correction=parsed is not StrategyName.ADAPTIVE_NO_DELTA,
            allow_layerwise_recompute=parsed is not StrategyName.ADAPTIVE_NO_PARTIAL,
            use_version_id=parsed is not StrategyName.ADAPTIVE_NO_VERSION,
            use_target_rules=parsed is not StrategyName.ADAPTIVE_NO_TARGET,
            use_update_norm=parsed is not StrategyName.ADAPTIVE_NO_NORM,
            periodic_refresh_interval=(
                None if parsed is StrategyName.ADAPTIVE_NO_PERIODIC else refresh_period
            ),
        )
        return AdaptiveStrategy(CachePlanner(policy), name=parsed)
    if parsed is StrategyName.DELTA_CORRECTION:
        return DeltaCorrectionStrategy(update_norm_threshold=update_norm_threshold)
    if parsed is StrategyName.BASE_CACHE_REUSE:
        return BaseCacheReuseStrategy()
    if parsed is StrategyName.ADAPTER_SPECIFIC_CACHE:
        return AdapterSpecificCacheStrategy()
    if parsed is StrategyName.STATIC_BASE_DELTA:
        return StaticBaseDeltaStrategy(update_norm_threshold=update_norm_threshold)
    if parsed is StrategyName.ALORA_PREFIX_REUSE:
        return AloraPrefixReuseStrategy()
    if parsed is StrategyName.LRAGENT_ADAPTER_CACHE:
        return LrAgentAdapterCacheStrategy()
    if parsed is StrategyName.FORKKV_BASE_DELTA:
        return ForkKvBaseDeltaStrategy(update_norm_threshold=update_norm_threshold)
    if parsed is StrategyName.ORACLE_PLANNER:
        return OraclePlannerStrategy(update_norm_threshold=update_norm_threshold)
    raise ValueError(f"Unsupported cache strategy: {name}")
