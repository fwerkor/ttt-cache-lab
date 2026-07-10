from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ttt_cache_lab.cache.planner import CachePlanner, PlannerDecision, PlannerPolicy
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
    ORACLE_PLANNER = "oracle_planner"
    ADAPTIVE = "adaptive"


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
            CacheAction.REUSE_FROZEN,
            CacheBlockState.VALID_FROZEN,
            None,
            "Baseline: keep the original pre-adaptation evidence fixed.",
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
        return StrategyDecision(
            self.name,
            CacheAction.PARTIAL_RECOMPUTE,
            CacheBlockState.INVALID,
            target.layer,
            "Recompute from the first affected layer onward.",
            recompute_fraction=0.5 if target.layer is None else 0.0,
        )


class AdaptiveStrategy(CacheStrategy):
    name = StrategyName.ADAPTIVE

    def __init__(self, planner: CachePlanner | None = None) -> None:
        self.planner = planner or CachePlanner()

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        decision: PlannerDecision = self.planner.plan(target, update_norm=update_norm, version_gap=step)
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
        self.planner = CachePlanner(PlannerPolicy(update_norm_threshold=update_norm_threshold))

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
        if target.kind in {ModuleKind.ATTENTION_K, ModuleKind.ATTENTION_V, ModuleKind.LORA_K, ModuleKind.LORA_V}:
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


class OraclePlannerStrategy(CacheStrategy):
    name = StrategyName.ORACLE_PLANNER

    def __init__(self, update_norm_threshold: float = 0.05) -> None:
        self.planner = CachePlanner(
            PlannerPolicy(update_norm_threshold=update_norm_threshold, allow_delta_correction=True)
        )

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        decision = self.planner.plan(target, update_norm=update_norm, version_gap=step)
        if decision.action is CacheAction.FULL_RECOMPUTE:
            return StrategyDecision(
                self.name,
                CacheAction.PARTIAL_RECOMPUTE if target.layer is not None else CacheAction.FULL_RECOMPUTE,
                CacheBlockState.INVALID,
                target.layer,
                "Oracle upper bound: choose the cheapest safe refresh action implied by target metadata.",
                recompute_fraction=0.0 if target.layer is not None else 1.0,
            )
        return StrategyDecision(
            self.name,
            decision.action,
            decision.state,
            decision.first_invalid_layer,
            f"Oracle upper bound: {decision.reason}",
            recompute_fraction=decision.recompute_fraction,
            reject_reuse=decision.reject_reuse,
        )


def build_strategy(name: str, *, refresh_period: int = 4, update_norm_threshold: float = 0.05) -> CacheStrategy:
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
    if parsed is StrategyName.ADAPTIVE:
        return AdaptiveStrategy(CachePlanner(PlannerPolicy(update_norm_threshold=update_norm_threshold)))
    if parsed is StrategyName.DELTA_CORRECTION:
        return DeltaCorrectionStrategy(update_norm_threshold=update_norm_threshold)
    if parsed is StrategyName.BASE_CACHE_REUSE:
        return BaseCacheReuseStrategy()
    if parsed is StrategyName.ADAPTER_SPECIFIC_CACHE:
        return AdapterSpecificCacheStrategy()
    if parsed is StrategyName.STATIC_BASE_DELTA:
        return StaticBaseDeltaStrategy(update_norm_threshold=update_norm_threshold)
    if parsed is StrategyName.ORACLE_PLANNER:
        return OraclePlannerStrategy(update_norm_threshold=update_norm_threshold)
    raise ValueError(f"Unsupported cache strategy: {name}")
