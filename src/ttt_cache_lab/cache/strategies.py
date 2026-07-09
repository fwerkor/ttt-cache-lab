from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ttt_cache_lab.cache.planner import CachePlanner, PlannerDecision, PlannerPolicy
from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.updates.targets import UpdateTarget


class StrategyName(StrEnum):
    FULL_RECOMPUTE = "full_recompute"
    STALE_REUSE = "stale_reuse"
    FROZEN_REUSE = "frozen_reuse"
    PERIODIC_REFRESH = "periodic_refresh"
    LAYERWISE_RECOMPUTE = "layerwise_recompute"
    DELTA_CORRECTION = "delta_correction"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class StrategyDecision:
    strategy: StrategyName
    action: CacheAction
    state: CacheBlockState
    first_invalid_layer: int | None
    reason: str


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
            )
        return StrategyDecision(
            self.name,
            CacheAction.REUSE_STALE,
            CacheBlockState.VALID_APPROX,
            None,
            f"Reuse until refresh period {self.period} is reached.",
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
        )


class AdaptiveStrategy(CacheStrategy):
    name = StrategyName.ADAPTIVE

    def __init__(self, planner: CachePlanner | None = None) -> None:
        self.planner = planner or CachePlanner()

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        decision: PlannerDecision = self.planner.plan(target, update_norm=update_norm)
        return StrategyDecision(
            self.name,
            decision.action,
            decision.state,
            decision.first_invalid_layer,
            decision.reason,
        )


class DeltaCorrectionStrategy(CacheStrategy):
    name = StrategyName.DELTA_CORRECTION

    def __init__(self, update_norm_threshold: float = 0.05) -> None:
        self.planner = CachePlanner(PlannerPolicy(update_norm_threshold=update_norm_threshold))

    def decide(self, target: UpdateTarget, *, step: int, update_norm: float) -> StrategyDecision:
        decision: PlannerDecision = self.planner.plan(target, update_norm=update_norm)
        return StrategyDecision(
            self.name,
            decision.action,
            decision.state,
            decision.first_invalid_layer,
            f"Delta-correction baseline: {decision.reason}",
        )


def build_strategy(name: str, *, refresh_period: int = 4, update_norm_threshold: float = 0.05) -> CacheStrategy:
    parsed = StrategyName(name)
    if parsed is StrategyName.FULL_RECOMPUTE:
        return FullRecomputeStrategy()
    if parsed is StrategyName.STALE_REUSE:
        return StaleReuseStrategy()
    if parsed is StrategyName.FROZEN_REUSE:
        return FrozenReuseStrategy()
    if parsed is StrategyName.PERIODIC_REFRESH:
        return PeriodicRefreshStrategy(period=refresh_period)
    if parsed is StrategyName.LAYERWISE_RECOMPUTE:
        return LayerwiseRecomputeStrategy()
    if parsed is StrategyName.ADAPTIVE:
        return AdaptiveStrategy()
    if parsed is StrategyName.DELTA_CORRECTION:
        return DeltaCorrectionStrategy(update_norm_threshold=update_norm_threshold)
    raise ValueError(f"Unsupported cache strategy: {name}")
