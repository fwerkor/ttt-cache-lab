from __future__ import annotations

from dataclasses import dataclass

from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.updates.targets import ModuleKind, UpdateTarget


@dataclass(frozen=True)
class PlannerDecision:
    state: CacheBlockState
    action: CacheAction
    reason: str
    first_invalid_layer: int | None = None
    recompute_fraction: float = 0.0
    reject_reuse: bool = False


@dataclass(frozen=True)
class PlannerPolicy:
    update_norm_threshold: float = 0.05
    version_gap_threshold: int = 8
    allow_delta_correction: bool = True
    allow_layerwise_recompute: bool = True
    reject_high_risk_reuse: bool = True


class CachePlanner:
    """Parameter-aware cache validity planner.

    The planner is still conservative, but it now uses version gap and
    accumulated update norm instead of only target kind. Its decisions are
    intended to be measured and refined by E3/E4 rather than treated as final.
    """

    def __init__(self, policy: PlannerPolicy | None = None) -> None:
        self.policy = policy or PlannerPolicy()

    def plan(self, target: UpdateTarget, *, update_norm: float, version_gap: int = 1) -> PlannerDecision:
        if version_gap == 0:
            return PlannerDecision(
                CacheBlockState.VALID_EXACT,
                CacheAction.REUSE_EXACT,
                "Cache version matches the current adapter version.",
            )

        if target.kind is ModuleKind.OUTPUT_HEAD:
            return PlannerDecision(
                CacheBlockState.VALID_EXACT,
                CacheAction.REUSE_EXACT,
                "Output-head updates do not change historical hidden states or K/V tensors.",
            )

        if target.kind in {ModuleKind.ATTENTION_Q, ModuleKind.LORA_Q}:
            if self._high_gap_or_norm(version_gap, update_norm):
                return PlannerDecision(
                    CacheBlockState.VALID_APPROX,
                    CacheAction.REUSE_STALE,
                    "Q-only update is cache-compatible, but high gap/norm is tracked as bounded-stale risk.",
                )
            return PlannerDecision(
                CacheBlockState.VALID_FROZEN,
                CacheAction.REUSE_FROZEN,
                "Q-only updates are cache-compatible under frozen-evidence semantics.",
            )

        if target.kind in {
            ModuleKind.ATTENTION_K,
            ModuleKind.ATTENTION_V,
            ModuleKind.ATTENTION_QV,
            ModuleKind.LORA_K,
            ModuleKind.LORA_V,
            ModuleKind.LORA_QV,
        }:
            if self.policy.allow_delta_correction and self._small_enough_for_delta(version_gap, update_norm):
                return PlannerDecision(
                    CacheBlockState.VALID_APPROX,
                    CacheAction.DELTA_CORRECT,
                    "K/V-affecting updates directly change cache but remain in the delta-correction region.",
                    first_invalid_layer=target.layer,
                    recompute_fraction=0.15,
                )
            if self.policy.allow_layerwise_recompute:
                return PlannerDecision(
                    CacheBlockState.INVALID,
                    CacheAction.PARTIAL_RECOMPUTE,
                    "K/V-affecting updates left the delta region; refresh from the affected layer onward.",
                    first_invalid_layer=target.layer,
                    recompute_fraction=0.5 if target.layer is None else 0.0,
                )
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.FULL_RECOMPUTE,
                "K/V-affecting update cannot be corrected under current policy.",
                first_invalid_layer=target.layer,
                recompute_fraction=1.0,
            )

        if target.kind in {
            ModuleKind.ATTENTION_O,
            ModuleKind.ATTENTION_ATTN,
            ModuleKind.MLP,
            ModuleKind.LORA_O,
            ModuleKind.LORA_ATTN,
            ModuleKind.LORA_ALL_LATE,
            ModuleKind.LORA_MLP,
        }:
            if target.layer is not None and self.policy.allow_layerwise_recompute:
                return PlannerDecision(
                    CacheBlockState.INVALID,
                    CacheAction.PARTIAL_RECOMPUTE,
                    "State-changing module updates require recomputing downstream layers.",
                    first_invalid_layer=target.layer,
                    recompute_fraction=0.0,
                )
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.FULL_RECOMPUTE,
                "State-changing module update without layer information requires full recompute.",
                recompute_fraction=1.0,
            )

        if target.kind is ModuleKind.NORM:
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.FULL_RECOMPUTE,
                "Norm updates are treated as high-risk because they affect downstream activations broadly.",
                first_invalid_layer=target.layer,
                recompute_fraction=1.0,
                reject_reuse=self.policy.reject_high_risk_reuse,
            )

        return PlannerDecision(
            CacheBlockState.INVALID,
            CacheAction.FULL_RECOMPUTE,
            "Unknown update target; conservative full recompute.",
            recompute_fraction=1.0,
            reject_reuse=self.policy.reject_high_risk_reuse,
        )

    def _small_enough_for_delta(self, version_gap: int, update_norm: float) -> bool:
        return update_norm <= self.policy.update_norm_threshold and version_gap <= self.policy.version_gap_threshold

    def _high_gap_or_norm(self, version_gap: int, update_norm: float) -> bool:
        return update_norm > self.policy.update_norm_threshold or version_gap > self.policy.version_gap_threshold
