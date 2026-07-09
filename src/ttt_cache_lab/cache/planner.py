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


@dataclass(frozen=True)
class PlannerPolicy:
    update_norm_threshold: float = 0.05
    allow_delta_correction: bool = True
    allow_layerwise_recompute: bool = True


class CachePlanner:
    """Parameter-aware cache validity planner.

    The first version is intentionally conservative. It encodes the hypotheses
    to be validated by experiments instead of pretending that the policy is final.
    """

    def __init__(self, policy: PlannerPolicy | None = None) -> None:
        self.policy = policy or PlannerPolicy()

    def plan(self, target: UpdateTarget, *, update_norm: float) -> PlannerDecision:
        if target.kind is ModuleKind.OUTPUT_HEAD:
            return PlannerDecision(
                CacheBlockState.VALID_EXACT,
                CacheAction.REUSE_EXACT,
                "Output-head updates do not change historical hidden states or K/V tensors.",
            )

        if target.kind in {ModuleKind.ATTENTION_Q, ModuleKind.LORA_Q}:
            return PlannerDecision(
                CacheBlockState.VALID_FROZEN,
                CacheAction.REUSE_FROZEN,
                "Q-only updates are cache-compatible under frozen-evidence semantics.",
            )

        if target.kind in {ModuleKind.ATTENTION_K, ModuleKind.ATTENTION_V, ModuleKind.LORA_K, ModuleKind.LORA_V}:
            if self.policy.allow_delta_correction and update_norm <= self.policy.update_norm_threshold:
                return PlannerDecision(
                    CacheBlockState.VALID_APPROX,
                    CacheAction.DELTA_CORRECT,
                    "K/V updates directly affect cache but small low-rank changes may be correctable.",
                    first_invalid_layer=target.layer,
                )
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.PARTIAL_RECOMPUTE,
                "K/V updates invalidate cache from the updated layer onward.",
                first_invalid_layer=target.layer,
            )

        if target.kind in {ModuleKind.ATTENTION_O, ModuleKind.MLP, ModuleKind.LORA_MLP}:
            if target.layer is not None and self.policy.allow_layerwise_recompute:
                return PlannerDecision(
                    CacheBlockState.INVALID,
                    CacheAction.PARTIAL_RECOMPUTE,
                    "State-changing module updates require recomputing downstream layers.",
                    first_invalid_layer=target.layer,
                )
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.FULL_RECOMPUTE,
                "State-changing module update without layer information requires full recompute.",
            )

        if target.kind is ModuleKind.NORM:
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.FULL_RECOMPUTE,
                "Norm updates are treated as high-risk because they affect downstream activations broadly.",
                first_invalid_layer=target.layer,
            )

        return PlannerDecision(
            CacheBlockState.INVALID,
            CacheAction.FULL_RECOMPUTE,
            "Unknown update target; conservative full recompute.",
        )
