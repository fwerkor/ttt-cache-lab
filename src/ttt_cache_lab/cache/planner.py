from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

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
class PlannerRuntime:
    total_cache_bytes: int = 0
    candidate_cache_bytes: int = 0
    full_recompute_latency: float = 1.0
    reuse_latency: float | None = None
    delta_correction_latency: float | None = None
    partial_recompute_latency: float | None = None
    model_name: str = ""
    context_length: int = 0
    lora_rank: int = 0
    configured_update_norm: float = 0.0
    update_mode: str = ""

    def action_latency(self, action: CacheAction, *, target: UpdateTarget | None = None) -> float:
        if action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
            return self.full_recompute_latency
        if (
            action in {CacheAction.REUSE_EXACT, CacheAction.REUSE_FROZEN, CacheAction.REUSE_STALE}
            and self.reuse_latency is not None
        ):
            return self.reuse_latency
        if action is CacheAction.DELTA_CORRECT and self.delta_correction_latency is not None:
            return self.delta_correction_latency
        if action is CacheAction.PARTIAL_RECOMPUTE and self.partial_recompute_latency is not None:
            return self.partial_recompute_latency
        return self.full_recompute_latency * _action_cost(action, target=target)


@dataclass(frozen=True)
class FailureMapCell:
    update_target: str
    version_gap: int
    cache_strategy: str
    model_name: str
    context_length: int
    lora_rank: int
    configured_update_norm: float
    update_mode: str
    task_drop_vs_full: float
    logits_kl_mean: float
    top1_agreement_mean: float
    false_safe_rate: float

    def safe(self, policy: PlannerPolicy) -> bool:
        return (
            self.logits_kl_mean <= policy.safe_kl_threshold
            and self.top1_agreement_mean >= policy.safe_top1_threshold
            and self.task_drop_vs_full <= policy.safe_task_drop_threshold
            and self.false_safe_rate == 0.0
        )


@dataclass(frozen=True)
class PlannerPolicy:
    update_norm_threshold: float = 0.05
    version_gap_threshold: int = 8
    error_proxy_threshold: float = 0.25
    latency_budget_fraction: float = 1.0
    memory_budget_bytes: int | None = None
    failure_map_path: Path | None = None
    safe_kl_threshold: float = 0.05
    safe_top1_threshold: float = 0.99
    safe_task_drop_threshold: float = 0.01
    allow_delta_correction: bool = True
    allow_layerwise_recompute: bool = True
    reject_high_risk_reuse: bool = True
    use_version_id: bool = True
    use_target_rules: bool = True
    use_update_norm: bool = True
    periodic_refresh_interval: int | None = None


class FailureMapIndex:
    def __init__(self, cells: list[FailureMapCell]) -> None:
        self.cells = cells

    @classmethod
    def from_csv(cls, path: Path) -> FailureMapIndex:
        if not path.exists():
            raise FileNotFoundError(path)
        cells: list[FailureMapCell] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                cells.append(
                    FailureMapCell(
                        update_target=row.get("update_target", ""),
                        version_gap=int(float(row.get("version_gap", "0") or 0)),
                        cache_strategy=row.get("cache_strategy", ""),
                        model_name=row.get("model_name", ""),
                        context_length=int(float(row.get("context_length", "0") or 0)),
                        lora_rank=int(float(row.get("lora_rank", "0") or 0)),
                        configured_update_norm=float(
                            row.get("configured_update_norm", "0") or 0.0
                        ),
                        update_mode=row.get("update_mode", ""),
                        task_drop_vs_full=float(row.get("task_drop_vs_full", "0") or 0.0),
                        logits_kl_mean=float(row.get("logits_kl_mean", "0") or 0.0),
                        top1_agreement_mean=float(row.get("top1_agreement_mean", "0") or 0.0),
                        false_safe_rate=float(row.get("false_safe_rate", "0") or 0.0),
                    )
                )
        return cls(cells)

    def nearest(
        self,
        target: UpdateTarget,
        version_gap: int,
        *,
        runtime: PlannerRuntime,
    ) -> list[FailureMapCell]:
        target_cells = [cell for cell in self.cells if cell.update_target == target.raw]
        if not target_cells:
            target_cells = [cell for cell in self.cells if cell.update_target == target.kind.value]
        if not target_cells:
            return []
        compatible = [cell for cell in target_cells if _matches_runtime(cell, runtime)]
        if compatible:
            target_cells = compatible
        nearest_gap = min({cell.version_gap for cell in target_cells}, key=lambda gap: abs(gap - version_gap))
        return [cell for cell in target_cells if cell.version_gap == nearest_gap]


class CachePlanner:
    """Evidence-aware cache planner with quality, latency, and memory constraints."""

    def __init__(self, policy: PlannerPolicy | None = None) -> None:
        self.policy = policy or PlannerPolicy()
        self.failure_map = (
            FailureMapIndex.from_csv(self.policy.failure_map_path)
            if self.policy.failure_map_path is not None
            else None
        )

    def plan(
        self,
        target: UpdateTarget,
        *,
        update_norm: float,
        version_gap: int = 1,
        runtime: PlannerRuntime | None = None,
    ) -> PlannerDecision:
        runtime = runtime or PlannerRuntime()
        effective_gap = version_gap if self.policy.use_version_id else 1
        effective_norm = update_norm if self.policy.use_update_norm else 0.0
        if self.policy.use_version_id and version_gap == 0:
            return PlannerDecision(
                CacheBlockState.VALID_EXACT,
                CacheAction.REUSE_EXACT,
                "Cache version matches the current adapter version.",
            )

        map_decision = self._failure_map_decision(target, effective_gap, runtime=runtime)
        if map_decision is not None:
            return map_decision

        if (
            self.policy.periodic_refresh_interval is not None
            and effective_gap >= self.policy.periodic_refresh_interval
        ):
            return self._refresh_decision(
                target,
                "Periodic safety fallback reached its configured version-gap interval.",
                runtime=runtime,
            )

        proxy = self._error_proxy(target, effective_gap, effective_norm)
        if proxy > self.policy.error_proxy_threshold:
            return self._refresh_decision(
                target,
                f"Error proxy {proxy:.6g} exceeds threshold {self.policy.error_proxy_threshold:.6g}.",
                runtime=runtime,
            )

        if not self.policy.use_target_rules:
            if self._small_enough_for_delta(effective_gap, effective_norm):
                return PlannerDecision(
                    CacheBlockState.VALID_APPROX,
                    CacheAction.REUSE_STALE,
                    "Target-rule ablation treats every update as generic bounded staleness.",
                )
            return self._refresh_decision(
                target,
                "Target-rule ablation has no safe generic reuse rule.",
                runtime=runtime,
            )

        if target.kind is ModuleKind.OUTPUT_HEAD:
            return PlannerDecision(
                CacheBlockState.VALID_EXACT,
                CacheAction.REUSE_EXACT,
                "Output-head updates do not change historical hidden states or K/V tensors.",
            )

        if target.kind in {ModuleKind.ATTENTION_Q, ModuleKind.LORA_Q}:
            if self._high_gap_or_norm(effective_gap, effective_norm):
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
            if (
                self.policy.allow_delta_correction
                and self._small_enough_for_delta(effective_gap, effective_norm)
                and self._fits_latency_budget(
                    CacheAction.DELTA_CORRECT, runtime=runtime, target=target
                )
                and not self._memory_pressure(runtime)
            ):
                return PlannerDecision(
                    CacheBlockState.VALID_APPROX,
                    CacheAction.DELTA_CORRECT,
                    "K/V-affecting update remains inside the quality, latency, and memory delta region.",
                    first_invalid_layer=target.layer,
                    recompute_fraction=0.15,
                )
            return self._refresh_decision(
                target,
                "K/V-affecting update is outside the configured correction region or budget.",
                runtime=runtime,
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
            return self._refresh_decision(
                target,
                "State-changing module updates require recomputing downstream layers.",
                runtime=runtime,
            )

        if target.kind is ModuleKind.NORM:
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.REJECT_UPDATE if self.policy.reject_high_risk_reuse else CacheAction.FULL_RECOMPUTE,
                "Norm updates reject cache reuse because they affect downstream activations broadly.",
                first_invalid_layer=target.layer,
                recompute_fraction=1.0,
                reject_reuse=self.policy.reject_high_risk_reuse,
            )

        return PlannerDecision(
            CacheBlockState.INVALID,
            CacheAction.REJECT_UPDATE if self.policy.reject_high_risk_reuse else CacheAction.FULL_RECOMPUTE,
            "Unknown update target rejects cache reuse and falls back to a full refresh.",
            recompute_fraction=1.0,
            reject_reuse=self.policy.reject_high_risk_reuse,
        )

    def _failure_map_decision(
        self,
        target: UpdateTarget,
        version_gap: int,
        *,
        runtime: PlannerRuntime,
    ) -> PlannerDecision | None:
        if self.failure_map is None:
            return None
        cells = self.failure_map.nearest(target, version_gap, runtime=runtime)
        if not cells:
            return None
        safe = [cell for cell in cells if cell.safe(self.policy)]
        candidates: list[tuple[float, FailureMapCell, CacheAction]] = []
        for cell in safe:
            action = _strategy_action(cell.cache_strategy, target=target)
            if action is None:
                continue
            if action is CacheAction.DELTA_CORRECT and not self.policy.allow_delta_correction:
                continue
            if action is CacheAction.PARTIAL_RECOMPUTE and (
                not self.policy.allow_layerwise_recompute or target.layer is None
            ):
                continue
            if not self._fits_latency_budget(action, runtime=runtime, target=target):
                continue
            if self._memory_pressure(runtime) and action is CacheAction.DELTA_CORRECT:
                continue
            candidates.append((runtime.action_latency(action, target=target), cell, action))
        if not candidates:
            return self._refresh_decision(
                target,
                "E3 failure map contains no safe strategy within the configured budgets.",
                runtime=runtime,
            )
        _, cell, action = min(candidates, key=lambda item: item[0])
        return PlannerDecision(
            _action_state(action),
            action,
            (
                f"E3 failure map selected {cell.cache_strategy} at nearest measured gap "
                f"{cell.version_gap}: KL={cell.logits_kl_mean:.6g}, "
                f"top1={cell.top1_agreement_mean:.6g}, task_drop={cell.task_drop_vs_full:.6g}."
            ),
            first_invalid_layer=(
                target.layer
                if action in {CacheAction.PARTIAL_RECOMPUTE, CacheAction.DELTA_CORRECT}
                else None
            ),
            recompute_fraction=_action_cost(action, target=target),
        )

    def _refresh_decision(
        self,
        target: UpdateTarget,
        reason: str,
        *,
        runtime: PlannerRuntime,
    ) -> PlannerDecision:
        if (
            target.layer is not None
            and self.policy.allow_layerwise_recompute
            and self._fits_latency_budget(
                CacheAction.PARTIAL_RECOMPUTE, runtime=runtime, target=target
            )
        ):
            return PlannerDecision(
                CacheBlockState.INVALID,
                CacheAction.PARTIAL_RECOMPUTE,
                reason,
                first_invalid_layer=target.layer,
            )
        return PlannerDecision(
            CacheBlockState.INVALID,
            CacheAction.FULL_RECOMPUTE,
            reason,
            recompute_fraction=1.0,
        )

    def _small_enough_for_delta(self, version_gap: int, update_norm: float) -> bool:
        return update_norm <= self.policy.update_norm_threshold and version_gap <= self.policy.version_gap_threshold

    def _high_gap_or_norm(self, version_gap: int, update_norm: float) -> bool:
        return update_norm > self.policy.update_norm_threshold or version_gap > self.policy.version_gap_threshold

    def _error_proxy(self, target: UpdateTarget, version_gap: int, update_norm: float) -> float:
        risk = {
            ModuleKind.OUTPUT_HEAD: 0.0,
            ModuleKind.ATTENTION_Q: 0.1,
            ModuleKind.LORA_Q: 0.1,
            ModuleKind.ATTENTION_K: 0.8,
            ModuleKind.ATTENTION_V: 0.8,
            ModuleKind.ATTENTION_QV: 0.9,
            ModuleKind.LORA_K: 0.8,
            ModuleKind.LORA_V: 0.8,
            ModuleKind.LORA_QV: 0.9,
            ModuleKind.ATTENTION_O: 1.0,
            ModuleKind.ATTENTION_ATTN: 1.0,
            ModuleKind.MLP: 1.0,
            ModuleKind.LORA_O: 1.0,
            ModuleKind.LORA_ATTN: 1.0,
            ModuleKind.LORA_ALL_LATE: 1.0,
            ModuleKind.LORA_MLP: 1.0,
            ModuleKind.NORM: 1.5,
        }.get(target.kind, 2.0)
        return risk * max(1, version_gap) * max(0.0, update_norm)

    def _fits_latency_budget(
        self,
        action: CacheAction,
        *,
        runtime: PlannerRuntime,
        target: UpdateTarget | None = None,
    ) -> bool:
        full_latency = max(1e-9, runtime.full_recompute_latency)
        action_latency = runtime.action_latency(action, target=target)
        return action_latency / full_latency <= self.policy.latency_budget_fraction

    def _memory_pressure(self, runtime: PlannerRuntime) -> bool:
        limit = self.policy.memory_budget_bytes
        if limit is None:
            return False
        return runtime.total_cache_bytes + runtime.candidate_cache_bytes > limit


def _matches_runtime(cell: FailureMapCell, runtime: PlannerRuntime) -> bool:
    if cell.model_name and runtime.model_name and cell.model_name != runtime.model_name:
        return False
    if cell.context_length and runtime.context_length and cell.context_length != runtime.context_length:
        return False
    if cell.lora_rank and runtime.lora_rank and cell.lora_rank != runtime.lora_rank:
        return False
    if (
        cell.configured_update_norm
        and runtime.configured_update_norm
        and not _close(cell.configured_update_norm, runtime.configured_update_norm)
    ):
        return False
    return not (
        cell.update_mode
        and runtime.update_mode
        and cell.update_mode != runtime.update_mode
    )


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 1e-9 * scale


def _strategy_action(strategy: str, *, target: UpdateTarget) -> CacheAction | None:
    if strategy in {"stale_reuse", "frozen_reuse", "base_cache_reuse"}:
        return CacheAction.REUSE_STALE
    if strategy in {"delta_correction", "static_base_delta", "forkkv_base_delta", "lragent_adapter_cache"}:
        return CacheAction.DELTA_CORRECT
    if strategy in {"layerwise_recompute", "adaptive", "oracle_planner"}:
        return CacheAction.PARTIAL_RECOMPUTE if target.layer is not None else CacheAction.FULL_RECOMPUTE
    if strategy == "full_recompute":
        return CacheAction.FULL_RECOMPUTE
    return None


def _action_cost(action: CacheAction, *, target: UpdateTarget | None = None) -> float:
    if action in {CacheAction.REUSE_EXACT, CacheAction.REUSE_FROZEN, CacheAction.REUSE_STALE}:
        return 0.05
    if action is CacheAction.DELTA_CORRECT:
        return 0.15
    if action is CacheAction.PARTIAL_RECOMPUTE:
        if target is not None and target.layer is not None:
            return 0.5
        return 1.0
    if action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
        return 0.25
    return 1.0


def _action_state(action: CacheAction) -> CacheBlockState:
    if action is CacheAction.REUSE_EXACT:
        return CacheBlockState.VALID_EXACT
    if action is CacheAction.REUSE_FROZEN:
        return CacheBlockState.VALID_FROZEN
    if action in {CacheAction.REUSE_STALE, CacheAction.DELTA_CORRECT}:
        return CacheBlockState.VALID_APPROX
    return CacheBlockState.INVALID
