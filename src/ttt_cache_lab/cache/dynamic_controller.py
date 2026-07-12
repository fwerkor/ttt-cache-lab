from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuousRiskEstimate:
    """Continuous cache-staleness estimate derived from per-cell projected drift."""

    score: float
    raw_score: float
    uncertainty: float
    control_score: float
    cell_scores: np.ndarray


@dataclass(frozen=True)
class DynamicBudgetPolicy:
    """Policy for continuous risk-to-budget control without system-side signals."""

    risk_scale: float = 0.002
    attention_floor: float = 0.01
    uncertainty_weight: float = 0.25
    activation_threshold: float = 0.65
    max_cells: int = 8
    budget_exponent: float = 1.0
    base_target_reduction: float = 0.05
    max_target_reduction: float = 0.35
    min_accept_reduction: float = 0.015
    marginal_reduction_floor: float = 0.002
    patience: int = 2
    objective_scale_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.risk_scale <= 0.0:
            raise ValueError("risk_scale must be positive")
        if self.attention_floor < 0.0:
            raise ValueError("attention_floor must be nonnegative")
        if not 0.0 <= self.uncertainty_weight <= 1.0:
            raise ValueError("uncertainty_weight must be in [0, 1]")
        if not 0.0 <= self.activation_threshold < 1.0:
            raise ValueError("activation_threshold must be in [0, 1)")
        if self.max_cells < 0:
            raise ValueError("max_cells must be nonnegative")
        if self.budget_exponent <= 0.0:
            raise ValueError("budget_exponent must be positive")
        if not 0.0 <= self.base_target_reduction <= 1.0:
            raise ValueError("base_target_reduction must be in [0, 1]")
        if not self.base_target_reduction <= self.max_target_reduction <= 1.0:
            raise ValueError("max_target_reduction must be in [base_target_reduction, 1]")
        if not 0.0 <= self.min_accept_reduction <= 1.0:
            raise ValueError("min_accept_reduction must be in [0, 1]")
        if self.marginal_reduction_floor < 0.0:
            raise ValueError("marginal_reduction_floor must be nonnegative")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.objective_scale_floor <= 0.0:
            raise ValueError("objective_scale_floor must be positive")


@dataclass(frozen=True)
class DynamicBudgetDecision:
    selected_count: int
    selected_score: float
    observed_steps: int
    budget_cap: int
    activated_risk: float
    target_relative_reduction: float
    achieved_relative_reduction: float
    stop_reason: str


class CellRiskLedger:
    """Per-cell state for cumulative projected drift and selective refresh.

    The preferred mode is ``cumulative=True``: callers supply the projected drift
    between the cell's cache version and the current parameter version. This
    avoids over-counting update norms and permits cancellation to be represented
    by the caller's cumulative projection.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("shape must contain positive dimensions")
        self.projected_drift = np.zeros(shape, dtype=np.float64)
        self.attention_mass = np.zeros(shape, dtype=np.float64)
        self.last_refresh_version = np.zeros(shape, dtype=np.int64)
        self.current_version = 0

    def observe(
        self,
        projected_drift: np.ndarray,
        attention_mass: np.ndarray,
        *,
        current_version: int,
        cumulative: bool = True,
    ) -> None:
        drift = np.asarray(projected_drift, dtype=np.float64)
        attention = np.asarray(attention_mass, dtype=np.float64)
        if drift.shape != self.projected_drift.shape or attention.shape != drift.shape:
            raise ValueError("projected drift and attention must match the ledger shape")
        if current_version < self.current_version:
            raise ValueError("current_version must be monotonic")
        drift = np.nan_to_num(np.abs(drift), nan=0.0, posinf=0.0, neginf=0.0)
        attention = np.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
        if cumulative:
            self.projected_drift[...] = drift
        else:
            self.projected_drift += drift
        self.attention_mass[...] = np.maximum(attention, 0.0)
        self.current_version = current_version

    def refresh(self, mask: np.ndarray, *, current_version: int | None = None) -> None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != self.projected_drift.shape:
            raise ValueError("refresh mask must match the ledger shape")
        version = self.current_version if current_version is None else current_version
        if version < self.current_version:
            raise ValueError("refresh version must not precede the observed version")
        self.projected_drift[selected] = 0.0
        self.last_refresh_version[selected] = version
        self.current_version = version

    def age(self) -> np.ndarray:
        return self.current_version - self.last_refresh_version

    def estimate(
        self,
        *,
        eligible: np.ndarray,
        policy: DynamicBudgetPolicy,
    ) -> ContinuousRiskEstimate:
        return estimate_continuous_risk(
            projected_drift=self.projected_drift,
            attention_mass=self.attention_mass,
            eligible=eligible,
            policy=policy,
        )


def estimate_continuous_risk(
    *,
    projected_drift: np.ndarray,
    attention_mass: np.ndarray,
    eligible: np.ndarray,
    policy: DynamicBudgetPolicy,
) -> ContinuousRiskEstimate:
    drift = np.asarray(projected_drift, dtype=np.float64)
    attention = np.asarray(attention_mass, dtype=np.float64)
    available = np.asarray(eligible, dtype=bool)
    if drift.shape != attention.shape or drift.shape != available.shape:
        raise ValueError("risk inputs must have identical shapes")

    safe_drift = np.nan_to_num(np.abs(drift), nan=0.0, posinf=0.0, neginf=0.0)
    safe_attention = np.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
    safe_attention = np.maximum(safe_attention, 0.0)
    cell_scores = np.where(
        available,
        safe_drift * (safe_attention + policy.attention_floor),
        0.0,
    )
    raw_score = float(np.sum(cell_scores))
    score = float(-math.expm1(-raw_score / policy.risk_scale))

    active = cell_scores[available]
    active_drift = safe_drift[available]
    uncertainty = _distribution_uncertainty(active, active_drift)
    control_score = float(
        np.clip(
            score + policy.uncertainty_weight * uncertainty * (1.0 - score),
            0.0,
            1.0,
        )
    )
    return ContinuousRiskEstimate(
        score=score,
        raw_score=raw_score,
        uncertainty=uncertainty,
        control_score=control_score,
        cell_scores=cell_scores,
    )


def normalized_probe_scores(
    *,
    stale_scores: dict[str, float],
    candidate_scores: dict[str, float],
    scale_floor: float,
) -> dict[str, float]:
    """Normalize candidate probe losses so stale reuse has score 1.0."""

    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")
    if not stale_scores or stale_scores.keys() != candidate_scores.keys():
        raise ValueError("stale and candidate probe scores must share nonempty keys")
    normalized: dict[str, float] = {}
    for name, stale_score in stale_scores.items():
        candidate_score = candidate_scores[name]
        if not math.isfinite(stale_score) or not math.isfinite(candidate_score):
            normalized[name] = math.inf
            continue
        scale = max(abs(stale_score), scale_floor)
        normalized[name] = 1.0 + (candidate_score - stale_score) / scale
    return normalized


def consensus_probe_score(
    *,
    stale_scores: dict[str, float],
    candidate_scores: dict[str, float],
    scale_floor: float,
) -> float:
    """Return the worst normalized candidate/stale score across probe horizons.

    Stale reuse has score 1.0. A candidate is below 1.0 only when every
    available reference horizon improves, so a single regressing horizon vetoes
    an otherwise favorable short probe.
    """

    normalized = normalized_probe_scores(
        stale_scores=stale_scores,
        candidate_scores=candidate_scores,
        scale_floor=scale_floor,
    )
    return max(normalized.values())


def dynamic_budget_parameters(
    *,
    risk: ContinuousRiskEstimate,
    available_cells: int,
    policy: DynamicBudgetPolicy,
) -> tuple[float, int, float]:
    if available_cells < 0:
        raise ValueError("available_cells must be nonnegative")
    activated_risk = max(
        0.0,
        (risk.control_score - policy.activation_threshold)
        / (1.0 - policy.activation_threshold),
    )
    continuous_cap = policy.max_cells * activated_risk**policy.budget_exponent
    budget_cap = min(available_cells, max(0, int(round(continuous_cap))))
    target_reduction = policy.base_target_reduction + activated_risk * (
        policy.max_target_reduction - policy.base_target_reduction
    )
    return activated_risk, budget_cap, target_reduction


def run_dynamic_budget_controller(
    *,
    stale_score: float,
    risk: ContinuousRiskEstimate,
    available_cells: int,
    evaluate_count: Callable[[int], float],
    policy: DynamicBudgetPolicy,
) -> DynamicBudgetDecision:
    if not math.isfinite(stale_score):
        raise ValueError("stale_score must be finite")

    activated_risk, budget_cap, target_reduction = dynamic_budget_parameters(
        risk=risk,
        available_cells=available_cells,
        policy=policy,
    )
    if budget_cap == 0:
        return DynamicBudgetDecision(
            selected_count=0,
            selected_score=stale_score,
            observed_steps=0,
            budget_cap=0,
            activated_risk=activated_risk,
            target_relative_reduction=target_reduction,
            achieved_relative_reduction=0.0,
            stop_reason="continuous_zero_budget",
        )

    scale = max(abs(stale_score), policy.objective_scale_floor)
    best_count = 0
    best_score = stale_score
    best_relative_reduction = 0.0
    stagnant_steps = 0
    observed_steps = 0
    stop_reason = "budget_cap_reached"

    for count in range(1, budget_cap + 1):
        score = float(evaluate_count(count))
        if not math.isfinite(score):
            stagnant_steps += 1
        else:
            previous_best = best_score
            if score < best_score:
                best_score = score
                best_count = count
                best_relative_reduction = max(0.0, (stale_score - score) / scale)
            marginal_reduction = max(0.0, (previous_best - best_score) / scale)
            if marginal_reduction <= policy.marginal_reduction_floor:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
        observed_steps = count
        if best_relative_reduction >= target_reduction:
            stop_reason = "quality_target_met"
            break
        if stagnant_steps >= policy.patience:
            stop_reason = "marginal_plateau"
            break

    if best_relative_reduction < policy.min_accept_reduction:
        return DynamicBudgetDecision(
            selected_count=0,
            selected_score=stale_score,
            observed_steps=observed_steps,
            budget_cap=budget_cap,
            activated_risk=activated_risk,
            target_relative_reduction=target_reduction,
            achieved_relative_reduction=0.0,
            stop_reason="insufficient_probe_improvement",
        )
    return DynamicBudgetDecision(
        selected_count=best_count,
        selected_score=best_score,
        observed_steps=observed_steps,
        budget_cap=budget_cap,
        activated_risk=activated_risk,
        target_relative_reduction=target_reduction,
        achieved_relative_reduction=best_relative_reduction,
        stop_reason=stop_reason,
    )


def _distribution_uncertainty(weighted: np.ndarray, drift: np.ndarray) -> float:
    if weighted.size <= 1:
        return 0.0
    weighted_total = float(np.sum(weighted))
    drift_total = float(np.sum(drift))
    if weighted_total <= 0.0 and drift_total <= 0.0:
        return 0.0
    weighted_prob = weighted / weighted_total if weighted_total > 0.0 else np.full(weighted.shape, 1.0 / weighted.size)
    drift_prob = drift / drift_total if drift_total > 0.0 else np.full(drift.shape, 1.0 / drift.size)
    disagreement = 0.5 * float(np.sum(np.abs(weighted_prob - drift_prob)))
    positive = weighted_prob[weighted_prob > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(weighted.size)
    return float(np.clip(0.5 * disagreement + 0.5 * entropy, 0.0, 1.0))
