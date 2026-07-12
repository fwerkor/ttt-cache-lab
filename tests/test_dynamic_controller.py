from __future__ import annotations

import numpy as np

from ttt_cache_lab.cache.dynamic_controller import (
    CellRiskLedger,
    ContinuousRiskEstimate,
    DynamicBudgetPolicy,
    consensus_probe_score,
    estimate_continuous_risk,
    run_dynamic_budget_controller,
)


def _risk(score: float) -> ContinuousRiskEstimate:
    return ContinuousRiskEstimate(
        score=score,
        raw_score=score,
        uncertainty=0.0,
        control_score=score,
        cell_scores=np.asarray([[score]], dtype=np.float64),
    )


def test_continuous_risk_uses_projected_drift_attention_and_eligibility() -> None:
    policy = DynamicBudgetPolicy(
        risk_scale=1.0,
        attention_floor=0.0,
        uncertainty_weight=0.0,
    )
    estimate = estimate_continuous_risk(
        projected_drift=np.asarray([[1.0, 9.0]], dtype=np.float64),
        attention_mass=np.asarray([[1.0, 1.0]], dtype=np.float64),
        eligible=np.asarray([[True, False]], dtype=bool),
        policy=policy,
    )

    assert np.isclose(estimate.raw_score, 1.0)
    assert np.isclose(estimate.score, 1.0 - np.exp(-1.0))
    assert estimate.cell_scores.tolist() == [[1.0, 0.0]]
    assert estimate.control_score == estimate.score


def test_cell_risk_ledger_tracks_cumulative_drift_and_selective_refresh() -> None:
    ledger = CellRiskLedger((1, 2))
    ledger.observe(
        np.asarray([[0.2, 0.4]], dtype=np.float64),
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        current_version=3,
        cumulative=True,
    )
    ledger.refresh(np.asarray([[True, False]], dtype=bool))

    assert ledger.projected_drift.tolist() == [[0.0, 0.4]]
    assert ledger.last_refresh_version.tolist() == [[3, 0]]
    assert ledger.age().tolist() == [[0, 3]]

    ledger.observe(
        np.asarray([[0.1, 0.25]], dtype=np.float64),
        np.asarray([[0.4, 0.6]], dtype=np.float64),
        current_version=4,
        cumulative=True,
    )
    assert ledger.projected_drift.tolist() == [[0.1, 0.25]]



def test_consensus_probe_requires_every_reference_horizon_to_improve() -> None:
    stale = {"reference_nll": 2.0, "reference_nll_4": 4.0}

    improving = consensus_probe_score(
        stale_scores=stale,
        candidate_scores={"reference_nll": 1.8, "reference_nll_4": 3.2},
        scale_floor=1e-3,
    )
    vetoed = consensus_probe_score(
        stale_scores=stale,
        candidate_scores={"reference_nll": 1.0, "reference_nll_4": 4.1},
        scale_floor=1e-3,
    )

    assert np.isclose(improving, 0.9)
    assert vetoed > 1.0

def test_dynamic_controller_dead_zone_avoids_probe_work() -> None:
    policy = DynamicBudgetPolicy(max_cells=8, activation_threshold=0.65)
    evaluated: list[int] = []

    def evaluate(count: int) -> float:
        evaluated.append(count)
        return 0.0

    decision = run_dynamic_budget_controller(
        stale_score=1.0,
        risk=_risk(0.64),
        available_cells=8,
        evaluate_count=evaluate,
        policy=policy,
    )

    assert decision.activated_risk == 0.0
    assert decision.budget_cap == 0
    assert decision.selected_count == 0
    assert evaluated == []


def test_dynamic_controller_maps_low_continuous_risk_to_zero_budget() -> None:
    policy = DynamicBudgetPolicy(max_cells=8)
    evaluated: list[int] = []

    def evaluate(count: int) -> float:
        evaluated.append(count)
        return 0.0

    decision = run_dynamic_budget_controller(
        stale_score=1.0,
        risk=_risk(0.01),
        available_cells=8,
        evaluate_count=evaluate,
        policy=policy,
    )

    assert decision.selected_count == 0
    assert decision.budget_cap == 0
    assert decision.observed_steps == 0
    assert decision.stop_reason == "continuous_zero_budget"
    assert evaluated == []


def test_dynamic_controller_stops_when_risk_scaled_quality_target_is_met() -> None:
    policy = DynamicBudgetPolicy(
        max_cells=4,
        activation_threshold=0.0,
        base_target_reduction=0.1,
        max_target_reduction=0.3,
        min_accept_reduction=0.01,
        marginal_reduction_floor=0.0,
        patience=2,
    )
    scores = {1: 0.8, 2: 0.6, 3: 0.5}
    decision = run_dynamic_budget_controller(
        stale_score=1.0,
        risk=_risk(0.75),
        available_cells=4,
        evaluate_count=scores.__getitem__,
        policy=policy,
    )

    assert decision.budget_cap == 3
    assert decision.selected_count == 2
    assert decision.observed_steps == 2
    assert decision.achieved_relative_reduction == 0.4
    assert decision.stop_reason == "quality_target_met"


def test_dynamic_controller_keeps_best_prefix_and_rolls_back_after_plateau() -> None:
    policy = DynamicBudgetPolicy(
        max_cells=4,
        activation_threshold=0.0,
        base_target_reduction=0.9,
        max_target_reduction=0.9,
        min_accept_reduction=0.01,
        marginal_reduction_floor=0.0,
        patience=2,
    )
    scores = {1: 0.7, 2: 0.8, 3: 0.9, 4: 0.95}
    decision = run_dynamic_budget_controller(
        stale_score=1.0,
        risk=_risk(1.0),
        available_cells=4,
        evaluate_count=scores.__getitem__,
        policy=policy,
    )

    assert decision.selected_count == 1
    assert decision.selected_score == 0.7
    assert decision.observed_steps == 3
    assert decision.stop_reason == "marginal_plateau"


def test_dynamic_controller_rejects_probe_path_without_material_improvement() -> None:
    policy = DynamicBudgetPolicy(
        max_cells=2,
        activation_threshold=0.0,
        min_accept_reduction=0.05,
        marginal_reduction_floor=0.0,
        patience=2,
    )
    scores = {1: 0.99, 2: 1.01}
    decision = run_dynamic_budget_controller(
        stale_score=1.0,
        risk=_risk(1.0),
        available_cells=2,
        evaluate_count=scores.__getitem__,
        policy=policy,
    )

    assert decision.selected_count == 0
    assert decision.selected_score == 1.0
    assert decision.stop_reason == "insufficient_probe_improvement"
