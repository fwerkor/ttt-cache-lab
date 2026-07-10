from pathlib import Path

from ttt_cache_lab.cache.planner import CachePlanner, PlannerPolicy, PlannerRuntime
from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.updates.targets import parse_update_target


def test_zero_gap_is_exact() -> None:
    decision = CachePlanner().plan(parse_update_target("lora.k"), update_norm=0.01, version_gap=0)
    assert decision.action is CacheAction.REUSE_EXACT
    assert decision.state is CacheBlockState.VALID_EXACT


def test_q_update_uses_frozen_reuse() -> None:
    decision = CachePlanner().plan(parse_update_target("attention.q"), update_norm=0.01)
    assert decision.action is CacheAction.REUSE_FROZEN
    assert decision.state is CacheBlockState.VALID_FROZEN


def test_large_gap_q_update_is_bounded_stale() -> None:
    decision = CachePlanner().plan(parse_update_target("attention.q"), update_norm=0.01, version_gap=16)
    assert decision.action is CacheAction.REUSE_STALE
    assert decision.state is CacheBlockState.VALID_APPROX


def test_small_lora_k_update_uses_delta() -> None:
    decision = CachePlanner().plan(parse_update_target("lora.k:2"), update_norm=0.01)
    assert decision.action is CacheAction.DELTA_CORRECT
    assert decision.first_invalid_layer == 2


def test_large_gap_lora_k_update_uses_partial_recompute() -> None:
    decision = CachePlanner().plan(parse_update_target("lora.k:2"), update_norm=0.01, version_gap=16)
    assert decision.action is CacheAction.PARTIAL_RECOMPUTE
    assert decision.first_invalid_layer == 2


def test_large_gap_all_layer_lora_k_uses_full_recompute() -> None:
    decision = CachePlanner().plan(parse_update_target("lora.k"), update_norm=0.01, version_gap=16)
    assert decision.action is CacheAction.FULL_RECOMPUTE
    assert decision.first_invalid_layer is None
    assert decision.recompute_fraction == 1.0


def test_layerwise_strategy_requires_layer_boundary() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    decision = build_strategy("layerwise_recompute").decide(
        parse_update_target("lora.k"), step=1, update_norm=0.01
    )
    assert decision.action is CacheAction.FULL_RECOMPUTE
    assert decision.recompute_fraction == 1.0


def test_norm_update_full_recompute() -> None:
    decision = CachePlanner().plan(parse_update_target("norm:1"), update_norm=0.01)
    assert decision.action is CacheAction.REJECT_UPDATE
    assert decision.reject_reuse


def test_delta_correction_strategy_keeps_own_identity() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    decision = build_strategy("delta_correction").decide(parse_update_target("lora.k"), step=1, update_norm=0.01)
    assert str(decision.strategy) == "delta_correction"
    assert decision.action is CacheAction.DELTA_CORRECT


def test_strategy_update_norm_threshold_is_configurable() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    target = parse_update_target("lora.k")
    low_threshold = build_strategy("adaptive", update_norm_threshold=0.0).decide(target, step=1, update_norm=0.01)
    default_threshold = build_strategy("adaptive", update_norm_threshold=0.05).decide(target, step=1, update_norm=0.01)
    assert low_threshold.action is CacheAction.FULL_RECOMPUTE
    assert default_threshold.action is CacheAction.DELTA_CORRECT


def test_threshold_refresh_strategy() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    strategy = build_strategy("threshold_refresh", update_norm_threshold=0.02)
    assert strategy.decide(parse_update_target("lora.k"), step=1, update_norm=0.01).action is CacheAction.REUSE_STALE
    assert strategy.decide(parse_update_target("lora.k"), step=1, update_norm=0.03).action is CacheAction.FULL_RECOMPUTE


def test_static_baseline_strategies_are_buildable() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    for name in (
        "no_adaptation",
        "base_cache_reuse",
        "adapter_specific_cache",
        "static_base_delta",
        "oracle_planner",
    ):
        decision = build_strategy(name).decide(parse_update_target("lora.k"), step=1, update_norm=0.01)
        assert str(decision.strategy) == name


def test_adaptive_component_ablation_strategies() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    target = parse_update_target("lora.k:2")
    assert (
        build_strategy("adaptive_no_version").decide(target, step=0, update_norm=0.01).action
        is not CacheAction.REUSE_EXACT
    )
    assert (
        build_strategy("adaptive_no_target").decide(target, step=1, update_norm=0.01).action is CacheAction.REUSE_STALE
    )
    assert (
        build_strategy("adaptive_no_norm").decide(target, step=1, update_norm=100.0).action is CacheAction.DELTA_CORRECT
    )
    assert (
        build_strategy("adaptive_no_delta").decide(target, step=1, update_norm=0.01).action
        is CacheAction.PARTIAL_RECOMPUTE
    )
    assert (
        build_strategy("adaptive_no_partial").decide(target, step=1, update_norm=1.0).action
        is CacheAction.FULL_RECOMPUTE
    )


def test_adaptive_periodic_fallback_ablation() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    target = parse_update_target("attention.q")
    full = build_strategy("adaptive", refresh_period=4).decide(target, step=8, update_norm=0.01)
    no_periodic = build_strategy("adaptive_no_periodic", refresh_period=4).decide(target, step=8, update_norm=0.01)
    assert full.action is CacheAction.FULL_RECOMPUTE
    assert no_periodic.action is CacheAction.REUSE_FROZEN



def test_related_work_baselines_have_distinct_actions() -> None:
    from ttt_cache_lab.cache.strategies import build_strategy

    target = parse_update_target("lora.k:1")
    alora = build_strategy("alora_prefix_reuse").decide(target, step=1, update_norm=0.01)
    lragent = build_strategy("lragent_adapter_cache").decide(target, step=1, update_norm=0.01)
    forkkv = build_strategy("forkkv_base_delta").decide(target, step=1, update_norm=0.01)
    assert alora.action is CacheAction.ALORA_SUFFIX_RECOMPUTE
    assert lragent.action is CacheAction.FULL_RECOMPUTE
    assert forkkv.action is CacheAction.DELTA_CORRECT
    assert str(alora.strategy) == "alora_prefix_reuse"
    assert str(lragent.strategy) == "lragent_adapter_cache"
    assert str(forkkv.strategy) == "forkkv_base_delta"



def _failure_map(path: Path, rows: list[str]) -> Path:
    path.write_text(
        "update_target,version_gap,cache_strategy,task_drop_vs_full,logits_kl_mean,"
        "top1_agreement_mean,false_safe_rate\n" + "".join(rows),
        encoding="utf-8",
    )
    return path


def test_planner_reads_safe_action_from_failure_map(tmp_path: Path) -> None:
    failure_map = _failure_map(
        tmp_path / "failure_map.csv",
        [
            "lora.k:2,4,stale_reuse,0.0,0.001,1.0,0.0\n",
            "lora.k:2,4,delta_correction,0.0,0.01,1.0,0.0\n",
            "lora.k:2,4,full_recompute,0.0,0.0,1.0,0.0\n",
        ],
    )
    planner = CachePlanner(PlannerPolicy(failure_map_path=failure_map))
    decision = planner.plan(parse_update_target("lora.k:2"), update_norm=0.1, version_gap=5)
    assert decision.action is CacheAction.REUSE_STALE
    assert "E3 failure map" in decision.reason


def test_planner_failure_map_skips_unsafe_reuse(tmp_path: Path) -> None:
    failure_map = _failure_map(
        tmp_path / "failure_map.csv",
        [
            "lora.k:2,2,stale_reuse,0.1,0.2,0.0,1.0\n",
            "lora.k:2,2,delta_correction,0.0,0.01,1.0,0.0\n",
        ],
    )
    planner = CachePlanner(PlannerPolicy(failure_map_path=failure_map))
    decision = planner.plan(parse_update_target("lora.k:2"), update_norm=0.01, version_gap=2)
    assert decision.action is CacheAction.DELTA_CORRECT


def test_planner_version_gap_threshold_is_configurable() -> None:
    target = parse_update_target("lora.k:2")
    strict = CachePlanner(PlannerPolicy(version_gap_threshold=1)).plan(
        target, update_norm=0.01, version_gap=2
    )
    relaxed = CachePlanner(PlannerPolicy(version_gap_threshold=4)).plan(
        target, update_norm=0.01, version_gap=2
    )
    assert strict.action is CacheAction.PARTIAL_RECOMPUTE
    assert relaxed.action is CacheAction.DELTA_CORRECT


def test_planner_error_proxy_forces_refresh() -> None:
    decision = CachePlanner(PlannerPolicy(error_proxy_threshold=0.001)).plan(
        parse_update_target("lora.k:2"), update_norm=0.01, version_gap=1
    )
    assert decision.action is CacheAction.PARTIAL_RECOMPUTE
    assert "Error proxy" in decision.reason


def test_planner_memory_pressure_disables_delta() -> None:
    decision = CachePlanner(PlannerPolicy(memory_budget_bytes=100)).plan(
        parse_update_target("lora.k:2"),
        update_norm=0.01,
        version_gap=1,
        runtime=PlannerRuntime(total_cache_bytes=90, candidate_cache_bytes=20),
    )
    assert decision.action is CacheAction.PARTIAL_RECOMPUTE
    assert "budget" in decision.reason


def test_planner_latency_budget_disables_delta() -> None:
    decision = CachePlanner(PlannerPolicy(latency_budget_fraction=0.1)).plan(
        parse_update_target("lora.k:2"), update_norm=0.01, version_gap=1
    )
    assert decision.action is CacheAction.FULL_RECOMPUTE
