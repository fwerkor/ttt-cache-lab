from ttt_cache_lab.cache.planner import CachePlanner
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
    low_threshold = build_strategy("adaptive", update_norm_threshold=0.0).decide(
        target, step=1, update_norm=0.01
    )
    default_threshold = build_strategy("adaptive", update_norm_threshold=0.05).decide(
        target, step=1, update_norm=0.01
    )
    assert low_threshold.action is CacheAction.PARTIAL_RECOMPUTE
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
