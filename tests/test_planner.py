from ttt_cache_lab.cache.planner import CachePlanner
from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.updates.targets import parse_update_target


def test_q_update_uses_frozen_reuse() -> None:
    decision = CachePlanner().plan(parse_update_target("attention.q"), update_norm=0.01)
    assert decision.action is CacheAction.REUSE_FROZEN
    assert decision.state is CacheBlockState.VALID_FROZEN


def test_small_lora_k_update_uses_delta() -> None:
    decision = CachePlanner().plan(parse_update_target("lora.k:2"), update_norm=0.01)
    assert decision.action is CacheAction.DELTA_CORRECT
    assert decision.first_invalid_layer == 2


def test_norm_update_full_recompute() -> None:
    decision = CachePlanner().plan(parse_update_target("norm:1"), update_norm=0.01)
    assert decision.action is CacheAction.FULL_RECOMPUTE


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
