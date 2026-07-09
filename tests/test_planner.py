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
