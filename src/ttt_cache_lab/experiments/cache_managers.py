from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ttt_cache_lab.cache.blocks import VersionedCacheManager
from ttt_cache_lab.cache.strategies import CacheStrategy


def build_strategy_managers(
    strategies: Sequence[CacheStrategy],
    *,
    max_cache_bytes: int | None,
    max_cache_entries: int | None,
    eviction_policy: Literal["lru", "fifo"],
) -> dict[str, VersionedCacheManager]:
    return {
        str(strategy.name): VersionedCacheManager(
            max_cache_bytes=max_cache_bytes,
            max_cache_entries=max_cache_entries,
            eviction_policy=eviction_policy,
        )
        for strategy in strategies
    }
