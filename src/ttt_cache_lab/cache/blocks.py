from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from ttt_cache_lab.cache.semantics import CacheBlockState, CacheSemantics
from ttt_cache_lab.models.interface import BackendOutput


@dataclass(frozen=True)
class CacheBlockMetadata:
    token_start: int
    token_end: int
    layer_id: int
    base_model_id: str
    adapter_id: str
    adapter_version: int
    cached_step: int
    update_target: str
    accumulated_update_norm: float
    state: CacheBlockState
    semantics: CacheSemantics
    precision: str
    attention_implementation: str


@dataclass(frozen=True)
class VersionedCacheEntry:
    output: BackendOutput
    blocks: tuple[CacheBlockMetadata, ...]


class VersionedCacheManager:
    """Capacity-bounded cache index shared across adapter identities and versions."""

    def __init__(
        self,
        *,
        max_cache_bytes: int | None = None,
        max_cache_entries: int | None = None,
        eviction_policy: Literal["lru", "fifo"] = "lru",
    ) -> None:
        if max_cache_bytes is not None and max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        if max_cache_entries is not None and max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be positive")
        self.max_cache_bytes = max_cache_bytes
        self.max_cache_entries = max_cache_entries
        self.eviction_policy = eviction_policy
        self._entries: OrderedDict[tuple[str, int], VersionedCacheEntry] = OrderedDict()
        self._entry_bytes: dict[tuple[str, int], int] = {}
        self._total_cache_bytes = 0
        self._eviction_count = 0

    def get(self, adapter_id: str, adapter_version: int) -> VersionedCacheEntry | None:
        key = (adapter_id, adapter_version)
        entry = self._entries.get(key)
        if entry is not None and self.eviction_policy == "lru":
            self._entries.move_to_end(key)
        return entry

    def put(self, adapter_id: str, adapter_version: int, entry: VersionedCacheEntry) -> bool:
        if any(block.adapter_version > adapter_version for block in entry.blocks):
            raise ValueError("Cache block version cannot be newer than the cache index")
        if any(block.adapter_id != adapter_id for block in entry.blocks):
            raise ValueError("Cache block adapter id does not match the cache index")
        key = (adapter_id, adapter_version)
        size = _entry_nbytes(entry)
        if self.max_cache_bytes is not None and size > self.max_cache_bytes:
            self.remove(adapter_id, adapter_version)
            return False

        previous = self._entries.pop(key, None)
        if previous is not None:
            self._total_cache_bytes -= self._entry_bytes.pop(key)
        self._entries[key] = entry
        self._entry_bytes[key] = size
        self._total_cache_bytes += size
        self._evict_to_capacity(protected_key=key)
        return key in self._entries

    def remove(self, adapter_id: str, adapter_version: int) -> bool:
        key = (adapter_id, adapter_version)
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        self._total_cache_bytes -= self._entry_bytes.pop(key)
        return True

    def versions(self, adapter_id: str) -> tuple[int, ...]:
        return tuple(sorted(version for key, version in self._entries if key == adapter_id))

    def entry_count(self) -> int:
        return len(self._entries)

    def total_block_count(self) -> int:
        return sum(len(entry.blocks) for entry in self._entries.values())

    def total_cache_bytes(self) -> int:
        return self._total_cache_bytes

    def eviction_count(self) -> int:
        return self._eviction_count

    def _evict_to_capacity(self, *, protected_key: tuple[str, int]) -> None:
        while self._over_capacity() and self._entries:
            victim = next(iter(self._entries))
            if victim == protected_key and len(self._entries) == 1:
                break
            if victim == protected_key:
                self._entries.move_to_end(victim)
                victim = next(iter(self._entries))
            self._entries.pop(victim)
            self._total_cache_bytes -= self._entry_bytes.pop(victim)
            self._eviction_count += 1

    def _over_capacity(self) -> bool:
        return bool(
            (self.max_cache_entries is not None and len(self._entries) > self.max_cache_entries)
            or (self.max_cache_bytes is not None and self._total_cache_bytes > self.max_cache_bytes)
        )


def _entry_nbytes(entry: VersionedCacheEntry) -> int:
    extras = entry.output.extras or {}
    cache_bytes = extras.get("cache_bytes")
    if isinstance(cache_bytes, int | float):
        return max(0, int(cache_bytes))
    return int(entry.output.cache_tensor.nbytes + entry.output.hidden_tensor.nbytes)
