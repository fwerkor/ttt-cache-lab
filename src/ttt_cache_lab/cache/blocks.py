from __future__ import annotations

from dataclasses import dataclass

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
    """Index complete per-layer cache entries by adapter identity and version."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int], VersionedCacheEntry] = {}

    def get(self, adapter_id: str, adapter_version: int) -> VersionedCacheEntry | None:
        return self._entries.get((adapter_id, adapter_version))

    def put(self, adapter_id: str, adapter_version: int, entry: VersionedCacheEntry) -> None:
        if any(block.adapter_version > adapter_version for block in entry.blocks):
            raise ValueError("Cache block version cannot be newer than the cache index")
        if any(block.adapter_id != adapter_id for block in entry.blocks):
            raise ValueError("Cache block adapter id does not match the cache index")
        self._entries[(adapter_id, adapter_version)] = entry

    def versions(self, adapter_id: str) -> tuple[int, ...]:
        return tuple(sorted(version for key, version in self._entries if key == adapter_id))

    def total_block_count(self) -> int:
        return sum(len(entry.blocks) for entry in self._entries.values())
