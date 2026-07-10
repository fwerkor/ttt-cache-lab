from __future__ import annotations

import numpy as np
import pytest

from ttt_cache_lab.cache.blocks import CacheBlockMetadata, VersionedCacheEntry, VersionedCacheManager
from ttt_cache_lab.cache.semantics import CacheBlockState, CacheSemantics
from ttt_cache_lab.models.interface import BackendOutput


def _block(*, adapter_id: str = "a", version: int = 1, layer: int = 0) -> CacheBlockMetadata:
    return CacheBlockMetadata(
        token_start=0,
        token_end=32,
        layer_id=layer,
        base_model_id="toy",
        adapter_id=adapter_id,
        adapter_version=version,
        cached_step=version,
        update_target="lora.k",
        accumulated_update_norm=0.1,
        state=CacheBlockState.VALID_EXACT,
        semantics=CacheSemantics.EXACT_CURRENT,
        precision="float32",
        attention_implementation="eager",
    )


def _output(version: int) -> BackendOutput:
    return BackendOutput(
        logits=np.zeros((1, 2)),
        cache_tensor=np.zeros((2, 2, 2)),
        hidden_tensor=np.zeros((2, 2)),
        parameter_version=version,
    )


def test_versioned_cache_manager_indexes_adapter_versions() -> None:
    manager = VersionedCacheManager()
    entry = VersionedCacheEntry(_output(1), (_block(layer=0), _block(layer=1)))
    manager.put("a", 1, entry)
    assert manager.get("a", 1) == entry
    assert manager.versions("a") == (1,)
    assert manager.total_block_count() == 2


def test_versioned_cache_manager_accepts_older_unaffected_blocks() -> None:
    manager = VersionedCacheManager()
    entry = VersionedCacheEntry(_output(2), (_block(version=1, layer=0), _block(version=2, layer=1)))
    manager.put("a", 2, entry)
    assert manager.get("a", 2) == entry


def test_versioned_cache_manager_rejects_future_block_metadata() -> None:
    manager = VersionedCacheManager()
    with pytest.raises(ValueError, match="newer"):
        manager.put("a", 2, VersionedCacheEntry(_output(2), (_block(version=3),)))
