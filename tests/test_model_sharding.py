from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from ttt_cache_lab.models.sharding import build_model_shard_plan, resolve_shard_device_ids


def test_qwen_64_layers_are_balanced_across_six_gpus() -> None:
    config = SimpleNamespace(model_type="qwen2", num_hidden_layers=64, tie_word_embeddings=False)
    plan = build_model_shard_plan(config, device_type="cuda", device_ids=[0, 1, 2, 3, 4, 5])
    counts = Counter(plan.layer_to_device)
    assert plan.input_device == "cuda:0"
    assert plan.device_map["model.embed_tokens"] == "cuda:0"
    assert plan.device_map["model.layers.63"] == "cuda:5"
    assert plan.device_map["model.norm"] == "cuda:5"
    assert plan.device_map["lm_head"] == "cuda:5"
    assert max(counts.values()) - min(counts.values()) <= 1
    assert list(dict.fromkeys(plan.layer_to_device)) == list(plan.devices)


def test_tied_gpt2_head_stays_with_embeddings() -> None:
    config = SimpleNamespace(model_type="gpt2", n_layer=4, tie_word_embeddings=True)
    plan = build_model_shard_plan(config, device_type="npu", device_ids=[2, 3])
    assert plan.device_map["transformer.wte"] == "npu:2"
    assert plan.device_map["lm_head"] == "npu:2"
    assert plan.layer_to_device == ("npu:2", "npu:2", "npu:3", "npu:3")


def test_sharding_rejects_unsupported_or_single_device_plans() -> None:
    config = SimpleNamespace(model_type="qwen2", num_hidden_layers=4, tie_word_embeddings=False)
    with pytest.raises(ValueError, match="CUDA or NPU"):
        build_model_shard_plan(config, device_type="cpu", device_ids=[0, 1])
    with pytest.raises(ValueError, match="at least two"):
        build_model_shard_plan(config, device_type="cuda", device_ids=[0])


def test_resolve_shard_ids_uses_all_visible_devices() -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3))
    assert resolve_shard_device_ids(fake_torch, device_type="cuda", configured=[]) == [0, 1, 2]
    assert resolve_shard_device_ids(fake_torch, device_type="cuda", configured=[1, 2]) == [1, 2]
