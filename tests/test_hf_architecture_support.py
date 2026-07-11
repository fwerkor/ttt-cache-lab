from __future__ import annotations

from types import SimpleNamespace

from ttt_cache_lab.models.hf import HuggingFaceBackend
from ttt_cache_lab.updates.targets import ModuleKind


def _backend_for_config(config: object) -> HuggingFaceBackend:
    backend = object.__new__(HuggingFaceBackend)
    backend.model = SimpleNamespace(config=config)
    return backend


def test_nested_text_config_drives_gemma3_dimensions() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=34,
        hidden_size=2560,
        intermediate_size=10240,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    backend = _backend_for_config(SimpleNamespace(model_type="gemma3", text_config=text_config))
    assert backend._infer_num_layers_from_config(backend.model.config) == 34
    assert backend._hidden_size() == 2560
    assert backend._intermediate_size(2560) == 10240


def test_gemma3_update_targets_ignore_vision_tower_modules() -> None:
    backend = _backend_for_config(SimpleNamespace(model_type="gemma3"))
    assert backend._is_text_decoder_module("model.language_model.layers.0.self_attn.q_proj")
    assert backend._is_text_decoder_module("lm_head")
    assert not backend._is_text_decoder_module("model.vision_tower.encoder.layers.0.self_attn.q_proj")


def test_nested_gemma3_decoder_layers_are_discovered() -> None:
    layers = [object(), object()]
    backend = _backend_for_config(SimpleNamespace(model_type="gemma3"))
    backend.model.model = SimpleNamespace(language_model=SimpleNamespace(layers=layers))
    assert backend._decoder_layers() == (layers, "gemma3")


def test_qwen_moe_module_matching_separates_sparse_components() -> None:
    backend = _backend_for_config(SimpleNamespace(model_type="qwen2_moe"))
    router = "model.layers.12.mlp.gate"
    shared = "model.layers.12.mlp.shared_expert.up_proj"
    shared_gate = "model.layers.12.mlp.shared_expert_gate"
    routed = "model.layers.12.mlp.experts.gate_up_proj"

    assert backend._module_matches_target(router, ModuleKind.MOE_ROUTER)
    assert backend._module_matches_target(router, ModuleKind.LORA_MOE_ROUTER)
    assert not backend._module_matches_target(shared_gate, ModuleKind.MOE_ROUTER)

    assert backend._module_matches_target(shared, ModuleKind.MOE_SHARED_EXPERT)
    assert backend._module_matches_target(shared, ModuleKind.LORA_MOE_SHARED_EXPERT)
    assert not backend._module_matches_target(shared_gate, ModuleKind.MOE_SHARED_EXPERT)

    assert backend._module_matches_target(routed, ModuleKind.MOE_ROUTED_EXPERTS)
    assert not backend._module_matches_target(shared, ModuleKind.MOE_ROUTED_EXPERTS)
