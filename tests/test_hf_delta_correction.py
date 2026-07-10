from types import SimpleNamespace
from typing import Any, cast

import pytest

from ttt_cache_lab.models.hf import HuggingFaceBackend
from ttt_cache_lab.models.lora import make_lora_linear


def test_lora_linear_computes_weight_delta_output() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    base = nn.Linear(3, 2, bias=False)
    module = make_lora_linear(torch, nn, base, rank=1, alpha=1.0)
    with torch.no_grad():
        module.lora_a.fill_(1.0)
        module.lora_b.zero_()
    old_state = module.lora_state()
    with torch.no_grad():
        module.lora_b.fill_(2.0)
    x = torch.ones(1, 4, 3)
    delta = module.lora_delta_output(x, old_state)
    assert delta.shape == (1, 4, 2)
    assert torch.allclose(delta, torch.full((1, 4, 2), 6.0))


def test_hf_weight_delta_patch_updates_kv_without_full_reference() -> None:
    torch = pytest.importorskip("torch")

    class FakeLora:
        lora_name = "model.layers.0.self_attn.k_proj"

        def lora_delta_output(self, cached_input: Any, old_state: dict[str, Any]) -> Any:
            del old_state
            return torch.ones(cached_input.shape[0], cached_input.shape[1], 4)

        def lora_state(self) -> dict[str, Any]:
            return {"a": torch.zeros(1, 4), "b": torch.zeros(4, 1), "scaling": 1.0}

    backend = cast(Any, object.__new__(HuggingFaceBackend))
    backend._lora_modules = [FakeLora()]
    old_past = ((torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2)),)
    old_cache = {
        "model.layers.0.self_attn.k_proj": {
            "input": torch.zeros(1, 2, 4),
            "a": torch.zeros(1, 4),
            "b": torch.zeros(4, 1),
            "scaling": 1.0,
            "layer": 0,
            "projection": "k",
        }
    }
    corrected, new_cache = HuggingFaceBackend._apply_lora_weight_delta_to_past(
        backend,
        old_past,
        old_cache,
        split_layer=0,
    )
    assert corrected is not None
    assert new_cache
    assert torch.allclose(corrected[0][0], torch.ones(1, 2, 2, 2))
    assert torch.allclose(corrected[0][1], torch.zeros(1, 2, 2, 2))


def test_hf_key_delta_is_rotated_before_cache_update() -> None:
    torch = pytest.importorskip("torch")

    class FakeRotary:
        def __call__(self, x: Any, position_ids: Any) -> tuple[Any, Any]:
            del x
            cos = torch.ones(position_ids.shape[0], position_ids.shape[1], 2)
            sin = torch.zeros_like(cos)
            cos[:, 1, :] = 0.0
            sin[:, 1, :] = 1.0
            return cos, sin

    backend = cast(Any, object.__new__(HuggingFaceBackend))
    backend.torch = torch
    backend.model = SimpleNamespace(model=SimpleNamespace(rotary_emb=FakeRotary()))
    key_delta = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    position_ids = torch.tensor([[0, 1]])

    rotated = HuggingFaceBackend._apply_rotary_to_key_delta(
        backend,
        key_delta,
        position_ids=position_ids,
    )

    assert rotated is not None
    expected = torch.tensor([[[[1.0, 2.0], [-4.0, 3.0]]]])
    assert torch.allclose(rotated, expected)


def test_hf_key_delta_refuses_unpositioned_rotary_cache_update() -> None:
    torch = pytest.importorskip("torch")

    class FakeRotary:
        def __call__(self, x: Any, position_ids: Any) -> tuple[Any, Any]:
            return torch.ones_like(x), torch.zeros_like(x)

    backend = cast(Any, object.__new__(HuggingFaceBackend))
    backend.torch = torch
    backend.model = SimpleNamespace(model=SimpleNamespace(rotary_emb=FakeRotary()))

    rotated = HuggingFaceBackend._apply_rotary_to_key_delta(
        backend,
        torch.ones(1, 1, 2, 2),
        position_ids=None,
    )

    assert rotated is None
