# ruff: noqa: E402
# mypy: disable-error-code="no-untyped-call,import-untyped"

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.hf_integration

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.models.hf import HuggingFaceBackend
from ttt_cache_lab.updates.targets import ModuleKind, UpdateTarget, parse_update_target


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("tiny-llama")
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[EOS]": 2,
        "filler": 3,
        "key": 4,
        "is": 5,
        "alpha": 6,
        "Answer": 7,
        ":": 8,
    }
    tokenizer_object = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(output)

    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=vocab["[PAD]"],
        eos_token_id=vocab["[EOS]"],
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(output)
    return output


def _backend(model_dir: Path) -> HuggingFaceBackend:
    return HuggingFaceBackend(
        model_name_or_path=str(model_dir),
        device="cpu",
        torch_dtype="float32",
        max_length=32,
        trust_remote_code=False,
        seed=7,
    )


def test_hf_lora_delta_and_native_layer_restart(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    sample = backend.prepare_sample(
        TaskSample(prompt="key is alpha Answer :", answer="alpha", metadata={}),
        context_length=16,
    )
    target = parse_update_target("lora.k:1", num_layers=backend.num_layers)
    assert backend.prepare_update_target(target, rank=2, alpha=4.0) == 1

    baseline = backend.prefill(sample.prompt)
    assert baseline.extras is not None
    assert baseline.extras["cache_bytes"] > baseline.cache_tensor.nbytes
    assert baseline.extras["token_length"] == 16

    update_norm = backend.train_lora_step(
        sample,
        target,
        rank=2,
        alpha=4.0,
        learning_rate=0.05,
    )
    assert update_norm > 0.0
    full = backend.full_recompute(sample.prompt, baseline)

    delta_decision = StrategyDecision(
        StrategyName.DELTA_CORRECTION,
        CacheAction.DELTA_CORRECT,
        CacheBlockState.VALID_APPROX,
        1,
        "integration",
    )
    delta = backend.apply_cache_strategy(
        baseline=baseline,
        full=full,
        updated=baseline,
        decision=delta_decision,
    )
    assert delta.extras is not None
    assert delta.extras["delta_mode"] == "lora_weight_delta"
    assert delta.extras["cache_maintenance_latency"] >= 0.0
    assert delta.extras["strategy_latency"] >= delta.extras["decode_latency"]

    partial_decision = StrategyDecision(
        StrategyName.LAYERWISE_RECOMPUTE,
        CacheAction.PARTIAL_RECOMPUTE,
        CacheBlockState.INVALID,
        1,
        "integration",
    )
    partial = backend.apply_cache_strategy(
        baseline=baseline,
        full=full,
        updated=baseline,
        decision=partial_decision,
    )
    assert partial.extras is not None
    assert partial.extras["partial_mode"] == "native_llama_like_layer_restart"
    assert partial.extras["cache_bytes"] > 0
    assert partial.logits.shape == full.logits.shape


def test_layer_specific_updates_never_fall_back_to_other_layers(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    impossible_lora = UpdateTarget(kind=ModuleKind.LORA_K, layer=99, raw="lora.k:99")
    assert backend.setup_lora(impossible_lora, rank=2, alpha=4.0) == 0
    with pytest.raises(ValueError, match="No HF parameters matched"):
        backend.simulate_update(
            backend.prefill("key is alpha Answer :"),
            UpdateTarget(kind=ModuleKind.ATTENTION_K, layer=99, raw="attention.k:99"),
            update_norm=0.01,
        )
