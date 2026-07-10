# ruff: noqa: E402
# mypy: disable-error-code="no-untyped-call,import-untyped"

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    tokenizer.chat_template = (
        "{% for message in messages %}<|{{ message['role'] }}|> "
        "{{ message['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}<|assistant|> {% endif %}"
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


def _backend(model_dir: Path, *, use_chat_template: bool = False) -> HuggingFaceBackend:
    return HuggingFaceBackend(
        model_name_or_path=str(model_dir),
        device="cpu",
        torch_dtype="float32",
        max_length=32,
        trust_remote_code=False,
        use_chat_template=use_chat_template,
        seed=7,
    )


def test_chat_template_prompt_preparation_preserves_context_and_format(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir, use_chat_template=True)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha",
            metadata={"max_generation_tokens": 4},
        ),
        context_length=24,
    )
    prepared = backend._prepared_input_ids[sample.prompt]
    assert prepared.shape[1] == 24
    assert sample.metadata["prompt_format"] == "chat_template"
    assert backend.use_chat_template is True
    output = backend.prefill(sample.prompt)
    assert output.extras is not None
    assert output.extras["token_length"] == 24


def test_manual_decode_stops_at_eos_and_reports_actual_token_count(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha",
            metadata={"max_generation_tokens": 8},
        ),
        context_length=16,
    )
    state = backend._encode_prompt(sample.prompt)
    eos = int(backend.tokenizer.eos_token_id)
    assert eos in backend._stop_token_ids

    class EosModel:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            logits = torch.zeros((1, 1, backend.tokenizer.vocab_size), dtype=torch.float32)
            logits[0, 0, eos] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=kwargs["past_key_values"])

    backend.model = EosModel()
    _, generated_text, _, generated_tokens = backend._generate_answer(state, ())
    assert generated_text == ""
    assert generated_tokens == 1


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

    exact_decision = StrategyDecision(
        StrategyName.ADAPTIVE,
        CacheAction.REUSE_EXACT,
        CacheBlockState.VALID_EXACT,
        None,
        "integration",
    )
    reused = backend.apply_cache_strategy(
        baseline=baseline,
        full=full,
        updated=baseline,
        decision=exact_decision,
    )
    assert reused.extras is not None
    assert isinstance(reused.extras["hidden_states"], tuple)

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
        baseline=reused,
        full=full,
        updated=baseline,
        decision=partial_decision,
    )
    assert partial.extras is not None
    assert partial.extras["partial_mode"] == "native_llama_like_layer_restart"
    assert partial.extras["cache_bytes"] > 0
    assert len(partial.extras["hidden_states"]) == backend.num_layers + 1
    assert partial.logits.shape == full.logits.shape
    np.testing.assert_allclose(partial.logits, full.logits, rtol=1e-5, atol=1e-5)

    repeated_partial = backend.apply_cache_strategy(
        baseline=partial,
        full=full,
        updated=baseline,
        decision=partial_decision,
    )
    assert repeated_partial.extras is not None
    assert repeated_partial.extras["partial_mode"] == "native_llama_like_layer_restart"
    assert len(repeated_partial.extras["hidden_states"]) == backend.num_layers + 1


def test_attention_capture_uses_decode_only_eager_and_restores_backend(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    backend.configure_metrics(capture_attention=True)
    original_implementation = backend._attention_implementation()

    output = backend.prefill("key is alpha Answer :")

    assert output.extras is not None
    attention_summary = output.extras["attention_summary"]
    assert attention_summary.shape[0] == backend.num_layers
    assert attention_summary.shape[1] > 0
    assert backend._attention_implementation() == original_implementation


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



def test_direct_update_uses_all_matching_parameters_and_global_norm(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    baseline = backend.prefill("key is alpha Answer :")
    target = parse_update_target("attention.k", num_layers=backend.num_layers)
    matching = backend._select_parameters(target)
    assert len(matching) == backend.num_layers
    backend.simulate_update(baseline, target, update_norm=0.125)
    measured = torch.sqrt(
        sum(torch.sum(delta.detach().double() ** 2) for _, delta in backend._deltas)
    ).item()
    assert measured == pytest.approx(0.125, rel=1e-6)



def test_alora_reuses_base_prefix_and_recomputes_adapter_suffix(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    raw = TaskSample(
        prompt="key is alpha <ADAPTER> Answer :",
        answer="alpha",
        metadata={"adapter_activation_marker": "<ADAPTER>"},
    )
    sample = backend.prepare_sample(raw, context_length=16)
    target = parse_update_target("lora.k:1", num_layers=backend.num_layers)
    backend.prepare_update_target(target, rank=2, alpha=4.0)
    baseline = backend.prefill(sample.prompt)
    backend.train_lora_step(
        sample,
        target,
        rank=2,
        alpha=4.0,
        learning_rate=0.05,
        target_update_norm=0.02,
    )
    full = backend.full_recompute(sample.prompt, baseline)
    decision = StrategyDecision(
        StrategyName.ALORA_PREFIX_REUSE,
        CacheAction.ALORA_SUFFIX_RECOMPUTE,
        CacheBlockState.VALID_EXACT,
        1,
        "integration",
    )
    output = backend.apply_cache_strategy(
        baseline=baseline,
        full=full,
        updated=baseline,
        decision=decision,
    )
    assert output.extras is not None
    assert output.extras["cache_mode"] == "alora_base_prefix_suffix_recompute"
    assert output.extras["alora_activation_boundary"] < output.extras["token_length"]
    assert output.extras["cache_maintenance_latency"] >= 0.0
    assert output.extras["cache_bytes"] > 0


def test_external_generation_length_is_capped_by_sample_metadata(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    raw = TaskSample(
        prompt="key is alpha Answer :",
        answer="alpha alpha alpha",
        metadata={"max_generation_tokens": 1},
    )
    sample = backend.prepare_sample(raw, context_length=16)
    assert backend._sample_answer_token_counts[sample.prompt] == 1
    output = backend.prefill(sample.prompt)
    assert output.extras is not None
    assert output.extras["generated_tokens"] == 1


def test_generation_budget_does_not_leak_reference_answer_length(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    raw = TaskSample(
        prompt="key is alpha Answer :",
        answer="alpha",
        metadata={"max_generation_tokens": 8},
    )
    sample = backend.prepare_sample(raw, context_length=16)
    assert backend._sample_answer_token_counts[sample.prompt] == 8
