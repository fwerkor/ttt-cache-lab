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
        "The": 9,
        "quiet": 10,
        "forest": 11,
        "contains": 12,
        "ordinary": 13,
        "prose": 14,
        "about": 15,
        "weather": 16,
        "windows": 17,
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
        num_hidden_layers=3,
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
    assert sample.metadata["neutral_padding_tokens"] > 0
    assert backend.use_chat_template is True
    output = backend.prefill(sample.prompt)
    assert output.extras is not None
    assert output.extras["token_length"] == 24


def test_neutral_padding_is_deterministic_diverse_and_special_free(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    first = backend._neutral_padding_ids(24, dtype=torch.long, prompt="prompt-a")
    repeated = backend._neutral_padding_ids(24, dtype=torch.long, prompt="prompt-a")
    second = backend._neutral_padding_ids(24, dtype=torch.long, prompt="prompt-b")

    assert torch.equal(first, repeated)
    assert first.shape == (1, 24)
    assert len(set(first[0].tolist())) > 1
    assert not set(first[0].tolist()) & set(backend.tokenizer.all_special_ids)
    assert not torch.equal(first, second)


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


def test_output_head_update_supports_tied_embeddings(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    backend.model.config.tie_word_embeddings = True
    backend.model.tie_weights()
    output_weight = backend.model.get_output_embeddings().weight
    input_weight = backend.model.get_input_embeddings().weight
    assert output_weight is input_weight

    target = parse_update_target("output_head", num_layers=backend.num_layers)
    selected = backend._select_parameters(target)
    assert selected == [output_weight]

    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha",
            metadata={"max_generation_tokens": 4},
        ),
        context_length=16,
    )
    baseline = backend.prefill(sample.prompt)
    before = output_weight.detach().clone()
    backend.simulate_update(baseline, target, update_norm=0.01)
    applied = torch.linalg.vector_norm((output_weight.detach() - before).float()).item()
    assert applied == pytest.approx(0.01, rel=1e-4, abs=1e-6)
    backend.restore_after_update()
    assert torch.allclose(output_weight.detach(), before, rtol=1e-5, atol=1e-6)


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

    window_decision = StrategyDecision(
        StrategyName.WINDOWED_RECOMPUTE,
        CacheAction.PARTIAL_RECOMPUTE,
        CacheBlockState.VALID_APPROX,
        1,
        "integration",
        last_recomputed_layer=2,
    )
    windowed = backend.apply_cache_strategy(
        baseline=reused,
        full=full,
        updated=baseline,
        decision=window_decision,
    )
    assert windowed.extras is not None
    assert windowed.extras["partial_mode"] == "native_llama_like_finite_window_restart"
    assert windowed.extras["partial_start_layer"] == 1
    assert windowed.extras["partial_end_layer"] == 2
    assert windowed.extras["partial_window_layers"] == 1
    assert full.extras is not None
    baseline_layers = backend._past_as_layers(reused.extras["past_key_values"])
    full_layers = backend._past_as_layers(full.extras["past_key_values"])
    window_layers = backend._past_as_layers(windowed.extras["past_key_values"])
    for layer_id in range(backend.num_layers):
        expected = full_layers[layer_id] if layer_id == 1 else baseline_layers[layer_id]
        for actual_tensor, expected_tensor in zip(window_layers[layer_id], expected, strict=True):
            if actual_tensor is None or expected_tensor is None:
                assert actual_tensor is expected_tensor
            else:
                assert torch.allclose(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)


def test_blockwise_cache_splice_selects_layer_token_cells(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha",
            metadata={"max_generation_tokens": 1},
        ),
        context_length=16,
    )
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
    assert baseline.extras is not None and full.extras is not None

    old_layers = backend._past_as_layers(baseline.extras["past_key_values"])
    current_layers = backend._past_as_layers(full.extras["past_key_values"])
    token_count = int(old_layers[0][0].shape[-2])
    block_size = 4
    block_count = int(np.ceil(token_count / block_size))
    mask = np.zeros((backend.num_layers, block_count), dtype=bool)
    mask[1, 1] = True
    mixed = backend.probe_blockwise_cache_splice(
        baseline=baseline,
        full=full,
        block_mask=mask,
        block_size=block_size,
    )
    assert mixed.extras is not None
    assert mixed.extras["selected_block_cells"] == 1
    mixed_layers = backend._past_as_layers(mixed.extras["past_key_values"])
    for layer_index in range(backend.num_layers):
        for item_index in (0, 1):
            actual = mixed_layers[layer_index][item_index]
            expected = old_layers[layer_index][item_index].clone()
            if layer_index == 1:
                expected[..., 4:8, :] = current_layers[layer_index][item_index][..., 4:8, :]
            assert torch.equal(actual, expected)

    empty = backend.probe_blockwise_cache_splice(
        baseline=baseline,
        full=full,
        block_mask=np.zeros_like(mask),
        block_size=block_size,
    )
    complete = backend.probe_blockwise_cache_splice(
        baseline=baseline,
        full=full,
        block_mask=np.ones_like(mask),
        block_size=block_size,
    )
    stale = backend._reuse_old_prefix_cache(baseline)
    np.testing.assert_allclose(empty.logits, stale.logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(complete.logits, full.logits, rtol=1e-5, atol=1e-5)


def test_block_sparse_lora_key_delta_repairs_only_selected_tokens(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    backend.configure_metrics(capture_attention=True)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha",
            metadata={"max_generation_tokens": 1},
        ),
        context_length=16,
    )
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
    assert baseline.extras is not None and full.extras is not None
    stale = backend.probe_blockwise_cache_splice(
        baseline=baseline,
        full=full,
        block_mask=np.zeros((backend.num_layers, 4), dtype=bool),
        block_size=4,
    )
    scores = backend.blockwise_lora_delta_scores(
        baseline=baseline,
        stale=stale,
        block_size=4,
    )
    assert scores["available"].shape == (backend.num_layers, 4)
    assert scores["available"][1].all()
    assert not scores["available"][0].any()
    assert not scores["available"][2].any()
    assert np.all(scores["stale_attention_mass"][1] >= 0.0)

    mask = np.zeros((backend.num_layers, 4), dtype=bool)
    mask[1, 1] = True
    sparse = backend.probe_blockwise_lora_delta(
        baseline=baseline,
        block_mask=mask,
        block_size=4,
    )
    assert sparse.extras is not None
    assert sparse.extras["cache_mode"] == "block_sparse_lora_weight_delta"
    assert sparse.extras["selected_direct_cells"] == 1
    old_layers = backend._past_as_layers(baseline.extras["past_key_values"])
    full_layers = backend._past_as_layers(full.extras["past_key_values"])
    sparse_layers = backend._past_as_layers(sparse.extras["past_key_values"])
    for layer_index in range(backend.num_layers):
        for item_index in (0, 1):
            actual = sparse_layers[layer_index][item_index]
            expected = old_layers[layer_index][item_index].clone()
            if layer_index == 1 and item_index == 0:
                expected[..., 4:8, :] = full_layers[layer_index][item_index][..., 4:8, :]
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    all_direct = np.zeros_like(mask)
    all_direct[1, :] = True
    repaired = backend.probe_blockwise_lora_delta(
        baseline=baseline,
        block_mask=all_direct,
        block_size=4,
    )
    assert repaired.extras is not None
    repaired_layers = backend._past_as_layers(repaired.extras["past_key_values"])
    assert torch.allclose(repaired_layers[1][0], full_layers[1][0], rtol=1e-5, atol=1e-6)
    assert torch.equal(repaired_layers[1][1], old_layers[1][1])


def test_block_sparse_lora_value_scores_emit_signed_corrections(
    tiny_llama_dir: Path,
) -> None:
    backend = _backend(tiny_llama_dir)
    backend.configure_metrics(capture_attention=True)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha",
            metadata={"max_generation_tokens": 1},
        ),
        context_length=16,
    )
    target = parse_update_target("lora.v:1", num_layers=backend.num_layers)
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
    stale = backend.probe_blockwise_cache_splice(
        baseline=baseline,
        full=full,
        block_mask=np.zeros((backend.num_layers, 4), dtype=bool),
        block_size=4,
    )

    scores = backend.blockwise_lora_delta_scores(
        baseline=baseline,
        stale=stale,
        block_size=4,
    )

    assert scores["signed_correction_vectors"].shape[:2] == (
        backend.num_layers,
        4,
    )
    assert scores["signed_correction_available"][1].all()
    assert not scores["signed_correction_available"][0].any()
    assert np.all(np.isfinite(scores["signed_correction_vectors"]))
    assert np.any(scores["signed_correction_norm"][1] > 0.0)
    assert np.all(np.abs(scores["signed_total_alignment"][1]) <= 1.0 + 1e-9)
    assert np.all(scores["signed_cancellation_ratio"][1] >= 0.0)
    assert np.all(scores["signed_cancellation_ratio"][1] <= 1.0 + 1e-9)


def test_reference_sequence_scoring_matches_first_token_logits(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha beta",
            metadata={"max_generation_tokens": 1},
        ),
        context_length=16,
    )
    baseline = backend.prefill(sample.prompt)
    assert baseline.extras is not None
    token_ids = backend.tokenizer(sample.answer, add_special_tokens=False)["input_ids"]
    assert len(token_ids) >= 2

    profile_metrics = backend.score_reference_sequence(
        baseline=baseline,
        past=baseline.extras["past_key_values"],
        reference_token_ids=token_ids,
        probe_lengths=(1, 2),
        return_profile=True,
    )
    reference_profile = profile_metrics.pop("_reference_log_probabilities")
    metrics = backend.score_reference_sequence(
        baseline=baseline,
        past=baseline.extras["past_key_values"],
        reference_token_ids=token_ids,
        probe_lengths=(1, 2),
        reference_log_probabilities=reference_profile,
    )
    logits = torch.tensor(baseline.logits[0], dtype=torch.float32)
    expected_first_nll = float(-torch.log_softmax(logits, dim=-1)[token_ids[0]])

    assert metrics["reference_probe_tokens"] == 2
    assert metrics["reference_token_nll_1"] == pytest.approx(expected_first_nll, rel=1e-5)
    assert np.isfinite(metrics["reference_token_nll_2"])
    assert metrics["reference_token_kl_1"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["reference_token_kl_2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["reference_sequence_kl"] == pytest.approx(0.0, abs=1e-6)
    assert len(metrics["reference_token_nll_values"]) == 2
    assert len(metrics["reference_token_kl_values"]) == 2


def test_prompt_suffix_scoring_matches_direct_causal_nll(
    tiny_llama_dir: Path,
) -> None:
    backend = _backend(tiny_llama_dir)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="alpha beta gamma delta epsilon zeta",
            answer="answer tokens must not be used",
            metadata={"max_generation_tokens": 1},
        ),
        context_length=16,
    )
    baseline = backend.prefill(sample.prompt)
    assert baseline.extras is not None
    state = baseline.extras["prompt_state"]
    metrics = backend.score_prompt_suffix(
        baseline=baseline,
        past=baseline.extras["past_key_values"],
        probe_lengths=(1, 2),
    )

    token_count = int(state.input_ids.shape[1])
    max_tokens = min(2, token_count - 1)
    suffix_start = token_count - max_tokens
    with torch.no_grad():
        direct = backend.model(
            input_ids=state.input_ids[:, : token_count - 1],
            use_cache=False,
        )
    direct_logits = direct.logits[:, suffix_start - 1 : token_count - 1, :].float()
    targets = state.input_ids[:, suffix_start:token_count]
    direct_nll = -torch.gather(
        torch.log_softmax(direct_logits, dim=-1),
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    assert metrics["prompt_suffix_probe_tokens"] == max_tokens
    assert metrics["prompt_suffix_nll_1"] == pytest.approx(
        float(direct_nll[0, 0]),
        rel=1e-5,
    )
    assert metrics["prompt_suffix_nll_2"] == pytest.approx(
        float(direct_nll[0].mean()),
        rel=1e-5,
    )
    assert len(metrics["prompt_suffix_token_nll_values"]) == max_tokens


def test_prompt_anchor_scoring_matches_direct_causal_nll(
    tiny_llama_dir: Path,
) -> None:
    backend = _backend(tiny_llama_dir)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="alpha beta gamma delta epsilon zeta eta theta",
            answer="unused answer",
            metadata={"max_generation_tokens": 1},
        ),
        context_length=16,
    )
    baseline = backend.prefill(sample.prompt)
    assert baseline.extras is not None
    state = baseline.extras["prompt_state"]
    cached_tokens = int(state.prefix_ids.shape[1])
    block_size = 2
    block_count = (cached_tokens + block_size - 1) // block_size
    mask = np.zeros((1, block_count), dtype=bool)
    mask[0, 0] = True
    metrics = backend.score_prompt_anchors(
        baseline=baseline,
        past=baseline.extras["past_key_values"],
        block_mask=mask,
        block_size=block_size,
        probe_lengths=(1, 2),
    )

    target_start = min(block_size, cached_tokens)
    max_tokens = min(2, int(state.input_ids.shape[1]) - target_start)
    with torch.no_grad():
        direct = backend.model(
            input_ids=state.input_ids[:, : target_start + max_tokens - 1],
            use_cache=False,
        )
    direct_logits = direct.logits[
        :, target_start - 1 : target_start + max_tokens - 1, :
    ].float()
    targets = state.input_ids[:, target_start : target_start + max_tokens]
    direct_nll = -torch.gather(
        torch.log_softmax(direct_logits, dim=-1),
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    assert metrics["prompt_anchor_count"] == 1
    assert metrics["prompt_anchor_probe_tokens"] == max_tokens
    assert metrics["prompt_anchor_b0_nll_1"] == pytest.approx(
        float(direct_nll[0, 0]),
        rel=1e-5,
    )
    assert metrics["prompt_anchor_b0_nll_2"] == pytest.approx(
        float(direct_nll[0].mean()),
        rel=1e-5,
    )


def test_lora_target_initialization_is_reproducible_and_updates_accumulate(
    tiny_llama_dir: Path,
) -> None:
    backend = _backend(tiny_llama_dir)
    sample = backend.prepare_sample(
        TaskSample(
            prompt="key is alpha Answer :",
            answer="alpha beta",
            metadata={"max_generation_tokens": 2},
        ),
        context_length=16,
    )
    target = parse_update_target("lora.k:1", num_layers=backend.num_layers)
    backend.prepare_update_target(target, rank=2, alpha=4.0)
    initial_fingerprint = backend.adapter_state_fingerprint()

    backend.train_lora_step(
        sample,
        target,
        rank=2,
        alpha=4.0,
        learning_rate=0.05,
        target_update_norm=0.02,
    )
    first_update_fingerprint = backend.adapter_state_fingerprint()
    backend.train_lora_step(
        sample,
        target,
        rank=2,
        alpha=4.0,
        learning_rate=0.05,
        target_update_norm=0.02,
    )
    second_update_fingerprint = backend.adapter_state_fingerprint()

    assert first_update_fingerprint != initial_fingerprint
    assert second_update_fingerprint != first_update_fingerprint

    backend.restore_after_update()
    backend.prepare_update_target(target, rank=2, alpha=4.0)
    assert backend.adapter_state_fingerprint() == initial_fingerprint


def test_attention_capture_uses_decode_only_eager_and_restores_backend(tiny_llama_dir: Path) -> None:
    backend = _backend(tiny_llama_dir)
    backend.configure_metrics(capture_attention=True)
    original_implementation = backend._attention_implementation()

    output = backend.prefill("key is alpha Answer :")

    assert output.extras is not None
    attention_summary = output.extras["attention_summary"]
    attention_head_summary = output.extras["attention_head_summary"]
    attention_input_summary = output.extras["attention_input_summary"]
    attention_output_summary = output.extras["attention_output_summary"]
    assert attention_summary.shape[0] == backend.num_layers
    assert attention_summary.shape[1] > 0
    assert attention_head_summary.shape[0] == backend.num_layers
    assert attention_head_summary.shape[1] == 2
    np.testing.assert_allclose(
        attention_summary,
        np.mean(attention_head_summary, axis=1),
        rtol=1e-6,
        atol=1e-7,
    )
    assert attention_input_summary.shape == (backend.num_layers, backend.hidden_size)
    assert attention_output_summary.shape == (backend.num_layers, backend.hidden_size)
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
    measured = torch.sqrt(sum(torch.sum(delta.detach().double() ** 2) for _, delta in backend._deltas)).item()
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
