from __future__ import annotations

from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.experiments.blockwise import (
    _attention_residual_metrics,
    _beam_sparse_objective_masks,
    _compact_evaluation_output,
    _completed_condition_keys,
    _condition_seed,
    _Evaluation,
    _expand_causal_wedge,
    _expand_downstream_columns,
    _generation_sample,
    _greedy_sparse_objective_masks,
    _joint_sparse_search_point,
    _logit_selection_metrics,
    _mask_from_order,
    _read_jsonl,
    _read_rows,
    _record,
    _reference_token_id,
    _SearchPoint,
    _signed_residual_best_prefix,
    _signed_residual_greedy_order,
    _sparse_objective_score,
    _swap_refine_sparse_objective_masks,
    _write_jsonl,
    _write_rows,
    _zero_probe_baseline_stale_kl,
)
from ttt_cache_lab.metrics.tensor import kl_divergence
from ttt_cache_lab.models.interface import BackendOutput


def test_signed_residual_greedy_uses_direction_and_tracks_energy() -> None:
    vectors = np.asarray(
        [
            [
                [2.0, 0.0],
                [1.0, 0.0],
                [-0.5, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    available = np.ones((1, 3), dtype=bool)

    order, marginals, initial_energy = _signed_residual_greedy_order(
        vectors,
        available,
    )

    assert order == [(0, 0), (0, 1), (0, 2)]
    assert initial_energy == 6.25
    assert marginals[0] == 6.0
    assert marginals[1] == 0.0
    mask = _mask_from_order((available.shape[0], available.shape[1]), order, 2)
    assert mask.tolist() == [[True, True, False]]
    final_residual = np.sum(vectors[~mask], axis=0)
    assert np.isclose(
        initial_energy - sum(marginals[:2]),
        float(np.dot(final_residual, final_residual)),
    )


def test_signed_residual_best_prefix_crosses_temporary_negative_gain() -> None:
    vectors = np.asarray([[[3.0], [-2.0]]], dtype=np.float64)
    available = np.ones((1, 2), dtype=bool)
    _, marginals, initial_energy = _signed_residual_greedy_order(vectors, available)

    assert marginals == [-3.0, 4.0]
    count, residual_energy = _signed_residual_best_prefix(
        marginals,
        initial_energy,
        cost_fraction=0.0,
    )
    assert count == 2
    assert residual_energy == 0.0
    expensive_count, expensive_energy = _signed_residual_best_prefix(
        marginals,
        initial_energy,
        cost_fraction=5.0,
    )
    assert expensive_count == 0
    assert expensive_energy == initial_energy


def test_signed_residual_greedy_is_layer_separable() -> None:
    vectors = np.asarray(
        [
            [[1.0], [1.0]],
            [[10.0], [-9.0]],
        ],
        dtype=np.float64,
    )
    available = np.ones((2, 2), dtype=bool)

    order, _, _ = _signed_residual_greedy_order(
        vectors,
        available,
        max_cells=2,
    )

    assert order[0] in {(0, 0), (0, 1)}
    assert order[1] in {(0, 0), (0, 1)}
    assert order[0] != order[1]


def test_expand_downstream_columns_respects_depth_and_eligibility() -> None:
    eligible = np.zeros((5, 4), dtype=bool)
    eligible[1:, :] = True
    direct = np.zeros_like(eligible)
    direct[1, 0] = True
    direct[3, 2] = True

    bounded = _expand_downstream_columns(direct, eligible, depth=2)
    expected_bounded = np.zeros_like(eligible)
    expected_bounded[1:3, 0] = True
    expected_bounded[3:5, 2] = True
    assert np.array_equal(bounded, expected_bounded)

    full = _expand_downstream_columns(direct, eligible, depth=None)
    expected_full = np.zeros_like(eligible)
    expected_full[1:, 0] = True
    expected_full[3:, 2] = True
    assert np.array_equal(full, expected_full)


def test_expand_causal_wedge_recomputes_later_tokens_downstream() -> None:
    eligible = np.zeros((5, 5), dtype=bool)
    eligible[1:, :] = True
    direct = np.zeros_like(eligible)
    direct[1, 2] = True
    direct[1, 4] = True

    bounded = _expand_causal_wedge(direct, eligible, depth=3)
    expected_bounded = np.zeros_like(eligible)
    expected_bounded[1, 2] = True
    expected_bounded[1, 4] = True
    expected_bounded[2:4, 2:] = True
    assert np.array_equal(bounded, expected_bounded)

    full = _expand_causal_wedge(direct, eligible, depth=None)
    expected_full = np.zeros_like(eligible)
    expected_full[1, 2] = True
    expected_full[1, 4] = True
    expected_full[2:, 2:] = True
    assert np.array_equal(full, expected_full)


def test_fast_zero_probe_kl_matches_reference_implementation() -> None:
    baseline = np.asarray([[2.0, 0.5, -1.0, 3.0]], dtype=np.float64)
    stale = np.asarray([[1.8, 0.7, -0.8, 3.1]], dtype=np.float64)
    expected = kl_divergence(baseline, stale)
    actual = _zero_probe_baseline_stale_kl(
        baseline_logits=baseline,
        stale_logits=stale,
    )
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def _evaluation(logits: list[float]) -> _Evaluation:
    output = BackendOutput(
        logits=np.asarray([logits], dtype=np.float64),
        cache_tensor=np.zeros((1, 1), dtype=np.float64),
        hidden_tensor=np.zeros((1, 1), dtype=np.float64),
        parameter_version=1,
        extras={},
    )
    return _Evaluation(output=output, logits_kl=0.0, top1_agreement=1.0)


def test_attention_residual_metrics_track_local_progress() -> None:
    inputs = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    def output(attention_output: list[list[float]]) -> BackendOutput:
        return BackendOutput(
            logits=np.zeros((1, 2), dtype=np.float64),
            cache_tensor=np.zeros((1, 1), dtype=np.float64),
            hidden_tensor=np.zeros((1, 1), dtype=np.float64),
            parameter_version=1,
            extras={
                "attention_input_summary": inputs.copy(),
                "attention_output_summary": np.asarray(attention_output, dtype=np.float64),
            },
        )

    baseline = output([[0.0, 0.0], [0.0, 0.0]])
    stale = output([[0.0, 0.0], [0.0, 0.0]])
    full = output([[1.0, 0.0], [2.0, 0.0]])
    candidate = output([[0.5, 0.0], [1.0, 0.0]])

    metrics = _attention_residual_metrics(
        candidate=candidate,
        full=full,
        stale=stale,
        baseline=baseline,
        target_layer=0,
    )

    assert metrics["target_attention_output_candidate_error_l2"] == 0.5
    assert metrics["target_attention_output_recovery_fraction"] == 0.5
    assert metrics["target_attention_output_delta_projection"] == 0.5
    assert metrics["target_attention_output_delta_cosine"] == 1.0
    assert metrics["target_attention_output_orthogonal_error_fraction"] == 0.0
    assert metrics["downstream_attention_output_recovery_fraction"] == 0.5
    assert metrics["target_attention_input_candidate_shift_relative"] == 0.0
    assert metrics["target_attention_input_full_shift_relative"] == 0.0


def test_condition_seed_ignores_runtime_metadata() -> None:
    condition = {
        "seed": 7,
        "sample_id": 1,
        "dataset_sample_id": "sample-1",
        "task_name": "multi_hop",
        "model_name": "qwen",
        "update_target": "lora.k_middle",
        "target_layer": 14,
        "version_gap": 4,
        "configured_update_norm": 0.01,
        "context_length": 512,
        "block_size": 64,
        "baseline_prefill_latency": 0.1,
    }
    changed_runtime = {**condition, "baseline_prefill_latency": 9.9}
    changed_axis = {**condition, "block_size": 128}

    assert _condition_seed(condition) == _condition_seed(changed_runtime)
    assert _condition_seed(condition) != _condition_seed(changed_axis)


def test_logit_selection_metrics_tracks_reference_and_confidence() -> None:
    weak_nll, weak_entropy, weak_confidence = _logit_selection_metrics(
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        reference_token_id=0,
    )
    strong_nll, strong_entropy, strong_confidence = _logit_selection_metrics(
        np.asarray([[4.0, 0.0]], dtype=np.float64),
        reference_token_id=0,
    )

    assert strong_nll < weak_nll
    assert strong_entropy < weak_entropy
    assert strong_confidence > weak_confidence


def test_greedy_sparse_reference_objective_selects_best_blocks() -> None:
    candidate_mask = np.ones((1, 3), dtype=bool)
    weights = np.asarray([1.0, 3.0, 2.0])

    def evaluate(mask: np.ndarray) -> _Evaluation:
        reference_logit = float(np.sum(weights[mask[0]]))
        return _evaluation([reference_logit, 0.0])

    selected = _greedy_sparse_objective_masks(
        evaluate=evaluate,
        candidate_mask=candidate_mask,
        max_cells=2,
        reference_token_id=0,
        objective="reference_nll",
    )

    assert selected[1][0].tolist() == [[False, True, False]]
    assert selected[2][0].tolist() == [[False, True, True]]


def test_sequence_sparse_objective_reads_probe_metadata() -> None:
    evaluation = _evaluation([0.0, 0.0])
    assert evaluation.output.extras is not None
    evaluation.output.extras["reference_token_nll_2"] = 0.125

    assert (
        _sparse_objective_score(
            evaluation,
            reference_token_id=0,
            objective="reference_nll_2",
        )
        == 0.125
    )


def test_reference_token_id_accepts_batch_encoding_like_mapping() -> None:
    backend = SimpleNamespace(tokenizer=lambda text, add_special_tokens=False: UserDict({"input_ids": [17, 18]}))
    sample = TaskSample(prompt="prompt", answer="answer", metadata={})

    assert _reference_token_id(backend, sample) == 17


def test_beam_and_swap_escape_a_greedy_pair_trap() -> None:
    candidate_mask = np.ones((1, 3), dtype=bool)
    scores = {
        (0,): 3.0,
        (1,): 2.0,
        (2,): 2.0,
        (0, 1): 3.1,
        (0, 2): 3.1,
        (1, 2): 6.0,
    }

    def evaluate(mask: np.ndarray) -> _Evaluation:
        selected = tuple(int(index) for index in np.flatnonzero(mask[0]))
        return _evaluation([scores[selected], 0.0])

    greedy = _greedy_sparse_objective_masks(
        evaluate=evaluate,
        candidate_mask=candidate_mask,
        max_cells=2,
        reference_token_id=0,
        objective="reference_nll",
    )
    beam = _beam_sparse_objective_masks(
        evaluate=evaluate,
        candidate_mask=candidate_mask,
        max_cells=2,
        reference_token_id=0,
        objective="reference_nll",
        beam_width=2,
    )
    swapped = _swap_refine_sparse_objective_masks(
        evaluate=evaluate,
        greedy_path=greedy,
        candidate_mask=candidate_mask,
        reference_token_id=0,
        objective="reference_nll",
        max_rounds=2,
    )

    assert greedy[2][0].tolist() == [[True, True, False]]
    assert beam[2].mask.tolist() == [[False, True, True]]
    assert swapped[2].mask.tolist() == [[False, True, True]]
    assert beam[2].probe_count > 0
    assert swapped[2].probe_count > 0


def test_joint_search_uses_cost_penalty_and_can_select_stale() -> None:
    stale = _evaluation([0.0, 0.0])
    one = _evaluation([4.0, 0.0])
    two = _evaluation([4.01, 0.0])
    path = {
        1: _SearchPoint(
            mask=np.asarray([[True, False]], dtype=bool),
            evaluation=one,
            score=_logit_selection_metrics(one.output.logits, reference_token_id=0)[0],
            probe_count=2,
        ),
        2: _SearchPoint(
            mask=np.asarray([[True, True]], dtype=bool),
            evaluation=two,
            score=_logit_selection_metrics(two.output.logits, reference_token_id=0)[0],
            probe_count=3,
        ),
    }

    unpenalized = _joint_sparse_search_point(
        stale=stale,
        path=path,
        reference_token_id=0,
        objective="reference_nll",
        direct_total=2,
        cost_penalty=0.0,
    )
    penalized = _joint_sparse_search_point(
        stale=stale,
        path=path,
        reference_token_id=0,
        objective="reference_nll",
        direct_total=2,
        cost_penalty=0.05,
    )
    harmful_path = {
        1: _SearchPoint(
            mask=np.asarray([[True]], dtype=bool),
            evaluation=_evaluation([-2.0, 0.0]),
            score=_logit_selection_metrics(
                _evaluation([-2.0, 0.0]).output.logits,
                reference_token_id=0,
            )[0],
            probe_count=1,
        )
    }
    stale_selected = _joint_sparse_search_point(
        stale=stale,
        path=harmful_path,
        reference_token_id=0,
        objective="reference_nll",
        direct_total=1,
        cost_penalty=0.0,
    )

    safety_gated = _joint_sparse_search_point(
        stale=stale,
        path=path,
        reference_token_id=0,
        objective="reference_nll",
        direct_total=2,
        cost_penalty=0.0,
        stale_margin=0.02,
    )

    assert int(np.count_nonzero(unpenalized.mask)) == 2
    assert int(np.count_nonzero(penalized.mask)) == 1
    assert int(np.count_nonzero(stale_selected.mask)) == 0
    assert int(np.count_nonzero(safety_gated.mask)) == 2
    assert unpenalized.probe_count == 3
    assert penalized.probe_count == 3
    assert safety_gated.probe_count == 3


def test_joint_search_rejects_improvement_below_stale_margin() -> None:
    stale = _evaluation([0.0, 0.0])
    slightly_better = _evaluation([0.01, 0.0])
    path = {
        1: _SearchPoint(
            mask=np.asarray([[True]], dtype=bool),
            evaluation=slightly_better,
            score=_logit_selection_metrics(
                slightly_better.output.logits,
                reference_token_id=0,
            )[0],
            probe_count=1,
        )
    }

    accepted = _joint_sparse_search_point(
        stale=stale,
        path=path,
        reference_token_id=0,
        objective="reference_nll",
        direct_total=1,
        cost_penalty=0.0,
        stale_margin=0.0,
    )
    rejected = _joint_sparse_search_point(
        stale=stale,
        path=path,
        reference_token_id=0,
        objective="reference_nll",
        direct_total=1,
        cost_penalty=0.0,
        stale_margin=0.01,
    )

    assert int(np.count_nonzero(accepted.mask)) == 1
    assert int(np.count_nonzero(rejected.mask)) == 0



def test_compact_evaluation_output_drops_heavy_state_and_keeps_probe_metrics() -> None:
    heavy_cache = object()
    output = BackendOutput(
        logits=np.asarray([[1.0, 2.0]], dtype=np.float64),
        cache_tensor=np.ones((8, 8), dtype=np.float64),
        hidden_tensor=np.ones((8, 8), dtype=np.float64),
        parameter_version=3,
        extras={
            "past_key_values": heavy_cache,
            "hidden_states": heavy_cache,
            "lora_cache": heavy_cache,
            "prompt_state": heavy_cache,
            "attention_summary": np.asarray([[0.25, 0.75]]),
            "cache_mode": "oracle_layer_token_block_splice",
            "strategy_flops": 123.0,
            "reference_token_nll_2": 0.125,
            "generated_text": "alpha beta",
            "generated_tokens": 2,
        },
    )

    compact = _compact_evaluation_output(output)

    assert compact.logits is output.logits
    assert compact.cache_tensor.size == 0
    assert compact.hidden_tensor.size == 0
    assert compact.parameter_version == 3
    assert compact.extras is not None
    assert compact.extras["cache_mode"] == "oracle_layer_token_block_splice"
    assert compact.extras["strategy_flops"] == 123.0
    assert compact.extras["reference_token_nll_2"] == 0.125
    assert compact.extras["generated_text"] == "alpha beta"
    assert compact.extras["generated_tokens"] == 2
    assert "attention_summary" in compact.extras
    assert "past_key_values" not in compact.extras
    assert "hidden_states" not in compact.extras
    assert "lora_cache" not in compact.extras
    assert "prompt_state" not in compact.extras


def test_blockwise_record_includes_generated_task_metrics() -> None:
    output = BackendOutput(
        logits=np.asarray([[1.0, 2.0]], dtype=np.float64),
        cache_tensor=np.empty((0,), dtype=np.float32),
        hidden_tensor=np.empty((0,), dtype=np.float32),
        parameter_version=1,
        extras={
            "generated_text": "answer",
            "generated_tokens": 3,
            "cache_mode": "test",
        },
    )
    evaluation = _Evaluation(
        output=output,
        logits_kl=0.25,
        top1_agreement=1.0,
        task_score=0.75,
    )

    record = _record(
        {
            "full_task_score": 1.0,
            "stale_task_score": 0.5,
            "reference_token_id": -1,
        },
        selector="test",
        requested_budget_fraction=0.25,
        mask=np.asarray([[True, False]], dtype=bool),
        eligible=np.asarray([[True, True]], dtype=bool),
        evaluation=evaluation,
        stale_kl=0.5,
    )

    assert record["task_score"] == 0.75
    assert record["task_drop_vs_full"] == 0.25
    assert record["task_gain_vs_stale"] == 0.25
    assert record["generated_text"] == "answer"
    assert record["generated_tokens"] == 3


def test_generation_sample_preserves_or_overrides_task_limit() -> None:
    sample = TaskSample(
        prompt="question",
        answer="answer",
        metadata={"max_generation_tokens": 16, "other": "value"},
    )

    preserved = _generation_sample(sample, max_generation_tokens=0)
    overridden = _generation_sample(sample, max_generation_tokens=4)

    assert preserved is sample
    assert overridden is not sample
    assert overridden.metadata["max_generation_tokens"] == 4
    assert overridden.metadata["other"] == "value"
    assert sample.metadata["max_generation_tokens"] == 16


def test_blockwise_checkpoint_round_trip_requires_all_artifacts(tmp_path: Path) -> None:
    condition = {
        "sample_id": 7,
        "update_target": "lora.mlp_middle",
        "block_size": 32,
        "version_gap": 4,
        "context_length": 4096,
    }
    records = [{**condition, "selector": "stale", "logits_kl": 1.0}]
    frontier = [{**condition, "token_block": 0, "marginal_kl_gain": 0.1}]
    masks = [{**condition, "selector": "oracle", "layer": 0, "token_block": 0}]

    records_path = tmp_path / "blockwise_records.jsonl"
    frontier_path = tmp_path / "block_frontier.csv"
    masks_path = tmp_path / "block_masks.csv"
    _write_jsonl(records_path, records)
    _write_rows(frontier_path, frontier)
    _write_rows(masks_path, masks)

    loaded_records = _read_jsonl(records_path)
    loaded_frontier = _read_rows(frontier_path)
    loaded_masks = _read_rows(masks_path)
    assert len(
        _completed_condition_keys(loaded_records, loaded_frontier, loaded_masks)
    ) == 1
    assert _completed_condition_keys(loaded_records, loaded_frontier, []) == set()
    assert not list(tmp_path.glob(".*.tmp-*"))
