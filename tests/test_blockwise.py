from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import numpy as np

from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.experiments.blockwise import (
    _beam_sparse_objective_masks,
    _Evaluation,
    _greedy_sparse_objective_masks,
    _joint_sparse_search_point,
    _logit_selection_metrics,
    _reference_token_id,
    _SearchPoint,
    _sparse_objective_score,
    _swap_refine_sparse_objective_masks,
)
from ttt_cache_lab.models.interface import BackendOutput


def _evaluation(logits: list[float]) -> _Evaluation:
    output = BackendOutput(
        logits=np.asarray([logits], dtype=np.float64),
        cache_tensor=np.zeros((1, 1), dtype=np.float64),
        hidden_tensor=np.zeros((1, 1), dtype=np.float64),
        parameter_version=1,
        extras={},
    )
    return _Evaluation(output=output, logits_kl=0.0, top1_agreement=1.0)


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
    backend = SimpleNamespace(
        tokenizer=lambda text, add_special_tokens=False: UserDict({"input_ids": [17, 18]})
    )
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

    assert int(np.count_nonzero(unpenalized.mask)) == 2
    assert int(np.count_nonzero(penalized.mask)) == 1
    assert int(np.count_nonzero(stale_selected.mask)) == 0
