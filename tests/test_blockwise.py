from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import numpy as np

from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.experiments.blockwise import (
    _compact_evaluation_output,
    _Evaluation,
    _greedy_sparse_objective_masks,
    _logit_selection_metrics,
    _reference_token_id,
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


def test_reference_token_id_accepts_batch_encoding_like_mapping() -> None:
    backend = SimpleNamespace(
        tokenizer=lambda text, add_special_tokens=False: UserDict({"input_ids": [17, 18]})
    )
    sample = TaskSample(prompt="prompt", answer="answer", metadata={})

    assert _reference_token_id(backend, sample) == 17


def test_compact_evaluation_output_drops_device_resident_state() -> None:
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
            "attention_summary": np.asarray([[0.25, 0.75]]),
            "cache_mode": "oracle_layer_token_block_splice",
            "strategy_flops": 123.0,
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
    assert "attention_summary" in compact.extras
    assert "past_key_values" not in compact.extras
    assert "hidden_states" not in compact.extras
    assert "lora_cache" not in compact.extras
