from typing import Any, cast

import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.models.hf import HuggingFaceBackend
from ttt_cache_lab.models.interface import BackendOutput


class _EmptyDecodeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return ""


class _TextDecodeTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, token_ids: list[int]) -> str:
        return self.text


def _backend_with_tokenizer(tokenizer: object) -> HuggingFaceBackend:
    backend = cast(Any, object.__new__(HuggingFaceBackend))
    backend.tokenizer = tokenizer
    return cast(HuggingFaceBackend, backend)


def test_hf_score_rejects_empty_decoded_token() -> None:
    backend = _backend_with_tokenizer(_EmptyDecodeTokenizer())
    sample = TaskSample(prompt="", answer="42", metadata={})
    output = BackendOutput(
        logits=np.array([[1.0]], dtype=np.float64),
        cache_tensor=np.zeros((1, 2, 1), dtype=np.float64),
        hidden_tensor=np.zeros((1, 1), dtype=np.float64),
        parameter_version=0,
    )
    assert backend.score_answer(sample, output) == 0.0


def test_hf_score_accepts_nonempty_prefix_match() -> None:
    backend = _backend_with_tokenizer(_TextDecodeTokenizer("4"))
    sample = TaskSample(prompt="", answer="42", metadata={})
    output = BackendOutput(
        logits=np.array([[1.0]], dtype=np.float64),
        cache_tensor=np.zeros((1, 2, 1), dtype=np.float64),
        hidden_tensor=np.zeros((1, 1), dtype=np.float64),
        parameter_version=0,
    )
    assert backend.score_answer(sample, output) == 1.0


def test_hf_unsupported_cache_surgery_charges_full_latency() -> None:
    backend = cast(Any, object.__new__(HuggingFaceBackend))
    backend._last_prefill_s = 3.0
    backend._last_stale_s = 0.1

    for action in (CacheAction.PARTIAL_RECOMPUTE, CacheAction.DELTA_CORRECT):
        decision = StrategyDecision(
            StrategyName.ADAPTIVE,
            action,
            CacheBlockState.INVALID,
            None,
            "test",
        )
        assert HuggingFaceBackend.estimate_latency(backend, decision, context_length=128) == 3.0
