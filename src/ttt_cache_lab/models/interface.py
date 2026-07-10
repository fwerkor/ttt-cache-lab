from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ttt_cache_lab.cache.strategies import StrategyDecision
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.updates.targets import UpdateTarget


@dataclass(frozen=True)
class BackendOutput:
    logits: np.ndarray
    cache_tensor: np.ndarray
    hidden_tensor: np.ndarray
    parameter_version: int
    extras: dict[str, Any] | None = None


class ModelBackend(Protocol):
    @property
    def num_layers(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def parameter_count(self) -> int: ...

    def prepare_sample(self, sample: TaskSample, *, context_length: int) -> TaskSample: ...

    def prefill(self, prompt: str) -> BackendOutput: ...

    def simulate_update(
        self, baseline: BackendOutput, target: UpdateTarget, *, update_norm: float
    ) -> BackendOutput: ...

    def full_recompute(self, prompt: str, updated: BackendOutput) -> BackendOutput: ...

    def apply_cache_strategy(
        self,
        *,
        baseline: BackendOutput,
        full: BackendOutput,
        updated: BackendOutput,
        decision: StrategyDecision,
    ) -> BackendOutput: ...

    def score_answer(self, sample: TaskSample, output: BackendOutput) -> float: ...

    def estimate_latency(self, decision: StrategyDecision, *, context_length: int) -> float: ...

    def estimate_flops(self, decision: StrategyDecision, *, context_length: int) -> float: ...

    def configure_metrics(self, *, capture_attention: bool) -> None: ...

    def last_adaptation_latency(self) -> float: ...

    def last_raw_update_norm(self) -> float: ...

    def last_applied_update_norm(self) -> float: ...

    def restore_after_update(self) -> None: ...
