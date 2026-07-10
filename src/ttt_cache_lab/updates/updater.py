from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.models.interface import BackendOutput, ModelBackend
from ttt_cache_lab.updates.targets import UpdateTarget


@dataclass(frozen=True)
class UpdateResult:
    output: BackendOutput
    update_norm: float
    step_count: int
    adaptation_latency: float


class TTTUpdater(Protocol):
    def update(
        self,
        baseline: BackendOutput,
        target: UpdateTarget,
        *,
        step_count: int,
        update_norm: float,
    ) -> UpdateResult: ...


class RandomPerturbationUpdater:
    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend

    def update(
        self,
        baseline: BackendOutput,
        target: UpdateTarget,
        *,
        step_count: int,
        update_norm: float,
    ) -> UpdateResult:
        if step_count < 1:
            raise ValueError("step_count must be at least 1")
        current = baseline
        total_latency = 0.0
        for _ in range(step_count):
            current = self.backend.simulate_update(current, target, update_norm=update_norm)
            total_latency += float(self.backend.last_adaptation_latency())
        return UpdateResult(
            output=current,
            update_norm=update_norm * step_count,
            step_count=step_count,
            adaptation_latency=total_latency,
        )


class SupervisedLoraUpdater:
    def __init__(
        self,
        backend: ModelBackend,
        sample: TaskSample,
        *,
        rank: int,
        alpha: float,
        learning_rate: float,
        freeze_base_model: bool,
    ) -> None:
        self.backend = backend
        self.sample = sample
        self.rank = rank
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.freeze_base_model = freeze_base_model

    def update(
        self,
        baseline: BackendOutput,
        target: UpdateTarget,
        *,
        step_count: int,
        update_norm: float,
    ) -> UpdateResult:
        del update_norm
        if step_count < 1:
            raise ValueError("step_count must be at least 1")
        train = getattr(self.backend, "train_lora_step", None)
        if not callable(train):
            raise RuntimeError("The selected backend does not implement supervised LoRA updates")
        total_norm = 0.0
        total_latency = 0.0
        for _ in range(step_count):
            total_norm += float(
                train(
                    self.sample,
                    target,
                    rank=self.rank,
                    alpha=self.alpha,
                    learning_rate=self.learning_rate,
                    freeze_base_model=self.freeze_base_model,
                )
            )
            total_latency += float(self.backend.last_adaptation_latency())
        next_version = int(getattr(self.backend, "parameter_version", baseline.parameter_version + step_count))
        return UpdateResult(
            output=BackendOutput(
                logits=baseline.logits,
                cache_tensor=baseline.cache_tensor,
                hidden_tensor=baseline.hidden_tensor,
                parameter_version=next_version,
                extras=baseline.extras,
            ),
            update_norm=total_norm,
            step_count=step_count,
            adaptation_latency=total_latency,
        )


def build_updater(
    backend: ModelBackend,
    *,
    mode: str,
    sample: TaskSample | None = None,
    target: UpdateTarget | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    learning_rate: float = 1e-3,
    freeze_base_model: bool = True,
) -> TTTUpdater:
    if mode == "lora_train" and target is not None and target.is_lora:
        if sample is None:
            raise ValueError("A task sample is required for supervised LoRA updates")
        return SupervisedLoraUpdater(
            backend,
            sample,
            rank=rank,
            alpha=alpha,
            learning_rate=learning_rate,
            freeze_base_model=freeze_base_model,
        )
    if mode in {"random", "lora_train"}:
        return RandomPerturbationUpdater(backend)
    raise ValueError(f"Unsupported dynamic update mode: {mode}")
