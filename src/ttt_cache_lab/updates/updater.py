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
    raw_update_norm: float
    update_scale: float
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
        total_raw_norm = 0.0
        total_applied_norm = 0.0
        for _ in range(step_count):
            current = self.backend.simulate_update(current, target, update_norm=update_norm)
            total_latency += float(self.backend.last_adaptation_latency())
            total_raw_norm += float(self.backend.last_raw_update_norm())
            total_applied_norm += float(self.backend.last_applied_update_norm())
        return UpdateResult(
            output=current,
            update_norm=total_applied_norm,
            raw_update_norm=total_raw_norm,
            update_scale=(total_applied_norm / total_raw_norm if total_raw_norm > 0.0 else 0.0),
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
        norm_control: str,
    ) -> None:
        self.backend = backend
        self.sample = sample
        self.rank = rank
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.freeze_base_model = freeze_base_model
        self.norm_control = norm_control

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
        train = getattr(self.backend, "train_lora_step", None)
        if not callable(train):
            raise RuntimeError("The selected backend does not implement supervised LoRA updates")
        total_norm = 0.0
        total_raw_norm = 0.0
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
                    target_update_norm=(
                        update_norm if self.norm_control == "target_l2" else None
                    ),
                    target_update_rms=(
                        update_norm if self.norm_control == "target_rms" else None
                    ),
                )
            )
            total_raw_norm += float(self.backend.last_raw_update_norm())
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
            raw_update_norm=total_raw_norm,
            update_scale=(total_norm / total_raw_norm if total_raw_norm > 0.0 else 0.0),
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
    norm_control: str = "target_l2",
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
            norm_control=norm_control,
        )
    if mode in {"random", "lora_train"}:
        return RandomPerturbationUpdater(backend)
    raise ValueError(f"Unsupported dynamic update mode: {mode}")
