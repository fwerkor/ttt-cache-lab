from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import CacheStrategy, build_strategy
from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.data.synthetic import SyntheticTaskFactory
from ttt_cache_lab.experiments.results import ExperimentArtifacts, ExperimentRecord, write_records
from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.models.interface import BackendOutput, ModelBackend
from ttt_cache_lab.updates.targets import UpdateTarget, parse_update_target


@dataclass
class _StrategyCache:
    output: BackendOutput
    cached_version: int


class VersionedExperimentRunner:
    """Run multi-step adapter-version experiments.

    This runner is the shared implementation for E1-E7. The exact experiment is
    selected by config: target set, version steps, cache strategies, adapter
    update mode, model, context length, and output directory.
    """

    def __init__(self, config: VersionedExperimentConfig) -> None:
        self.config = config

    def run(self) -> ExperimentArtifacts:
        data = SyntheticTaskFactory(self.config.seed).build(
            self.config.data.task,
            num_samples=self.config.data.num_samples,
            context_length=self.config.data.context_length,
            answer_length=self.config.data.answer_length,
        )
        backend = build_backend(self.config.model, seed=self.config.seed)
        strategies = [
            build_strategy(name, refresh_period=self.config.cache.refresh_period)
            for name in self.config.cache.strategies
        ]
        max_version = max(self.config.version_steps or [0])
        target_steps = set(self.config.version_steps)
        records: list[ExperimentRecord] = []

        for sample_id, sample in enumerate(data):
            for target_name in self.config.updates.targets:
                target = parse_update_target(target_name, num_layers=self.config.model.num_layers)
                backend.restore_after_update()
                cached_v0 = backend.prefill(sample.prompt)
                strategy_caches = {
                    str(strategy.name): _StrategyCache(output=cached_v0, cached_version=0) for strategy in strategies
                }
                accumulated_update_norm = 0.0
                current = cached_v0

                if 0 in target_steps:
                    full = backend.full_recompute(sample.prompt, cached_v0)
                    self._record_step(
                        records,
                        backend=backend,
                        sample_id=sample_id,
                        sample_answer=sample,
                        target=target,
                        target_name=target_name,
                        strategies=strategies,
                        strategy_caches=strategy_caches,
                        current=current,
                        full=full,
                        adapter_version=0,
                        accumulated_update_norm=0.0,
                    )

                for step in range(1, max_version + 1):
                    update_norm = self._update_one_version(backend, sample.prompt, target, current)
                    accumulated_update_norm += update_norm
                    current = BackendOutput(
                        logits=current.logits,
                        cache_tensor=current.cache_tensor,
                        hidden_tensor=current.hidden_tensor,
                        parameter_version=step,
                        extras=current.extras,
                    )
                    if step not in target_steps:
                        continue
                    full = backend.full_recompute(sample.prompt, current)
                    self._record_step(
                        records,
                        backend=backend,
                        sample_id=sample_id,
                        sample_answer=sample,
                        target=target,
                        target_name=target_name,
                        strategies=strategies,
                        strategy_caches=strategy_caches,
                        current=current,
                        full=full,
                        adapter_version=step,
                        accumulated_update_norm=accumulated_update_norm,
                    )
                backend.restore_after_update()

        return write_records(records, self.config.output_dir)

    def _update_one_version(
        self,
        backend: ModelBackend,
        prompt: str,
        target: UpdateTarget,
        current: BackendOutput,
    ) -> float:
        if self.config.adapter.update_mode == "lora_train" and hasattr(backend, "train_lora_step"):
            norms = []
            for _ in range(self.config.adapter.train_steps_per_version):
                norm = backend.train_lora_step(
                    prompt,
                    target,
                    rank=self.config.adapter.lora_rank,
                    alpha=self.config.adapter.lora_alpha,
                    learning_rate=self.config.adapter.learning_rate,
                    freeze_base_model=self.config.adapter.freeze_base_model,
                )
                norms.append(float(norm))
            return sum(norms)
        backend.simulate_update(current, target, update_norm=self.config.updates.update_norm)
        return self.config.updates.update_norm

    def _record_step(
        self,
        records: list[ExperimentRecord],
        *,
        backend: ModelBackend,
        sample_id: int,
        sample_answer: object,
        target: UpdateTarget,
        target_name: str,
        strategies: Sequence[CacheStrategy],
        strategy_caches: dict[str, _StrategyCache],
        current: BackendOutput,
        full: BackendOutput,
        adapter_version: int,
        accumulated_update_norm: float,
    ) -> None:
        for strategy in strategies:
            decision = strategy.decide(
                target,
                step=adapter_version,
                update_norm=max(accumulated_update_norm, self.config.updates.update_norm),
            )
            cache_key = str(decision.strategy)
            cached = strategy_caches[cache_key]
            approx = backend.apply_cache_strategy(
                baseline=cached.output,
                full=full,
                updated=current,
                decision=decision,
            )
            if decision.action in {
                CacheAction.FULL_RECOMPUTE,
                CacheAction.REUSE_EXACT,
                CacheAction.PARTIAL_RECOMPUTE,
                CacheAction.DELTA_CORRECT,
            }:
                strategy_caches[cache_key] = _StrategyCache(output=approx, cached_version=adapter_version)
            records.append(
                ExperimentRecord(
                    sample_id=sample_id,
                    update_target=target_name,
                    cache_strategy=str(decision.strategy),
                    action=str(decision.action),
                    cache_state=str(decision.state),
                    first_invalid_layer=decision.first_invalid_layer,
                    task_score=backend.score_answer(sample_answer, approx),  # type: ignore[arg-type]
                    logits_kl=kl_divergence(full.logits, approx.logits),
                    top1_agreement=top1_agreement(full.logits, approx.logits),
                    relative_error=relative_error(full.cache_tensor, approx.cache_tensor),
                    latency_units=backend.estimate_latency(decision, context_length=self.config.data.context_length),
                    reason=decision.reason,
                    experiment_id=self.config.experiment_id,
                    adapter_version=adapter_version,
                    cached_version=cached.cached_version,
                    version_gap=adapter_version - cached.cached_version,
                    update_step=adapter_version,
                    accumulated_update_norm=accumulated_update_norm,
                    lora_rank=self.config.adapter.lora_rank,
                    update_mode=self.config.adapter.update_mode,
                )
            )


def write_version_summary(input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["experiment_id"], row["update_target"], row["cache_strategy"], row["adapter_version"])
            groups.setdefault(key, []).append(row)
    fields = [
        "experiment_id",
        "update_target",
        "cache_strategy",
        "adapter_version",
        "count",
        "task_score_mean",
        "logits_kl_mean",
        "top1_agreement_mean",
        "relative_error_mean",
        "latency_units_mean",
        "accumulated_update_norm_mean",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            writer.writerow(
                {
                    "experiment_id": key[0],
                    "update_target": key[1],
                    "cache_strategy": key[2],
                    "adapter_version": key[3],
                    "count": len(rows),
                    "task_score_mean": _mean(rows, "task_score"),
                    "logits_kl_mean": _mean(rows, "logits_kl"),
                    "top1_agreement_mean": _mean(rows, "top1_agreement"),
                    "relative_error_mean": _mean(rows, "relative_error"),
                    "latency_units_mean": _mean(rows, "latency_units"),
                    "accumulated_update_norm_mean": _mean(rows, "accumulated_update_norm"),
                }
            )


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)
