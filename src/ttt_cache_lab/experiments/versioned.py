from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import CacheStrategy, StrategyDecision, StrategyName, build_strategy
from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.data.synthetic import SyntheticTaskFactory, TaskSample
from ttt_cache_lab.experiments.metrics import (
    estimate_recompute_fraction,
    is_cache_hit,
    is_false_safe,
    is_refresh_action,
    output_cache_bytes,
    output_memory_allocated,
    output_strategy_mode,
)
from ttt_cache_lab.experiments.results import ExperimentArtifacts, ExperimentRecord, write_records
from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.models.interface import BackendOutput, ModelBackend
from ttt_cache_lab.updates.targets import UpdateTarget, parse_update_target


@dataclass
class _StrategyCache:
    output: BackendOutput
    cached_version: int
    refresh_count: int = 0
    version_outputs: dict[int, BackendOutput] = field(default_factory=dict)


@dataclass(frozen=True)
class _VersionUpdate:
    output: BackendOutput
    update_norm: float


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
            build_strategy(
                name,
                refresh_period=self.config.cache.refresh_period,
                update_norm_threshold=self.config.cache.update_norm_threshold,
            )
            for name in self.config.cache.strategies
        ]
        max_version = max(self.config.version_steps or [0])
        target_steps = set(self.config.version_steps)
        records: list[ExperimentRecord] = []

        for sample_id, sample in enumerate(data):
            sample = backend.prepare_sample(sample, context_length=self.config.data.context_length)
            for target_name in self.config.updates.targets:
                target = parse_update_target(target_name, num_layers=backend.num_layers)
                backend.restore_after_update()
                self._prepare_backend_for_target(backend, target)
                cached_v0 = backend.prefill(sample.prompt)
                strategy_caches = {
                    str(strategy.name): _StrategyCache(
                        output=cached_v0,
                        cached_version=0,
                        version_outputs={0: cached_v0},
                    )
                    for strategy in strategies
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
                    version_update = self._update_one_version(backend, sample, target, current)
                    accumulated_update_norm += version_update.update_norm
                    current = version_update.output
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

    def _prepare_backend_for_target(self, backend: ModelBackend, target: UpdateTarget) -> None:
        if self.config.adapter.update_mode != "lora_train":
            return
        prepare = getattr(backend, "prepare_update_target", None)
        if not callable(prepare):
            return
        prepare(
            target,
            rank=self.config.adapter.lora_rank,
            alpha=self.config.adapter.lora_alpha,
            freeze_base_model=self.config.adapter.freeze_base_model,
        )

    def _update_one_version(
        self,
        backend: ModelBackend,
        sample: TaskSample,
        target: UpdateTarget,
        current: BackendOutput,
    ) -> _VersionUpdate:
        if self.config.adapter.update_mode == "lora_train" and hasattr(backend, "train_lora_step"):
            norms = []
            for _ in range(self.config.adapter.train_steps_per_version):
                norm = backend.train_lora_step(
                    sample,
                    target,
                    rank=self.config.adapter.lora_rank,
                    alpha=self.config.adapter.lora_alpha,
                    learning_rate=self.config.adapter.learning_rate,
                    freeze_base_model=self.config.adapter.freeze_base_model,
                )
                norms.append(float(norm))
            next_version = int(getattr(backend, "parameter_version", current.parameter_version + 1))
            return _VersionUpdate(
                output=BackendOutput(
                    logits=current.logits,
                    cache_tensor=current.cache_tensor,
                    hidden_tensor=current.hidden_tensor,
                    parameter_version=next_version,
                    extras=current.extras,
                ),
                update_norm=sum(norms),
            )
        updated = backend.simulate_update(current, target, update_norm=self.config.updates.update_norm)
        return _VersionUpdate(output=updated, update_norm=self.config.updates.update_norm)

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
            cache_key = str(strategy.name)
            cached = strategy_caches[cache_key]
            version_gap = adapter_version - cached.cached_version
            decision = strategy.decide(
                target,
                step=version_gap,
                update_norm=accumulated_update_norm,
            )
            baseline_output = cached.output
            if strategy.name is StrategyName.NO_ADAPTATION:
                decision = StrategyDecision(
                    decision.strategy,
                    CacheAction.REUSE_EXACT,
                    CacheBlockState.VALID_EXACT,
                    None,
                    "No-adaptation baseline keeps the original model and v0 output fixed.",
                )
                baseline_output = cached.version_outputs[0]
            elif strategy.name is StrategyName.ADAPTER_SPECIFIC_CACHE:
                existing = cached.version_outputs.get(adapter_version)
                if existing is not None:
                    decision = StrategyDecision(
                        decision.strategy,
                        CacheAction.REUSE_EXACT,
                        CacheBlockState.VALID_EXACT,
                        None,
                        "Adapter-version cache hit.",
                    )
                    baseline_output = existing
                else:
                    decision = StrategyDecision(
                        decision.strategy,
                        CacheAction.FULL_RECOMPUTE,
                        CacheBlockState.INVALID,
                        None,
                        "Adapter-version cache miss; build a dedicated cache entry.",
                        recompute_fraction=1.0,
                    )
            elif adapter_version == cached.cached_version and decision.action is not CacheAction.FULL_RECOMPUTE:
                decision = StrategyDecision(
                    decision.strategy,
                    CacheAction.REUSE_EXACT,
                    CacheBlockState.VALID_EXACT,
                    None,
                    "Cache version matches adapter version; reuse is exact.",
                )
            approx = backend.apply_cache_strategy(
                baseline=baseline_output,
                full=full,
                updated=current,
                decision=decision,
            )
            new_refresh_count = cached.refresh_count + (1 if is_refresh_action(decision) else 0)
            if strategy.name is StrategyName.ADAPTER_SPECIFIC_CACHE:
                cached.version_outputs[adapter_version] = approx
                cached.output = approx
                cached.cached_version = adapter_version
                cached.refresh_count = new_refresh_count
            elif strategy.name is not StrategyName.NO_ADAPTATION and decision.action in {
                CacheAction.FULL_RECOMPUTE,
                CacheAction.REUSE_EXACT,
                CacheAction.PARTIAL_RECOMPUTE,
                CacheAction.DELTA_CORRECT,
            }:
                cached.output = approx
                cached.cached_version = adapter_version
                cached.refresh_count = new_refresh_count
                cached.version_outputs[adapter_version] = approx
            top1 = top1_agreement(full.logits, approx.logits)
            records.append(
                ExperimentRecord(
                    sample_id=sample_id,
                    update_target=target_name,
                    cache_strategy=str(decision.strategy),
                    action=str(decision.action),
                    cache_state=str(decision.state),
                    first_invalid_layer=decision.first_invalid_layer,
                    task_score=(
                        backend.score_answer(sample_answer, approx)  # type: ignore[arg-type]
                        if self.config.metrics.compute_task_metrics
                        else 0.0
                    ),
                    logits_kl=(
                        kl_divergence(full.logits, approx.logits)
                        if self.config.metrics.compute_tensor_metrics
                        else 0.0
                    ),
                    top1_agreement=top1,
                    relative_error=(
                        relative_error(full.cache_tensor, approx.cache_tensor)
                        if self.config.metrics.compute_tensor_metrics
                        else 0.0
                    ),
                    latency_units=backend.estimate_latency(decision, context_length=self.config.data.context_length),
                    reason=decision.reason,
                    experiment_id=self.config.experiment_id,
                    adapter_version=adapter_version,
                    cached_version=cached.cached_version,
                    version_gap=version_gap,
                    update_step=adapter_version,
                    accumulated_update_norm=accumulated_update_norm,
                    lora_rank=self.config.adapter.lora_rank,
                    update_mode=self.config.adapter.update_mode,
                    hidden_relative_error=(
                        relative_error(full.hidden_tensor, approx.hidden_tensor)
                        if self.config.metrics.compute_tensor_metrics
                        else 0.0
                    ),
                    cache_bytes=output_cache_bytes(approx),
                    memory_allocated=output_memory_allocated(approx),
                    recompute_fraction=estimate_recompute_fraction(decision, num_layers=backend.num_layers),
                    cache_hit=is_cache_hit(decision),
                    refresh_count=new_refresh_count,
                    rejected_reuse=decision.reject_reuse,
                    false_safe=is_false_safe(decision, full=full, approx=approx),
                    strategy_mode=output_strategy_mode(approx),
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
        "hidden_relative_error_mean",
        "latency_units_mean",
        "recompute_fraction_mean",
        "cache_hit_rate",
        "refresh_count_mean",
        "false_safe_rate",
        "accumulated_update_norm_mean",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, records in sorted(groups.items()):
            experiment_id, target, strategy, version = key
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "update_target": target,
                    "cache_strategy": strategy,
                    "adapter_version": version,
                    "count": len(records),
                    "task_score_mean": _mean(records, "task_score"),
                    "logits_kl_mean": _mean(records, "logits_kl"),
                    "top1_agreement_mean": _mean(records, "top1_agreement"),
                    "relative_error_mean": _mean(records, "relative_error"),
                    "hidden_relative_error_mean": _mean(records, "hidden_relative_error"),
                    "latency_units_mean": _mean(records, "latency_units"),
                    "recompute_fraction_mean": _mean(records, "recompute_fraction"),
                    "cache_hit_rate": _mean_bool(records, "cache_hit"),
                    "refresh_count_mean": _mean(records, "refresh_count"),
                    "false_safe_rate": _mean_bool(records, "false_safe"),
                    "accumulated_update_norm_mean": _mean(records, "accumulated_update_norm"),
                }
            )


def _mean(records: list[dict[str, str]], field: str) -> float:
    values = [float(record[field]) for record in records if record.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _mean_bool(records: list[dict[str, str]], field: str) -> float:
    values = [record.get(field, "False").lower() == "true" for record in records]
    return sum(1.0 for value in values if value) / len(values) if values else 0.0
