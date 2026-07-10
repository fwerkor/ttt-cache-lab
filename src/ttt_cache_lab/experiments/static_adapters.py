from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName, build_strategy
from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.data.loader import build_task_samples
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.experiments.metrics import (
    estimate_recompute_fraction,
    is_cache_hit,
    is_false_safe,
    is_refresh_action,
    output_cache_bytes,
    output_cache_maintenance_latency,
    output_decode_latency,
    output_memory_allocated,
    output_peak_memory_allocated,
    output_strategy_latency,
    output_strategy_mode,
    output_throughput,
)
from ttt_cache_lab.experiments.results import ExperimentArtifacts, ExperimentRecord, write_records
from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.models.interface import BackendOutput, ModelBackend
from ttt_cache_lab.updates.targets import UpdateTarget, parse_update_target


@dataclass(frozen=True)
class _StaticAdapter:
    adapter_id: int
    current: BackendOutput
    update_norm: float
    state: Any | None = None


class StaticAdapterExperimentRunner:
    """Evaluate cache reuse while repeatedly switching among fixed adapters."""

    def __init__(self, config: VersionedExperimentConfig) -> None:
        self.config = config
        if config.adapter.update_mode != "static_lora":
            raise ValueError("StaticAdapterExperimentRunner requires adapter.update_mode=static_lora")
        if not config.adapter.static_adapter_sequence:
            raise ValueError("adapter.static_adapter_sequence must not be empty")
        if min(config.adapter.static_adapter_sequence) < 0:
            raise ValueError("static adapter ids must be non-negative")

    def run(self) -> ExperimentArtifacts:
        data = build_task_samples(self.config.data, seed=self.config.seed)
        backend = build_backend(self.config.model, seed=self.config.seed)
        strategies = [
            build_strategy(
                name,
                refresh_period=self.config.cache.refresh_period,
                update_norm_threshold=self.config.cache.update_norm_threshold,
            )
            for name in self.config.cache.strategies
        ]
        records: list[ExperimentRecord] = []
        for sample_id, sample in enumerate(data):
            sample = backend.prepare_sample(sample, context_length=self.config.data.context_length)
            for target_name in self.config.updates.targets:
                target = parse_update_target(target_name, num_layers=backend.num_layers)
                backend.restore_after_update()
                self._prepare_backend(backend, target)
                baseline = backend.prefill(sample.prompt)
                adapters = self._build_static_adapters(backend, sample, target, baseline)
                per_adapter_cache: dict[str, dict[int, BackendOutput]] = {
                    str(strategy.name): {0: baseline} for strategy in strategies
                }
                latest_cache: dict[str, tuple[int, BackendOutput]] = {
                    str(strategy.name): (0, baseline) for strategy in strategies
                }
                refresh_counts = {str(strategy.name): 0 for strategy in strategies}

                for sequence_step, adapter_number in enumerate(self.config.adapter.static_adapter_sequence):
                    adapter = adapters[adapter_number]
                    self._activate_adapter(backend, adapter)
                    full = backend.full_recompute(sample.prompt, adapter.current)
                    for strategy in strategies:
                        key = str(strategy.name)
                        cached_adapter, cached_output = latest_cache[key]
                        version_gap = 0 if cached_adapter == adapter_number else 1
                        decision = strategy.decide(
                            target,
                            step=version_gap,
                            update_norm=adapter.update_norm,
                        )
                        baseline_output = baseline
                        if strategy.name is StrategyName.NO_ADAPTATION:
                            decision = StrategyDecision(
                                strategy.name,
                                CacheAction.REUSE_EXACT,
                                CacheBlockState.VALID_EXACT,
                                None,
                                "No-adaptation baseline uses the base model and base cache.",
                            )
                            baseline_output = baseline
                        elif strategy.name in {
                            StrategyName.ADAPTER_SPECIFIC_CACHE,
                            StrategyName.LRAGENT_ADAPTER_CACHE,
                        }:
                            existing = per_adapter_cache[key].get(adapter_number)
                            if existing is None:
                                decision = StrategyDecision(
                                    strategy.name,
                                    CacheAction.FULL_RECOMPUTE,
                                    CacheBlockState.INVALID,
                                    None,
                                    "Static adapter cache miss; build a dedicated entry.",
                                    recompute_fraction=1.0,
                                )
                            else:
                                decision = StrategyDecision(
                                    strategy.name,
                                    CacheAction.REUSE_EXACT,
                                    CacheBlockState.VALID_EXACT,
                                    None,
                                    "Static adapter cache hit.",
                                )
                                baseline_output = existing
                        elif version_gap == 0 and decision.action is not CacheAction.FULL_RECOMPUTE:
                            decision = StrategyDecision(
                                strategy.name,
                                CacheAction.REUSE_EXACT,
                                CacheBlockState.VALID_EXACT,
                                None,
                                "The fixed adapter already matches the cached adapter entry.",
                            )
                            baseline_output = per_adapter_cache[key].get(adapter_number, cached_output)
                        elif strategy.name not in {
                            StrategyName.BASE_CACHE_REUSE,
                            StrategyName.STATIC_BASE_DELTA,
                            StrategyName.FORKKV_BASE_DELTA,
                            StrategyName.ALORA_PREFIX_REUSE,
                            StrategyName.STALE_REUSE,
                            StrategyName.FROZEN_REUSE,
                        }:
                            baseline_output = cached_output

                        approx = backend.apply_cache_strategy(
                            baseline=baseline_output,
                            full=full,
                            updated=adapter.current,
                            decision=decision,
                        )
                        if strategy.name in {
                            StrategyName.ADAPTER_SPECIFIC_CACHE,
                            StrategyName.LRAGENT_ADAPTER_CACHE,
                        }:
                            per_adapter_cache[key][adapter_number] = approx
                        if strategy.name not in {
                            StrategyName.NO_ADAPTATION,
                            StrategyName.ALORA_PREFIX_REUSE,
                        } and decision.action in {
                            CacheAction.FULL_RECOMPUTE,
                            CacheAction.REUSE_EXACT,
                            CacheAction.PARTIAL_RECOMPUTE,
                            CacheAction.DELTA_CORRECT,
                            CacheAction.REJECT_UPDATE,
                        }:
                            latest_cache[key] = (adapter_number, approx)
                        if is_refresh_action(decision):
                            refresh_counts[key] += 1

                        top1 = top1_agreement(full.logits, approx.logits)
                        fallback_latency = backend.estimate_latency(
                            decision,
                            context_length=self.config.data.context_length,
                        )
                        strategy_latency = output_strategy_latency(approx, fallback=fallback_latency)
                        decode_latency = output_decode_latency(approx)
                        maintenance_latency = output_cache_maintenance_latency(approx)
                        task_score = (
                            backend.score_answer(sample, approx)
                            if self.config.metrics.compute_task_metrics
                            else 0.0
                        )
                        full_task_score = (
                            backend.score_answer(sample, full)
                            if self.config.metrics.compute_task_metrics
                            else 0.0
                        )
                        logits_kl_value = (
                            kl_divergence(full.logits, approx.logits)
                            if self.config.metrics.compute_tensor_metrics
                            else 0.0
                        )
                        records.append(
                            ExperimentRecord(
                                sample_id=sample_id,
                                update_target=target_name,
                                cache_strategy=str(decision.strategy),
                                action=str(decision.action),
                                cache_state=str(decision.state),
                                first_invalid_layer=decision.first_invalid_layer,
                                task_score=task_score,
                                logits_kl=logits_kl_value,
                                top1_agreement=top1,
                                relative_error=(
                                    relative_error(full.cache_tensor, approx.cache_tensor)
                                    if self.config.metrics.compute_tensor_metrics
                                    else 0.0
                                ),
                                latency_units=strategy_latency,
                                reason=decision.reason,
                                experiment_id=self.config.experiment_id,
                                adapter_id=f"static-{adapter_number}",
                                adapter_version=adapter_number,
                                cached_version=cached_adapter,
                                version_gap=version_gap,
                                update_step=sequence_step,
                                accumulated_update_norm=adapter.update_norm,
                                update_norm_since_cache=adapter.update_norm if version_gap else 0.0,
                                lora_rank=self.config.adapter.lora_rank,
                                update_mode=self.config.adapter.update_mode,
                                hidden_relative_error=(
                                    relative_error(full.hidden_tensor, approx.hidden_tensor)
                                    if self.config.metrics.compute_tensor_metrics
                                    else 0.0
                                ),
                                cache_bytes=output_cache_bytes(approx),
                                memory_allocated=output_memory_allocated(approx),
                                peak_memory_allocated=output_peak_memory_allocated(approx),
                                adaptation_latency=0.0,
                                cache_maintenance_latency=maintenance_latency,
                                decode_latency=decode_latency,
                                end_to_end_latency=strategy_latency,
                                throughput_tokens_per_s=output_throughput(approx, latency=strategy_latency),
                                recompute_fraction=estimate_recompute_fraction(
                                    decision,
                                    num_layers=backend.num_layers,
                                ),
                                cache_hit=is_cache_hit(decision),
                                refresh_count=refresh_counts[key],
                                rejected_reuse=(
                                    decision.reject_reuse or decision.action is CacheAction.REJECT_UPDATE
                                ),
                                false_safe=is_false_safe(
                                    decision,
                                    full=full,
                                    approx=approx,
                                    full_task_score=full_task_score,
                                    approx_task_score=task_score,
                                    kl_threshold=self.config.cache.oracle_kl_threshold,
                                    top1_threshold=self.config.cache.oracle_top1_threshold,
                                    task_drop_threshold=self.config.cache.oracle_task_drop_threshold,
                                ),
                                strategy_mode=output_strategy_mode(approx),
                                cache_block_count=backend.num_layers,
                                cache_entry_count=len(per_adapter_cache[key]),
                            )
                        )
                backend.restore_after_update()
        return write_records(records, self.config.output_dir)

    def _prepare_backend(self, backend: ModelBackend, target: UpdateTarget) -> None:
        prepare = getattr(backend, "prepare_update_target", None)
        if callable(prepare):
            prepare(
                target,
                rank=self.config.adapter.lora_rank,
                alpha=self.config.adapter.lora_alpha,
                freeze_base_model=self.config.adapter.freeze_base_model,
            )

    def _build_static_adapters(
        self,
        backend: ModelBackend,
        sample: TaskSample,
        target: UpdateTarget,
        baseline: BackendOutput,
    ) -> dict[int, _StaticAdapter]:
        adapter_ids = sorted(set(self.config.adapter.static_adapter_sequence))
        snapshot = getattr(backend, "snapshot_adapter_state", None)
        load = getattr(backend, "load_adapter_state", None)
        train = getattr(backend, "train_lora_step", None)
        if callable(snapshot) and callable(load) and callable(train):
            zero_state = snapshot()
            adapters = {0: _StaticAdapter(0, baseline, 0.0, zero_state)}
            for adapter_id in adapter_ids:
                if adapter_id == 0:
                    continue
                load(zero_state, version=0)
                norm = 0.0
                for _ in range(self.config.adapter.train_steps_per_version):
                    norm += float(
                        train(
                            sample,
                            target,
                            rank=self.config.adapter.lora_rank,
                            alpha=self.config.adapter.lora_alpha,
                            learning_rate=self.config.adapter.learning_rate * adapter_id,
                            freeze_base_model=self.config.adapter.freeze_base_model,
                        )
                    )
                state = snapshot()
                current = BackendOutput(
                    logits=baseline.logits,
                    cache_tensor=baseline.cache_tensor,
                    hidden_tensor=baseline.hidden_tensor,
                    parameter_version=adapter_id,
                    extras=baseline.extras,
                )
                adapters[adapter_id] = _StaticAdapter(adapter_id, current, norm, state)
            load(zero_state, version=0)
            return adapters

        adapters = {0: _StaticAdapter(0, baseline, 0.0)}
        for adapter_id in adapter_ids:
            if adapter_id == 0:
                continue
            norm = self.config.updates.update_norm * adapter_id
            current = backend.simulate_update(baseline, target, update_norm=norm)
            adapters[adapter_id] = _StaticAdapter(adapter_id, current, norm)
        return adapters

    def _activate_adapter(self, backend: ModelBackend, adapter: _StaticAdapter) -> None:
        if adapter.state is None:
            return
        load = getattr(backend, "load_adapter_state", None)
        if not callable(load):
            raise RuntimeError("Backend produced adapter snapshots but cannot restore them")
        load(adapter.state, version=adapter.adapter_id)
