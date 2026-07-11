from __future__ import annotations

import random
from functools import partial

from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName, build_strategy
from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.data.loader import build_task_samples
from ttt_cache_lab.experiments.measurement import execute_strategy, measure_backend_call
from ttt_cache_lab.experiments.metrics import (
    attention_distribution_shift,
    estimate_recompute_fraction,
    is_cache_hit,
    is_false_safe,
    is_refresh_action,
    output_baseline_fidelity,
    output_cache_bytes,
    output_full_recompute_flops,
    output_memory_allocated,
    output_peak_memory_allocated,
    output_physical_cache_bytes,
    output_strategy_available,
    output_strategy_fallback,
    output_strategy_flops,
    output_strategy_mode,
    output_throughput,
)
from ttt_cache_lab.experiments.planner_runtime import build_planner_runtime
from ttt_cache_lab.experiments.provenance import baseline_provenance, planner_provenance
from ttt_cache_lab.experiments.results import ExperimentArtifacts, ExperimentRecord, write_records
from ttt_cache_lab.experiments.run_metadata import (
    collect_run_metadata,
    record_run_fields,
    write_run_metadata,
)
from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.updates.targets import parse_update_target
from ttt_cache_lab.updates.updater import RandomPerturbationUpdater


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def run(self) -> ExperimentArtifacts:
        data = build_task_samples(self.config.data, seed=self.config.seed)
        backend = build_backend(self.config.model, seed=self.config.seed)
        backend.configure_metrics(capture_attention=self.config.metrics.compute_attention_metrics)
        run_metadata = collect_run_metadata(self.config)
        metadata_path = write_run_metadata(self.config.output_dir, run_metadata)
        strategies = [
            build_strategy(
                name,
                refresh_period=self.config.cache.refresh_period,
                update_norm_threshold=self.config.cache.update_norm_threshold,
                recompute_window_size=self.config.cache.recompute_window_size,
                version_gap_threshold=self.config.cache.version_gap_threshold,
                error_proxy_threshold=self.config.cache.error_proxy_threshold,
                latency_budget_fraction=self.config.cache.latency_budget_fraction,
                memory_budget_bytes=self.config.cache.max_cache_bytes,
                failure_map_path=self.config.cache.failure_map_path,
                safe_kl_threshold=self.config.cache.oracle_kl_threshold,
                safe_top1_threshold=self.config.cache.oracle_top1_threshold,
                safe_task_drop_threshold=self.config.cache.oracle_task_drop_threshold,
            )
            for name in self.config.cache.strategies
        ]
        records: list[ExperimentRecord] = []
        full_decision = StrategyDecision(
            StrategyName.FULL_RECOMPUTE,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            None,
            "FLOP accounting probe.",
            recompute_fraction=1.0,
        )

        for sample_id, sample in enumerate(data):
            sample = backend.prepare_sample(sample, context_length=self.config.data.context_length)
            baseline = backend.prefill(sample.prompt)
            for target_name in self.config.updates.targets:
                target = parse_update_target(target_name, num_layers=backend.num_layers)
                update_result = RandomPerturbationUpdater(backend).update(
                    baseline,
                    target,
                    step_count=self.config.updates.step_count,
                    update_norm=self.config.updates.update_norm,
                )
                updated = update_result.output
                full = backend.full_recompute(sample.prompt, updated)
                baseline_task_score = (
                    backend.score_answer(sample, baseline)
                    if self.config.metrics.compute_task_metrics
                    else 0.0
                )
                full_task_score = (
                    backend.score_answer(sample, full)
                    if self.config.metrics.compute_task_metrics
                    else 0.0
                )

                for strategy in strategies:
                    decision = strategy.decide_with_runtime(
                        target,
                        step=self.config.updates.step_count,
                        update_norm=self.config.updates.update_norm,
                        runtime=build_planner_runtime(
                            backend,
                            target,
                            context_length=self.config.data.context_length,
                            total_cache_bytes=output_cache_bytes(baseline),
                            candidate_cache_bytes=output_cache_bytes(baseline),
                            model_name=(
                                self.config.model.model_name_or_path
                                or self.config.model.modelscope_model_id
                                or "toy"
                            ),
                            lora_rank=0,
                            configured_update_norm=self.config.updates.update_norm,
                            update_mode="random",
                        ),
                    )
                    fallback_latency = backend.estimate_latency(
                        decision,
                        context_length=self.config.data.context_length,
                    )
                    measurement = measure_backend_call(
                        partial(
                            execute_strategy,
                            backend,
                            prompt=sample.prompt,
                            baseline=baseline,
                            full=full,
                            updated=updated,
                            decision=decision,
                        ),
                        warmup_runs=self.config.measurement.warmup_runs,
                        timed_runs=self.config.measurement.timed_runs,
                        fallback_latency=fallback_latency,
                    )
                    approx = measurement.output
                    top1 = top1_agreement(full.logits, approx.logits)
                    strategy_latency = measurement.latency_p50
                    decode_latency = measurement.decode_latency_p50
                    maintenance_latency = measurement.cache_maintenance_latency_p50
                    adaptation_latency = (
                        0.0
                        if strategy.name is StrategyName.NO_ADAPTATION
                        else update_result.adaptation_latency
                    )
                    task_score = (
                        backend.score_answer(sample, approx)
                        if self.config.metrics.compute_task_metrics
                        else 0.0
                    )
                    logits_kl_value = (
                        kl_divergence(full.logits, approx.logits)
                        if self.config.metrics.compute_tensor_metrics
                        else 0.0
                    )
                    strategy_flops = (
                        output_strategy_flops(
                            approx,
                            fallback=backend.estimate_flops(
                                decision,
                                context_length=self.config.data.context_length,
                            ),
                        )
                        if self.config.metrics.compute_flops_metrics
                        else 0.0
                    )
                    full_recompute_flops = (
                        output_full_recompute_flops(
                            full,
                            fallback=backend.estimate_flops(
                                full_decision,
                                context_length=self.config.data.context_length,
                            ),
                        )
                        if self.config.metrics.compute_flops_metrics
                        else 0.0
                    )
                    attention_shift_value = (
                        attention_distribution_shift(full, approx)
                        if self.config.metrics.compute_attention_metrics
                        else None
                    )
                    baseline_fidelity, baseline_source, baseline_reference = baseline_provenance(
                        decision.strategy, output_baseline_fidelity(approx)
                    )
                    planner_source, failure_map_path, failure_map_sha256 = planner_provenance(
                        decision.strategy,
                        self.config.cache.failure_map_path,
                    )
                    records.append(
                        ExperimentRecord(
                            sample_id=sample_id,
                            update_target=target_name,
                            cache_strategy=str(decision.strategy),
                            action=str(decision.action),
                            cache_state=str(decision.state),
                            first_invalid_layer=decision.first_invalid_layer,
                            last_recomputed_layer=(
                                min(backend.num_layers, decision.last_recomputed_layer)
                                if decision.last_recomputed_layer is not None
                                else None
                            ),
                            recompute_window_size=(
                                max(
                                    0,
                                    min(backend.num_layers, decision.last_recomputed_layer)
                                    - decision.first_invalid_layer,
                                )
                                if decision.first_invalid_layer is not None
                                and decision.last_recomputed_layer is not None
                                else 0
                            ),
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
                            timing_warmup_runs=measurement.warmup_runs,
                            timing_runs=measurement.timed_runs,
                            latency_mean=measurement.latency_mean,
                            latency_p50=measurement.latency_p50,
                            latency_p95=measurement.latency_p95,
                            latency_std=measurement.latency_std,
                            accumulated_update_norm=update_result.update_norm,
                            accumulated_raw_update_norm=update_result.raw_update_norm,
                            update_norm_since_cache=update_result.update_norm,
                            raw_update_norm_since_cache=update_result.raw_update_norm,
                            update_scale=update_result.update_scale,
                            norm_control="target_l2",
                            hidden_relative_error=(
                                relative_error(full.hidden_tensor, approx.hidden_tensor)
                                if self.config.metrics.compute_tensor_metrics
                                else 0.0
                            ),
                            cache_bytes=output_cache_bytes(approx),
                            physical_cache_bytes=output_physical_cache_bytes(approx),
                            memory_allocated=output_memory_allocated(approx),
                            peak_memory_allocated=output_peak_memory_allocated(approx),
                            adaptation_latency=adaptation_latency,
                            cache_maintenance_latency=maintenance_latency,
                            decode_latency=decode_latency,
                            end_to_end_latency=adaptation_latency + strategy_latency,
                            throughput_tokens_per_s=output_throughput(approx, latency=strategy_latency),
                            recompute_fraction=estimate_recompute_fraction(
                                decision, num_layers=backend.num_layers
                            ),
                            cache_hit=is_cache_hit(decision),
                            refresh_count=1 if is_refresh_action(decision) else 0,
                            rejected_reuse=(decision.reject_reuse or decision.action is CacheAction.REJECT_UPDATE),
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
                            strategy_available=output_strategy_available(approx),
                            strategy_fallback=output_strategy_fallback(approx),
                            baseline_fidelity=baseline_fidelity,
                            baseline_source=baseline_source,
                            baseline_reference=baseline_reference,
                            cache_block_count=backend.num_layers,
                            cache_entry_count=1,
                            total_cache_bytes=output_cache_bytes(approx),
                            context_length=self.config.data.context_length,
                            model_name=(
                                self.config.model.model_name_or_path
                                or self.config.model.modelscope_model_id
                                or "toy"
                            ),
                            model_num_layers=backend.num_layers,
                            model_hidden_size=backend.hidden_size,
                            model_parameter_count=backend.parameter_count,
                            configured_update_norm=self.config.updates.update_norm,
                            baseline_task_score=baseline_task_score,
                            full_task_score=full_task_score,
                            adaptation_gain_vs_base=full_task_score - baseline_task_score,
                            attention_shift=attention_shift_value,
                            attention_metric_available=attention_shift_value is not None,
                            strategy_flops=strategy_flops,
                            full_recompute_flops=full_recompute_flops,
                            flops_fraction=(
                                strategy_flops / full_recompute_flops
                                if full_recompute_flops > 0.0
                                else 0.0
                            ),
                            planner_source=planner_source,
                            failure_map_path=failure_map_path,
                            failure_map_sha256=failure_map_sha256,
                            cache_manager_scope="condition",
                            **record_run_fields(self.config, approx, run_metadata, sample=sample),
                        )
                    )
                backend.restore_after_update()
                if self.config.checkpoint_each_target:
                    write_records(
                        records,
                        self.config.output_dir,
                        merge_existing=self.config.resume,
                        metadata_path=metadata_path,
                    )

        return write_records(
            records,
            self.config.output_dir,
            merge_existing=self.config.resume,
            metadata_path=metadata_path,
        )
