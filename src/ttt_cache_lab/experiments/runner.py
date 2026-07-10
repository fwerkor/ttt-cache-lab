from __future__ import annotations

import random

from ttt_cache_lab.cache.planner import PlannerRuntime
from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName, build_strategy
from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.data.loader import build_task_samples
from ttt_cache_lab.experiments.metrics import (
    attention_distribution_shift,
    estimate_recompute_fraction,
    is_cache_hit,
    is_false_safe,
    is_refresh_action,
    output_baseline_fidelity,
    output_cache_bytes,
    output_cache_maintenance_latency,
    output_decode_latency,
    output_full_recompute_flops,
    output_memory_allocated,
    output_peak_memory_allocated,
    output_strategy_flops,
    output_strategy_latency,
    output_strategy_mode,
    output_throughput,
)
from ttt_cache_lab.experiments.provenance import planner_provenance
from ttt_cache_lab.experiments.results import ExperimentArtifacts, ExperimentRecord, write_records
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
        strategies = [
            build_strategy(
                name,
                refresh_period=self.config.cache.refresh_period,
                update_norm_threshold=self.config.cache.update_norm_threshold,
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

                for strategy in strategies:
                    decision = strategy.decide_with_runtime(
                        target,
                        step=self.config.updates.step_count,
                        update_norm=self.config.updates.update_norm,
                        runtime=PlannerRuntime(
                            total_cache_bytes=output_cache_bytes(baseline),
                            candidate_cache_bytes=output_cache_bytes(baseline),
                            full_recompute_latency=max(
                                1e-9,
                                backend.estimate_latency(
                                    strategy.decide(
                                        target,
                                        step=self.config.updates.step_count,
                                        update_norm=self.config.updates.update_norm,
                                    ),
                                    context_length=self.config.data.context_length,
                                ),
                            ),
                            model_name=(
                                self.config.model.model_name_or_path
                                or self.config.model.modelscope_model_id
                                or "toy"
                            ),
                            context_length=self.config.data.context_length,
                            configured_update_norm=self.config.updates.update_norm,
                            update_mode="random",
                        ),
                    )
                    approx = backend.apply_cache_strategy(
                        baseline=baseline,
                        full=full,
                        updated=updated,
                        decision=decision,
                    )
                    top1 = top1_agreement(full.logits, approx.logits)
                    fallback_latency = backend.estimate_latency(
                        decision,
                        context_length=self.config.data.context_length,
                    )
                    strategy_latency = output_strategy_latency(approx, fallback=fallback_latency)
                    decode_latency = output_decode_latency(approx)
                    maintenance_latency = output_cache_maintenance_latency(approx)
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
                            hidden_relative_error=(
                                relative_error(full.hidden_tensor, approx.hidden_tensor)
                                if self.config.metrics.compute_tensor_metrics
                                else 0.0
                            ),
                            cache_bytes=output_cache_bytes(approx),
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
                            baseline_fidelity=output_baseline_fidelity(approx),
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
                            model_hidden_size=self.config.model.hidden_size,
                            configured_update_norm=self.config.updates.update_norm,
                            attention_shift=(
                                attention_distribution_shift(full, approx)
                                if self.config.metrics.compute_attention_metrics
                                else 0.0
                            ),
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
                        )
                    )
                backend.restore_after_update()

        return write_records(records, self.config.output_dir)
