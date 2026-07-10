from __future__ import annotations

import random

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import build_strategy
from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.data.loader import build_task_samples
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
from ttt_cache_lab.updates.targets import parse_update_target


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

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
            baseline = backend.prefill(sample.prompt)
            for target_name in self.config.updates.targets:
                target = parse_update_target(target_name, num_layers=backend.num_layers)
                updated = backend.simulate_update(baseline, target, update_norm=self.config.updates.update_norm)
                full = backend.full_recompute(sample.prompt, updated)

                for strategy in strategies:
                    decision = strategy.decide(
                        target,
                        step=self.config.updates.step_count,
                        update_norm=self.config.updates.update_norm,
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
                    adaptation_latency = float(backend.last_adaptation_latency())
                    records.append(
                        ExperimentRecord(
                            sample_id=sample_id,
                            update_target=target_name,
                            cache_strategy=str(decision.strategy),
                            action=str(decision.action),
                            cache_state=str(decision.state),
                            first_invalid_layer=decision.first_invalid_layer,
                            task_score=(
                                backend.score_answer(sample, approx)
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
                            false_safe=is_false_safe(decision, full=full, approx=approx),
                            strategy_mode=output_strategy_mode(approx),
                            cache_block_count=backend.num_layers,
                            cache_entry_count=1,
                        )
                    )
                backend.restore_after_update()

        return write_records(records, self.config.output_dir)
