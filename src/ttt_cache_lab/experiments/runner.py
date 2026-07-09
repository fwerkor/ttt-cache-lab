from __future__ import annotations

import random

from ttt_cache_lab.cache.strategies import build_strategy
from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.data.synthetic import SyntheticTaskFactory
from ttt_cache_lab.experiments.results import ExperimentArtifacts, ExperimentRecord, write_records
from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.updates.targets import parse_update_target


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

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
        records: list[ExperimentRecord] = []

        for sample_id, sample in enumerate(data):
            baseline = backend.prefill(sample.prompt)
            for target_name in self.config.updates.targets:
                target = parse_update_target(target_name, num_layers=self.config.model.num_layers)
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
                    records.append(
                        ExperimentRecord(
                            sample_id=sample_id,
                            update_target=target_name,
                            cache_strategy=str(decision.strategy),
                            action=str(decision.action),
                            cache_state=str(decision.state),
                            first_invalid_layer=decision.first_invalid_layer,
                            task_score=backend.score_answer(sample, approx),
                            logits_kl=kl_divergence(full.logits, approx.logits),
                            top1_agreement=top1_agreement(full.logits, approx.logits),
                            relative_error=relative_error(full.cache_tensor, approx.cache_tensor),
                            latency_units=backend.estimate_latency(
                                decision, context_length=self.config.data.context_length
                            ),
                            reason=decision.reason,
                        )
                    )
                backend.restore_after_update()

        return write_records(records, self.config.output_dir)
