from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ttt_cache_lab.cache.blocks import CacheBlockMetadata, VersionedCacheEntry, VersionedCacheManager
from ttt_cache_lab.cache.semantics import CacheAction, CacheBlockState, CacheSemantics
from ttt_cache_lab.cache.strategies import CacheStrategy, StrategyDecision, StrategyName, build_strategy
from ttt_cache_lab.configs import VersionedExperimentConfig
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
from ttt_cache_lab.experiments.planner_runtime import build_planner_runtime
from ttt_cache_lab.experiments.provenance import planner_provenance
from ttt_cache_lab.experiments.results import ExperimentArtifacts, ExperimentRecord, write_records
from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.models.interface import BackendOutput, ModelBackend
from ttt_cache_lab.updates.targets import UpdateTarget, parse_update_target
from ttt_cache_lab.updates.updater import TTTUpdater, UpdateResult, build_updater


@dataclass
class _StrategyCache:
    output: BackendOutput
    original_output: BackendOutput
    cached_version: int
    manager: VersionedCacheManager
    refresh_count: int = 0
    cached_update_norm: float = 0.0
    blocks: tuple[CacheBlockMetadata, ...] = field(default_factory=tuple)


class VersionedExperimentRunner:
    """Run multi-step adapter-version experiments.

    This runner is the shared implementation for E1-E7. The exact experiment is
    selected by config: target set, version steps, cache strategies, adapter
    update mode, model, context length, and output directory.
    """

    def __init__(self, config: VersionedExperimentConfig) -> None:
        self.config = config

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
        strategy_managers = {
            str(strategy.name): VersionedCacheManager(
                max_cache_bytes=self.config.cache.max_cache_bytes,
                max_cache_entries=self.config.cache.max_cache_entries,
                eviction_policy=self.config.cache.eviction_policy,
            )
            for strategy in strategies
        }
        cached_version = self.config.cached_version
        if cached_version < 0:
            raise ValueError("cached_version must be non-negative")
        if any(step < cached_version for step in self.config.version_steps):
            raise ValueError("version_steps cannot contain versions older than cached_version")
        max_version = max(self.config.version_steps or [cached_version])
        target_steps = set(self.config.version_steps)
        records: list[ExperimentRecord] = []

        for sample_id, sample in enumerate(data):
            sample = backend.prepare_sample(sample, context_length=self.config.data.context_length)
            for target_name in self.config.updates.targets:
                target = parse_update_target(target_name, num_layers=backend.num_layers)
                backend.restore_after_update()
                self._prepare_backend_for_target(backend, target)
                base_v0 = backend.prefill(sample.prompt)
                updater = build_updater(
                    backend,
                    mode=self.config.adapter.update_mode,
                    sample=sample,
                    target=target,
                    rank=self.config.adapter.lora_rank,
                    alpha=self.config.adapter.lora_alpha,
                    learning_rate=self.config.adapter.learning_rate,
                    freeze_base_model=self.config.adapter.freeze_base_model,
                )
                accumulated_update_norm = 0.0
                accumulated_adaptation_latency = 0.0
                current = base_v0
                for _ in range(1, cached_version + 1):
                    version_update = self._update_one_version(updater, target, current)
                    accumulated_update_norm += version_update.update_norm
                    accumulated_adaptation_latency += version_update.adaptation_latency
                    current = version_update.output
                cached_output = (
                    backend.full_recompute(sample.prompt, current)
                    if cached_version > 0
                    else base_v0
                )
                current = cached_output

                adapter_id = f"sample-{sample_id}:{target_name}"
                base_blocks = self._make_cache_blocks(
                    output=base_v0,
                    adapter_id=adapter_id,
                    adapter_version=0,
                    cached_step=0,
                    target_name=target_name,
                    accumulated_update_norm=0.0,
                    state=CacheBlockState.VALID_EXACT,
                )
                cached_blocks = self._make_cache_blocks(
                    output=cached_output,
                    adapter_id=adapter_id,
                    adapter_version=cached_version,
                    cached_step=cached_version,
                    target_name=target_name,
                    accumulated_update_norm=accumulated_update_norm,
                    state=CacheBlockState.VALID_EXACT,
                )
                strategy_caches = {}
                for strategy in strategies:
                    manager = strategy_managers[str(strategy.name)]
                    manager.put(adapter_id, 0, VersionedCacheEntry(base_v0, base_blocks))
                    if cached_version != 0:
                        manager.put(
                            adapter_id,
                            cached_version,
                            VersionedCacheEntry(cached_output, cached_blocks),
                        )
                    strategy_caches[str(strategy.name)] = _StrategyCache(
                        output=cached_output,
                        original_output=base_v0,
                        cached_version=cached_version,
                        cached_update_norm=accumulated_update_norm,
                        blocks=cached_blocks,
                        manager=manager,
                    )

                if cached_version in target_steps:
                    self._record_step(
                        records,
                        backend=backend,
                        sample_id=sample_id,
                        sample_answer=sample,
                        target=target,
                        target_name=target_name,
                        adapter_id=adapter_id,
                        strategies=strategies,
                        strategy_caches=strategy_caches,
                        current=current,
                        full=cached_output,
                        adapter_version=cached_version,
                        accumulated_update_norm=accumulated_update_norm,
                        accumulated_adaptation_latency=accumulated_adaptation_latency,
                    )

                for step in range(cached_version + 1, max_version + 1):
                    version_update = self._update_one_version(updater, target, current)
                    accumulated_update_norm += version_update.update_norm
                    accumulated_adaptation_latency += version_update.adaptation_latency
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
                        adapter_id=adapter_id,
                        strategies=strategies,
                        strategy_caches=strategy_caches,
                        current=current,
                        full=full,
                        adapter_version=step,
                        accumulated_update_norm=accumulated_update_norm,
                        accumulated_adaptation_latency=accumulated_adaptation_latency,
                    )
                backend.restore_after_update()

        return write_records(records, self.config.output_dir)

    def _prepare_backend_for_target(self, backend: ModelBackend, target: UpdateTarget) -> None:
        if self.config.adapter.update_mode != "lora_train" or not target.is_lora:
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
        updater: TTTUpdater,
        target: UpdateTarget,
        current: BackendOutput,
    ) -> UpdateResult:
        step_count = (
            self.config.adapter.train_steps_per_version
            if self.config.adapter.update_mode == "lora_train" and target.is_lora
            else self.config.updates.step_count
        )
        return updater.update(
            current,
            target,
            step_count=step_count,
            update_norm=self.config.updates.update_norm,
        )

    def _record_step(
        self,
        records: list[ExperimentRecord],
        *,
        backend: ModelBackend,
        sample_id: int,
        sample_answer: object,
        target: UpdateTarget,
        target_name: str,
        adapter_id: str,
        strategies: Sequence[CacheStrategy],
        strategy_caches: dict[str, _StrategyCache],
        current: BackendOutput,
        full: BackendOutput,
        adapter_version: int,
        accumulated_update_norm: float,
        accumulated_adaptation_latency: float,
    ) -> None:
        full_decision = StrategyDecision(
            StrategyName.FULL_RECOMPUTE,
            CacheAction.FULL_RECOMPUTE,
            CacheBlockState.INVALID,
            None,
            "FLOP accounting probe.",
            recompute_fraction=1.0,
        )
        for strategy in strategies:
            cache_key = str(strategy.name)
            cached = strategy_caches[cache_key]
            record_cached_version = cached.cached_version
            version_gap = adapter_version - record_cached_version
            update_norm_since_cache = max(0.0, accumulated_update_norm - cached.cached_update_norm)
            decision = strategy.decide_with_runtime(
                target,
                step=version_gap,
                update_norm=update_norm_since_cache,
                runtime=build_planner_runtime(
                    backend,
                    target,
                    context_length=self.config.data.context_length,
                    total_cache_bytes=cached.manager.total_cache_bytes(),
                    candidate_cache_bytes=output_cache_bytes(cached.output),
                    model_name=(
                        self.config.model.model_name_or_path
                        or self.config.model.modelscope_model_id
                        or "toy"
                    ),
                    lora_rank=self.config.adapter.lora_rank,
                    configured_update_norm=self.config.updates.update_norm,
                    update_mode=self.config.adapter.update_mode,
                ),
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
                baseline_output = cached.original_output
            elif strategy.name is StrategyName.ADAPTER_SPECIFIC_CACHE:
                entry = cached.manager.get(adapter_id, adapter_version)
                existing = entry.output if entry is not None else None
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
            elif (
                strategy.name is not StrategyName.ADAPTIVE_NO_VERSION
                and adapter_version == cached.cached_version
                and decision.action is not CacheAction.FULL_RECOMPUTE
            ):
                decision = StrategyDecision(
                    decision.strategy,
                    CacheAction.REUSE_EXACT,
                    CacheBlockState.VALID_EXACT,
                    None,
                    "Cache version matches adapter version; reuse is exact.",
                )
            if strategy.name is StrategyName.ORACLE_PLANNER:
                decision, approx = self._run_measured_oracle(
                    backend=backend,
                    baseline=baseline_output,
                    full=full,
                    current=current,
                    target=target,
                    sample=sample_answer,
                )
            else:
                approx = backend.apply_cache_strategy(
                    baseline=baseline_output,
                    full=full,
                    updated=current,
                    decision=decision,
                )
            new_refresh_count = cached.refresh_count + (1 if is_refresh_action(decision) else 0)
            block_state = (
                CacheBlockState.VALID_EXACT
                if decision.action
                in {
                    CacheAction.FULL_RECOMPUTE,
                    CacheAction.REUSE_EXACT,
                    CacheAction.PARTIAL_RECOMPUTE,
                    CacheAction.REJECT_UPDATE,
                }
                else decision.state
            )
            new_blocks = self._make_cache_blocks(
                output=approx,
                adapter_id=adapter_id,
                adapter_version=adapter_version,
                cached_step=adapter_version,
                target_name=target_name,
                accumulated_update_norm=accumulated_update_norm,
                state=block_state,
                previous=cached.blocks,
                first_invalid_layer=decision.first_invalid_layer,
                action=decision.action,
            )
            if strategy.name is StrategyName.ADAPTER_SPECIFIC_CACHE:
                cached.manager.put(adapter_id, adapter_version, VersionedCacheEntry(approx, new_blocks))
                cached.output = approx
                cached.cached_version = adapter_version
                cached.cached_update_norm = accumulated_update_norm
                cached.blocks = new_blocks
                cached.refresh_count = new_refresh_count
            elif strategy.name not in {
                StrategyName.NO_ADAPTATION,
                StrategyName.ALORA_PREFIX_REUSE,
            } and decision.action in {
                CacheAction.FULL_RECOMPUTE,
                CacheAction.REUSE_EXACT,
                CacheAction.PARTIAL_RECOMPUTE,
                CacheAction.DELTA_CORRECT,
                CacheAction.REJECT_UPDATE,
            }:
                cached.output = approx
                cached.cached_version = adapter_version
                cached.cached_update_norm = accumulated_update_norm
                cached.blocks = new_blocks
                cached.refresh_count = new_refresh_count
                cached.manager.put(adapter_id, adapter_version, VersionedCacheEntry(approx, new_blocks))
            top1 = top1_agreement(full.logits, approx.logits)
            fallback_latency = backend.estimate_latency(
                decision,
                context_length=self.config.data.context_length,
            )
            strategy_latency = output_strategy_latency(approx, fallback=fallback_latency)
            decode_latency = output_decode_latency(approx)
            maintenance_latency = output_cache_maintenance_latency(approx)
            task_score = (
                backend.score_answer(sample_answer, approx)  # type: ignore[arg-type]
                if self.config.metrics.compute_task_metrics
                else 0.0
            )
            full_task_score = (
                backend.score_answer(sample_answer, full)  # type: ignore[arg-type]
                if self.config.metrics.compute_task_metrics
                else 0.0
            )
            logits_kl_value = (
                kl_divergence(full.logits, approx.logits)
                if self.config.metrics.compute_tensor_metrics
                else 0.0
            )
            strategy_adaptation_latency = (
                0.0
                if strategy.name is StrategyName.NO_ADAPTATION
                else accumulated_adaptation_latency
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
                    experiment_id=self.config.experiment_id,
                    adapter_id=adapter_id,
                    adapter_version=adapter_version,
                    cached_version=record_cached_version,
                    version_gap=version_gap,
                    update_step=adapter_version,
                    accumulated_update_norm=accumulated_update_norm,
                    update_norm_since_cache=update_norm_since_cache,
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
                    adaptation_latency=strategy_adaptation_latency,
                    cache_maintenance_latency=maintenance_latency,
                    decode_latency=decode_latency,
                    end_to_end_latency=strategy_adaptation_latency + strategy_latency,
                    throughput_tokens_per_s=output_throughput(approx, latency=strategy_latency),
                    recompute_fraction=estimate_recompute_fraction(decision, num_layers=backend.num_layers),
                    cache_hit=is_cache_hit(decision),
                    refresh_count=new_refresh_count,
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
                    cache_block_count=cached.manager.total_block_count(),
                    cache_entry_count=cached.manager.entry_count(),
                    total_cache_bytes=cached.manager.total_cache_bytes(),
                    evicted_cache_entries=cached.manager.eviction_count(),
                    context_length=self.config.data.context_length,
                    model_name=(
                        self.config.model.model_name_or_path
                        or self.config.model.modelscope_model_id
                        or "toy"
                    ),
                    model_num_layers=backend.num_layers,
                    model_hidden_size=self.config.model.hidden_size,
                    configured_update_norm=self.config.updates.update_norm,
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
                )
            )

    def _make_cache_blocks(
        self,
        *,
        output: BackendOutput,
        adapter_id: str,
        adapter_version: int,
        cached_step: int,
        target_name: str,
        accumulated_update_norm: float,
        state: CacheBlockState,
        previous: tuple[CacheBlockMetadata, ...] = (),
        first_invalid_layer: int | None = None,
        action: CacheAction = CacheAction.FULL_RECOMPUTE,
    ) -> tuple[CacheBlockMetadata, ...]:
        extras = output.extras or {}
        token_length = int(extras.get("token_length", self.config.data.context_length))
        model_id = self.config.model.model_name_or_path or "toy"
        precision = self.config.model.torch_dtype
        attention_impl = str(extras.get("attention_implementation", "transformers_default"))
        semantics = {
            CacheBlockState.VALID_EXACT: CacheSemantics.EXACT_CURRENT,
            CacheBlockState.VALID_FROZEN: CacheSemantics.FROZEN_EVIDENCE,
        }.get(state, CacheSemantics.BOUNDED_STALE)
        blocks = []
        for layer_id in range(len(output.cache_tensor)):
            if (
                previous
                and first_invalid_layer is not None
                and layer_id < first_invalid_layer
                and action in {CacheAction.PARTIAL_RECOMPUTE, CacheAction.DELTA_CORRECT}
            ):
                blocks.append(previous[layer_id])
                continue
            blocks.append(
                CacheBlockMetadata(
                    token_start=0,
                    token_end=token_length,
                    layer_id=layer_id,
                    base_model_id=model_id,
                    adapter_id=adapter_id,
                    adapter_version=adapter_version,
                    cached_step=cached_step,
                    update_target=target_name,
                    accumulated_update_norm=accumulated_update_norm,
                    state=state,
                    semantics=semantics,
                    precision=precision,
                    attention_implementation=attention_impl,
                )
            )
        return tuple(blocks)

    def _run_measured_oracle(
        self,
        *,
        backend: ModelBackend,
        baseline: BackendOutput,
        full: BackendOutput,
        current: BackendOutput,
        target: UpdateTarget,
        sample: object,
    ) -> tuple[StrategyDecision, BackendOutput]:
        candidates = [
            StrategyDecision(
                StrategyName.ORACLE_PLANNER,
                CacheAction.REUSE_STALE,
                CacheBlockState.VALID_APPROX,
                None,
                "Oracle candidate: stale reuse.",
            ),
            StrategyDecision(
                StrategyName.ORACLE_PLANNER,
                CacheAction.DELTA_CORRECT,
                CacheBlockState.VALID_APPROX,
                target.layer,
                "Oracle candidate: delta correction.",
                recompute_fraction=0.15,
            ),
        ]
        if target.layer is not None:
            candidates.append(
                StrategyDecision(
                    StrategyName.ORACLE_PLANNER,
                    CacheAction.PARTIAL_RECOMPUTE,
                    CacheBlockState.INVALID,
                    target.layer,
                    "Oracle candidate: native layer restart.",
                )
            )
        candidates.append(
            StrategyDecision(
                StrategyName.ORACLE_PLANNER,
                CacheAction.FULL_RECOMPUTE,
                CacheBlockState.INVALID,
                None,
                "Oracle candidate: full recompute.",
                recompute_fraction=1.0,
            )
        )

        full_score = backend.score_answer(sample, full)  # type: ignore[arg-type]
        feasible: list[tuple[float, StrategyDecision, BackendOutput, float, float, float]] = []
        for candidate in candidates:
            try:
                output = backend.apply_cache_strategy(
                    baseline=baseline,
                    full=full,
                    updated=current,
                    decision=candidate,
                )
            except RuntimeError:
                continue
            mode = output_strategy_mode(output)
            if candidate.action is CacheAction.DELTA_CORRECT and mode.startswith("unavailable_"):
                continue
            candidate_kl = kl_divergence(full.logits, output.logits)
            candidate_top1 = top1_agreement(full.logits, output.logits)
            candidate_score = backend.score_answer(sample, output)  # type: ignore[arg-type]
            safe = (
                candidate_kl <= self.config.cache.oracle_kl_threshold
                and candidate_top1 >= self.config.cache.oracle_top1_threshold
                and full_score - candidate_score <= self.config.cache.oracle_task_drop_threshold
            )
            if candidate.action is CacheAction.FULL_RECOMPUTE:
                safe = True
            if not safe:
                continue
            fallback_cost = backend.estimate_latency(candidate, context_length=self.config.data.context_length)
            cost = output_strategy_latency(output, fallback=fallback_cost)
            feasible.append((cost, candidate, output, candidate_kl, candidate_top1, candidate_score))

        if not feasible:
            raise RuntimeError("Measured oracle found no feasible action, including full recompute")
        cost, selected, output, selected_kl, selected_top1, selected_score = min(
            feasible,
            key=lambda item: (item[0], item[1].recompute_fraction),
        )
        decision = StrategyDecision(
            StrategyName.ORACLE_PLANNER,
            selected.action,
            selected.state,
            selected.first_invalid_layer,
            (
                f"Measured oracle selected {selected.action}: latency={cost:.6g}, "
                f"KL={selected_kl:.6g}, top1={selected_top1:.3f}, task={selected_score:.3f}."
            ),
            recompute_fraction=selected.recompute_fraction,
        )
        return decision, output



def write_version_summary(input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    sweep_fields = [field for field in fieldnames if field.startswith("sweep.")]
    dimension_fields = [
        field
        for field in (
            "run_name",
            *sweep_fields,
            "experiment_id",
            "model_name",
            "model_num_layers",
            "model_hidden_size",
            "context_length",
            "task_name",
            "update_target",
            "cache_strategy",
            "adapter_version",
            "cached_version",
            "version_gap",
            "lora_rank",
            "configured_update_norm",
            "update_mode",
            "norm_control",
            "seed",
        )
        if field in fieldnames
    ]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in dimension_fields)
        groups.setdefault(key, []).append(row)

    metric_fields = [
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
        "update_norm_since_cache_mean",
        "cache_block_count_mean",
        "adaptation_latency_mean",
        "cache_maintenance_latency_mean",
        "decode_latency_mean",
        "end_to_end_latency_mean",
        "throughput_tokens_per_s_mean",
        "peak_memory_allocated_mean",
        "cache_entry_count_mean",
        "total_cache_bytes_mean",
        "evicted_cache_entries_mean",
        "attention_shift_mean",
        "attention_metric_available_rate",
        "strategy_flops_mean",
        "full_recompute_flops_mean",
        "flops_fraction_mean",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*dimension_fields, *metric_fields])
        writer.writeheader()
        for key, records in sorted(groups.items()):
            dimensions = dict(zip(dimension_fields, key, strict=True))
            writer.writerow(
                {
                    **dimensions,
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
                    "update_norm_since_cache_mean": _mean(records, "update_norm_since_cache"),
                    "cache_block_count_mean": _mean(records, "cache_block_count"),
                    "adaptation_latency_mean": _mean(records, "adaptation_latency"),
                    "cache_maintenance_latency_mean": _mean(records, "cache_maintenance_latency"),
                    "decode_latency_mean": _mean(records, "decode_latency"),
                    "end_to_end_latency_mean": _mean(records, "end_to_end_latency"),
                    "throughput_tokens_per_s_mean": _mean(records, "throughput_tokens_per_s"),
                    "peak_memory_allocated_mean": _mean(records, "peak_memory_allocated"),
                    "cache_entry_count_mean": _mean(records, "cache_entry_count"),
                    "total_cache_bytes_mean": _mean(records, "total_cache_bytes"),
                    "evicted_cache_entries_mean": _mean(records, "evicted_cache_entries"),
                    "attention_shift_mean": _mean(records, "attention_shift"),
                    "attention_metric_available_rate": _mean_bool(
                        records, "attention_metric_available"
                    ),
                    "strategy_flops_mean": _mean(records, "strategy_flops"),
                    "full_recompute_flops_mean": _mean(records, "full_recompute_flops"),
                    "flops_fraction_mean": _mean(records, "flops_fraction"),
                }
            )


def _mean(records: list[dict[str, str]], field: str) -> float:
    values = [float(record[field]) for record in records if record.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else 0.0


def _mean_bool(records: list[dict[str, str]], field: str) -> float:
    values = [record.get(field, "False").lower() == "true" for record in records]
    return sum(1.0 for value in values if value) / len(values) if values else 0.0
