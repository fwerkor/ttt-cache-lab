from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.data.loader import build_task_samples
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.experiments.block_ranker import (
    load_block_ranker,
    route_committed_candidate,
    route_reference_candidate,
    route_zero_probe_recompute,
    score_block_features,
)
from ttt_cache_lab.metrics.tensor import kl_divergence, top1_agreement
from ttt_cache_lab.models.factory import build_backend
from ttt_cache_lab.models.interface import BackendOutput, ModelBackend
from ttt_cache_lab.updates.targets import UpdateTarget, parse_update_target
from ttt_cache_lab.updates.updater import build_updater


@dataclass(frozen=True)
class BlockwiseArtifacts:
    records_csv: Path
    records_jsonl: Path
    frontier_csv: Path
    masks_csv: Path
    features_csv: Path
    report_markdown: Path


@dataclass(frozen=True)
class _Evaluation:
    output: BackendOutput
    logits_kl: float
    top1_agreement: float


@dataclass(frozen=True)
class _SearchPoint:
    mask: np.ndarray
    evaluation: _Evaluation
    score: float
    probe_count: int


_EVALUATION_HEAVY_EXTRA_KEYS = frozenset(
    {
        "past_key_values",
        "hidden_states",
        "lora_cache",
        "prompt_state",
    }
)


def _compact_evaluation_output(output: BackendOutput) -> BackendOutput:
    """Drop device-resident candidate state after all probe metrics are computed."""
    extras = output.extras or {}
    compact_extras = {
        key: value
        for key, value in extras.items()
        if key not in _EVALUATION_HEAVY_EXTRA_KEYS
    }
    return BackendOutput(
        logits=output.logits,
        cache_tensor=np.empty((0,), dtype=np.float32),
        hidden_tensor=np.empty((0,), dtype=np.float32),
        parameter_version=output.parameter_version,
        extras=compact_extras,
    )


def _release_accelerator_cache(backend: ModelBackend) -> None:
    """Return released candidate buffers to the active accelerator allocator."""
    torch = getattr(backend, "torch", None)
    if torch is None:
        return
    for name in ("npu", "cuda"):
        accelerator = getattr(torch, name, None)
        empty_cache = getattr(accelerator, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
            return


def run_blockwise_exploration(
    config: VersionedExperimentConfig,
    *,
    block_sizes: tuple[int, ...],
    version_gap: int,
    budget_fractions: tuple[float, ...],
    oracle_candidate_limit: int = 24,
    oracle_max_cells: int = 16,
    direct_oracle_max_blocks: int = 0,
    sparse_beam_widths: tuple[int, ...] = (2, 4),
    sparse_cost_penalties: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01),
    sparse_swap_rounds: int = 4,
    reference_probe_lengths: tuple[int, ...] = (1,),
    sparse_stale_margins: tuple[float, ...] = (0.0,),
    compute_cache_surgery_oracles: bool = True,
    compute_structured_sparse_search: bool = True,
    sparse_policy_only: bool = False,
    sparse_policy_variant: str = "reference",
    sparse_ranker_path: Path | None = None,
) -> BlockwiseArtifacts:
    if not block_sizes or any(block_size <= 0 for block_size in block_sizes):
        raise ValueError("block sizes must be positive")
    if len(set(block_sizes)) != len(block_sizes):
        raise ValueError("block sizes must be unique")
    if version_gap <= 0:
        raise ValueError("version_gap must be positive")
    if not budget_fractions or any(value <= 0.0 or value > 1.0 for value in budget_fractions):
        raise ValueError("budget fractions must be in (0, 1]")
    if oracle_candidate_limit < 1 or oracle_max_cells < 1:
        raise ValueError("oracle limits must be positive")
    if direct_oracle_max_blocks < 0:
        raise ValueError("direct_oracle_max_blocks must be nonnegative")
    if any(width < 1 for width in sparse_beam_widths):
        raise ValueError("sparse beam widths must be positive")
    if len(set(sparse_beam_widths)) != len(sparse_beam_widths):
        raise ValueError("sparse beam widths must be unique")
    if any(penalty < 0.0 for penalty in sparse_cost_penalties):
        raise ValueError("sparse cost penalties must be nonnegative")
    if len(set(sparse_cost_penalties)) != len(sparse_cost_penalties):
        raise ValueError("sparse cost penalties must be unique")
    if sparse_swap_rounds < 0:
        raise ValueError("sparse_swap_rounds must be nonnegative")
    if not reference_probe_lengths or any(length <= 0 for length in reference_probe_lengths):
        raise ValueError("reference probe lengths must be positive")
    if len(set(reference_probe_lengths)) != len(reference_probe_lengths):
        raise ValueError("reference probe lengths must be unique")
    if any(margin < 0.0 for margin in sparse_stale_margins):
        raise ValueError("sparse stale margins must be nonnegative")
    if len(set(sparse_stale_margins)) != len(sparse_stale_margins):
        raise ValueError("sparse stale margins must be unique")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_ranker = (
        load_block_ranker(sparse_ranker_path) if sparse_ranker_path is not None else None
    )
    if sparse_policy_only and sparse_ranker is None:
        raise ValueError("sparse_policy_only requires a fitted sparse ranker")
    if sparse_policy_variant not in {
        "reference",
        "baseline_reference",
        "committed",
        "recompute",
    }:
        raise ValueError(
            "sparse_policy_variant must be 'reference', 'baseline_reference', "
            "'committed', or 'recompute'"
        )
    backend = build_backend(config.model, seed=config.seed)
    backend.configure_metrics(capture_attention=True)
    splice = getattr(backend, "probe_blockwise_cache_splice", None)
    if not callable(splice):
        raise RuntimeError("The selected backend does not implement blockwise cache splicing")

    records_path = output_dir / "blockwise_records.csv"
    records_jsonl_path = output_dir / "blockwise_records.jsonl"
    frontier_path = output_dir / "block_frontier.csv"
    masks_path = output_dir / "block_masks.csv"
    features_path = output_dir / "block_features.csv"
    records: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    completed_conditions: set[tuple[int, str, int, int, int]] = set()
    if config.resume:
        records = _read_jsonl(records_jsonl_path)
        frontier_rows = _read_rows(frontier_path)
        mask_rows = _read_rows(masks_path)
        feature_rows = _read_rows(features_path)
        completed_conditions = _completed_condition_keys(
            records,
            frontier_rows,
            mask_rows,
        )
        records = [row for row in records if _condition_key(row) in completed_conditions]
        frontier_rows = [
            row for row in frontier_rows if _condition_key(row) in completed_conditions
        ]
        mask_rows = [
            row for row in mask_rows if _condition_key(row) in completed_conditions
        ]
        feature_rows = [
            row for row in feature_rows if _condition_key(row) in completed_conditions
        ]
    samples = build_task_samples(config.data, seed=config.seed)
    try:
        for sample_id, raw_sample in enumerate(samples):
            sample = _single_token_sample(raw_sample)
            sample = backend.prepare_sample(sample, context_length=config.data.context_length)
            for target_name in config.updates.targets:
                pending_block_sizes = tuple(
                    block_size
                    for block_size in block_sizes
                    if (
                        sample_id,
                        target_name,
                        block_size,
                        version_gap,
                        config.data.context_length,
                    )
                    not in completed_conditions
                )
                if not pending_block_sizes:
                    continue
                target = parse_update_target(target_name, num_layers=backend.num_layers)
                backend.restore_after_update()
                _prepare_target(backend, config, target)
                baseline = backend.prefill(sample.prompt)
                fingerprint = getattr(backend, "adapter_state_fingerprint", None)
                initial_adapter_fingerprint = (
                    str(fingerprint()) if callable(fingerprint) else "unavailable"
                )
                updater = build_updater(
                    backend,
                    mode=config.adapter.update_mode,
                    sample=sample,
                    target=target,
                    rank=config.adapter.lora_rank,
                    alpha=config.adapter.lora_alpha,
                    learning_rate=config.adapter.learning_rate,
                    freeze_base_model=config.adapter.freeze_base_model,
                    norm_control=config.adapter.norm_control,
                )
                current = baseline
                accumulated_update_norm = 0.0
                for _ in range(version_gap):
                    result = updater.update(
                        current,
                        target,
                        step_count=(
                            config.adapter.train_steps_per_version
                            if config.adapter.update_mode == "lora_train" and target.is_lora
                            else config.updates.step_count
                        ),
                        update_norm=config.updates.update_norm,
                    )
                    current = result.output
                    accumulated_update_norm += result.update_norm
                final_adapter_fingerprint = (
                    str(fingerprint()) if callable(fingerprint) else "unavailable"
                )
                full = backend.full_recompute(sample.prompt, current)
                baseline_extras = baseline.extras or {}
                full_extras = full.extras or {}
                base_condition = {
                    "sample_id": sample_id,
                    "dataset_sample_id": str(sample.metadata.get("dataset_sample_id", sample_id)),
                    "task_name": config.data.task,
                    "model_name": config.model.model_name_or_path or "toy",
                    "update_target": target_name,
                    "target_layer": target.layer if target.layer is not None else 0,
                    "version_gap": version_gap,
                    "configured_update_norm": config.updates.update_norm,
                    "accumulated_update_norm": accumulated_update_norm,
                    "initial_adapter_fingerprint": initial_adapter_fingerprint,
                    "final_adapter_fingerprint": final_adapter_fingerprint,
                    "context_length": config.data.context_length,
                    "baseline_prefill_latency": float(
                        baseline_extras.get("prefill_latency", 0.0)
                    ),
                    "baseline_decode_latency": float(
                        baseline_extras.get("decode_latency", 0.0)
                    ),
                    "full_recompute_prefill_latency": float(
                        full_extras.get("prefill_latency", 0.0)
                    ),
                    "full_recompute_decode_latency": float(
                        full_extras.get("decode_latency", 0.0)
                    ),
                    "full_recompute_strategy_latency": float(
                        full_extras.get("strategy_latency", 0.0)
                    ),
                    "full_recompute_flops": float(
                        full_extras.get("full_recompute_flops", 0.0)
                    ),
                    "seed": config.seed,
                }
                reference_token_ids = _reference_token_ids(backend, sample)
                for block_size in pending_block_sizes:
                    condition = {
                        **base_condition,
                        "block_size": block_size,
                        "reference_token_id": reference_token_ids[0] if reference_token_ids else -1,
                        "reference_token_count": len(reference_token_ids),
                    }
                    (
                        condition_records,
                        condition_frontier,
                        condition_masks,
                        condition_features,
                    ) = _explore_condition(
                        backend=backend,
                        sample=sample,
                        splice=splice,
                        baseline=baseline,
                        full=full,
                        target=target,
                        condition=condition,
                        budget_fractions=budget_fractions,
                        oracle_candidate_limit=oracle_candidate_limit,
                        oracle_max_cells=oracle_max_cells,
                        direct_oracle_max_blocks=direct_oracle_max_blocks,
                        sparse_beam_widths=sparse_beam_widths,
                        sparse_cost_penalties=sparse_cost_penalties,
                        sparse_swap_rounds=sparse_swap_rounds,
                        reference_probe_lengths=reference_probe_lengths,
                        sparse_stale_margins=sparse_stale_margins,
                        compute_cache_surgery_oracles=compute_cache_surgery_oracles,
                        compute_structured_sparse_search=compute_structured_sparse_search,
                        sparse_policy_only=sparse_policy_only,
                        sparse_policy_variant=sparse_policy_variant,
                        sparse_ranker=sparse_ranker,
                        sparse_ranker_path=sparse_ranker_path,
                    )
                    records.extend(condition_records)
                    frontier_rows.extend(condition_frontier)
                    mask_rows.extend(condition_masks)
                    feature_rows.extend(condition_features)
                    # Checkpoint each expensive block-size condition so a later
                    # failure cannot discard all previously completed search.
                    _write_rows(records_path, records)
                    _write_jsonl(records_jsonl_path, records)
                    _write_rows(frontier_path, frontier_rows)
                    _write_rows(masks_path, mask_rows)
                    _write_rows(features_path, feature_rows)
                    completed_conditions.add(_condition_key(condition))
                    _release_accelerator_cache(backend)
    finally:
        backend.restore_after_update()

    report_path = output_dir / "blockwise_report.md"
    report_path.write_text(_report(records, frontier_rows), encoding="utf-8")
    return BlockwiseArtifacts(
        records_csv=records_path,
        records_jsonl=records_jsonl_path,
        frontier_csv=frontier_path,
        masks_csv=masks_path,
        features_csv=features_path,
        report_markdown=report_path,
    )


def _explore_condition(
    *,
    backend: ModelBackend,
    sample: TaskSample,
    splice: Any,
    baseline: BackendOutput,
    full: BackendOutput,
    target: UpdateTarget,
    condition: dict[str, Any],
    budget_fractions: tuple[float, ...],
    oracle_candidate_limit: int,
    oracle_max_cells: int,
    direct_oracle_max_blocks: int,
    sparse_beam_widths: tuple[int, ...],
    sparse_cost_penalties: tuple[float, ...],
    sparse_swap_rounds: int,
    reference_probe_lengths: tuple[int, ...],
    sparse_stale_margins: tuple[float, ...],
    compute_cache_surgery_oracles: bool,
    compute_structured_sparse_search: bool,
    sparse_policy_only: bool,
    sparse_policy_variant: str,
    sparse_ranker: dict[str, Any] | None,
    sparse_ranker_path: Path | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    old_layers = _past_layers(baseline)
    new_layers = _past_layers(full)
    if len(old_layers) != len(new_layers):
        raise ValueError("Old and current cache layer counts differ")
    token_count = int(old_layers[0][0].shape[-2])
    block_size = int(condition["block_size"])
    block_count = math.ceil(token_count / block_size)
    layer_count = len(old_layers)
    start_layer = target.layer if target.layer is not None else 0
    start_layer = max(0, min(layer_count - 1, start_layer))
    eligible = np.zeros((layer_count, block_count), dtype=bool)
    eligible[start_layer:, :] = True
    total_eligible = int(np.count_nonzero(eligible))
    reference_token_ids = _reference_token_ids(backend, sample)
    effective_reference_lengths = tuple(
        sorted(
            {
                length
                for length in reference_probe_lengths
                if length <= len(reference_token_ids)
            }
        )
    )
    sequence_scorer = getattr(backend, "score_reference_sequence", None)
    full_reference_log_probabilities = None
    if (
        callable(sequence_scorer)
        and full.extras
        and effective_reference_lengths
        and max(effective_reference_lengths) > 1
    ):
        full_past = full.extras.get("past_key_values")
        if full_past is not None:
            full_reference_metrics = sequence_scorer(
                baseline=baseline,
                past=full_past,
                reference_token_ids=reference_token_ids,
                probe_lengths=effective_reference_lengths,
                return_profile=True,
            )
            if isinstance(full_reference_metrics, dict):
                full_reference_log_probabilities = full_reference_metrics.pop(
                    "_reference_log_probabilities",
                    None,
                )
                full.extras.update(full_reference_metrics)

    def enrich_reference_metrics(output: BackendOutput) -> None:
        if (
            not callable(sequence_scorer)
            or not output.extras
            or not effective_reference_lengths
            or max(effective_reference_lengths) <= 1
        ):
            return
        past = output.extras.get("past_key_values")
        if past is None:
            return
        metrics = sequence_scorer(
            baseline=baseline,
            past=past,
            reference_token_ids=reference_token_ids,
            probe_lengths=effective_reference_lengths,
            reference_log_probabilities=full_reference_log_probabilities,
        )
        if isinstance(metrics, dict):
            output.extras.update(metrics)

    kv_scores, attention_scores = _cell_scores(
        old_layers,
        new_layers,
        full,
        block_size=block_size,
        block_count=block_count,
    )
    condition.update(
        _prefix_hidden_cascade_metrics(
            baseline=baseline,
            full=full,
            target_layer=start_layer,
        )
    )
    condition.update(
        _cache_cascade_metrics(
            kv_scores=kv_scores,
            target_layer=start_layer,
        )
    )
    cache: dict[bytes, _Evaluation] = {}
    stale_output_reference: BackendOutput | None = None

    def enrich_local_attention_metrics(output: BackendOutput) -> None:
        if output.extras is None or stale_output_reference is None:
            return
        output.extras.update(
            _attention_residual_metrics(
                candidate=output,
                full=full,
                stale=stale_output_reference,
                baseline=baseline,
                target_layer=start_layer,
            )
        )

    def evaluate(mask: np.ndarray) -> _Evaluation:
        key = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached
        output = splice(
            baseline=baseline,
            full=full,
            block_mask=mask,
            block_size=block_size,
        )
        enrich_local_attention_metrics(output)
        enrich_reference_metrics(output)
        value = _Evaluation(
            output=_compact_evaluation_output(output),
            logits_kl=kl_divergence(full.logits, output.logits),
            top1_agreement=top1_agreement(full.logits, output.logits),
        )
        del output
        cache[key] = value
        return value

    empty = np.zeros_like(eligible)
    stale = evaluate(empty)
    stale_output_reference = stale.output
    router_feature_started = time.perf_counter()
    if sparse_policy_variant == "recompute":
        condition["router_baseline_stale_kl"] = (
            _zero_probe_baseline_stale_kl(
                baseline_logits=baseline.logits,
                stale_logits=stale.output.logits,
            )
        )
    else:
        condition.update(
            _zero_probe_failure_metrics(
                baseline_logits=baseline.logits,
                stale_logits=stale.output.logits,
            )
        )
    condition["router_feature_latency"] = (
        time.perf_counter() - router_feature_started
    )
    enrich_local_attention_metrics(stale.output)
    enrich_reference_metrics(stale.output)
    sparse_probe = getattr(backend, "probe_blockwise_lora_delta", None)
    sparse_score_fn = getattr(backend, "blockwise_lora_delta_scores", None)
    sparse_cache: dict[bytes, _Evaluation] = {}

    def probe_sparse(mask: np.ndarray) -> _Evaluation:
        if not callable(sparse_probe):
            raise RuntimeError("Backend does not implement block-sparse LoRA delta repair")
        output = sparse_probe(
            baseline=baseline,
            block_mask=mask,
            block_size=block_size,
        )
        enrich_local_attention_metrics(output)
        enrich_reference_metrics(output)
        value = _Evaluation(
            output=_compact_evaluation_output(output),
            logits_kl=kl_divergence(full.logits, output.logits),
            top1_agreement=top1_agreement(full.logits, output.logits),
        )
        del output
        return value

    def evaluate_sparse(mask: np.ndarray) -> _Evaluation:
        key = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        cached = sparse_cache.get(key)
        if cached is not None:
            return cached
        value = probe_sparse(mask)
        sparse_cache[key] = value
        return value

    sparse_scores: dict[str, np.ndarray] | None = None
    if callable(sparse_score_fn) and callable(sparse_probe):
        try:
            sparse_scores = sparse_score_fn(
                baseline=baseline,
                stale=stale.output,
                block_size=block_size,
            )
        except ValueError:
            sparse_scores = None
    records = [
        _record(
            condition,
            selector="stale",
            requested_budget_fraction=0.0,
            mask=empty,
            eligible=eligible,
            evaluation=stale,
            stale_kl=stale.logits_kl,
        )
    ]
    if sparse_ranker is not None and sparse_policy_variant == "recompute":
        trigger_recompute, risk_score, recompute_policy = route_zero_probe_recompute(
            sparse_ranker,
            update_target=str(condition["update_target"]),
            condition=condition,
        )
        if trigger_recompute:
            policy_evaluation = _Evaluation(
                output=full,
                logits_kl=0.0,
                top1_agreement=1.0,
            )
            action_latency = float(
                condition.get("full_recompute_strategy_latency", 0.0)
            )
            action_flops = float(condition.get("full_recompute_flops", 0.0))
            cache_latency = float(
                condition.get("full_recompute_prefill_latency", 0.0)
            )
            decode_latency = float(
                condition.get("full_recompute_decode_latency", 0.0)
            )
        else:
            policy_evaluation = stale
            stale_extras = stale.output.extras or {}
            cache_latency = float(stale_extras.get("cache_maintenance_latency", 0.0))
            decode_latency = float(stale_extras.get("decode_latency", 0.0))
            action_latency = cache_latency + decode_latency
            action_flops = float(stale_extras.get("strategy_flops", 0.0))
        router_latency = float(condition.get("router_feature_latency", 0.0))
        records.append(
            _record(
                condition,
                selector="zero_probe_recompute_router",
                requested_budget_fraction=1.0 if trigger_recompute else 0.0,
                mask=empty,
                eligible=eligible,
                evaluation=policy_evaluation,
                stale_kl=stale.logits_kl,
                selection_metadata={
                    "selection_objective": str(
                        recompute_policy.get(
                            "objective", "zero_probe_direct_full_recompute"
                        )
                    ),
                    "selected_full_recompute_fallback": trigger_recompute,
                    "selected_stale_action": not trigger_recompute,
                    "router_action": (
                        "full_recompute" if trigger_recompute else "stale"
                    ),
                    "router_risk_feature": str(
                        recompute_policy.get("risk_feature", "")
                    ),
                    "router_risk_score": risk_score,
                    "router_risk_threshold": float(
                        recompute_policy.get("risk_threshold", math.inf)
                    ),
                    "router_trigger_quantile": float(
                        recompute_policy.get("trigger_quantile", 0.0)
                    ),
                    "router_calibration_conditions": int(
                        recompute_policy.get("calibration_conditions", 0)
                    ),
                    "router_calibration_trigger_rate": float(
                        recompute_policy.get("calibration_trigger_rate", 0.0)
                    ),
                    "router_calibration_weighted_kl_recovery": float(
                        recompute_policy.get(
                            "calibration_weighted_kl_recovery", 0.0
                        )
                    ),
                    "router_runtime_forward_count": 0,
                    "router_runtime_uses_full_reference": False,
                    "planner_probe_latency": 0.0,
                    "planner_probe_flops": 0.0,
                    "search_probe_count": 0,
                    "search_reference_token_evaluations": 0,
                    "router_decision_latency": router_latency,
                    "end_to_end_planner_latency": action_latency + router_latency,
                    "strategy_flops": action_flops,
                    "cache_maintenance_latency": cache_latency,
                    "decode_latency": decode_latency,
                    "sparse_ranker_path": str(sparse_ranker_path),
                },
            )
        )
    mask_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    desired_counts = sorted(
        {
            max(1, min(total_eligible, int(math.ceil(total_eligible * fraction))))
            for fraction in budget_fractions
        }
    )
    random_scores = np.random.default_rng(
        _condition_seed(condition)
    ).random(eligible.shape)
    layer_prefix_scores = np.zeros_like(kv_scores)
    for layer_index in range(layer_count):
        for block_index in range(block_count):
            layer_prefix_scores[layer_index, block_index] = -(
                layer_index * block_count + block_index
            )

    if compute_cache_surgery_oracles:
        selectors = {
            "random": random_scores,
            "kv_drift": kv_scores,
            "attention_weighted_kv_drift": attention_scores,
            "layer_prefix": layer_prefix_scores,
        }
        for selector, scores in selectors.items():
            for count in desired_counts:
                mask = _top_mask(scores, eligible, count)
                evaluation = evaluate(mask)
                records.append(
                    _record(
                        condition,
                        selector=selector,
                        requested_budget_fraction=count / total_eligible,
                        mask=mask,
                        eligible=eligible,
                        evaluation=evaluation,
                        stale_kl=stale.logits_kl,
                    )
                )
                mask_rows.extend(
                    _mask_rows(condition, selector, count / total_eligible, mask)
                )

    if sparse_scores is not None:
        direct_available = np.asarray(sparse_scores["available"], dtype=bool) & eligible
        direct_total = int(np.count_nonzero(direct_available))
        condition_feature_rows: list[dict[str, Any]] = []
        for layer_index, block_index in np.argwhere(direct_available):
            token_start = int(block_index) * block_size
            token_end = min(token_count, token_start + block_size)
            feature_row = {
                **condition,
                "layer": int(layer_index),
                "token_block": int(block_index),
                "token_start": token_start,
                "token_end": token_end,
                "token_center_fraction": (
                    (token_start + token_end) / 2.0 / max(token_count, 1)
                ),
                "token_length_fraction": (token_end - token_start) / max(token_count, 1),
                "layer_fraction": int(layer_index) / max(layer_count - 1, 1),
                "direct_available_cells": direct_total,
                "stale_attention_mass": float(
                    sparse_scores["stale_attention_mass"][layer_index, block_index]
                ),
                "input_weight_bound": float(
                    sparse_scores["input_weight_bound"][layer_index, block_index]
                ),
                "attention_input_bound": float(
                    sparse_scores["attention_input_bound"][layer_index, block_index]
                ),
                "predicted_delta_norm": float(
                    sparse_scores["predicted_delta_norm"][layer_index, block_index]
                ),
                "attention_predicted_delta": float(
                    sparse_scores["attention_predicted_delta"][layer_index, block_index]
                ),
            }
            for signed_name in (
                "signed_correction_norm",
                "signed_total_alignment",
                "signed_total_projection",
                "signed_first_residual_gain",
                "signed_cancellation_ratio",
            ):
                signed_values = sparse_scores.get(signed_name)
                if isinstance(signed_values, np.ndarray):
                    feature_row[signed_name] = float(
                        signed_values[layer_index, block_index]
                    )
            condition_feature_rows.append(feature_row)
            feature_rows.append(feature_row)
        sparse_selectors = {
            "sparse_attention_mass": "stale_attention_mass",
            "sparse_input_bound": "input_weight_bound",
            "sparse_attention_input_bound": "attention_input_bound",
            "sparse_predicted_delta_norm": "predicted_delta_norm",
            "sparse_attention_predicted_delta": "attention_predicted_delta",
        }
        if direct_total > 0:
            if not sparse_policy_only:
                for selector, score_name in sparse_selectors.items():
                    scores = np.asarray(sparse_scores[score_name], dtype=np.float64)
                    for count in desired_counts:
                        direct_count = min(count, direct_total)
                        mask = _top_mask(scores, direct_available, direct_count)
                        evaluation = evaluate_sparse(mask)
                        records.append(
                            _record(
                                condition,
                                selector=selector,
                                requested_budget_fraction=count / total_eligible,
                                mask=mask,
                                eligible=eligible,
                                evaluation=evaluation,
                                stale_kl=stale.logits_kl,
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(condition, selector, count / total_eligible, mask)
                        )
            if (
                not sparse_policy_only
                and isinstance(
                    sparse_scores.get("signed_correction_vectors"), np.ndarray
                )
            ):
                signed_available = (
                    np.asarray(
                        sparse_scores.get(
                            "signed_correction_available",
                            direct_available,
                        ),
                        dtype=bool,
                    )
                    & direct_available
                )
                signed_started = time.perf_counter()
                signed_order, signed_marginals, signed_initial_energy = (
                    _signed_residual_greedy_order(
                        np.asarray(
                            sparse_scores["signed_correction_vectors"],
                            dtype=np.float64,
                        ),
                        signed_available,
                        max_cells=int(np.count_nonzero(signed_available)),
                    )
                )
                signed_decision_latency = time.perf_counter() - signed_started
                for count in desired_counts:
                    direct_count = min(count, len(signed_order))
                    signed_mask = _mask_from_order(
                        direct_available.shape,
                        signed_order,
                        direct_count,
                    )
                    signed_evaluation = (
                        evaluate_sparse(signed_mask) if direct_count > 0 else stale
                    )
                    final_energy = signed_initial_energy - float(
                        np.sum(signed_marginals[:direct_count])
                    )
                    records.append(
                        _record(
                            condition,
                            selector="sparse_signed_residual_greedy",
                            requested_budget_fraction=direct_count / total_eligible,
                            mask=signed_mask,
                            eligible=eligible,
                            evaluation=signed_evaluation,
                            stale_kl=stale.logits_kl,
                            selection_metadata={
                                "selection_objective": (
                                    "signed_attention_value_residual_reduction"
                                ),
                                "signed_residual_initial_energy": (
                                    signed_initial_energy
                                ),
                                "signed_residual_final_energy_fraction": (
                                    max(final_energy, 0.0)
                                    / max(signed_initial_energy, 1e-30)
                                ),
                                "signed_residual_last_marginal": (
                                    signed_marginals[direct_count - 1]
                                    if direct_count > 0
                                    else 0.0
                                ),
                                "router_runtime_forward_count": 0,
                                "router_runtime_uses_kl": False,
                                "planner_probe_latency": 0.0,
                                "planner_probe_flops": 0.0,
                                "search_probe_count": 0,
                                "search_reference_token_evaluations": 0,
                                "joint_budget_selection": False,
                                "router_decision_latency": signed_decision_latency,
                            },
                        )
                    )
                    mask_rows.extend(
                        _mask_rows(
                            condition,
                            "sparse_signed_residual_greedy",
                            direct_count / total_eligible,
                            signed_mask,
                        )
                    )

                auto_thresholds = (
                    ("000", 0.0),
                    ("001", 0.01),
                    ("0025", 0.025),
                    ("005", 0.05),
                    ("010", 0.10),
                )
                for threshold_name, threshold_fraction in auto_thresholds:
                    auto_count, final_energy = _signed_residual_best_prefix(
                        signed_marginals,
                        signed_initial_energy,
                        cost_fraction=threshold_fraction,
                    )
                    auto_mask = _mask_from_order(
                        direct_available.shape,
                        signed_order,
                        auto_count,
                    )
                    auto_evaluation = (
                        evaluate_sparse(auto_mask) if auto_count > 0 else stale
                    )
                    auto_selector = (
                        f"sparse_signed_residual_auto_{threshold_name}"
                    )
                    records.append(
                        _record(
                            condition,
                            selector=auto_selector,
                            requested_budget_fraction=auto_count / total_eligible,
                            mask=auto_mask,
                            eligible=eligible,
                            evaluation=auto_evaluation,
                            stale_kl=stale.logits_kl,
                            selection_metadata={
                                "selection_objective": (
                                    "signed_attention_value_dynamic_"
                                    "residual_reduction"
                                ),
                                "signed_residual_cell_cost_fraction": (
                                    threshold_fraction
                                ),
                                "signed_residual_initial_energy": (
                                    signed_initial_energy
                                ),
                                "signed_residual_final_energy_fraction": (
                                    max(final_energy, 0.0)
                                    / max(signed_initial_energy, 1e-30)
                                ),
                                "signed_residual_last_marginal": (
                                    signed_marginals[auto_count - 1]
                                    if auto_count > 0
                                    else 0.0
                                ),
                                "router_runtime_forward_count": 0,
                                "router_runtime_uses_kl": False,
                                "planner_probe_latency": 0.0,
                                "planner_probe_flops": 0.0,
                                "search_probe_count": 0,
                                "search_reference_token_evaluations": 0,
                                "joint_budget_selection": True,
                                "router_decision_latency": signed_decision_latency,
                            },
                        )
                    )
                    mask_rows.extend(
                        _mask_rows(
                            condition,
                            auto_selector,
                            auto_count / total_eligible,
                            auto_mask,
                        )
                    )
            if sparse_ranker is not None and not sparse_policy_only:
                learned_vector, learned_default_count = score_block_features(
                    sparse_ranker,
                    update_target=str(condition["update_target"]),
                    feature_rows=condition_feature_rows,
                )
                learned_scores = np.full(eligible.shape, -math.inf, dtype=np.float64)
                for feature_row, score in zip(
                    condition_feature_rows, learned_vector, strict=True
                ):
                    learned_scores[
                        int(feature_row["layer"]), int(feature_row["token_block"])
                    ] = float(score)
                learned_counts = sorted(
                    {min(count, direct_total) for count in desired_counts}
                )
                for direct_count in learned_counts:
                    learned_mask = _top_mask(
                        learned_scores, direct_available, direct_count
                    )
                    learned_evaluation = evaluate_sparse(learned_mask)
                    learned_budget = direct_count / total_eligible
                    records.append(
                        _record(
                            condition,
                            selector="sparse_learned_ranker",
                            requested_budget_fraction=learned_budget,
                            mask=learned_mask,
                            eligible=eligible,
                            evaluation=learned_evaluation,
                            stale_kl=stale.logits_kl,
                            selection_metadata={
                                "selection_objective": "static_learned_ranker",
                                "search_probe_count": 0,
                                "search_reference_token_evaluations": 0,
                                "joint_budget_selection": False,
                                "sparse_ranker_path": str(sparse_ranker_path),
                                "ranker_default_count": learned_default_count,
                            },
                        )
                    )
                    mask_rows.extend(
                        _mask_rows(
                            condition,
                            "sparse_learned_ranker",
                            learned_budget,
                            learned_mask,
                        )
                    )

                default_count = max(0, min(learned_default_count, direct_total))
                if default_count == 0:
                    default_mask = np.zeros_like(direct_available)
                    default_evaluation = stale
                else:
                    default_mask = _top_mask(
                        learned_scores, direct_available, default_count
                    )
                    default_evaluation = evaluate_sparse(default_mask)
                default_budget = default_count / total_eligible
                records.append(
                    _record(
                        condition,
                        selector="sparse_learned_default_budget",
                        requested_budget_fraction=default_budget,
                        mask=default_mask,
                        eligible=eligible,
                        evaluation=default_evaluation,
                        stale_kl=stale.logits_kl,
                        selection_metadata={
                            "selection_objective": "static_learned_ranker",
                            "search_probe_count": 0,
                            "search_reference_token_evaluations": 0,
                            "joint_budget_selection": True,
                            "selected_stale_action": default_count == 0,
                            "sparse_ranker_path": str(sparse_ranker_path),
                            "ranker_default_count": learned_default_count,
                        },
                    )
                )
                mask_rows.extend(
                    _mask_rows(
                        condition,
                        "sparse_learned_default_budget",
                        default_budget,
                        default_mask,
                    )
                )

                models = sparse_ranker.get("models", {})
                target_model = models.get(str(condition["update_target"]))
                one_probe_policy = (
                    target_model.get("one_probe_policy")
                    if isinstance(target_model, dict)
                    else None
                )
                if isinstance(one_probe_policy, dict):
                    candidate_selector = str(
                        one_probe_policy.get("candidate_selector", "")
                    )
                    candidate_score_name = sparse_selectors.get(candidate_selector)
                    candidate_count = max(
                        0,
                        min(
                            int(one_probe_policy.get("candidate_count", 0)),
                            direct_total,
                        ),
                    )
                    nll_margin = float(
                        one_probe_policy.get("reference_nll_margin", 0.0)
                    )
                    if candidate_score_name is not None and candidate_count > 0:
                        candidate_scores = np.asarray(
                            sparse_scores[candidate_score_name], dtype=np.float64
                        )
                        candidate_mask = _top_mask(
                            candidate_scores,
                            direct_available,
                            candidate_count,
                        )
                        candidate_evaluation = evaluate_sparse(candidate_mask)
                        reference_token_id = int(
                            condition.get("reference_token_id", -1)
                        )
                        stale_nll, stale_entropy, stale_max_probability = (
                            _logit_selection_metrics(
                                stale.output.logits,
                                reference_token_id=reference_token_id,
                            )
                        )
                        (
                            candidate_nll,
                            candidate_entropy,
                            candidate_max_probability,
                        ) = _logit_selection_metrics(
                            candidate_evaluation.output.logits,
                            reference_token_id=reference_token_id,
                        )
                        nll_improvement = stale_nll - candidate_nll
                        accepted = bool(
                            np.isfinite(nll_improvement)
                            and nll_improvement > nll_margin
                        )
                        if accepted:
                            selected_mask = candidate_mask
                            selected_evaluation = candidate_evaluation
                            selected_count = candidate_count
                        else:
                            selected_mask = np.zeros_like(direct_available)
                            selected_evaluation = stale
                            selected_count = 0
                        candidate_extras = candidate_evaluation.output.extras or {}
                        stale_extras = stale.output.extras or {}
                        candidate_maintenance = float(
                            candidate_extras.get("cache_maintenance_latency", 0.0)
                        )
                        candidate_decode = float(
                            candidate_extras.get("decode_latency", 0.0)
                        )
                        candidate_latency = candidate_maintenance + candidate_decode
                        stale_decode = float(stale_extras.get("decode_latency", 0.0))
                        selected_budget = selected_count / total_eligible
                        records.append(
                            _record(
                                condition,
                                selector="sparse_one_probe_policy",
                                requested_budget_fraction=selected_budget,
                                mask=selected_mask,
                                eligible=eligible,
                                evaluation=selected_evaluation,
                                stale_kl=stale.logits_kl,
                                selection_metadata={
                                    "selection_objective": "reference_nll_stale_gate",
                                    "search_probe_count": 1,
                                    "search_reference_token_evaluations": 1,
                                    "joint_budget_selection": True,
                                    "selected_stale_action": not accepted,
                                    "safety_gate_passed": accepted,
                                    "selection_stale_margin": nll_margin,
                                    "selection_stale_score": stale_nll,
                                    "selection_raw_score": candidate_nll,
                                    "selection_objective_improvement_vs_stale": (
                                        nll_improvement
                                    ),
                                    "sparse_ranker_path": str(sparse_ranker_path),
                                    "candidate_selector": candidate_selector,
                                    "candidate_selected_cells": candidate_count,
                                    "candidate_reference_token_nll": candidate_nll,
                                    "candidate_output_entropy": candidate_entropy,
                                    "candidate_output_max_probability": (
                                        candidate_max_probability
                                    ),
                                    "stale_output_entropy": stale_entropy,
                                    "stale_output_max_probability": stale_max_probability,
                                    "candidate_logits_kl": (
                                        candidate_evaluation.logits_kl
                                    ),
                                    "candidate_cache_maintenance_latency": (
                                        candidate_maintenance
                                    ),
                                    "candidate_decode_latency": candidate_decode,
                                    "planner_probe_latency": candidate_latency,
                                    "end_to_end_planner_latency": (
                                        stale_decode + candidate_latency
                                    ),
                                    "candidate_strategy_flops": float(
                                        candidate_extras.get("strategy_flops", 0.0)
                                    ),
                                    "one_probe_calibration_harmful": int(
                                        one_probe_policy.get(
                                            "calibration_harmful", 0
                                        )
                                    ),
                                    "one_probe_calibration_mean_gain": float(
                                        one_probe_policy.get(
                                            "calibration_mean_gain", 0.0
                                        )
                                    ),
                                },
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(
                                condition,
                                "sparse_one_probe_policy",
                                selected_budget,
                                selected_mask,
                            )
                        )

                confidence_probe_policy = (
                    target_model.get("confidence_probe_policy")
                    if isinstance(target_model, dict)
                    else None
                )
                if isinstance(confidence_probe_policy, dict):
                    candidate_selector = str(
                        confidence_probe_policy.get("candidate_selector", "")
                    )
                    candidate_score_name = sparse_selectors.get(candidate_selector)
                    candidate_count = max(
                        0,
                        min(
                            int(confidence_probe_policy.get("candidate_count", 0)),
                            direct_total,
                        ),
                    )
                    confidence_margin = float(
                        confidence_probe_policy.get("confidence_margin", 0.0)
                    )
                    if candidate_score_name is not None and candidate_count > 0:
                        candidate_scores = np.asarray(
                            sparse_scores[candidate_score_name], dtype=np.float64
                        )
                        candidate_mask = _top_mask(
                            candidate_scores,
                            direct_available,
                            candidate_count,
                        )
                        candidate_evaluation = evaluate_sparse(candidate_mask)
                        reference_token_id = int(
                            condition.get("reference_token_id", -1)
                        )
                        stale_nll, stale_entropy, stale_max_probability = (
                            _logit_selection_metrics(
                                stale.output.logits,
                                reference_token_id=reference_token_id,
                            )
                        )
                        (
                            candidate_nll,
                            candidate_entropy,
                            candidate_max_probability,
                        ) = _logit_selection_metrics(
                            candidate_evaluation.output.logits,
                            reference_token_id=reference_token_id,
                        )
                        confidence_improvement = (
                            candidate_max_probability - stale_max_probability
                        )
                        accepted = bool(
                            np.isfinite(confidence_improvement)
                            and confidence_improvement > confidence_margin
                        )
                        if accepted:
                            selected_mask = candidate_mask
                            selected_evaluation = candidate_evaluation
                            selected_count = candidate_count
                        else:
                            selected_mask = np.zeros_like(direct_available)
                            selected_evaluation = stale
                            selected_count = 0
                        candidate_extras = candidate_evaluation.output.extras or {}
                        stale_extras = stale.output.extras or {}
                        candidate_maintenance = float(
                            candidate_extras.get("cache_maintenance_latency", 0.0)
                        )
                        candidate_decode = float(
                            candidate_extras.get("decode_latency", 0.0)
                        )
                        candidate_latency = candidate_maintenance + candidate_decode
                        stale_decode = float(stale_extras.get("decode_latency", 0.0))
                        selected_budget = selected_count / total_eligible
                        records.append(
                            _record(
                                condition,
                                selector="sparse_one_probe_confidence_gate",
                                requested_budget_fraction=selected_budget,
                                mask=selected_mask,
                                eligible=eligible,
                                evaluation=selected_evaluation,
                                stale_kl=stale.logits_kl,
                                selection_metadata={
                                    "selection_objective": "confidence_stale_gate",
                                    "search_probe_count": 1,
                                    "search_reference_token_evaluations": 0,
                                    "joint_budget_selection": True,
                                    "selected_stale_action": not accepted,
                                    "safety_gate_passed": accepted,
                                    "selection_stale_margin": confidence_margin,
                                    "selection_stale_score": stale_max_probability,
                                    "selection_raw_score": candidate_max_probability,
                                    "selection_objective_improvement_vs_stale": (
                                        confidence_improvement
                                    ),
                                    "sparse_ranker_path": str(sparse_ranker_path),
                                    "candidate_selector": candidate_selector,
                                    "candidate_selected_cells": candidate_count,
                                    "candidate_reference_token_nll": candidate_nll,
                                    "candidate_output_entropy": candidate_entropy,
                                    "candidate_output_max_probability": (
                                        candidate_max_probability
                                    ),
                                    "stale_reference_token_nll": stale_nll,
                                    "stale_output_entropy": stale_entropy,
                                    "stale_output_max_probability": (
                                        stale_max_probability
                                    ),
                                    "candidate_logits_kl": (
                                        candidate_evaluation.logits_kl
                                    ),
                                    "candidate_cache_maintenance_latency": (
                                        candidate_maintenance
                                    ),
                                    "candidate_decode_latency": candidate_decode,
                                    "planner_probe_latency": candidate_latency,
                                    "end_to_end_planner_latency": (
                                        stale_decode + candidate_latency
                                    ),
                                    "candidate_strategy_flops": float(
                                        candidate_extras.get("strategy_flops", 0.0)
                                    ),
                                    "confidence_probe_calibration_harmful": int(
                                        confidence_probe_policy.get(
                                            "calibration_harmful", 0
                                        )
                                    ),
                                    "confidence_probe_calibration_mean_gain": float(
                                        confidence_probe_policy.get(
                                            "calibration_mean_gain", 0.0
                                        )
                                    ),
                                },
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(
                                condition,
                                "sparse_one_probe_confidence_gate",
                                selected_budget,
                                selected_mask,
                            )
                        )

            if sparse_ranker is not None:
                target_model = sparse_ranker.get("models", {}).get(
                    str(condition["update_target"])
                )
                if isinstance(target_model, dict) and not sparse_policy_only:
                    reference_pool_policy = target_model.get(
                        "reference_candidate_pool_policy"
                    )
                    if isinstance(reference_pool_policy, dict):
                        raw_candidates = reference_pool_policy.get("candidates", [])
                        reference_margin = float(
                            reference_pool_policy.get("reference_nll_margin", 0.0)
                        )
                        reference_token_id = int(
                            condition.get("reference_token_id", -1)
                        )
                        stale_nll, stale_entropy, stale_max_probability = (
                            _logit_selection_metrics(
                                stale.output.logits,
                                reference_token_id=reference_token_id,
                            )
                        )
                        selected_mask = np.zeros_like(direct_available)
                        selected_evaluation = stale
                        selected_nll = stale_nll
                        selected_count = 0
                        evaluated_masks: set[bytes] = set()
                        candidate_metadata: list[dict[str, Any]] = []
                        planner_probe_latency = 0.0
                        planner_probe_flops = 0.0
                        for candidate in raw_candidates:
                            if not isinstance(candidate, dict):
                                continue
                            candidate_selector = str(
                                candidate.get("candidate_selector", "")
                            )
                            candidate_score_name = sparse_selectors.get(
                                candidate_selector
                            )
                            candidate_count = max(
                                0,
                                min(
                                    int(candidate.get("candidate_count", 0)),
                                    direct_total,
                                ),
                            )
                            if candidate_score_name is None or candidate_count <= 0:
                                continue
                            candidate_scores = np.asarray(
                                sparse_scores[candidate_score_name], dtype=np.float64
                            )
                            candidate_mask = _top_mask(
                                candidate_scores,
                                direct_available,
                                candidate_count,
                            )
                            mask_key = np.ascontiguousarray(
                                candidate_mask, dtype=np.uint8
                            ).tobytes()
                            if mask_key in evaluated_masks:
                                continue
                            evaluated_masks.add(mask_key)
                            candidate_evaluation = evaluate_sparse(candidate_mask)
                            (
                                candidate_nll,
                                candidate_entropy,
                                candidate_max_probability,
                            ) = _logit_selection_metrics(
                                candidate_evaluation.output.logits,
                                reference_token_id=reference_token_id,
                            )
                            candidate_extras = candidate_evaluation.output.extras or {}
                            candidate_latency = float(
                                candidate_extras.get("cache_maintenance_latency", 0.0)
                            ) + float(candidate_extras.get("decode_latency", 0.0))
                            candidate_flops = float(
                                candidate_extras.get("strategy_flops", 0.0)
                            )
                            planner_probe_latency += candidate_latency
                            planner_probe_flops += candidate_flops
                            candidate_metadata.append(
                                {
                                    "selector": candidate_selector,
                                    "selected_cells": candidate_count,
                                    "reference_token_nll": candidate_nll,
                                    "output_entropy": candidate_entropy,
                                    "output_max_probability": candidate_max_probability,
                                    "logits_kl": candidate_evaluation.logits_kl,
                                    "latency": candidate_latency,
                                    "strategy_flops": candidate_flops,
                                    **{
                                        key: float(value)
                                        for key, value in candidate_extras.items()
                                        if key.startswith(
                                            (
                                                "target_attention_",
                                                "final_attention_",
                                                "downstream_attention_",
                                            )
                                        )
                                        and isinstance(value, int | float)
                                    },
                                }
                            )
                            if candidate_nll < selected_nll - reference_margin - 1e-15:
                                selected_mask = candidate_mask
                                selected_evaluation = candidate_evaluation
                                selected_nll = candidate_nll
                                selected_count = candidate_count
                        selected_budget = selected_count / total_eligible
                        stale_extras = stale.output.extras or {}
                        records.append(
                            _record(
                                condition,
                                selector="sparse_reference_pool_policy",
                                requested_budget_fraction=selected_budget,
                                mask=selected_mask,
                                eligible=eligible,
                                evaluation=selected_evaluation,
                                stale_kl=stale.logits_kl,
                                selection_metadata={
                                    "selection_objective": (
                                        "distilled_reference_candidate_pool"
                                    ),
                                    "search_probe_count": len(evaluated_masks),
                                    "search_reference_token_evaluations": len(
                                        evaluated_masks
                                    ),
                                    "joint_budget_selection": True,
                                    "selected_stale_action": selected_count == 0,
                                    "safety_gate_passed": selected_count > 0,
                                    "selection_stale_margin": reference_margin,
                                    "selection_stale_score": stale_nll,
                                    "selection_raw_score": selected_nll,
                                    "selection_objective_improvement_vs_stale": (
                                        stale_nll - selected_nll
                                    ),
                                    "stale_output_entropy": stale_entropy,
                                    "stale_output_max_probability": (
                                        stale_max_probability
                                    ),
                                    "sparse_ranker_path": str(sparse_ranker_path),
                                    "reference_candidate_pool": json.dumps(
                                        candidate_metadata, sort_keys=True
                                    ),
                                    "planner_probe_latency": planner_probe_latency,
                                    "end_to_end_planner_latency": (
                                        float(
                                            stale_extras.get("decode_latency", 0.0)
                                        )
                                        + planner_probe_latency
                                    ),
                                    "planner_probe_flops": planner_probe_flops,
                                    "reference_pool_calibration_harmful": int(
                                        reference_pool_policy.get(
                                            "calibration_harmful", 0
                                        )
                                    ),
                                    "reference_pool_calibration_weighted_recovery": (
                                        float(
                                            reference_pool_policy.get(
                                                "calibration_weighted_recovery", 0.0
                                            )
                                        )
                                    ),
                                },
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(
                                condition,
                                "sparse_reference_pool_policy",
                                selected_budget,
                                selected_mask,
                            )
                        )

                if (
                    isinstance(target_model, dict)
                    and sparse_policy_variant == "committed"
                ):
                    committed_policy_name = (
                        "baseline_committed_candidate_router_policy"
                    )
                    committed_policy = target_model.get(committed_policy_name)
                    if isinstance(committed_policy, dict):
                        _, committed_stale_entropy, _ = _logit_selection_metrics(
                            stale.output.logits,
                            reference_token_id=int(
                                condition.get("reference_token_id", -1)
                            ),
                        )
                        candidate, predicted_gains, lower_bounds = (
                            route_committed_candidate(
                                sparse_ranker,
                                update_target=str(condition["update_target"]),
                                feature_rows=condition_feature_rows,
                                stale_output_entropy=committed_stale_entropy,
                                policy_name=committed_policy_name,
                            )
                        )
                        selected_mask = np.zeros_like(direct_available)
                        selected_evaluation = stale
                        selected_count = 0
                        selected_selector = "stale"
                        if candidate is not None:
                            candidate_selector = str(
                                candidate.get("candidate_selector", "")
                            )
                            candidate_score_name = sparse_selectors.get(
                                candidate_selector
                            )
                            candidate_count = max(
                                0,
                                min(
                                    int(candidate.get("candidate_count", 0)),
                                    direct_total,
                                ),
                            )
                            if (
                                candidate_score_name is not None
                                and candidate_count > 0
                            ):
                                candidate_scores = np.asarray(
                                    sparse_scores[candidate_score_name],
                                    dtype=np.float64,
                                )
                                selected_mask = _top_mask(
                                    candidate_scores,
                                    direct_available,
                                    candidate_count,
                                )
                                selected_evaluation = evaluate_sparse(selected_mask)
                                selected_count = candidate_count
                                selected_selector = candidate_selector
                        accepted = selected_count > 0
                        selected_budget = selected_count / total_eligible
                        selected_extras = selected_evaluation.output.extras or {}
                        committed_latency = float(
                            selected_extras.get("cache_maintenance_latency", 0.0)
                        ) + float(selected_extras.get("decode_latency", 0.0))
                        best_lower_bound = float(np.max(lower_bounds))
                        records.append(
                            _record(
                                condition,
                                selector="sparse_committed_router_policy",
                                requested_budget_fraction=selected_budget,
                                mask=selected_mask,
                                eligible=eligible,
                                evaluation=selected_evaluation,
                                stale_kl=stale.logits_kl,
                                selection_metadata={
                                    "selection_objective": (
                                        "conservative_absolute_kl_gain_lower_bound"
                                    ),
                                    "search_probe_count": 0,
                                    "search_reference_token_evaluations": 0,
                                    "joint_budget_selection": True,
                                    "selected_stale_action": not accepted,
                                    "safety_gate_passed": accepted,
                                    "candidate_selector": selected_selector,
                                    "candidate_selected_cells": selected_count,
                                    "candidate_logits_kl": (
                                        selected_evaluation.logits_kl
                                    ),
                                    "planner_probe_latency": 0.0,
                                    "end_to_end_planner_latency": committed_latency,
                                    "planner_probe_flops": 0.0,
                                    "router_scores": json.dumps(
                                        predicted_gains.tolist()
                                    ),
                                    "router_lower_bounds": json.dumps(
                                        lower_bounds.tolist()
                                    ),
                                    "router_selected_lower_bound": best_lower_bound,
                                    "router_minimum_lower_bound": float(
                                        committed_policy.get(
                                            "minimum_lower_bound", 0.0
                                        )
                                    ),
                                    "router_overprediction_quantile": float(
                                        committed_policy.get(
                                            "overprediction_quantile", 1.0
                                        )
                                    ),
                                    "router_objective": str(
                                        committed_policy.get("objective", "")
                                    ),
                                    "router_ridge": float(
                                        committed_policy.get("ridge", 0.0)
                                    ),
                                    "router_runtime_forward_count": 0,
                                    "router_runtime_uses_kl": False,
                                    "router_runtime_guard": json.dumps(
                                        committed_policy.get("runtime_guard", {}),
                                        sort_keys=True,
                                    ),
                                    "router_calibration_material_harmful": int(
                                        committed_policy.get(
                                            "material_harmful", 0
                                        )
                                    ),
                                    "router_calibration_material_recovery": float(
                                        committed_policy.get(
                                            "material_weighted_recovery", 0.0
                                        )
                                    ),
                                    "router_calibration_accept_rate": float(
                                        committed_policy.get("accept_rate", 0.0)
                                    ),
                                    "sparse_ranker_path": str(sparse_ranker_path),
                                },
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(
                                condition,
                                "sparse_committed_router_policy",
                                selected_budget,
                                selected_mask,
                            )
                        )

                if (
                    isinstance(target_model, dict)
                    and sparse_policy_variant != "committed"
                ):
                    baseline_reference_variant = (
                        sparse_policy_variant == "baseline_reference"
                    )
                    router_policy_name = (
                        "baseline_reference_candidate_router_policy"
                        if baseline_reference_variant
                        else "reference_candidate_router_policy"
                    )
                    router_policy = target_model.get(router_policy_name)
                    if isinstance(router_policy, dict):
                        _, router_scores = route_reference_candidate(
                            sparse_ranker,
                            update_target=str(condition["update_target"]),
                            feature_rows=condition_feature_rows,
                            policy_name=router_policy_name,
                        )
                        raw_candidates = router_policy.get("candidates", [])
                        candidate_order = np.argsort(-router_scores)
                        first_index = int(candidate_order[0])
                        second_index = int(candidate_order[1])
                        score_margin = float(
                            router_scores[first_index] - router_scores[second_index]
                        )
                        second_probe_margin = (
                            0.0
                            if baseline_reference_variant
                            else float(
                                router_policy.get("second_probe_score_margin", 0.0)
                            )
                        )
                        probe_indices = [first_index]
                        if score_margin < second_probe_margin:
                            probe_indices.append(second_index)
                        reference_token_id = int(
                            condition.get("reference_token_id", -1)
                        )
                        gate_output = (
                            baseline if baseline_reference_variant else stale.output
                        )
                        gate_nll, gate_entropy, gate_max_probability = (
                            _logit_selection_metrics(
                                gate_output.logits,
                                reference_token_id=reference_token_id,
                            )
                        )
                        reference_margin = float(
                            router_policy.get("reference_nll_margin", 0.0)
                        )
                        selected_mask = np.zeros_like(direct_available)
                        selected_evaluation = (
                            _Evaluation(
                                output=full,
                                logits_kl=0.0,
                                top1_agreement=1.0,
                            )
                            if baseline_reference_variant
                            else stale
                        )
                        selected_nll = gate_nll
                        selected_count = 0
                        selected_selector = (
                            "full_recompute"
                            if baseline_reference_variant
                            else "stale"
                        )
                        selected_entropy = gate_entropy
                        selected_max_probability = gate_max_probability
                        planner_probe_latency = 0.0
                        planner_probe_flops = 0.0
                        router_evaluated_masks: set[bytes] = set()
                        probe_metadata: list[dict[str, Any]] = []
                        for candidate_index in probe_indices:
                            candidate = raw_candidates[candidate_index]
                            if not isinstance(candidate, dict):
                                continue
                            candidate_selector = str(
                                candidate.get("candidate_selector", "")
                            )
                            candidate_score_name = sparse_selectors.get(
                                candidate_selector
                            )
                            candidate_count = max(
                                0,
                                min(
                                    int(candidate.get("candidate_count", 0)),
                                    direct_total,
                                ),
                            )
                            if candidate_score_name is None or candidate_count <= 0:
                                continue
                            candidate_scores = np.asarray(
                                sparse_scores[candidate_score_name], dtype=np.float64
                            )
                            candidate_mask = _top_mask(
                                candidate_scores,
                                direct_available,
                                candidate_count,
                            )
                            mask_key = np.ascontiguousarray(
                                candidate_mask, dtype=np.uint8
                            ).tobytes()
                            if mask_key in router_evaluated_masks:
                                continue
                            router_evaluated_masks.add(mask_key)
                            candidate_evaluation = evaluate_sparse(candidate_mask)
                            (
                                candidate_nll,
                                candidate_entropy,
                                candidate_max_probability,
                            ) = _logit_selection_metrics(
                                candidate_evaluation.output.logits,
                                reference_token_id=reference_token_id,
                            )
                            candidate_extras = candidate_evaluation.output.extras or {}
                            candidate_maintenance = float(
                                candidate_extras.get(
                                    "cache_maintenance_latency", 0.0
                                )
                            )
                            candidate_decode = float(
                                candidate_extras.get("decode_latency", 0.0)
                            )
                            candidate_latency = (
                                candidate_maintenance + candidate_decode
                            )
                            candidate_flops = float(
                                candidate_extras.get("strategy_flops", 0.0)
                            )
                            planner_probe_latency += candidate_latency
                            planner_probe_flops += candidate_flops
                            probe_metadata.append(
                                {
                                    "candidate_index": candidate_index,
                                    "selector": candidate_selector,
                                    "selected_cells": candidate_count,
                                    "router_score": float(
                                        router_scores[candidate_index]
                                    ),
                                    "reference_token_nll": candidate_nll,
                                    "output_entropy": candidate_entropy,
                                    "output_max_probability": (
                                        candidate_max_probability
                                    ),
                                    "logits_kl": candidate_evaluation.logits_kl,
                                    "latency": candidate_latency,
                                    "strategy_flops": candidate_flops,
                                    **{
                                        key: float(value)
                                        for key, value in candidate_extras.items()
                                        if key.startswith(
                                            (
                                                "target_attention_",
                                                "final_attention_",
                                                "downstream_attention_",
                                            )
                                        )
                                        and isinstance(value, int | float)
                                    },
                                }
                            )
                            if (
                                candidate_nll
                                < selected_nll - reference_margin - 1e-15
                            ):
                                selected_mask = candidate_mask
                                selected_evaluation = candidate_evaluation
                                selected_nll = candidate_nll
                                selected_count = candidate_count
                                selected_selector = candidate_selector
                                selected_entropy = candidate_entropy
                                selected_max_probability = (
                                    candidate_max_probability
                                )
                        accepted = selected_count > 0
                        nll_improvement = gate_nll - selected_nll
                        stale_extras = stale.output.extras or {}
                        selected_budget = selected_count / total_eligible
                        records.append(
                            _record(
                                condition,
                                selector=(
                                    "sparse_baseline_reference_router_policy"
                                    if baseline_reference_variant
                                    else "sparse_reference_router_policy"
                                ),
                                requested_budget_fraction=selected_budget,
                                mask=selected_mask,
                                eligible=eligible,
                                evaluation=selected_evaluation,
                                stale_kl=stale.logits_kl,
                                selection_metadata={
                                    "selection_objective": (
                                        "distilled_baseline_reference_router"
                                        if baseline_reference_variant
                                        else "distilled_adaptive_reference_router"
                                    ),
                                    "search_probe_count": len(router_evaluated_masks),
                                    "search_reference_token_evaluations": len(
                                        router_evaluated_masks
                                    ),
                                    "joint_budget_selection": True,
                                    "selected_stale_action": not accepted,
                                    "safety_gate_passed": accepted,
                                    "selection_stale_margin": reference_margin,
                                    "selection_stale_score": gate_nll,
                                    "selection_raw_score": selected_nll,
                                    "selection_objective_improvement_vs_stale": (
                                        nll_improvement
                                    ),
                                    "candidate_selector": selected_selector,
                                    "candidate_selected_cells": selected_count,
                                    "candidate_reference_token_nll": selected_nll,
                                    "candidate_output_entropy": selected_entropy,
                                    "candidate_output_max_probability": (
                                        selected_max_probability
                                    ),
                                    "stale_output_entropy": gate_entropy,
                                    "stale_output_max_probability": (
                                        gate_max_probability
                                    ),
                                    "reference_gate_source": (
                                        "baseline_old_parameters"
                                        if baseline_reference_variant
                                        else "stale_current_parameters"
                                    ),
                                    "selected_full_recompute_fallback": bool(
                                        baseline_reference_variant and not accepted
                                    ),
                                    "candidate_logits_kl": (
                                        selected_evaluation.logits_kl
                                    ),
                                    "planner_probe_latency": planner_probe_latency,
                                    "end_to_end_planner_latency": (
                                        planner_probe_latency
                                        + (
                                            float(
                                                condition.get(
                                                    "full_recompute_strategy_latency",
                                                    0.0,
                                                )
                                            )
                                            if baseline_reference_variant and not accepted
                                            else (
                                                0.0
                                                if baseline_reference_variant
                                                else float(
                                                    stale_extras.get(
                                                        "decode_latency", 0.0
                                                    )
                                                )
                                            )
                                        )
                                    ),
                                    "planner_probe_flops": planner_probe_flops,
                                    "router_scores": json.dumps(
                                        router_scores.tolist()
                                    ),
                                    "router_score_margin": score_margin,
                                    "router_second_probe_score_margin": (
                                        second_probe_margin
                                    ),
                                    "router_second_probe_triggered": (
                                        len(probe_indices) > 1
                                    ),
                                    "router_probe_candidates": json.dumps(
                                        probe_metadata, sort_keys=True
                                    ),
                                    "router_objective": str(
                                        router_policy.get("objective", "")
                                    ),
                                    "router_ridge": float(
                                        router_policy.get("ridge", 0.0)
                                    ),
                                    "router_calibration_material_harmful": int(
                                        router_policy.get("material_harmful", 0)
                                    ),
                                    "router_calibration_material_recovery": float(
                                        router_policy.get(
                                            "material_weighted_recovery", 0.0
                                        )
                                    ),
                                    "router_second_probe_calibration_harmful": int(
                                        router_policy.get(
                                            "second_probe_material_harmful", 0
                                        )
                                    ),
                                    "router_second_probe_calibration_recovery": float(
                                        router_policy.get(
                                            "second_probe_material_weighted_recovery",
                                            0.0,
                                        )
                                    ),
                                    "router_second_probe_calibration_average_probes": float(
                                        router_policy.get(
                                            "second_probe_average_probes", 1.0
                                        )
                                    ),
                                    "sparse_ranker_path": str(sparse_ranker_path),
                                },
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(
                                condition,
                                (
                                    "sparse_baseline_reference_router_policy"
                                    if baseline_reference_variant
                                    else "sparse_reference_router_policy"
                                ),
                                selected_budget,
                                selected_mask,
                            )
                        )

            if not sparse_policy_only:
                all_direct = direct_available.copy()
                all_direct_evaluation = evaluate_sparse(all_direct)
                records.append(
                    _record(
                        condition,
                        selector="sparse_all_direct",
                        requested_budget_fraction=direct_total / total_eligible,
                        mask=all_direct,
                        eligible=eligible,
                        evaluation=all_direct_evaluation,
                        stale_kl=stale.logits_kl,
                    )
                )
                mask_rows.extend(
                    _mask_rows(
                        condition,
                        "sparse_all_direct",
                        direct_total / total_eligible,
                        all_direct,
                    )
                )
            if compute_structured_sparse_search and not sparse_policy_only:
                direct_indices = [tuple(index) for index in np.argwhere(direct_available)]
                search_counts = sorted({min(count, direct_total) for count in desired_counts})
                reference_token_id = int(condition.get("reference_token_id", -1))
                reference_objectives = [("reference", "reference_nll")]
                for length in effective_reference_lengths:
                    if length <= 1:
                        continue
                    key = f"reference_token_nll_{length}"
                    if stale.output.extras and key in stale.output.extras:
                        reference_objectives.append((f"reference{length}", f"reference_nll_{length}"))

                if direct_total <= direct_oracle_max_blocks:
                    for direct_count in search_counts:
                        best_splice_mask: np.ndarray | None = None
                        best_splice_eval: _Evaluation | None = None
                        best_sparse_mask: np.ndarray | None = None
                        best_sparse_eval: _Evaluation | None = None
                        best_reference_mask: np.ndarray | None = None
                        best_reference_eval: _Evaluation | None = None
                        best_reference_nll = math.inf
                        best_sequence: dict[str, tuple[float, np.ndarray, _Evaluation]] = {}
                        best_sequence_kl: dict[str, tuple[float, np.ndarray, _Evaluation]] = {}
                        best_confidence_mask: np.ndarray | None = None
                        best_confidence_eval: _Evaluation | None = None
                        best_confidence = -math.inf
                        for chosen in combinations(direct_indices, direct_count):
                            mask = np.zeros_like(direct_available)
                            for index in chosen:
                                mask[index] = True
                            if compute_cache_surgery_oracles:
                                splice_eval = evaluate(mask)
                                if (
                                    best_splice_eval is None
                                    or splice_eval.logits_kl
                                    < best_splice_eval.logits_kl - 1e-15
                                ):
                                    best_splice_mask = mask.copy()
                                    best_splice_eval = splice_eval
                            sparse_eval = probe_sparse(mask)
                            if (
                                best_sparse_eval is None
                                or sparse_eval.logits_kl < best_sparse_eval.logits_kl - 1e-15
                            ):
                                best_sparse_mask = mask.copy()
                                best_sparse_eval = sparse_eval
                            reference_nll, _, max_probability = _logit_selection_metrics(
                                sparse_eval.output.logits,
                                reference_token_id=reference_token_id,
                            )
                            if np.isfinite(reference_nll) and reference_nll < best_reference_nll:
                                best_reference_nll = reference_nll
                                best_reference_mask = mask.copy()
                                best_reference_eval = sparse_eval
                            for label, objective in reference_objectives[1:]:
                                score = _sparse_objective_score(
                                    sparse_eval,
                                    reference_token_id=reference_token_id,
                                    objective=objective,
                                )
                                previous = best_sequence.get(label)
                                if np.isfinite(score) and (previous is None or score < previous[0]):
                                    best_sequence[label] = (score, mask.copy(), sparse_eval)
                                length = int(objective.rsplit("_", 1)[1])
                                sequence_kl = float(
                                    (sparse_eval.output.extras or {}).get(
                                        f"reference_token_kl_{length}",
                                        math.nan,
                                    )
                                )
                                previous_kl = best_sequence_kl.get(label)
                                if np.isfinite(sequence_kl) and (
                                    previous_kl is None or sequence_kl < previous_kl[0]
                                ):
                                    best_sequence_kl[label] = (
                                        sequence_kl,
                                        mask.copy(),
                                        sparse_eval,
                                    )
                            if max_probability > best_confidence:
                                best_confidence = max_probability
                                best_confidence_mask = mask.copy()
                                best_confidence_eval = sparse_eval
                        if best_sparse_mask is None or best_sparse_eval is None:
                            continue
                        budget = direct_count / total_eligible
                        exhaustive_rows = [
                            ("sparse_delta_oracle", best_sparse_mask, best_sparse_eval),
                        ]
                        if best_splice_mask is not None and best_splice_eval is not None:
                            exhaustive_rows.insert(
                                0,
                                ("direct_splice_oracle", best_splice_mask, best_splice_eval),
                            )
                        if best_reference_mask is not None and best_reference_eval is not None:
                            exhaustive_rows.append(
                                (
                                    "sparse_reference_objective_oracle",
                                    best_reference_mask,
                                    best_reference_eval,
                                )
                            )
                        if best_confidence_mask is not None and best_confidence_eval is not None:
                            exhaustive_rows.append(
                                (
                                    "sparse_confidence_objective_oracle",
                                    best_confidence_mask,
                                    best_confidence_eval,
                                )
                            )
                        for label, (_, sequence_mask, sequence_eval) in best_sequence.items():
                            exhaustive_rows.append(
                                (
                                    f"sparse_{label}_objective_oracle",
                                    sequence_mask,
                                    sequence_eval,
                                )
                            )
                        for label, (_, sequence_mask, sequence_eval) in best_sequence_kl.items():
                            length_label = label.removeprefix("reference")
                            exhaustive_rows.append(
                                (
                                    f"sparse_sequence{length_label}_delta_oracle",
                                    sequence_mask,
                                    sequence_eval,
                                )
                            )
                        exhaustive_probe_count = math.comb(direct_total, direct_count)
                        for selector, mask, evaluation in exhaustive_rows:
                            records.append(
                                _record(
                                    condition,
                                    selector=selector,
                                    requested_budget_fraction=budget,
                                    mask=mask,
                                    eligible=eligible,
                                    evaluation=evaluation,
                                    stale_kl=stale.logits_kl,
                                    selection_metadata={
                                        "search_probe_count": exhaustive_probe_count,
                                        "selection_objective": (
                                            "logits_kl"
                                            if selector in {"direct_splice_oracle", "sparse_delta_oracle"}
                                            else (
                                                f"reference_sequence_kl_{selector.removeprefix('sparse_sequence').removesuffix('_delta_oracle')}"
                                                if selector.startswith("sparse_sequence")
                                                else (
                                                    "reference_nll"
                                                    if selector.startswith("sparse_reference")
                                                    else "confidence"
                                                )
                                            )
                                        ),
                                        "joint_budget_selection": False,
                                    },
                                )
                            )
                            mask_rows.extend(_mask_rows(condition, selector, budget, mask))

                max_direct_count = max(search_counts, default=0)

                def to_search_points(
                    path_by_count: dict[int, tuple[np.ndarray, _Evaluation]],
                    *,
                    objective: str,
                ) -> dict[int, _SearchPoint]:
                    points: dict[int, _SearchPoint] = {}
                    for count, (mask, evaluation) in path_by_count.items():
                        points[count] = _SearchPoint(
                            mask=mask.copy(),
                            evaluation=evaluation,
                            score=_sparse_objective_score(
                                evaluation,
                                reference_token_id=reference_token_id,
                                objective=objective,
                            ),
                            probe_count=sum(direct_total - step for step in range(count)),
                        )
                    return points

                def objective_probe_length(objective: str) -> int:
                    if objective.startswith("reference_nll_"):
                        return int(objective.rsplit("_", 1)[1])
                    return 1

                fixed_paths: dict[str, tuple[str, dict[int, _SearchPoint]]] = {}
                reference_path_groups: dict[
                    str, tuple[str, dict[str, dict[int, _SearchPoint]]]
                ] = {}
                for label, objective in reference_objectives:
                    selector_prefix = f"sparse_{label}"
                    greedy_raw = _greedy_sparse_objective_masks(
                        evaluate=evaluate_sparse,
                        candidate_mask=direct_available,
                        max_cells=max_direct_count,
                        reference_token_id=reference_token_id,
                        objective=objective,
                    )
                    greedy_points = to_search_points(greedy_raw, objective=objective)
                    paths: dict[str, dict[int, _SearchPoint]] = {
                        f"{selector_prefix}_greedy": greedy_points,
                    }
                    for beam_width in sparse_beam_widths:
                        paths[f"{selector_prefix}_beam{beam_width}"] = (
                            _beam_sparse_objective_masks(
                                evaluate=evaluate_sparse,
                                candidate_mask=direct_available,
                                max_cells=max_direct_count,
                                reference_token_id=reference_token_id,
                                objective=objective,
                                beam_width=beam_width,
                            )
                        )
                    if sparse_swap_rounds > 0:
                        paths[f"{selector_prefix}_swap"] = (
                            _swap_refine_sparse_objective_masks(
                                evaluate=evaluate_sparse,
                                greedy_path=greedy_raw,
                                candidate_mask=direct_available,
                                reference_token_id=reference_token_id,
                                objective=objective,
                                max_rounds=sparse_swap_rounds,
                            )
                        )
                    reference_path_groups[label] = (objective, paths)
                    fixed_paths.update(
                        {
                            selector: (objective, path_by_count)
                            for selector, path_by_count in paths.items()
                        }
                    )

                confidence_greedy_raw = _greedy_sparse_objective_masks(
                    evaluate=evaluate_sparse,
                    candidate_mask=direct_available,
                    max_cells=max_direct_count,
                    reference_token_id=reference_token_id,
                    objective="confidence",
                )
                fixed_paths["sparse_confidence_greedy"] = (
                    "confidence",
                    to_search_points(confidence_greedy_raw, objective="confidence"),
                )

                for selector, (objective, path_by_count) in fixed_paths.items():
                    probe_length = objective_probe_length(objective)
                    for direct_count in search_counts:
                        point = path_by_count.get(direct_count)
                        if point is None:
                            continue
                        budget = direct_count / total_eligible
                        records.append(
                            _record(
                                condition,
                                selector=selector,
                                requested_budget_fraction=budget,
                                mask=point.mask,
                                eligible=eligible,
                                evaluation=point.evaluation,
                                stale_kl=stale.logits_kl,
                                selection_metadata={
                                    "search_probe_count": point.probe_count,
                                    "search_reference_token_evaluations": (
                                        point.probe_count * probe_length
                                        if objective.startswith("reference_nll")
                                        else 0
                                    ),
                                    "selection_objective": objective,
                                    "selection_score": point.score,
                                    "reference_probe_length": probe_length,
                                    "joint_budget_selection": False,
                                    "search_beam_width": (
                                        int(selector.rsplit("beam", 1)[1])
                                        if "beam" in selector
                                        else 1
                                    ),
                                    "search_swap_rounds": (
                                        sparse_swap_rounds if selector.endswith("swap") else 0
                                    ),
                                },
                            )
                        )
                        mask_rows.extend(_mask_rows(condition, selector, budget, point.mask))

                for _, (objective, paths) in reference_path_groups.items():
                    probe_length = objective_probe_length(objective)
                    for selector, path_by_count in paths.items():
                        stale_objective_score = _sparse_objective_score(
                            stale,
                            reference_token_id=reference_token_id,
                            objective=objective,
                        )
                        for cost_penalty in sparse_cost_penalties:
                            for stale_margin in sparse_stale_margins:
                                joint = _joint_sparse_search_point(
                                    stale=stale,
                                    path=path_by_count,
                                    reference_token_id=reference_token_id,
                                    objective=objective,
                                    direct_total=direct_total,
                                    cost_penalty=cost_penalty,
                                    stale_margin=stale_margin,
                                )
                                selected_count = int(np.count_nonzero(joint.mask))
                                penalty_label = f"{cost_penalty:g}".replace(".", "p")
                                margin_label = f"{stale_margin:g}".replace(".", "p")
                                joint_selector = f"{selector}_joint_p{penalty_label}"
                                if stale_margin > 0.0:
                                    joint_selector += f"_m{margin_label}"
                                budget = selected_count / total_eligible
                                raw_selected_score = _sparse_objective_score(
                                    joint.evaluation,
                                    reference_token_id=reference_token_id,
                                    objective=objective,
                                )
                                objective_improvement = stale_objective_score - raw_selected_score
                                records.append(
                                    _record(
                                        condition,
                                        selector=joint_selector,
                                        requested_budget_fraction=budget,
                                        mask=joint.mask,
                                        eligible=eligible,
                                        evaluation=joint.evaluation,
                                        stale_kl=stale.logits_kl,
                                        selection_metadata={
                                            "search_probe_count": joint.probe_count,
                                            "search_reference_token_evaluations": (
                                                joint.probe_count * probe_length
                                            ),
                                            "selection_objective": objective,
                                            "selection_score": joint.score,
                                            "selection_raw_score": raw_selected_score,
                                            "selection_stale_score": stale_objective_score,
                                            "selection_objective_improvement_vs_stale": (
                                                objective_improvement
                                            ),
                                            "selection_cost_penalty": cost_penalty,
                                            "selection_stale_margin": stale_margin,
                                            "reference_probe_length": probe_length,
                                            "joint_budget_selection": True,
                                            "selected_stale_action": selected_count == 0,
                                            "safety_gate_passed": (
                                                selected_count > 0
                                                and objective_improvement >= stale_margin - 1e-15
                                            ),
                                            "search_beam_width": (
                                                int(selector.rsplit("beam", 1)[1])
                                                if "beam" in selector
                                                else 1
                                            ),
                                            "search_swap_rounds": (
                                                sparse_swap_rounds if selector.endswith("swap") else 0
                                            ),
                                        },
                                    )
                                )
                                mask_rows.extend(
                                    _mask_rows(condition, joint_selector, budget, joint.mask)
                                )

    frontier_rows: list[dict[str, Any]] = []
    if compute_cache_surgery_oracles:
        greedy_limit = min(max(desired_counts), oracle_max_cells, total_eligible)
        candidate_mask = _candidate_pool(
            kv_scores,
            attention_scores,
            eligible,
            limit=max(oracle_candidate_limit, greedy_limit),
        )
        greedy_masks = _greedy_oracle_masks(
            evaluate=evaluate,
            candidate_mask=candidate_mask,
            max_cells=greedy_limit,
        )
        for count in desired_counts:
            actual = min(count, greedy_limit)
            mask = greedy_masks[actual]
            evaluation = evaluate(mask)
            records.append(
                _record(
                    condition,
                    selector="greedy_oracle",
                    requested_budget_fraction=count / total_eligible,
                    mask=mask,
                    eligible=eligible,
                    evaluation=evaluation,
                    stale_kl=stale.logits_kl,
                )
            )
            mask_rows.extend(
                _mask_rows(condition, "greedy_oracle", count / total_eligible, mask)
            )

        frontier_rows, profiles = _column_frontier_profiles(
            evaluate=evaluate,
            condition=condition,
            eligible=eligible,
            start_layer=start_layer,
            token_count=token_count,
            stale_kl=stale.logits_kl,
        )
        for count in desired_counts:
            mask = _assemble_frontier_mask(profiles, eligible, cell_budget=count)
            evaluation = evaluate(mask)
            records.append(
                _record(
                    condition,
                    selector="marginal_frontier",
                    requested_budget_fraction=count / total_eligible,
                    mask=mask,
                    eligible=eligible,
                    evaluation=evaluation,
                    stale_kl=stale.logits_kl,
                )
            )
            mask_rows.extend(
                _mask_rows(condition, "marginal_frontier", count / total_eligible, mask)
            )

        complete = eligible.copy()
        complete_eval = evaluate(complete)
        records.append(
            _record(
                condition,
                selector="eligible_full",
                requested_budget_fraction=1.0,
                mask=complete,
                eligible=eligible,
                evaluation=complete_eval,
                stale_kl=stale.logits_kl,
            )
        )
    return records, frontier_rows, mask_rows, feature_rows


def _cell_scores(
    old_layers: list[Any],
    new_layers: list[Any],
    full: BackendOutput,
    *,
    block_size: int,
    block_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    layer_count = len(old_layers)
    kv = np.zeros((layer_count, block_count), dtype=np.float64)
    attention = np.zeros_like(kv)
    attention_summary = None
    if full.extras is not None:
        candidate = full.extras.get("attention_summary")
        if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
            attention_summary = candidate
    for layer_index, (old_layer, new_layer) in enumerate(
        zip(old_layers, new_layers, strict=True)
    ):
        sequence = int(old_layer[0].shape[-2])
        for block_index in range(block_count):
            start = block_index * block_size
            end = min(sequence, start + block_size)
            numerator = 0.0
            denominator = 0.0
            for item_index in (0, 1):
                old = old_layer[item_index][..., start:end, :].detach().float()
                new = new_layer[item_index][..., start:end, :].detach().float()
                difference = new - old
                numerator += float((difference * difference).sum().cpu())
                denominator += float((new * new).sum().cpu())
            relative = math.sqrt(numerator) / max(math.sqrt(denominator), 1e-12)
            kv[layer_index, block_index] = relative
            mass = 0.0
            if attention_summary is not None and layer_index < attention_summary.shape[0]:
                mass = float(np.sum(attention_summary[layer_index, start:end]))
            attention[layer_index, block_index] = relative * mass
    return kv, attention


def _greedy_oracle_masks(
    *,
    evaluate: Any,
    candidate_mask: np.ndarray,
    max_cells: int,
) -> dict[int, np.ndarray]:
    current = np.zeros_like(candidate_mask)
    result = {0: current.copy()}
    candidates = [tuple(index) for index in np.argwhere(candidate_mask)]
    for step in range(1, max_cells + 1):
        best_cell: tuple[int, int] | None = None
        best_kl = math.inf
        for layer_index, block_index in candidates:
            if current[layer_index, block_index]:
                continue
            trial = current.copy()
            trial[layer_index, block_index] = True
            value = evaluate(trial).logits_kl
            if value < best_kl - 1e-15:
                best_kl = value
                best_cell = (layer_index, block_index)
        if best_cell is None:
            result[step] = current.copy()
            continue
        current[best_cell] = True
        result[step] = current.copy()
    return result


def _greedy_sparse_objective_masks(
    *,
    evaluate: Any,
    candidate_mask: np.ndarray,
    max_cells: int,
    reference_token_id: int,
    objective: str,
) -> dict[int, tuple[np.ndarray, _Evaluation]]:
    if objective != "confidence" and not objective.startswith("reference_nll"):
        raise ValueError(f"Unsupported sparse objective: {objective}")
    current = np.zeros_like(candidate_mask)
    candidates = [tuple(index) for index in np.argwhere(candidate_mask)]
    result: dict[int, tuple[np.ndarray, _Evaluation]] = {}
    for step in range(1, max_cells + 1):
        best_cell: tuple[int, int] | None = None
        best_evaluation: _Evaluation | None = None
        best_score = math.inf
        for layer_index, block_index in candidates:
            if current[layer_index, block_index]:
                continue
            trial = current.copy()
            trial[layer_index, block_index] = True
            evaluation = evaluate(trial)
            score = _sparse_objective_score(
                evaluation,
                reference_token_id=reference_token_id,
                objective=objective,
            )
            if not np.isfinite(score):
                continue
            if score < best_score - 1e-15:
                best_score = score
                best_cell = (layer_index, block_index)
                best_evaluation = evaluation
        if best_cell is None or best_evaluation is None:
            break
        current[best_cell] = True
        result[step] = (current.copy(), best_evaluation)
    return result


def _sparse_objective_score(
    evaluation: _Evaluation,
    *,
    reference_token_id: int,
    objective: str,
) -> float:
    reference_nll, _, max_probability = _logit_selection_metrics(
        evaluation.output.logits,
        reference_token_id=reference_token_id,
    )
    if objective == "confidence":
        return float(-max_probability)
    if objective == "reference_nll":
        return float(reference_nll)
    if objective.startswith("reference_nll_"):
        extras = evaluation.output.extras or {}
        value = extras.get(f"reference_token_nll_{objective.rsplit('_', 1)[1]}", math.nan)
        return float(value)
    raise ValueError(f"Unsupported sparse objective: {objective}")


def _beam_sparse_objective_masks(
    *,
    evaluate: Any,
    candidate_mask: np.ndarray,
    max_cells: int,
    reference_token_id: int,
    objective: str,
    beam_width: int,
) -> dict[int, _SearchPoint]:
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    candidates = [tuple(index) for index in np.argwhere(candidate_mask)]
    empty = np.zeros_like(candidate_mask)
    beam = [empty]
    result: dict[int, _SearchPoint] = {}
    probe_count = 0
    for step in range(1, min(max_cells, len(candidates)) + 1):
        expanded: dict[bytes, tuple[np.ndarray, _Evaluation, float]] = {}
        for parent in beam:
            for layer_index, block_index in candidates:
                if parent[layer_index, block_index]:
                    continue
                trial = parent.copy()
                trial[layer_index, block_index] = True
                key = np.ascontiguousarray(trial, dtype=np.uint8).tobytes()
                if key in expanded:
                    continue
                evaluation = evaluate(trial)
                probe_count += 1
                score = _sparse_objective_score(
                    evaluation,
                    reference_token_id=reference_token_id,
                    objective=objective,
                )
                if not np.isfinite(score):
                    continue
                expanded[key] = (trial, evaluation, score)
        if not expanded:
            break
        ranked = sorted(
            expanded.values(),
            key=lambda item: (item[2], np.ascontiguousarray(item[0], dtype=np.uint8).tobytes()),
        )
        beam = [item[0] for item in ranked[:beam_width]]
        best_mask, best_evaluation, best_score = ranked[0]
        result[step] = _SearchPoint(
            mask=best_mask.copy(),
            evaluation=best_evaluation,
            score=best_score,
            probe_count=probe_count,
        )
    return result


def _swap_refine_sparse_objective_masks(
    *,
    evaluate: Any,
    greedy_path: dict[int, tuple[np.ndarray, _Evaluation]],
    candidate_mask: np.ndarray,
    reference_token_id: int,
    objective: str,
    max_rounds: int,
) -> dict[int, _SearchPoint]:
    candidates = [tuple(index) for index in np.argwhere(candidate_mask)]
    total_candidates = len(candidates)
    result: dict[int, _SearchPoint] = {}
    for count, (initial_mask, initial_evaluation) in greedy_path.items():
        current = initial_mask.copy()
        current_evaluation = initial_evaluation
        current_score = _sparse_objective_score(
            current_evaluation,
            reference_token_id=reference_token_id,
            objective=objective,
        )
        probe_count = sum(total_candidates - step for step in range(count))
        for _ in range(max_rounds):
            selected = [index for index in candidates if current[index]]
            unselected = [index for index in candidates if not current[index]]
            best_mask = current
            best_evaluation = current_evaluation
            best_score = current_score
            for remove_index in selected:
                for add_index in unselected:
                    trial = current.copy()
                    trial[remove_index] = False
                    trial[add_index] = True
                    evaluation = evaluate(trial)
                    probe_count += 1
                    score = _sparse_objective_score(
                        evaluation,
                        reference_token_id=reference_token_id,
                        objective=objective,
                    )
                    if not np.isfinite(score):
                        continue
                    trial_key = np.ascontiguousarray(trial, dtype=np.uint8).tobytes()
                    best_key = np.ascontiguousarray(best_mask, dtype=np.uint8).tobytes()
                    if score < best_score - 1e-15 or (
                        abs(score - best_score) <= 1e-15 and trial_key < best_key
                    ):
                        best_mask = trial.copy()
                        best_evaluation = evaluation
                        best_score = score
            if best_score >= current_score - 1e-15:
                break
            current = best_mask
            current_evaluation = best_evaluation
            current_score = best_score
        result[count] = _SearchPoint(
            mask=current.copy(),
            evaluation=current_evaluation,
            score=current_score,
            probe_count=probe_count,
        )
    return result


def _joint_sparse_search_point(
    *,
    stale: _Evaluation,
    path: dict[int, _SearchPoint],
    reference_token_id: int,
    objective: str,
    direct_total: int,
    cost_penalty: float,
    stale_margin: float = 0.0,
) -> _SearchPoint:
    if direct_total <= 0:
        raise ValueError("direct_total must be positive")
    if stale_margin < 0.0:
        raise ValueError("stale_margin must be nonnegative")
    empty = np.zeros_like(next(iter(path.values())).mask) if path else np.zeros((1, 1), dtype=bool)
    total_probe_count = max((point.probe_count for point in path.values()), default=0)
    stale_score = _sparse_objective_score(
        stale,
        reference_token_id=reference_token_id,
        objective=objective,
    )
    best = _SearchPoint(
        mask=empty,
        evaluation=stale,
        score=stale_score,
        probe_count=total_probe_count,
    )
    best_penalized = stale_score
    acceptance_threshold = stale_score - stale_margin
    for count, point in path.items():
        penalized = point.score + cost_penalty * count / direct_total
        if penalized > acceptance_threshold - 1e-15:
            continue
        selected_best = int(np.count_nonzero(best.mask))
        if penalized < best_penalized - 1e-15 or (
            abs(penalized - best_penalized) <= 1e-15 and count < selected_best
        ):
            best = _SearchPoint(
                mask=point.mask.copy(),
                evaluation=point.evaluation,
                score=penalized,
                probe_count=total_probe_count,
            )
            best_penalized = penalized
    return best


def _column_frontier_profiles(
    *,
    evaluate: Any,
    condition: dict[str, Any],
    eligible: np.ndarray,
    start_layer: int,
    token_count: int,
    stale_kl: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    layer_count, block_count = eligible.shape
    block_size = int(condition["block_size"])
    for block_index in range(block_count):
        best_kl = stale_kl
        best_end = start_layer
        curve: list[float] = []
        for end_layer in range(start_layer + 1, layer_count + 1):
            mask = np.zeros_like(eligible)
            mask[start_layer:end_layer, block_index] = True
            value = evaluate(mask).logits_kl
            curve.append(value)
            if value < best_kl - 1e-15:
                best_kl = value
                best_end = end_layer
        gain = stale_kl - best_kl
        depth = best_end - start_layer
        row = {
            **condition,
            "token_block": block_index,
            "token_start": block_index * block_size,
            "token_end": min(token_count, (block_index + 1) * block_size),
            "best_end_layer": best_end,
            "best_depth": depth,
            "stale_logits_kl": stale_kl,
            "best_logits_kl": best_kl,
            "marginal_kl_gain": gain,
            "gain_per_cell": gain / depth if depth > 0 else 0.0,
            "curve_nonmonotonic": any(
                curve[index + 1] > curve[index] + 1e-15
                for index in range(len(curve) - 1)
            ),
            "curve": json.dumps(curve),
        }
        rows.append(row)
        profiles.append(row)
    return rows, profiles


def _assemble_frontier_mask(
    profiles: list[dict[str, Any]],
    eligible: np.ndarray,
    *,
    cell_budget: int,
) -> np.ndarray:
    mask = np.zeros_like(eligible)
    remaining = cell_budget
    start_layer = int(np.argwhere(eligible)[0][0])
    ordered = sorted(
        profiles,
        key=lambda row: (
            -float(row["gain_per_cell"]),
            -float(row["marginal_kl_gain"]),
            int(row["token_block"]),
        ),
    )
    for profile in ordered:
        depth = int(profile["best_depth"])
        if depth <= 0 or depth > remaining or float(profile["marginal_kl_gain"]) <= 0.0:
            continue
        block = int(profile["token_block"])
        mask[start_layer : start_layer + depth, block] = True
        remaining -= depth
    return mask




def _prefix_hidden_cascade_metrics(
    *,
    baseline: BackendOutput,
    full: BackendOutput,
    target_layer: int,
) -> dict[str, float]:
    """Summarize how a target-layer update propagates across prefix hidden states."""
    baseline_extras = baseline.extras or {}
    full_extras = full.extras or {}
    baseline_states = baseline_extras.get("hidden_states")
    full_states = full_extras.get("hidden_states")
    if not isinstance(baseline_states, tuple | list) or not isinstance(
        full_states, tuple | list
    ):
        return {}
    if len(baseline_states) != len(full_states) or not baseline_states:
        return {}
    input_index = max(0, min(target_layer, len(baseline_states) - 1))
    output_index = max(0, min(target_layer + 1, len(baseline_states) - 1))

    def relative_drift(index: int) -> tuple[float, Any | None]:
        old = baseline_states[index]
        new = full_states[index]
        if (
            old is None
            or new is None
            or not hasattr(old, "detach")
            or not hasattr(new, "detach")
            or tuple(old.shape) != tuple(new.shape)
        ):
            return math.nan, None
        old_float = old.detach().float()
        new_float = new.detach().float()
        difference = new_float - old_float
        numerator = float((difference * difference).sum().sqrt().cpu())
        denominator = float((new_float * new_float).sum().sqrt().cpu())
        return numerator / max(denominator, 1e-12), difference

    input_drift, _ = relative_drift(input_index)
    target_drift, target_difference = relative_drift(output_index)
    final_drift, _ = relative_drift(len(baseline_states) - 1)
    metrics = {
        "target_prefix_input_drift_relative": input_drift,
        "target_prefix_output_drift_relative": target_drift,
        "final_prefix_hidden_drift_relative": final_drift,
        "prefix_hidden_cascade_amplification": final_drift / max(target_drift, 1e-12),
    }
    if target_difference is None:
        return metrics
    token_energy = (target_difference * target_difference).sum(dim=-1)
    while token_energy.ndim > 1:
        token_energy = token_energy.sum(dim=0)
    energy = np.asarray(token_energy.detach().float().cpu().numpy(), dtype=np.float64)
    energy = energy.reshape(-1)
    total = float(np.sum(energy))
    if total <= 1e-24 or not len(energy):
        metrics.update(
            {
                "target_prefix_drift_top10_fraction": 0.0,
                "target_prefix_drift_entropy": 0.0,
                "target_prefix_drift_first_half_fraction": 0.0,
            }
        )
        return metrics
    top_count = max(1, int(math.ceil(0.1 * len(energy))))
    probabilities = energy / total
    nonzero = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    normalized_entropy = entropy / max(math.log(len(energy)), 1e-12)
    split = max(1, len(energy) // 2)
    metrics.update(
        {
            "target_prefix_drift_top10_fraction": float(
                np.sum(np.sort(energy)[-top_count:]) / total
            ),
            "target_prefix_drift_entropy": normalized_entropy,
            "target_prefix_drift_first_half_fraction": float(
                np.sum(energy[:split]) / total
            ),
        }
    )
    return metrics


def _cache_cascade_metrics(
    *,
    kv_scores: np.ndarray,
    target_layer: int,
) -> dict[str, float]:
    """Summarize full-reference cache drift beyond the directly updated layer."""
    scores = np.asarray(kv_scores, dtype=np.float64)
    if scores.ndim != 2 or not (0 <= target_layer < scores.shape[0]):
        return {}

    def rms(values: np.ndarray) -> float:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        return float(np.sqrt(np.mean(flat * flat))) if len(flat) else 0.0

    target_rms = rms(scores[target_layer])
    downstream = scores[target_layer + 1 :]
    downstream_rms = rms(downstream)
    downstream_layer_rms = (
        np.sqrt(np.mean(downstream * downstream, axis=1))
        if downstream.size
        else np.zeros(0, dtype=np.float64)
    )
    final_rms = rms(scores[-1])
    return {
        "target_cache_drift_rms": target_rms,
        "downstream_cache_drift_rms": downstream_rms,
        "final_cache_drift_rms": final_rms,
        "cache_cascade_amplification": downstream_rms / max(target_rms, 1e-12),
        "cache_cascade_final_amplification": final_rms / max(target_rms, 1e-12),
        "downstream_cache_drift_max_layer_rms": (
            float(np.max(downstream_layer_rms)) if len(downstream_layer_rms) else 0.0
        ),
        "downstream_cache_drift_layer_rms_std": (
            float(np.std(downstream_layer_rms)) if len(downstream_layer_rms) else 0.0
        ),
    }


def _attention_residual_metrics(
    *,
    candidate: BackendOutput,
    full: BackendOutput,
    stale: BackendOutput,
    baseline: BackendOutput,
    target_layer: int,
) -> dict[str, float]:
    """Compare a candidate with full recompute in attention-output space."""

    def summary(output: BackendOutput, name: str) -> np.ndarray | None:
        extras = output.extras or {}
        value = extras.get(name)
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            return None
        return np.asarray(value, dtype=np.float64)

    candidate_output = summary(candidate, "attention_output_summary")
    full_output = summary(full, "attention_output_summary")
    stale_output = summary(stale, "attention_output_summary")
    baseline_output = summary(baseline, "attention_output_summary")
    candidate_input = summary(candidate, "attention_input_summary")
    full_input = summary(full, "attention_input_summary")
    stale_input = summary(stale, "attention_input_summary")
    baseline_input = summary(baseline, "attention_input_summary")
    required = (candidate_output, full_output, stale_output)
    if any(value is None for value in required):
        return {}
    assert candidate_output is not None
    assert full_output is not None
    assert stale_output is not None
    if not (
        candidate_output.shape == full_output.shape == stale_output.shape
        and 0 <= target_layer < candidate_output.shape[0]
    ):
        return {}

    def vector_metrics(
        candidate_vector: np.ndarray,
        full_vector: np.ndarray,
        stale_vector: np.ndarray,
        *,
        prefix: str,
    ) -> dict[str, float]:
        candidate_vector = np.asarray(candidate_vector, dtype=np.float64).reshape(-1)
        full_vector = np.asarray(full_vector, dtype=np.float64).reshape(-1)
        stale_vector = np.asarray(stale_vector, dtype=np.float64).reshape(-1)
        stale_delta = full_vector - stale_vector
        candidate_delta = candidate_vector - stale_vector
        stale_error = float(np.linalg.norm(stale_delta))
        candidate_error = float(np.linalg.norm(full_vector - candidate_vector))
        full_scale = float(np.linalg.norm(full_vector))
        delta_squared = float(np.dot(stale_delta, stale_delta))
        candidate_delta_norm = float(np.linalg.norm(candidate_delta))
        projection = (
            float(np.dot(candidate_delta, stale_delta) / delta_squared)
            if delta_squared > 1e-24
            else 0.0
        )
        orthogonal = candidate_delta - projection * stale_delta
        cosine = (
            float(np.dot(candidate_delta, stale_delta) / (candidate_delta_norm * stale_error))
            if candidate_delta_norm > 1e-12 and stale_error > 1e-12
            else 0.0
        )
        return {
            f"{prefix}_stale_error_l2": stale_error,
            f"{prefix}_candidate_error_l2": candidate_error,
            f"{prefix}_candidate_error_relative": candidate_error / max(full_scale, 1e-12),
            f"{prefix}_recovery_fraction": (
                (stale_error - candidate_error) / max(stale_error, 1e-12)
            ),
            f"{prefix}_delta_projection": projection,
            f"{prefix}_delta_cosine": cosine,
            f"{prefix}_orthogonal_error_fraction": float(np.linalg.norm(orthogonal))
            / max(stale_error, 1e-12),
        }

    metrics = vector_metrics(
        candidate_output[target_layer],
        full_output[target_layer],
        stale_output[target_layer],
        prefix="target_attention_output",
    )
    metrics.update(
        vector_metrics(
            candidate_output[-1],
            full_output[-1],
            stale_output[-1],
            prefix="final_attention_output",
        )
    )
    metrics.update(
        vector_metrics(
            candidate_output[target_layer:],
            full_output[target_layer:],
            stale_output[target_layer:],
            prefix="downstream_attention_output",
        )
    )

    if all(
        value is not None
        for value in (candidate_input, full_input, stale_input, baseline_input)
    ):
        assert candidate_input is not None
        assert full_input is not None
        assert stale_input is not None
        assert baseline_input is not None
        if (
            candidate_input.shape
            == full_input.shape
            == stale_input.shape
            == baseline_input.shape
            and target_layer < candidate_input.shape[0]
        ):
            scale = max(
                float(np.linalg.norm(baseline_input[target_layer])),
                1e-12,
            )
            metrics.update(
                {
                    "target_attention_input_candidate_shift_relative": float(
                        np.linalg.norm(
                            candidate_input[target_layer] - baseline_input[target_layer]
                        )
                    )
                    / scale,
                    "target_attention_input_stale_shift_relative": float(
                        np.linalg.norm(stale_input[target_layer] - baseline_input[target_layer])
                    )
                    / scale,
                    "target_attention_input_full_shift_relative": float(
                        np.linalg.norm(full_input[target_layer] - baseline_input[target_layer])
                    )
                    / scale,
                }
            )
    if baseline_output is not None and baseline_output.shape == full_output.shape:
        scale = max(float(np.linalg.norm(baseline_output[target_layer])), 1e-12)
        metrics["target_attention_output_full_shift_vs_baseline_relative"] = float(
            np.linalg.norm(full_output[target_layer] - baseline_output[target_layer])
        ) / scale
    return metrics


def _record(
    condition: dict[str, Any],
    *,
    selector: str,
    requested_budget_fraction: float,
    mask: np.ndarray,
    eligible: np.ndarray,
    evaluation: _Evaluation,
    stale_kl: float,
    selection_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = int(np.count_nonzero(mask))
    eligible_count = int(np.count_nonzero(eligible))
    structure = _mask_structure(mask)
    extras = evaluation.output.extras or {}
    reference_token_id = int(condition.get("reference_token_id", -1))
    reference_nll, output_entropy, output_max_probability = _logit_selection_metrics(
        evaluation.output.logits,
        reference_token_id=reference_token_id,
    )
    sequence_metrics = {
        key: value
        for key, value in extras.items()
        if (
            key.startswith("reference_token_nll_")
            or key.startswith("reference_token_kl_")
        )
        and isinstance(value, int | float)
    }
    attention_residual_metrics = {
        key: value
        for key, value in extras.items()
        if key.startswith(
            (
                "target_attention_",
                "final_attention_",
                "downstream_attention_",
            )
        )
        and isinstance(value, int | float)
    }
    return {
        **condition,
        "selector": selector,
        "requested_budget_fraction": requested_budget_fraction,
        "selected_cells": selected,
        "eligible_cells": eligible_count,
        "selected_fraction": selected / eligible_count if eligible_count else 0.0,
        "logits_kl": evaluation.logits_kl,
        "top1_agreement": evaluation.top1_agreement,
        "reference_token_nll": reference_nll,
        "output_entropy": output_entropy,
        "output_max_probability": output_max_probability,
        "reference_sequence_nll": float(extras.get("reference_sequence_nll", math.nan)),
        "reference_sequence_kl": float(extras.get("reference_sequence_kl", math.nan)),
        "reference_probe_tokens": int(extras.get("reference_probe_tokens", 0)),
        "reference_probe_latency": float(extras.get("reference_probe_latency", 0.0)),
        **sequence_metrics,
        **attention_residual_metrics,
        "stale_logits_kl": stale_kl,
        "kl_gain_vs_stale": stale_kl - evaluation.logits_kl,
        "beneficial_vs_stale": evaluation.logits_kl <= stale_kl + 1e-15,
        "cache_mode": str(extras.get("cache_mode", "unknown")),
        "strategy_flops": float(extras.get("strategy_flops", 0.0)),
        "cache_maintenance_latency": float(
            extras.get("cache_maintenance_latency", 0.0)
        ),
        "decode_latency": float(extras.get("decode_latency", 0.0)),
        "selected_direct_cells": int(extras.get("selected_direct_cells", 0)),
        "available_direct_cells": int(extras.get("available_direct_cells", 0)),
        "selected_direct_fraction": float(
            extras.get("selected_direct_fraction", 0.0)
        ),
        "residual_cache_bytes": int(extras.get("residual_cache_bytes", 0)),
        "mask_hash": hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8)).hexdigest()[:16],
        **structure,
        **(selection_metadata or {}),
    }


def _mask_structure(mask: np.ndarray) -> dict[str, Any]:
    selected = np.argwhere(mask)
    if len(selected) == 0:
        return {
            "active_layers": 0,
            "active_token_blocks": 0,
            "layer_span": 0,
            "token_block_span": 0,
            "rectangle_fill": 0.0,
            "column_contiguity": 1.0,
            "row_contiguity": 1.0,
            "connected_components": 0,
            "frontier_depth_std": 0.0,
        }
    layers = sorted(set(int(value) for value in selected[:, 0]))
    blocks = sorted(set(int(value) for value in selected[:, 1]))
    layer_span = layers[-1] - layers[0] + 1
    block_span = blocks[-1] - blocks[0] + 1
    column_contiguous = []
    depths = []
    for block in blocks:
        values = sorted(int(value) for value in np.argwhere(mask[:, block]).reshape(-1))
        column_contiguous.append(values[-1] - values[0] + 1 == len(values))
        depths.append(values[-1] - values[0] + 1)
    row_contiguous = []
    for layer in layers:
        values = sorted(int(value) for value in np.argwhere(mask[layer]).reshape(-1))
        row_contiguous.append(values[-1] - values[0] + 1 == len(values))
    return {
        "active_layers": len(layers),
        "active_token_blocks": len(blocks),
        "layer_span": layer_span,
        "token_block_span": block_span,
        "rectangle_fill": len(selected) / (layer_span * block_span),
        "column_contiguity": float(np.mean(column_contiguous)),
        "row_contiguity": float(np.mean(row_contiguous)),
        "connected_components": _connected_components(mask),
        "frontier_depth_std": float(np.std(depths)) if depths else 0.0,
    }


def _connected_components(mask: np.ndarray) -> int:
    remaining = {tuple(index) for index in np.argwhere(mask)}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            layer, block = stack.pop()
            for neighbor in (
                (layer - 1, block),
                (layer + 1, block),
                (layer, block - 1),
                (layer, block + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _candidate_pool(
    kv_scores: np.ndarray,
    attention_scores: np.ndarray,
    eligible: np.ndarray,
    *,
    limit: int,
) -> np.ndarray:
    total = int(np.count_nonzero(eligible))
    limit = min(limit, total)
    pool = np.zeros_like(eligible)
    each = max(1, limit // 2)
    for scores in (kv_scores, attention_scores):
        pool |= _top_mask(scores, eligible, each)
    if int(np.count_nonzero(pool)) < limit:
        pool |= _top_mask(kv_scores + attention_scores, eligible, limit)
    return pool


def _signed_residual_greedy_order(
    correction_vectors: np.ndarray,
    available: np.ndarray,
    *,
    max_cells: int | None = None,
) -> tuple[list[tuple[int, int]], list[float], float]:
    """Order cells by their marginal reduction of signed correction residual."""
    vectors = np.asarray(correction_vectors, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if vectors.ndim != 3 or vectors.shape[:2] != mask.shape:
        raise ValueError(
            "correction_vectors must have shape [layers, blocks, features] "
            "matching available"
        )
    candidates = [
        tuple(int(value) for value in index) for index in np.argwhere(mask)
    ]
    limit = len(candidates) if max_cells is None else min(max_cells, len(candidates))
    if limit < 0:
        raise ValueError("max_cells must be nonnegative")
    residual = np.sum(np.where(mask[..., None], vectors, 0.0), axis=1)
    initial_energy = float(np.sum(residual * residual))
    order: list[tuple[int, int]] = []
    marginal_gains: list[float] = []
    remaining = set(candidates)
    while remaining and len(order) < limit:
        scored: list[tuple[float, int, int]] = []
        for layer, block in remaining:
            vector = vectors[layer, block]
            marginal = float(
                2.0 * np.dot(residual[layer], vector) - np.dot(vector, vector)
            )
            scored.append((marginal, layer, block))
        marginal, layer, block = max(
            scored,
            key=lambda item: (item[0], -item[1], -item[2]),
        )
        order.append((layer, block))
        marginal_gains.append(marginal)
        residual[layer] -= vectors[layer, block]
        remaining.remove((layer, block))
    return order, marginal_gains, initial_energy


def _signed_residual_best_prefix(
    marginal_gains: list[float],
    initial_energy: float,
    *,
    cost_fraction: float,
) -> tuple[int, float]:
    """Choose the residual-greedy prefix minimizing residual plus cell cost."""
    if initial_energy < 0.0:
        raise ValueError("initial_energy must be nonnegative")
    if cost_fraction < 0.0:
        raise ValueError("cost_fraction must be nonnegative")
    best_count = 0
    best_energy = initial_energy
    best_objective = initial_energy
    cumulative_gain = 0.0
    cell_cost = cost_fraction * initial_energy
    for count, marginal in enumerate(marginal_gains, start=1):
        cumulative_gain += marginal
        residual_energy = max(initial_energy - cumulative_gain, 0.0)
        objective = residual_energy + cell_cost * count
        if (objective, count) < (best_objective, best_count):
            best_count = count
            best_energy = residual_energy
            best_objective = objective
    return best_count, best_energy


def _mask_from_order(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    count: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for layer, block in order[: max(0, count)]:
        mask[layer, block] = True
    return mask


def _top_mask(scores: np.ndarray, eligible: np.ndarray, count: int) -> np.ndarray:
    indices = np.argwhere(eligible)
    ordered = sorted(
        (tuple(index) for index in indices),
        key=lambda index: (-float(scores[index]), index[0], index[1]),
    )
    mask = np.zeros_like(eligible)
    for index in ordered[:count]:
        mask[index] = True
    return mask


def _mask_rows(
    condition: dict[str, Any],
    selector: str,
    budget_fraction: float,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            **condition,
            "selector": selector,
            "requested_budget_fraction": budget_fraction,
            "layer": int(layer),
            "token_block": int(block),
        }
        for layer, block in np.argwhere(mask)
    ]


def _past_layers(output: BackendOutput) -> list[Any]:
    if not output.extras or output.extras.get("past_key_values") is None:
        raise ValueError("Backend output does not contain past_key_values")
    past = output.extras["past_key_values"]
    if hasattr(past, "to_legacy_cache"):
        return list(past.to_legacy_cache())
    return list(past)


def _reference_token_ids(backend: ModelBackend, sample: TaskSample) -> list[int]:
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        return []
    encoded = tokenizer(sample.answer, add_special_tokens=False)
    getter = getattr(encoded, "get", None)
    if not callable(getter):
        return []
    token_ids = getter("input_ids", [])
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _reference_token_id(backend: ModelBackend, sample: TaskSample) -> int:
    token_ids = _reference_token_ids(backend, sample)
    return token_ids[0] if token_ids else -1


def _logit_selection_metrics(
    logits: np.ndarray,
    *,
    reference_token_id: int,
) -> tuple[float, float, float]:
    values = np.asarray(logits, dtype=np.float64).reshape(-1, logits.shape[-1])[0]
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    denominator = float(np.sum(exp_values))
    probabilities = exp_values / max(denominator, 1e-300)
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    max_probability = float(np.max(probabilities))
    if reference_token_id < 0 or reference_token_id >= len(values):
        return math.nan, entropy, max_probability
    log_partition = float(np.log(max(denominator, 1e-300)) + np.max(values))
    return log_partition - float(values[reference_token_id]), entropy, max_probability


def _zero_probe_baseline_stale_kl(
    *,
    baseline_logits: np.ndarray,
    stale_logits: np.ndarray,
) -> float:
    """Compute the one runtime risk feature with a single float32 softmax pair."""

    baseline_values = np.asarray(baseline_logits, dtype=np.float32).reshape(
        -1, baseline_logits.shape[-1]
    )[0]
    stale_values = np.asarray(stale_logits, dtype=np.float32).reshape(
        -1, stale_logits.shape[-1]
    )[0]
    if baseline_values.shape != stale_values.shape:
        raise ValueError("Baseline and stale logits must have the same vocabulary shape")
    baseline_shifted = baseline_values - np.max(baseline_values)
    stale_shifted = stale_values - np.max(stale_values)
    baseline_log_probabilities = baseline_shifted - np.log(
        np.sum(np.exp(baseline_shifted), dtype=np.float64)
    )
    stale_log_probabilities = stale_shifted - np.log(
        np.sum(np.exp(stale_shifted), dtype=np.float64)
    )
    baseline_probabilities = np.exp(baseline_log_probabilities)
    value = np.sum(
        baseline_probabilities
        * (baseline_log_probabilities - stale_log_probabilities),
        dtype=np.float64,
    )
    return max(float(value), 0.0)


def _zero_probe_failure_metrics(
    *,
    baseline_logits: np.ndarray,
    stale_logits: np.ndarray,
) -> dict[str, float]:
    """Summarize old-model versus stale-cache output drift without a new forward."""

    baseline_values = np.asarray(baseline_logits, dtype=np.float64).reshape(
        -1, baseline_logits.shape[-1]
    )[0]
    stale_values = np.asarray(stale_logits, dtype=np.float64).reshape(
        -1, stale_logits.shape[-1]
    )[0]
    if baseline_values.shape != stale_values.shape:
        raise ValueError("Baseline and stale logits must have the same vocabulary shape")

    def probabilities(values: np.ndarray) -> np.ndarray:
        shifted = values - float(np.max(values))
        exp_values = np.exp(shifted)
        return exp_values / max(float(np.sum(exp_values)), 1e-300)

    baseline_probabilities = probabilities(baseline_values)
    stale_probabilities = probabilities(stale_values)
    midpoint = 0.5 * (baseline_probabilities + stale_probabilities)
    epsilon = 1e-300
    js = 0.5 * np.sum(
        baseline_probabilities
        * (
            np.log(baseline_probabilities + epsilon)
            - np.log(midpoint + epsilon)
        )
    )
    js += 0.5 * np.sum(
        stale_probabilities
        * (
            np.log(stale_probabilities + epsilon)
            - np.log(midpoint + epsilon)
        )
    )

    baseline_positive = baseline_probabilities[baseline_probabilities > 0.0]
    stale_positive = stale_probabilities[stale_probabilities > 0.0]
    baseline_entropy = float(
        -np.sum(baseline_positive * np.log(baseline_positive))
    )
    stale_entropy = float(-np.sum(stale_positive * np.log(stale_positive)))
    baseline_max_probability = float(np.max(baseline_probabilities))
    stale_max_probability = float(np.max(stale_probabilities))
    delta = stale_values - baseline_values
    baseline_norm = max(float(np.linalg.norm(baseline_values)), 1e-12)
    stale_norm = max(float(np.linalg.norm(stale_values)), 1e-12)
    cosine = float(
        np.dot(baseline_values, stale_values) / (baseline_norm * stale_norm)
    )
    return {
        "router_baseline_stale_kl": kl_divergence(
            baseline_values[None, :], stale_values[None, :]
        ),
        "router_stale_baseline_kl": kl_divergence(
            stale_values[None, :], baseline_values[None, :]
        ),
        "router_baseline_stale_js": float(js),
        "router_baseline_stale_top1_agreement": top1_agreement(
            baseline_values[None, :], stale_values[None, :]
        ),
        "router_baseline_entropy": baseline_entropy,
        "router_stale_entropy": stale_entropy,
        "router_entropy_delta": stale_entropy - baseline_entropy,
        "router_baseline_max_probability": baseline_max_probability,
        "router_stale_max_probability": stale_max_probability,
        "router_max_probability_delta": (
            stale_max_probability - baseline_max_probability
        ),
        "router_baseline_stale_logit_relative_l2": (
            float(np.linalg.norm(delta)) / baseline_norm
        ),
        "router_baseline_stale_logit_cosine": cosine,
    }


def _prepare_target(
    backend: ModelBackend,
    config: VersionedExperimentConfig,
    target: UpdateTarget,
) -> None:
    if config.adapter.update_mode != "lora_train" or not target.is_lora:
        return
    prepare = getattr(backend, "prepare_update_target", None)
    if not callable(prepare):
        raise RuntimeError("Backend does not implement LoRA target preparation")
    prepare(
        target,
        rank=config.adapter.lora_rank,
        alpha=config.adapter.lora_alpha,
        freeze_base_model=config.adapter.freeze_base_model,
    )


def _single_token_sample(sample: TaskSample) -> TaskSample:
    metadata = dict(sample.metadata)
    metadata["max_generation_tokens"] = 1
    return TaskSample(prompt=sample.prompt, answer=sample.answer, metadata=metadata)


def _condition_seed(condition: dict[str, Any]) -> int:
    stable_fields = {
        key: condition.get(key)
        for key in (
            "seed",
            "sample_id",
            "dataset_sample_id",
            "task_name",
            "model_name",
            "update_target",
            "target_layer",
            "version_gap",
            "configured_update_norm",
            "context_length",
            "block_size",
        )
    }
    payload = json.dumps(stable_fields, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _condition_key(row: dict[str, Any]) -> tuple[int, str, int, int, int]:
    return (
        int(row["sample_id"]),
        str(row["update_target"]),
        int(row["block_size"]),
        int(row["version_gap"]),
        int(row["context_length"]),
    )


def _completed_condition_keys(
    records: list[dict[str, Any]],
    frontier_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
) -> set[tuple[int, str, int, int, int]]:
    if not records or not frontier_rows or not mask_rows:
        return set()
    return (
        {_condition_key(row) for row in records}
        & {_condition_key(row) for row in frontier_rows}
        & {_condition_key(row) for row in mask_rows}
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            rows.append(payload)
    return rows


def _commit_temp_file(path: Path, temporary: Path) -> None:
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fields = sorted({field for row in rows for field in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    _commit_temp_file(path, temporary)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _commit_temp_file(path, temporary)


def _report(records: list[dict[str, Any]], frontier_rows: list[dict[str, Any]]) -> str:
    lines = ["# Blockwise cache exploration", ""]
    if not records:
        return "\n".join([*lines, "No records.", ""])
    selectors = sorted({str(row["selector"]) for row in records})
    lines.extend(
        [
            "## Selector summary",
            "",
            "| Selector | Rows | Mean KL | Mean gain vs stale | Mean selected fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for selector in selectors:
        group = [row for row in records if row["selector"] == selector]
        lines.append(
            f"| {selector} | {len(group)} | "
            f"{np.mean([float(row['logits_kl']) for row in group]):.6g} | "
            f"{np.mean([float(row['kl_gain_vs_stale']) for row in group]):.6g} | "
            f"{np.mean([float(row['selected_fraction']) for row in group]):.4f} |"
        )
    if frontier_rows:
        positive = [row for row in frontier_rows if float(row["marginal_kl_gain"]) > 0.0]
        nonmonotonic = [row for row in frontier_rows if bool(row["curve_nonmonotonic"])]
        depths = [int(row["best_depth"]) for row in positive]
        lines.extend(
            [
                "",
                "## Frontier structure",
                "",
                f"- Positive marginal token blocks: {len(positive)}/{len(frontier_rows)}",
                f"- Nonmonotonic per-block depth curves: {len(nonmonotonic)}/{len(frontier_rows)}",
                f"- Best-depth standard deviation: {float(np.std(depths)) if depths else 0.0:.4f}",
            ]
        )
    lines.append("")
    return "\n".join(lines)
