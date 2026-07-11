from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.data.loader import build_task_samples
from ttt_cache_lab.data.synthetic import TaskSample
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
    report_markdown: Path


@dataclass(frozen=True)
class _Evaluation:
    output: BackendOutput
    logits_kl: float
    top1_agreement: float


def run_blockwise_exploration(
    config: VersionedExperimentConfig,
    *,
    block_sizes: tuple[int, ...],
    version_gap: int,
    budget_fractions: tuple[float, ...],
    oracle_candidate_limit: int = 24,
    oracle_max_cells: int = 16,
    direct_oracle_max_blocks: int = 0,
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

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(config.model, seed=config.seed)
    backend.configure_metrics(capture_attention=True)
    splice = getattr(backend, "probe_blockwise_cache_splice", None)
    if not callable(splice):
        raise RuntimeError("The selected backend does not implement blockwise cache splicing")

    records: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    samples = build_task_samples(config.data, seed=config.seed)
    try:
        for sample_id, raw_sample in enumerate(samples):
            sample = _single_token_sample(raw_sample)
            sample = backend.prepare_sample(sample, context_length=config.data.context_length)
            for target_name in config.updates.targets:
                target = parse_update_target(target_name, num_layers=backend.num_layers)
                backend.restore_after_update()
                _prepare_target(backend, config, target)
                baseline = backend.prefill(sample.prompt)
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
                full = backend.full_recompute(sample.prompt, current)
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
                    "context_length": config.data.context_length,
                    "seed": config.seed,
                }
                for block_size in block_sizes:
                    condition = {
                        **base_condition,
                        "block_size": block_size,
                        "reference_token_id": _reference_token_id(backend, sample),
                    }
                    condition_records, condition_frontier, condition_masks = _explore_condition(
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
                    )
                    records.extend(condition_records)
                    frontier_rows.extend(condition_frontier)
                    mask_rows.extend(condition_masks)
                _write_rows(output_dir / "blockwise_records.csv", records)
                _write_jsonl(output_dir / "blockwise_records.jsonl", records)
                _write_rows(output_dir / "block_frontier.csv", frontier_rows)
                _write_rows(output_dir / "block_masks.csv", mask_rows)
    finally:
        backend.restore_after_update()

    report_path = output_dir / "blockwise_report.md"
    report_path.write_text(_report(records, frontier_rows), encoding="utf-8")
    return BlockwiseArtifacts(
        records_csv=output_dir / "blockwise_records.csv",
        records_jsonl=output_dir / "blockwise_records.jsonl",
        frontier_csv=output_dir / "block_frontier.csv",
        masks_csv=output_dir / "block_masks.csv",
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    del sample
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
    kv_scores, attention_scores = _cell_scores(
        old_layers,
        new_layers,
        full,
        block_size=block_size,
        block_count=block_count,
    )
    cache: dict[bytes, _Evaluation] = {}

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
        value = _Evaluation(
            output=output,
            logits_kl=kl_divergence(full.logits, output.logits),
            top1_agreement=top1_agreement(full.logits, output.logits),
        )
        cache[key] = value
        return value

    empty = np.zeros_like(eligible)
    stale = evaluate(empty)
    sparse_probe = getattr(backend, "probe_blockwise_lora_delta", None)
    sparse_score_fn = getattr(backend, "blockwise_lora_delta_scores", None)
    sparse_cache: dict[bytes, _Evaluation] = {}

    def evaluate_sparse(mask: np.ndarray) -> _Evaluation:
        if not callable(sparse_probe):
            raise RuntimeError("Backend does not implement block-sparse LoRA delta repair")
        key = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        cached = sparse_cache.get(key)
        if cached is not None:
            return cached
        output = sparse_probe(
            baseline=baseline,
            block_mask=mask,
            block_size=block_size,
        )
        value = _Evaluation(
            output=output,
            logits_kl=kl_divergence(full.logits, output.logits),
            top1_agreement=top1_agreement(full.logits, output.logits),
        )
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
    mask_rows: list[dict[str, Any]] = []
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
            mask_rows.extend(_mask_rows(condition, selector, count / total_eligible, mask))

    if sparse_scores is not None:
        direct_available = np.asarray(sparse_scores["available"], dtype=bool) & eligible
        direct_total = int(np.count_nonzero(direct_available))
        sparse_selectors = {
            "sparse_attention_mass": "stale_attention_mass",
            "sparse_input_bound": "input_weight_bound",
            "sparse_attention_input_bound": "attention_input_bound",
            "sparse_predicted_delta_norm": "predicted_delta_norm",
            "sparse_attention_predicted_delta": "attention_predicted_delta",
        }
        if direct_total > 0:
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
            if 0 < direct_total <= direct_oracle_max_blocks:
                direct_indices = [tuple(index) for index in np.argwhere(direct_available)]
                seen_direct_counts: set[int] = set()
                for count in desired_counts:
                    direct_count = min(count, direct_total)
                    if direct_count in seen_direct_counts:
                        continue
                    seen_direct_counts.add(direct_count)
                    best_splice_mask: np.ndarray | None = None
                    best_splice_eval: _Evaluation | None = None
                    best_sparse_mask: np.ndarray | None = None
                    best_sparse_eval: _Evaluation | None = None
                    best_reference_mask: np.ndarray | None = None
                    best_reference_eval: _Evaluation | None = None
                    best_reference_nll = math.inf
                    best_confidence_mask: np.ndarray | None = None
                    best_confidence_eval: _Evaluation | None = None
                    best_confidence = -math.inf
                    reference_token_id = int(condition.get("reference_token_id", -1))
                    for chosen in combinations(direct_indices, direct_count):
                        mask = np.zeros_like(direct_available)
                        for index in chosen:
                            mask[index] = True
                        splice_eval = evaluate(mask)
                        if (
                            best_splice_eval is None
                            or splice_eval.logits_kl < best_splice_eval.logits_kl - 1e-15
                        ):
                            best_splice_mask = mask.copy()
                            best_splice_eval = splice_eval
                        sparse_eval = evaluate_sparse(mask)
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
                        if max_probability > best_confidence:
                            best_confidence = max_probability
                            best_confidence_mask = mask.copy()
                            best_confidence_eval = sparse_eval
                    if (
                        best_splice_mask is None
                        or best_splice_eval is None
                        or best_sparse_mask is None
                        or best_sparse_eval is None
                    ):
                        continue
                    budget = direct_count / total_eligible
                    records.append(
                        _record(
                            condition,
                            selector="direct_splice_oracle",
                            requested_budget_fraction=budget,
                            mask=best_splice_mask,
                            eligible=eligible,
                            evaluation=best_splice_eval,
                            stale_kl=stale.logits_kl,
                        )
                    )
                    records.append(
                        _record(
                            condition,
                            selector="sparse_delta_oracle",
                            requested_budget_fraction=budget,
                            mask=best_sparse_mask,
                            eligible=eligible,
                            evaluation=best_sparse_eval,
                            stale_kl=stale.logits_kl,
                        )
                    )
                    if best_reference_mask is not None and best_reference_eval is not None:
                        records.append(
                            _record(
                                condition,
                                selector="sparse_reference_objective_oracle",
                                requested_budget_fraction=budget,
                                mask=best_reference_mask,
                                eligible=eligible,
                                evaluation=best_reference_eval,
                                stale_kl=stale.logits_kl,
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(
                                condition,
                                "sparse_reference_objective_oracle",
                                budget,
                                best_reference_mask,
                            )
                        )
                    if best_confidence_mask is not None and best_confidence_eval is not None:
                        records.append(
                            _record(
                                condition,
                                selector="sparse_confidence_objective_oracle",
                                requested_budget_fraction=budget,
                                mask=best_confidence_mask,
                                eligible=eligible,
                                evaluation=best_confidence_eval,
                                stale_kl=stale.logits_kl,
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(
                                condition,
                                "sparse_confidence_objective_oracle",
                                budget,
                                best_confidence_mask,
                            )
                        )
                    mask_rows.extend(
                        _mask_rows(
                            condition,
                            "direct_splice_oracle",
                            budget,
                            best_splice_mask,
                        )
                    )
                    mask_rows.extend(
                        _mask_rows(
                            condition,
                            "sparse_delta_oracle",
                            budget,
                            best_sparse_mask,
                        )
                    )
                max_direct_count = max(seen_direct_counts) if seen_direct_counts else 0
                objective_paths = {
                    "sparse_reference_greedy": _greedy_sparse_objective_masks(
                        evaluate=evaluate_sparse,
                        candidate_mask=direct_available,
                        max_cells=max_direct_count,
                        reference_token_id=int(condition.get("reference_token_id", -1)),
                        objective="reference_nll",
                    ),
                    "sparse_confidence_greedy": _greedy_sparse_objective_masks(
                        evaluate=evaluate_sparse,
                        candidate_mask=direct_available,
                        max_cells=max_direct_count,
                        reference_token_id=int(condition.get("reference_token_id", -1)),
                        objective="confidence",
                    ),
                }
                for selector, path_by_count in objective_paths.items():
                    for direct_count in sorted(seen_direct_counts):
                        selected = path_by_count.get(direct_count)
                        if selected is None:
                            continue
                        mask, evaluation = selected
                        budget = direct_count / total_eligible
                        records.append(
                            _record(
                                condition,
                                selector=selector,
                                requested_budget_fraction=budget,
                                mask=mask,
                                eligible=eligible,
                                evaluation=evaluation,
                                stale_kl=stale.logits_kl,
                            )
                        )
                        mask_rows.extend(
                            _mask_rows(condition, selector, budget, mask)
                        )

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
    return records, frontier_rows, mask_rows


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
    if objective not in {"reference_nll", "confidence"}:
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
            reference_nll, _, max_probability = _logit_selection_metrics(
                evaluation.output.logits,
                reference_token_id=reference_token_id,
            )
            score = reference_nll if objective == "reference_nll" else -max_probability
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


def _record(
    condition: dict[str, Any],
    *,
    selector: str,
    requested_budget_fraction: float,
    mask: np.ndarray,
    eligible: np.ndarray,
    evaluation: _Evaluation,
    stale_kl: float,
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


def _reference_token_id(backend: ModelBackend, sample: TaskSample) -> int:
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        return -1
    encoded = tokenizer(sample.answer, add_special_tokens=False)
    getter = getattr(encoded, "get", None)
    if not callable(getter):
        return -1
    token_ids = getter("input_ids", [])
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return int(token_ids[0]) if token_ids else -1


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
    payload = json.dumps(condition, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


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
