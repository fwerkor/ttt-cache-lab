from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
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
    block_size: int,
    version_gap: int,
    budget_fractions: tuple[float, ...],
    oracle_candidate_limit: int = 24,
    oracle_max_cells: int = 16,
) -> BlockwiseArtifacts:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if version_gap <= 0:
        raise ValueError("version_gap must be positive")
    if not budget_fractions or any(value <= 0.0 or value > 1.0 for value in budget_fractions):
        raise ValueError("budget fractions must be in (0, 1]")
    if oracle_candidate_limit < 1 or oracle_max_cells < 1:
        raise ValueError("oracle limits must be positive")

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
                condition = {
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
                    "block_size": block_size,
                    "seed": config.seed,
                }
                condition_records, condition_frontier, condition_masks = _explore_condition(
                    backend=backend,
                    splice=splice,
                    baseline=baseline,
                    full=full,
                    target=target,
                    condition=condition,
                    budget_fractions=budget_fractions,
                    oracle_candidate_limit=oracle_candidate_limit,
                    oracle_max_cells=oracle_max_cells,
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
    splice: Any,
    baseline: BackendOutput,
    full: BackendOutput,
    target: UpdateTarget,
    condition: dict[str, Any],
    budget_fractions: tuple[float, ...],
    oracle_candidate_limit: int,
    oracle_max_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    return {
        **condition,
        "selector": selector,
        "requested_budget_fraction": requested_budget_fraction,
        "selected_cells": selected,
        "eligible_cells": eligible_count,
        "selected_fraction": selected / eligible_count if eligible_count else 0.0,
        "logits_kl": evaluation.logits_kl,
        "top1_agreement": evaluation.top1_agreement,
        "stale_logits_kl": stale_kl,
        "kl_gain_vs_stale": stale_kl - evaluation.logits_kl,
        "beneficial_vs_stale": evaluation.logits_kl <= stale_kl + 1e-15,
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
