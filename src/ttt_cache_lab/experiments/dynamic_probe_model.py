from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ttt_cache_lab.cache.dynamic_probe_model import prompt_anchor_feature_values


@dataclass(frozen=True)
class _TargetTrainingPolicy:
    feature_names: tuple[str, ...]
    ridge: float
    positive_weight: float
    threshold: float
    selection_policy: str
    candidate_schedule: str


_DEFAULT_POLICIES = {
    "lora.k_middle": _TargetTrainingPolicy(
        feature_names=(
            "activated",
            "count",
            "worst",
            "mean",
            "median",
            "std",
            "improved_frac",
            "improved_1pct",
            "anchor_count",
            "anchor_max_mean",
            "anchor_max_max",
            "anchor_mean_mean",
            "block_mean",
            "block_span",
        ),
        ridge=1.0,
        positive_weight=0.5,
        threshold=0.6,
        selection_policy="largest_accepted_count",
        candidate_schedule="all",
    ),
    "lora.v_middle": _TargetTrainingPolicy(
        feature_names=(
            "activated",
            "count",
            "worst",
            "mean",
            "std",
            "improved_frac",
            "block_mean",
        ),
        ridge=0.3,
        positive_weight=0.5,
        threshold=0.55,
        selection_policy="max_probability",
        candidate_schedule="prefix2_plus_cap",
    ),
}


def fit_dynamic_probe_model(
    input_dirs: list[Path],
    *,
    output_path: Path,
) -> Path:
    rows_by_target: dict[str, list[tuple[dict[str, float], int]]] = {}
    source_files: list[str] = []
    condition_counts: dict[str, set[tuple[str, str]]] = {}

    for input_dir in input_dirs:
        records_path = input_dir / "blockwise_records.csv"
        if not records_path.exists():
            raise FileNotFoundError(f"Missing blockwise records: {records_path}")
        source_files.append(str(records_path))
        with records_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("selector") != "sparse_dynamic_controller":
                    continue
                if row.get("dynamic_probe_source") != "prompt_anchor":
                    continue
                trace_raw = row.get("dynamic_probe_trace", "")
                if not trace_raw:
                    continue
                target = str(row.get("update_target", ""))
                policy = _policy_for_target(target)
                if policy is None:
                    continue
                risk_score = float(row["continuous_risk_score"])
                control_score = float(row["continuous_control_score"])
                activated_risk = float(row["dynamic_activated_risk"])
                budget_cap = int(float(row["dynamic_budget_cap_cells"]))
                eligible_cells = int(float(row.get("eligible_cells", 0)))
                block_size = int(float(row.get("block_size", 1)))
                context_length = int(float(row.get("context_length", 0)))
                token_block_count = _infer_token_block_count(
                    eligible_cells=eligible_cells,
                    context_length=context_length,
                    block_size=block_size,
                )
                stale_kl = float(row["stale_logits_kl"])
                trace = json.loads(trace_raw)
                if not isinstance(trace, list):
                    raise ValueError("dynamic_probe_trace must be a list")
                condition_counts.setdefault(target, set()).add(
                    (str(records_path), str(row.get("sample_id", "")))
                )
                for point in trace:
                    if not isinstance(point, dict):
                        continue
                    features = prompt_anchor_feature_values(
                        point,
                        risk_score=risk_score,
                        control_score=control_score,
                        activated_risk=activated_risk,
                        budget_cap=budget_cap,
                        token_block_count=token_block_count,
                    )
                    candidate_kl = float(point["evaluation_only_logits_kl"])
                    label = int(candidate_kl < stale_kl - 1e-12)
                    rows_by_target.setdefault(target, []).append((features, label))

    targets: dict[str, Any] = {}
    for target, policy in _DEFAULT_POLICIES.items():
        rows = rows_by_target.get(target, [])
        if not rows:
            continue
        feature_matrix = np.asarray(
            [
                [float(features[name]) for name in policy.feature_names]
                for features, _ in rows
            ],
            dtype=np.float64,
        )
        labels = np.asarray([label for _, label in rows], dtype=np.float64)
        mean = feature_matrix.mean(axis=0)
        scale = feature_matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (feature_matrix - mean) / scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        coefficients = _fit_logistic(
            design,
            labels,
            ridge=policy.ridge,
            positive_weight=policy.positive_weight,
        )
        probabilities = _sigmoid(design @ coefficients)
        accepted = probabilities >= policy.threshold
        accepted_count = int(np.count_nonzero(accepted))
        accepted_precision = (
            float(np.mean(labels[accepted])) if accepted_count else math.nan
        )
        targets[target] = {
            "model_type": "logistic",
            "feature_names": list(policy.feature_names),
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "coefficients": coefficients.tolist(),
            "threshold": policy.threshold,
            "metadata": {
                "ridge": policy.ridge,
                "positive_weight": policy.positive_weight,
                "training_points": len(rows),
                "training_conditions": len(condition_counts.get(target, set())),
                "positive_rate": float(np.mean(labels)),
                "training_accept_count": accepted_count,
                "training_accept_precision": accepted_precision,
                "selection_policy": policy.selection_policy,
                "candidate_schedule": policy.candidate_schedule,
            },
        }

    if not targets:
        raise ValueError("No prompt-anchor calibration rows were found")
    payload = {
        "format_version": 1,
        "probe_source": "prompt_anchor",
        "targets": targets,
        "metadata": {
            "source_files": source_files,
            "label": "candidate_logits_kl_below_stale_logits_kl",
            "runtime_uses_kl": False,
            "selection_note": (
                "Hyperparameters were selected with grouped cross-validation on "
                "B24+B26 calibration conditions."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _policy_for_target(target: str) -> _TargetTrainingPolicy | None:
    direct = _DEFAULT_POLICIES.get(target)
    if direct is not None:
        return direct
    if ".k" in target:
        return _DEFAULT_POLICIES["lora.k_middle"]
    if ".v" in target:
        return _DEFAULT_POLICIES["lora.v_middle"]
    return None


def _infer_token_block_count(
    *,
    eligible_cells: int,
    context_length: int,
    block_size: int,
) -> int:
    from_context = math.ceil(context_length / max(block_size, 1))
    if from_context > 0:
        return from_context
    return max(eligible_cells, 1)


def _fit_logistic(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
    positive_weight: float,
    iterations: int = 100,
) -> np.ndarray:
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if positive_weight <= 0.0:
        raise ValueError("positive_weight must be positive")
    weights = np.where(labels > 0.5, positive_weight, 1.0)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    for _ in range(iterations):
        probabilities = _sigmoid(design @ coefficients)
        gradient = design.T @ (weights * (probabilities - labels))
        gradient += regularizer @ coefficients
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + regularizer
        hessian += np.eye(hessian.shape[0], dtype=np.float64) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return coefficients


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)
