from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MAINLINE_TAGS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "a1", "w1", "w2", "w4")
_RECOMMENDED_SYNTHETIC_SCORERS = {
    "passkey": "numeric_match",
    "key_value": "contains",
    "multi_needle": "contains",
    "variable_tracking": "contains",
    "multi_hop_tracing": "contains",
    "needle_absent": "prefix_match",
    "aggregation": "prefix_match",
    "common_words": "set_f1",
}


def _mainline_configs() -> list[tuple[Path, dict[str, Any]]]:
    root = Path(__file__).parents[1] / "configs"
    configs: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = payload.get("base", payload)
        experiment_id = str(config.get("experiment_id", "")).lower()
        if not any(tag in experiment_id for tag in _MAINLINE_TAGS):
            continue
        model = config.get("model", {}) or {}
        if model.get("backend", "toy") == "toy":
            continue
        configs.append((path, config))
    return configs


def test_non_toy_mainline_configs_enable_task_viability_gate() -> None:
    missing = []
    for path, config in _mainline_configs():
        viability = config.get("task_viability", {}) or {}
        if not viability.get("enabled", False):
            missing.append(str(path))
    assert not missing, f"Task viability gate is disabled for: {missing}"


def test_non_toy_mainline_synthetic_tasks_use_robust_scorers_and_difficulty() -> None:
    failures = []
    for path, config in _mainline_configs():
        data = config.get("data", {}) or {}
        if data.get("source", "synthetic") != "synthetic":
            continue
        task = str(data.get("task", ""))
        expected = _RECOMMENDED_SYNTHETIC_SCORERS.get(task)
        if expected is not None and data.get("scorer", "exact_match") != expected:
            failures.append(f"{path}: scorer={data.get('scorer', 'exact_match')} expected={expected}")
        if "synthetic_difficulty" not in data:
            failures.append(f"{path}: missing synthetic_difficulty")
    assert not failures, "\n".join(failures)


def test_controlled_e2_tasks_are_model_calibrated() -> None:
    failures = []
    for path, config in _mainline_configs():
        experiment_id = str(config.get("experiment_id", "")).lower()
        data = config.get("data", {}) or {}
        if "e2" not in experiment_id or data.get("source", "synthetic") != "synthetic":
            continue
        model = config.get("model", {}) or {}
        model_name = str(model.get("model_name_or_path") or model.get("modelscope_model_id") or "").lower()
        if "32b" in model_name or "14b" in model_name:
            expected = ("common_words", "set_f1", "hard")
        elif any(size in model_name for size in ("7b", "4b", "3b", "a2.7b")):
            expected = ("variable_tracking", "contains", "easy")
        else:
            expected = ("passkey", "numeric_match", "easy")
        actual = (
            data.get("task"),
            data.get("scorer", "exact_match"),
            data.get("synthetic_difficulty"),
        )
        if actual != expected:
            failures.append(f"{path}: task/scorer/difficulty={actual} expected={expected}")
    assert not failures, "\n".join(failures)


def test_paper_e2_ability_track_uses_held_out_samples_and_meaningful_update_scale() -> None:
    failures = []
    for path, config in _mainline_configs():
        if "configs/paper/drift" not in str(path):
            continue
        experiment_id = str(config.get("experiment_id", "")).lower()
        if "e2" not in experiment_id:
            continue
        data = config.get("data", {}) or {}
        updates = config.get("updates", {}) or {}
        if int(data.get("sample_offset", 0)) < 96:
            failures.append(f"{path}: sample_offset must be at least 96")
        if float(updates.get("update_norm", 0.0)) < 1e-3:
            failures.append(f"{path}: update_norm must be at least 1e-3")
    assert not failures, "\n".join(failures)


def test_non_toy_mainline_sample_counts_have_statistical_floor() -> None:
    failures = []
    for path, config in _mainline_configs():
        data = config.get("data", {}) or {}
        samples = int(data.get("num_samples", 0))
        context = int(data.get("context_length", 0))
        experiment_id = str(config.get("experiment_id", "")).lower()
        source = data.get("source", "synthetic")
        if source == "huggingface":
            minimum = 96
        elif any(tag in experiment_id for tag in ("w1", "w2", "w4")) or "e6" in experiment_id and context >= 32768:
            minimum = 16
        else:
            minimum = 24
        if samples < minimum:
            failures.append(f"{path}: num_samples={samples} minimum={minimum}")
    assert not failures, "\n".join(failures)
