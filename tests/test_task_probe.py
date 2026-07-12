from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.experiments.task_probe import run_task_probe


def _config(tmp_path: Path) -> VersionedExperimentConfig:
    return VersionedExperimentConfig.model_validate(
        {
            "name": "task-probe-test",
            "output_dir": tmp_path / "unused",
            "model": {
                "backend": "toy",
                "num_layers": 4,
                "hidden_size": 16,
                "vocab_size": 128,
            },
            "data": {
                "source": "synthetic",
                "task": "passkey",
                "num_samples": 4,
                "context_length": 64,
                "answer_length": 4,
                "max_generation_tokens": 16,
                "selection_seed": 2027,
            },
        }
    )


def test_task_probe_writes_records_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    artifacts = run_task_probe(_config(tmp_path), output_dir=output_dir, max_samples=3)

    assert artifacts.summary.sample_count == 3
    assert artifacts.summary.mean_score == pytest.approx(1.0)
    assert artifacts.summary.degenerate_all_one is True
    assert artifacts.summary.degenerate_all_zero is False
    assert len(artifacts.records) == 3
    assert artifacts.records_jsonl.exists()
    assert artifacts.records_csv.exists()
    assert artifacts.summary_json.exists()

    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 3
    assert summary["perfect_fraction"] == pytest.approx(1.0)
    with artifacts.records_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["task_name"] for row in rows} == {"passkey"}


def test_task_probe_threshold_failure_keeps_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    with pytest.raises(RuntimeError, match="exceeds the required maximum"):
        run_task_probe(
            _config(tmp_path),
            output_dir=output_dir,
            max_samples=2,
            max_mean_score=0.9,
        )

    assert (output_dir / "task_probe.jsonl").exists()
    assert (output_dir / "task_probe.csv").exists()
    assert (output_dir / "task_probe_summary.json").exists()


def test_task_probe_fraction_thresholds_keep_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    with pytest.raises(RuntimeError, match="perfect fraction"):
        run_task_probe(
            _config(tmp_path),
            output_dir=output_dir,
            max_samples=2,
            max_perfect_fraction=0.5,
        )

    assert (output_dir / "task_probe.jsonl").exists()
    assert (output_dir / "task_probe.csv").exists()
    assert (output_dir / "task_probe_summary.json").exists()


def test_task_probe_rejects_invalid_thresholds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_task_probe(
            _config(tmp_path),
            output_dir=tmp_path / "probe",
            min_mean_score=-0.1,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        run_task_probe(
            _config(tmp_path),
            output_dir=tmp_path / "probe",
            min_mean_score=0.8,
            max_mean_score=0.2,
        )
