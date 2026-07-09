from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.experiments.versioned import VersionedExperimentRunner, write_version_summary


def test_versioned_runner_writes_version_fields(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned",
            "experiment_id": "unit_e2",
            "seed": 1,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["full_recompute", "stale_reuse", "adaptive"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [0, 1, 2],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    assert artifacts.csv_path.exists()
    assert len(artifacts.records) == 9
    assert {record.adapter_version for record in artifacts.records} == {0, 1, 2}
    assert all(record.experiment_id == "unit_e2" for record in artifacts.records)
    output = tmp_path / "version_summary.csv"
    write_version_summary(artifacts.csv_path, output)
    assert output.exists()


def test_versioned_runner_preserves_random_update_drift(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned-drift",
            "experiment_id": "unit_drift",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 4, "hidden_size": 32, "vocab_size": 256},
            "data": {"task": "passkey", "num_samples": 8, "context_length": 128, "answer_length": 2},
            "updates": {"targets": ["lora.q", "lora.k", "lora.v"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["stale_reuse"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    means: dict[str, float] = {}
    for target in {record.update_target for record in artifacts.records}:
        values = [record.relative_error for record in artifacts.records if record.update_target == target]
        means[target] = sum(values) / len(values)
    assert len(set(round(value, 12) for value in means.values())) > 1


def test_versioned_runner_keeps_delta_correction_separate(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned-delta",
            "experiment_id": "unit_delta",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["adaptive", "delta_correction"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    assert {record.cache_strategy for record in artifacts.records} == {"adaptive", "delta_correction"}
    keys = [
        (record.sample_id, record.update_target, record.cache_strategy, record.adapter_version)
        for record in artifacts.records
    ]
    assert len(keys) == len(set(keys))


def test_versioned_runner_marks_zero_gap_reuse_exact(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned-zero-gap",
            "experiment_id": "unit_zero_gap",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["adaptive", "delta_correction"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [0],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    assert {record.action for record in artifacts.records} == {"reuse_exact"}
    assert {record.cache_state for record in artifacts.records} == {"valid_exact"}
    assert {record.version_gap for record in artifacts.records} == {0}
