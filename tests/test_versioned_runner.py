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
