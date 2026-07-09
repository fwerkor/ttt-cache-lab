from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.experiments.report import generate_report
from ttt_cache_lab.experiments.versioned import VersionedExperimentRunner


def test_generate_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-report",
            "experiment_id": "unit_report",
            "seed": 1,
            "output_dir": run_dir,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["full_recompute", "stale_reuse"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [0, 1],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    report = generate_report(artifacts.csv_path, tmp_path / "report")
    assert report.exists()
    assert (tmp_path / "report" / "logits_kl_by_version.svg").exists()
