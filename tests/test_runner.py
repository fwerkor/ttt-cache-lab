from pathlib import Path

from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.experiments.runner import ExperimentRunner


def test_runner_writes_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "unit",
            "seed": 1,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 2, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["attention.q", "lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["full_recompute", "adaptive"]},
        }
    )
    artifacts = ExperimentRunner(config).run()
    assert artifacts.jsonl_path.exists()
    assert artifacts.csv_path.exists()
    assert len(artifacts.records) == 8
