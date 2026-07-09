from pathlib import Path

from ttt_cache_lab.configs import SweepConfig
from ttt_cache_lab.experiments.sweep import run_sweep


def test_sweep_expands_and_runs(tmp_path: Path) -> None:
    config = SweepConfig.model_validate(
        {
            "name": "unit-sweep",
            "output_dir": tmp_path,
            "base": {
                "name": "base",
                "seed": 1,
                "output_dir": tmp_path / "unused",
                "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
                "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
                "updates": {"targets": ["attention.q"], "step_count": 1, "update_norm": 0.01},
                "cache": {"strategies": ["full_recompute", "stale_reuse"]},
            },
            "axes": [{"path": "updates.update_norm", "values": [0.01, 0.02]}],
        }
    )
    assert len(config.expand()) == 2
    artifacts = run_sweep(config)
    assert artifacts.merged_records_csv.exists()
    assert artifacts.grouped_csv.exists()
