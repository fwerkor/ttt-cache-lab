import csv
from pathlib import Path

from ttt_cache_lab.configs import SweepConfig, VersionedSweepConfig
from ttt_cache_lab.experiments.sweep import run_sweep, run_versioned_sweep


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
    with artifacts.merged_records_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["sweep.updates.update_norm"] for row in rows} == {"0.01", "0.02"}



def test_versioned_sweep_expands_rank_and_update_norm(tmp_path: Path) -> None:
    config = VersionedSweepConfig.model_validate(
        {
            "name": "unit-versioned-sweep",
            "output_dir": tmp_path,
            "base": {
                "name": "base",
                "experiment_id": "e5",
                "seed": 1,
                "output_dir": tmp_path / "unused",
                "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
                "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
                "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
                "cache": {"strategies": ["full_recompute", "delta_correction"]},
                "adapter": {"update_mode": "random", "lora_rank": 4},
                "version_steps": [0, 1],
            },
            "axes": [
                {"path": "adapter.lora_rank", "values": [4, 8]},
                {"path": "updates.update_norm", "values": [0.01, 0.02]},
            ],
        }
    )
    assert len(config.expand()) == 4
    artifacts = run_versioned_sweep(config)
    assert len(artifacts.run_dirs) == 4
    assert artifacts.merged_records_csv.exists()
    assert artifacts.grouped_csv.exists()
    with artifacts.merged_records_csv.open(newline="", encoding="utf-8") as handle:
        merged = list(csv.DictReader(handle))
    assert {row["sweep.adapter.lora_rank"] for row in merged} == {"4", "8"}
    assert {row["sweep.updates.update_norm"] for row in merged} == {"0.01", "0.02"}
    with artifacts.grouped_csv.open(newline="", encoding="utf-8") as handle:
        grouped = list(csv.DictReader(handle))
    assert len(grouped) == 16
    assert len({row["run_name"] for row in grouped}) == 4
    assert {row["sweep.adapter.lora_rank"] for row in grouped} == {"4", "8"}
    assert {row["sweep.updates.update_norm"] for row in grouped} == {"0.01", "0.02"}
