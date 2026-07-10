import json
from pathlib import Path

import pytest

from ttt_cache_lab.configs import ExperimentConfig
from ttt_cache_lab.experiments.failures import capture_run_failure


def test_capture_run_failure_writes_structured_manifest(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "unit-failure",
            "output_dir": tmp_path,
            "model": {"backend": "toy"},
        }
    )

    def fail() -> None:
        raise RuntimeError("simulated OOM")

    with pytest.raises(RuntimeError, match="simulated OOM"):
        capture_run_failure(tmp_path, config, fail)

    payload = json.loads((tmp_path / "run_failure.json").read_text(encoding="utf-8"))
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_message"] == "simulated OOM"
    assert len(payload["config_sha256"]) == 64
    assert "simulated OOM" in payload["traceback"]


def test_successful_retry_removes_stale_failure_manifest(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "unit-retry",
            "output_dir": tmp_path,
            "model": {"backend": "toy"},
        }
    )
    failure_path = tmp_path / "run_failure.json"
    failure_path.write_text('{"error_type": "RuntimeError"}\n', encoding="utf-8")

    result = capture_run_failure(tmp_path, config, lambda: "completed")

    assert result == "completed"
    assert not failure_path.exists()
