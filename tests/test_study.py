from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ttt_cache_lab.experiments.study import (
    expand_study,
    run_study_job,
    select_study_jobs,
    write_study_plan,
)


def _manifest(tmp_path: Path) -> Path:
    config = tmp_path / "toy.yaml"
    config.write_text(
        """
name: toy-study
output_dir: ignored
experiment_id: e2
model:
  backend: toy
data:
  task: passkey
  num_samples: 1
  context_length: 64
  answer_length: 2
updates:
  targets: [lora.q]
cache:
  strategies: [full_recompute, stale_reuse]
adapter:
  update_mode: random
version_steps: [0, 1]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "study.yaml"
    manifest.write_text(
        f"""
name: test-study
output_dir: {tmp_path / 'runs'}
jobs:
  - name: calibration
    config: {config}
    seeds: [3, 5]
    tags: [calibration, toy]
  - name: heldout
    config: {config}
    seeds: [7]
    tags: [test, toy]
    required_paths: [{tmp_path / 'calibration.done'}]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_study_expansion_and_sharding_are_stable(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _, jobs = expand_study(manifest)
    assert [(job.index, job.name, job.seed) for job in jobs] == [
        (0, "calibration", 3),
        (1, "calibration", 5),
        (2, "heldout", 7),
    ]
    assert [job.index for job in select_study_jobs(manifest, tag="calibration")] == [0, 1]
    assert [job.index for job in select_study_jobs(manifest, shard_index=1, num_shards=2)] == [1]


def test_study_plan_writes_job_matrix_and_commands(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    matrix, commands = write_study_plan(manifest, tmp_path / "plan")
    with matrix.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[2]["name"] == "heldout"
    assert "--job-index 2" in commands.read_text(encoding="utf-8")


def test_study_job_requires_declared_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    job = select_study_jobs(manifest, job_index=2)[0]
    with pytest.raises(FileNotFoundError, match="missing required artifacts"):
        run_study_job(job)
