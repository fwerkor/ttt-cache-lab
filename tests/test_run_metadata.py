from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ttt_cache_lab.experiments.run_metadata import _git_state


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_git_state_ignores_untracked_artifacts_but_detects_tracked_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")
    expected_commit = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "result.jsonl").write_text("{}\n", encoding="utf-8")
    assert _git_state() == (expected_commit, False)

    tracked.write_text("modified\n", encoding="utf-8")
    assert _git_state() == (expected_commit, True)
