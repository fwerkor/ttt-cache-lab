from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ttt_cache_lab.models.interface import BackendOutput

_SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
_PACKAGE_NAMES = (
    "ttt-cache-lab",
    "numpy",
    "pydantic",
    "PyYAML",
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "modelscope",
    "torch-npu",
)


def collect_run_metadata(config: BaseModel) -> dict[str, Any]:
    config_payload = config.model_dump(mode="json")
    encoded_config = json.dumps(
        config_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    commit, dirty = _git_state()
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config_payload,
        "config_sha256": hashlib.sha256(encoded_config).hexdigest(),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            package: version
            for package in _PACKAGE_NAMES
            if (version := _package_version(package)) is not None
        },
        "environment": {
            key: os.environ[key]
            for key in _SAFE_ENVIRONMENT_KEYS
            if key in os.environ
        },
    }


def record_run_fields(
    config: BaseModel,
    output: BackendOutput,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    model = payload.get("model", {})
    data = payload.get("data", {})
    extras = output.extras or {}
    return {
        "seed": int(payload.get("seed", 0)),
        "task_name": str(data.get("task", "")),
        "backend_name": str(model.get("backend", "")),
        "torch_dtype": str(model.get("torch_dtype", "")),
        "attention_implementation": str(
            extras.get("attention_implementation", "")
        ),
        "git_commit": str(metadata.get("git_commit", "")),
        "run_config_sha256": str(metadata.get("config_sha256", "")),
    }


def write_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.SubprocessError):
        return "", False
