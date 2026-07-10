from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ttt_cache_lab.experiments.run_metadata import collect_run_metadata


def capture_run_failure[T](
    output_dir: Path,
    config: BaseModel,
    operation: Callable[[], T],
) -> T:
    try:
        return operation()
    except Exception as exc:
        write_run_failure(output_dir, config=config, error=exc)
        raise


def write_run_failure(
    output_dir: Path,
    *,
    config: BaseModel,
    error: BaseException,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_failure.json"
    temporary = path.with_name(f".{path.name}.tmp")
    metadata = collect_run_metadata(config)
    payload: dict[str, Any] = {
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "config_sha256": metadata["config_sha256"],
        "git_commit": metadata["git_commit"],
        "config": metadata["config"],
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path
