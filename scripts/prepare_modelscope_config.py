#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml


def _snapshot_download(model_id: str, cache_dir: Path) -> str:
    try:
        from modelscope import snapshot_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "ModelScope support requires the modelscope package. Install it in the Ascend environment with:\n"
            "  pip install modelscope\n"
            "Then rerun the Ascend script."
        ) from exc

    print(f"Downloading/resolving ModelScope model: {model_id}", file=sys.stderr)
    print(f"ModelScope cache dir: {cache_dir}", file=sys.stderr)
    local_path = snapshot_download(model_id, cache_dir=str(cache_dir))
    return str(local_path)


def _stable_name(config: Path, model_id: str) -> str:
    digest = hashlib.sha256(f"{config}:{model_id}".encode()).hexdigest()[:10]
    return f"{config.stem}.modelscope.{digest}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a ModelScope model and write a temporary config that loads the local snapshot."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/modelscope"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/modelscope_configs"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)

    model = payload.setdefault("model", {})
    if not isinstance(model, dict):
        raise SystemExit("Invalid config: model must be a mapping")

    model_id = model.get("modelscope_model_id") or model.get("model_name_or_path")
    if not model_id:
        raise SystemExit("Invalid config: model.modelscope_model_id or model.model_name_or_path is required")

    local_path = _snapshot_download(str(model_id), args.cache_dir)
    model["model_name_or_path"] = local_path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / _stable_name(args.config, str(model_id))
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    print(output)


if __name__ == "__main__":
    main()
