from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_formal_matrix() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "formal_matrix_20260712.py"
    spec = importlib.util.spec_from_file_location("formal_matrix_20260712", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_small_queue_ignores_global_model_shard_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_formal_matrix()
    monkeypatch.setenv("FORMAL_MODEL_PARALLELISM", "model_shard")
    assert module.queue_parallelism("small0") == "single"


def test_large_queue_respects_parallelism_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_formal_matrix()
    monkeypatch.setenv("FORMAL_MODEL_PARALLELISM", "single")
    assert module.queue_parallelism("seven13") == "single"
