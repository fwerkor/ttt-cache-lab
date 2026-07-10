from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from ttt_cache_lab.cache.strategies import StrategyName


@lru_cache(maxsize=32)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planner_provenance(
    strategy: StrategyName,
    failure_map_path: Path | None,
) -> tuple[str, str, str]:
    if strategy is StrategyName.ORACLE_PLANNER:
        return ("measured_oracle", "", "")
    if strategy.value.startswith("adaptive"):
        if failure_map_path is None:
            return ("heuristic", "", "")
        resolved = failure_map_path.resolve()
        return ("failure_map", str(resolved), file_sha256(resolved))
    return ("fixed_policy", "", "")
