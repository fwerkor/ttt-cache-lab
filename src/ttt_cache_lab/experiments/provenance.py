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


_RELATED_BASELINES = {
    StrategyName.ALORA_PREFIX_REUSE: (
        "paper_reimplementation",
        "aLoRA",
        "invocation-prefix adapter reuse semantics",
    ),
    StrategyName.LRAGENT_ADAPTER_CACHE: (
        "paper_reimplementation",
        "LRAgent",
        "shared base cache plus adapter low-rank component",
    ),
    StrategyName.FORKKV_BASE_DELTA: (
        "paper_reimplementation",
        "ForkKV",
        "copy-on-write base plus residual cache",
    ),
    StrategyName.BASE_CACHE_REUSE: (
        "adapted_baseline",
        "static adapter reuse",
        "base-prefix cache reuse control",
    ),
    StrategyName.ADAPTER_SPECIFIC_CACHE: (
        "adapted_baseline",
        "per-adapter cache",
        "dedicated cache entry per fixed adapter",
    ),
    StrategyName.STATIC_BASE_DELTA: (
        "adapted_baseline",
        "base-plus-delta cache",
        "static base and adapter delta decomposition",
    ),
}


def baseline_provenance(
    strategy: StrategyName,
    observed_fidelity: str,
) -> tuple[str, str, str]:
    configured = _RELATED_BASELINES.get(strategy)
    if configured is not None:
        fidelity, source, reference = configured
        return (observed_fidelity or fidelity, source, reference)
    return (observed_fidelity or "native_baseline", strategy.value, "local implementation")


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
