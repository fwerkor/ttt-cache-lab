from __future__ import annotations

from enum import StrEnum


class CacheSemantics(StrEnum):
    EXACT_CURRENT = "exact_current"
    FROZEN_EVIDENCE = "frozen_evidence"
    BOUNDED_STALE = "bounded_stale"


class CacheBlockState(StrEnum):
    VALID_EXACT = "valid_exact"
    VALID_FROZEN = "valid_frozen"
    VALID_APPROX = "valid_approx"
    INVALID = "invalid"


class CacheAction(StrEnum):
    REUSE_EXACT = "reuse_exact"
    REUSE_FROZEN = "reuse_frozen"
    REUSE_STALE = "reuse_stale"
    DELTA_CORRECT = "delta_correct"
    PARTIAL_RECOMPUTE = "partial_recompute"
    ALORA_SUFFIX_RECOMPUTE = "alora_suffix_recompute"
    FULL_RECOMPUTE = "full_recompute"
    REJECT_UPDATE = "reject_update"
