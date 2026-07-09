from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ttt_cache_lab.models.interface import BackendOutput
from ttt_cache_lab.updates.targets import UpdateTarget


@dataclass(frozen=True)
class UpdateResult:
    output: BackendOutput
    update_norm: float
    step_count: int


class TTTUpdater(Protocol):
    def update(
        self,
        baseline: BackendOutput,
        target: UpdateTarget,
        *,
        step_count: int,
        update_norm: float,
    ) -> UpdateResult: ...
