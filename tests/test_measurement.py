from __future__ import annotations

import gc
import weakref

import numpy as np

from ttt_cache_lab.experiments.measurement import measure_backend_call
from ttt_cache_lab.models.interface import BackendOutput


def test_measure_backend_call_excludes_warmups_and_reports_percentiles() -> None:
    calls = 0

    def execute() -> BackendOutput:
        nonlocal calls
        calls += 1
        latency = float(calls)
        return BackendOutput(
            logits=np.zeros((1, 2)),
            cache_tensor=np.zeros((1, 2)),
            hidden_tensor=np.zeros((1, 2)),
            parameter_version=0,
            extras={
                "strategy_latency": latency,
                "decode_latency": latency / 2.0,
                "cache_maintenance_latency": latency / 4.0,
            },
        )

    measured = measure_backend_call(
        execute,
        warmup_runs=2,
        timed_runs=3,
        fallback_latency=99.0,
    )
    assert calls == 5
    assert measured.latency_mean == 4.0
    assert measured.latency_p50 == 4.0
    assert measured.latency_p95 == 4.9
    assert measured.decode_latency_p50 == 2.0
    assert measured.cache_maintenance_latency_p50 == 1.0
    assert measured.output.extras is not None
    assert measured.output.extras["strategy_latency"] == 5.0


def test_measure_backend_call_does_not_retain_every_timed_output() -> None:
    references: list[weakref.ReferenceType[BackendOutput]] = []

    def execute() -> BackendOutput:
        output = BackendOutput(
            logits=np.zeros((1, 2)),
            cache_tensor=np.zeros((128, 128)),
            hidden_tensor=np.zeros((128, 128)),
            parameter_version=0,
            extras={"strategy_latency": 1.0},
        )
        references.append(weakref.ref(output))
        return output

    measured = measure_backend_call(execute, warmup_runs=2, timed_runs=5, fallback_latency=1.0)
    gc.collect()
    assert measured.output is references[-1]()
    assert sum(reference() is not None for reference in references) == 1
