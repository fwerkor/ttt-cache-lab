from __future__ import annotations

import pytest

from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.models.toy import ToyBackend
from ttt_cache_lab.updates.targets import parse_update_target
from ttt_cache_lab.updates.updater import RandomPerturbationUpdater, build_updater


def test_random_updater_executes_all_requested_steps() -> None:
    backend = ToyBackend(num_layers=2, hidden_size=8, vocab_size=16, seed=7)
    sample = TaskSample(prompt="key 1", answer="1", metadata={})
    baseline = backend.prefill(sample.prompt)
    target = parse_update_target("attention.k", num_layers=backend.num_layers)
    result = RandomPerturbationUpdater(backend).update(
        baseline,
        target,
        step_count=3,
        update_norm=0.02,
    )
    assert result.output.parameter_version == 3
    assert result.step_count == 3
    assert result.update_norm == 0.06
    assert result.adaptation_latency == 0.0


def test_lora_mode_uses_random_updater_for_non_lora_controls() -> None:
    backend = ToyBackend(num_layers=2, hidden_size=8, vocab_size=16, seed=7)
    sample = TaskSample(prompt="key 1", answer="1", metadata={})
    target = parse_update_target("norm:1", num_layers=backend.num_layers)
    updater = build_updater(backend, mode="lora_train", sample=sample, target=target)
    result = updater.update(backend.prefill(sample.prompt), target, step_count=1, update_norm=0.01)
    assert result.output.parameter_version == 1



def test_supervised_lora_updater_honors_target_update_norm() -> None:
    backend = ToyBackend(num_layers=2, hidden_size=8, vocab_size=16, seed=7)
    sample = TaskSample(prompt="key 1", answer="1", metadata={})
    target = parse_update_target("lora.k:1", num_layers=backend.num_layers)
    updater = build_updater(
        backend,
        mode="lora_train",
        sample=sample,
        target=target,
        rank=4,
        learning_rate=0.5,
    )
    result = updater.update(
        backend.prefill(sample.prompt),
        target,
        step_count=2,
        update_norm=0.03,
    )
    assert result.update_norm == 0.06
    assert result.raw_update_norm == pytest.approx(0.45)
    assert result.update_scale == result.update_norm / result.raw_update_norm
    assert result.step_count == 2


def test_supervised_lora_updater_can_keep_raw_learning_rate_update() -> None:
    backend = ToyBackend(num_layers=2, hidden_size=8, vocab_size=16, seed=7)
    sample = TaskSample(prompt="key 1", answer="1", metadata={})
    target = parse_update_target("lora.k:1", num_layers=backend.num_layers)
    updater = build_updater(
        backend,
        mode="lora_train",
        sample=sample,
        target=target,
        rank=4,
        learning_rate=0.5,
        norm_control="none",
    )
    result = updater.update(
        backend.prefill(sample.prompt),
        target,
        step_count=2,
        update_norm=0.03,
    )
    assert result.raw_update_norm == pytest.approx(0.45)
    assert result.update_norm == pytest.approx(result.raw_update_norm)
    assert result.update_scale == 1.0
