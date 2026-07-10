from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.experiments.versioned import VersionedExperimentRunner, write_version_summary


def test_versioned_runner_writes_version_fields(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned",
            "experiment_id": "unit_e2",
            "seed": 1,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["full_recompute", "stale_reuse", "adaptive"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [0, 1, 2],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    assert artifacts.csv_path.exists()
    assert len(artifacts.records) == 9
    assert {record.adapter_version for record in artifacts.records} == {0, 1, 2}
    assert all(record.experiment_id == "unit_e2" for record in artifacts.records)
    output = tmp_path / "version_summary.csv"
    write_version_summary(artifacts.csv_path, output)
    assert output.exists()


def test_versioned_runner_preserves_random_update_drift(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned-drift",
            "experiment_id": "unit_drift",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 4, "hidden_size": 32, "vocab_size": 256},
            "data": {"task": "passkey", "num_samples": 8, "context_length": 128, "answer_length": 2},
            "updates": {"targets": ["lora.q", "lora.k", "lora.v"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["stale_reuse"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    means: dict[str, float] = {}
    for target in {record.update_target for record in artifacts.records}:
        values = [record.relative_error for record in artifacts.records if record.update_target == target]
        means[target] = sum(values) / len(values)
    assert len(set(round(value, 12) for value in means.values())) > 1


def test_global_layerwise_target_records_full_recompute(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-global-layerwise",
            "experiment_id": "unit_global_layerwise",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 4, "hidden_size": 16, "vocab_size": 32},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.1},
            "cache": {"strategies": ["layerwise_recompute"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    record = VersionedExperimentRunner(config).run().records[0]
    assert record.action == "full_recompute"
    assert record.first_invalid_layer is None
    assert record.recompute_fraction == 1.0


def test_versioned_runner_keeps_delta_correction_separate(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned-delta",
            "experiment_id": "unit_delta",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["adaptive", "delta_correction"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    assert {record.cache_strategy for record in artifacts.records} == {"adaptive", "delta_correction"}
    keys = [
        (record.sample_id, record.update_target, record.cache_strategy, record.adapter_version)
        for record in artifacts.records
    ]
    assert len(keys) == len(set(keys))


def test_versioned_runner_marks_zero_gap_reuse_exact(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-versioned-zero-gap",
            "experiment_id": "unit_zero_gap",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.01},
            "cache": {"strategies": ["adaptive", "delta_correction"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [0],
        }
    )
    artifacts = VersionedExperimentRunner(config).run()
    assert {record.action for record in artifacts.records} == {"reuse_exact"}
    assert {record.cache_state for record in artifacts.records} == {"valid_exact"}
    assert {record.version_gap for record in artifacts.records} == {0}


def test_no_adaptation_keeps_original_cache_version(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-no-adaptation",
            "experiment_id": "unit_no_adapt",
            "seed": 3,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "update_norm": 0.05},
            "cache": {"strategies": ["no_adaptation"]},
            "adapter": {"update_mode": "random"},
            "version_steps": [1, 2],
        }
    )
    records = VersionedExperimentRunner(config).run().records
    assert {record.action for record in records} == {"reuse_exact"}
    assert {record.cached_version for record in records} == {0}
    assert [record.version_gap for record in records] == [1, 2]
    assert {record.adaptation_latency for record in records} == {0.0}
    assert all(record.end_to_end_latency == record.latency_units for record in records)


def test_adapter_specific_cache_builds_each_unseen_version(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-adapter-cache",
            "experiment_id": "unit_adapter_cache",
            "seed": 3,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "update_norm": 0.05},
            "cache": {"strategies": ["adapter_specific_cache"]},
            "adapter": {"update_mode": "random"},
            "version_steps": [0, 1, 2],
        }
    )
    records = VersionedExperimentRunner(config).run().records
    assert [record.action for record in records] == ["reuse_exact", "full_recompute", "full_recompute"]


def test_oracle_selects_measured_safe_candidate(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-measured-oracle",
            "experiment_id": "unit_oracle",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 4, "hidden_size": 32, "vocab_size": 256},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["output_head"], "update_norm": 0.001},
            "cache": {"strategies": ["oracle_planner"]},
            "adapter": {"update_mode": "random"},
            "version_steps": [1],
        }
    )
    record = VersionedExperimentRunner(config).run().records[0]
    assert record.action == "reuse_stale"
    assert record.reason.startswith("Measured oracle selected")



def test_cached_version_initializes_cache_after_real_updates(tmp_path: Path) -> None:
    import pytest

    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-cached-version",
            "experiment_id": "unit_cached_version",
            "seed": 5,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "step_count": 1, "update_norm": 0.02},
            "cache": {"strategies": ["stale_reuse"]},
            "adapter": {"update_mode": "random"},
            "cached_version": 2,
            "version_steps": [2, 3],
        }
    )
    records = VersionedExperimentRunner(config).run().records
    assert [record.adapter_version for record in records] == [2, 3]
    assert records[0].action == "reuse_exact"
    assert records[0].cached_version == 2
    assert records[0].update_norm_since_cache == 0.0
    assert records[1].cached_version == 2
    assert records[1].update_norm_since_cache == pytest.approx(0.02)
    assert records[0].accumulated_update_norm == pytest.approx(0.04)
    assert records[1].accumulated_update_norm == pytest.approx(0.06)


def test_cached_version_rejects_backward_version_steps(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-cached-version-invalid",
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"]},
            "cache": {"strategies": ["stale_reuse"]},
            "cached_version": 2,
            "version_steps": [1, 2],
        }
    )
    import pytest

    with pytest.raises(ValueError, match="older than cached_version"):
        VersionedExperimentRunner(config).run()



def test_versioned_cache_capacity_is_global_across_samples_and_versions(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-capacity",
            "experiment_id": "unit_capacity",
            "seed": 3,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 16},
            "data": {"task": "passkey", "num_samples": 2, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "update_norm": 0.05},
            "cache": {"strategies": ["adapter_specific_cache"], "max_cache_entries": 2},
            "adapter": {"update_mode": "random"},
            "version_steps": [0, 1, 2],
        }
    )
    records = VersionedExperimentRunner(config).run().records
    assert max(record.cache_entry_count for record in records) <= 2
    assert records[-1].evicted_cache_entries > 0
    assert all(record.total_cache_bytes >= record.cache_bytes for record in records)



def test_related_work_baselines_use_distinct_cache_representations(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-related-work",
            "experiment_id": "unit_related_work",
            "seed": 5,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 4, "hidden_size": 16, "vocab_size": 32},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k:1"], "update_norm": 0.01},
            "cache": {"strategies": ["lragent_adapter_cache", "forkkv_base_delta"]},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    records = VersionedExperimentRunner(config).run().records
    by_strategy = {record.cache_strategy: record for record in records}
    lragent = by_strategy["lragent_adapter_cache"]
    forkkv = by_strategy["forkkv_base_delta"]
    assert lragent.action == "delta_correct"
    assert forkkv.action == "delta_correct"
    assert lragent.strategy_mode == "lragent_shared_base_plus_low_rank_component"
    assert forkkv.strategy_mode == "forkkv_copy_on_write_residual"
    assert lragent.baseline_fidelity == "paper_reimplementation"
    assert forkkv.baseline_fidelity == "paper_reimplementation"
    assert lragent.cache_bytes < forkkv.cache_bytes



def test_versioned_runner_records_failure_map_provenance(tmp_path: Path) -> None:
    failure_map = tmp_path / "failure_map.csv"
    failure_map.write_text(
        "update_target,version_gap,cache_strategy,task_drop_vs_full,logits_kl_mean,"
        "top1_agreement_mean,false_safe_rate\n"
        "lora.k:1,1,stale_reuse,0.0,0.001,1.0,0.0\n",
        encoding="utf-8",
    )
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-failure-map-provenance",
            "experiment_id": "unit_failure_map",
            "seed": 9,
            "output_dir": tmp_path / "run",
            "model": {"backend": "toy", "num_layers": 4, "hidden_size": 16, "vocab_size": 32},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k:1"], "update_norm": 0.01},
            "cache": {"strategies": ["adaptive"], "failure_map_path": failure_map},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    record = VersionedExperimentRunner(config).run().records[0]
    assert record.planner_source == "failure_map"
    assert record.failure_map_path == str(failure_map.resolve())
    assert len(record.failure_map_sha256) == 64


def test_versioned_runner_records_attention_shift_and_action_flops(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-attention-flops",
            "experiment_id": "unit_attention_flops",
            "seed": 9,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 4, "hidden_size": 16, "vocab_size": 32},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k:1"], "update_norm": 0.1},
            "cache": {"strategies": ["full_recompute", "stale_reuse", "layerwise_recompute"]},
            "metrics": {"compute_attention_metrics": True, "compute_flops_metrics": True},
            "adapter": {"update_mode": "random", "lora_rank": 4},
            "version_steps": [1],
        }
    )
    records = VersionedExperimentRunner(config).run().records
    by_strategy = {record.cache_strategy: record for record in records}
    full = by_strategy["full_recompute"]
    stale = by_strategy["stale_reuse"]
    partial = by_strategy["layerwise_recompute"]
    assert full.attention_shift == 0.0
    assert stale.attention_shift > 0.0
    assert full.strategy_flops == full.full_recompute_flops
    assert full.flops_fraction == 1.0
    assert stale.strategy_flops == 0.0
    assert partial.strategy_flops < partial.full_recompute_flops
    assert 0.0 < partial.flops_fraction < 1.0
