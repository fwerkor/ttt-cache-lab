from __future__ import annotations

from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.experiments.study import expand_study


def test_all_paper_configs_parse_and_use_fixed_dataset_selection() -> None:
    paths = sorted(Path("configs/paper").glob("*/*.yaml"))
    assert len(paths) >= 40
    configs = [VersionedExperimentConfig.from_yaml(path) for path in paths]
    assert all(config.data.evaluation_partition in {"calibration", "validation", "test"} for config in configs)
    assert all(config.data.selection_seed > 0 for config in configs)
    assert all(config.adapter.update_mode in {"lora_train", "static_lora"} for config in configs)
    assert any(config.experiment_id == "e1_static_adapter_baseline" for config in configs)
    assert any(config.experiment_id == "e2_version_drift" for config in configs)
    assert any(config.experiment_id == "e5_delta_correction" for config in configs)
    assert all(config.updates.update_norm > 0.0 for config in configs)


def test_paper_matrix_contains_large_models_real_tasks_and_cache_pressure() -> None:
    paths = sorted(Path("configs/paper").glob("*/*.yaml"))
    configs = [VersionedExperimentConfig.from_yaml(path) for path in paths]
    model_names = {config.model.model_name_or_path for config in configs}
    assert "Qwen/Qwen2.5-7B-Instruct" in model_names
    assert "Qwen/Qwen2.5-14B-Instruct" in model_names
    assert "Qwen/Qwen2.5-32B-Instruct" in model_names
    assert "mistralai/Mistral-7B-Instruct-v0.3" in model_names
    assert any(config.data.benchmark_name == "LongBench-v2" for config in configs)
    assert any(config.data.benchmark_name == "LongBench" for config in configs)
    assert any(config.experiment_id == "e8_cache_pressure" for config in configs)
    assert any(config.model.max_length == 65536 for config in configs)


def test_controlled_calibration_covers_six_tasks_at_every_qwen_scale() -> None:
    expected_tasks = {
        "multi_needle",
        "needle_absent",
        "multi_hop_tracing",
        "aggregation",
        "common_words",
        "variable_tracking",
    }
    for model_key in ("qwen_1_5b", "qwen_7b", "qwen_14b", "qwen_32b"):
        paths = sorted(Path("configs/paper/calibration").glob(f"e3_{model_key}_*.yaml"))
        configs = [VersionedExperimentConfig.from_yaml(path) for path in paths]
        assert {config.data.task for config in configs} == expected_tasks


def test_synthetic_paper_configs_use_explicit_nontruncating_generation_budgets() -> None:
    paths = sorted(Path("configs/paper").glob("*/*.yaml"))
    configs = [VersionedExperimentConfig.from_yaml(path) for path in paths]
    synthetic = [config for config in configs if config.data.source == "synthetic"]
    assert synthetic
    assert all(config.data.max_generation_tokens >= 32 for config in synthetic)
    assert all(
        config.data.max_generation_tokens >= config.data.answer_length
        for config in synthetic
    )


def test_needle_absence_uses_answer_prefix_scoring() -> None:
    paths = sorted(Path("configs/paper/calibration").glob("e3_qwen_*_needle_absent.yaml"))
    assert len(paths) == 4
    configs = [VersionedExperimentConfig.from_yaml(path) for path in paths]
    assert all(config.data.scorer == "prefix_match" for config in configs)


def test_controlled_calibration_uses_frozen_model_specific_viability_cells() -> None:
    expected = {
        "qwen_1_5b": {
            "context_length": 4096,
            "difficulties": {
                "multi_needle": "medium",
                "needle_absent": "hard",
                "multi_hop_tracing": "easy",
                "aggregation": "easy",
                "common_words": "easy",
                "variable_tracking": "easy",
            },
        },
        "qwen_7b": {
            "context_length": 8192,
            "difficulties": {
                "multi_needle": "hard",
                "needle_absent": "hard",
                "multi_hop_tracing": "medium",
                "aggregation": "medium",
                "common_words": "hard",
                "variable_tracking": "medium",
            },
        },
        "qwen_14b": {
            "context_length": 16384,
            "difficulties": {
                "multi_needle": "hard",
                "needle_absent": "hard",
                "multi_hop_tracing": "medium",
                "aggregation": "hard",
                "common_words": "hard",
                "variable_tracking": "hard",
            },
        },
        "qwen_32b": {
            "context_length": 16384,
            "difficulties": {
                "multi_needle": "hard",
                "needle_absent": "hard",
                "multi_hop_tracing": "hard",
                "aggregation": "hard",
                "common_words": "hard",
                "variable_tracking": "hard",
            },
        },
    }
    for model_key, model_expected in expected.items():
        paths = sorted(Path("configs/paper/calibration").glob(f"e3_{model_key}_*.yaml"))
        configs = [VersionedExperimentConfig.from_yaml(path) for path in paths]
        assert all(
            config.data.context_length == model_expected["context_length"]
            for config in configs
        )
        assert {
            config.data.task: config.data.synthetic_difficulty for config in configs
        } == model_expected["difficulties"]


def test_longbench_v2_partitions_are_disjoint_for_qwen_7b() -> None:
    validation = VersionedExperimentConfig.from_yaml(
        Path("configs/paper/validation/e4_qwen_7b_longbench_v2_validation.yaml")
    )
    test = VersionedExperimentConfig.from_yaml(
        Path("configs/paper/test/e4_qwen_7b_longbench_v2_test.yaml")
    )
    ablation = VersionedExperimentConfig.from_yaml(
        Path("configs/paper/ablation/e7_qwen_7b_longbench_v2.yaml")
    )
    intervals = [
        range(validation.data.sample_offset, validation.data.sample_offset + validation.data.num_samples),
        range(test.data.sample_offset, test.data.sample_offset + test.data.num_samples),
        range(ablation.data.sample_offset, ablation.data.sample_offset + ablation.data.num_samples),
    ]
    sets = [set(interval) for interval in intervals]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_study_manifest_expands_every_config_to_three_seeds() -> None:
    _, jobs = expand_study(Path("configs/paper/study.yaml"))
    config_count = len(list(Path("configs/paper").glob("*/*.yaml")))
    assert len(jobs) == config_count * 3
    assert {job.seed for job in jobs} == {7, 17, 29}
    dependent_stages = {"validation", "test", "delta", "scaling", "ablation", "workload"}
    assert all(job.required_paths for job in jobs if dependent_stages.intersection(job.tags))
    assert all(not job.required_paths for job in jobs if {"baseline", "calibration", "drift"}.intersection(job.tags))
