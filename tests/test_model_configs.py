from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig, VersionedSweepConfig

REAL_MODEL_CONFIGS = [
    Path("configs/experiments/e2_version_drift_qwen_0_5b.yaml"),
    Path("configs/experiments/e2_version_drift_qwen_1_5b.yaml"),
    Path("configs/experiments/e2_version_drift_qwen_7b.yaml"),
    Path("configs/experiments/e2_version_drift_llama_3_2_1b.yaml"),
    Path("configs/experiments/e2_version_drift_mistral_7b_v0_3.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_qwen_0_5b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_qwen_7b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_llama_3_2_1b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_mistral_7b_v0_3.yaml"),
    Path("configs/experiments/e5_delta_correction_qwen_0_5b.yaml"),
    Path("configs/experiments/ascend_e5_delta_correction_qwen_0_5b.yaml"),
    Path("configs/experiments/ascend_e6_scaling_qwen_1_5b_4k.yaml"),
    Path("configs/experiments/ascend_e6_scaling_qwen_1_5b_8k.yaml"),
    Path("configs/experiments/ascend_e6_scaling_qwen_7b_16k.yaml"),
]


def test_real_model_configs_load_without_model_downloads() -> None:
    for path in REAL_MODEL_CONFIGS:
        config = VersionedExperimentConfig.from_yaml(path)
        assert config.model.model_name_or_path
        assert config.adapter.update_mode == "lora_train"
        assert "stale_reuse" in config.cache.strategies
        assert set(config.updates.targets) & {"lora.k", "lora.k_late"}
        assert set(config.updates.targets) & {"lora.v", "lora.v_late"}


def test_all_versioned_experiment_configs_parse() -> None:
    for path in Path("configs/experiments").glob("*.yaml"):
        VersionedExperimentConfig.from_yaml(path)


def test_ascend_configs_have_modelscope_ids() -> None:
    for path in REAL_MODEL_CONFIGS:
        config = VersionedExperimentConfig.from_yaml(path)
        if config.model.backend == "ascend_hf":
            assert config.model.modelscope_model_id


def test_default_ascend_parallel_excludes_large_manual_templates() -> None:
    script = Path("scripts/run_ascend_e2_parallel.sh").read_text(encoding="utf-8")
    assert "ascend_e2_version_drift_qwen_7b.yaml" not in script
    assert "ascend_e2_version_drift_mistral_7b_v0_3.yaml" not in script
    assert "ascend_e2_version_drift_llama_3_2_1b.yaml" in script


def test_ascend_7b_uses_four_npu_model_sharding() -> None:
    config = VersionedExperimentConfig.from_yaml(
        Path("configs/experiments/ascend_e2_version_drift_qwen_7b.yaml")
    )
    assert config.model.device == "npu"
    assert config.model.parallelism == "model_shard"
    assert config.model.device_ids == [0, 1, 2, 3]


def test_attention_metric_sweep_uses_eager_attention() -> None:
    config = VersionedSweepConfig.from_yaml(
        Path("configs/versioned_sweep_e7_boundary_qwen_0_5b.yaml")
    )
    assert config.base.model.attention_implementation == "eager"


def test_llama_templates_use_small_1b_model() -> None:
    hf = VersionedExperimentConfig.from_yaml(Path("configs/experiments/e2_version_drift_llama_3_2_1b.yaml"))
    ascend = VersionedExperimentConfig.from_yaml(Path("configs/experiments/ascend_e2_version_drift_llama_3_2_1b.yaml"))
    assert hf.model.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct"
    assert ascend.model.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct"
    assert ascend.model.modelscope_model_id == "LLM-Research/Llama-3.2-1B-Instruct"
