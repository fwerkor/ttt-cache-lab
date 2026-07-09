from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig

REAL_MODEL_CONFIGS = [
    Path("configs/experiments/e2_version_drift_qwen_0_5b.yaml"),
    Path("configs/experiments/e2_version_drift_qwen_1_5b.yaml"),
    Path("configs/experiments/e2_version_drift_qwen_7b.yaml"),
    Path("configs/experiments/e2_version_drift_llama_3_1_8b.yaml"),
    Path("configs/experiments/e2_version_drift_mistral_7b_v0_3.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_qwen_0_5b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_qwen_7b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_llama_3_1_8b.yaml"),
    Path("configs/experiments/ascend_e2_version_drift_mistral_7b_v0_3.yaml"),
]


def test_real_model_configs_load_without_model_downloads() -> None:
    for path in REAL_MODEL_CONFIGS:
        config = VersionedExperimentConfig.from_yaml(path)
        assert config.model.model_name_or_path
        assert config.adapter.update_mode == "lora_train"
        assert "stale_reuse" in config.cache.strategies
        assert set(config.updates.targets) >= {"lora.q", "lora.k", "lora.v"}


def test_ascend_configs_have_modelscope_ids() -> None:
    for path in REAL_MODEL_CONFIGS:
        config = VersionedExperimentConfig.from_yaml(path)
        if config.model.backend == "ascend_hf":
            assert config.model.modelscope_model_id
