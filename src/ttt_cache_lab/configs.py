from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    backend: Literal["toy", "hf", "ascend_hf"] = "toy"
    model_name_or_path: str | None = None
    modelscope_model_id: str | None = None
    revision: str | None = None
    attention_implementation: str | None = None
    num_layers: int = 4
    hidden_size: int = 32
    vocab_size: int = 256
    device: str = "auto"
    torch_dtype: str = "auto"
    max_length: int = 2048
    trust_remote_code: bool = False
    parallelism: Literal["single", "model_shard"] = "single"
    device_ids: list[int] = Field(default_factory=list)


class DataConfig(BaseModel):
    source: Literal["synthetic", "jsonl", "huggingface"] = "synthetic"
    task: str = "passkey"
    num_samples: int = 16
    context_length: int = 512
    answer_length: int = 4
    dataset_path: Path | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "test"
    prompt_field: str = "prompt"
    answer_field: str = "answer"
    scorer: Literal["exact_match", "contains", "token_f1", "rouge_l", "code_similarity"] = "exact_match"
    context_field: str | None = None
    question_field: str | None = None
    prompt_template: str = "{context}\n\nQuestion: {question}\nAnswer:"
    truncation_strategy: Literal["error", "left", "middle"] = "error"
    adapter_activation_marker: str | None = None


class UpdateConfig(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["attention.q"])
    step_count: int = 1
    update_norm: float = 0.01


class CacheConfig(BaseModel):
    strategies: list[str] = Field(default_factory=lambda: ["full_recompute", "stale_reuse"])
    refresh_period: int = 4
    update_norm_threshold: float = 0.05
    version_gap_threshold: int = Field(default=8, ge=0)
    error_proxy_threshold: float = Field(default=0.25, ge=0.0)
    latency_budget_fraction: float = Field(default=1.0, gt=0.0)
    failure_map_path: Path | None = None
    oracle_kl_threshold: float = 0.05
    oracle_top1_threshold: float = 0.99
    oracle_task_drop_threshold: float = 0.01
    max_cache_bytes: int | None = Field(default=None, gt=0)
    max_cache_entries: int | None = Field(default=None, gt=0)
    eviction_policy: Literal["lru", "fifo"] = "lru"
    manager_scope: Literal["condition", "sample", "global_workload"] = "condition"


class MetricsConfig(BaseModel):
    compute_tensor_metrics: bool = True
    compute_task_metrics: bool = True
    compute_attention_metrics: bool = False
    compute_flops_metrics: bool = True


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    output_dir: Path = Path("runs/default")
    resume: bool = False
    checkpoint_each_target: bool = True
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    updates: UpdateConfig = Field(default_factory=UpdateConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)


class SweepAxis(BaseModel):
    path: str
    values: list[Any]


class SweepConfig(BaseModel):
    name: str
    base: ExperimentConfig
    output_dir: Path = Path("runs/sweep")
    axes: list[SweepAxis] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> SweepConfig:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)

    def expand(self) -> list[ExperimentConfig]:
        configs: list[ExperimentConfig] = []

        def rec(index: int, payload: dict[str, Any], suffix: list[str]) -> None:
            if index == len(self.axes):
                item = ExperimentConfig.model_validate(payload)
                clean_suffix = "__".join(suffix) if suffix else "base"
                item.name = f"{self.name}-{clean_suffix}"
                item.output_dir = self.output_dir / clean_suffix
                configs.append(item)
                return
            axis = self.axes[index]
            for value in axis.values:
                next_payload = deepcopy(payload)
                _set_dotted(next_payload, axis.path, value)
                safe_value = str(value).replace("/", "_").replace(".", "p").replace(" ", "_")
                rec(index + 1, next_payload, [*suffix, f"{axis.path.replace('.', '_')}={safe_value}"])

        rec(0, self.base.model_dump(mode="json"), [])
        return configs


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    current: dict[str, Any] = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


class AdapterConfig(BaseModel):
    update_mode: Literal["random", "lora_train", "static_lora"] = "random"
    norm_control: Literal["target_l2", "none"] = "target_l2"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    learning_rate: float = 1e-3
    train_steps_per_version: int = 1
    freeze_base_model: bool = True
    static_adapter_sequence: list[int] = Field(default_factory=lambda: [0, 1, 2, 0, 1, 2])


class VersionedExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    output_dir: Path = Path("runs/versioned")
    resume: bool = False
    checkpoint_each_target: bool = True
    experiment_id: str = "versioned"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    updates: UpdateConfig = Field(default_factory=UpdateConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    version_steps: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])
    cached_version: int = 0

    @classmethod
    def from_yaml(cls, path: Path) -> VersionedExperimentConfig:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)


class VersionedSweepConfig(BaseModel):
    name: str
    base: VersionedExperimentConfig
    output_dir: Path = Path("runs/versioned-sweep")
    axes: list[SweepAxis] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> VersionedSweepConfig:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)

    def expand(self) -> list[VersionedExperimentConfig]:
        configs: list[VersionedExperimentConfig] = []

        def rec(index: int, payload: dict[str, Any], suffix: list[str]) -> None:
            if index == len(self.axes):
                item = VersionedExperimentConfig.model_validate(payload)
                clean_suffix = "__".join(suffix) if suffix else "base"
                item.name = f"{self.name}-{clean_suffix}"
                item.output_dir = self.output_dir / clean_suffix
                configs.append(item)
                return
            axis = self.axes[index]
            for value in axis.values:
                next_payload = deepcopy(payload)
                _set_dotted(next_payload, axis.path, value)
                safe_value = str(value).replace("/", "_").replace(".", "p").replace(" ", "_")
                rec(index + 1, next_payload, [*suffix, f"{axis.path.replace('.', '_')}={safe_value}"])

        rec(0, self.base.model_dump(mode="json"), [])
        return configs
