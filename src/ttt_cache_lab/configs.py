from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    backend: Literal["toy", "hf"] = "toy"
    model_name_or_path: str | None = None
    num_layers: int = 4
    hidden_size: int = 32
    vocab_size: int = 256
    device: str = "auto"
    torch_dtype: str = "auto"
    max_length: int = 2048
    trust_remote_code: bool = False


class DataConfig(BaseModel):
    task: Literal["passkey", "key_value"] = "passkey"
    num_samples: int = 16
    context_length: int = 512
    answer_length: int = 4


class UpdateConfig(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["attention.q"])
    step_count: int = 1
    update_norm: float = 0.01


class CacheConfig(BaseModel):
    strategies: list[str] = Field(default_factory=lambda: ["full_recompute", "stale_reuse"])
    refresh_period: int = 4
    update_norm_threshold: float = 0.05


class MetricsConfig(BaseModel):
    compute_tensor_metrics: bool = True
    compute_task_metrics: bool = True


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    output_dir: Path = Path("runs/default")
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
