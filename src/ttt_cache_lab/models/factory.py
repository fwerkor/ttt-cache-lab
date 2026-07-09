from __future__ import annotations

from ttt_cache_lab.configs import ModelConfig
from ttt_cache_lab.models.interface import ModelBackend
from ttt_cache_lab.models.toy import ToyBackend


def build_backend(config: ModelConfig, *, seed: int) -> ModelBackend:
    if config.backend == "toy":
        return ToyBackend(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            seed=seed,
        )
    if config.backend == "hf":
        from ttt_cache_lab.models.hf import HuggingFaceBackend

        if not config.model_name_or_path:
            raise ValueError("model.model_name_or_path is required for the hf backend")
        return HuggingFaceBackend(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            torch_dtype=config.torch_dtype,
            max_length=config.max_length,
            trust_remote_code=config.trust_remote_code,
            seed=seed,
        )
    raise ValueError(f"Unsupported model backend: {config.backend}")
