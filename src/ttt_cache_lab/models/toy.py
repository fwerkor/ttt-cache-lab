from __future__ import annotations

import hashlib
import re

import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import StrategyDecision
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.models.interface import BackendOutput
from ttt_cache_lab.updates.targets import ModuleKind, UpdateTarget


class ToyBackend:
    """Small deterministic backend for CI and planner experiments.

    It is not a language model. It creates prompt-dependent tensors with shapes
    resembling per-layer cache summaries, then injects target-dependent drift so
    cache strategies can be tested without downloading model weights.
    """

    def __init__(self, *, num_layers: int, hidden_size: int, vocab_size: int, seed: int) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seed = seed

    def _rng_for(self, text: str, version: int) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}:{version}:{text}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**32)
        return np.random.default_rng(seed)

    def prefill(self, prompt: str) -> BackendOutput:
        rng = self._rng_for(prompt, 0)
        cache = rng.normal(size=(self.num_layers, 2, self.hidden_size)).astype(np.float64)
        hidden = rng.normal(size=(self.num_layers, self.hidden_size)).astype(np.float64)
        logits = rng.normal(size=(1, self.vocab_size)).astype(np.float64)
        answer = self._extract_answer(prompt)
        if answer:
            logits[0, self._answer_bucket(answer)] += 8.0
        return BackendOutput(logits=logits, cache_tensor=cache, hidden_tensor=hidden, parameter_version=0)

    def simulate_update(self, baseline: BackendOutput, target: UpdateTarget, *, update_norm: float) -> BackendOutput:
        drift = self._target_drift(target, update_norm)
        return BackendOutput(
            logits=baseline.logits + drift["logits"],
            cache_tensor=baseline.cache_tensor + drift["cache"],
            hidden_tensor=baseline.hidden_tensor + drift["hidden"],
            parameter_version=baseline.parameter_version + 1,
        )

    def full_recompute(self, prompt: str, updated: BackendOutput) -> BackendOutput:
        rng = self._rng_for(prompt, updated.parameter_version)
        noise_scale = 0.01
        return BackendOutput(
            logits=updated.logits + rng.normal(scale=noise_scale, size=updated.logits.shape),
            cache_tensor=updated.cache_tensor + rng.normal(scale=noise_scale, size=updated.cache_tensor.shape),
            hidden_tensor=updated.hidden_tensor + rng.normal(scale=noise_scale, size=updated.hidden_tensor.shape),
            parameter_version=updated.parameter_version,
        )

    def apply_cache_strategy(
        self,
        *,
        baseline: BackendOutput,
        full: BackendOutput,
        updated: BackendOutput,
        decision: StrategyDecision,
    ) -> BackendOutput:
        if decision.action is CacheAction.FULL_RECOMPUTE:
            return full
        if decision.action is CacheAction.REUSE_EXACT:
            return full
        if decision.action is CacheAction.REUSE_FROZEN:
            return BackendOutput(
                logits=updated.logits,
                cache_tensor=baseline.cache_tensor,
                hidden_tensor=updated.hidden_tensor,
                parameter_version=updated.parameter_version,
            )
        if decision.action is CacheAction.REUSE_STALE:
            return BackendOutput(
                logits=updated.logits,
                cache_tensor=baseline.cache_tensor,
                hidden_tensor=baseline.hidden_tensor,
                parameter_version=updated.parameter_version,
            )
        if decision.action is CacheAction.DELTA_CORRECT:
            corrected = baseline.cache_tensor + 0.75 * (full.cache_tensor - baseline.cache_tensor)
            return BackendOutput(
                logits=0.75 * full.logits + 0.25 * updated.logits,
                cache_tensor=corrected,
                hidden_tensor=updated.hidden_tensor,
                parameter_version=updated.parameter_version,
            )
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            layer = decision.first_invalid_layer or 0
            cache = baseline.cache_tensor.copy()
            hidden = baseline.hidden_tensor.copy()
            cache[layer:] = full.cache_tensor[layer:]
            hidden[layer:] = full.hidden_tensor[layer:]
            return BackendOutput(
                logits=0.9 * full.logits + 0.1 * updated.logits,
                cache_tensor=cache,
                hidden_tensor=hidden,
                parameter_version=updated.parameter_version,
            )
        return full

    def score_answer(self, sample: TaskSample, output: BackendOutput) -> float:
        # Deterministic proxy score: useful for smoke tests, not a real LM metric.
        target_bucket = self._answer_bucket(sample.answer)
        predicted = int(np.argmax(output.logits, axis=-1)[0])
        return 1.0 if predicted == target_bucket else 0.0

    def estimate_latency(self, decision: StrategyDecision, *, context_length: int) -> float:
        base = max(1.0, context_length / 1024.0)
        if decision.action is CacheAction.FULL_RECOMPUTE:
            return 10.0 * base
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            first = decision.first_invalid_layer or 0
            fraction = max(0.1, (self.num_layers - first) / self.num_layers)
            return 10.0 * base * fraction
        if decision.action is CacheAction.DELTA_CORRECT:
            return 2.0 * base
        return 1.0 * base

    def restore_after_update(self) -> None:
        return None

    def train_lora_step(
        self,
        prompt: str,
        target: UpdateTarget,
        *,
        rank: int,
        alpha: float,
        learning_rate: float,
        freeze_base_model: bool = True,
    ) -> float:
        del prompt, rank, alpha, freeze_base_model
        # Toy mode has no persistent parameters. Return a deterministic update-norm proxy.
        return learning_rate * self._target_multiplier(target)

    def _target_multiplier(self, target: UpdateTarget) -> float:
        return {
            ModuleKind.ATTENTION_Q: 0.2,
            ModuleKind.LORA_Q: 0.15,
            ModuleKind.ATTENTION_K: 0.9,
            ModuleKind.ATTENTION_V: 0.9,
            ModuleKind.LORA_K: 0.45,
            ModuleKind.LORA_V: 0.45,
            ModuleKind.ATTENTION_O: 0.7,
            ModuleKind.ATTENTION_QV: 0.85,
            ModuleKind.ATTENTION_ATTN: 0.95,
            ModuleKind.MLP: 0.8,
            ModuleKind.LORA_QV: 0.5,
            ModuleKind.LORA_ATTN: 0.65,
            ModuleKind.LORA_ALL_LATE: 0.9,
            ModuleKind.LORA_MLP: 0.5,
            ModuleKind.NORM: 1.2,
            ModuleKind.OUTPUT_HEAD: 0.05,
            ModuleKind.UNKNOWN: 1.0,
        }.get(target.kind, 1.0)

    def _target_drift(self, target: UpdateTarget, update_norm: float) -> dict[str, np.ndarray]:
        digest = hashlib.sha256(
            f"{self.seed}:{target.kind.value}:{target.layer}:{round(update_norm, 8)}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**32)
        rng = np.random.default_rng(seed)
        scale = update_norm * self._target_multiplier(target)
        cache = rng.normal(scale=scale, size=(self.num_layers, 2, self.hidden_size))
        hidden = rng.normal(scale=scale, size=(self.num_layers, self.hidden_size))
        logits = rng.normal(scale=scale, size=(1, self.vocab_size))
        if target.layer is not None:
            mask = np.zeros_like(cache)
            mask[target.layer :] = 1.0
            cache *= mask
            hmask = np.zeros_like(hidden)
            hmask[target.layer :] = 1.0
            hidden *= hmask
        return {"cache": cache, "hidden": hidden, "logits": logits}

    def _answer_bucket(self, answer: str) -> int:
        return sum(ord(ch) for ch in answer) % self.vocab_size

    def _extract_answer(self, prompt: str) -> str | None:
        passkey = re.search(r"secret passkey is ([0-9]+)", prompt, flags=re.IGNORECASE)
        if passkey:
            return passkey.group(1)
        key_value = re.search(r"^([^:\n]+):\s*([A-Za-z0-9_]+)$", prompt, flags=re.MULTILINE)
        if key_value:
            return key_value.group(2)
        return None
