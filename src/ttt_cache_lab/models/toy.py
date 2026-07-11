from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName
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
        self._capture_attention_metrics = False
        self._sample_answers: dict[str, str] = {}
        self._last_raw_update_norm = 0.0
        self._last_applied_update_norm = 0.0
        self._last_updated_parameter_count = 0
        self._last_applied_update_rms = 0.0

    @property
    def parameter_count(self) -> int:
        return self.num_layers * self.hidden_size * self.hidden_size + self.hidden_size * self.vocab_size

    def prepare_sample(self, sample: TaskSample, *, context_length: int) -> TaskSample:
        del context_length
        self._sample_answers[sample.prompt] = sample.answer
        return sample

    def _rng_for(self, text: str, version: int) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}:{version}:{text}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**32)
        return np.random.default_rng(seed)

    def prefill(self, prompt: str) -> BackendOutput:
        rng = self._rng_for(prompt, 0)
        cache = rng.normal(size=(self.num_layers, 2, self.hidden_size)).astype(np.float64)
        hidden = rng.normal(size=(self.num_layers, self.hidden_size)).astype(np.float64)
        logits = rng.normal(size=(1, self.vocab_size)).astype(np.float64)
        answer = self._sample_answers.get(prompt) or self._extract_answer(prompt)
        if answer:
            logits[0, self._answer_bucket(answer)] += 8.0
        return BackendOutput(
            logits=logits,
            cache_tensor=cache,
            hidden_tensor=hidden,
            parameter_version=0,
            extras=self._attention_extras(cache),
        )

    def simulate_update(self, baseline: BackendOutput, target: UpdateTarget, *, update_norm: float) -> BackendOutput:
        self._last_raw_update_norm = update_norm
        self._last_applied_update_norm = update_norm
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
        cache = updated.cache_tensor + rng.normal(scale=noise_scale, size=updated.cache_tensor.shape)
        return BackendOutput(
            logits=updated.logits + rng.normal(scale=noise_scale, size=updated.logits.shape),
            cache_tensor=cache,
            hidden_tensor=updated.hidden_tensor + rng.normal(scale=noise_scale, size=updated.hidden_tensor.shape),
            parameter_version=updated.parameter_version,
            extras=self._attention_extras(cache),
        )

    def apply_cache_strategy(
        self,
        *,
        baseline: BackendOutput,
        full: BackendOutput,
        updated: BackendOutput,
        decision: StrategyDecision,
    ) -> BackendOutput:
        if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
            return full
        if decision.action is CacheAction.REUSE_EXACT:
            return baseline
        if decision.action is CacheAction.REUSE_FROZEN:
            return BackendOutput(
                logits=updated.logits,
                cache_tensor=baseline.cache_tensor,
                hidden_tensor=updated.hidden_tensor,
                parameter_version=updated.parameter_version,
                extras=self._attention_extras(baseline.cache_tensor),
            )
        if decision.action is CacheAction.REUSE_STALE:
            return BackendOutput(
                logits=updated.logits,
                cache_tensor=baseline.cache_tensor,
                hidden_tensor=baseline.hidden_tensor,
                parameter_version=updated.parameter_version,
                extras=self._attention_extras(baseline.cache_tensor),
            )
        if decision.action is CacheAction.DELTA_CORRECT:
            corrected = baseline.cache_tensor + 0.75 * (full.cache_tensor - baseline.cache_tensor)
            delta_mode = "toy_delta_correction"
            logical_cache_bytes = int(corrected.nbytes)
            if decision.strategy is StrategyName.LRAGENT_ADAPTER_CACHE:
                delta_mode = "lragent_shared_base_plus_low_rank_component"
                logical_cache_bytes = max(1, int(corrected.nbytes * 0.1))
            elif decision.strategy is StrategyName.FORKKV_BASE_DELTA:
                delta_mode = "forkkv_copy_on_write_residual"
                logical_cache_bytes = max(1, int(corrected.nbytes * 0.2))
            return BackendOutput(
                logits=0.75 * full.logits + 0.25 * updated.logits,
                cache_tensor=corrected,
                hidden_tensor=updated.hidden_tensor,
                parameter_version=updated.parameter_version,
                extras={
                    "delta_mode": delta_mode,
                    "cache_bytes": logical_cache_bytes,
                    "physical_cache_bytes": int(corrected.nbytes),
                    "baseline_fidelity": "paper_reimplementation",
                    **self._attention_extras(corrected),
                },
            )
        if decision.action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
            return BackendOutput(
                logits=full.logits.copy(),
                cache_tensor=full.cache_tensor.copy(),
                hidden_tensor=full.hidden_tensor.copy(),
                parameter_version=updated.parameter_version,
                extras={
                    "cache_mode": "alora_base_prefix_suffix_recompute",
                    **self._attention_extras(full.cache_tensor),
                },
            )
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            if decision.first_invalid_layer is None:
                raise ValueError("Partial recompute requires an explicit first-invalid layer")
            start = decision.first_invalid_layer
            end = min(
                self.num_layers,
                decision.last_recomputed_layer
                if decision.last_recomputed_layer is not None
                else self.num_layers,
            )
            if end <= start:
                raise ValueError("Partial recompute requires a non-empty layer interval")
            cache = baseline.cache_tensor.copy()
            hidden = baseline.hidden_tensor.copy()
            cache[start:end] = full.cache_tensor[start:end]
            hidden[start:end] = full.hidden_tensor[start:end]
            fraction = (end - start) / max(1, self.num_layers - start)
            logits = baseline.logits + fraction * (full.logits - baseline.logits)
            return BackendOutput(
                logits=logits,
                cache_tensor=cache,
                hidden_tensor=hidden,
                parameter_version=updated.parameter_version,
                extras={
                    "partial_start_layer": start,
                    "partial_end_layer": end,
                    "partial_window_layers": end - start,
                    "partial_mode": (
                        "toy_suffix_recompute"
                        if end == self.num_layers
                        else "toy_finite_window_recompute"
                    ),
                    **self._attention_extras(cache),
                },
            )
        return full

    def score_answer(self, sample: TaskSample, output: BackendOutput) -> float:
        # Deterministic proxy score: useful for smoke tests, not a real LM metric.
        target_bucket = self._answer_bucket(sample.answer)
        predicted = int(np.argmax(output.logits, axis=-1)[0])
        return 1.0 if predicted == target_bucket else 0.0

    def estimate_latency(self, decision: StrategyDecision, *, context_length: int) -> float:
        base = max(1.0, context_length / 1024.0)
        if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
            return 10.0 * base
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            if decision.first_invalid_layer is None:
                raise ValueError("Partial recompute requires an explicit first-invalid layer")
            first = decision.first_invalid_layer
            end = min(
                self.num_layers,
                decision.last_recomputed_layer
                if decision.last_recomputed_layer is not None
                else self.num_layers,
            )
            fraction = max(0.1, max(0, end - first) / self.num_layers)
            return 10.0 * base * fraction
        if decision.action is CacheAction.DELTA_CORRECT:
            return 2.0 * base
        if decision.action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
            return 2.5 * base
        return 1.0 * base

    def configure_metrics(self, *, capture_attention: bool) -> None:
        self._capture_attention_metrics = capture_attention

    def estimate_flops(self, decision: StrategyDecision, *, context_length: int) -> float:
        tokens = max(1, context_length)
        hidden = self.hidden_size
        per_layer = 12.0 * tokens * hidden * hidden + 4.0 * tokens * tokens * hidden
        full = self.num_layers * per_layer
        if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
            return full
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            first = decision.first_invalid_layer or 0
            end = min(
                self.num_layers,
                decision.last_recomputed_layer
                if decision.last_recomputed_layer is not None
                else self.num_layers,
            )
            return max(0, end - first) * per_layer
        if decision.action is CacheAction.DELTA_CORRECT:
            return 4.0 * tokens * hidden * max(1, hidden // 8)
        if decision.action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
            return full * (decision.recompute_fraction or 0.25)
        return 0.0

    def _attention_extras(self, cache: np.ndarray) -> dict[str, np.ndarray]:
        if not self._capture_attention_metrics:
            return {}
        scores = cache[:, 0, :].astype(np.float64)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return {
            "attention_summary": probabilities,
            "attention_input_summary": cache[:, 0, :],
            "attention_output_summary": cache.mean(axis=1),
        }

    def last_adaptation_latency(self) -> float:
        return 0.0

    def last_raw_update_norm(self) -> float:
        return self._last_raw_update_norm

    def last_applied_update_norm(self) -> float:
        return self._last_applied_update_norm

    def last_updated_parameter_count(self) -> int:
        return self._last_updated_parameter_count

    def last_applied_update_rms(self) -> float:
        return self._last_applied_update_rms

    def restore_after_update(self) -> None:
        self._last_raw_update_norm = 0.0
        self._last_applied_update_norm = 0.0
        self._last_updated_parameter_count = 0
        self._last_applied_update_rms = 0.0

    def train_lora_step(
        self,
        sample: TaskSample,
        target: UpdateTarget,
        *,
        rank: int,
        alpha: float,
        learning_rate: float,
        freeze_base_model: bool = True,
        target_update_norm: float | None = None,
        target_update_rms: float | None = None,
    ) -> float:
        del sample, alpha, freeze_base_model
        # Toy mode has no persistent parameters. Return a deterministic update-norm proxy.
        measured = learning_rate * self._target_multiplier(target)
        updated_parameter_count = max(1, rank * self.hidden_size * 2)
        if target_update_norm is not None and target_update_rms is not None:
            raise ValueError("target_update_norm and target_update_rms are mutually exclusive")
        applied = measured
        if target_update_norm is not None:
            applied = target_update_norm
        elif target_update_rms is not None:
            applied = target_update_rms * math.sqrt(updated_parameter_count)
        self._last_raw_update_norm = measured
        self._last_applied_update_norm = applied
        self._last_updated_parameter_count = updated_parameter_count
        self._last_applied_update_rms = applied / math.sqrt(updated_parameter_count)
        return applied

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
            ModuleKind.MOE_ROUTER: 1.0,
            ModuleKind.MOE_SHARED_EXPERT: 0.8,
            ModuleKind.MOE_ROUTED_EXPERTS: 0.8,
            ModuleKind.LORA_QV: 0.5,
            ModuleKind.LORA_ATTN: 0.65,
            ModuleKind.LORA_ALL_LATE: 0.9,
            ModuleKind.LORA_MLP: 0.5,
            ModuleKind.LORA_MOE_ROUTER: 0.7,
            ModuleKind.LORA_MOE_SHARED_EXPERT: 0.5,
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
