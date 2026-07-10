from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import StrategyDecision
from ttt_cache_lab.data.synthetic import TaskSample
from ttt_cache_lab.models.accelerator import memory_allocated, reset_peak_memory, resolve_device, synchronize
from ttt_cache_lab.models.interface import BackendOutput
from ttt_cache_lab.updates.targets import ModuleKind, UpdateTarget


@dataclass(frozen=True)
class _PromptState:
    prompt: str
    input_ids: Any
    prefix_ids: Any
    probe_ids: Any
    attention_mask: Any


class HuggingFaceBackend:
    """Experimental HF backend for real stale-cache measurements.

    The backend uses a prefix/probe split. It pre-fills all tokens except the
    final probe token and then evaluates the probe token with `past_key_values`.
    After a controlled parameter perturbation, full recompute uses a fresh
    prefix under the new parameter version, while stale/frozen reuse evaluates
    the probe token with the old prefix cache.

    The backend also implements measurable cache-surgery paths for layer-wise
    recomputation and delta correction. The experiment runner still computes a
    full reference first so metrics can be measured, but strategy application no
    longer returns that reference unchanged.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        torch_dtype: str,
        max_length: int,
        trust_remote_code: bool,
        seed: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps
            raise RuntimeError("Install the HF backend with: pip install -e '.[hf]'") from exc

        self.torch = torch
        self.seed = seed
        torch.manual_seed(seed)
        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype(torch_dtype)
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.num_layers = self._infer_num_layers()
        self.max_length = max_length
        self.parameter_version = 0
        self._deltas: list[tuple[Any, Any]] = []
        self._last_state: _PromptState | None = None
        self._last_prefill_s = 0.0
        self._last_stale_s = 0.0
        self._last_partial_s = 0.0
        self._last_delta_s = 0.0
        self._lora_modules: list[Any] = []
        self._active_lora_modules: list[Any] = []
        self._lora_target_key: str | None = None
        self._prepared_input_ids: dict[str, Any] = {}
        self._sample_answers: dict[str, str] = {}
        self._sample_answer_token_counts: dict[str, int] = {}

    def _infer_num_layers(self) -> int:
        config = self.model.config
        for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError(f"Cannot infer transformer layer count from {type(config).__name__}")

    def prepare_sample(self, sample: TaskSample, *, context_length: int) -> TaskSample:
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        if context_length > self.max_length:
            raise ValueError(
                f"Requested context_length={context_length} exceeds model.max_length={self.max_length}"
            )
        encoded = self.tokenizer(sample.prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        current = int(input_ids.shape[1])
        if current > context_length:
            raise ValueError(
                "Synthetic prompt tokenization exceeded the requested context length; "
                f"generated={current}, requested={context_length}. "
                "Reduce filler density instead of silently truncating."
            )
        if current < context_length:
            filler = self.tokenizer(" filler", add_special_tokens=False).get("input_ids", [])
            filler_id = int(filler[0]) if filler else int(self.tokenizer.eos_token_id or 0)
            padding = self.torch.full((1, context_length - current), filler_id, dtype=input_ids.dtype)
            input_ids = self.torch.cat([padding, input_ids], dim=1)
        self._prepared_input_ids[sample.prompt] = input_ids
        answer_ids = self.tokenizer(sample.answer, add_special_tokens=False).get("input_ids", [])
        self._sample_answers[sample.prompt] = sample.answer
        self._sample_answer_token_counts[sample.prompt] = max(1, len(answer_ids))
        metadata = dict(sample.metadata)
        metadata["token_length"] = context_length
        return TaskSample(prompt=sample.prompt, answer=sample.answer, metadata=metadata)

    def _resolve_device(self, device: str) -> str:
        return resolve_device(self.torch, device)

    def _resolve_dtype(self, dtype: str) -> Any | None:
        if dtype == "auto":
            return None
        mapping = {
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {dtype}")
        return mapping[dtype]

    def prefill(self, prompt: str) -> BackendOutput:
        state = self._encode_prompt(prompt)
        self._last_state = state
        reset_peak_memory(self.torch, self.device)
        synchronize(self.torch, self.device)
        start = time.perf_counter()
        self._set_lora_capture(True)
        with self.torch.no_grad():
            prefill = self.model(input_ids=state.prefix_ids, use_cache=True, output_hidden_states=True)
            self._set_lora_capture(False)
            probe_logits, generated_text, decode_s = self._generate_answer(state, prefill.past_key_values)
        synchronize(self.torch, self.device)
        self._last_prefill_s = time.perf_counter() - start
        return BackendOutput(
            logits=self._to_numpy(probe_logits),
            cache_tensor=self._summarize_past(prefill.past_key_values),
            hidden_tensor=self._summarize_hidden(prefill.hidden_states),
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": prefill.past_key_values,
                "hidden_states": tuple(hidden.detach() for hidden in prefill.hidden_states),
                "prompt_state": state,
                "lora_cache": self._snapshot_lora_cache(),
                "memory_allocated": memory_allocated(self.torch, self.device),
                "generated_text": generated_text,
                "decode_latency": decode_s,
            },
        )

    def simulate_update(self, baseline: BackendOutput, target: UpdateTarget, *, update_norm: float) -> BackendOutput:
        selected = self._select_parameters(target)
        if not selected:
            raise ValueError(f"No HF parameters matched update target {target.raw!r}")
        self.torch.manual_seed(self.seed + self.parameter_version + 1)
        for param in selected:
            if not param.requires_grad or not param.is_floating_point():
                continue
            noise = self.torch.randn_like(param) * update_norm
            with self.torch.no_grad():
                param.add_(noise)
            self._deltas.append((param, noise))
        self.parameter_version += 1
        return BackendOutput(
            logits=baseline.logits,
            cache_tensor=baseline.cache_tensor,
            hidden_tensor=baseline.hidden_tensor,
            parameter_version=self.parameter_version,
            extras=baseline.extras,
        )

    def full_recompute(self, prompt: str, updated: BackendOutput) -> BackendOutput:
        state = self._encode_prompt(prompt)
        reset_peak_memory(self.torch, self.device)
        synchronize(self.torch, self.device)
        start = time.perf_counter()
        self._set_lora_capture(True)
        with self.torch.no_grad():
            prefill = self.model(input_ids=state.prefix_ids, use_cache=True, output_hidden_states=True)
            self._set_lora_capture(False)
            probe_logits, generated_text, decode_s = self._generate_answer(state, prefill.past_key_values)
        synchronize(self.torch, self.device)
        self._last_prefill_s = time.perf_counter() - start
        return BackendOutput(
            logits=self._to_numpy(probe_logits),
            cache_tensor=self._summarize_past(prefill.past_key_values),
            hidden_tensor=self._summarize_hidden(prefill.hidden_states),
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": prefill.past_key_values,
                "hidden_states": tuple(hidden.detach() for hidden in prefill.hidden_states),
                "prompt_state": state,
                "lora_cache": self._snapshot_lora_cache(),
                "memory_allocated": memory_allocated(self.torch, self.device),
                "generated_text": generated_text,
                "decode_latency": decode_s,
            },
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
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            return self._partial_recompute_prefix_cache(baseline=baseline, full=full, decision=decision)
        if decision.action is CacheAction.DELTA_CORRECT:
            return self._delta_correct_prefix_cache(baseline=baseline, full=full, decision=decision)
        if decision.action in {CacheAction.REUSE_STALE, CacheAction.REUSE_FROZEN}:
            return self._reuse_old_prefix_cache(baseline)
        return full

    def score_answer(self, sample: TaskSample, output: BackendOutput) -> float:
        extras = output.extras or {}
        generated = extras.get("generated_text")
        if isinstance(generated, str):
            return 1.0 if generated.strip() == sample.answer.strip() else 0.0
        logits = output.logits[0]
        top_token = int(np.argmax(logits))
        decoded = self.tokenizer.decode([top_token]).strip()
        return 1.0 if decoded and decoded == sample.answer.strip() else 0.0

    def estimate_latency(self, decision: StrategyDecision, *, context_length: int) -> float:
        if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
            return self._last_prefill_s or 1.0
        if decision.action in {CacheAction.REUSE_STALE, CacheAction.REUSE_FROZEN}:
            return self._last_stale_s or max(1e-6, (self._last_prefill_s or 1.0) / 10.0)
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            return getattr(self, "_last_partial_s", 0.0) or max(1e-6, (self._last_prefill_s or 1.0) * 0.5)
        if decision.action is CacheAction.DELTA_CORRECT:
            return getattr(self, "_last_delta_s", 0.0) or max(1e-6, (self._last_prefill_s or 1.0) * 0.15)
        return max(1e-6, (self._last_prefill_s or 1.0) / 10.0)

    def restore_after_update(self) -> None:
        for param, delta in reversed(self._deltas):
            with self.torch.no_grad():
                param.sub_(delta)
        self._deltas.clear()
        self.reset_lora_adapters()
        self.parameter_version = 0

    def setup_lora(self, target: UpdateTarget, *, rank: int, alpha: float, freeze_base_model: bool = True) -> int:
        from torch import nn

        from ttt_cache_lab.models.lora import is_lora_linear, make_lora_linear

        target_key = f"{target.kind.value}:{target.layer}:{rank}:{alpha}"

        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad_(False)

        self._active_lora_modules = []
        for module in self._lora_modules:
            for param in module.lora_parameters():
                param.requires_grad_(False)

        filters = self._target_filters(target.kind)
        replaced = 0
        seen_active: set[int] = set()
        for parent, child_name, module_name, module in list(self._iter_named_modules_with_parent()):
            lower = module_name.lower()
            if target.layer is not None and not self._name_matches_layer(lower, target.layer):
                continue
            if not any(part in lower for part in filters):
                continue
            if is_lora_linear(module):
                if not getattr(module, "lora_name", ""):
                    module.lora_name = module_name
                if id(module) not in seen_active:
                    self._activate_lora_module(module)
                    seen_active.add(id(module))
                    replaced += 1
                continue
            if isinstance(module, nn.Linear):
                wrapped = make_lora_linear(self.torch, nn, module, rank=rank, alpha=alpha)
                wrapped.lora_name = module_name
                wrapped.to(self.device)
                setattr(parent, child_name, wrapped)
                self._lora_modules.append(wrapped)
                self._activate_lora_module(wrapped)
                seen_active.add(id(wrapped))
                replaced += 1
        if replaced == 0 and target.layer is not None:
            return self.setup_lora(
                UpdateTarget(kind=target.kind, layer=None, raw=target.raw),
                rank=rank,
                alpha=alpha,
                freeze_base_model=freeze_base_model,
            )
        self._lora_target_key = target_key
        return replaced

    def prepare_update_target(
        self,
        target: UpdateTarget,
        *,
        rank: int,
        alpha: float,
        freeze_base_model: bool = True,
    ) -> int:
        return self.setup_lora(
            target,
            rank=rank,
            alpha=alpha,
            freeze_base_model=freeze_base_model,
        )

    def train_lora_step(
        self,
        sample: TaskSample,
        target: UpdateTarget,
        *,
        rank: int,
        alpha: float,
        learning_rate: float,
        freeze_base_model: bool = True,
    ) -> float:
        count = self.setup_lora(target, rank=rank, alpha=alpha, freeze_base_model=freeze_base_model)
        if count == 0:
            raise ValueError(f"No Linear modules matched LoRA target {target.raw!r}")
        self.model.train()
        prompt_ids = self._prepared_input_ids.get(sample.prompt)
        if prompt_ids is None:
            prompt_ids = self.tokenizer(
                sample.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )["input_ids"]
        answer_ids = self.tokenizer(
            sample.answer,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        if answer_ids.shape[1] == 0:
            self.model.eval()
            return 0.0
        input_ids = self.torch.cat([prompt_ids, answer_ids], dim=1).to(self.device)
        labels = self.torch.full_like(input_ids, -100)
        labels[:, -answer_ids.shape[1] :] = answer_ids.to(self.device)
        attention_mask = self.torch.ones_like(input_ids, device=self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss
        loss.backward()
        total_norm = 0.0
        with self.torch.no_grad():
            for module in self._active_lora_modules:
                for param in module.lora_parameters():
                    if param.grad is None:
                        continue
                    delta = -learning_rate * param.grad
                    total_norm += float(self.torch.linalg.vector_norm(delta.detach().float()).cpu())
                    param.add_(delta)
                    param.grad = None
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        self.parameter_version += 1
        return total_norm

    def reset_lora_adapters(self) -> None:
        for module in self._lora_modules:
            if hasattr(module, "reset_lora"):
                module.reset_lora()
            for param in module.lora_parameters():
                param.requires_grad_(False)
        self._active_lora_modules = []

    def _activate_lora_module(self, module: Any) -> None:
        for param in module.lora_parameters():
            param.requires_grad_(True)
        self._active_lora_modules.append(module)

    def _iter_named_modules_with_parent(self) -> list[tuple[Any, str, str, Any]]:
        items: list[tuple[Any, str, str, Any]] = []
        for module_name, module in self.model.named_modules():
            for child_name, child in module.named_children():
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                items.append((module, child_name, full_name, child))
        return items

    def _encode_prompt(self, prompt: str) -> _PromptState:
        prepared = self._prepared_input_ids.get(prompt)
        if prepared is None:
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            input_ids = prepared.to(self.device)
            attention_mask = self.torch.ones_like(input_ids, device=self.device)
        if input_ids.shape[1] < 2:
            eos = self.tokenizer.eos_token_id or 0
            input_ids = self.torch.cat([input_ids, self.torch.tensor([[eos]], device=self.device)], dim=1)
        return _PromptState(
            prompt=prompt,
            input_ids=input_ids,
            prefix_ids=input_ids[:, :-1],
            probe_ids=input_ids[:, -1:],
            attention_mask=attention_mask,
        )

    def _generate_answer(self, state: _PromptState, past: Any) -> tuple[Any, str, float]:
        max_new_tokens = self._sample_answer_token_counts.get(state.prompt, 1)
        current = state.probe_ids
        current_past = past
        generated: list[Any] = []
        first_logits = None
        start = time.perf_counter()
        for _ in range(max_new_tokens):
            result = self.model(input_ids=current, past_key_values=current_past, use_cache=True)
            logits = result.logits[:, -1, :]
            if first_logits is None:
                first_logits = logits
            next_token = self.torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            current = next_token
            current_past = result.past_key_values
        synchronize(self.torch, self.device)
        decode_s = time.perf_counter() - start
        if first_logits is None:
            raise RuntimeError("Answer generation produced no logits")
        generated_ids = self.torch.cat(generated, dim=1)[0].detach().cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return first_logits, generated_text, decode_s

    def _reuse_old_prefix_cache(self, baseline: BackendOutput) -> BackendOutput:
        if not baseline.extras:
            raise ValueError("Baseline output does not contain cached HF state")
        past = baseline.extras["past_key_values"]
        result = self._probe_with_past(
            baseline=baseline,
            past=past,
            cache_tensor=baseline.cache_tensor,
            hidden_tensor=baseline.hidden_tensor,
        )
        self._last_stale_s = float(result.extras.get("strategy_latency", 0.0)) if result.extras else 0.0
        return result

    def _partial_recompute_prefix_cache(
        self,
        *,
        baseline: BackendOutput,
        full: BackendOutput,
        decision: StrategyDecision,
    ) -> BackendOutput:
        if not baseline.extras:
            raise ValueError("Partial recompute requires cached HF state from the baseline output")
        del full
        native = self._native_partial_recompute_prefix_cache(baseline=baseline, decision=decision)
        if native is None:
            raise RuntimeError(
                f"Native partial recompute is unsupported for {type(self.model).__name__}; "
                "refusing to substitute a full-reference KV splice."
            )
        self._last_partial_s = float(native.extras.get("strategy_latency", 0.0)) if native.extras else 0.0
        return native

    def _native_partial_recompute_prefix_cache(
        self,
        *,
        baseline: BackendOutput,
        decision: StrategyDecision,
    ) -> BackendOutput | None:
        if not baseline.extras:
            return None
        state = baseline.extras.get("prompt_state")
        hidden_states = baseline.extras.get("hidden_states")
        old_past = baseline.extras.get("past_key_values")
        if not isinstance(state, _PromptState) or not isinstance(hidden_states, tuple) or old_past is None:
            return None
        split_layer = decision.first_invalid_layer or 0
        layer_container, family = self._decoder_layers()
        if layer_container is None or split_layer < 0 or split_layer > len(layer_container):
            return None
        old_layers = self._past_as_layers(old_past)
        if len(old_layers) != len(layer_container) or len(hidden_states) < len(layer_container) + 1:
            return None

        replacements: list[tuple[int, Any]] = []
        try:
            for layer_index in range(split_layer):
                original = layer_container[layer_index]
                cached_hidden = hidden_states[layer_index + 1]
                cached_past = old_layers[layer_index]
                wrapper = self._make_cached_decoder_layer(
                    family=family,
                    layer_index=layer_index,
                    cached_hidden=cached_hidden,
                    cached_past=cached_past,
                )
                replacements.append((layer_index, original))
                layer_container[layer_index] = wrapper

            reset_peak_memory(self.torch, self.device)
            synchronize(self.torch, self.device)
            start = time.perf_counter()
            self._set_lora_capture(True)
            with self.torch.no_grad():
                prefill = self.model(
                    input_ids=state.prefix_ids,
                    attention_mask=self.torch.ones_like(state.prefix_ids, device=self.device),
                    use_cache=True,
                    output_hidden_states=True,
                )
                self._set_lora_capture(False)
                probe_logits, generated_text, decode_s = self._generate_answer(state, prefill.past_key_values)
            synchronize(self.torch, self.device)
            latency = time.perf_counter() - start
        finally:
            self._set_lora_capture(False)
            for layer_index, original in replacements:
                layer_container[layer_index] = original

        lora_cache = dict(baseline.extras.get("lora_cache", {}))
        lora_cache.update(self._snapshot_lora_cache())
        return BackendOutput(
            logits=self._to_numpy(probe_logits),
            cache_tensor=self._summarize_past(prefill.past_key_values),
            hidden_tensor=self._summarize_hidden(prefill.hidden_states),
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": prefill.past_key_values,
                "hidden_states": tuple(hidden.detach() for hidden in prefill.hidden_states),
                "prompt_state": state,
                "lora_cache": lora_cache,
                "memory_allocated": memory_allocated(self.torch, self.device),
                "strategy_latency": latency,
                "decode_latency": decode_s,
                "generated_text": generated_text,
                "partial_mode": f"native_{family}_layer_restart",
            },
        )

    def _decoder_layers(self) -> tuple[Any | None, str]:
        backbone = getattr(self.model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is not None:
            return layers, "llama_like"
        transformer = getattr(self.model, "transformer", None)
        blocks = getattr(transformer, "h", None)
        if blocks is not None:
            return blocks, "gpt2"
        return None, "unknown"

    def _past_as_layers(self, past: Any) -> list[Any]:
        if hasattr(past, "to_legacy_cache"):
            return list(past.to_legacy_cache())
        return list(past)

    def _make_cached_decoder_layer(
        self,
        *,
        family: str,
        layer_index: int,
        cached_hidden: Any,
        cached_past: Any,
    ) -> Any:
        torch = self.torch

        from types import MethodType

        module = torch.nn.Module()
        if family == "gpt2":
            def gpt2_forward(_module: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
                del _module, args
                output: tuple[Any, ...] = (cached_hidden,)
                if bool(kwargs.get("use_cache", False)):
                    output += (cached_past,)
                if bool(kwargs.get("output_attentions", False)):
                    output += (None,)
                return output

            module.forward = MethodType(gpt2_forward, module)
            return module

        def llama_forward(_module: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            del _module, args
            cache = kwargs.get("past_key_value")
            if cache is not None and hasattr(cache, "update"):
                key, value = cached_past[:2]
                cache.update(key, value, layer_index, {})
                present = cache
            else:
                present = cached_past
            output: tuple[Any, ...] = (cached_hidden,)
            if bool(kwargs.get("output_attentions", False)):
                output += (None,)
            if bool(kwargs.get("use_cache", False)):
                output += (present,)
            return output

        module.forward = MethodType(llama_forward, module)
        return module

    def _delta_correct_prefix_cache(
        self,
        *,
        baseline: BackendOutput,
        full: BackendOutput,
        decision: StrategyDecision,
    ) -> BackendOutput:
        del full
        if not baseline.extras:
            raise ValueError("Delta correction requires cached HF state from the baseline output")
        split_layer = decision.first_invalid_layer or 0
        corrected_past, corrected_lora_cache = self._apply_lora_weight_delta_to_past(
            baseline.extras["past_key_values"],
            baseline.extras.get("lora_cache", {}),
            split_layer=split_layer,
        )
        if corrected_past is None:
            result = self._reuse_old_prefix_cache(baseline)
            if result.extras is not None:
                result.extras["delta_mode"] = "unavailable_weight_delta_fallback_to_stale"
            self._last_delta_s = self._last_stale_s
            return result
        result = self._probe_with_past(
            baseline=baseline,
            past=corrected_past,
            cache_tensor=self._summarize_past(corrected_past),
            hidden_tensor=baseline.hidden_tensor,
            lora_cache=corrected_lora_cache,
            extra_metadata={"delta_mode": "lora_weight_delta"},
        )
        self._last_delta_s = float(result.extras.get("strategy_latency", 0.0)) if result.extras else 0.0
        return result

    def _probe_with_past(
        self,
        *,
        baseline: BackendOutput,
        past: Any,
        cache_tensor: np.ndarray,
        hidden_tensor: np.ndarray,
        lora_cache: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> BackendOutput:
        if not baseline.extras:
            raise ValueError("Baseline output does not contain prompt state")
        state = baseline.extras["prompt_state"]
        synchronize(self.torch, self.device)
        start = time.perf_counter()
        with self.torch.no_grad():
            probe_logits, generated_text, decode_s = self._generate_answer(state, past)
        synchronize(self.torch, self.device)
        latency = time.perf_counter() - start
        extras = {
            "past_key_values": past,
            "prompt_state": state,
            "lora_cache": lora_cache if lora_cache is not None else baseline.extras.get("lora_cache", {}),
            "memory_allocated": memory_allocated(self.torch, self.device),
            "strategy_latency": latency,
            "generated_text": generated_text,
            "decode_latency": decode_s,
        }
        if extra_metadata:
            extras.update(extra_metadata)
        return BackendOutput(
            logits=self._to_numpy(probe_logits),
            cache_tensor=cache_tensor,
            hidden_tensor=hidden_tensor,
            parameter_version=self.parameter_version,
            extras=extras,
        )

    def _set_lora_capture(self, enabled: bool) -> None:
        for module in self._lora_modules:
            if hasattr(module, "capture_lora_input"):
                module.capture_lora_input = enabled
                if enabled:
                    module.cached_lora_input = None

    def _snapshot_lora_cache(self) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        for module in self._lora_modules:
            name = str(getattr(module, "lora_name", ""))
            if not name or not hasattr(module, "lora_state"):
                continue
            state = module.lora_state()
            cached_input = getattr(module, "cached_lora_input", None)
            if cached_input is not None:
                state["input"] = cached_input.detach()
            state["name"] = name
            state["layer"] = self._module_layer(name)
            state["projection"] = self._module_projection(name)
            cache[name] = state
        return cache

    def _apply_lora_weight_delta_to_past(
        self,
        past_key_values: Any,
        old_lora_cache: Any,
        *,
        split_layer: int,
    ) -> tuple[Any | None, dict[str, Any]]:
        if not isinstance(old_lora_cache, dict) or not old_lora_cache:
            return None, {}
        module_by_name = {
            str(getattr(module, "lora_name", "")): module
            for module in self._lora_modules
            if getattr(module, "lora_name", "")
        }
        layers = [tuple(layer) for layer in past_key_values]
        corrected = [list(layer) for layer in layers]
        new_lora_cache: dict[str, Any] = {}
        applied = False
        for name, old_state in old_lora_cache.items():
            if not isinstance(old_state, dict):
                continue
            module = module_by_name.get(str(name))
            if module is None or not hasattr(module, "lora_delta_output"):
                continue
            projection = str(old_state.get("projection", ""))
            if projection not in {"k", "v"}:
                continue
            layer = old_state.get("layer")
            if not isinstance(layer, int) or layer < split_layer or layer >= len(corrected):
                continue
            cached_input = old_state.get("input")
            if cached_input is None:
                continue
            delta = module.lora_delta_output(cached_input, old_state)
            item_index = 0 if projection == "k" else 1
            if item_index >= len(corrected[layer]):
                continue
            projected = self._reshape_projection_delta(delta, corrected[layer][item_index])
            if projected is None:
                continue
            corrected[layer][item_index] = corrected[layer][item_index] + projected.to(
                device=corrected[layer][item_index].device,
                dtype=corrected[layer][item_index].dtype,
            )
            state = module.lora_state()
            state["input"] = cached_input.detach()
            state["name"] = name
            state["layer"] = layer
            state["projection"] = projection
            new_lora_cache[str(name)] = state
            applied = True
        if not applied:
            return None, {}
        for name, old_state in old_lora_cache.items():
            if str(name) not in new_lora_cache and isinstance(old_state, dict):
                module = module_by_name.get(str(name))
                if module is not None and hasattr(module, "lora_state"):
                    state = module.lora_state()
                    if "input" in old_state:
                        state["input"] = old_state["input"]
                    state["name"] = name
                    state["layer"] = old_state.get("layer")
                    state["projection"] = old_state.get("projection")
                    new_lora_cache[str(name)] = state
        return tuple(tuple(layer) for layer in corrected), new_lora_cache

    def _reshape_projection_delta(self, delta: Any, target_past: Any) -> Any | None:
        if not hasattr(delta, "reshape") or not hasattr(target_past, "shape"):
            return None
        if delta.ndim != 3 or target_past.ndim != 4:
            return None
        batch, seq_len, features = delta.shape
        if int(target_past.shape[0]) != int(batch):
            return None
        if int(target_past.shape[2]) == int(seq_len):
            heads = int(target_past.shape[1])
            head_dim = int(target_past.shape[3])
            if features != heads * head_dim:
                return None
            return delta.reshape(batch, seq_len, heads, head_dim).transpose(1, 2).contiguous()
        if int(target_past.shape[1]) == int(seq_len):
            heads = int(target_past.shape[2])
            head_dim = int(target_past.shape[3])
            if features != heads * head_dim:
                return None
            return delta.reshape(batch, seq_len, heads, head_dim).contiguous()
        return None

    def _module_layer(self, name: str) -> int | None:
        import re

        for pattern in (r"layers\.(\d+)", r"h\.(\d+)", r"blocks?\.(\d+)", r"decoder\.(\d+)"):
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None

    def _module_projection(self, name: str) -> str:
        lower = name.lower()
        if "k_proj" in lower or ".key" in lower or lower.endswith("key"):
            return "k"
        if "v_proj" in lower or ".value" in lower or lower.endswith("value"):
            return "v"
        if "q_proj" in lower or ".query" in lower or lower.endswith("query"):
            return "q"
        return "unknown"

    def _select_parameters(self, target: UpdateTarget) -> list[Any]:
        filters = self._target_filters(target.kind)
        selected = []
        for name, param in self.model.named_parameters():
            lower = name.lower()
            if target.layer is not None and not self._name_matches_layer(lower, target.layer):
                continue
            if any(part in lower for part in filters):
                selected.append(param)
        if target.layer is not None and not selected:
            return self._select_parameters(UpdateTarget(kind=target.kind, layer=None, raw=target.raw))
        return selected[:4]

    def _target_filters(self, kind: ModuleKind) -> tuple[str, ...]:
        mapping = {
            ModuleKind.ATTENTION_Q: ("q_proj", "query", "c_attn"),
            ModuleKind.ATTENTION_K: ("k_proj", "key", "c_attn"),
            ModuleKind.ATTENTION_V: ("v_proj", "value", "c_attn"),
            ModuleKind.ATTENTION_O: ("o_proj", "out_proj", "attn.c_proj", "attention.dense"),
            ModuleKind.ATTENTION_QV: ("q_proj", "query", "v_proj", "value", "c_attn"),
            ModuleKind.ATTENTION_ATTN: (
                "q_proj",
                "query",
                "k_proj",
                "key",
                "v_proj",
                "value",
                "o_proj",
                "out_proj",
                "c_attn",
                "attn.c_proj",
                "attention.dense",
            ),
            ModuleKind.MLP: ("mlp", "gate_proj", "up_proj", "down_proj", "mlp.c_fc", "mlp.c_proj"),
            ModuleKind.NORM: ("norm", "ln_"),
            ModuleKind.OUTPUT_HEAD: ("lm_head",),
            ModuleKind.LORA_Q: ("q_proj", "query", "c_attn"),
            ModuleKind.LORA_K: ("k_proj", "key", "c_attn"),
            ModuleKind.LORA_V: ("v_proj", "value", "c_attn"),
            ModuleKind.LORA_O: ("o_proj", "out_proj", "attn.c_proj", "attention.dense"),
            ModuleKind.LORA_QV: ("q_proj", "query", "v_proj", "value", "c_attn"),
            ModuleKind.LORA_ATTN: (
                "q_proj",
                "query",
                "k_proj",
                "key",
                "v_proj",
                "value",
                "o_proj",
                "out_proj",
                "c_attn",
                "attn.c_proj",
                "attention.dense",
            ),
            ModuleKind.LORA_ALL_LATE: (
                "q_proj",
                "query",
                "k_proj",
                "key",
                "v_proj",
                "value",
                "o_proj",
                "out_proj",
                "mlp",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            ModuleKind.LORA_MLP: ("mlp", "gate_proj", "up_proj", "down_proj", "mlp.c_fc", "mlp.c_proj"),
            ModuleKind.UNKNOWN: tuple(),
        }
        return mapping.get(kind, tuple())

    def _name_matches_layer(self, name: str, layer: int) -> bool:
        candidates = (
            f"layers.{layer}.",
            f"h.{layer}.",
            f"block.{layer}.",
            f"blocks.{layer}.",
            f"decoder.{layer}.",
        )
        return any(candidate in name for candidate in candidates)

    def _splice_past(self, old_past: Any, new_past: Any, *, split_layer: int) -> Any:
        old_layers = list(old_past)
        new_layers = list(new_past)
        merged = [
            old if idx < split_layer else new
            for idx, (old, new) in enumerate(zip(old_layers, new_layers, strict=True))
        ]
        return tuple(merged)

    def _blend_past(self, old_past: Any, new_past: Any, *, split_layer: int, alpha: float) -> Any:
        old_layers = list(old_past)
        new_layers = list(new_past)
        blended = []
        for idx, (old_layer, new_layer) in enumerate(zip(old_layers, new_layers, strict=True)):
            if idx < split_layer:
                blended.append(old_layer)
                continue
            blended.append(self._blend_past_layer(old_layer, new_layer, alpha=alpha))
        return tuple(blended)

    def _blend_past_layer(self, old_layer: Any, new_layer: Any, *, alpha: float) -> Any:
        old_items = list(old_layer)
        new_items = list(new_layer)
        out = []
        for idx, (old_item, new_item) in enumerate(zip(old_items, new_items, strict=True)):
            if idx < 2 and hasattr(old_item, "detach") and hasattr(new_item, "detach"):
                out.append(old_item + (new_item - old_item) * alpha)
            else:
                out.append(new_item)
        return tuple(out)

    def _splice_summary(self, old: np.ndarray, new: np.ndarray, *, split_layer: int) -> np.ndarray:
        if old.shape != new.shape:
            return new
        merged = old.copy()
        merged[split_layer:] = new[split_layer:]
        return merged

    def _blend_summary(self, old: np.ndarray, new: np.ndarray, *, split_layer: int, alpha: float) -> np.ndarray:
        if old.shape != new.shape:
            return new
        blended = old.copy()
        blended[split_layer:] = old[split_layer:] + (new[split_layer:] - old[split_layer:]) * alpha
        return blended

    def _summarize_past(self, past_key_values: Any) -> np.ndarray:
        rows = []
        for layer in past_key_values:
            key, value = layer[:2]
            key_summary = key.detach().float().mean(dim=tuple(range(key.ndim - 1))).cpu().numpy()
            value_summary = value.detach().float().mean(dim=tuple(range(value.ndim - 1))).cpu().numpy()
            rows.append(np.stack([key_summary, value_summary], axis=0))
        return np.stack(rows, axis=0).astype(np.float64)

    def _summarize_hidden(self, hidden_states: Any) -> np.ndarray:
        if not hidden_states:
            return np.zeros((1, 1), dtype=np.float64)
        rows = []
        for hidden in list(hidden_states)[1:]:
            summary = hidden.detach().float().mean(dim=tuple(range(hidden.ndim - 1))).cpu().numpy()
            rows.append(summary)
        if not rows:
            return np.zeros((1, 1), dtype=np.float64)
        return np.stack(rows, axis=0).astype(np.float64)

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        result = tensor.detach().float().cpu().numpy().astype(np.float64)
        return cast(np.ndarray, result)
