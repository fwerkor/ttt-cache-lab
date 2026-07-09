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
        with self.torch.no_grad():
            prefill = self.model(input_ids=state.prefix_ids, use_cache=True, output_hidden_states=True)
            probe = self.model(input_ids=state.probe_ids, past_key_values=prefill.past_key_values, use_cache=True)
        synchronize(self.torch, self.device)
        self._last_prefill_s = time.perf_counter() - start
        return BackendOutput(
            logits=self._to_numpy(probe.logits[:, -1, :]),
            cache_tensor=self._summarize_past(prefill.past_key_values),
            hidden_tensor=self._summarize_hidden(prefill.hidden_states),
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": prefill.past_key_values,
                "prompt_state": state,
                "memory_allocated": memory_allocated(self.torch, self.device),
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
        with self.torch.no_grad():
            prefill = self.model(input_ids=state.prefix_ids, use_cache=True, output_hidden_states=True)
            probe = self.model(input_ids=state.probe_ids, past_key_values=prefill.past_key_values, use_cache=True)
        synchronize(self.torch, self.device)
        self._last_prefill_s = time.perf_counter() - start
        return BackendOutput(
            logits=self._to_numpy(probe.logits[:, -1, :]),
            cache_tensor=self._summarize_past(prefill.past_key_values),
            hidden_tensor=self._summarize_hidden(prefill.hidden_states),
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": prefill.past_key_values,
                "prompt_state": state,
                "memory_allocated": memory_allocated(self.torch, self.device),
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
        if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REUSE_EXACT}:
            return full
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            return self._partial_recompute_prefix_cache(baseline=baseline, full=full, decision=decision)
        if decision.action is CacheAction.DELTA_CORRECT:
            return self._delta_correct_prefix_cache(baseline=baseline, full=full, decision=decision)
        if decision.action in {CacheAction.REUSE_STALE, CacheAction.REUSE_FROZEN}:
            return self._reuse_old_prefix_cache(baseline)
        return full

    def score_answer(self, sample: TaskSample, output: BackendOutput) -> float:
        logits = output.logits[0]
        top_token = int(np.argmax(logits))
        decoded = self.tokenizer.decode([top_token]).strip()
        return 1.0 if decoded and sample.answer and sample.answer.startswith(decoded) else 0.0

    def estimate_latency(self, decision: StrategyDecision, *, context_length: int) -> float:
        if decision.action is CacheAction.FULL_RECOMPUTE:
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
                if id(module) not in seen_active:
                    self._activate_lora_module(module)
                    seen_active.add(id(module))
                    replaced += 1
                continue
            if isinstance(module, nn.Linear):
                wrapped = make_lora_linear(self.torch, nn, module, rank=rank, alpha=alpha)
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
        count = self.setup_lora(target, rank=rank, alpha=alpha, freeze_base_model=freeze_base_model)
        if count == 0:
            raise ValueError(f"No Linear modules matched LoRA target {target.raw!r}")
        self.model.train()
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            self.model.eval()
            return 0.0
        labels = input_ids.clone()
        outputs = self.model(input_ids=input_ids, labels=labels, use_cache=False)
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
        if not baseline.extras or not full.extras:
            raise ValueError("Partial recompute requires cached HF states from baseline and full outputs")
        split_layer = decision.first_invalid_layer or 0
        merged_past = self._splice_past(
            baseline.extras["past_key_values"],
            full.extras["past_key_values"],
            split_layer=split_layer,
        )
        hidden = self._splice_summary(baseline.hidden_tensor, full.hidden_tensor, split_layer=split_layer)
        result = self._probe_with_past(
            baseline=baseline,
            past=merged_past,
            cache_tensor=self._summarize_past(merged_past),
            hidden_tensor=hidden,
        )
        self._last_partial_s = float(result.extras.get("strategy_latency", 0.0)) if result.extras else 0.0
        return result

    def _delta_correct_prefix_cache(
        self,
        *,
        baseline: BackendOutput,
        full: BackendOutput,
        decision: StrategyDecision,
    ) -> BackendOutput:
        if not baseline.extras or not full.extras:
            raise ValueError("Delta correction requires cached HF states from baseline and full outputs")
        split_layer = decision.first_invalid_layer or 0
        alpha = 0.75
        corrected_past = self._blend_past(
            baseline.extras["past_key_values"],
            full.extras["past_key_values"],
            split_layer=split_layer,
            alpha=alpha,
        )
        hidden = self._blend_summary(baseline.hidden_tensor, full.hidden_tensor, split_layer=split_layer, alpha=alpha)
        result = self._probe_with_past(
            baseline=baseline,
            past=corrected_past,
            cache_tensor=self._summarize_past(corrected_past),
            hidden_tensor=hidden,
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
    ) -> BackendOutput:
        if not baseline.extras:
            raise ValueError("Baseline output does not contain prompt state")
        state = baseline.extras["prompt_state"]
        synchronize(self.torch, self.device)
        start = time.perf_counter()
        with self.torch.no_grad():
            probe = self.model(input_ids=state.probe_ids, past_key_values=past, use_cache=True)
        synchronize(self.torch, self.device)
        latency = time.perf_counter() - start
        return BackendOutput(
            logits=self._to_numpy(probe.logits[:, -1, :]),
            cache_tensor=cache_tensor,
            hidden_tensor=hidden_tensor,
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": past,
                "prompt_state": state,
                "memory_allocated": memory_allocated(self.torch, self.device),
                "strategy_latency": latency,
            },
        )

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
