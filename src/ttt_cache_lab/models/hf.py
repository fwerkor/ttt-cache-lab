from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ttt_cache_lab.cache.semantics import CacheAction
from ttt_cache_lab.cache.strategies import StrategyDecision, StrategyName
from ttt_cache_lab.data.scoring import score_prediction
from ttt_cache_lab.data.synthetic import TaskSample, neutral_background_sentences
from ttt_cache_lab.models.accelerator import (
    max_memory_allocated,
    memory_allocated,
    reset_peak_memory,
    resolve_device,
    synchronize,
)
from ttt_cache_lab.models.interface import BackendOutput
from ttt_cache_lab.updates.targets import ModuleKind, UpdateTarget


@dataclass(frozen=True)
class _PromptState:
    prompt: str
    input_ids: Any
    prefix_ids: Any
    probe_ids: Any
    attention_mask: Any
    activation_boundary: int | None = None


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
        use_chat_template: bool = False,
        revision: str | None = None,
        attention_implementation: str | None = None,
        parallelism: str = "single",
        device_ids: list[int] | None = None,
        seed: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps
            raise RuntimeError("Install the HF backend with: pip install -e '.[hf]'") from exc

        self.torch = torch
        self.seed = seed
        torch.manual_seed(seed)
        dtype = self._resolve_dtype(torch_dtype)
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            load_kwargs["revision"] = revision
        if attention_implementation is not None:
            load_kwargs["attn_implementation"] = attention_implementation
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        tokenizer_factory = cast(Any, AutoTokenizer)
        model_factory = cast(Any, AutoModelForCausalLM)
        config_factory = cast(Any, AutoConfig)
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
        self.tokenizer: Any = tokenizer_factory.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.use_chat_template = use_chat_template
        if self.use_chat_template and not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("model.use_chat_template=true requires a tokenizer with a configured chat template")
        if parallelism == "model_shard":
            from ttt_cache_lab.models.sharding import build_model_shard_plan, resolve_shard_device_ids

            resolved = self._resolve_device(device)
            device_type = resolved.split(":", maxsplit=1)[0]
            ids = resolve_shard_device_ids(
                torch,
                device_type=device_type,
                configured=list(device_ids or []),
            )
            config_kwargs: dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
            }
            if revision is not None:
                config_kwargs["revision"] = revision
            model_config = config_factory.from_pretrained(
                model_name_or_path,
                **config_kwargs,
            )
            shard_plan = build_model_shard_plan(
                model_config,
                device_type=device_type,
                device_ids=ids,
            )
            load_kwargs["device_map"] = shard_plan.device_map
            load_kwargs["low_cpu_mem_usage"] = True
            self.device = shard_plan.input_device
            self.devices = shard_plan.devices
            self.model: Any = model_factory.from_pretrained(model_name_or_path, **load_kwargs)
            self.parallelism = "model_shard"
            self.layer_to_device = shard_plan.layer_to_device
        elif parallelism == "single":
            self.device = self._resolve_device(device)
            self.devices = (self.device,)
            self.model = model_factory.from_pretrained(model_name_or_path, **load_kwargs)
            self.model.to(self.device)
            self.parallelism = "single"
            self.layer_to_device = tuple(
                self.device for _ in range(self._infer_num_layers_from_config(self.model.config))
            )
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism}")
        self.model.eval()
        self._stop_token_ids = self._resolve_stop_token_ids()
        self.num_layers = self._infer_num_layers()
        self._parameter_count = sum(int(parameter.numel()) for parameter in self.model.parameters())
        self.max_length = max_length
        self.parameter_version = 0
        self._deltas: list[tuple[Any, Any]] = []
        self._last_prefill_s = 0.0
        self._last_stale_s = 0.0
        self._last_partial_s = 0.0
        self._last_delta_s = 0.0
        self._last_alora_s = 0.0
        self._last_adaptation_s = 0.0
        self._last_raw_update_norm = 0.0
        self._last_applied_update_norm = 0.0
        self._last_updated_parameter_count = 0
        self._last_applied_update_rms = 0.0
        self._lora_modules: list[Any] = []
        self._active_lora_modules: list[Any] = []
        self._prepared_input_ids: dict[str, Any] = {}
        self._sample_answer_token_counts: dict[str, int] = {}
        self._sample_activation_boundaries: dict[str, int] = {}
        self._neutral_padding_pool: tuple[int, ...] | None = None
        self._alora_base_cache: dict[str, tuple[Any, np.ndarray]] = {}
        self._capture_attention_metrics = False
        self._last_attention_summary: np.ndarray | None = None
        self._last_attention_input_summary: np.ndarray | None = None
        self._last_attention_output_summary: np.ndarray | None = None
        self._last_correction_flops = 0.0
        self._last_low_rank_cache_bytes = 0
        self._last_residual_cache_bytes = 0

    @property
    def hidden_size(self) -> int:
        return self._hidden_size()

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def _infer_num_layers(self) -> int:
        return self._infer_num_layers_from_config(self.model.config)

    def _infer_num_layers_from_config(self, config: Any) -> int:
        for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError(f"Cannot infer transformer layer count from {type(config).__name__}")

    def prepare_sample(self, sample: TaskSample, *, context_length: int) -> TaskSample:
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        if context_length > self.max_length:
            raise ValueError(f"Requested context_length={context_length} exceeds model.max_length={self.max_length}")

        input_ids, content_start, content_end = self._tokenize_prompt_ids(sample.prompt)
        current = int(input_ids.shape[1])
        activation_boundary = self._activation_boundary(
            input_ids,
            str(sample.metadata.get("adapter_activation_marker", "")),
        )

        if current > context_length:
            strategy = str(sample.metadata.get("truncation_strategy", "error"))
            if strategy == "error":
                raise ValueError(
                    "Prompt tokenization exceeded the requested context length; "
                    f"generated={current}, requested={context_length}. "
                    "Set data.truncation_strategy to left or middle for external datasets."
                )
            if strategy not in {"left", "middle"}:
                raise ValueError(f"Unsupported truncation strategy: {strategy}")

            if self.use_chat_template:
                prefix = input_ids[:, :content_start]
                content = input_ids[:, content_start:content_end]
                suffix = input_ids[:, content_end:]
                available = context_length - int(prefix.shape[1]) - int(suffix.shape[1])
                if available < 1:
                    raise ValueError("Chat-template control tokens leave no room for prompt content")
                content_length = int(content.shape[1])
                activation_in_content = activation_boundary - content_start if activation_boundary is not None else None
                if strategy == "left":
                    removed = content_length - available
                    content = content[:, -available:]
                    if activation_in_content is not None:
                        activation_in_content -= removed
                else:
                    left = (available + 1) // 2
                    right = available - left
                    right_start = content_length - right
                    content = self.torch.cat(
                        [content[:, :left], content[:, -right:] if right else content[:, :0]],
                        dim=1,
                    )
                    if activation_in_content is not None:
                        if activation_in_content <= left:
                            pass
                        elif activation_in_content >= right_start:
                            activation_in_content = left + activation_in_content - right_start
                        else:
                            raise ValueError("Middle truncation removed the adapter activation marker")
                input_ids = self.torch.cat([prefix, content, suffix], dim=1)
                if activation_in_content is not None:
                    activation_boundary = int(prefix.shape[1]) + activation_in_content
            else:
                original_length = current
                if strategy == "left":
                    removed = current - context_length
                    input_ids = input_ids[:, -context_length:]
                    if activation_boundary is not None:
                        activation_boundary -= removed
                else:
                    left = (context_length + 1) // 2
                    right = context_length - left
                    right_start = original_length - right
                    input_ids = self.torch.cat(
                        [input_ids[:, :left], input_ids[:, -right:] if right else input_ids[:, :0]],
                        dim=1,
                    )
                    if activation_boundary is not None:
                        if activation_boundary <= left:
                            pass
                        elif activation_boundary >= right_start:
                            activation_boundary = left + activation_boundary - right_start
                        else:
                            raise ValueError("Middle truncation removed the adapter activation marker")
            current = int(input_ids.shape[1])

        neutral_padding_tokens = 0
        if current < context_length:
            pad_count = context_length - current
            padding = self._neutral_padding_ids(
                pad_count,
                dtype=input_ids.dtype,
                prompt=sample.prompt,
            )
            insertion = content_start if self.use_chat_template else 0
            input_ids = self.torch.cat(
                [input_ids[:, :insertion], padding, input_ids[:, insertion:]],
                dim=1,
            )
            neutral_padding_tokens = pad_count
            if activation_boundary is not None and activation_boundary >= insertion:
                activation_boundary += pad_count

        if activation_boundary is not None:
            if activation_boundary <= 0 or activation_boundary >= context_length:
                raise ValueError("Adapter activation marker must remain inside the retained prompt context")
            self._sample_activation_boundaries[sample.prompt] = activation_boundary
        self._prepared_input_ids[sample.prompt] = input_ids
        answer_ids = self.tokenizer(sample.answer, add_special_tokens=False).get("input_ids", [])
        reference_tokens = max(1, len(answer_ids))
        configured_limit = max(1, int(sample.metadata.get("max_generation_tokens", reference_tokens)))
        self._sample_answer_token_counts[sample.prompt] = configured_limit
        metadata = dict(sample.metadata)
        metadata["token_length"] = context_length
        metadata["neutral_padding_tokens"] = neutral_padding_tokens
        metadata["prompt_format"] = "chat_template" if self.use_chat_template else "plain"
        return TaskSample(prompt=sample.prompt, answer=sample.answer, metadata=metadata)

    def _tokenize_prompt_ids(self, prompt: str) -> tuple[Any, int, int]:
        if not self.use_chat_template:
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"]
            return input_ids, 0, int(input_ids.shape[1])

        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        content_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        start = self._find_token_subsequence(
            input_ids[0].tolist(),
            content_ids[0].tolist(),
        )
        if start is None:
            raise RuntimeError("Could not locate user content inside the tokenizer chat template")
        return input_ids, start, start + int(content_ids.shape[1])

    def _activation_boundary(self, input_ids: Any, marker: str) -> int | None:
        if not marker:
            return None
        marker_ids = self.tokenizer(marker, add_special_tokens=False).get("input_ids", [])
        if not marker_ids:
            raise ValueError("Configured adapter activation marker tokenized to an empty sequence")
        start = self._find_token_subsequence(input_ids[0].tolist(), list(marker_ids))
        if start is None:
            raise ValueError("Configured adapter activation marker is absent from the tokenized prompt")
        return start + len(marker_ids)

    @staticmethod
    def _find_token_subsequence(tokens: list[int], subsequence: list[int]) -> int | None:
        if not subsequence or len(subsequence) > len(tokens):
            return None
        stop = len(tokens) - len(subsequence) + 1
        for index in range(stop):
            if tokens[index : index + len(subsequence)] == subsequence:
                return index
        return None

    def _neutral_padding_ids(self, count: int, *, dtype: Any, prompt: str) -> Any:
        if count < 1:
            return self.torch.empty((1, 0), dtype=dtype)
        if self._neutral_padding_pool is None:
            background = " ".join(neutral_background_sentences(512))
            raw_ids = self.tokenizer(background, add_special_tokens=False).get("input_ids", [])
            special_ids = {int(value) for value in getattr(self.tokenizer, "all_special_ids", [])}
            pool = tuple(int(value) for value in raw_ids if int(value) not in special_ids)
            if not pool:
                fallback = int(self.tokenizer.unk_token_id or self.tokenizer.eos_token_id or 0)
                pool = (fallback,)
            self._neutral_padding_pool = pool
        pool = self._neutral_padding_pool
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], byteorder="big") % len(pool)
        rotated = pool[offset:] + pool[:offset]
        repeats = (count + len(rotated) - 1) // len(rotated)
        values = (rotated * repeats)[:count]
        return self.torch.tensor([values], dtype=dtype)

    def _resolve_device(self, device: str) -> str:
        return resolve_device(self.torch, device)

    def _resolve_stop_token_ids(self) -> frozenset[int]:
        stop_ids: set[int] = set()
        for raw in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
        ):
            if isinstance(raw, int):
                stop_ids.add(raw)
            elif isinstance(raw, list | tuple | set):
                stop_ids.update(int(value) for value in raw if isinstance(value, int))
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            for token in ("<|im_end|>", "<|eot_id|>", "<|end_of_turn|>"):
                token_id = convert(token)
                if isinstance(token_id, int) and token_id >= 0:
                    stop_ids.add(token_id)
        return frozenset(stop_ids)

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
        reset_peak_memory(self.torch, self.devices)
        synchronize(self.torch, self.devices)
        start = time.perf_counter()
        self._set_lora_capture(True)
        with self.torch.no_grad():
            prefill = self.model(input_ids=state.prefix_ids, use_cache=True, output_hidden_states=True)
            self._set_lora_capture(False)
        synchronize(self.torch, self.devices)
        prefill_s = time.perf_counter() - start
        with self.torch.no_grad():
            probe_logits, generated_text, decode_s, generated_tokens = self._generate_answer(
                state, prefill.past_key_values
            )
        self._last_prefill_s = prefill_s
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
                "memory_allocated": memory_allocated(self.torch, self.devices),
                "peak_memory_allocated": max_memory_allocated(self.torch, self.devices),
                "cache_bytes": self._past_nbytes(prefill.past_key_values),
                "token_length": int(state.input_ids.shape[1]),
                "attention_implementation": self._attention_implementation(),
                "generated_text": generated_text,
                "generated_tokens": generated_tokens,
                "prefill_latency": prefill_s,
                "cache_maintenance_latency": prefill_s,
                "decode_latency": decode_s,
                "strategy_latency": prefill_s + decode_s,
                "strategy_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1])),
                "full_recompute_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1])),
                **self._attention_metadata(),
                **self._alora_base_extras(state),
            },
        )

    def simulate_update(self, baseline: BackendOutput, target: UpdateTarget, *, update_norm: float) -> BackendOutput:
        selected = self._select_parameters(target)
        if not selected:
            raise ValueError(f"No HF parameters matched update target {target.raw!r}")
        if update_norm < 0.0:
            raise ValueError("update_norm must be non-negative")
        self.torch.manual_seed(self.seed + self.parameter_version + 1)
        synchronize(self.torch, self.devices)
        start = time.perf_counter()
        pending: list[tuple[Any, Any]] = []
        squared_norm = self.torch.zeros((), dtype=self.torch.float64, device=self.device)
        for param in selected:
            if not param.is_floating_point():
                continue
            noise = self.torch.randn_like(param)
            pending.append((param, noise))
            squared_norm += self.torch.sum(noise.detach().double() ** 2).to(self.device)
        raw_norm = float(self.torch.sqrt(squared_norm).cpu())
        scale = update_norm / raw_norm if raw_norm > 0.0 else 0.0
        self._last_raw_update_norm = raw_norm
        self._last_applied_update_norm = update_norm if raw_norm > 0.0 else 0.0
        with self.torch.no_grad():
            for param, noise in pending:
                delta = noise * scale
                param.add_(delta)
                self._deltas.append((param, delta))
        synchronize(self.torch, self.devices)
        self._last_adaptation_s = time.perf_counter() - start
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
        reset_peak_memory(self.torch, self.devices)
        synchronize(self.torch, self.devices)
        start = time.perf_counter()
        self._set_lora_capture(True)
        with self.torch.no_grad():
            prefill = self.model(input_ids=state.prefix_ids, use_cache=True, output_hidden_states=True)
            self._set_lora_capture(False)
        synchronize(self.torch, self.devices)
        prefill_s = time.perf_counter() - start
        with self.torch.no_grad():
            probe_logits, generated_text, decode_s, generated_tokens = self._generate_answer(
                state, prefill.past_key_values
            )
        self._last_prefill_s = prefill_s
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
                "memory_allocated": memory_allocated(self.torch, self.devices),
                "peak_memory_allocated": max_memory_allocated(self.torch, self.devices),
                "cache_bytes": self._past_nbytes(prefill.past_key_values),
                "token_length": int(state.input_ids.shape[1]),
                "attention_implementation": self._attention_implementation(),
                "generated_text": generated_text,
                "generated_tokens": generated_tokens,
                "prefill_latency": prefill_s,
                "cache_maintenance_latency": prefill_s,
                "decode_latency": decode_s,
                "strategy_latency": prefill_s + decode_s,
                "strategy_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1])),
                "full_recompute_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1])),
                **self._attention_metadata(),
                **self._alora_base_extras(state),
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
            if decision.strategy is StrategyName.NO_ADAPTATION:
                return self._cached_output_with_runtime(baseline, cache_mode="no_adaptation")
            return self._reuse_old_prefix_cache(baseline, cache_mode="exact_reuse")
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            return self._partial_recompute_prefix_cache(baseline=baseline, full=full, decision=decision)
        if decision.action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
            return self._alora_suffix_recompute(baseline)
        if decision.action is CacheAction.DELTA_CORRECT:
            return self._delta_correct_prefix_cache(baseline=baseline, full=full, decision=decision)
        if decision.action in {CacheAction.REUSE_STALE, CacheAction.REUSE_FROZEN}:
            return self._reuse_old_prefix_cache(baseline)
        return full

    def score_answer(self, sample: TaskSample, output: BackendOutput) -> float:
        extras = output.extras or {}
        generated = extras.get("generated_text")
        if isinstance(generated, str):
            return score_prediction(sample, generated)
        logits = output.logits[0]
        top_token = int(np.argmax(logits))
        decoded = self.tokenizer.decode([top_token]).strip()
        return score_prediction(sample, decoded)

    def estimate_latency(self, decision: StrategyDecision, *, context_length: int) -> float:
        if decision.action in {CacheAction.FULL_RECOMPUTE, CacheAction.REJECT_UPDATE}:
            return self._last_prefill_s or 1.0
        if decision.action in {CacheAction.REUSE_STALE, CacheAction.REUSE_FROZEN}:
            return self._last_stale_s or max(1e-6, (self._last_prefill_s or 1.0) / 10.0)
        if decision.action is CacheAction.PARTIAL_RECOMPUTE:
            return getattr(self, "_last_partial_s", 0.0) or max(1e-6, (self._last_prefill_s or 1.0) * 0.5)
        if decision.action is CacheAction.DELTA_CORRECT:
            return getattr(self, "_last_delta_s", 0.0) or max(1e-6, (self._last_prefill_s or 1.0) * 0.15)
        if decision.action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
            return self._last_alora_s or max(1e-6, (self._last_prefill_s or 1.0) * 0.25)
        return max(1e-6, (self._last_prefill_s or 1.0) / 10.0)

    def configure_metrics(self, *, capture_attention: bool) -> None:
        self._capture_attention_metrics = capture_attention

    def estimate_flops(self, decision: StrategyDecision, *, context_length: int) -> float:
        full = self._full_prefill_flops(max(1, context_length))
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
            return full * max(0, end - first) / max(1, self.num_layers)
        if decision.action is CacheAction.DELTA_CORRECT:
            if self._last_correction_flops > 0.0:
                return self._last_correction_flops
            rank = max((int(getattr(module, "rank", 0)) for module in self._active_lora_modules), default=8)
            hidden = self._hidden_size()
            return 4.0 * max(1, context_length) * hidden * max(1, rank)
        if decision.action is CacheAction.ALORA_SUFFIX_RECOMPUTE:
            return full * (decision.recompute_fraction or 0.25)
        return 0.0

    def _full_prefill_flops(self, tokens: int) -> float:
        hidden = self._hidden_size()
        intermediate = self._intermediate_size(hidden)
        heads = max(1, int(getattr(self.model.config, "num_attention_heads", 1) or 1))
        kv_heads = int(getattr(self.model.config, "num_key_value_heads", heads) or heads)
        kv_width = hidden * kv_heads / heads
        projection = 2.0 * tokens * hidden * (2.0 * hidden + 2.0 * kv_width)
        attention = 4.0 * tokens * tokens * hidden
        mlp = 6.0 * tokens * hidden * intermediate
        return self.num_layers * (projection + attention + mlp)

    def _hidden_size(self) -> int:
        for name in ("hidden_size", "n_embd", "d_model"):
            value = getattr(self.model.config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return 1

    def _intermediate_size(self, hidden: int) -> int:
        for name in ("intermediate_size", "n_inner", "ffn_dim"):
            value = getattr(self.model.config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return 4 * hidden

    def last_adaptation_latency(self) -> float:
        return self._last_adaptation_s

    def last_raw_update_norm(self) -> float:
        return self._last_raw_update_norm

    def last_applied_update_norm(self) -> float:
        return self._last_applied_update_norm

    def last_updated_parameter_count(self) -> int:
        return self._last_updated_parameter_count

    def last_applied_update_rms(self) -> float:
        return self._last_applied_update_rms

    def restore_after_update(self) -> None:
        for param, delta in reversed(self._deltas):
            with self.torch.no_grad():
                param.sub_(delta)
        self._deltas.clear()
        self.reset_lora_adapters()
        self.parameter_version = 0
        self._last_adaptation_s = 0.0
        self._last_raw_update_norm = 0.0
        self._last_applied_update_norm = 0.0
        self._last_updated_parameter_count = 0
        self._last_applied_update_rms = 0.0

    def setup_lora(self, target: UpdateTarget, *, rank: int, alpha: float, freeze_base_model: bool = True) -> int:
        from torch import nn

        from ttt_cache_lab.models.lora import is_lora_linear, make_lora_linear

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
            if child_name == "base" and is_lora_linear(parent):
                continue
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
                wrapped.to(module.weight.device)
                setattr(parent, child_name, wrapped)
                self._lora_modules.append(wrapped)
                self._activate_lora_module(wrapped)
                seen_active.add(id(wrapped))
                replaced += 1
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
        target_update_norm: float | None = None,
        target_update_rms: float | None = None,
    ) -> float:
        count = self.setup_lora(target, rank=rank, alpha=alpha, freeze_base_model=freeze_base_model)
        if count == 0:
            raise ValueError(f"No Linear modules matched LoRA target {target.raw!r}")
        self.model.train()
        synchronize(self.torch, self.devices)
        adaptation_start = time.perf_counter()
        prompt_ids = self._prepared_input_ids.get(sample.prompt)
        if prompt_ids is None:
            prompt_ids, _, _ = self._tokenize_prompt_ids(sample.prompt)
            if prompt_ids.shape[1] > self.max_length:
                raise ValueError("Unprepared training prompt exceeds model.max_length; call prepare_sample first")
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
        pending_updates: list[tuple[Any, Any]] = []
        updated_parameter_count = 0
        squared_norm = self.torch.zeros((), dtype=self.torch.float64, device=self.device)
        with self.torch.no_grad():
            for module in self._active_lora_modules:
                for param in module.lora_parameters():
                    if param.grad is None:
                        continue
                    delta = -learning_rate * param.grad
                    pending_updates.append((param, delta))
                    updated_parameter_count += int(param.numel())
                    squared_norm += self.torch.sum(delta.detach().double() ** 2).to(self.device)
            raw_norm = float(self.torch.sqrt(squared_norm).cpu())
            if target_update_norm is not None and target_update_rms is not None:
                raise ValueError("target_update_norm and target_update_rms are mutually exclusive")
            target_l2 = target_update_norm
            if target_update_rms is not None:
                if target_update_rms < 0.0:
                    raise ValueError("target_update_rms must be non-negative")
                target_l2 = target_update_rms * math.sqrt(updated_parameter_count)
            scale = 1.0
            if target_l2 is not None:
                if target_l2 < 0.0:
                    raise ValueError("target_update_norm must be non-negative")
                if raw_norm > 0.0:
                    scale = target_l2 / raw_norm
            applied_norm = raw_norm * scale
            self._last_raw_update_norm = raw_norm
            self._last_applied_update_norm = applied_norm
            self._last_updated_parameter_count = updated_parameter_count
            self._last_applied_update_rms = (
                applied_norm / math.sqrt(updated_parameter_count)
                if updated_parameter_count > 0
                else 0.0
            )
            for param, delta in pending_updates:
                param.add_(delta * scale)
                param.grad = None
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        synchronize(self.torch, self.devices)
        self._last_adaptation_s = time.perf_counter() - adaptation_start
        self.parameter_version += 1
        return self._last_applied_update_norm

    def snapshot_adapter_state(self) -> tuple[tuple[Any, Any], ...]:
        return tuple((module.lora_a.detach().clone(), module.lora_b.detach().clone()) for module in self._lora_modules)

    def load_adapter_state(self, state: tuple[tuple[Any, Any], ...], *, version: int) -> None:
        if len(state) != len(self._lora_modules):
            raise ValueError("Adapter snapshot does not match the installed LoRA modules")
        with self.torch.no_grad():
            for module, (lora_a, lora_b) in zip(self._lora_modules, state, strict=True):
                module.lora_a.copy_(lora_a.to(device=module.lora_a.device, dtype=module.lora_a.dtype))
                module.lora_b.copy_(lora_b.to(device=module.lora_b.device, dtype=module.lora_b.dtype))
        self.parameter_version = version

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
            input_ids, _, _ = self._tokenize_prompt_ids(prompt)
            if input_ids.shape[1] > self.max_length:
                raise ValueError(
                    "Unprepared prompt exceeds model.max_length; call prepare_sample with an explicit truncation policy"
                )
            input_ids = input_ids.to(self.device)
            attention_mask = self.torch.ones_like(input_ids, device=self.device)
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
            activation_boundary=self._sample_activation_boundaries.get(prompt),
        )

    def _generate_answer(self, state: _PromptState, past: Any) -> tuple[Any, str, float, int]:
        max_new_tokens = self._sample_answer_token_counts.get(state.prompt, 1)
        current = state.probe_ids
        current_past = self._clone_past(past)
        generated: list[Any] = []
        first_logits = None
        self._last_attention_summary = None
        self._last_attention_input_summary = None
        self._last_attention_output_summary = None
        start = time.perf_counter()
        for generation_step in range(max_new_tokens):
            capture_attention = self._capture_attention_metrics and generation_step == 0
            previous_attention_implementation: str | None = None
            setter_candidate = getattr(self.model, "set_attn_implementation", None)
            set_attention_implementation: Callable[[str], Any] | None = (
                cast(Callable[[str], Any], setter_candidate) if callable(setter_candidate) else None
            )
            if capture_attention and set_attention_implementation is not None:
                current_implementation = self._attention_implementation()
                if current_implementation != "eager" and current_implementation != "transformers_default":
                    set_attention_implementation("eager")
                    previous_attention_implementation = current_implementation
            attention_handles: list[Any] = []
            captured_attention_inputs: dict[int, np.ndarray] = {}
            captured_attention_outputs: dict[int, np.ndarray] = {}
            if capture_attention:
                attention_handles = self._register_attention_hooks(
                    captured_attention_inputs,
                    captured_attention_outputs,
                )
            try:
                result = self.model(
                    input_ids=current,
                    past_key_values=current_past,
                    use_cache=True,
                    output_attentions=capture_attention,
                )
            finally:
                for handle in attention_handles:
                    handle.remove()
                if previous_attention_implementation is not None and set_attention_implementation is not None:
                    set_attention_implementation(previous_attention_implementation)
            logits = result.logits[:, -1, :]
            if first_logits is None:
                first_logits = logits
            if capture_attention:
                self._last_attention_summary = self._summarize_attentions(getattr(result, "attentions", None))
                self._last_attention_input_summary = self._stack_attention_vectors(
                    captured_attention_inputs
                )
                self._last_attention_output_summary = self._stack_attention_vectors(
                    captured_attention_outputs
                )
            next_token = self.torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            token_id = int(next_token[0, 0].item())
            if token_id in self._stop_token_ids:
                break
            current = next_token
            current_past = result.past_key_values
        synchronize(self.torch, self.devices)
        decode_s = time.perf_counter() - start
        if first_logits is None:
            raise RuntimeError("Answer generation produced no logits")
        generated_ids = self.torch.cat(generated, dim=1)[0].detach().cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return first_logits, generated_text, decode_s, len(generated)

    def _summarize_attentions(self, attentions: Any) -> np.ndarray | None:
        if not isinstance(attentions, tuple | list) or not attentions:
            return None
        layers: list[np.ndarray] = []
        for attention in attentions:
            if attention is None or not hasattr(attention, "ndim") or attention.ndim != 4:
                continue
            last_query = attention[:, :, -1, :].detach().float().mean(dim=(0, 1))
            layers.append(last_query.cpu().numpy())
        if not layers or len({layer.shape for layer in layers}) != 1:
            return None
        return np.stack(layers, axis=0)

    def _attention_metadata(self) -> dict[str, np.ndarray]:
        metadata: dict[str, np.ndarray] = {}
        if self._last_attention_summary is not None:
            metadata["attention_summary"] = self._last_attention_summary
        if self._last_attention_input_summary is not None:
            metadata["attention_input_summary"] = self._last_attention_input_summary
        if self._last_attention_output_summary is not None:
            metadata["attention_output_summary"] = self._last_attention_output_summary
        return metadata

    def _register_attention_hooks(
        self,
        captured_inputs: dict[int, np.ndarray],
        captured_outputs: dict[int, np.ndarray],
    ) -> list[Any]:
        layers, _ = self._decoder_layers()
        if layers is None:
            return []
        handles: list[Any] = []
        for layer_index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attention is None or not hasattr(attention, "register_forward_hook"):
                continue

            def capture_input(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                tensor = kwargs.get("hidden_states")
                if tensor is None and args:
                    tensor = args[0]
                if tensor is None or not hasattr(tensor, "detach"):
                    return
                value = tensor.detach().float()
                if value.ndim >= 3:
                    value = value[:, -1, :]
                captured_inputs[index] = value.reshape(-1).cpu().numpy()

            def capture_output(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                tensor = output[0] if isinstance(output, tuple | list) and output else output
                if tensor is None or not hasattr(tensor, "detach"):
                    return
                value = tensor.detach().float()
                if value.ndim >= 3:
                    value = value[:, -1, :]
                captured_outputs[index] = value.reshape(-1).cpu().numpy()

            handles.append(
                attention.register_forward_pre_hook(capture_input, with_kwargs=True)
            )
            handles.append(attention.register_forward_hook(capture_output))
        return handles

    def _stack_attention_vectors(self, captured: dict[int, np.ndarray]) -> np.ndarray | None:
        if len(captured) != self.num_layers:
            return None
        ordered = [captured[index] for index in range(self.num_layers)]
        if len({value.shape for value in ordered}) != 1:
            return None
        return np.stack(ordered, axis=0)

    def _set_lora_enabled(self, enabled: bool) -> None:
        for module in self._lora_modules:
            if hasattr(module, "lora_enabled"):
                module.lora_enabled = enabled

    def _alora_base_extras(self, state: _PromptState) -> dict[str, Any]:
        boundary = state.activation_boundary
        if boundary is None:
            return {}
        cached = self._alora_base_cache.get(state.prompt)
        if cached is None:
            base_ids = state.prefix_ids[:, :boundary]
            if base_ids.shape[1] == 0:
                raise ValueError("aLoRA base prefix is empty")
            self._set_lora_enabled(False)
            try:
                with self.torch.no_grad():
                    base = self.model(input_ids=base_ids, use_cache=True, output_hidden_states=True)
            finally:
                self._set_lora_enabled(True)
            cached = (
                self._clone_past(base.past_key_values),
                self._summarize_hidden(base.hidden_states),
            )
            self._alora_base_cache[state.prompt] = cached
        past, hidden = cached
        return {
            "alora_base_past": past,
            "alora_base_hidden": hidden,
            "alora_activation_boundary": boundary,
        }

    def _alora_suffix_recompute(self, baseline: BackendOutput) -> BackendOutput:
        if not baseline.extras:
            raise ValueError("aLoRA prefix reuse requires cached prompt state")
        state = baseline.extras.get("prompt_state")
        boundary = baseline.extras.get("alora_activation_boundary")
        base_past = baseline.extras.get("alora_base_past")
        if not isinstance(state, _PromptState) or not isinstance(boundary, int) or base_past is None:
            raise ValueError(
                "aLoRA prefix reuse requires data.adapter_activation_marker and a prepared base-prefix cache"
            )
        past = self._clone_past(base_past)
        suffix_ids = state.prefix_ids[:, boundary:]
        reset_peak_memory(self.torch, self.devices)
        synchronize(self.torch, self.devices)
        start = time.perf_counter()
        self._set_lora_enabled(True)
        self._set_lora_capture(True)
        hidden_tensor = baseline.extras.get("alora_base_hidden", baseline.hidden_tensor)
        with self.torch.no_grad():
            if suffix_ids.shape[1] > 0:
                suffix = self.model(
                    input_ids=suffix_ids,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = suffix.past_key_values
                hidden_tensor = self._summarize_hidden(suffix.hidden_states)
            self._set_lora_capture(False)
        synchronize(self.torch, self.devices)
        maintenance_s = time.perf_counter() - start
        with self.torch.no_grad():
            probe_logits, generated_text, decode_s, generated_tokens = self._generate_answer(state, past)
        latency = maintenance_s + decode_s
        self._last_alora_s = latency
        return BackendOutput(
            logits=self._to_numpy(probe_logits),
            cache_tensor=self._summarize_past(past),
            hidden_tensor=np.asarray(hidden_tensor),
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": past,
                "prompt_state": state,
                "lora_cache": self._snapshot_lora_cache(),
                "memory_allocated": memory_allocated(self.torch, self.devices),
                "peak_memory_allocated": max_memory_allocated(self.torch, self.devices),
                "cache_bytes": self._past_nbytes(past),
                "token_length": int(state.input_ids.shape[1]),
                "attention_implementation": self._attention_implementation(),
                "strategy_latency": latency,
                "cache_maintenance_latency": maintenance_s,
                "decode_latency": decode_s,
                "generated_tokens": generated_tokens,
                "generated_text": generated_text,
                "cache_mode": "alora_base_prefix_suffix_recompute",
                "strategy_flops": self._full_prefill_flops(int(suffix_ids.shape[1])),
                "full_recompute_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1])),
                **self._attention_metadata(),
                **self._alora_base_extras(state),
            },
        )

    def _cached_output_with_runtime(self, baseline: BackendOutput, *, cache_mode: str) -> BackendOutput:
        extras = dict(baseline.extras or {})
        decode_s = float(extras.get("decode_latency", 0.0))
        extras.update(
            {
                "cache_mode": cache_mode,
                "cache_maintenance_latency": 0.0,
                "strategy_latency": decode_s,
                "strategy_flops": 0.0,
            }
        )
        return BackendOutput(
            logits=baseline.logits,
            cache_tensor=baseline.cache_tensor,
            hidden_tensor=baseline.hidden_tensor,
            parameter_version=baseline.parameter_version,
            extras=extras,
        )

    def _reuse_old_prefix_cache(
        self,
        baseline: BackendOutput,
        *,
        cache_mode: str = "stale_reuse",
        reset_memory_stats: bool = True,
    ) -> BackendOutput:
        if not baseline.extras:
            raise ValueError("Baseline output does not contain cached HF state")
        past = baseline.extras["past_key_values"]
        result = self._probe_with_past(
            baseline=baseline,
            past=past,
            cache_tensor=baseline.cache_tensor,
            hidden_tensor=baseline.hidden_tensor,
            extra_metadata={"cache_mode": cache_mode, "strategy_flops": 0.0},
            reset_memory_stats=reset_memory_stats,
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
        if decision.first_invalid_layer is None:
            raise ValueError("Partial recompute requires an explicit first-invalid layer")
        state = baseline.extras.get("prompt_state")
        hidden_states = baseline.extras.get("hidden_states")
        old_past = baseline.extras.get("past_key_values")
        if not isinstance(state, _PromptState) or not isinstance(hidden_states, tuple) or old_past is None:
            return None
        start_layer = decision.first_invalid_layer
        layer_container, family = self._decoder_layers()
        if layer_container is None or start_layer < 0 or start_layer > len(layer_container):
            return None
        end_layer = min(
            len(layer_container),
            decision.last_recomputed_layer
            if decision.last_recomputed_layer is not None
            else len(layer_container),
        )
        if end_layer <= start_layer:
            raise ValueError("Partial recompute requires a non-empty layer interval")
        old_layers = self._past_as_layers(old_past)
        if len(old_layers) != len(layer_container) or len(hidden_states) < len(layer_container) + 1:
            return None

        replacements: list[tuple[int, Any]] = []
        try:
            for layer_index in range(len(layer_container)):
                if start_layer <= layer_index < end_layer:
                    continue
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

            reset_peak_memory(self.torch, self.devices)
            synchronize(self.torch, self.devices)
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
            synchronize(self.torch, self.devices)
            maintenance_s = time.perf_counter() - start
        finally:
            self._set_lora_capture(False)
            for layer_index, original in replacements:
                layer_container[layer_index] = original

        with self.torch.no_grad():
            probe_logits, generated_text, decode_s, generated_tokens = self._generate_answer(
                state, prefill.past_key_values
            )
        latency = maintenance_s + decode_s
        recomputed_hidden_states = tuple(hidden.detach() for hidden in prefill.hidden_states)
        expected_window_states = end_layer - start_layer + 1
        if len(recomputed_hidden_states) == len(layer_container) + 1:
            merged_hidden_states = recomputed_hidden_states
        elif len(recomputed_hidden_states) == expected_window_states:
            merged_hidden_states = (
                hidden_states[: start_layer + 1]
                + recomputed_hidden_states[1:]
                + hidden_states[end_layer + 1 :]
            )
        else:
            raise RuntimeError(
                "Native partial recompute returned an unexpected hidden-state history: "
                f"got {len(recomputed_hidden_states)}, expected {expected_window_states} window states "
                f"or {len(layer_container) + 1} full states."
            )

        lora_cache = dict(baseline.extras.get("lora_cache", {}))
        lora_cache.update(self._snapshot_lora_cache())
        finite_window = end_layer < len(layer_container)
        window_layers = end_layer - start_layer
        return BackendOutput(
            logits=self._to_numpy(probe_logits),
            cache_tensor=self._summarize_past(prefill.past_key_values),
            hidden_tensor=self._summarize_hidden(merged_hidden_states),
            parameter_version=self.parameter_version,
            extras={
                "past_key_values": prefill.past_key_values,
                "hidden_states": merged_hidden_states,
                "prompt_state": state,
                "lora_cache": lora_cache,
                "memory_allocated": memory_allocated(self.torch, self.devices),
                "peak_memory_allocated": max_memory_allocated(self.torch, self.devices),
                "cache_bytes": self._past_nbytes(prefill.past_key_values),
                "token_length": int(state.input_ids.shape[1]),
                "attention_implementation": self._attention_implementation(),
                "strategy_latency": latency,
                "cache_maintenance_latency": maintenance_s,
                "decode_latency": decode_s,
                "generated_tokens": generated_tokens,
                "generated_text": generated_text,
                "partial_start_layer": start_layer,
                "partial_end_layer": end_layer,
                "partial_window_layers": window_layers,
                "partial_mode": (
                    f"native_{family}_finite_window_restart"
                    if finite_window
                    else f"native_{family}_layer_restart"
                ),
                "strategy_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1]))
                * window_layers
                / max(1, self.num_layers),
                "full_recompute_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1])),
                **self._attention_metadata(),
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

    def _clone_past(self, past: Any) -> Any:
        layers = tuple(
            tuple(tensor.detach().clone() if hasattr(tensor, "detach") else tensor for tensor in layer)
            for layer in self._past_as_layers(past)
        )
        return self._restore_past_type(past, layers)

    def _restore_past_type(self, original: Any, layers: tuple[tuple[Any, ...], ...]) -> Any:
        factory = getattr(type(original), "from_legacy_cache", None)
        if callable(factory):
            return factory(layers)
        if hasattr(original, "update") and hasattr(original, "get_seq_length"):
            cache_type = type(original)
            try:
                return cache_type(layers, config=self.model.config)
            except TypeError:
                try:
                    return cache_type(layers)
                except TypeError:
                    pass
        return layers

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

        def llama_forward(_module: Any, *args: Any, **kwargs: Any) -> Any:
            del _module, args
            cache = kwargs.get("past_key_values") or kwargs.get("past_key_value")
            if cache is not None and hasattr(cache, "update"):
                key, value = cached_past[:2]
                cache.update(key, value, layer_index, {})
            return cached_hidden

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
        reset_peak_memory(self.torch, self.devices)
        synchronize(self.torch, self.devices)
        start = time.perf_counter()
        corrected_past, corrected_lora_cache = self._apply_lora_weight_delta_to_past(
            baseline.extras["past_key_values"],
            baseline.extras.get("lora_cache", {}),
            split_layer=split_layer,
            position_ids=self._prefix_position_ids(baseline.extras.get("prompt_state")),
        )
        if corrected_past is None:
            result = self._reuse_old_prefix_cache(baseline, reset_memory_stats=False)
            if result.extras is not None:
                result.extras["delta_mode"] = "unavailable_weight_delta_fallback_to_stale"
            synchronize(self.torch, self.devices)
            total_s = time.perf_counter() - start
            if result.extras is not None:
                decode_s = float(result.extras.get("decode_latency", 0.0))
                result.extras["cache_maintenance_latency"] = max(0.0, total_s - decode_s)
                result.extras["strategy_latency"] = total_s
            self._last_delta_s = total_s
            return result
        delta_mode = "lora_weight_delta"
        logical_cache_bytes = self._past_nbytes(corrected_past)
        if decision.strategy is StrategyName.LRAGENT_ADAPTER_CACHE:
            delta_mode = "lragent_shared_base_plus_low_rank_component"
            logical_cache_bytes = self._last_low_rank_cache_bytes
        elif decision.strategy is StrategyName.FORKKV_BASE_DELTA:
            delta_mode = "forkkv_copy_on_write_residual"
            logical_cache_bytes = self._last_residual_cache_bytes
        result = self._probe_with_past(
            baseline=baseline,
            past=corrected_past,
            cache_tensor=self._summarize_past(corrected_past),
            hidden_tensor=baseline.hidden_tensor,
            lora_cache=corrected_lora_cache,
            extra_metadata={
                "delta_mode": delta_mode,
                "cache_bytes": logical_cache_bytes,
                "physical_cache_bytes": self._past_nbytes(corrected_past),
                "baseline_fidelity": "paper_reimplementation",
            },
        )
        synchronize(self.torch, self.devices)
        total_s = time.perf_counter() - start
        if result.extras is not None:
            decode_s = float(result.extras.get("decode_latency", 0.0))
            result.extras["cache_maintenance_latency"] = max(0.0, total_s - decode_s)
            result.extras["strategy_latency"] = total_s
            result.extras["peak_memory_allocated"] = max_memory_allocated(self.torch, self.devices)
            result.extras["strategy_flops"] = self._last_correction_flops
            result.extras["delta_raw_l2"] = self._last_delta_raw_l2
            result.extras["delta_stored_l2"] = self._last_delta_stored_l2
            result.extras["delta_raw_max_abs"] = self._last_delta_raw_max_abs
            result.extras["delta_stored_max_abs"] = self._last_delta_stored_max_abs
            result.extras["delta_changed_fraction"] = self._last_delta_changed_fraction
            result.extras["delta_quantization_retention"] = self._last_delta_quantization_retention
        self._last_delta_s = total_s
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
        reset_memory_stats: bool = True,
    ) -> BackendOutput:
        if not baseline.extras:
            raise ValueError("Baseline output does not contain prompt state")
        state = baseline.extras["prompt_state"]
        if reset_memory_stats:
            reset_peak_memory(self.torch, self.devices)
        synchronize(self.torch, self.devices)
        start = time.perf_counter()
        with self.torch.no_grad():
            probe_logits, generated_text, decode_s, generated_tokens = self._generate_answer(state, past)
        synchronize(self.torch, self.devices)
        latency = time.perf_counter() - start
        extras = {
            "past_key_values": past,
            "hidden_states": baseline.extras.get("hidden_states"),
            "prompt_state": state,
            "lora_cache": lora_cache if lora_cache is not None else baseline.extras.get("lora_cache", {}),
            "memory_allocated": memory_allocated(self.torch, self.devices),
            "peak_memory_allocated": max_memory_allocated(self.torch, self.devices),
            "cache_bytes": self._past_nbytes(past),
            "token_length": int(state.input_ids.shape[1]),
            "attention_implementation": self._attention_implementation(),
            "strategy_latency": latency,
            "cache_maintenance_latency": max(0.0, latency - decode_s),
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "decode_latency": decode_s,
            "strategy_flops": 0.0,
            "full_recompute_flops": self._full_prefill_flops(int(state.prefix_ids.shape[1])),
            **self._attention_metadata(),
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
        position_ids: Any | None = None,
    ) -> tuple[Any | None, dict[str, Any]]:
        self._last_low_rank_cache_bytes = 0
        self._last_residual_cache_bytes = 0
        self._last_delta_raw_l2 = 0.0
        self._last_delta_stored_l2 = 0.0
        self._last_delta_raw_max_abs = 0.0
        self._last_delta_stored_max_abs = 0.0
        self._last_delta_changed_fraction = 0.0
        self._last_delta_quantization_retention = 0.0
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
        low_rank_cache_bytes = 0
        residual_cache_bytes = 0
        correction_flops = 0.0
        raw_delta_squared = 0.0
        stored_delta_squared = 0.0
        raw_delta_max_abs = 0.0
        stored_delta_max_abs = 0.0
        changed_elements = 0
        corrected_elements = 0
        self._last_correction_flops = 0.0
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
            if projection == "k":
                projected = self._apply_rotary_to_key_delta(
                    projected,
                    position_ids=position_ids,
                )
                if projected is None:
                    return None, {}
            rank = int(old_state.get("rank", 0) or 0)
            if rank > 0 and hasattr(cached_input, "shape") and len(cached_input.shape) >= 2:
                batch = int(cached_input.shape[0])
                tokens = int(cached_input.shape[1])
                input_width = int(cached_input.shape[-1])
                output_width = int(delta.shape[-1])
                low_rank_cache_bytes += batch * tokens * rank * int(cached_input.element_size())
                correction_flops += 4.0 * batch * tokens * rank * (input_width + output_width)
            residual_cache_bytes += int(projected.numel() * projected.element_size())
            target_tensor = corrected[layer][item_index]
            projected_on_target = projected.to(
                device=target_tensor.device,
                dtype=target_tensor.dtype,
            )
            corrected_tensor = target_tensor + projected_on_target
            stored_delta = corrected_tensor - target_tensor
            raw_float = projected.detach().float()
            stored_float = stored_delta.detach().float()
            raw_delta_squared += float(self.torch.sum(raw_float * raw_float).cpu())
            stored_delta_squared += float(self.torch.sum(stored_float * stored_float).cpu())
            raw_delta_max_abs = max(raw_delta_max_abs, float(raw_float.abs().max().cpu()))
            stored_delta_max_abs = max(stored_delta_max_abs, float(stored_float.abs().max().cpu()))
            changed_elements += int(self.torch.count_nonzero(stored_delta).cpu())
            corrected_elements += int(stored_delta.numel())
            corrected[layer][item_index] = corrected_tensor
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
        corrected_layers = tuple(tuple(layer) for layer in corrected)
        self._last_correction_flops = correction_flops
        self._last_low_rank_cache_bytes = low_rank_cache_bytes
        self._last_residual_cache_bytes = residual_cache_bytes
        self._last_delta_raw_l2 = raw_delta_squared**0.5
        self._last_delta_stored_l2 = stored_delta_squared**0.5
        self._last_delta_raw_max_abs = raw_delta_max_abs
        self._last_delta_stored_max_abs = stored_delta_max_abs
        self._last_delta_changed_fraction = (
            changed_elements / corrected_elements if corrected_elements > 0 else 0.0
        )
        self._last_delta_quantization_retention = (
            self._last_delta_stored_l2 / self._last_delta_raw_l2
            if self._last_delta_raw_l2 > 0.0
            else 0.0
        )
        return self._restore_past_type(past_key_values, corrected_layers), new_lora_cache

    def _prefix_position_ids(self, state: Any) -> Any | None:
        if not isinstance(state, _PromptState):
            return None
        batch = int(state.prefix_ids.shape[0])
        sequence = int(state.prefix_ids.shape[1])
        positions = self.torch.arange(sequence, device=state.prefix_ids.device)
        return positions.unsqueeze(0).expand(batch, -1)

    def _apply_rotary_to_key_delta(
        self,
        key_delta: Any,
        *,
        position_ids: Any | None,
    ) -> Any | None:
        model = getattr(self, "model", None)
        backbone = getattr(model, "model", None)
        rotary = getattr(backbone, "rotary_emb", None)
        if not callable(rotary):
            return key_delta
        if position_ids is None or key_delta.ndim != 4:
            return None

        sequence = int(position_ids.shape[-1])
        if int(key_delta.shape[2]) == sequence:
            unsqueeze_dim = 1
        elif int(key_delta.shape[1]) == sequence:
            unsqueeze_dim = 2
        else:
            return None
        if int(key_delta.shape[-1]) % 2 != 0:
            return None

        positions = position_ids.to(device=key_delta.device)
        cos, sin = rotary(key_delta, positions)
        cos = cos.to(device=key_delta.device, dtype=key_delta.dtype).unsqueeze(unsqueeze_dim)
        sin = sin.to(device=key_delta.device, dtype=key_delta.dtype).unsqueeze(unsqueeze_dim)
        if int(cos.shape[-1]) != int(key_delta.shape[-1]):
            return None
        half = int(key_delta.shape[-1]) // 2
        rotated_half = self.torch.cat(
            (-key_delta[..., half:], key_delta[..., :half]),
            dim=-1,
        )
        return key_delta * cos + rotated_half * sin

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
        if target.kind is ModuleKind.OUTPUT_HEAD:
            get_output_embeddings = getattr(self.model, "get_output_embeddings", None)
            if callable(get_output_embeddings):
                output_embeddings = get_output_embeddings()
                if output_embeddings is not None:
                    parameters = [
                        parameter
                        for parameter in output_embeddings.parameters()
                        if parameter.is_floating_point()
                    ]
                    if parameters:
                        return parameters

        filters = self._target_filters(target.kind)
        selected = []
        for name, param in self.model.named_parameters():
            lower = name.lower()
            if target.layer is not None and not self._name_matches_layer(lower, target.layer):
                continue
            if any(part in lower for part in filters):
                selected.append(param)
        return selected

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

    def _past_nbytes(self, past_key_values: Any) -> int:
        total = 0
        for layer in self._past_as_layers(past_key_values):
            for tensor in layer[:2]:
                if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                    total += int(tensor.numel()) * int(tensor.element_size())
        return total

    def _attention_implementation(self) -> str:
        value = getattr(self.model.config, "_attn_implementation", None)
        return str(value or "transformers_default")

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
