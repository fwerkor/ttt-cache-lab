# Design notes

## Cache semantics

The framework separates three cache semantics:

1. **Exact current-parameter semantics**: cached K/V match the current adapter/model version.
2. **Frozen-evidence semantics**: cached K/V remain fixed evidence from an earlier or base model.
3. **Bounded-stale semantics**: reuse is approximate and must be evaluated against explicit quality thresholds.

Each cache entry is indexed by adapter identity and version. Each layer-level block records token range, layer, base model, adapter/version, update target, accumulated update norm, validity state, precision, and attention implementation.

## Actions

The planner and baseline strategies can execute:

- `REUSE_EXACT`
- `REUSE_FROZEN`
- `REUSE_STALE`
- `DELTA_CORRECT`
- `PARTIAL_RECOMPUTE`
- `ALORA_SUFFIX_RECOMPUTE`
- `FULL_RECOMPUTE`
- `REJECT_UPDATE`

`REJECT_UPDATE` rejects cache reuse and executes a safe full refresh. `PARTIAL_RECOMPUTE` restarts the decoder at the first invalid layer without reading the full-reference cache. `ALORA_SUFFIX_RECOMPUTE` reuses a base-model prefix before an invocation marker and recomputes the post-marker suffix with the active adapter.

## Baselines

- `adapter_specific_cache` and `lragent_adapter_cache` maintain complete entries per fixed adapter identity.
- `static_base_delta` and `forkkv_base_delta` retain a base cache and patch K/V with measured LoRA projection deltas.
- `alora_prefix_reuse` stores a base prefix with LoRA disabled and applies the adapter only after the configured marker.
- `oracle_planner` evaluates actual candidate outputs against the full reference and selects the lowest measured safe latency.

## Measurement boundary

The repository is an experiment framework rather than a production serving engine. The backends expose real Transformers `past_key_values`, LoRA updates, native layer restart, multi-device model sharding, and detailed cost metrics. Integration with serving schedulers such as vLLM or SGLang remains outside the experiment runtime and should use the same backend/cache metadata interfaces.
