# Design notes

## Semantics

The framework separates three cache semantics.

1. **Exact current-parameter semantics**: cached K/V must match a full prefill under the current parameter version.
2. **Frozen-evidence semantics**: cached K/V are treated as fixed evidence from the pre-adaptation model.
3. **Bounded-stale semantics**: stale cache reuse is allowed if measured or predicted error remains below a threshold.

## Planner states

Each cache block can be classified as:

- `VALID_EXACT`
- `VALID_FROZEN`
- `VALID_APPROX`
- `INVALID`

The planner maps update targets and update magnitudes to one of:

- `REUSE_EXACT`
- `REUSE_FROZEN`
- `REUSE_STALE`
- `DELTA_CORRECT`
- `PARTIAL_RECOMPUTE`
- `FULL_RECOMPUTE`
- `REJECT_UPDATE`

## First implementation boundary

The first version does not attempt to be a production serving engine. It provides controlled, reproducible experiments. Integration with vLLM/SGLang/HuggingFace generation should remain behind backend interfaces.
