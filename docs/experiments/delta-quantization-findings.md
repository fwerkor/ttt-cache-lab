# Delta-correction quantization findings

## Scope and provenance

This note records the first completed investigation of direct LoRA K/V cache correction under BF16 cache storage.

| Phase | Result root | Model/sample budget | Git revision |
|---|---|---|---|
| L2 threshold pilot | `runs/delta_quantization` | 0.5B: 1 sample; 1.5B: 2 samples | `bf8d9388861fad6dd8ead2f7edd4b2dfc8c5d087` |
| L2 confirmation | `runs/delta_quantization_confirm` | 0.5B/1.5B: 4 samples; 7B: 2 samples | `852bb974c1757c77377a75176f98d66e5495adc7` |
| Cross-model RMS calibration | `runs/delta_rms_calibration` | 1 sample per model and RMS point | `ad831937cbabaa1fd6c933466bd714cdaeac0821` |

All recorded runs report `git_dirty=false`. The implementation and sweep utilities were validated by 164 passing tests.

## Why stale and delta were initially identical

At target L2 update norm `1e-5`, the RoPE-corrected K delta was commonly much smaller than one BF16 representable step around the existing cache value. A direct diagnostic on Qwen2.5-0.5B measured:

- maximum raw rotated K delta: about `2.2e-6`;
- median BF16 spacing around cached K values: about `3.1e-2`;
- changed K elements after storage: `53 / 199680` (`0.0265%`);
- stale and corrected probe logits: bit-identical.

The equality therefore came from cache-dtype quantization, not from a zero LoRA delta or a silent stale fallback.

## Added measurements

Each actual delta execution now records:

- `delta_raw_l2` and `delta_stored_l2`;
- raw and stored maximum absolute delta;
- `delta_changed_fraction`;
- `delta_quantization_retention = stored_l2 / raw_l2`;
- actual strategy action and fallback rate;
- updated parameter count and applied update RMS.

The summary script only treats records with `action=delta_correct` as executed delta correction. Full-recompute safety fallbacks are reported separately.

## Multi-sample L2 confirmation

The table reports mean KL improvement `(stale - delta) / stale`. Positive values favor delta correction.

| Model | Target | L2 norm | Retention | Mean KL improvement | Delta better rate |
|---|---|---:|---:|---:|---:|
| 0.5B | all-layer K | `1e-3` | 0.154 | +33.8% | 75% |
| 0.5B | all-layer K | `1e-2` | 0.483 | -93.0% | 25% |
| 0.5B | late K | `1e-3` | 0.486 | +59.6% | 50% |
| 0.5B | late K | `1e-2` | 0.957 | +99.3% | 75% |
| 0.5B | all-layer V | `1e-3` | 0.325 | +47.0% | 100% |
| 0.5B | all-layer V | `1e-2` | 0.817 | +60.2% | 75% |
| 0.5B | late V | `1e-3`, `1e-2` | 0.463, 0.972 | 100% | 100% |
| 1.5B | all-layer K | `1e-3` | 0.079 | +38.3% | 50% |
| 1.5B | all-layer K | `1e-2` | 0.309 | -32.8% | 50% |
| 1.5B | late K | `1e-3` | 0.302 | +98.9% | 50% |
| 1.5B | late K | `1e-2` | 0.890 | +24.0% | 25% |
| 1.5B | all-layer V | `1e-3` | 0.441 | -158.8% | 25% |
| 1.5B | all-layer V | `1e-2` | 0.780 | -40.3% | 50% |
| 1.5B | late V | `1e-3`, `1e-2` | 0.181, 0.602 | 100% | 100% |
| 7B | late V | `1e-3`, `1e-2` | 0.211, 0.705 | corrected KL = 0 | observed in both points |

### Interpretation

1. **Quantization retention is necessary but not sufficient.** Higher retention makes the correction numerically visible, but all-layer K/V can still improve or worsen because hidden-state propagation is omitted.
2. **Late V is the strongest stable region.** It produced zero corrected KL in every observed late-V condition across 0.5B, 1.5B, and 7B. This matches the dependency structure: final-layer V changes the cached V directly and has no downstream decoder layers whose hidden inputs must be reconstructed.
3. **Late K is useful but less exact.** It often sharply reduced KL, but RoPE and BF16 operation ordering prevent the additive correction from consistently matching full recomputation.
4. **All-layer correction is not a universal strategy.** Its quality is model-, sample-, norm-, and projection-dependent. It must be selected from calibrated evidence rather than enabled by target type alone.

## Cross-model RMS calibration

A fixed total L2 norm is not comparable across targets or model sizes. `target_rms` converts the configured per-parameter RMS to an applied L2 norm using:

`target_l2 = target_rms * sqrt(updated_parameter_count)`.

The RMS calibration forced actual delta execution by setting the evaluation-only update-norm threshold to `1.0`.

Selected single-sample results:

| Model | Target | RMS | Retention | Stale KL | Delta KL | Improvement |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | all-layer K | `1e-5` | 0.335 | `9.13e-4` | `7.42e-4` | +18.7% |
| 0.5B | all-layer K | `1e-4` | 0.909 | `1.20e-3` | `2.64e-3` | -120.2% |
| 0.5B | all-layer V | `1e-5` | 0.668 | `1.48e-4` | `9.73e-5` | +34.2% |
| 0.5B | all-layer V | `1e-4` | 1.013 | `8.73e-2` | `5.86e-2` | +32.9% |
| 1.5B | all-layer K | `1e-4` | 0.857 | `1.12e-3` | `5.15e-4` | +54.0% |
| 1.5B | all-layer V | `1e-5` | 0.688 | `1.21e-4` | `2.53e-5` | +79.1% |
| 1.5B | all-layer V | `1e-4` | 0.973 | `8.48e-3` | `4.65e-3` | +45.2% |
| 1.5B | late K | `1e-4` | 0.849 | `3.92e-5` | `1.74e-7` | +99.6% |
| 1.5B | late V | `1e-5`, `1e-4` | 0.192, 0.602 | nonzero stale KL | `0`, `0` | 100% |

The RMS results confirm that a representable direct term can be valuable, but they also reproduce the non-monotonic all-layer K behavior. A larger update does not guarantee a better correction.

## 7B metric limitation

On Qwen2.5-7B, the parameter update and internal drift are real:

- all-layer cache relative error: about `2.4e-4` to `5.6e-4`;
- hidden-state relative error: about `2.7e-3` to `4.4e-3`;
- quantization retention at RMS `1e-4`: about 0.895 for K and 0.955 for V.

However, the current one-token probe logits KL remains around `1e-10`. The existing passkey task scores are also unusable because the older E2 task configuration yields zero scores. Therefore, the 7B runs establish cache-level behavior and correction cost, but they do not yet establish task-level quality impact.

## Planner implications

The current evidence does not support globally removing delta correction. It supports a narrower policy:

- permit late-V delta correction after calibration;
- permit late-K only in empirically safe cells;
- treat all-layer K/V delta correction as optional and failure-map-controlled;
- reject delta when the stored correction is numerically negligible;
- use suffix or full recomputation when propagation error dominates;
- never count a full-recompute fallback as a successful delta result.

The planner should eventually include projection, layer position, update RMS, quantization retention, context, model, and empirical false-safe rate in its decision key.

## Remaining work

- rerun task-level experiments with the corrected long-context task suite;
- add multiple seeds for the RMS sweep;
- measure projection-recompute correction for K, which may avoid additive RoPE/BF16 rounding mismatch;
- compare delta against exact suffix recomputation on the same late-layer conditions;
- calibrate planner thresholds on training tasks and report false-safe rate on held-out tasks.
