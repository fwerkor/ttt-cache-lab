# Delta-correction quantization sweep

## Question

The existing all-layer LoRA experiments used a total update norm of `1e-5`. At BF16 cache precision, the direct K/V correction can be smaller than one representable cache step, making delta correction numerically indistinguishable from stale reuse. This sweep separates three effects:

1. cache quantization loss;
2. direct K/V weight-delta correction;
3. hidden-state propagation after early or all-layer updates.

## Experimental axes

- Models: Qwen2.5-0.5B, 1.5B, and 7B.
- Update norms: `1e-5` through `1e-2`, with the larger models using the informative subset.
- Targets:
  - all-layer `lora.k` and `lora.v`;
  - final-layer `lora.k_late` and `lora.v_late`.
- Strategies: full recompute, stale reuse, and direct delta correction.
- Version gap: one update for the threshold scan.

The final-layer targets are controls: their direct K/V correction is not followed by additional decoder layers, so a retained correction should be much closer to the full reference than an all-layer correction if propagation is the limiting factor.

## Recorded diagnostics

Every delta-correction record includes:

- `delta_raw_l2`: L2 norm before cache-dtype quantization;
- `delta_stored_l2`: L2 norm of the change actually stored in the cache;
- `delta_raw_max_abs` and `delta_stored_max_abs`;
- `delta_changed_fraction`: fraction of corrected K/V elements that changed after writing to cache dtype;
- `delta_quantization_retention`: `stored_l2 / raw_l2`.

The sweep summary additionally reports paired stale-versus-delta KL improvement and maintenance overhead.

## Interpretation

A delta result is considered numerically testable only after the stored correction is nontrivial. The primary evidence is the joint relationship between retention and quality:

- low retention and identical stale/delta KL: quantization-limited;
- high retention and late-layer improvement: direct correction works locally, while all-layer failure is propagation-limited;
- high retention with no late-layer improvement: the direct correction implementation or underlying approximation remains inadequate;
- quality improvement smaller than maintenance overhead: correct but not useful for the planner.

No universal threshold is fixed in advance. The analysis reports the empirical transition and compares it across model size, target, and layer placement.

## Pilot and confirmation phases

The broad pilot sweep locates the numerical transition with a small sample budget. The confirmatory sweep then reruns the informative `1e-3` and `1e-2` points with 4 samples on 0.5B/1.5B and 2 samples on 7B. The confirmation specification is `configs/sweeps/ascend_delta_quantization_confirm.yaml`.

## Cross-model update normalization

A fixed target L2 norm is diluted as the number of updated LoRA parameters grows. The RMS calibration therefore uses `adapter.norm_control: target_rms`, where the configured update value is converted to `L2 = RMS * sqrt(updated_parameter_count)`. Each record includes `updated_parameter_count` and `applied_update_rms`. The calibration specification is `configs/sweeps/ascend_delta_rms_calibration.yaml`. It raises `cache.update_norm_threshold` to `1.0` only to force execution of the direct-delta algorithm; this is an evaluation override, not a recommended deployment safety threshold.

Sweep summaries distinguish the requested `delta_correction` strategy from the action actually executed. `delta_execution_rate` and `delta_fallback_rate` prevent full-recompute safety fallbacks from being counted as successful delta corrections.

## Reproduction

Generate concrete pilot configs:

```bash
python scripts/generate_delta_quantization_sweep.py
```

Generate confirmatory configs:

```bash
python scripts/generate_delta_quantization_sweep.py \
  --spec configs/sweeps/ascend_delta_quantization_confirm.yaml
```

Generate RMS calibration configs:

```bash
python scripts/generate_delta_quantization_sweep.py \
  --spec configs/sweeps/ascend_delta_rms_calibration.yaml
```

Run a generated config with the normal Ascend launcher:

```bash
bash scripts/run_ascend_e2_single.sh \
  runs/delta_quantization_configs/qwen_0_5b/1em03.yaml
```

Aggregate completed runs:

```bash
python scripts/summarize_delta_quantization_sweep.py
```
