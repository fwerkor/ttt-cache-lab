# CacheBlend-Inspired Influence Ranker

## Goal

Identify a small set of directly repairable layer-token KV blocks that achieves quality close to the exhaustive `sparse_delta_oracle`, without evaluating candidate masks with model forwards.

This line uses LoRA parameter deltas only as block-selection signals. It does not revive delta correction as a cache-repair action.

## Feature

Magnitude-only features such as `predicted_delta_norm` estimate how much a block changes, but not whether that change affects the current attention output. The new feature estimates a signed first-order output correction for every directly repairable block.

For a V update, the block contribution follows the existing first-order form

\[
\Delta o_B \approx \sum_{t \in B} a_t\,\Delta v_t.
\]

For a K update, the implementation adds

\[
\Delta o_B \approx \sum_{t \in B} a_t\,\Delta s_t\,(v_t-o),
\qquad
\Delta s_t = \frac{q^\top \Delta k_t}{\sqrt d}.
\]

The per-block correction is projected through the attention output projection. `signed_total_alignment` ranks blocks by alignment with the total predicted output correction. This requires no candidate repair forward.

## Validation

### Multi-hop tracing, unseen samples

Model and setting: Qwen2.5-1.5B, context 512, LoRA middle-layer K/V update, version gap 4, block size 64, eight directly repairable blocks. Feature choices used samples 0-3; held-out evaluation used samples 4-7.

For K updates over the nontrivial 2/4/6-block budgets:

- stale mean KL: 0.02987;
- `signed_total_alignment` mean KL: 0.01645;
- aggregate oracle-gain capture: 56.1%;
- beneficial in 66.7% of condition-budget points.

At the intended two-block budget:

- selected KL: 0.01264;
- exhaustive oracle KL: 0.00643;
- oracle-gain capture: 73.5%.

For V updates, the two-block result also captured 75.5% of oracle gain, but larger budgets were less stable. The feature should therefore not yet be treated as a universal V selector.

### Cross-task aggregation

Two aggregation samples were evaluated without using them for feature selection.

For K updates:

- two blocks: KL 0.000701 versus stale 0.001937 and oracle 0.00000947, capturing 64.1% of oracle gain;
- four blocks: KL 0.000567, capturing 71.1% of oracle gain;
- six blocks became less reliable, confirming non-monotonic repair behavior.

For V updates, signed alignment was not consistently beneficial across budgets.

### Cost

Measured block-score extraction latency after the first invocation:

- K: 13.7 ms, 9.6% of full-recompute strategy latency;
- V: 11.0 ms, 7.7% of full-recompute strategy latency.

The first invocation was about 0.11 s because of accelerator warm-up. Runtime comparisons should warm the scorer before reporting steady-state latency.

## Conclusions

1. The useful CacheBlend transfer is impact-based block ranking, not cache delta correction.
2. Signed output alignment generalizes better than delta magnitude for K-cache repair.
3. Exact oracle-mask overlap is not the only target: different masks can provide comparable KL recovery, so oracle-gain capture is the primary selector metric.
4. The strongest current operating point is a very small K-block budget, especially two blocks.
5. A fitted percentile-rank fusion of delta magnitude and alignment improved calibration overlap but failed on held-out KL and was removed.
6. V-cache selection remains unresolved and should retain separate calibration or fallback behavior.

## Artifacts

The isolated NPU evaluation artifacts are stored under:

- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_k_oracle_heldout4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_oracle_heldout4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_k_aggregation_2`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_aggregation_2`
