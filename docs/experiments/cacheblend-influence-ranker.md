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

### V-cache selector validation at larger scale

The initial V-cache conclusions were based on too few conditions. A later
validation fixed the operating point to a two-block budget and evaluated 114
held-out conditions that were not used for selector calibration:

- synthetic multi-hop tracing: samples 8-31, 24 conditions;
- synthetic aggregation: samples 200-225, 26 conditions;
- LongBench HotpotQA: offsets 4-35, 32 conditions;
- LongBench 2WikiMQA: offsets 4-35, 32 conditions.

The model and cache setting remained Qwen2.5-1.5B, context length 512, a
middle-layer V update, version gap 4, block size 64, and two repaired blocks
out of eight directly repairable blocks. All selectors below used zero
candidate-repair forwards.

The strongest common selector was `sparse_signed_first_residual_gain`. For a
block correction vector \(c_B\) and total predicted correction \(C\), its score
is

\[
2\langle c_B, C\rangle - \lVert c_B\rVert^2,
\]

which estimates the first-step decrease in residual correction energy. Across
the 114 held-out conditions it achieved:

- aggregate KL reduction versus stale reuse: 29.9%;
- beneficial-condition rate: 67.5%;
- harmful-condition rate: 32.5%;
- worst absolute KL increase: 0.00499;
- positive aggregate KL reduction on all eight evaluation splits;
- mean online score-plus-repair latency: 34.6 ms, versus 171.9 ms for full
  recomputation, a 79.9% reduction.

Plain attention mass was nearly tied in aggregate KL reduction at 29.8%, but
had a slightly lower beneficial-condition rate of 66.7%. Several magnitude and
residual selectors produced exactly the same attention-dominated masks on all
114 conditions. `sparse_signed_first_residual_gain` was the only selector on
the empirical recovery/tail-harm Pareto frontier.

The later `sparse_retrieval_headwise_gain` experiment improved the earlier
small V-cache splits, but did not remain the best method after expansion. On
the 114-condition validation it reduced KL by 20.6%, was beneficial in 62.3%
of conditions, and had a worst KL increase of 0.0107. It remains useful as a
diagnostic signed feature rather than the default V-cache selector.

Two additional attempts were rejected:

- an equal-rank Borda fusion of attention mass and head-wise gain improved the
  first 50 held-out conditions, but fell to 21.8% aggregate KL reduction after
  adding 64 new conditions and was removed;
- a learned block ranker trained on 16 KL-oracle calibration conditions
  collapsed to a fixed block pair on held-out data and achieved only 22.2% KL
  reduction on its 50-condition evaluation.

These larger results also show that individual V-cache repairs remain
non-monotonic. The two-block operating point is an aggregate policy, not a
per-condition safety guarantee.

### V-cache budget sweep

The fixed two-block operating point was checked against one, four, six, and
eight repaired blocks. A complete five-budget sweep was run on 64 held-out
conditions. Aggregate KL reductions were:

- one block: 15.3%;
- two blocks: 29.3%;
- four blocks: 25.3%;
- six blocks: 29.8%;
- eight blocks: 26.3%.

The small apparent advantage of six blocks on this split did not persist after
adding the other 50 held-out conditions. On the full 114-condition comparison:

- two blocks reduced KL by 29.9%;
- six blocks reduced KL by 28.3%;
- two blocks produced lower KL on 65 conditions, versus 49 for six blocks;
- both budgets were beneficial on 67.5% of conditions;
- mean cache-maintenance latency was 8.74 ms for two blocks and 13.88 ms for
  six blocks;
- worst absolute KL increase was 0.00499 for two blocks and 0.00575 for six
  blocks.

The paired bootstrap interval for the six-minus-two aggregate recovery
difference included zero, so there is no evidence that the extra four blocks
improve expected recovery. The six-block repair also added 5.13 ms of mean
cache-maintenance cost.

Budget effects were strongly non-monotonic. None of the 64 complete sweep
conditions improved monotonically across one, two, four, six, and eight
blocks. The per-condition best budget was distributed across all five choices:
12, 17, 12, 10, and 13 conditions respectively. Two blocks therefore remain
the preferred fixed operating point because they provide the best aggregate
recovery-efficiency trade-off, not because additional repair is guaranteed to
hurt.

The non-monotonic curves leave room for a dynamic budget policy. An oracle that
chooses between two and six blocks independently for each of the 114 held-out
conditions would reduce KL by 40.1%, improve 78.9% of conditions, and limit the
worst KL increase to 0.00259. This is substantially better than either fixed
budget.

However, the available zero-forward features did not recover this upper bound.
A grouped calibration procedure used 16 separate conditions and left out each
task family in turn while selecting one-feature thresholds over score-tail,
attention, correction-magnitude, cascade, and router features. Its selected
rule thresholded the third-highest first-residual score. Applied unchanged to
the 114 held-out conditions, it chose six blocks 45 times and reduced KL by
29.5%, slightly below the fixed two-block result of 29.9%. It captured no
positive fraction of the available two-versus-six oracle gain.

Dynamic V-cache budgeting therefore remains a planner problem. The current
method uses two blocks without a hand-written budget gate.

### Task-quality and answer-trajectory validation

The KL experiments above measure the next-token output distribution. A later
validation added full autoregressive generation and task-score fields to the
blockwise runner, then tested whether V-cache KL recovery transfers to answer
quality.

The usable real-task settings were Qwen2.5-7B with a 4096-token context for
2WikiMQA and an 8192-token context for HotpotQA. Both used a middle-layer V
LoRA update with configured norm 0.001. At version gap 4, the two-block
`signed_first_residual_gain` selector reduced next-token KL by 36.8% on the
first four 2WikiMQA conditions and 33.6% on the first four HotpotQA conditions.
However, the fresh update itself changed mean token F1 by only 0.00285 on
2WikiMQA and by zero on HotpotQA, so these conditions contained essentially no
task-score gain to recover.

Version-gap sweeps confirmed that this was not specific to gap 4. Across 40
middle-layer V sample-gap conditions at gaps 0, 4, 8, 16, and 32, fresh
recomputation, stale reuse, and the pre-update baseline produced identical
token F1. Updating all V projections on 2WikiMQA also produced no F1 changes
across 16 sample-gap conditions. The corresponding all-V HotpotQA run at an
8192-token context exceeded single-NPU memory during LoRA training.

Because exact generated-answer F1 was insensitive, a second evaluation used
teacher-forced correct-answer probes. The selector remained answer-free; answer
tokens were used only as an evaluation metric. On 32 new held-out real-task
conditions, `signed_first_residual_gain` achieved:

- next-token KL reduction: 27.9%;
- first correct-answer-token NLL gain: 0.0326 on average, with a bootstrap
  interval that included zero;
- two-token answer NLL gain: -0.0354, with a 95% bootstrap interval entirely
  below zero;
- four-token answer NLL gain: -0.0203, also with a 95% bootstrap interval
  below zero.

Thus, the current first-residual selector improves the immediate output
distribution but does not preserve the subsequent correct-answer trajectory.
`predicted_delta_norm` was significantly less harmful than first-residual on
the two- and four-token NLL probes, but its own mean gains remained negative
and statistically indistinguishable from zero. No evaluated zero-forward
selector demonstrated positive multi-token answer recovery.

The V-cache claim must therefore remain limited to next-token distribution
fidelity and latency. A task-quality V planner requires a multi-token,
answer-free influence objective rather than another refinement of the current
single-step residual score.

### K-cache causal propagation validation

The direct K-cache experiments initially treated a selected block as an
independent target-layer correction. This is incomplete: changing a prefix K
block changes the attention outputs of later positions, and those changes
propagate into downstream-layer K/V states. Two fresh-cache splice analyses
were therefore added:

- vertical propagation replaces the selected token columns in downstream
  layers;
- causal propagation replaces the selected target-layer K blocks and the
  causally affected suffix blocks in downstream layers.

These are analysis primitives using the full-fresh cache, not deployable
selective-recomputation kernels.

On 16 new Qwen2.5-1.5B synthetic conditions, the two-block
`signed_first_residual_gain` selector always selected token blocks `(0, 7)`.
Its direct K correction reduced aggregate KL by only 4.0%. Propagating the same
columns through all downstream layers reduced KL by 71.8% while replacing 25%
of eligible layer-token cells. A full causal wedge reduced KL by 84.8%, but it
replaced 94.6% of eligible cells because selecting block 0 makes nearly the
entire suffix causally dependent. Thus, the original selector-action pair is
not computationally useful even though its full causal repair is accurate.

An exhaustive search over all 28 two-block causal wedges revealed a much
better cost-quality region. On the same 16 conditions:

- fixed `(6, 7)` replaced 25.0% of eligible cells and reduced KL by 79.9%;
- fixed `(5, 7)` replaced 36.6% of cells and reduced KL by 81.4%;
- a per-condition oracle constrained to at most 50% of cells reduced KL by
  93.1% and never increased KL.

The `(5, 7)` pair was selected using only these 16 conditions and then frozen.
On the independent original ten synthetic conditions, it reduced KL by 90.9%,
improved nine of ten conditions, and reduced one- and two-token trajectory KL
by 95.8%. Across all 26 synthetic conditions, `(5, 7)` reduced KL by 87.6%; its
bootstrap 95% interval was 74.8%--92.0%. The cheaper `(6, 7)` pair reduced KL
by 84.9% across the same 26 conditions.

The real-task result is less uniform. Eight new Qwen2.5-7B conditions used a
4096-token 2WikiMQA context and an 8192-token HotpotQA context. Direct
two-block K correction increased aggregate next-token KL by 18.3%. Causal
late-block repair substantially improved 2WikiMQA: `(4, 5)` reduced next-token
KL by 96.3%. On HotpotQA, however, stale next-token KL was already very small,
and every fixed late pair increased that metric. Despite this, `(4, 5)` reduced
material one- and two-token trajectory KL on all HotpotQA conditions by 82.5%
and 72.7%, respectively. Selecting a fixed pair on one real task did not
transfer reliably to the other.

The K-cache conclusion therefore changes in two ways. First, downstream causal
propagation is the dominant missing component; direct target-layer correction
is not a sufficient repair action. Second, the planner should trade source
block value against causal recomputation cost and favor late blocks when their
quality is comparable. Fixed `(5, 7)` or `(6, 7)` rules are strong synthetic
baselines, not final task-independent planners. A deployable method still
requires a bounded selected-token downstream-recomputation kernel and an
answer-free rule for choosing both source blocks and propagation depth.

### Cost

Measured block-score extraction latency after the first invocation:

- K: 13.7 ms, 9.6% of full-recompute strategy latency;
- V: 11.0 ms, 7.7% of full-recompute strategy latency.

The first invocation was about 0.11 s because of accelerator warm-up. Runtime comparisons should warm the scorer before reporting steady-state latency.

## Conclusions

1. The useful CacheBlend transfer is impact-based block ranking, not cache delta correction.
2. Direct target-layer K correction is incomplete; selected K blocks require
   bounded causal downstream propagation.
3. Cost-aware late K blocks can recover most of the fresh distribution at a
   fraction of the full causal-wedge cost, but no task-independent pair is yet
   validated.
4. Exact oracle-mask overlap is not the only target: different masks can provide comparable KL recovery, so oracle-gain capture is the primary selector metric.
5. The strongest current K analysis uses two source blocks plus downstream
   propagation; two direct blocks alone are not a sufficient operating point.
6. A fitted percentile-rank fusion of delta magnitude and alignment improved calibration overlap but failed on held-out KL and was removed.
7. For V-cache repair at the fixed two-block operating point,
   `signed_first_residual_gain` is the current preferred zero-forward selector;
   larger repair budgets and per-condition safety remain unresolved.

## Artifacts

The isolated NPU evaluation artifacts are stored under:

- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_k_oracle_heldout4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_oracle_heldout4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_k_aggregation_2`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_aggregation_2`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_multi_8_15`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_multi_16_31`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_aggregation_200_209`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_aggregation_210_225`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_hotpotqa_offset4_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_hotpotqa_offset20_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_2wikimqa_offset4_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_borda_2wikimqa_offset20_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_rankers/v_mixed_cal16.json`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget_sweep_multi_16_31`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget_sweep_aggregation_210_225`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget_sweep_hotpotqa_offset20_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget_sweep_2wikimqa_offset20_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget6_multi_8_15`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget6_aggregation_200_209`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget6_hotpotqa_offset4_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_budget6_2wikimqa_offset4_16`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_dynamic_cal6_multi_0_3`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_dynamic_cal6_hotpotqa_0_3`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/b5_v_dynamic_cal6_2wikimqa_0_3`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_taskscore_qwen7b_2wikimqa_ctx4096_offset36_4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_taskscore_qwen7b_hotpotqa_ctx8192_offset36_4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_gap_screen_qwen7b_2wikimqa_ctx4096_offset36_4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_gap_screen_qwen7b_hotpotqa_ctx8192_offset36_4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_gap_screen_allv_qwen7b_2wikimqa_ctx4096_offset36_4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_refnll_qwen7b_2wikimqa_expanded_offset40_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_refnll_qwen7b_2wikimqa_expanded_offset48_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_refnll_qwen7b_hotpotqa_expanded_offset40_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/v_refnll_qwen7b_hotpotqa_expanded_offset48_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_compare_multi_offset8_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_compare_aggregation_offset202_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_pairs_multi_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_pairs_aggregation_8`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_pairs_old_multi_0_3`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_pairs_old_multi_4_7`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_pairs_old_aggregation_200_201`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_pairs_qwen7b_2wikimqa_ctx4096_offset80_4`
- `/mnt/caoyuhang/cyh/ttt-cache-influence-eval/runs/influence_eval/k_causal_pairs_qwen7b_hotpotqa_ctx8192_offset80_4`
