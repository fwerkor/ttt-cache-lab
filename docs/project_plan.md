# Project plan: versioned KV cache management for inference-time adapter training

This document fixes the project direction and the experiment plan. Future changes should be checked against this document before adding new code or experiments.

## 1. One-sentence goal

Study and build **versioned KV cache management for inference-time adapter training**, where an adapter is updated during inference and therefore evolves through versions `v0 -> v1 -> v2 -> ...` while KV cache blocks may have been produced by older versions.

## 2. Non-goals

The project is not primarily about:

- generic KV cache compression;
- generic vLLM / PagedAttention memory allocation;
- static multi-LoRA serving only;
- reproducing qTTT as the final contribution;
- only showing that Q-only updates can reuse K/V;
- building a production serving engine before the research effect is proven.

These topics can appear as baselines or implementation references, but they are not the central claim.

## 3. Core research question

During inference-time adapter training, a model state changes over time:

```text
base model + adapter version v0
  -> train one/few steps on current input/task
base model + adapter version v1
  -> train again
base model + adapter version v2
  -> ...
```

A KV cache block is tied to the parameter/adaptor version that produced it:

```text
KV block = f(prefix tokens, layer, base model, adapter, adapter version, precision, attention implementation)
```

The central question is:

> When the adapter version changes, which cached K/V blocks remain usable, which can be corrected, which should be refreshed, and which must be recomputed?

## 4. Positioning against related work

The important boundary is not simply "qTTT is Q-only". Related work already covers several broader cache-reuse cases.

| Area | What it handles | What remains open for this project |
|---|---|---|
| qTTT / query-only TTT | Avoids K/V invalidation by restricting updates to query projection | Does not manage cache under continuously evolving adapters |
| aLoRA / cross-model reuse | Reuses base-model prefix cache for static activated adapters | Assumes adapter activation/weights are static during serving |
| S-LoRA / multi-LoRA serving | Manages many static LoRA weights and KV caches | Focuses on serving many adapters, not online adapter version drift |
| LRAgent / ForkKV-like systems | Share/decompose KV cache for multi-LoRA or multi-agent scenarios | Mainly handles multiple static adapters/agents, not one adapter updated every step |
| PagedAttention / vLLM | Efficient KV cache allocation, paging, prefix sharing | Usually assumes fixed model parameters for cache validity |

This project should emphasize **adapter version evolution**:

```text
static adapter A / static adapter B  !=  online adapter v0 -> v1 -> v2 -> ...
```

## 5. Intended contributions

### C1. Versioned cache validity model

Define metadata and validity states for KV cache blocks under adapter evolution.

A cache block should eventually carry metadata like:

```text
{
  token_span,
  layer_id,
  base_model_id,
  adapter_id,
  adapter_version,
  cached_step,
  update_target_set,
  accumulated_update_norm,
  semantics
}
```

Validity states:

- `EXACT_VALID`: equivalent to full recomputation under the current adapter version.
- `FROZEN_VALID`: valid under frozen-evidence semantics.
- `DELTA_VALID`: can be made close to current-version cache through low-cost correction.
- `STALE_VALID`: stale but below an error/risk threshold.
- `INVALID`: must be refreshed or recomputed.

### C2. Empirical characterization of cache drift under adapter evolution

Measure how cache/logit/task error grows as the adapter version gap increases:

```text
cached version: v0
current version: v1, v2, v4, v8, v16, ...
```

Characterize drift by:

- update target: LoRA-Q / LoRA-K / LoRA-V / LoRA-MLP;
- layer position: early / middle / late;
- update magnitude: learning rate, update norm, number of steps;
- context length;
- model scale.

### C3. Versioned cache planner

Design a planner that chooses among:

- reuse old cache;
- reuse under frozen-evidence semantics;
- incremental delta correction;
- layer/block refresh;
- periodic refresh;
- full recomputation;
- rejecting unsafe reuse.

### C4. End-to-end evaluation

Show that the versioned planner preserves most of the benefit of full recomputation while reducing recomputation cost, and is more stable than always reusing stale cache.

## 6. Experiment dependency graph

The experiments are not independent. They should be performed in this order.

```text
E1. Static-adapter baseline alignment
  -> clarifies what existing cache-reuse ideas already cover

E2. Adapter-version drift characterization
  -> proves online adapter evolution creates a new cache-consistency problem

E3. Update-target x version-gap failure map
  -> identifies which updates are safe, correctable, or unsafe

E4. Versioned planner main experiment
  -> evaluates the proposed method against baselines

E5. Delta correction / base+delta cache experiment
  -> tests the most important low-cost maintenance mechanism

E6. Context-length and model-scale experiments
  -> tests whether the result scales

E7. Ablations and failure-boundary experiments
  -> explains which components matter and when reuse should be rejected
```

## 7. Experiment E1: static-adapter baseline alignment

### Purpose

Clarify the boundary with static adapter reuse work. This experiment is not the main contribution; it prevents the paper from overclaiming novelty.

### Question

What can static adapter cache reuse or base+delta cache decomposition already solve when adapters do not change during inference?

### Setup

Use a base model with one or more fixed adapters. Do not train adapters during this experiment.

Suggested initial models:

- Qwen2.5-0.5B or tiny GPT-2 for smoke tests;
- Qwen2.5-1.5B for first real runs;
- Qwen2.5-7B for main runs.

### Methods to compare

- `full_recompute`: recompute prefix under each adapter.
- `base_cache_reuse`: reuse base prefix cache where valid.
- `adapter_specific_cache`: cache per adapter.
- `alora_like_reuse`: invocation-before/after split, if implemented.
- `base_delta_cache`: base KV plus adapter-dependent delta, simplified baseline.

### Metrics

- logits KL against full recompute;
- top-1 agreement;
- task score;
- memory footprint;
- prefill/recompute latency;
- cache bytes per request/adaptor.

### Deliverables

- `runs/e1_static_adapter_baseline/merged_records.csv`
- table: static adapter reuse baseline comparison;
- plot: latency/memory vs quality for static reuse methods;
- short note in `docs/results/e1_static_adapter_baseline.md` explaining what is and is not covered by static-adapter methods.

### Success criterion

The experiment should show that static reuse helps in static adapter settings, but it should not be presented as solving online adapter version drift.

## 8. Experiment E2: adapter-version drift characterization

### Purpose

This is the core problem-motivation experiment.

### Question

How does stale-cache error grow as an adapter is updated through versions during inference-time training?

### Setup

For each sample:

```text
1. Load base model and attach adapter version v0.
2. Prefill prefix under v0 and save KV_v0.
3. Train adapter for T steps on the current task/input: v0 -> v1 -> ... -> vT.
4. At selected steps t in {1, 2, 4, 8, 16}, compute:
   a. full recompute under vt;
   b. reuse KV_v0 under vt;
   c. optionally reuse the latest refreshed cache version.
5. Compare stale outputs to full recompute outputs.
```

### Adapter update targets

Start with:

- LoRA-Q;
- LoRA-K;
- LoRA-V;
- LoRA-QV;
- LoRA-MLP-late.

Later add:

- LoRA-O;
- LoRA-attn;
- LoRA-all-late;
- early-layer LoRA-MLP;
- Norm or output-head as negative/control cases.

### Tasks

Synthetic first:

- passkey retrieval;
- key-value retrieval;
- multi-needle retrieval;
- variable tracking.

Then real tasks:

- LongBench subset;
- code/repo-level QA subset.

### Metrics

- logits KL vs full recompute;
- top-1 agreement vs full recompute;
- relative KV summary error;
- attention distribution shift, once available;
- task score;
- cache error as a function of `version_gap`;
- accumulated adapter update norm.

### Main plots

- line plot: `version_gap` vs logits KL for each update target;
- line plot: `version_gap` vs task score drop;
- table: first step at which stale reuse exceeds an error threshold.

### Deliverables

- `runs/e2_version_drift/merged_records.csv`
- `runs/e2_version_drift/grouped.csv`
- `figures/e2_version_gap_vs_kl.pdf`
- `figures/e2_version_gap_vs_task_drop.pdf`
- `docs/results/e2_version_drift.md`

### Success criterion

There must be structured drift patterns. Useful expected patterns:

- LoRA-Q drifts slowly;
- LoRA-K/V drifts faster;
- LoRA-MLP-late has localized or moderate drift;
- drift grows with version gap and update norm.

If all targets drift identically, the planner idea is weak and the project direction should be reconsidered.

## 8.5 Discovery gate W1/W2: finite propagation and minimum recompute windows

The original E3 design only compares stale reuse, full recompute, and suffix recompute from the first affected layer to the model end. That matrix can produce a useful failure map, but it cannot establish the more novel claim that update effects decay after a bounded number of layers. Before expanding E3 to all large models, run two discovery experiments.

### W1: finite recompute-window sweep

For an update beginning at layer `s`, construct a mixed prefix cache:

```text
layers [0, s)       : reuse the cached-version K/V
layers [s, e)       : recompute under the current parameters
layers [e, L)       : reuse the cached-version K/V
```

The exclusive end layer `e` is evaluated through window sizes 1, 2, 4, 8, 16, and 32. All six windows are distinct strategies inside the same run (`windowed_recompute_1` through `windowed_recompute_32`), so they share one model load, one sample, and exactly the same parameter-update trajectory. The sweep expands only update targets; it must not retrain the adapter independently for each window. Compare the paired windows with stale reuse, suffix `layerwise_recompute`, and full recompute. The first-stage matrix uses Qwen2.5-1.5B at 4K and Qwen2.5-7B at 8K, eight calibrated samples, Q/K/V/MLP updates at early, middle, and late positions, and version gaps 1, 4, and 16.

Primary outputs:

- safe-window rate across samples;
- minimum window satisfying KL, top-1, task-drop, and false-safe thresholds;
- minimum beneficial window that also improves or preserves KL, top-1, and task drop relative to paired stale reuse;
- best-KL window and the number of KL monotonicity violations as the window grows;
- latency/FLOP fraction of the selected window;
- frequency with which no tested finite window is safe or no finite window improves on stale reuse.

Run:

```bash
python -m ttt_cache_lab.cli versioned-sweep \
  --config configs/paper/discovery/w1_qwen_1_5b_multi_hop_window_sweep.yaml
python -m ttt_cache_lab.cli window-analysis \
  --input runs/paper/discovery/w1_qwen_1_5b_multi_hop_window/merged_records.csv \
  --output-dir runs/paper/discovery/w1_qwen_1_5b_multi_hop_window/analysis
```

### W2: layerwise propagation probe

At each selected version, compare the cached-version and current full-recompute states at every decoder layer. Record sampled-token hidden-state, K, and V relative error, cosine distance, and norm ratio. The probe samples a fixed number of token positions on device instead of copying complete long-context activations to the host.

Primary outputs:

- drift curves by layer;
- peak drift layer;
- tail-to-peak ratio;
- first layer after which drift remains below a fixed fraction of the peak;
- profile label: strong decay, partial decay, persistent, or late amplification.

Run:

```bash
python -m ttt_cache_lab.cli versioned-run \
  --config configs/paper/discovery/w2_qwen_1_5b_propagation.yaml
python -m ttt_cache_lab.cli propagation-analysis \
  --input runs/paper/discovery/w2_qwen_1_5b_propagation/propagation_records.csv \
  --output-dir runs/paper/discovery/w2_qwen_1_5b_propagation/analysis
```

### Go/no-go rule

Proceed to a learned or threshold-based window predictor only if W1 finds finite windows materially shorter than suffix recompute in a nontrivial fraction of conditions and W2 shows a reproducible decay or recovery pattern. If most conditions require recomputation to the model end and propagation remains persistent, retain E3 as a negative measurement study but do not position finite-window planning as the main contribution.

## 9. Experiment E3: update-target x version-gap failure map

### Purpose

Convert E2's curves into a decision map for cache planning.

### Question

For each update target and version gap, should cache be reused, corrected, refreshed, or fully recomputed?

### Matrix

Rows:

- LoRA-Q;
- LoRA-K;
- LoRA-V;
- LoRA-QV;
- LoRA-O;
- LoRA-MLP-late;
- LoRA-MLP-early;
- Norm;
- output head.

For models with tied input/output embeddings, `output_head` updates the shared embedding/output matrix returned by `get_output_embeddings()`; it is not interpreted as a nonexistent independent `lm_head` parameter.

Columns:

- step/version gap 1;
- 2;
- 4;
- 8;
- 16;
- 32 if feasible.

Cell values:

- logits KL;
- task score drop;
- relative cache error;
- top-1 disagreement rate.

### Procedure

Use the same multi-step adapter update runner as E2. Aggregate results by update target and version gap.

### Deliverables

- heatmap: update target x version gap -> logits KL;
- heatmap: update target x version gap -> task score drop;
- table: recommended policy per cell (`reuse`, `delta`, `refresh`, `full_recompute`);
- `docs/results/e3_failure_map.md`.

### Success criterion

The heatmap should support target-specific rules rather than a single universal refresh interval.

## 10. Experiment E4: versioned cache planner main experiment

### Purpose

Evaluate the actual proposed system policy.

### Question

Can a versioned cache planner reduce recomputation cost while preserving quality during online adapter training?

### Baselines

- `no_adaptation`: no adapter training.
- `full_recompute_every_step`: gold quality, highest cost.
- `always_stale`: cheapest, often unsafe.
- `periodic_refresh_N`: refresh every N steps.
- `static_base_delta`: static decomposition baseline where applicable.
- `threshold_refresh`: refresh when accumulated update norm exceeds threshold.
- `versioned_planner`: proposed method.
- `oracle_planner`: upper bound using future/full-recompute error information.

### Proposed planner inputs

- cached adapter version;
- current adapter version;
- version gap;
- update target;
- layer id;
- accumulated update norm;
- optional error proxy;
- latency/memory budget.

### Proposed planner outputs

- stale/frozen reuse;
- delta correction;
- partial refresh;
- full recompute;
- reject reuse.

### Metrics

Quality:

- task score;
- score drop vs full recompute;
- logits KL vs full recompute;
- top-1 agreement.

Cost:

- prefill/recompute latency;
- total adaptation+decode latency;
- recompute FLOPs proxy;
- KV memory footprint;
- number of refreshes;
- cache hit/reuse ratio.

### Main plots

- accuracy-latency Pareto plot;
- recompute cost vs task score;
- refresh count vs task score;
- table: relative speedup over full recompute at equal/near-equal score.

### Deliverables

- `runs/e4_planner_main/merged_records.csv`
- `figures/e4_accuracy_latency_pareto.pdf`
- `figures/e4_recompute_cost_vs_score.pdf`
- `docs/results/e4_planner_main.md`

### Success criterion

The planner should be strictly better than always stale and periodic refresh in at least one important region, and should substantially reduce recomputation cost compared with full recompute while keeping most of the score.

## 11. Experiment E5: delta correction and base+delta cache

### Purpose

Test the most important low-cost cache maintenance mechanism and clarify the boundary with base+adapter cache decomposition work.

### Question

When LoRA-K/V changes during online training, can we update cached K/V through low-rank delta correction rather than full recomputation?

### Approximation

For LoRA-K:

```text
K = h (W_K + B A)
```

If hidden state `h` is treated as unchanged for the correction step:

```text
K_new ≈ K_old + h * Δ(B A)
```

For LoRA-V:

```text
V_new ≈ V_old + h * Δ(B A)
```

This approximation must be evaluated against full recompute.

### Methods

- full recompute;
- stale old delta cache;
- incremental delta correction;
- periodic rebuild of delta cache;
- planner-selected delta vs refresh.

### Variables

- LoRA rank: 4, 8, 16;
- version gap: 1, 2, 4, 8, 16;
- update norm / learning rate;
- layer position: early, middle, late;
- update target: LoRA-K, LoRA-V, LoRA-QV.

### Metrics

- corrected KV error vs full recompute;
- logits KL;
- task score;
- correction latency;
- correction FLOPs proxy;
- memory for base and delta components.

### Deliverables

- `runs/e5_delta_correction/merged_records.csv`
- plot: version gap vs correction error;
- plot: rank/update norm vs correction quality;
- table: regions where delta correction is safe;
- `docs/results/e5_delta_correction.md`.

### Success criterion

Delta correction does not need to work everywhere. It is useful if it works for a clearly defined region such as small-step LoRA-K/V updates in late or middle layers.

## 12. Experiment E6: context-length and model-scale scaling

### Purpose

Demonstrate that the problem and method matter more as context length and model scale increase.

### Questions

- Does full recompute become increasingly expensive with longer contexts?
- Does the versioned planner's relative benefit grow with context length?
- Do drift patterns and planner decisions transfer across model sizes?

### Context lengths

Minimum:

- 4K;
- 8K;
- 16K;
- 32K.

If resources permit:

- 64K;
- 128K.

### Models

Minimum:

- Qwen2.5-1.5B;
- Qwen2.5-7B.

Preferred:

- Qwen2.5-1.5B;
- Qwen2.5-7B;
- Qwen2.5-14B;
- Qwen2.5-Coder-7B.

### Methods

Keep the method set small:

- full recompute;
- always stale;
- periodic refresh;
- versioned planner;
- oracle planner.

### Metrics

- latency;
- memory;
- task score;
- logits KL;
- recompute ratio;
- refresh count;
- throughput if serving-style batching is implemented.

### Deliverables

- `runs/e6_scaling/merged_records.csv`
- plot: context length vs latency;
- plot: context length vs score;
- plot: model scale vs relative speedup;
- `docs/results/e6_scaling.md`.

### Success criterion

The planner's speed/cost advantage should be more visible at 16K+ contexts and for 7B+ models.

## 13. Experiment E7: ablations and failure boundaries

### Purpose

Show which planner components matter and when cache reuse is unsafe.

### Ablations

Compare:

- full versioned planner;
- no version id;
- no update-target-specific rules;
- no update-norm threshold;
- no delta correction;
- no partial/layer refresh;
- no periodic fallback;
- always stale;
- always full recompute.

### Failure boundary variables

- learning rate / update norm;
- number of update steps;
- LoRA rank;
- early vs late layer;
- Norm updates;
- multi-layer updates;
- exact-value tasks such as passkey, numbers, code symbols;
- very long contexts.

### Metrics

- failure rate;
- task score drop;
- logits KL;
- top-1 disagreement;
- number of times planner rejects reuse;
- false-safe rate: planner reuses cache when full recompute would disagree.

### Deliverables

- ablation table;
- failure-boundary table;
- plot: update norm vs failure rate;
- `docs/results/e7_ablation_failure.md`.

### Success criterion

The planner should avoid unsafe reuse in high-risk regions. Negative results are acceptable if they define clear boundaries.

## 14. Implementation milestones

### M0. Current framework baseline

Already present:

- toy backend;
- HF backend with full recompute and stale/frozen prefix-cache reuse;
- update target taxonomy;
- cache planner skeleton;
- sweep runner;
- first-table reporting.

### M1. Real LoRA injection and online update

Implement:

- attach LoRA modules to selected Q/K/V/O/MLP projections;
- train LoRA for one or more steps on a synthetic task loss;
- record adapter version and accumulated update norm;
- restore/reset adapter state between samples.

Deliverable:

- `configs/e2_lora_drift_qwen_0_5b.yaml`
- working run that records `adapter_version`, `cached_version`, `version_gap`.

### M2. Multi-step versioned runner

Change runner from single update to multi-step evolution:

```text
prefill at v0
for t in steps:
    train adapter to vt
    full recompute reference
    apply each cache strategy
    record drift and score
```

Deliverable:

- `versioned_records.csv` with per-step records.

### M3. Static adapter baselines

Implement simplified static baselines:

- per-adapter full cache;
- base cache reuse where valid;
- base+delta cache summary baseline.

Deliverable:

- E1 runnable config and result table.

### M4. Delta correction

Implement LoRA-K/V correction:

- capture hidden states needed for correction;
- compute delta from LoRA weight changes;
- apply correction to cached K/V;
- compare with full recompute.

Deliverable:

- E5 runnable config and correction-error plot.

### M5. Versioned planner

Implement planner using:

- version gap;
- update target;
- accumulated update norm;
- layer position;
- optional error proxy.

Deliverable:

- E4 planner main experiment.

### M6. Scaling experiments

Run on real GPU hardware:

- Qwen2.5-1.5B and 7B;
- 4K/8K/16K/32K contexts;
- later 14B and 64K if feasible.

Deliverable:

- E6 scaling plots.

## 15. Suggested hardware usage

Use smaller GPUs for development and sweeps:

- 4x T10: smoke tests, 0.5B/1.5B, short contexts, config sweeps.

Use Ascend as an available platform for real experiments:

- 8x Ascend 910B: available platform for Qwen2.5-1.5B/7B/14B, 16K/32K contexts, and multi-step LoRA updates.
- Use the eight cards first as parallel sweep workers, one process per visible NPU.
- Distributed model parallelism can be added later only if 14B/32K+ runs require it.

CUDA GPUs can remain as optional validation or fallback; the research story should not depend on a specific accelerator vendor.

## 16. Minimal viable paper experiment set

If time is limited, produce the following minimum set:

1. E2 drift curves for Qwen2.5-1.5B and Qwen2.5-7B.
2. E3 update-target x version-gap heatmaps.
3. E4 planner main Pareto plot.
4. E5 delta correction only for LoRA-K/V.
5. E6 scaling only for 4K/8K/16K/32K.
6. E7 failure-boundary table for update norm and version gap.

This minimum set should be enough to support the core claim if the results are positive.

## 17. Decision gates

Use these gates to avoid drifting into weak directions.

### Gate 1: Is adapter-version drift structured?

After E2, continue only if drift depends meaningfully on update target, version gap, layer position, or update norm.

If every update behaves the same, the planner has little basis.

### Gate 2: Does online LoRA adaptation actually change task behavior?

After M1/M2, continue only if LoRA training produces measurable changes in loss/logits/task score.

If adaptation has no effect, cache management is not meaningful.

### Gate 3: Is there a region between stale and full recompute?

Continue only if at least one strategy such as delta correction, periodic refresh, or versioned planner provides a useful middle point:

```text
cost lower than full recompute
quality better or safer than always stale
```

### Gate 4: Does the benefit grow with context length?

If 16K/32K shows no cost benefit, the system angle is weak.

## 18. Naming convention for future runs

Use stable output directories:

```text
runs/e1_static_adapter_baseline/
runs/e2_version_drift/
runs/e3_failure_map/
runs/e4_planner_main/
runs/e5_delta_correction/
runs/e6_scaling/
runs/e7_ablation_failure/
```

Use matching docs:

```text
docs/results/e1_static_adapter_baseline.md
docs/results/e2_version_drift.md
docs/results/e3_failure_map.md
docs/results/e4_planner_main.md
docs/results/e5_delta_correction.md
docs/results/e6_scaling.md
docs/results/e7_ablation_failure.md
```

## 19. What should not be claimed until implemented

Do not claim any of the following until code and experiments support them:

- real qTTT is reproduced;
- native mid-layer recomputation generalizes beyond the explicitly tested Llama/Qwen-like and GPT-2 decoder families;
- LoRA-weight-delta correction is validated on real models across safe regions rather than only unit tests and toy runs;
- planner beats full recompute/stale baselines on real tasks;
- multi-GPU serving performance is measured;
- results generalize to 32K/64K contexts.

## 20. Current status as of this document

Implemented:

- toy backend with task-correct scoring for passkey, key-value, multi-needle, and variable-tracking diagnostics;
- HF/Ascend backend for full recompute, stale/frozen prefix cache reuse, native tested-family layer restart, and LoRA-weight-delta K/V correction;
- LoRA injection/training for selected `torch.nn.Linear` projections;
- multi-step adapter version evolution;
- versioned cache metadata and planner cost/safety metrics in records;
- expanded update target taxonomy including QV/attention and early/middle/late layer positions;
- static adapter baseline strategies, threshold refresh, delta correction, oracle planner, and adaptive planner;
- E1-E7 toy templates plus selected HF/Ascend E2/E5/E6 templates;
- condition-preserving failure-map, Pareto, version-summary, E5 safe-region, E6 scaling, E7 paired-ablation, and adaptation-effect reports;
- E3 failure-map calibration wired into E4 with artifact hashes and runtime latency budgets;
- explicit cache-manager scopes, raw/applied update diagnostics, attention availability, baseline provenance, atomic checkpoints, resume, cross-run merging, and structured failure manifests;
- CI with lint, typecheck, tests.

Still limited:

- real-model validation of LoRA-weight-delta correction safe regions;
- native layer restart validation outside the currently tested decoder families and under large sharded models;
- official upstream aLoRA/LRAgent/ForkKV comparisons; local variants are explicitly labeled as paper reimplementations or adapted baselines;
- completed long-context 7B/32B accelerator results;
- paper-quality final plots populated with real runs.

The immediate next step is hardware execution: run the small-model smoke test, generate E3, run E4 against the exact calibrated map, then execute E5/E6/E7 and merge the model/context runs for the final analysis.


## 21. Frozen paper-scale protocol

The earlier minimal experiment set in this roadmap is superseded for the main submission by [`paper_experiment_protocol.md`](paper_experiment_protocol.md).

The frozen study now includes:

- deterministic and disjoint calibration, validation, main-test, and ablation partitions;
- six controlled diagnostic task families plus LongBench, LongBench v2, and repository-code tasks;
- three model/update seeds with fixed dataset membership;
- Qwen2.5-7B as the complete primary evaluation;
- Qwen2.5-14B and Qwen2.5-32B as required scale evidence rather than optional examples;
- Mistral-7B cross-family transfer and Qwen2.5-Coder-7B code evaluation;
- E1-E8, adding sustained cache-pressure and tail-latency evaluation;
- conservative failure-map selection across all compatible calibration seeds and tasks;
- paired cluster-bootstrap intervals, Wilson false-safe bounds, and repeated p50/p95 performance measurement;
- a 72-configuration, 216-job shardable manifest under `configs/paper/`, with all six controlled tasks represented at 1.5B, 7B, 14B, and 32B.

The paper experiment order is now enforced by artifact dependencies: E4/E5/E6/E7/E8 jobs require the finalized E3 failure map. Test results must not be used to modify that artifact.
