# Paper experiment protocol

This document freezes the paper-scale evaluation protocol for versioned KV-cache management under inference-time adapter evolution. Changes to tasks, partitions, seeds, safety thresholds, model roles, or primary metrics must be reviewed as protocol changes rather than ordinary configuration edits.

## 1. Claims under evaluation

The paper evaluates four claims:

1. KV-cache drift under online adapter evolution is structured by update target, layer position, update magnitude, version gap, task, context length, and model scale.
2. Low-cost actions such as stale reuse, delta correction, and downstream-layer recomputation occupy useful quality-cost regions between unconditional reuse and full recomputation.
3. A planner calibrated only on designated calibration data transfers to held-out tasks and samples without using test outcomes.
4. The benefit grows at long context, remains visible on 32B headline conditions, and survives a sustained capacity-limited trace on the 7B primary model.

The experiment framework must not claim production serving integration, universal safety, or generalization beyond the measured models and workloads.

## 2. Experiment groups

| Group | Question | Primary evidence |
|---|---|---|
| E1 | What do static-adapter cache methods already solve? | 7B LongBench v2 static-adapter comparison |
| E2 | How does cache error grow with adapter versions? | 7B controlled/LongBench-v2 drift plus a 32B controlled confirmation |
| E3 | Which target/gap regions permit reuse, correction, or refresh? | Six-task exhaustive 1.5B calibration plus three representative tasks at 7B/32B |
| E4 | Does the calibrated planner improve the quality-cost frontier? | Held-out LongBench-v2 plus representative QA, retrieval, summarization, code, cross-family, and 32B evaluation |
| E5 | Where is low-rank K/V delta correction safe? | A 7B boundary grid over rank 4/16, update norm 1e-4/1e-3, and layer position |
| E6 | How do cost and benefit scale? | 4K/16K/64K tests on 7B and 8K/32K headline tests on 32B |
| E7 | Which planner components matter? | Held-out LongBench v2 component ablations |
| E8 | What happens during a sustained capacity-limited trace? | A 7B global-cache-manager trace with LRU eviction, p95/p99 latency, and false-safe rate |
| A1 | Do the discovered failure boundaries transfer across decoder structures? | Three-task lightweight screening on dense GQA, sliding-window attention, local/global attention, and sparse MoE models |

## 3. Task hierarchy

### 3.1 Controlled long-context diagnostics

The controlled suite isolates cache-consistency mechanisms:

- `multi_needle`: retrieve a selected value among multiple needles;
- `needle_absent`: reject a missing record instead of hallucinating a value;
- `multi_hop_tracing`: follow a dispersed pointer chain;
- `aggregation`: compare the frequencies of two event labels inside an explicitly delimited ledger;
- `common_words`: compute an intersection across long lists;
- `variable_tracking`: recover the final value after dispersed state updates.

The framework also retains version-routed `passkey` and `key_value` tasks. Passkey difficulty changes the number of routing hops rather than only adding filler: easy is revision-to-value, medium is slot-to-revision-to-value, and hard is profile-to-slot-to-revision-to-value. Controlled samples are generated deterministically from `data.selection_seed`; model/update seeds never change sample membership. `answer_length` controls the generated reference value, while paper-scale synthetic configs reserve at least 32 generation tokens so instruction-style prefixes do not truncate the answer.

E2 uses model-specific controlled tasks selected by baseline viability probes: Qwen2.5-1.5B uses easy version-routed passkey, Qwen2.5-7B uses easy variable tracking, and Qwen2.5-14B/32B use hard common-words with set-F1. This avoids the observed floor/ceiling discontinuity of applying one retrieval task to every scale.
E2 also separates micro-drift and task-ability evidence. Small-update runs characterize distributional sensitivity; paper controlled and realistic E2 runs use `update_norm: 1e-3` on held-out samples and must report whether fresh-cache adaptation gain is available before interpreting gain-retention ratios.

E3 uses model-calibrated contexts and semantic difficulty rather than forcing the same 16K hard task onto every scale. The core calibration matrix is:

| Model | Context | Core tasks |
|---|---:|---|
| Qwen2.5-1.5B-Instruct | 4K | multi-needle, needle absent, multi-hop, aggregation, common words, variable tracking |
| Qwen2.5-7B-Instruct | 8K | multi-needle, multi-hop, aggregation |
| Qwen2.5-32B-Instruct | 16K | multi-needle, multi-hop, aggregation |

The original six-task 7B/14B/32B grid remains in `configs/paper/study_extended.yaml`. Core task difficulties retain the previously probed model-specific values. Results from different cells remain explicit through `context_length` and `synthetic_difficulty` and must not be pooled as if they were the same workload.

E3 records only four actions in the formal failure map: full recomputation, stale reuse, delta correction, and layerwise recomputation. The already-running 1.5B aggregation configuration is an eight-strategy superset whose config hash is preserved; failure-map generation filters that artifact to the four core actions.

Before expanding E3 beyond the ongoing Qwen2.5-1.5B calibration, W1-W4 form a discovery gate. W1 evaluates finite recompute windows as paired strategies within one model/update trajectory and reports both the smallest safe interval and the smallest interval that improves or preserves quality relative to paired stale reuse; it also records any KL monotonicity violations as the window grows. Window sizes must not be compared across independently retrained runs. W2 records per-layer hidden/K/V drift and classifies whether propagation decays, persists, or amplifies. W3 evaluates local-boundary and whole-stale-suffix signals using sample-held-out prediction. W4 evaluates layer-token cache-splice masks at equal cache-cell budgets. Every W4 block size and selector must share the same trained adapter state and old/full cache pair. W4 is an oracle cache-surgery study, so its quality gains are not reported as deployable speedups until a real sparse-recompute or copy-on-write path exists. These experiments use calibration samples only. The selected 32B E3 expansion and any learned planner proceed only if the corresponding discovery mechanism produces a reproducible quality-cost advantage. The 14B expansion remains optional in the extended matrix. Otherwise that mechanism is reported as a negative result and the main study remains a failure-boundary measurement.

### 3.2 Real long-context tasks

The primary realistic benchmark is LongBench v2. Its multiple-choice schema gives deterministic automatic scoring while covering document QA, multi-document QA, long in-context learning, dialogue history, code-repository understanding, and structured-data understanding.

The original LongBench suite supplies open-ended evaluation:

- `hotpotqa` and `2wikimqa` for QA;
- `passage_retrieval_en` for retrieval;
- `passage_count` for aggregation;
- `gov_report` for summarization;
- `lcc` and `repobench-p` for code and repository completion.

Qwen2.5-Coder-7B is used for the code-specific tasks. General-language tasks use Qwen2.5-7B unless the experiment explicitly studies model scale or cross-family transfer.

## 4. Frozen data partitions

All external datasets are shuffled once with `selection_seed: 2027`. The model seed controls adapter optimization only.

For LongBench v2:

| Partition | Offset | Count | Use |
|---|---:|---:|---|
| validation | 0 | 96 | baseline and threshold selection |
| main test | 96 | 256 on Qwen-7B; 96 on other models | held-out E2/E4 evaluation |
| ablation test | 352 | 96 | E7 only |

These index ranges are disjoint. Test outcomes must not be used to regenerate the E3 failure map or tune periodic/threshold baselines.

Controlled calibration uses samples beginning at offset 0. Controlled E2, E5, E6, and E8 use later offsets so that the planner is not evaluated on its calibration instances.

Every record stores:

- `dataset_sample_id`;
- `evaluation_partition`;
- `selection_seed`;
- benchmark, task family, split, and category;
- model/update seed.

## 5. Model roles

| Model | Role |
|---|---|
| Qwen2.5-1.5B-Instruct | Exhaustive target/layer mechanism sweep |
| Qwen2.5-7B-Instruct | Primary complete paper evaluation |
| Qwen2.5-14B-Instruct | Extended-matrix intermediate-scale confirmation only |
| Qwen2.5-32B-Instruct | Large-model headline and scaling evidence |
| Mistral-7B-Instruct-v0.3 | Existing held-out cross-family LongBench-v2 transfer |
| Mistral-7B-Instruct-v0.1 | Sliding-window-attention architecture screening |
| Llama-3.2-3B-Instruct | Independent dense GQA family screening |
| Gemma-3-4B-IT | Alternating local/global-attention architecture screening |
| Qwen1.5-MoE-A2.7B-Chat | Sparse-MoE screening with separate router, shared-expert, and routed-expert targets |
| Qwen2.5-Coder-7B-Instruct | Code and repository tasks |

The 32B core configurations use explicit model-layer sharding across all visible accelerators. The 14B configurations remain available only in the extended matrix. The 7B configurations default to one visible accelerator, except long-context runs that may be changed to model sharding without changing task membership or analysis semantics. A1 architecture-screening configs are opt-in and are not referenced by default launch scripts; each controlled configuration uses at least 48 samples, six update targets, four cache strategies, and version gaps 1/4/16 before any model is promoted to a full matrix.

Records use backend-reported layer count, hidden size, and parameter count. Configuration defaults are not accepted as model-scale measurements.

## 6. Calibration and held-out evaluation

E3 produces the only failure-map artifact consumed by the planner:

```text
runs/paper/calibration/final/failure_map/failure_map.csv
```

A strategy is eligible only when it is safe across every compatible calibration cell for the requested model, context length, LoRA rank, update norm, and update mode. One safe seed or task cannot make a strategy eligible.

The execution order is fixed:

1. run E1 and E2, which do not consume the failure map;
2. run all core `calibration` jobs;
3. merge calibration records and generate the failure map;
4. run validation jobs and freeze baseline/planner hyperparameters;
5. run E4/E5/E6/E7/E8 test stages without modifying the map;
6. merge each stage and generate statistics and experiment-specific analyses.

Any change to the failure map after observing test results invalidates that test run.

## 7. Randomness and uncertainty

The paper manifest uses model/update seeds:

```text
7, 17, 29
```

Dataset membership remains fixed across seeds. Primary quality comparisons are paired on the same dataset sample, seed, target, adapter version, rank, update norm, and context.

Reports include:

- mean, median, standard deviation, and 95% cluster-bootstrap confidence intervals;
- paired task-score and latency differences against full recomputation;
- paired speedup;
- p50, p90, p95, and p99 latency;
- Wilson confidence intervals for false-safe rates, including a nonzero upper bound when no failure is observed.

Bootstrap clusters are `(seed, dataset_sample_id)`. The default paper analysis uses 5,000 resamples.

## 8. Performance measurement

Quality-heavy runs use one strategy execution per condition. Core E6 performance runs use:

- 1 warm-up execution;
- 5 measured executions;
- p50 as the primary latency;
- p95 as the tail-latency result;
- separately recorded adaptation, cache-maintenance, and decode latency.

A 3-warm-up/10-measurement confirmation pass is optional only when a headline cell exhibits unstable timing; it is not part of the default frozen matrix. E8 derives p50/p95/p99 from the sustained request trace and executes each request condition once rather than repeatedly converting the trace into a microbenchmark. Full recomputation is executed again during every configured timed repetition. Reusing a previously computed reference as a timing result is prohibited.

The run metadata records git commit, config hash, package versions, dtype, attention implementation, visible accelerators, and the exact failure-map hash.

## 9. Main model/task matrix

The default core matrix contains 47 configurations and expands to 141 jobs at three seeds. The original 84-configuration/252-job matrix is retained as `configs/paper/study_extended.yaml` and is not a submission prerequisite. Twelve A1 jobs still cover four additional architectures across multi-hop tracing, multi-needle retrieval, and variable tracking.

The principal core results are:

- E4 on Qwen2.5-7B over 256 held-out LongBench-v2 samples;
- E4 on representative QA, retrieval, summarization, and code tasks;
- E4 transfer on Qwen2.5-32B and Mistral-7B;
- E5 boundary combinations on Qwen2.5-7B;
- E6 context scaling through 64K for 7B and 32K for 32B;
- E8 a capacity-limited trace on 7B.

The selected 32B runs are primary headline evidence, not optional smoke tests. If a selected 32B condition is infeasible, its failure manifest and resource limit must be reported rather than silently replacing it with a smaller model.

## 10. Execution

Generate the stable job matrix:

```bash
python -m ttt_cache_lab.cli study-plan \
  --manifest configs/paper/study.yaml
```


The archived extended matrix can be inspected or run explicitly with:

```bash
python -m ttt_cache_lab.cli study-plan \
  --manifest configs/paper/study_extended.yaml
```

It must not be substituted for or merged into the core progress denominator.

Run a core tag locally:

```bash
python -m ttt_cache_lab.cli study-run \
  --manifest configs/paper/study.yaml \
  --tag calibration
```

Distribute jobs across workers or accelerator groups:

```bash
scripts/run_paper_shard.sh configs/paper/study.yaml 0 8
scripts/run_paper_shard.sh configs/paper/study.yaml 1 8
```

Finalize calibration before any dependent stage:

```bash
scripts/finalize_paper_calibration.sh
```

Then run and finalize stages:

```bash
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag validation
scripts/finalize_paper_stage.sh validation

python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag test
scripts/finalize_paper_stage.sh test
```

The same finalizer accepts `baseline`, `drift`, `delta`, `scaling`, `ablation`, and `workload`.

## 11. Minimum acceptance conditions

The paper claims are supported only if all of the following hold:

- online adaptation produces a measurable gain over the unadapted baseline on at least part of the task suite;
- drift differs systematically by target, layer, gap, or update norm;
- the planner improves the held-out quality-cost frontier over tuned periodic and update-norm baselines;
- its false-safe confidence bound remains acceptably low on held-out data;
- benefits persist on 7B and are confirmed on the selected 32B headline conditions;
- 16K+ contexts show a materially stronger cost advantage than short contexts;
- E5 identifies a nontrivial delta-correction safe region, or clearly documents that none exists;
- E8 does not hide cache-capacity failures behind aggregate mean latency.

Negative results remain publishable evidence when they establish clear rejection boundaries. They must not be removed from the task or model matrix after observation.
