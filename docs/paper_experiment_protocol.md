# Paper experiment protocol

This document freezes the paper-scale evaluation protocol for versioned KV-cache management under inference-time adapter evolution. Changes to tasks, partitions, seeds, safety thresholds, model roles, or primary metrics must be reviewed as protocol changes rather than ordinary configuration edits.

## 1. Claims under evaluation

The paper evaluates four claims:

1. KV-cache drift under online adapter evolution is structured by update target, layer position, update magnitude, version gap, task, context length, and model scale.
2. Low-cost actions such as stale reuse, delta correction, and downstream-layer recomputation occupy useful quality-cost regions between unconditional reuse and full recomputation.
3. A planner calibrated only on designated calibration data transfers to held-out tasks and samples without using test outcomes.
4. The benefit grows at long context and remains visible on 7B, 14B, and 32B models under sustained cache pressure.

The experiment framework must not claim production serving integration, universal safety, or generalization beyond the measured models and workloads.

## 2. Experiment groups

| Group | Question | Primary evidence |
|---|---|---|
| E1 | What do static-adapter cache methods already solve? | 7B LongBench v2 static-adapter comparison |
| E2 | How does cache error grow with adapter versions? | 7B/14B/32B controlled and LongBench v2 drift curves |
| E3 | Which target/gap regions permit reuse, correction, or refresh? | Multi-task calibration failure map at 1.5B/7B/14B/32B |
| E4 | Does the calibrated planner improve the quality-cost frontier? | Held-out LongBench, LongBench v2, and cross-family evaluation |
| E5 | Where is low-rank K/V delta correction safe? | Rank × update-norm × layer-position grids on 7B and 32B |
| E6 | How do cost and benefit scale? | 4K–64K context tests on 7B and 8K–32K tests on 14B/32B |
| E7 | Which planner components matter? | Held-out LongBench v2 component ablations |
| E8 | What happens during a sustained capacity-limited trace? | Global cache manager, LRU eviction, p95 latency, and false-safe rate |

## 3. Task hierarchy

### 3.1 Controlled long-context diagnostics

The controlled suite isolates cache-consistency mechanisms:

- `multi_needle`: retrieve a selected value among multiple needles;
- `needle_absent`: reject a missing record instead of hallucinating a value;
- `multi_hop_tracing`: follow a dispersed pointer chain;
- `aggregation`: compare the frequencies of two event labels inside an explicitly delimited ledger;
- `common_words`: compute an intersection across long lists;
- `variable_tracking`: recover the final value after dispersed state updates.

The framework also retains `passkey` and `key_value` for smoke tests. Controlled samples are generated deterministically from `data.selection_seed`; model/update seeds never change sample membership. `answer_length` controls the generated reference value, while the decode budget is independently set by `max_generation_tokens` (16 by default) so instruction-style response prefixes do not truncate the answer or leak the reference tokenizer length.

E3 uses model-calibrated contexts and semantic difficulty rather than forcing the same 16K hard task onto every scale. The frozen calibration matrix is:

| Model | Context | Multi-needle | Needle absent | Multi-hop | Aggregation | Common words | Variable tracking |
|---|---:|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 4K | medium | hard | easy | easy | easy | easy |
| Qwen2.5-7B-Instruct | 8K | hard | hard | medium | medium | hard | easy |
| Qwen2.5-14B-Instruct | 16K | hard | hard | medium | hard | hard | hard |
| Qwen2.5-32B-Instruct | 16K | hard | hard | hard | hard | hard | hard |

These settings are selected only from baseline task-viability probes. They are fixed before versioned cache experiments and remain explicit record dimensions through `context_length` and `synthetic_difficulty`; results from different cells must not be pooled as if they were the same workload.

Before expanding E3 beyond the ongoing Qwen2.5-1.5B calibration, W1 and W2 form a discovery gate. W1 evaluates finite recompute windows as paired strategies within one model/update trajectory and selects the smallest interval that is safe across samples; window sizes must not be compared across independently retrained runs. W2 records per-layer hidden/K/V drift and classifies whether propagation decays, persists, or amplifies. These experiments use calibration samples only. The 14B/32B E3 expansion and any learned window predictor proceed only if bounded windows materially outperform suffix recompute and the layerwise profiles show a reproducible recovery pattern. Otherwise the finite-window hypothesis is reported as a negative result and the main study remains a failure-boundary measurement.

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
| validation | 0 | 64 | baseline and threshold selection |
| main test | 64 | 256 on Qwen-7B; 96 on other models | held-out E2/E4 evaluation |
| ablation test | 320 | 96 | E7 only |

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
| Qwen2.5-14B-Instruct | Intermediate-scale transfer and performance confirmation |
| Qwen2.5-32B-Instruct | Large-model headline and scaling evidence |
| Mistral-7B-Instruct-v0.3 | Cross-family transfer |
| Qwen2.5-Coder-7B-Instruct | Code and repository tasks |

The 14B and 32B configurations use explicit model-layer sharding across all visible accelerators. The 7B configurations default to one visible accelerator, except long-context runs that may be changed to model sharding without changing task membership or analysis semantics.

Records use backend-reported layer count, hidden size, and parameter count. Configuration defaults are not accepted as model-scale measurements.

## 6. Calibration and held-out evaluation

E3 produces the only failure-map artifact consumed by the planner:

```text
runs/paper/calibration/final/failure_map/failure_map.csv
```

A strategy is eligible only when it is safe across every compatible calibration cell for the requested model, context length, LoRA rank, update norm, and update mode. One safe seed or task cannot make a strategy eligible.

The execution order is fixed:

1. run E1 and E2, which do not consume the failure map;
2. run all `calibration` jobs;
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

Quality-heavy runs use one strategy execution per condition. Dedicated E6 performance runs use:

- 3 warm-up executions;
- 10 measured executions;
- p50 as the primary latency;
- p95 as the tail-latency result;
- separately recorded adaptation, cache-maintenance, and decode latency.

Full recomputation is executed again during each timed repetition. Reusing a previously computed reference as a timing result is prohibited.

The run metadata records git commit, config hash, package versions, dtype, attention implementation, visible accelerators, and the exact failure-map hash.

## 9. Main model/task matrix

The checked-in matrix contains 72 configurations and expands to 216 jobs at three seeds. Every Qwen calibration scale covers all six controlled task families, so task coverage does not silently shrink as model size increases.

The principal results are:

- E4 on Qwen2.5-7B over 256 held-out LongBench v2 samples;
- E4 on seven open-ended LongBench tasks, including two code tasks;
- E4 transfer on Qwen2.5-14B, Qwen2.5-32B, and Mistral-7B;
- E5 rank/update-norm grids on Qwen2.5-7B and Qwen2.5-32B;
- E6 context scaling through 64K for 7B and through 32K for 14B/32B;
- E8 capacity-limited traces on 7B and 32B.

The 32B runs are part of the planned primary evidence, not optional smoke tests. If a specific 32B condition is infeasible, its failure manifest and resource limit must be reported rather than silently replacing it with a smaller model.

## 10. Execution

Generate the stable job matrix:

```bash
python -m ttt_cache_lab.cli study-plan \
  --manifest configs/paper/study.yaml
```

Run a tag locally:

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
- benefits persist on 7B and are confirmed on 14B or 32B;
- 16K+ contexts show a materially stronger cost advantage than short contexts;
- E5 identifies a nontrivial delta-correction safe region, or clearly documents that none exists;
- E8 does not hide cache-capacity failures behind aggregate mean latency.

Negative results remain publishable evidence when they establish clear rejection boundaries. They must not be removed from the task or model matrix after observation.
