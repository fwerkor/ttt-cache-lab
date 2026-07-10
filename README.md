# TTT Cache Lab

TTT Cache Lab is an experiment framework for studying **versioned KV cache management under inference-time adapter evolution**.

The project focuses on the following setting:

```text
adapter version v0
  -> train/update during inference
adapter version v1
  -> train/update again
adapter version v2
  -> ...
```

A KV cache block is produced by a specific model/adaptor version. Once the adapter changes, the old cache may be exact, approximately usable, stale-but-tolerable, or invalid. This repository provides the code structure needed to measure that drift and evaluate cache-maintenance strategies.

The framework supports lightweight toy diagnostics and Hugging Face experiments on CPU or CUDA GPUs.

## Research question

> During inference-time adapter training, when an adapter evolves from `v0` to `v1`, `v2`, and later versions, which KV cache blocks can be reused, which can be corrected, which should be refreshed, and which must be recomputed?

The project is not just about Q-only qTTT. Existing work already covers parts of static adapter reuse and multi-LoRA serving. This repository targets the dynamic case where the **same adapter keeps changing during inference**.

## Current implementation status

Implemented:

- synthetic diagnostics plus local JSONL and Hugging Face dataset loaders, including LongBench-style field mapping;
- exact tokenizer-level context sizing with explicit `error`, `left`, and `middle` truncation policies;
- Q/K/V/O/QV/attention/MLP/Norm/output-head update targets and layer-positioned LoRA variants;
- answer-supervised online LoRA updates, global-L2 update-norm control, and multi-token exact-match generation scoring;
- versioned per-layer cache metadata, adapter/version indexing, relative update norm since the last refresh, and executable rejection semantics;
- full recomputation, stale/frozen reuse, LoRA K/V delta correction, native Llama/GPT-2 layer restart, periodic/threshold policies, measured oracle selection, and adaptive planning;
- fixed-adapter E1 baselines, including executable aLoRA-style invocation-prefix reuse, LRAgent-style per-adapter caches, and ForkKV-style base/delta decomposition;
- real KV tensor byte counts, peak allocated memory, adaptation/cache/decode/end-to-end latency, throughput, cache-entry counts, and configurable tensor/task metrics;
- E1-E7 toy and Hugging Face templates; E5 rank/update-norm sweeps; E6 exact 4K/8K/16K/32K context sweeps;
- single-model layer sharding across multiple CUDA GPUs, including a 6×GPU Qwen2.5-32B template;
- strategy-aware failure maps, configurable safety thresholds, Markdown reports, SVG trends, and quality-cost Pareto figures;
- CI for linting, strict type checking, unit tests, and offline tiny-Llama integration tests that execute real LoRA, KV delta correction, and native layer restart paths.

Remaining validation work is experimental rather than placeholder implementation: the large-model paths still need to be run on the selected GPU infrastructure, and the adapted related-work baselines should be compared against official upstream implementations where licensing and environments permit.

## Repository layout

```text
src/ttt_cache_lab/
  cache/         cache validity semantics, strategies, and planner
  data/          synthetic, JSONL, and Hugging Face task loaders
  experiments/  single-step runner, versioned runner, sweep, summaries, reports
  metrics/      cache/logit/task metrics
  models/       toy and Hugging Face backends
  updates/      update targets and executable random/LoRA updaters
configs/
  feasibility_*.yaml          early single-step configs
  sweep_*.yaml                single-step sweep configs
  versioned_sweep_*.yaml      E5/E6 versioned sweep configs
  experiments/                E1-E7 experiment configs
docs/
  project_plan.md             full research and experiment roadmap
  runbook.md                  general run instructions
scripts/
  run_toy_study.sh            run all E1-E7 toy templates
  run_model_sharded.sh        one-model multi-GPU layer-sharding launcher
tests/
  unit tests for configs, runners, metrics, planner, reports, and backends
```

## Install

For local development and toy experiments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For Hugging Face experiments:

```bash
pip install -e '.[dev,hf]'
```

## Validate the repository

```bash
ruff check src tests
mypy src tests
pytest
```

The base CI runs the lightweight suite. A separate offline integration job constructs a tiny Llama locally and executes real HF cache paths without downloading model weights.

## Toy experiments

Toy experiments are useful for validating the pipeline without model downloads.

Run one versioned experiment:

```bash
python -m ttt_cache_lab.cli versioned-run \
  --config configs/experiments/e2_version_drift_toy.yaml \
  --version-summary
```

Generate the report:

```bash
python -m ttt_cache_lab.cli version-report \
  --input runs/e2_version_drift/summary.csv \
  --output-dir runs/e2_version_drift/report
```

Generate E3 and E4/E7 analysis tables:

```bash
python -m ttt_cache_lab.cli failure-map \
  --input runs/e3_failure_map/summary.csv \
  --output-dir runs/e3_failure_map/failure_map

python -m ttt_cache_lab.cli pareto \
  --input runs/e4_planner_main/summary.csv \
  --output-dir runs/e4_planner_main/pareto
```

Run all E1-E7 toy templates:

```bash
scripts/run_toy_study.sh
```

## Single-step feasibility commands

The original single-step runner is still available for quick checks.

```bash
python -m ttt_cache_lab.cli run --config configs/feasibility_toy.yaml
python -m ttt_cache_lab.cli summarize --input runs/feasibility-toy/summary.csv
python -m ttt_cache_lab.cli first-table --input runs/feasibility-toy/summary.csv
```

Sweep examples:

```bash
python -m ttt_cache_lab.cli sweep --config configs/sweep_toy_update_norm.yaml
python -m ttt_cache_lab.cli versioned-sweep \
  --config configs/versioned_sweep_e5_delta_qwen_0_5b.yaml
python -m ttt_cache_lab.cli versioned-sweep \
  --config configs/versioned_sweep_e6_context_qwen_1_5b.yaml
```

## Main experiment groups

The full plan is in [`docs/project_plan.md`](docs/project_plan.md). The runnable templates currently cover:

| Group | Purpose | Example config |
|---|---|---|
| E1 | Static adapters and aLoRA/LRAgent/ForkKV-style baselines | `configs/experiments/e1_static_adapter_baseline_qwen_0_5b.yaml` |
| E2 | Adapter-version drift characterization | `configs/experiments/e2_version_drift_qwen_1_5b.yaml` |
| E2 cross-family | Qwen/LLaMA family generality and manual Mistral check | `configs/experiments/e2_version_drift_llama_3_2_1b.yaml`; Mistral 7B remains a manual large-model template |
| E3 | Update-target × version-gap failure map | `configs/experiments/e3_failure_map_qwen_0_5b.yaml` |
| E4 | Versioned planner main experiment | `configs/experiments/e4_planner_main_qwen_0_5b.yaml` |
| E5 | Delta correction and rank/update-norm sweep | `configs/versioned_sweep_e5_delta_qwen_0_5b.yaml` |
| E6 | Exact 4K-32K context and model-scale scaling | `configs/versioned_sweep_e6_context_qwen_1_5b.yaml` |
| E7 | Planner-component ablations and failure boundaries | `configs/experiments/e7_ablation_failure_qwen_0_5b.yaml` |

## Output files

Most experiment runs write:

```text
runs/<experiment>/records.jsonl          raw records
runs/<experiment>/summary.csv            flat CSV records
runs/<experiment>/version_summary.csv    grouped means by version/target/strategy
runs/<experiment>/report/report.md       Markdown report
runs/<experiment>/report/*.svg           metric-vs-version plots
runs/e3_failure_map/failure_map/*        E3 policy table and heatmap
runs/e4_planner_main/pareto/*            E4 quality-cost Pareto table
```

Important columns include:

```text
experiment_id
update_target
cache_strategy
action
adapter_version
cached_version
version_gap
accumulated_update_norm
task_score
logits_kl
top1_agreement
relative_error
hidden_relative_error
update_norm_since_cache
cache_bytes
peak_memory_allocated
adaptation_latency
cache_maintenance_latency
decode_latency
end_to_end_latency
throughput_tokens_per_s
cache_entry_count
recompute_fraction
cache_hit
refresh_count
false_safe
strategy_mode
```

## Backends

| Backend | Config value | Purpose |
|---|---|---|
| Toy | `toy` | CI, dry runs, pipeline validation |
| HuggingFace | `hf` | CPU/CUDA experiments; `model.parallelism: model_shard` partitions decoder layers across visible GPUs |

## Example config fragment

```yaml
model:
  backend: hf
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  device: cuda:0
  torch_dtype: bfloat16
  max_length: 4096
  trust_remote_code: true

adapter:
  update_mode: lora_train
  lora_rank: 8
  lora_alpha: 16.0
  learning_rate: 0.0002
  train_steps_per_version: 1
  freeze_base_model: true

version_steps: [0, 1, 2, 4, 8]
```

## Documentation

- [`docs/project_plan.md`](docs/project_plan.md): complete research plan, experiment dependency graph, deliverables, and decision gates.
- [`docs/runbook.md`](docs/runbook.md): detailed command reference.
- [`docs/design.md`](docs/design.md): cache semantics and strategy design notes.
- [`docs/experiment_plan.md`](docs/experiment_plan.md): earlier feasibility experiment plan.

## Current recommended next step

Run the small-model Hugging Face experiment first, then execute the real E2/E3/E4/E7 configs and E5/E6 sweeps on the selected GPU environment:

```bash
python -m ttt_cache_lab.cli versioned-run \
  --config configs/experiments/e2_version_drift_qwen_0_5b.yaml \
  --version-summary
```

If that passes, run E2 on Qwen2.5-1.5B and generate the report:

```bash
python -m ttt_cache_lab.cli versioned-run \
  --config configs/experiments/e2_version_drift_qwen_1_5b.yaml \
  --version-summary
```

## License

MIT.
