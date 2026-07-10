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

The framework supports both HuggingFace/CUDA-style runs and Ascend 910B runs through torch-npu. The current configs include an Ascend path because that hardware is readily available for the planned experiments.

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
- E1-E7 toy, Hugging Face, and Ascend templates; E5 rank/update-norm sweeps; E6 exact 4K/8K/16K/32K context sweeps;
- single-model layer sharding across multiple CUDA GPUs or Ascend NPUs, including 6×GPU and 8×NPU Qwen2.5-32B templates;
- strategy-aware failure maps, configurable safety thresholds, Markdown reports, SVG trends, and quality-cost Pareto figures;
- CI for linting, strict type checking, unit tests, and offline tiny-Llama integration tests that execute real LoRA, KV delta correction, and native layer restart paths.

Remaining validation work is experimental rather than placeholder implementation: the large-model paths still need to be run on the target 6×A6000 and 8×Ascend 910B machines, and the adapted related-work baselines should be compared against official upstream implementations where licensing and environments permit.

## Repository layout

```text
src/ttt_cache_lab/
  cache/         cache validity semantics, strategies, and planner
  data/          synthetic, JSONL, and Hugging Face task loaders
  experiments/  single-step runner, versioned runner, sweep, summaries, reports
  metrics/      cache/logit/task metrics
  models/       toy, HuggingFace, Ascend torch-npu backends
  updates/      update targets and executable random/LoRA updaters
configs/
  feasibility_*.yaml          early single-step configs
  sweep_*.yaml                single-step sweep configs
  versioned_sweep_*.yaml      E5/E6 versioned sweep configs
  experiments/                E1-E7 and Ascend experiment configs
docs/
  project_plan.md             full research and experiment roadmap
  ascend.md                   Ascend 910B runbook
  runbook.md                  general run instructions
scripts/
  run_toy_study.sh            run all E1-E7 toy templates
  run_ascend_*.sh             Ascend smoke/single/parallel-worker launchers
  run_model_sharded.sh        one-model multi-GPU/NPU layer-sharding launcher
  prepare_modelscope_config.py  resolve ModelScope weights and emit local-path configs
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

For HuggingFace / Ascend experiments:

```bash
pip install -e '.[dev,hf,modelscope]'
```

`modelscope` is used by the Ascend launcher scripts to download weights into a local snapshot directory before Transformers loads the model. `torch-npu` is not pinned in `pyproject.toml` because it must match the server's CANN and PyTorch versions. Install torch-npu according to the Ascend server environment.

## Validate the repository

```bash
ruff check src tests
mypy src tests
pytest
```

The base CI runs the lightweight suite. A separate offline integration job constructs a tiny Llama locally and executes real HF cache paths without downloading model weights.

## Ascend 910B quick start

Check the NPU environment first:

```bash
python - <<'PY'
import torch
import torch_npu
print('torch', torch.__version__)
print('npu available', torch.npu.is_available())
print('device count', torch.npu.device_count())
PY
```

Run the Ascend smoke test:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 scripts/run_ascend_smoke.sh
```

Expected outputs:

```text
runs/ascend_smoke_qwen_0_5b/summary.csv
runs/ascend_smoke_qwen_0_5b/version_summary.csv
runs/ascend_smoke_qwen_0_5b/report/report.md
runs/ascend_smoke_qwen_0_5b/report/*.svg
```

Run the first real E2 version-drift experiment on Qwen2.5-1.5B:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 scripts/run_ascend_e2_single.sh \
  configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
```

Run a manual large-model template after checking memory headroom:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 scripts/run_ascend_e2_single.sh \
  configs/experiments/ascend_e2_version_drift_qwen_7b.yaml
```

Use the 8-card machine as parallel experiment workers first:

```bash
scripts/run_ascend_e2_parallel.sh
```

This launches independent processes with different `ASCEND_RT_VISIBLE_DEVICES` values for experiment-level parallelism. For one model distributed across several devices, use `scripts/run_model_sharded.sh`; each Ascend launcher resolves the configured ModelScope model into `${MODELSCOPE_CACHE_DIR:-models/modelscope}` before running.

Run Qwen2.5-32B across all eight NPUs:

```bash
scripts/run_model_sharded.sh \
  configs/experiments/ascend_e2_version_drift_qwen_32b_8npu.yaml
```

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
| E2 | Adapter-version drift characterization | `configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml` |
| E2 cross-family | Qwen/LLaMA family generality and manual Mistral check | `configs/experiments/ascend_e2_version_drift_llama_3_2_1b.yaml`; Mistral 7B remains a manual large-model template |
| E3 | Update-target × version-gap failure map | `configs/experiments/e3_failure_map_qwen_0_5b.yaml` |
| E4 | Versioned planner main experiment | `configs/experiments/e4_planner_main_qwen_0_5b.yaml` |
| E5 | Delta correction and rank/update-norm sweep | `configs/versioned_sweep_e5_delta_qwen_0_5b.yaml` |
| E6 | Exact 4K-32K context and model-scale scaling | `configs/versioned_sweep_e6_context_qwen_1_5b.yaml` |
| E7 | Planner-component ablations and failure boundaries | `configs/experiments/e7_ablation_failure_qwen_0_5b.yaml` |

Ascend-specific configs:

```text
configs/experiments/ascend_smoke_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
configs/experiments/ascend_e2_version_drift_llama_3_2_1b.yaml
configs/experiments/ascend_e2_version_drift_qwen_7b.yaml         # manual large-model template
configs/experiments/ascend_e2_version_drift_mistral_7b_v0_3.yaml  # manual large-model template
configs/experiments/ascend_e2_version_drift_qwen_32b_8npu.yaml     # 8-NPU model sharding
configs/experiments/ascend_e5_delta_correction_qwen_0_5b.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_4k.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_8k.yaml
configs/experiments/ascend_e6_scaling_qwen_7b_16k.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_32k.yaml
configs/experiments/ascend_e6_scaling_qwen_7b_32k.yaml
```

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
| Ascend HuggingFace | `ascend_hf` | torch-npu experiments; ModelScope launchers prepare local snapshots and `model_shard` partitions layers across NPUs |

## Example config fragment

```yaml
model:
  backend: ascend_hf
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  device: npu:0
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
- [`docs/ascend.md`](docs/ascend.md): Ascend 910B environment and runbook.
- [`docs/runbook.md`](docs/runbook.md): detailed command reference.
- [`docs/design.md`](docs/design.md): cache semantics and strategy design notes.
- [`docs/experiment_plan.md`](docs/experiment_plan.md): earlier feasibility experiment plan.

## Current recommended next step

Run the Ascend smoke test on the 8×910B server, then execute the real E2/E3/E4/E7 configs and E5/E6 sweeps:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 scripts/run_ascend_smoke.sh
```

If that passes, run E2 on Qwen2.5-1.5B and generate the report:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 scripts/run_ascend_e2_single.sh \
  configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
```

## License

MIT.
