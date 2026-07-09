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

- synthetic long-context tasks: passkey, key-value retrieval, multi-needle retrieval, and variable tracking;
- update-target taxonomy: Q/K/V/O/QV/attention/MLP/Norm/output head and LoRA variants with early/middle/late layer positions;
- cache validity semantics and strategy abstractions;
- adaptive planner with update-target, version-gap, and accumulated-update-norm inputs;
- single-step feasibility runner;
- multi-step versioned runner for adapter-version drift experiments;
- toy backend for CI and dry runs;
- HuggingFace backend for real `past_key_values` experiments;
- Ascend HuggingFace backend using torch-npu (`model.backend: ascend_hf`);
- ModelScope download preparation for Ascend runs;
- simple LoRA wrapper and online LoRA update path for `torch.nn.Linear` projections;
- HF/Ascend cache-surgery paths for layer-wise cache splice and LoRA-weight-delta K/V correction;
- result summaries, first feasibility tables, Markdown reports, SVG trend plots, failure-map tables, and Pareto tables;
- E1-E7 experiment templates from the project plan, including static baselines, threshold refresh, oracle planner, E5 real-model smoke configs, and E6 scaling configs;
- CI with linting, type checking, and tests.

Still limited:

- HF/Ascend delta correction now uses cached LoRA projection inputs plus A/B weight snapshots to patch K/V without reading the full-reference cache;
- HF/Ascend partial recompute records `strategy_mode`; generic Transformers use `fallback_past_key_values_layer_splice`, while model-specific native mid-layer restart can be plugged into `_native_partial_recompute_prefix_cache`;
- optional distributed backend for larger models, if single-card torch-npu is insufficient;
- full reproduction of aLoRA/LRAgent/ForkKV-style baselines;
- final paper-quality plots and real 910B experiment results.

## Repository layout

```text
src/ttt_cache_lab/
  cache/         cache validity semantics, strategies, and planner
  data/          synthetic long-context tasks
  experiments/  single-step runner, versioned runner, sweep, summaries, reports
  metrics/      cache/logit/task metrics
  models/       toy, HuggingFace, Ascend torch-npu backends
  updates/      parameter-update target taxonomy
configs/
  feasibility_*.yaml          early single-step configs
  sweep_*.yaml                sweep configs
  experiments/                E1-E7 and Ascend experiment configs
docs/
  project_plan.md             full research and experiment roadmap
  ascend.md                   Ascend 910B runbook
  runbook.md                  general run instructions
scripts/
  run_toy_study.sh            run all E1-E7 toy templates
  run_ascend_*.sh             Ascend smoke/single/parallel launchers; use ModelScope by default
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

The CI runs the same checks without downloading model weights.

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

This launches independent processes with different `ASCEND_RT_VISIBLE_DEVICES` values. By default it runs Qwen2.5-0.5B, Qwen2.5-1.5B, and Llama-3.2-1B. Qwen2.5-7B and Mistral-7B templates remain available for manual runs only until a distributed backend or shorter-context validation path is added. Each Ascend launcher resolves the configured ModelScope model into `${MODELSCOPE_CACHE_DIR:-models/modelscope}` before running.

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

Sweep example:

```bash
python -m ttt_cache_lab.cli sweep --config configs/sweep_toy_update_norm.yaml
python -m ttt_cache_lab.cli first-table --input runs/sweep-toy-update-norm/merged_records.csv
```

## Main experiment groups

The full plan is in [`docs/project_plan.md`](docs/project_plan.md). The runnable templates currently cover:

| Group | Purpose | Example config |
|---|---|---|
| E1 | Static-adapter baseline alignment | `configs/experiments/e1_static_adapter_baseline_toy.yaml` |
| E2 | Adapter-version drift characterization | `configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml` |
| E2 cross-family | Qwen/LLaMA family generality and manual Mistral check | `configs/experiments/ascend_e2_version_drift_llama_3_2_1b.yaml`; Mistral 7B remains a manual large-model template |
| E3 | Update-target × version-gap failure map | `configs/experiments/e3_failure_map_toy.yaml` |
| E4 | Versioned planner main experiment | `configs/experiments/e4_planner_main_toy.yaml` |
| E5 | Delta correction / base+delta cache experiment | `configs/experiments/e5_delta_correction_toy.yaml` |
| E6 | Context-length and model-scale scaling | `configs/experiments/ascend_e6_scaling_qwen_7b_16k.yaml` |
| E7 | Ablations and failure boundaries | `configs/experiments/e7_ablation_failure_toy.yaml` |

Ascend-specific configs:

```text
configs/experiments/ascend_smoke_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
configs/experiments/ascend_e2_version_drift_llama_3_2_1b.yaml
configs/experiments/ascend_e2_version_drift_qwen_7b.yaml         # manual large-model template
configs/experiments/ascend_e2_version_drift_mistral_7b_v0_3.yaml  # manual large-model template
configs/experiments/ascend_e5_delta_correction_qwen_0_5b.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_4k.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_8k.yaml
configs/experiments/ascend_e6_scaling_qwen_7b_16k.yaml            # manual scaling template
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
latency_units
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
| HuggingFace | `hf` | CUDA/CPU fallback and optional validation |
| Ascend HuggingFace | `ascend_hf` | Real experiment backend on Ascend 910B through torch-npu; launcher scripts prepare local model snapshots through ModelScope |

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

Run the Ascend smoke test on the 8×910B server:

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
