# Runbook

## 1. Lightweight local run

This path requires only the base dependencies and runs without downloading model weights.

```bash
git clone https://github.com/fwerkor/ttt-cache-lab.git
cd ttt-cache-lab
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m ttt_cache_lab.cli run --config configs/feasibility_toy.yaml
python -m ttt_cache_lab.cli summarize \
  --input runs/feasibility-toy/summary.csv \
  --output runs/feasibility-toy/grouped.csv
python -m ttt_cache_lab.cli first-table \
  --input runs/feasibility-toy/summary.csv
```

Equivalent shortcut:

```bash
make test
make run-toy
```

## 2. Tiny HuggingFace smoke run

This uses a tiny public GPT-2 model. It is meant to check that real `past_key_values`
can be captured and reused. It is not a meaningful paper experiment.

```bash
pip install -e '.[dev,hf]'
python -m ttt_cache_lab.cli run --config configs/feasibility_hf_tiny.yaml
python -m ttt_cache_lab.cli summarize --input runs/feasibility-hf-tiny/summary.csv
```

## 3. Small real-model feasibility run

This is the first useful run for the research question. Use GPU if available.

```bash
pip install -e '.[dev,hf]'
python -m ttt_cache_lab.cli run --config configs/feasibility_hf_qwen_0_5b.yaml
python -m ttt_cache_lab.cli summarize \
  --input runs/feasibility-hf-qwen-0-5b/summary.csv \
  --output runs/feasibility-hf-qwen-0-5b/grouped.csv
```

If the machine has multiple GPUs, choose one explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m ttt_cache_lab.cli run \
  --config configs/feasibility_hf_qwen_0_5b.yaml
```

## 4. Useful knobs

Edit the YAML config rather than the code.

- `model.model_name_or_path`: local path or HuggingFace model ID.
- `model.max_length`: truncation length for the prompt.
- `data.num_samples`: number of synthetic examples.
- `data.context_length`: approximate synthetic context length.
- `updates.targets`: update targets, such as `attention.q`, `attention.k`, `lora.v`, `mlp.late`.
- `updates.update_norm`: random perturbation magnitude for the controlled update.
- `cache.strategies`: cache policies to compare.

List supported target names:

```bash
python -m ttt_cache_lab.cli list-targets
```

## 5. Output files

Each run writes:

- `records.jsonl`: one record per sample × update target × cache strategy;
- `summary.csv`: flat CSV with raw records;
- optional grouped CSV from the `summarize` command.

The main columns are:

- `update_target`
- `cache_strategy`
- `action`
- `task_score`
- `logits_kl`
- `top1_agreement`
- `relative_error`
- `latency_units`

## 6. Current limitations

The HF backend currently implements actual full recomputation and stale/frozen prefix-cache reuse. Layer-wise recomputation and delta correction are planner-level actions but use full recomputation as an upper-bound placeholder in the HF backend. Their latency is charged as full recomputation until real per-layer cache surgery exists. Implementing actual per-layer cache surgery is the next major step.


## 7. Sweep run

Use this when you want the first feasibility table over several update magnitudes
and context lengths.

```bash
python -m ttt_cache_lab.cli sweep --config configs/sweep_toy_update_norm.yaml
python -m ttt_cache_lab.cli summarize --input runs/sweep-toy-update-norm/merged_records.csv
python -m ttt_cache_lab.cli first-table --input runs/sweep-toy-update-norm/merged_records.csv
```

The sweep file has a `base` experiment plus `axes`. Each `axes[*].path` is a dotted
path into the experiment config, for example `updates.update_norm` or
`data.context_length`.

## Project direction

The full project plan and experiment blueprint are in [`project_plan.md`](project_plan.md).


## 8. Planned E1-E7 experiment templates

Toy templates exist for all planned experiment groups under `configs/experiments/`.

```bash
scripts/run_toy_study.sh
```

Run a single template:

```bash
python -m ttt_cache_lab.cli versioned-run   --config configs/experiments/e2_version_drift_toy.yaml   --version-summary
```

The first real HF LoRA template is:

```bash
pip install -e '.[dev,hf]'
CUDA_VISIBLE_DEVICES=0 python -m ttt_cache_lab.cli versioned-run   --config configs/experiments/e2_version_drift_qwen_0_5b.yaml   --version-summary
```

Current implementation note: the HF path implements real LoRA wrapping for `torch.nn.Linear` projections and real gradient steps. GPT-2 style `Conv1D` projections still fall back to perturbation-style experiments unless a Linear target is available.

## 9. Implemented experiment coverage

The current code implements runnable templates for all planned experiment groups:

| Group | Toy config | Main output |
|---|---|---|
| E1 static-adapter baseline | `configs/experiments/e1_static_adapter_baseline_toy.yaml` | `runs/e1_static_adapter_baseline/version_summary.csv` |
| E2 version drift | `configs/experiments/e2_version_drift_toy.yaml` | `runs/e2_version_drift/version_summary.csv` |
| E3 failure map | `configs/experiments/e3_failure_map_toy.yaml` | `runs/e3_failure_map/version_summary.csv` |
| E4 planner main | `configs/experiments/e4_planner_main_toy.yaml` | `runs/e4_planner_main/version_summary.csv` |
| E5 delta correction | `configs/experiments/e5_delta_correction_toy.yaml` | `runs/e5_delta_correction/version_summary.csv` |
| E6 scaling | `configs/experiments/e6_scaling_toy.yaml` | `runs/e6_scaling/version_summary.csv` |
| E7 ablation/failure | `configs/experiments/e7_ablation_failure_toy.yaml` | `runs/e7_ablation_failure/version_summary.csv` |

Run all toy templates:

```bash
scripts/run_toy_study.sh
```

Run the first real LoRA drift template:

```bash
pip install -e '.[dev,hf]'
CUDA_VISIBLE_DEVICES=0 python -m ttt_cache_lab.cli versioned-run \
  --config configs/experiments/e2_version_drift_qwen_0_5b.yaml \
  --version-summary
```

The HF LoRA path currently supports `torch.nn.Linear` projections, which covers Qwen/LLaMA-style projection layers. GPT-2-style fused `Conv1D` modules are not wrapped by the current LoRA implementation.


## 10. Generate experiment reports

After any `versioned-run`, generate Markdown and SVG plots:

```bash
python -m ttt_cache_lab.cli version-report   --input runs/e2_version_drift/summary.csv   --output-dir runs/e2_version_drift/report
```

The report directory contains:

- `report.md`
- `logits_kl_by_version.svg`
- `relative_error_by_version.svg`
- `task_score_by_version.svg`
- `latency_units_by_version.svg`

Larger Qwen and cross-family templates are available for GPU runs:

```bash
configs/experiments/e2_version_drift_qwen_1_5b.yaml
configs/experiments/e2_version_drift_qwen_7b.yaml
configs/experiments/e2_version_drift_llama_3_1_8b.yaml
configs/experiments/e2_version_drift_mistral_7b_v0_3.yaml
```


## 11. Ascend support

Ascend runs use `model.backend: ascend_hf` on the 8xAscend 910B server. The Ascend scripts resolve `model.modelscope_model_id` through ModelScope before loading the local snapshot. See [`ascend.md`](ascend.md).
