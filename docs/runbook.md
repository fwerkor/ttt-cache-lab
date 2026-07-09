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

The HF backend currently implements actual full recomputation and stale/frozen prefix-cache reuse. Layer-wise recomputation and delta correction are planner-level actions but use full recomputation as an upper-bound placeholder in the HF backend. Implementing actual per-layer cache surgery is the next major step.


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
