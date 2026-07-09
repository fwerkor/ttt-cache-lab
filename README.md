# TTT Cache Lab

Experimental framework for studying **KV cache consistency and reuse under inference-time parameter evolution** in long-context language models.

The repository is intentionally structured around the paper question rather than around a single model implementation:

> When model parameters change during inference-time training or online adaptation, which KV cache blocks remain valid, which ones can be reused under frozen-evidence semantics, which ones can be approximately corrected, and which ones must be recomputed?

## Current status

This is a scaffold for the first feasibility study. It contains:

- synthetic long-context task generators;
- update-target and cache-semantics abstractions;
- cache strategy interfaces and an adaptive planner skeleton;
- tensor-level metrics for cache/logit drift;
- a lightweight toy backend so CI can run without downloading large models;
- CLI entry points and experiment config templates;
- pytest coverage for the non-model components.

The HuggingFace backend is optional and isolated behind backend interfaces, so CI does not download model weights. vLLM/Ascend integrations are future backends.

## Research questions

1. How do Q/K/V/O/MLP/Norm/LoRA updates invalidate KV cache under different semantics?
2. Is there useful update space beyond Q-only qTTT that can be maintained cheaper than full recomputation?
3. Can parameter-aware planning choose between exact reuse, frozen reuse, stale reuse, partial recomputation, delta correction, and full recomputation?
4. How do the trade-offs scale with context length, update step count, and update norm?

## Repository layout

```text
src/ttt_cache_lab/
  cache/        cache validity semantics, strategies, and planner
  data/         synthetic long-context tasks
  experiments/ experiment runner and result schemas
  metrics/     cache/logit/task metrics
  models/      backend interfaces and toy backend
  updates/     parameter-update target taxonomy and updater interfaces
configs/       YAML experiment templates
docs/          experiment plan and design notes
scripts/       convenience launch scripts
tests/         unit tests for the scaffold
```

## Quick start

Toy run, no model download:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m ttt_cache_lab.cli run --config configs/feasibility_toy.yaml
python -m ttt_cache_lab.cli summarize --input runs/feasibility-toy/summary.csv
```

Tiny HuggingFace smoke run:

```bash
pip install -e '.[dev,hf]'
python -m ttt_cache_lab.cli run --config configs/feasibility_hf_tiny.yaml
python -m ttt_cache_lab.cli summarize --input runs/feasibility-hf-tiny/summary.csv
```

Small real-model feasibility run:

```bash
pip install -e '.[dev,hf]'
python -m ttt_cache_lab.cli run --config configs/feasibility_hf_qwen_0_5b.yaml
python -m ttt_cache_lab.cli summarize --input runs/feasibility-hf-qwen-0-5b/summary.csv
```

See [`docs/runbook.md`](docs/runbook.md) for detailed instructions.

The experiments write JSONL and CSV summaries under `runs/`.

## First feasibility target

The first table to produce is:

| Update target | Full recompute score | Stale cache score | Frozen reuse score | Layer-wise recompute score | Latency vs full |
|---|---:|---:|---:|---:|---:|
| Q | | | | | |
| K | | | | | |
| V | | | | | |
| O | | | | | |
| MLP-late | | | | | |
| LoRA-Q | | | | | |
| LoRA-K | | | | | |
| LoRA-V | | | | | |
| LoRA-MLP-late | | | | | |

A positive result is a region where an update target is more useful than Q-only while cache maintenance is substantially cheaper than full recomputation.

## License

MIT.
