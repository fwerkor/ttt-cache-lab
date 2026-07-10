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
- `model.revision`: immutable model commit/revision for reproducible runs.
- `model.attention_implementation`: explicit backend such as `eager`; use `eager` when collecting attention metrics.
- `model.max_length`: truncation length for the prompt.
- `data.source`: `synthetic`, `jsonl`, or `huggingface`.
- `data.num_samples`: number of examples to run.
- `data.selection_seed` and `data.sample_offset`: fixed dataset membership independent of the model/update seed.
- `data.evaluation_partition`: explicit `calibration`, `validation`, or `test` provenance.
- `data.choice_fields`, `filters`, and metadata fields: mapping for LongBench v2 and other structured benchmarks.
- `data.context_length`: exact tokenizer-level prompt length after the configured truncation/padding policy.
- `data.answer_length`: synthetic answer construction length or expected task answer length.
- `data.max_generation_tokens`: decode budget, defaulting to 16 and never lower than `answer_length`; it is intentionally independent of the reference answer's tokenizer length.
- `data.truncation_strategy`: `error`, `left`, or `middle` for overlength external data.
- `data.adapter_activation_marker`: optional invocation marker for the aLoRA-style prefix-reuse baseline.
- `updates.targets`: update targets, such as `attention.q`, `attention.k`, `lora.v`, `mlp.late`.
- `updates.update_norm`: random perturbation magnitude or target L2 norm for the controlled update.
- `adapter.norm_control`: `target_l2` or `none`; the latter preserves raw learning-rate updates.
- `cache.strategies`: cache policies to compare.
- `cache.manager_scope`: `condition` by default; use `sample` or `global_workload` only for workload-style capacity experiments.
- `resume`: merge completed checkpoint records from an interrupted run.
- `checkpoint_each_target`: atomically save after each sample × target condition.
- `measurement.warmup_runs` and `measurement.timed_runs`: robust performance repetitions; use dedicated performance configs rather than repeating the whole quality matrix unnecessarily.

List supported target names:

```bash
python -m ttt_cache_lab.cli list-targets
```

## 5. Output files

Each run writes:

- `records.jsonl`: one record per sample × update target × cache strategy;
- `summary.csv`: flat CSV with raw records;
- `run_metadata.json`: config hash, git state, platform, package versions, and visible-device environment;
- `run_failure.json`: structured exception/OOM/unsupported manifest when a run fails;
- optional grouped CSV from the `summarize` command.

`records.jsonl` and `summary.csv` are written through same-directory temporary files and atomically replaced.

The main columns are:

- `update_target`
- `cache_strategy`
- `action`
- `task_score`
- `logits_kl`
- `top1_agreement`
- `relative_error`
- `hidden_relative_error`
- `latency_units`
- `recompute_fraction`
- `cache_hit`
- `refresh_count`
- `false_safe`
- `strategy_mode`, `strategy_available`, `strategy_fallback`
- `baseline_fidelity`, `baseline_source`, `baseline_reference`
- `cache_bytes`, `physical_cache_bytes`, `peak_memory_allocated`
- `accumulated_raw_update_norm`, `accumulated_update_norm`, `update_scale`
- `baseline_task_score`, `full_task_score`, `adaptation_gain_vs_base`
- `adaptation_latency`, `cache_maintenance_latency`, `decode_latency`, `end_to_end_latency`
- `throughput_tokens_per_s`, `cache_entry_count`
- `dataset_sample_id`, `evaluation_partition`, `benchmark_name`, `task_family`
- `model_parameter_count`, actual backend hidden size and layer count
- `timing_runs`, `latency_mean`, `latency_p50`, `latency_p95`, `latency_std`

## 6. Analysis commands

```bash
python -m ttt_cache_lab.cli version-report \
  --input runs/e2_version_drift/summary.csv \
  --output-dir runs/e2_version_drift/report

python -m ttt_cache_lab.cli failure-map \
  --input runs/e3_failure_map/summary.csv \
  --output-dir runs/e3_failure_map/failure_map

python -m ttt_cache_lab.cli pareto \
  --input runs/e4_planner_main/summary.csv \
  --output-dir runs/e4_planner_main/pareto

python -m ttt_cache_lab.cli statistics \
  --input runs/e4_planner_main/summary.csv \
  --output-dir runs/e4_planner_main/statistics \
  --bootstrap-resamples 5000

python -m ttt_cache_lab.cli study-analysis \
  --input runs/e4_planner_main/summary.csv \
  --output-dir runs/e4_planner_main/analysis
```

Run E3 and E4 as one calibrated dependency chain:

```bash
scripts/run_calibrated_planner.sh \
  configs/experiments/e3_failure_map_toy.yaml \
  configs/experiments/e4_planner_main_toy.yaml
```

E4 records expose `planner_source=failure_map` and the exact artifact SHA-256 when the calibrated map is used.

The HF/Ascend backend implements full recomputation, stale/frozen reuse, LoRA-weight-delta K/V correction, native Llama/GPT-2 decoder-layer restart, and aLoRA-style base-prefix reuse with suffix-only recomputation. Delta correction and layer restart do not read the full-reference cache used for evaluation metrics. Unsupported model families fail explicitly instead of substituting a full-reference splice.

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


## 8. E1-E7 experiment templates

Toy templates and real Qwen/Ascend templates exist under `configs/experiments/`.

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

The HF path implements answer-supervised LoRA wrapping for `torch.nn.Linear` projections with global-L2 update-norm control. Non-LoRA controls use normalized direct parameter updates. Layer-specific targets fail when no matching module exists rather than silently broadening the update.

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

The current LoRA wrapper targets `torch.nn.Linear`, covering Qwen/LLaMA/Mistral-style projections. GPT-2-style fused `Conv1D` modules remain available for direct-update and cache-path tests but are not wrapped as trainable LoRA modules.


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

Cross-family and manual large-model templates are available for GPU/NPU runs:

```bash
configs/experiments/e2_version_drift_qwen_1_5b.yaml
configs/experiments/e2_version_drift_llama_3_2_1b.yaml
configs/experiments/e2_version_drift_qwen_7b.yaml        # manual large-model template
configs/experiments/e2_version_drift_mistral_7b_v0_3.yaml # manual large-model template
```


## 11. Ascend support

Ascend runs use `model.backend: ascend_hf` on the 8xAscend 910B server. The Ascend scripts resolve `model.modelscope_model_id` through ModelScope before loading the local snapshot. See [`ascend.md`](ascend.md).


## 12. Real datasets

Local JSONL example:

```yaml
data:
  source: jsonl
  dataset_path: data/repo_qa.jsonl
  task: repo_qa
  prompt_field: prompt
  answer_field: answer
  context_length: 8192
  truncation_strategy: middle
```

Hugging Face / LongBench-style example:

```yaml
data:
  source: huggingface
  dataset_name: THUDM/LongBench
  dataset_config: passage_retrieval_en
  dataset_split: test
  context_field: context
  question_field: input
  answer_field: answers
  prompt_template: "{context}\n\nQuestion: {question}\nAnswer:"
  context_length: 8192
  truncation_strategy: middle
```

## 13. Versioned sweeps

```bash
python -m ttt_cache_lab.cli versioned-sweep \
  --config configs/versioned_sweep_e5_delta_qwen_0_5b.yaml
python -m ttt_cache_lab.cli versioned-sweep \
  --config configs/versioned_sweep_e6_context_qwen_1_5b.yaml
```

The first sweep varies LoRA rank and the measured global update norm. The second runs exact 4K, 8K, 16K, and 32K token contexts.

## 14. One model across multiple devices

Set `model.parallelism: model_shard` and list the device IDs. Decoder layers are divided as evenly as possible while embeddings remain on the first device and the final norm/head on the last device.

```bash
scripts/run_model_sharded.sh \
  configs/experiments/e2_version_drift_qwen_32b_6gpu.yaml

scripts/run_model_sharded.sh \
  configs/experiments/ascend_e2_version_drift_qwen_32b_8npu.yaml
```

The launcher uses Hugging Face Accelerate device maps. This is model layer sharding, distinct from `run_ascend_e2_parallel.sh`, which runs independent experiments on separate cards.


## 15. Resume, merge, and failure recovery

For long runs, set:

```yaml
resume: true
checkpoint_each_target: true
```

Resume is record-level: completed conditions are preserved and de-duplicated, while model updates required to reconstruct in-memory state are replayed.

Merge multiple completed model/context runs for one E6 analysis:

```bash
python -m ttt_cache_lab.cli merge-records \
  --input runs/e6_qwen_1_5b/records.jsonl runs/e6_qwen_7b/records.jsonl \
  --output-dir runs/e6_merged
python -m ttt_cache_lab.cli version-report \
  --input runs/e6_merged/summary.csv \
  --output-dir runs/e6_merged/report
```

When a CLI run raises an exception, inspect `run_failure.json`. OOM and unsupported backend/model paths are retained as structured artifacts instead of only terminal output.

## 16. Baseline fidelity

The local aLoRA-, LRAgent-, and ForkKV-style methods are labeled `paper_reimplementation`. Simplified static controls are labeled `adapted_baseline`. These labels prevent local implementations from being presented as official upstream reproductions. Official-code comparisons remain a separate experimental validation step.


## 13. Paper-scale E1-E8 campaign

The authoritative protocol is [`paper_experiment_protocol.md`](paper_experiment_protocol.md). Do not tune planner thresholds or regenerate the failure map after observing held-out test results.

Inspect the 198-job matrix without running models:

```bash
python -m ttt_cache_lab.cli study-plan --manifest configs/paper/study.yaml
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag qwen_32b --dry-run
```

Recommended stage order:

```bash
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag baseline
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag drift

python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag calibration
scripts/finalize_paper_calibration.sh

python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag validation
scripts/finalize_paper_stage.sh validation

python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag test
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag delta
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag scaling
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag ablation
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag workload
```

Finalize each stage independently so partial hardware failures remain visible:

```bash
scripts/finalize_paper_stage.sh baseline
scripts/finalize_paper_stage.sh drift
scripts/finalize_paper_stage.sh test
scripts/finalize_paper_stage.sh delta
scripts/finalize_paper_stage.sh scaling
scripts/finalize_paper_stage.sh ablation
scripts/finalize_paper_stage.sh workload
```

Distribute the full manifest over eight workers or accelerator groups:

```bash
for shard in $(seq 0 7); do
  scripts/run_paper_shard.sh configs/paper/study.yaml "$shard" 8 &
done
wait
```

Each dependent job declares the finalized failure map as a required artifact. It exits before loading a model when calibration has not been finalized.
