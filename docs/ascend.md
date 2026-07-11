# Ascend 910B runbook

This document describes how to run the experiment framework on Ascend 910B through torch-npu. CUDA/HuggingFace support remains available separately; Ascend runs use `model.backend: ascend_hf`.

## 1. Environment expectation

The `ascend_hf` backend expects a working CANN + PyTorch + torch-npu environment. Verify the server first:

```bash
python - <<'PY'
import torch
import torch_npu
print('torch', torch.__version__)
print('npu available', torch.npu.is_available())
print('device count', torch.npu.device_count())
PY
```

## 2. Install this project

```bash
git pull
pip install -e '.[dev,hf,modelscope]'
```

Ascend launchers download model snapshots through ModelScope and rewrite a temporary config under `${TTT_CACHE_CONFIG_DIR:-runs/modelscope_configs}`. Use `MODELSCOPE_CACHE_DIR` to choose the shared model cache location. The project does not pin torch-npu in `pyproject.toml` because torch-npu must match the server CANN/PyTorch version.

## 3. Smoke test

```bash
ASCEND_RT_VISIBLE_DEVICES=0 scripts/run_ascend_smoke.sh
```

Expected outputs:

```text
runs/ascend_smoke_qwen_0_5b/summary.csv
runs/ascend_smoke_qwen_0_5b/version_summary.csv
runs/ascend_smoke_qwen_0_5b/report/report.md
```

## 4. First real E2 run

```bash
ASCEND_RT_VISIBLE_DEVICES=0 scripts/run_ascend_e2_single.sh   configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
```

## 5. Use 8 cards as independent experiment workers

Use the eight cards as independent experiment workers for configs/seeds/targets. Example:

```bash
scripts/run_ascend_e2_parallel.sh
```

This starts independent Python processes with different `ASCEND_RT_VISIBLE_DEVICES` values. It is useful for parallel seeds/configs and is separate from one-model multi-NPU sharding. Each process resolves its ModelScope model into `${MODELSCOPE_CACHE_DIR:-models/modelscope}` before loading Transformers.

## 6. Shard one model across multiple NPUs

The HF/torch-npu backend supports explicit decoder-layer sharding through an Accelerate device map. The checked-in 7B E2 configuration uses four NPUs: single-NPU LoRA training at 8K context exhausts a 64 GiB 910B, and two-way sharding reaches roughly 57–60 GiB per device in the full matrix. Four-way sharding completed the QV and late-MLP v0–v8 stress conditions with substantial headroom:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 scripts/run_model_sharded.sh \
  configs/experiments/ascend_e2_version_drift_qwen_7b.yaml
```

The 32B template uses all eight NPUs:

```bash
scripts/run_model_sharded.sh \
  configs/experiments/ascend_e2_version_drift_qwen_32b_8npu.yaml
```

The corresponding config sets:

```yaml
model:
  backend: ascend_hf
  device: npu
  parallelism: model_shard
  device_ids: [0, 1, 2, 3, 4, 5, 6, 7]
```

Embeddings are placed on the first NPU, decoder layers are balanced across all listed NPUs, and the final norm/head are placed on the last NPU unless tied embeddings require the head on the first device.

## 7. Main configs

```text
configs/experiments/ascend_smoke_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
configs/experiments/ascend_e2_version_drift_llama_3_2_1b.yaml
configs/experiments/ascend_e2_version_drift_llama_3_2_3b.yaml
configs/experiments/ascend_e2_version_drift_gemma_3_4b.yaml
configs/experiments/ascend_e2_version_drift_qwen1_5_moe_a2_7b.yaml
configs/experiments/ascend_e2_version_drift_qwen_7b.yaml          # 4-NPU model sharding
configs/experiments/ascend_e2_version_drift_mistral_7b_v0_1.yaml  # sliding-window screening
configs/experiments/ascend_e2_version_drift_mistral_7b_v0_3.yaml  # existing manual large-model template
configs/experiments/ascend_e2_version_drift_qwen_32b_8npu.yaml     # 8-NPU model sharding
configs/experiments/ascend_e5_delta_correction_qwen_0_5b.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_4k.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_8k.yaml
configs/experiments/ascend_e6_scaling_qwen_7b_16k.yaml
configs/experiments/ascend_e6_scaling_qwen_1_5b_32k.yaml
configs/experiments/ascend_e6_scaling_qwen_7b_32k.yaml
```

Architecture-screening configs remain platform-neutral under `configs/paper/architecture/`. Run one on Ascend with ModelScope resolution and an isolated visible card:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 scripts/run_ascend_architecture_single.sh \
  configs/paper/architecture/a1_gemma_3_4b_multi_hop_tracing.yaml
```

They are intentionally excluded from default parallel launchers.

## 8. Validation notes

- `ascend_hf` uses torch-npu through Hugging Face Transformers; torch-npu must match the server PyTorch/CANN versions.
- Ascend scripts resolve `model.modelscope_model_id` to a local snapshot before model loading.
- Delta correction uses cached LoRA projection inputs and cached-version A/B snapshots; it does not read the full-reference cache.
- Native decoder-layer restart is implemented for Llama/Qwen/Mistral-like, nested Gemma 3 text, and GPT-2-like model layouts. Unsupported layouts fail explicitly.
- Qwen2-MoE targets distinguish router, shared-expert, and fused routed-expert parameters; the fused routed-expert target currently uses controlled direct parameter perturbation because it is not an `nn.Linear` LoRA injection point.
- aLoRA-style prefix reuse disables LoRA before the configured invocation marker, caches that base prefix, and recomputes only the post-marker suffix under the active adapter.
- Multi-NPU sharding logic is covered by unit tests, but throughput, HCCL behavior, and memory headroom must still be validated on the target 8×910B machine.
