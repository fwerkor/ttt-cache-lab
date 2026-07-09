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
pip install -e '.[dev,hf]'
```

The project does not pin torch-npu in `pyproject.toml` because torch-npu must match the server CANN/PyTorch version.

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

## 5. Use 8 cards as parallel experiment workers first

Use the eight cards as independent experiment workers for configs/seeds/targets. Example:

```bash
scripts/run_ascend_e2_parallel.sh
```

This starts independent Python processes with different `ASCEND_RT_VISIBLE_DEVICES` values.

## 6. Main configs

```text
configs/experiments/ascend_smoke_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_0_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
configs/experiments/ascend_e2_version_drift_qwen_7b.yaml
configs/experiments/ascend_e6_scaling_qwen_7b_16k.yaml
```

## 7. Current limitations

- `ascend_hf` uses torch-npu through HuggingFace Transformers.
- Multi-card model parallelism is optional future work and should only be added if single-card runs cannot cover the target model/context scale.
- The recommended first use of 8x910B is parallel sweeps, one process per visible NPU.
- Real delta KV correction and real layer-wise partial recomputation are still future work.
