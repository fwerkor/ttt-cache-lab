#!/usr/bin/env bash
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
config=$(python scripts/prepare_modelscope_config.py \
  --config configs/experiments/ascend_smoke_qwen_0_5b.yaml \
  --cache-dir "${MODELSCOPE_CACHE_DIR:-models/modelscope}" \
  --output-dir "${TTT_CACHE_CONFIG_DIR:-runs/modelscope_configs}")
python -m ttt_cache_lab.cli versioned-run --config "$config" --version-summary
python -m ttt_cache_lab.cli version-report \
  --input runs/ascend_smoke_qwen_0_5b/summary.csv \
  --output-dir runs/ascend_smoke_qwen_0_5b/report
