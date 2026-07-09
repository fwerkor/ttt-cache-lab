#!/usr/bin/env bash
set -euo pipefail
configs=(
  configs/experiments/ascend_e2_version_drift_qwen_0_5b.yaml
  configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml
  configs/experiments/ascend_e2_version_drift_qwen_7b.yaml
)
for i in "${!configs[@]}"; do
  cfg=${configs[$i]}
  log="runs/ascend_parallel_${i}.log"
  mkdir -p runs
  (
    export ASCEND_RT_VISIBLE_DEVICES=$i
    echo "==> device=$ASCEND_RT_VISIBLE_DEVICES config=$cfg"
    resolved_config=$(python scripts/prepare_modelscope_config.py \
      --config "$cfg" \
      --cache-dir "${MODELSCOPE_CACHE_DIR:-models/modelscope}" \
      --output-dir "${TTT_CACHE_CONFIG_DIR:-runs/modelscope_configs}")
    python -m ttt_cache_lab.cli versioned-run --config "$resolved_config" --version-summary
  ) >"$log" 2>&1 &
  echo "started $cfg on visible device $i, log=$log"
done
wait
