#!/usr/bin/env bash
set -euo pipefail
config=${1:-configs/experiments/ascend_e2_version_drift_qwen_1_5b.yaml}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
python -m ttt_cache_lab.cli versioned-run --config "$config" --version-summary
out_dir=$(python -c "import yaml; print(yaml.safe_load(open('$config', encoding='utf-8'))['output_dir'])")
python -m ttt_cache_lab.cli version-report   --input "$out_dir/summary.csv"   --output-dir "$out_dir/report"
