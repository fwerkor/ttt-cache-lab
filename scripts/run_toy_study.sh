#!/usr/bin/env bash
set -euo pipefail
while IFS= read -r config; do
  [[ -z "$config" ]] && continue
  echo "==> $config"
  output_dir="$(python - "$config" <<'PY'
import sys
import yaml
with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    print(yaml.safe_load(handle)['output_dir'])
PY
)"
  python -m ttt_cache_lab.cli versioned-run --config "$config" --version-summary
  python -m ttt_cache_lab.cli version-report --input "$output_dir/summary.csv" --output-dir "$output_dir/report"
  case "$output_dir" in
    *e3_failure_map*)
      python -m ttt_cache_lab.cli failure-map --input "$output_dir/summary.csv" --output-dir "$output_dir/failure_map"
      ;;
    *e4_planner_main*|*e6_scaling*|*e7_ablation_failure*)
      python -m ttt_cache_lab.cli pareto --input "$output_dir/summary.csv" --output-dir "$output_dir/pareto"
      ;;
  esac
done < configs/experiments/study_toy_all.txt
