#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: $0 <validation|test|scaling|ablation|workload> [study-root] [output-root]}"
study_root="${2:-runs/paper/study}"
output_root="${3:-runs/paper/results/$stage}"
case "$stage" in
  validation|test|scaling|ablation|workload) ;;
  *) echo "Unsupported stage: $stage" >&2; exit 2 ;;
esac
mkdir -p "$output_root"

mapfile -t records < <(find "$study_root" -path "*/${stage}_*/seed-*/records.jsonl" -type f | sort)
if [[ ${#records[@]} -eq 0 ]]; then
  echo "No $stage records found under $study_root" >&2
  exit 1
fi

python -m ttt_cache_lab.cli merge-records   --input "${records[@]}"   --output-dir "$output_root/merged"
python -m ttt_cache_lab.cli statistics   --input "$output_root/merged/summary.csv"   --output-dir "$output_root/statistics"   --bootstrap-resamples 5000
python -m ttt_cache_lab.cli study-analysis   --input "$output_root/merged/summary.csv"   --output-dir "$output_root/analysis"
python -m ttt_cache_lab.cli version-report   --input "$output_root/merged/summary.csv"   --output-dir "$output_root/report"
