#!/usr/bin/env bash
set -euo pipefail

study_root="${1:-runs/paper/study}"
output_root="${2:-runs/paper/calibration/final}"
mkdir -p "$output_root"

mapfile -t records < <(find "$study_root" -path '*/calibration_*/seed-*/records.jsonl' -type f | sort)
if [[ ${#records[@]} -eq 0 ]]; then
  echo "No calibration records found under $study_root" >&2
  exit 1
fi

python -m ttt_cache_lab.cli merge-records   --input "${records[@]}"   --output-dir "$output_root/merged"
python -m ttt_cache_lab.cli failure-map   --input "$output_root/merged/summary.csv"   --output-dir "$output_root/failure_map"
python -m ttt_cache_lab.cli statistics   --input "$output_root/merged/summary.csv"   --output-dir "$output_root/statistics"   --bootstrap-resamples 5000
python -m ttt_cache_lab.cli study-analysis   --input "$output_root/merged/summary.csv"   --output-dir "$output_root/analysis"

sha256sum "$output_root/failure_map/failure_map.csv" > "$output_root/failure_map.sha256"
echo "Finalized calibration artifact: $output_root/failure_map/failure_map.csv"
