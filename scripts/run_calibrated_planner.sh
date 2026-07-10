#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <e3-config.yaml> <e4-config.yaml>" >&2
  exit 2
fi

e3_config="$1"
e4_config="$2"

read_config_value() {
  python - "$1" "$2" <<'PY'
import sys
import yaml

path, dotted = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for part in dotted.split("."):
    value = value[part]
print(value)
PY
}

e3_output="$(read_config_value "$e3_config" output_dir)"
e4_output="$(read_config_value "$e4_config" output_dir)"
configured_map="$(read_config_value "$e4_config" cache.failure_map_path)"
generated_map="$e3_output/failure_map/failure_map.csv"

if [[ "$configured_map" != "$generated_map" ]]; then
  echo "E4 failure_map_path must equal the E3 artifact path" >&2
  echo "configured: $configured_map" >&2
  echo "generated:  $generated_map" >&2
  exit 2
fi

python -m ttt_cache_lab.cli versioned-run --config "$e3_config" --version-summary
python -m ttt_cache_lab.cli failure-map \
  --input "$e3_output/summary.csv" \
  --output-dir "$e3_output/failure_map"
python -m ttt_cache_lab.cli version-report \
  --input "$e3_output/summary.csv" \
  --output-dir "$e3_output/report"

python -m ttt_cache_lab.cli versioned-run --config "$e4_config" --version-summary
python -m ttt_cache_lab.cli version-report \
  --input "$e4_output/summary.csv" \
  --output-dir "$e4_output/report"
python -m ttt_cache_lab.cli pareto \
  --input "$e4_output/summary.csv" \
  --output-dir "$e4_output/pareto"
