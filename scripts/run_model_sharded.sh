#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment-config.yaml>" >&2
  exit 2
fi

config="$1"
if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2
  exit 2
fi

readarray -t metadata < <(
  python - "$config" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
model = payload.get("model", {})
adapter = payload.get("adapter", {})
print(model.get("parallelism", "single"))
print(model.get("backend", "toy"))
print("1" if model.get("modelscope_model_id") else "0")
print(adapter.get("update_mode", "random"))
PY
)

parallelism="${metadata[0]}"
backend="${metadata[1]}"
has_modelscope_id="${metadata[2]}"
update_mode="${metadata[3]}"

if [[ "$parallelism" != "model_shard" ]]; then
  echo "Config must set model.parallelism: model_shard" >&2
  exit 2
fi

resolved_config="$config"
if [[ "$backend" == "ascend_hf" && "$has_modelscope_id" == "1" ]]; then
  resolved_config="$({
    python scripts/prepare_modelscope_config.py \
      --config "$config" \
      --cache-dir "${MODELSCOPE_CACHE_DIR:-models/modelscope}" \
      --output-dir "${TTT_CACHE_CONFIG_DIR:-runs/modelscope_configs}"
  })"
fi

command_name="versioned-run"
if [[ "$update_mode" == "static_lora" ]]; then
  command_name="static-run"
fi

python -m ttt_cache_lab.cli "$command_name" \
  --config "$resolved_config" \
  --version-summary
