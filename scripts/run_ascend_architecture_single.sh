#!/usr/bin/env bash
set -euo pipefail

config=${1:?usage: run_ascend_architecture_single.sh <architecture-config.yaml>}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

resolved_config=$(python scripts/prepare_modelscope_config.py \
  --config "$config" \
  --cache-dir "${MODELSCOPE_CACHE_DIR:-models/modelscope}" \
  --output-dir "${TTT_CACHE_CONFIG_DIR:-runs/modelscope_configs}")

ascend_config=$(python - "$resolved_config" "${TTT_CACHE_CONFIG_DIR:-runs/modelscope_configs}" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
model = payload["model"]
model["backend"] = "ascend_hf"
model["device"] = "npu:0"
model["torch_dtype"] = "bfloat16"
output = output_dir / f"{source.stem}.ascend.yaml"
output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print(output)
PY
)

python -m ttt_cache_lab.cli versioned-run \
  --config "$ascend_config" \
  --version-summary
