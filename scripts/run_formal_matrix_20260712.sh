#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${TTT_CACHE_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
RUN_ROOT="$ROOT/runs/formal_20260712"
SUPERVISOR_ROOT="$RUN_ROOT/logs_v2/supervisors"
ENV_ROOT=${TTT_CACHE_ENV_ROOT:-/mnt/caoyuhang/cyh/envs/torch-npu-2.7.1-cann-8.5.1-py312}
ALL_USABLE_DEVICES=${TTT_CACHE_NPU_DEVICES:-0,1,3,4,5,6,7}

cd "$ROOT"
mkdir -p "$SUPERVISOR_ROOT"
export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

PIDS=()
LAST_PID=""

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

queue_is_complete() {
  local queue=$1
  local marker="$RUN_ROOT/status_v2/$queue.done.json"
  [[ -f "$marker" ]] || return 1
  python - "$marker" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if int(payload.get("failed", 1)) == 0 else 1)
PY
}

run_queue() {
  local queue=$1
  local devices=$2
  local log_path="$SUPERVISOR_ROOT/$queue.log"
  if queue_is_complete "$queue"; then
    log "queue $queue already complete; skipping"
    LAST_PID=""
    return 0
  fi
  log "starting queue=$queue devices=$devices log=$log_path"
  (
    export ASCEND_RT_VISIBLE_DEVICES="$devices"
    python scripts/formal_matrix_20260712.py --queue "$queue"
  ) >>"$log_path" 2>&1 &
  LAST_PID=$!
  PIDS+=("$LAST_PID")
}

wait_for() {
  local label=$1
  local pid=$2
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if wait "$pid"; then
    log "$label process exited normally"
  else
    log "$label process exited abnormally; continuing so completed checkpoints are preserved"
  fi
}

log "formal matrix started"
log "commit=$(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"
log "excluded physical NPU 2"

# Phase 1: the three main model-size bands run concurrently.
run_queue small0 0
small_pid=$LAST_PID
run_queue seven13 1,3
seven_pid=$LAST_PID
run_queue fourteen4567 4,5,6,7
fourteen_pid=$LAST_PID

wait_for seven13 "$seven_pid"
wait_for fourteen4567 "$fourteen_pid"

# Phase 2: keep the two released resource pools busy while 1.5B continues.
run_queue arch13 1,3
arch_pid=$LAST_PID
run_queue sevenlong4567 4,5,6,7
sevenlong_pid=$LAST_PID
wait_for arch13 "$arch_pid"
wait_for sevenlong4567 "$sevenlong_pid"
wait_for small0 "$small_pid"

# Phase 3: long 14B and all 32B work use every healthy card.
run_queue longall "$ALL_USABLE_DEVICES"
longall_pid=$LAST_PID
wait_for longall "$longall_pid"

# A minimal 32B/16K run checks the final seven-card allocation. Failure does
# not suppress the formal queue because its 8K and delta subsets may still run.
preflight_cfg="$RUN_ROOT/preflight_configs/smoke_32b_7npu_16k_gc.yaml"
preflight_out="$RUN_ROOT/preflight/smoke_32b_7npu_16k_gc"
if [[ ! -f "$preflight_out/version_summary.csv" ]]; then
  python - "$preflight_cfg" "$preflight_out" <<'PY'
from pathlib import Path
import sys
import yaml
cfg_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
payload = yaml.safe_load(Path("configs/paper/drift/e2_qwen_32b_controlled.yaml").read_text())
payload["name"] = "smoke_32b_7npu_16k_gc"
payload["output_dir"] = str(out_dir)
payload["resume"] = False
payload["checkpoint_each_target"] = False
payload["model"]["model_name_or_path"] = "/mnt/caoyuhang/cyh/models/modelscope/models/Qwen--Qwen2.5-32B-Instruct/snapshots/master"
payload["model"]["parallelism"] = "model_shard"
payload["model"]["device"] = "auto"
payload["model"]["device_ids"] = []
payload["data"]["num_samples"] = 1
payload["data"]["sample_offset"] = 0
payload["data"]["context_length"] = 16384
payload["task_viability"]["enabled"] = False
payload["updates"]["targets"] = ["lora.q"]
payload["cache"]["strategies"] = ["full_recompute", "stale_reuse"]
payload["version_steps"] = [0, 1]
payload["measurement"]["warmup_runs"] = 0
payload["measurement"]["timed_runs"] = 1
cfg_path.parent.mkdir(parents=True, exist_ok=True)
cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
  log "starting 32B seven-card preflight"
  ASCEND_RT_VISIBLE_DEVICES="$ALL_USABLE_DEVICES" \
    python -m ttt_cache_lab.cli versioned-run --config "$preflight_cfg" --version-summary \
    >>"$SUPERVISOR_ROOT/preflight_32b_7npu_16k.log" 2>&1 || \
    log "32B seven-card preflight failed; formal queue will still attempt viable subsets"
fi

run_queue thirtysix "$ALL_USABLE_DEVICES"
thirty_pid=$LAST_PID
wait_for thirtysix "$thirty_pid"

python scripts/finalize_formal_20260712.py \
  >>"$SUPERVISOR_ROOT/finalize.log" 2>&1 || log "finalization reported errors"

log "formal matrix finished"
