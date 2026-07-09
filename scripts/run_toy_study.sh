#!/usr/bin/env bash
set -euo pipefail
while IFS= read -r config; do
  [[ -z "$config" ]] && continue
  echo "==> $config"
  python -m ttt_cache_lab.cli versioned-run --config "$config" --version-summary
done < configs/experiments/study_toy_all.txt
