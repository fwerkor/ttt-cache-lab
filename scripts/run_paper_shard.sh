#!/usr/bin/env bash
set -euo pipefail

manifest="${1:-configs/paper/study.yaml}"
shard_index="${2:?Usage: $0 [manifest] <shard-index> <num-shards>}"
num_shards="${3:?Usage: $0 [manifest] <shard-index> <num-shards>}"

python -m ttt_cache_lab.cli study-run   --manifest "$manifest"   --shard-index "$shard_index"   --num-shards "$num_shards"
