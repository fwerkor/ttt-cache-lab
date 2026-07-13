#!/usr/bin/env python3
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "formal_20260712"
CONFIG_ROOT = RUN_ROOT / "configs_v2"
LOG_ROOT = RUN_ROOT / "logs_v2"
STATUS_ROOT = RUN_ROOT / "status_v2"
SEEDS = (7, 17, 29)

MODEL_PATHS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "/mnt/caoyuhang/cyh/models/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master",
    "Qwen/Qwen2.5-7B-Instruct": "/mnt/caoyuhang/cyh/models/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    "Qwen/Qwen2.5-14B-Instruct": "/mnt/caoyuhang/cyh/models/modelscope/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master",
    "Qwen/Qwen2.5-32B-Instruct": "/mnt/caoyuhang/cyh/models/modelscope/models/Qwen--Qwen2.5-32B-Instruct/snapshots/master",
    "google/gemma-3-4b-it": "/mnt/caoyuhang/cyh/models/modelscope/models/LLM-Research--gemma-3-4b-it/snapshots/master",
    "meta-llama/Llama-3.2-3B-Instruct": "/mnt/caoyuhang/cyh/models/modelscope/models/LLM-Research--Llama-3.2-3B-Instruct/snapshots/master",
    "mistralai/Mistral-7B-Instruct-v0.1": "/mnt/caoyuhang/cyh/models/modelscope/models/AI-ModelScope--Mistral-7B-Instruct-v0.1/snapshots/master",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": "/mnt/caoyuhang/cyh/models/modelscope/models/Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/master",
}
LONG_BENCH_V2 = Path("/mnt/caoyuhang/cyh/datasets/longbench_v2/data.jsonl")


@dataclasses.dataclass(frozen=True)
class Job:
    name: str
    config: str
    runner: str = "versioned"
    seeds: tuple[int, ...] = SEEDS
    mode: str = "normal"


def glob_jobs(prefix: str, pattern: str, *, runner: str = "versioned", mode: str = "normal") -> list[Job]:
    return [Job(f"{prefix}_{p.stem}", str(p.relative_to(ROOT)), runner, SEEDS, mode) for p in sorted(ROOT.glob(pattern))]


def build_queues() -> dict[str, list[Job]]:
    small0 = [
        Job("w1_qwen_1_5b", "configs/paper/discovery/w1_qwen_1_5b_multi_hop_window_sweep.yaml", "sweep"),
        Job("w2_qwen_1_5b", "configs/paper/discovery/w2_qwen_1_5b_propagation.yaml"),
        Job("w3_qwen_1_5b", "configs/paper/discovery/w3_qwen_1_5b_boundary_predictor.yaml"),
        Job("w4_qwen_1_5b", "configs/paper/discovery/w4_qwen_1_5b_blockwise_oracle.yaml", "blockwise"),
    ] + glob_jobs("e3", "configs/paper/calibration/e3_qwen_1_5b_*.yaml")

    seven13 = [
        Job("e1_qwen_7b_longbench_v2", "configs/paper/baseline/e1_qwen_7b_longbench_v2.yaml", "static"),
        Job("e2_qwen_7b_controlled", "configs/paper/drift/e2_qwen_7b_controlled.yaml"),
        Job("e2_qwen_7b_longbench_v2", "configs/paper/drift/e2_qwen_7b_longbench_v2.yaml"),
        Job("w1_qwen_7b", "configs/paper/discovery/w1_qwen_7b_multi_hop_window_sweep.yaml", "sweep"),
        Job("w2_qwen_7b", "configs/paper/discovery/w2_qwen_7b_propagation.yaml"),
        Job("w3_qwen_7b", "configs/paper/discovery/w3_qwen_7b_boundary_predictor.yaml"),
    ]
    seven13 += glob_jobs("e3", "configs/paper/calibration/e3_qwen_7b_*.yaml")
    seven13 += glob_jobs("e5", "configs/paper/delta/e5_qwen_7b_*.yaml")
    for context in (4096, 8192, 16384):
        seven13.append(Job(f"e6_qwen_7b_{context}_fixed", f"configs/paper/scaling/e6_qwen_7b_{context}.yaml", mode="e6_fixed"))

    arch13 = glob_jobs("a1", "configs/paper/architecture/*.yaml")

    fourteen4567 = [
        Job("e2_qwen_14b_controlled", "configs/paper/drift/e2_qwen_14b_controlled.yaml"),
        Job("e2_qwen_14b_longbench_v2", "configs/paper/drift/e2_qwen_14b_longbench_v2.yaml"),
    ]
    fourteen4567 += glob_jobs("e3", "configs/paper/calibration/e3_qwen_14b_*.yaml")
    for context in (8192, 16384):
        fourteen4567.append(Job(f"e6_qwen_14b_{context}_fixed", f"configs/paper/scaling/e6_qwen_14b_{context}.yaml", mode="e6_fixed"))

    sevenlong4567 = [
        Job("e6_qwen_7b_32768_fixed", "configs/paper/scaling/e6_qwen_7b_32768.yaml", mode="e6_fixed"),
        Job("e6_qwen_7b_65536_fixed", "configs/paper/scaling/e6_qwen_7b_65536.yaml", mode="e6_fixed"),
    ]
    longall = [
        Job("e6_qwen_14b_32768_fixed", "configs/paper/scaling/e6_qwen_14b_32768.yaml", mode="e6_fixed"),
    ]

    thirtysix = [
        Job("e2_qwen_32b_controlled", "configs/paper/drift/e2_qwen_32b_controlled.yaml"),
        Job("e2_qwen_32b_longbench_v2", "configs/paper/drift/e2_qwen_32b_longbench_v2.yaml"),
    ]
    thirtysix += glob_jobs("e3", "configs/paper/calibration/e3_qwen_32b_*.yaml")
    thirtysix += glob_jobs("e5", "configs/paper/delta/e5_qwen_32b_*.yaml")
    for context in (8192, 16384, 32768):
        thirtysix.append(Job(f"e6_qwen_32b_{context}_fixed", f"configs/paper/scaling/e6_qwen_32b_{context}.yaml", mode="e6_fixed"))

    return {
        "small0": small0,
        "seven13": seven13,
        "arch13": arch13,
        "fourteen4567": fourteen4567,
        "sevenlong4567": sevenlong4567,
        "longall": longall,
        "thirtysix": thirtysix,
    }


def queue_parallelism(queue: str) -> str:
    override = os.environ.get("FORMAL_MODEL_PARALLELISM")
    if override is not None:
        if override not in {"single", "model_shard"}:
            raise ValueError(
                "FORMAL_MODEL_PARALLELISM must be 'single' or 'model_shard'"
            )
        return override
    return "single" if queue == "small0" else "model_shard"


def patch_model(model: dict[str, Any], queue: str) -> None:
    model_name = str(model.get("model_name_or_path", ""))
    local = MODEL_PATHS.get(model_name)
    if local is None:
        raise KeyError(f"No local model mapping for {model_name!r}")
    if not Path(local).exists():
        raise FileNotFoundError(local)
    model["model_name_or_path"] = local
    model["device"] = "auto"
    model["parallelism"] = queue_parallelism(queue)
    model["device_ids"] = []


def patch_data(data: dict[str, Any]) -> None:
    if data.get("source") == "huggingface" and data.get("task") == "longbench_v2":
        if not LONG_BENCH_V2.exists():
            raise FileNotFoundError(LONG_BENCH_V2)
        data["source"] = "jsonl"
        data["dataset_path"] = str(LONG_BENCH_V2)


def prepare_config(job: Job, seed: int, output_dir: Path, queue: str) -> Path:
    source = ROOT / job.config
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if job.runner == "sweep":
        base = payload["base"]
        payload["output_dir"] = str(output_dir)
        base["output_dir"] = str(output_dir / "base")
        base["seed"] = seed
        base["resume"] = True
        base["checkpoint_each_target"] = True
        patch_model(base["model"], queue)
        patch_data(base["data"])
    else:
        payload["output_dir"] = str(output_dir)
        payload["seed"] = seed
        payload["resume"] = True
        payload["checkpoint_each_target"] = True
        patch_model(payload["model"], queue)
        patch_data(payload["data"])
        if "alora_prefix_reuse" in payload.get("cache", {}).get("strategies", []):
            payload["data"].setdefault("adapter_activation_marker", "<|adapter_activation|>")
        if job.mode == "e6_fixed":
            cache = payload["cache"]
            cache["strategies"] = [s for s in cache.get("strategies", []) if s in {"full_recompute", "stale_reuse", "periodic_refresh", "threshold_refresh"}]
            cache["failure_map_path"] = None
    destination = CONFIG_ROOT / queue / f"{job.name}.seed-{seed}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


def command_for(job: Job, config: Path) -> list[str]:
    base = [sys.executable, "-m", "ttt_cache_lab.cli"]
    if job.runner == "static":
        return base + ["static-run", "--config", str(config), "--version-summary"]
    if job.runner == "sweep":
        return base + ["versioned-sweep", "--config", str(config)]
    if job.runner == "blockwise":
        return base + ["blockwise-explore", "--config", str(config), "--block-sizes", "32", "64", "128", "--version-gap", "4", "--budget-fractions", "0.01", "0.02", "0.05", "0.10", "0.20"]
    return base + ["versioned-run", "--config", str(config), "--version-summary"]


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def append_status(queue: str, record: dict[str, Any]) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    with (STATUS_ROOT / f"{queue}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_one(queue: str, job: Job, seed: int) -> bool:
    out = RUN_ROOT / "results_v2" / queue / job.name / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    success = out / ".success"
    failed = out / ".failed"
    if success.exists():
        print(f"[skip] {queue}/{job.name}/seed-{seed}", flush=True)
        return True
    config = prepare_config(job, seed, out, queue)
    log = LOG_ROOT / queue / f"{job.name}.seed-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = command_for(job, config)
    append_status(queue, {"event": "start", "job": job.name, "seed": seed, "time": utcnow(), "config": str(config)})
    print(f"[start] {queue}/{job.name}/seed-{seed}", flush=True)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT / "src"),
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_NPU_ALLOC_CONF": env.get("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"),
    })
    rc = 1
    for attempt in (1, 2):
        started = time.monotonic()
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== attempt {attempt} started {utcnow()} =====\n")
            handle.flush()
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
            handle.write(f"\n===== attempt {attempt} exit={rc} duration_s={time.monotonic()-started:.3f} =====\n")
        if rc == 0:
            break
        time.sleep(20)
    event = "success" if rc == 0 else "failed"
    marker = success if rc == 0 else failed
    other = failed if rc == 0 else success
    other.unlink(missing_ok=True)
    marker.write_text(json.dumps({"event": event, "time": utcnow(), "config": str(config), "log": str(log), "return_code": rc}) + "\n", encoding="utf-8")
    append_status(queue, {"event": event, "job": job.name, "seed": seed, "time": utcnow(), "return_code": rc, "log": str(log)})
    print(f"[{event}] {queue}/{job.name}/seed-{seed}", flush=True)
    return rc == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, choices=sorted(build_queues()))
    parser.add_argument("--job", help="Run only the exact named job in the selected queue")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Run only one seed")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue after a failed run; the safe default is fail-fast",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    jobs = build_queues()[args.queue]
    if args.job is not None:
        jobs = [job for job in jobs if job.name == args.job]
        if not jobs:
            parser.error(f"queue {args.queue!r} has no job named {args.job!r}")
    if args.list:
        for job in jobs:
            seeds = (args.seed,) if args.seed is not None else job.seeds
            print(dataclasses.replace(job, seeds=seeds))
        return 0
    selected_runs = sum(1 if args.seed is not None else len(job.seeds) for job in jobs)
    print(
        f"queue={args.queue} visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
        f"parallelism={queue_parallelism(args.queue)} "
        f"jobs={len(jobs)} runs={selected_runs} fail_fast={not args.continue_on_failure}",
        flush=True,
    )
    ok = bad = 0
    aborted = False
    for job in jobs:
        seeds = (args.seed,) if args.seed is not None else job.seeds
        for seed in seeds:
            if run_one(args.queue, job, seed):
                ok += 1
            else:
                bad += 1
                if not args.continue_on_failure:
                    aborted = True
                    break
        if aborted:
            break
    full_queue = args.job is None and args.seed is None
    if full_queue and not aborted:
        done = STATUS_ROOT / f"{args.queue}.done.json"
        done.write_text(
            json.dumps(
                {"queue": args.queue, "finished": utcnow(), "success": ok, "failed": bad},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[queue-done] {args.queue} success={ok} failed={bad}", flush=True)
    else:
        state = "aborted" if aborted else "selection-done"
        print(f"[queue-{state}] {args.queue} success={ok} failed={bad}", flush=True)
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
