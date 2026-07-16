#!/usr/bin/env python3
# ruff: noqa: E501
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "formal_20260712"
RESULT_ROOT = RUN_ROOT / "results_v2"
FINAL_ROOT = RUN_ROOT / "final_v2"


def run(cmd: list[str], *, log_name: str) -> bool:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = FINAL_ROOT / log_name
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"exit={proc.returncode}\n")
    print(f"[finalize] {log_name}: rc={proc.returncode}", flush=True)
    return proc.returncode == 0


def successful_result_dirs() -> list[Path]:
    return sorted(marker.parent for marker in RESULT_ROOT.rglob(".success"))


def merge_e3() -> None:
    inputs = [
        d / "records.jsonl" for d in successful_result_dirs() if "/e3_" in str(d) and (d / "records.jsonl").exists()
    ]
    if not inputs:
        print("[finalize] no successful E3 records found", flush=True)
        return
    merged_dir = FINAL_ROOT / "e3_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "ttt_cache_lab.cli",
        "merge-records",
        "--input",
        *map(str, inputs),
        "--output-dir",
        str(merged_dir),
    ]
    if not run(cmd, log_name="e3_merge.log"):
        return
    merged = merged_dir / "merged_records.csv"
    if merged.exists():
        run(
            [
                sys.executable,
                "-m",
                "ttt_cache_lab.cli",
                "failure-map",
                "--input",
                str(merged),
                "--output-dir",
                str(FINAL_ROOT / "failure_map"),
            ],
            log_name="e3_failure_map.log",
        )


def analyze_w() -> None:
    for seed_dir in successful_result_dirs():
        rel_name = seed_dir.parent.name
        analysis = seed_dir / "analysis"
        if "w1_" in rel_name:
            # The sweep-level file preserves the sweep axes required by
            # window-analysis. merge-records emits summary.csv and drops those
            # axes, so using it here silently skipped W1 finalization.
            merged = seed_dir / "merged_records.csv"
            if merged.exists():
                run(
                    [
                        sys.executable,
                        "-m",
                        "ttt_cache_lab.cli",
                        "window-analysis",
                        "--input",
                        str(merged),
                        "--output-dir",
                        str(analysis / "window"),
                    ],
                    log_name=f"{rel_name}-{seed_dir.name}-analysis.log",
                )
        elif "w2_" in rel_name:
            source = seed_dir / "propagation_records.csv"
            if source.exists():
                run(
                    [
                        sys.executable,
                        "-m",
                        "ttt_cache_lab.cli",
                        "propagation-analysis",
                        "--input",
                        str(source),
                        "--output-dir",
                        str(analysis / "propagation"),
                    ],
                    log_name=f"{rel_name}-{seed_dir.name}-analysis.log",
                )
        elif "w3_" in rel_name:
            boundary = seed_dir / "boundary_records.csv"
            summary = seed_dir / "summary.csv"
            if boundary.exists() and summary.exists():
                run(
                    [
                        sys.executable,
                        "-m",
                        "ttt_cache_lab.cli",
                        "boundary-analysis",
                        "--boundary-input",
                        str(boundary),
                        "--summary-input",
                        str(summary),
                        "--output-dir",
                        str(analysis / "boundary"),
                    ],
                    log_name=f"{rel_name}-{seed_dir.name}-analysis.log",
                )


def write_summary() -> None:
    statuses: list[dict[str, object]] = []
    for marker_name, state in ((".success", "success"), (".failed", "failed")):
        for marker in sorted(RESULT_ROOT.rglob(marker_name)):
            seed_dir = marker.parent
            try:
                details = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                details = {}
            statuses.append(
                {
                    "queue": seed_dir.parents[1].name,
                    "job": seed_dir.parent.name,
                    "seed": seed_dir.name.removeprefix("seed-"),
                    "state": state,
                    "details": details,
                }
            )
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    (FINAL_ROOT / "status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (FINAL_ROOT / "status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["queue", "job", "seed", "state"])
        writer.writeheader()
        for item in statuses:
            writer.writerow({key: item[key] for key in ("queue", "job", "seed", "state")})
    counts: dict[str, int] = {}
    for item in statuses:
        key = str(item["state"])
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# Formal experiment queue status",
        "",
        f"- Successful runs: {counts.get('success', 0)}",
        f"- Failed runs: {counts.get('failed', 0)}",
        "",
    ]
    for queue in sorted({str(item["queue"]) for item in statuses}):
        q = [item for item in statuses if item["queue"] == queue]
        lines.append(f"## {queue}")
        lines.append("")
        lines.append(f"- Success: {sum(item['state'] == 'success' for item in q)}")
        lines.append(f"- Failed: {sum(item['state'] == 'failed' for item in q)}")
        lines.append("")
    (FINAL_ROOT / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    write_summary()
    analyze_w()
    merge_e3()
    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
