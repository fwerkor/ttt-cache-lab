from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

NUMERIC_FIELDS = [
    "task_score",
    "logits_kl",
    "top1_agreement",
    "relative_error",
    "latency_units",
]


@dataclass(frozen=True)
class SummaryRow:
    update_target: str
    cache_strategy: str
    count: int
    task_score_mean: float
    logits_kl_mean: float
    top1_agreement_mean: float
    relative_error_mean: float
    latency_units_mean: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "update_target": self.update_target,
            "cache_strategy": self.cache_strategy,
            "count": self.count,
            "task_score_mean": self.task_score_mean,
            "logits_kl_mean": self.logits_kl_mean,
            "top1_agreement_mean": self.top1_agreement_mean,
            "relative_error_mean": self.relative_error_mean,
            "latency_units_mean": self.latency_units_mean,
        }


def summarize_csv(path: Path) -> list[SummaryRow]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            groups[(row["update_target"], row["cache_strategy"])].append(row)

    rows = []
    for (target, strategy), records in sorted(groups.items()):
        means = {}
        for field in NUMERIC_FIELDS:
            means[field] = sum(float(record[field]) for record in records) / len(records)
        rows.append(
            SummaryRow(
                update_target=target,
                cache_strategy=strategy,
                count=len(records),
                task_score_mean=means["task_score"],
                logits_kl_mean=means["logits_kl"],
                top1_agreement_mean=means["top1_agreement"],
                relative_error_mean=means["relative_error"],
                latency_units_mean=means["latency_units"],
            )
        )
    return rows


def write_summary(rows: list[SummaryRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].to_dict().keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def to_markdown(rows: list[SummaryRow]) -> str:
    headers = [
        "update_target",
        "cache_strategy",
        "n",
        "task",
        "kl",
        "top1",
        "rel_err",
        "latency",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.update_target,
                    row.cache_strategy,
                    str(row.count),
                    f"{row.task_score_mean:.4f}",
                    f"{row.logits_kl_mean:.6g}",
                    f"{row.top1_agreement_mean:.4f}",
                    f"{row.relative_error_mean:.6g}",
                    f"{row.latency_units_mean:.6g}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
