import csv
from pathlib import Path

import pytest

from ttt_cache_lab.experiments.results import ExperimentRecord, write_records
from ttt_cache_lab.experiments.study_analysis import generate_study_analysis


def _record(
    *,
    experiment_id: str,
    strategy: str,
    version: int,
    gap: int,
    task_score: float,
    logits_kl: float,
    top1: float,
    rank: int = 4,
    norm: float = 0.01,
    context: int = 128,
    model: str = "toy",
) -> ExperimentRecord:
    return ExperimentRecord(
        sample_id=0,
        update_target="lora.k:1",
        cache_strategy=strategy,
        action="full_recompute" if strategy == "full_recompute" else "reuse_stale",
        cache_state="invalid" if strategy == "full_recompute" else "valid_approx",
        first_invalid_layer=None,
        task_score=task_score,
        logits_kl=logits_kl,
        top1_agreement=top1,
        relative_error=logits_kl,
        latency_units=10.0 if strategy == "full_recompute" else 1.0,
        reason="test",
        experiment_id=experiment_id,
        adapter_id="a",
        adapter_version=version,
        cached_version=max(0, version - gap),
        version_gap=gap,
        update_step=version,
        accumulated_update_norm=norm * version,
        update_norm_since_cache=norm * max(1, gap),
        lora_rank=rank,
        hidden_relative_error=logits_kl,
        cache_bytes=100,
        end_to_end_latency=10.0 if strategy == "full_recompute" else 1.0,
        throughput_tokens_per_s=1.0,
        cache_hit=strategy != "full_recompute",
        false_safe=logits_kl > 0.05 or top1 < 0.99,
        cache_entry_count=version + 1,
        total_cache_bytes=100 * (version + 1),
        context_length=context,
        model_name=model,
        model_num_layers=4,
        model_hidden_size=32,
        configured_update_norm=norm,
    )


def test_generate_dedicated_e1_to_e7_outputs(tmp_path: Path) -> None:
    records = []
    for experiment in (
        "e1_static",
        "e2_drift",
        "e3_failure_map",
        "e4_planner",
        "e5_delta",
        "e6_scaling",
        "e7_ablation",
    ):
        records.extend(
            [
                _record(
                    experiment_id=experiment,
                    strategy="full_recompute",
                    version=1,
                    gap=1,
                    task_score=1.0,
                    logits_kl=0.0,
                    top1=1.0,
                ),
                _record(
                    experiment_id=experiment,
                    strategy="stale_reuse",
                    version=1,
                    gap=1,
                    task_score=0.8,
                    logits_kl=0.2,
                    top1=0.0,
                ),
            ]
        )
    records.extend(
        [
            _record(
                experiment_id="e5_delta",
                strategy="full_recompute",
                version=2,
                gap=2,
                task_score=1.0,
                logits_kl=0.0,
                top1=1.0,
                rank=8,
                norm=0.02,
            ),
            _record(
                experiment_id="e5_delta",
                strategy="delta_correction",
                version=2,
                gap=2,
                task_score=1.0,
                logits_kl=0.01,
                top1=1.0,
                rank=8,
                norm=0.02,
            ),
            _record(
                experiment_id="e6_scaling",
                strategy="full_recompute",
                version=2,
                gap=2,
                task_score=1.0,
                logits_kl=0.0,
                top1=1.0,
                context=256,
                model="toy-large",
            ),
        ]
    )
    source = write_records(records, tmp_path / "run").csv_path
    output = tmp_path / "analysis"
    artifacts = generate_study_analysis(source, output)
    assert artifacts
    expected = {
        "e1_cache_cost.csv",
        "e1_memory_latency.svg",
        "e2_version_drift.csv",
        "e2_first_boundary.csv",
        "e2_task_drop_by_gap.svg",
        "e3_records.csv",
        "e4_planner_comparison.csv",
        "e4_quality_cost.svg",
        "e4_records.csv",
        "e5_safe_region.csv",
        "e5_safe_region_heatmap.svg",
        "e6_context_model_scaling.csv",
        "e6_latency_by_context.svg",
        "e6_speedup_by_context.svg",
        "e6_task_drop_by_context.svg",
        "e7_failure_boundary.csv",
        "e7_false_safe_rate.svg",
        "e7_ablation_effect.csv",
        "adaptation_effect.csv",
    }
    assert expected <= {path.name for path in artifacts}
    assert all((output / name).exists() for name in expected)
    assert (output / "e3_failure_map" / "failure_map.csv").exists()
    assert (output / "e3_failure_map" / "attention_shift_heatmap.svg").exists()
    assert (output / "e4_pareto" / "pareto.csv").exists()
    assert (output / "e4_pareto" / "pareto.svg").exists()
    with (output / "e5_safe_region.csv").open(newline="", encoding="utf-8") as handle:
        e5_rows = list(csv.DictReader(handle))
    assert "relative_error_mean" in e5_rows[0]
    assert "physical_cache_bytes_mean" in e5_rows[0]
    assert "fallback_rate" in e5_rows[0]


def test_e7_analysis_computes_paired_ablation_effects(tmp_path: Path) -> None:
    adaptive = _record(
        experiment_id="e7_ablation",
        strategy="adaptive",
        version=2,
        gap=2,
        task_score=0.9,
        logits_kl=0.01,
        top1=1.0,
    )
    ablated = _record(
        experiment_id="e7_ablation",
        strategy="adaptive_no_delta",
        version=2,
        gap=2,
        task_score=0.7,
        logits_kl=0.2,
        top1=0.0,
    )
    full = _record(
        experiment_id="e7_ablation",
        strategy="full_recompute",
        version=2,
        gap=2,
        task_score=1.0,
        logits_kl=0.0,
        top1=1.0,
    )
    source = write_records([full, adaptive, ablated], tmp_path / "run").csv_path
    output = tmp_path / "analysis"
    generate_study_analysis(source, output)
    with (output / "e7_ablation_effect.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["cache_strategy"] == "adaptive_no_delta"
    assert float(rows[0]["task_score_delta_vs_adaptive_mean"]) == pytest.approx(-0.2)


def test_e2_analysis_preserves_context_and_update_norm_axes(tmp_path: Path) -> None:
    records = []
    for context, norm in ((128, 0.01), (256, 0.02)):
        records.extend(
            [
                _record(
                    experiment_id="e2_drift",
                    strategy="full_recompute",
                    version=1,
                    gap=1,
                    task_score=1.0,
                    logits_kl=0.0,
                    top1=1.0,
                    context=context,
                    norm=norm,
                ),
                _record(
                    experiment_id="e2_drift",
                    strategy="stale_reuse",
                    version=1,
                    gap=1,
                    task_score=0.9,
                    logits_kl=0.01,
                    top1=1.0,
                    context=context,
                    norm=norm,
                ),
            ]
        )
    source = write_records(records, tmp_path / "run").csv_path
    output = tmp_path / "analysis"
    generate_study_analysis(source, output)
    with (output / "e2_version_drift.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stale = [row for row in rows if row["cache_strategy"] == "stale_reuse"]
    assert {(row["context_length"], row["configured_update_norm"]) for row in stale} == {
        ("128", "0.01"),
        ("256", "0.02"),
    }
