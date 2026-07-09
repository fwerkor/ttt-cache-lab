from pathlib import Path

from ttt_cache_lab.experiments.failure_map import generate_failure_map
from ttt_cache_lab.experiments.pareto import generate_pareto

HEADER = (
    "sample_id,update_target,cache_strategy,action,cache_state,first_invalid_layer,"
    "task_score,logits_kl,top1_agreement,relative_error,latency_units,reason,"
    "experiment_id,adapter_version,cached_version,version_gap,update_step,accumulated_update_norm,"
    "lora_rank,update_mode,hidden_relative_error,cache_bytes,memory_allocated,recompute_fraction,"
    "cache_hit,refresh_count,rejected_reuse,false_safe\n"
)
FULL_ROW = [
    "0",
    "lora.k",
    "full_recompute",
    "full_recompute",
    "invalid",
    "",
    "1",
    "0",
    "1",
    "0",
    "10",
    "x",
    "e",
    "1",
    "0",
    "1",
    "1",
    "0.01",
    "8",
    "random",
    "0",
    "1",
    "0",
    "1",
    "False",
    "1",
    "False",
    "False",
]
STALE_ROW = [
    "0",
    "lora.k",
    "stale_reuse",
    "reuse_stale",
    "valid_approx",
    "",
    "0",
    "0.2",
    "0",
    "0.5",
    "1",
    "x",
    "e",
    "1",
    "0",
    "1",
    "1",
    "0.01",
    "8",
    "random",
    "0.5",
    "1",
    "0",
    "0",
    "True",
    "0",
    "False",
    "True",
]


def _row(values: list[str], *, experiment: str, task_score: str | None = None) -> str:
    item = values.copy()
    item[12] = experiment
    if task_score is not None:
        item[6] = task_score
    return ",".join(item) + "\n"


def test_generate_failure_map(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    source.write_text(HEADER + _row(FULL_ROW, experiment="e3") + _row(STALE_ROW, experiment="e3"), encoding="utf-8")
    policy = generate_failure_map(source, tmp_path / "failure")
    assert policy.exists()
    assert "full_recompute" in policy.read_text(encoding="utf-8")
    assert (tmp_path / "failure" / "failure_map.csv").exists()
    assert (tmp_path / "failure" / "logits_kl_heatmap.svg").exists()


def test_generate_pareto(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    source.write_text(
        HEADER + _row(FULL_ROW, experiment="e4") + _row(STALE_ROW, experiment="e4", task_score="0.5"),
        encoding="utf-8",
    )
    output = generate_pareto(source, tmp_path / "pareto")
    assert output.exists()
    assert "dominated" in output.read_text(encoding="utf-8")
    assert (tmp_path / "pareto" / "pareto.md").exists()
