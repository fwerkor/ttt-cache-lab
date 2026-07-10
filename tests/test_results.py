from pathlib import Path

from ttt_cache_lab.experiments.results import (
    ExperimentRecord,
    merge_record_files,
    read_records,
    write_records,
)


def _record(*, sample_id: int, strategy: str, task_score: float) -> ExperimentRecord:
    return ExperimentRecord(
        sample_id=sample_id,
        update_target="lora.k",
        cache_strategy=strategy,
        action="reuse_stale",
        cache_state="valid_approx",
        first_invalid_layer=None,
        task_score=task_score,
        logits_kl=0.01,
        top1_agreement=1.0,
        relative_error=0.1,
        latency_units=1.0,
        reason="test",
        experiment_id="unit",
        adapter_id=f"sample-{sample_id}:lora.k",
        adapter_version=1,
        cached_version=0,
        update_step=1,
        lora_rank=4,
        configured_update_norm=0.01,
        context_length=64,
        model_name="toy",
        seed=7,
    )


def test_write_records_merges_resume_checkpoints_by_identity(tmp_path: Path) -> None:
    first = _record(sample_id=0, strategy="stale_reuse", task_score=0.5)
    write_records([first], tmp_path)

    replacement = _record(sample_id=0, strategy="stale_reuse", task_score=0.75)
    second = _record(sample_id=1, strategy="stale_reuse", task_score=1.0)
    artifacts = write_records([replacement, second], tmp_path, merge_existing=True)

    records = read_records(artifacts.jsonl_path)
    assert len(records) == 2
    assert {record.sample_id for record in records} == {0, 1}
    assert next(record for record in records if record.sample_id == 0).task_score == 0.75
    assert not list(tmp_path.glob(".*.tmp"))


def test_merge_record_files_combines_distinct_runs(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = write_records(
        [_record(sample_id=0, strategy="stale_reuse", task_score=0.5)],
        first_dir,
    )
    second = write_records(
        [_record(sample_id=1, strategy="full_recompute", task_score=1.0)],
        second_dir,
    )
    merged = merge_record_files(
        [first.jsonl_path, second.jsonl_path],
        tmp_path / "merged",
    )
    assert len(merged.records) == 2
    assert {record.sample_id for record in merged.records} == {0, 1}


def test_write_records_replaces_old_run_when_resume_is_disabled(tmp_path: Path) -> None:
    write_records([_record(sample_id=0, strategy="stale_reuse", task_score=0.5)], tmp_path)
    artifacts = write_records(
        [_record(sample_id=1, strategy="full_recompute", task_score=1.0)],
        tmp_path,
        merge_existing=False,
    )
    records = read_records(artifacts.jsonl_path)
    assert len(records) == 1
    assert records[0].sample_id == 1
