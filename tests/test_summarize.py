from pathlib import Path

from ttt_cache_lab.experiments.summarize import summarize_csv, to_markdown, write_summary


def test_summarize_csv(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    source.write_text(
        "sample_id,update_target,cache_strategy,action,cache_state,first_invalid_layer,"
        "task_score,logits_kl,top1_agreement,relative_error,latency_units,reason\n"
        "0,attention.q,full_recompute,full_recompute,invalid,,1,0.0,1.0,0.0,10,x\n"
        "1,attention.q,full_recompute,full_recompute,invalid,,0,0.2,0.5,0.1,12,x\n",
        encoding="utf-8",
    )
    rows = summarize_csv(source)
    assert len(rows) == 1
    assert rows[0].count == 2
    assert rows[0].task_score_mean == 0.5
    markdown = to_markdown(rows)
    assert "attention.q" in markdown
    output = tmp_path / "grouped.csv"
    write_summary(rows, output)
    assert output.exists()
