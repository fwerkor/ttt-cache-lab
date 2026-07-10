from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from ttt_cache_lab.configs import DataConfig
from ttt_cache_lab.data.loader import build_task_samples


def test_jsonl_loader_maps_context_question_and_answer(tmp_path: Path) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "context": "repository context",
                "question": "Which module owns the cache?",
                "answers": ["planner"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = DataConfig.model_validate(
        {
            "source": "jsonl",
            "task": "repo_qa",
            "dataset_path": dataset,
            "num_samples": 1,
            "context_field": "context",
            "question_field": "question",
            "answer_field": "answers",
            "truncation_strategy": "middle",
        }
    )
    sample = build_task_samples(config, seed=7)[0]
    assert "repository context" in sample.prompt
    assert "Which module owns the cache?" in sample.prompt
    assert sample.answer == "planner"
    assert sample.metadata["source"] == "jsonl"
    assert sample.metadata["truncation_strategy"] == "middle"


def test_huggingface_loader_uses_config_and_seed(monkeypatch: object) -> None:
    class FakeDataset(list[dict[str, object]]):
        def shuffle(self, *, seed: int) -> FakeDataset:
            assert seed == 9
            return self

    calls: list[tuple[tuple[str, ...], str]] = []

    def load_dataset(*args: str, split: str) -> FakeDataset:
        calls.append((args, split))
        return FakeDataset(
            [
                {
                    "context": "long context",
                    "input": "Find the stored identifier.",
                    "answers": ["alpha-7"],
                }
            ]
        )

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))  # type: ignore[attr-defined]
    config = DataConfig.model_validate(
        {
            "source": "huggingface",
            "task": "longbench",
            "dataset_name": "THUDM/LongBench",
            "dataset_config": "passage_retrieval_en",
            "dataset_split": "test",
            "num_samples": 1,
            "context_field": "context",
            "question_field": "input",
            "answer_field": "answers",
        }
    )
    sample = build_task_samples(config, seed=9)[0]
    assert calls == [(('THUDM/LongBench', 'passage_retrieval_en'), 'test')]
    assert sample.answer == "alpha-7"


def test_synthetic_loader_adds_runtime_metadata() -> None:
    config = DataConfig(task="passkey", num_samples=1, context_length=64, answer_length=2)
    sample = build_task_samples(config, seed=1)[0]
    assert sample.metadata["source"] == "synthetic"
    assert sample.metadata["record_index"] == 0



def test_activation_marker_is_inserted_before_answer() -> None:
    config = DataConfig(
        task="passkey",
        num_samples=1,
        context_length=64,
        answer_length=2,
        adapter_activation_marker="<ADAPTER>",
    )
    sample = build_task_samples(config, seed=1)[0]
    assert "<ADAPTER> Answer:" in sample.prompt
    assert sample.metadata["adapter_activation_marker"] == "<ADAPTER>"
