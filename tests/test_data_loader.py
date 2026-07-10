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
    assert sample.metadata["answers"] == ("planner",)
    assert sample.metadata["scorer"] == "exact_match"
    assert sample.metadata["source"] == "jsonl"
    assert sample.metadata["truncation_strategy"] == "middle"
    assert sample.metadata["max_generation_tokens"] == config.answer_length


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


def test_selection_seed_and_offset_define_disjoint_model_seed_independent_partitions(tmp_path: Path) -> None:
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"prompt": f"prompt-{index}", "answer": f"answer-{index}", "id": index}) + "\n"
            for index in range(12)
        ),
        encoding="utf-8",
    )
    calibration = DataConfig(
        source="jsonl",
        task="partitioned",
        dataset_path=dataset,
        id_field="id",
        selection_seed=123,
        sample_offset=0,
        num_samples=4,
        evaluation_partition="calibration",
    )
    test = calibration.model_copy(
        update={"sample_offset": 4, "evaluation_partition": "test"}
    )
    first = build_task_samples(calibration, seed=1)
    repeated = build_task_samples(calibration, seed=999)
    held_out = build_task_samples(test, seed=1)
    assert [sample.metadata["dataset_sample_id"] for sample in first] == [
        sample.metadata["dataset_sample_id"] for sample in repeated
    ]
    assert {sample.metadata["dataset_sample_id"] for sample in first}.isdisjoint(
        {sample.metadata["dataset_sample_id"] for sample in held_out}
    )
    assert all(sample.metadata["evaluation_partition"] == "test" for sample in held_out)


def test_multiple_choice_loader_formats_choices_and_normalizes_answer(tmp_path: Path) -> None:
    dataset = tmp_path / "choices.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "context": "Long repository context",
                "question": "Which module owns the planner?",
                "choices": ["cache", "runtime", "planner", "metrics"],
                "answer": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = DataConfig(
        source="jsonl",
        task="longbench_v2",
        dataset_path=dataset,
        context_field="context",
        question_field="question",
        choices_field="choices",
        answer_field="answer",
        prompt_template="{context}\nQuestion: {question}\n{choices}\nAnswer:",
        scorer="multiple_choice",
        num_samples=1,
    )
    sample = build_task_samples(config, seed=7)[0]
    assert "A. cache" in sample.prompt
    assert "C. planner" in sample.prompt
    assert sample.answer == "C"
    assert sample.metadata["answers"] == ("C", "planner")


def test_huggingface_indexed_dataset_selects_partition_without_full_iteration(monkeypatch: object) -> None:
    class IndexedDataset:
        def __init__(self, records: list[dict[str, object]]) -> None:
            self.records = records
            self.selected_indices: list[int] | None = None

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, object]:
            return self.records[index]

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("The full dataset must not be materialized")

        def shuffle(self, *, seed: int) -> IndexedDataset:
            assert seed == 2027
            return self

        def select(self, indices: list[int]) -> IndexedDataset:
            self.selected_indices = indices
            return IndexedDataset([self.records[index] for index in indices])

    dataset = IndexedDataset(
        [{"prompt": f"prompt-{index}", "answer": f"answer-{index}"} for index in range(100)]
    )

    def load_dataset(*args: str, split: str) -> IndexedDataset:
        assert args == ("example/large",)
        assert split == "test"
        return dataset

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))  # type: ignore[attr-defined]
    config = DataConfig(
        source="huggingface",
        task="large",
        dataset_name="example/large",
        dataset_split="test",
        selection_seed=2027,
        sample_offset=20,
        num_samples=3,
    )
    samples = build_task_samples(config, seed=7)
    assert dataset.selected_indices == [20, 21, 22]
    assert [sample.answer for sample in samples] == ["answer-20", "answer-21", "answer-22"]
