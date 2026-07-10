from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from ttt_cache_lab.configs import DataConfig
from ttt_cache_lab.data.synthetic import SyntheticTaskFactory, TaskSample

SYNTHETIC_TASKS = {"passkey", "key_value", "multi_needle", "variable_tracking"}


def build_task_samples(config: DataConfig, *, seed: int) -> list[TaskSample]:
    if config.source == "synthetic":
        if config.task not in SYNTHETIC_TASKS:
            raise ValueError(f"Unsupported synthetic task: {config.task}")
        samples = SyntheticTaskFactory(seed).build(
            config.task,
            num_samples=config.num_samples,
            context_length=config.context_length,
            answer_length=config.answer_length,
        )
        return [_with_runtime_metadata(sample, config=config, index=index) for index, sample in enumerate(samples)]
    if config.source == "jsonl":
        if config.dataset_path is None:
            raise ValueError("data.dataset_path is required when data.source=jsonl")
        records = _read_jsonl(config.dataset_path)
        return _records_to_samples(records, config=config)
    if config.source == "huggingface":
        if not config.dataset_name:
            raise ValueError("data.dataset_name is required when data.source=huggingface")
        records = _load_huggingface(config, seed=seed)
        return _records_to_samples(records, config=config)
    raise ValueError(f"Unsupported data source: {config.source}")


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record {line_number} in {path} is not an object")
            yield payload


def _load_huggingface(config: DataConfig, *, seed: int) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Hugging Face dataset loading requires: pip install -e '.[hf]'") from exc

    args: list[str] = [str(config.dataset_name)]
    if config.dataset_config:
        args.append(config.dataset_config)
    dataset = load_dataset(*args, split=config.dataset_split)
    shuffle = getattr(dataset, "shuffle", None)
    if callable(shuffle):
        dataset = shuffle(seed=seed)
    return cast(Iterable[Mapping[str, Any]], dataset)


def _records_to_samples(records: Iterable[Mapping[str, Any]], *, config: DataConfig) -> list[TaskSample]:
    samples: list[TaskSample] = []
    for index, record in enumerate(records):
        if len(samples) >= config.num_samples:
            break
        prompt = _build_prompt(record, config=config)
        answer = _coerce_answer(_field(record, config.answer_field))
        if not prompt.strip():
            raise ValueError(f"Dataset record {index} produced an empty prompt")
        if not answer.strip():
            raise ValueError(f"Dataset record {index} produced an empty answer")
        samples.append(
            TaskSample(
                prompt=prompt,
                answer=answer,
                metadata={
                    "task": config.task,
                    "source": config.source,
                    "record_index": index,
                    "truncation_strategy": config.truncation_strategy,
                },
            )
        )
    if not samples:
        raise ValueError("The configured dataset produced no samples")
    return samples


def _build_prompt(record: Mapping[str, Any], *, config: DataConfig) -> str:
    if config.context_field is None and config.question_field is None:
        return str(_field(record, config.prompt_field))
    context = "" if config.context_field is None else str(_field(record, config.context_field))
    question = "" if config.question_field is None else str(_field(record, config.question_field))
    return config.prompt_template.format(context=context, question=question)


def _field(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Dataset record does not contain field {path!r}")
        value = value[part]
    return value


def _coerce_answer(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        if not value:
            return ""
        return str(value[0])
    if value is None:
        return ""
    return str(value)


def _with_runtime_metadata(sample: TaskSample, *, config: DataConfig, index: int) -> TaskSample:
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "source": config.source,
            "record_index": index,
            "truncation_strategy": config.truncation_strategy,
        }
    )
    return TaskSample(prompt=sample.prompt, answer=sample.answer, metadata=metadata)
