from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ttt_cache_lab.configs import DataConfig
from ttt_cache_lab.data.synthetic import SyntheticTaskFactory, TaskSample

SYNTHETIC_TASKS = {
    "passkey",
    "key_value",
    "multi_needle",
    "variable_tracking",
    "needle_absent",
    "multi_hop_tracing",
    "aggregation",
    "common_words",
}


def build_task_samples(config: DataConfig, *, seed: int) -> list[TaskSample]:
    del seed  # Dataset selection is intentionally independent of model/update randomness.
    if config.source == "synthetic":
        if config.task not in SYNTHETIC_TASKS:
            raise ValueError(f"Unsupported synthetic task: {config.task}")
        pool_size = config.sample_offset + config.num_samples
        samples = SyntheticTaskFactory(config.selection_seed).build(
            config.task,
            num_samples=pool_size,
            context_length=config.context_length,
            answer_length=config.answer_length,
        )
        selected = samples[config.sample_offset : config.sample_offset + config.num_samples]
        return [
            _with_runtime_metadata(
                sample,
                config=config,
                index=config.sample_offset + index,
            )
            for index, sample in enumerate(selected)
        ]
    if config.source == "jsonl":
        if config.dataset_path is None:
            raise ValueError("data.dataset_path is required when data.source=jsonl")
        records = _read_jsonl(config.dataset_path)
        return _records_to_samples(_select_records(records, config=config), config=config)
    if config.source == "huggingface":
        if not config.dataset_name:
            raise ValueError("data.dataset_name is required when data.source=huggingface")
        records = _load_huggingface(config)
        return _records_to_samples(_select_records(records, config=config), config=config)
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


def _load_huggingface(config: DataConfig) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face dataset loading requires: pip install -e '.[hf]'") from exc

    args: list[str] = [str(config.dataset_name)]
    if config.dataset_config:
        args.append(config.dataset_config)
    dataset = load_dataset(*args, split=config.dataset_split)
    return cast(Iterable[Mapping[str, Any]], dataset)


def _select_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config: DataConfig,
) -> list[tuple[int, Mapping[str, Any]]]:
    indexed_selection = _select_indexed_dataset(records, config=config)
    if indexed_selection is not None:
        return indexed_selection
    indexed = [
        (index, record)
        for index, record in enumerate(records)
        if _matches_filters(record, config.filters)
    ]
    if config.shuffle:
        random.Random(config.selection_seed).shuffle(indexed)
    start = config.sample_offset
    selected = indexed[start : start + config.num_samples]
    if not selected:
        raise ValueError(
            "The configured dataset selection produced no samples; check filters, sample_offset, and num_samples"
        )
    return selected


def _select_indexed_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    config: DataConfig,
) -> list[tuple[int, Mapping[str, Any]]] | None:
    dataset: Any = records
    select = getattr(dataset, "select", None)
    if not callable(select) or not hasattr(dataset, "__len__"):
        return None
    if config.filters:
        filter_records = getattr(dataset, "filter", None)
        if not callable(filter_records):
            return None
        dataset = filter_records(lambda record: _matches_filters(record, config.filters))
    if config.shuffle:
        shuffle = getattr(dataset, "shuffle", None)
        if not callable(shuffle):
            return None
        dataset = shuffle(seed=config.selection_seed)
    total = int(len(dataset))
    start = min(config.sample_offset, total)
    stop = min(total, start + config.num_samples)
    if start >= stop:
        raise ValueError(
            "The configured dataset selection produced no samples; check filters, sample_offset, and num_samples"
        )
    selected_dataset = dataset.select(list(range(start, stop)))
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for selection_index in range(len(selected_dataset)):
        record = selected_dataset[selection_index]
        if not isinstance(record, Mapping):
            raise TypeError("Indexed dataset selection returned a non-mapping record")
        selected.append((start + selection_index, record))
    return selected


def _matches_filters(record: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for path, expected in filters.items():
        try:
            actual = _field(record, path)
        except KeyError:
            return False
        if isinstance(expected, list | tuple | set):
            if actual not in expected and str(actual) not in {str(value) for value in expected}:
                return False
        elif actual != expected and str(actual) != str(expected):
            return False
    return True


def _records_to_samples(
    records: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    config: DataConfig,
) -> list[TaskSample]:
    samples: list[TaskSample] = []
    for selection_index, (record_index, record) in enumerate(records):
        choices = (
            _coerce_choices(_field(record, config.choices_field))
            if config.choices_field
            else _choices_from_fields(record, config.choice_fields)
        )
        prompt = _attach_activation_marker(
            _build_prompt(record, config=config, choices=choices),
            config=config,
        )
        raw_answer = _field(record, config.answer_field)
        answers = _coerce_answers(raw_answer)
        choice_labels = tuple(label for label, _ in choices)
        choice_values = tuple(value for _, value in choices)
        if config.scorer == "multiple_choice":
            answers = _multiple_choice_references(
                raw_answer,
                labels=choice_labels,
                choices=choice_values,
            )
        answer = answers[0] if answers else ""
        if not prompt.strip():
            raise ValueError(f"Dataset record {record_index} produced an empty prompt")
        if not answer.strip():
            raise ValueError(f"Dataset record {record_index} produced an empty answer")
        category = (
            str(_field(record, config.category_field))
            if config.category_field
            else ""
        )
        selected_metadata = {
            field: _field(record, field)
            for field in config.metadata_fields
            if _has_field(record, field)
        }
        samples.append(
            TaskSample(
                prompt=prompt,
                answer=answer,
                metadata={
                    "task": config.task,
                    "task_family": config.task_family,
                    "benchmark_name": config.benchmark_name or config.dataset_name or config.source,
                    "evaluation_partition": config.evaluation_partition,
                    "source": config.source,
                    "dataset_split": config.dataset_split,
                    "record_index": record_index,
                    "selection_index": selection_index,
                    "dataset_sample_id": _dataset_sample_id(record, config=config, record_index=record_index),
                    "category": category,
                    "dataset_metadata": selected_metadata,
                    "truncation_strategy": config.truncation_strategy,
                    "adapter_activation_marker": config.adapter_activation_marker or "",
                    "scorer": config.scorer,
                    "max_generation_tokens": max(1, config.answer_length),
                    "answers": answers,
                    "choice_labels": choice_labels,
                    "choices": choice_values,
                },
            )
        )
    return samples


def _build_prompt(
    record: Mapping[str, Any],
    *,
    config: DataConfig,
    choices: tuple[tuple[str, str], ...],
) -> str:
    if config.context_field is None and config.question_field is None and not choices:
        return str(_field(record, config.prompt_field))
    context = "" if config.context_field is None else str(_field(record, config.context_field))
    question = "" if config.question_field is None else str(_field(record, config.question_field))
    choices_text = "\n".join(
        config.choice_template.format(label=label, choice=choice)
        for label, choice in choices
    )
    return config.prompt_template.format(
        context=context,
        question=question,
        choices=choices_text,
    )


def _field(record: Mapping[str, Any], path: str | None) -> Any:
    if path is None:
        return None
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Dataset record does not contain field {path!r}")
        value = value[part]
    return value


def _has_field(record: Mapping[str, Any], path: str) -> bool:
    try:
        _field(record, path)
    except KeyError:
        return False
    return True


def _coerce_answers(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if str(item).strip())
    if value is None:
        return ()
    return (str(value),)



def _choices_from_fields(
    record: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    choices: list[tuple[str, str]] = []
    for index, field in enumerate(fields):
        label = field.rsplit("_", maxsplit=1)[-1]
        if len(label) != 1 or not label.isalpha():
            label = _choice_label(index)
        choices.append((label.upper(), str(_field(record, field))))
    return tuple(choices)

def _coerce_choices(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple((str(label), str(choice)) for label, choice in value.items())
    if isinstance(value, list | tuple):
        return tuple((_choice_label(index), str(choice)) for index, choice in enumerate(value))
    raise ValueError("choices_field must resolve to a list, tuple, or mapping")


def _multiple_choice_references(
    value: Any,
    *,
    labels: tuple[str, ...],
    choices: tuple[str, ...],
) -> tuple[str, ...]:
    raw = _coerce_answers(value)
    if not raw:
        return ()
    first = raw[0].strip()
    if first.isdigit() and labels:
        index = int(first)
        if 0 <= index < len(labels):
            return (labels[index], choices[index])
        if 1 <= index <= len(labels):
            return (labels[index - 1], choices[index - 1])
    for label, choice in zip(labels, choices, strict=True):
        if first.casefold() == label.casefold() or first.casefold() == choice.casefold():
            return (label, choice)
    return raw


def _choice_label(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return str(index + 1)


def _dataset_sample_id(
    record: Mapping[str, Any],
    *,
    config: DataConfig,
    record_index: int,
) -> str:
    if config.id_field and _has_field(record, config.id_field):
        return str(_field(record, config.id_field))
    try:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(record_index)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _attach_activation_marker(prompt: str, *, config: DataConfig) -> str:
    marker = config.adapter_activation_marker
    if not marker or marker in prompt:
        return prompt
    answer_index = prompt.rfind("Answer:")
    if answer_index >= 0:
        return prompt[:answer_index] + marker + " " + prompt[answer_index:]
    return prompt + "\n" + marker


def _with_runtime_metadata(sample: TaskSample, *, config: DataConfig, index: int) -> TaskSample:
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "task": config.task,
            "task_family": config.task_family or "controlled",
            "benchmark_name": config.benchmark_name or "synthetic",
            "evaluation_partition": config.evaluation_partition,
            "source": config.source,
            "dataset_split": "generated",
            "record_index": index,
            "selection_index": index - config.sample_offset,
            "dataset_sample_id": f"synthetic-{config.task}-{config.selection_seed}-{index}",
            "category": config.task_family or "controlled",
            "dataset_metadata": {},
            "truncation_strategy": config.truncation_strategy,
            "adapter_activation_marker": config.adapter_activation_marker or "",
            "scorer": config.scorer,
            "max_generation_tokens": max(1, config.answer_length),
            "answers": (sample.answer,),
            "choice_labels": (),
            "choices": (),
        }
    )
    prompt = _attach_activation_marker(sample.prompt, config=config)
    return TaskSample(prompt=prompt, answer=sample.answer, metadata=metadata)
