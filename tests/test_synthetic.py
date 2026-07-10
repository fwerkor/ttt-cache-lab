import re

from ttt_cache_lab.data.synthetic import SyntheticTaskFactory


def test_passkey_task_is_deterministic() -> None:
    first = SyntheticTaskFactory(7).passkey(context_length=64, answer_length=4)
    second = SyntheticTaskFactory(7).passkey(context_length=64, answer_length=4)
    assert first == second
    assert first.answer in first.prompt


def test_key_value_task_contains_target_pair() -> None:
    sample = SyntheticTaskFactory(7).key_value(context_length=64, answer_length=4)
    assert str(sample.metadata["key"]) in sample.prompt
    assert sample.answer in sample.prompt


def test_multi_needle_task_contains_target_answer() -> None:
    sample = SyntheticTaskFactory(7).multi_needle(context_length=1024, answer_length=4)
    assert sample.metadata["task"] == "multi_needle"
    assert str(sample.metadata["target"]) in sample.prompt
    assert sample.answer in sample.prompt


def test_variable_tracking_task_contains_final_answer() -> None:
    sample = SyntheticTaskFactory(7).variable_tracking(context_length=1024, answer_length=4)
    assert sample.metadata["task"] == "variable_tracking"
    assert str(sample.metadata["variable"]) in sample.prompt
    assert sample.answer in sample.prompt


def test_build_supports_new_tasks() -> None:
    factory = SyntheticTaskFactory(7)
    for task in ("multi_needle", "variable_tracking"):
        samples = factory.build(task, num_samples=2, context_length=512, answer_length=4)
        assert len(samples) == 2
        assert all(sample.answer for sample in samples)


def test_extended_long_context_tasks_are_deterministic() -> None:
    for task in ("needle_absent", "multi_hop_tracing", "aggregation", "common_words"):
        first = SyntheticTaskFactory(19).build(task, num_samples=2, context_length=2048, answer_length=4)
        second = SyntheticTaskFactory(19).build(task, num_samples=2, context_length=2048, answer_length=4)
        assert first == second
        assert all(sample.answer for sample in first)


def test_synthetic_prompts_reserve_tokenizer_headroom() -> None:
    tasks = (
        "passkey",
        "key_value",
        "multi_needle",
        "needle_absent",
        "multi_hop_tracing",
        "aggregation",
        "common_words",
        "variable_tracking",
    )
    context_length = 4096
    for task in tasks:
        sample = SyntheticTaskFactory(2027).build(
            task,
            num_samples=1,
            context_length=context_length,
            answer_length=4,
        )[0]
        assert len(sample.prompt.split()) <= int(context_length * 0.45)


def test_key_value_target_key_has_one_definition() -> None:
    sample = SyntheticTaskFactory(7).key_value(context_length=4096, answer_length=4)
    key = str(sample.metadata["key"])
    definitions = re.findall(rf"^{re.escape(key)}:\s", sample.prompt, flags=re.MULTILINE)
    assert len(definitions) == 1


def test_multi_hop_sources_have_one_outgoing_edge() -> None:
    sample = SyntheticTaskFactory(7).multi_hop_tracing(context_length=4096, answer_length=4)
    sources = re.findall(r"^(entity_\d+) points to ", sample.prompt, flags=re.MULTILINE)
    assert len(sources) == len(set(sources))


def test_aggregation_reference_matches_generated_target_count() -> None:
    sample = SyntheticTaskFactory(7).aggregation(context_length=4096, answer_length=4)
    target = str(sample.metadata["target"])
    facts = sample.prompt.split("\nQuestion:", maxsplit=1)[0]
    assert facts.splitlines().count(f"Event belongs to {target}.") == int(sample.answer)


def test_variable_tracking_distractors_never_reassign_target() -> None:
    sample = SyntheticTaskFactory(7).variable_tracking(context_length=4096, answer_length=4)
    variable = str(sample.metadata["variable"])
    assignments = re.findall(rf"{re.escape(variable)} = [a-z]+\.", sample.prompt)
    assert len(assignments) == int(sample.metadata["updates"])


def test_synthetic_difficulty_controls_structural_complexity() -> None:
    easy_needle = SyntheticTaskFactory(7).multi_needle(
        context_length=16384,
        answer_length=4,
        difficulty="easy",
    )
    hard_needle = SyntheticTaskFactory(7).multi_needle(
        context_length=16384,
        answer_length=4,
        difficulty="hard",
    )
    assert easy_needle.metadata["needle_count"] < hard_needle.metadata["needle_count"]

    easy_hop = SyntheticTaskFactory(7).multi_hop_tracing(
        context_length=16384,
        answer_length=4,
        difficulty="easy",
    )
    hard_hop = SyntheticTaskFactory(7).multi_hop_tracing(
        context_length=16384,
        answer_length=4,
        difficulty="hard",
    )
    assert easy_hop.metadata["hop_count"] < hard_hop.metadata["hop_count"]

    easy_tracking = SyntheticTaskFactory(7).variable_tracking(
        context_length=16384,
        answer_length=4,
        difficulty="easy",
    )
    hard_tracking = SyntheticTaskFactory(7).variable_tracking(
        context_length=16384,
        answer_length=4,
        difficulty="hard",
    )
    assert easy_tracking.metadata["updates"] < hard_tracking.metadata["updates"]

    easy_set = SyntheticTaskFactory(7).common_words(
        context_length=16384,
        answer_length=4,
        difficulty="easy",
    )
    hard_set = SyntheticTaskFactory(7).common_words(
        context_length=16384,
        answer_length=4,
        difficulty="hard",
    )
    assert easy_set.metadata["list_count"] < hard_set.metadata["list_count"]

    easy_aggregation = SyntheticTaskFactory(7).aggregation(
        context_length=16384,
        answer_length=4,
        difficulty="easy",
    )
    hard_aggregation = SyntheticTaskFactory(7).aggregation(
        context_length=16384,
        answer_length=4,
        difficulty="hard",
    )
    easy_semantic_events = easy_aggregation.prompt.count("Event belongs to group_")
    hard_semantic_events = hard_aggregation.prompt.count("Event belongs to group_")
    assert easy_semantic_events < hard_semantic_events

    easy_edges = easy_hop.prompt.count(" points to ")
    hard_edges = hard_hop.prompt.count(" points to ")
    assert easy_edges < hard_edges

    assert int(easy_set.metadata["unique_per_list"]) < int(
        hard_set.metadata["unique_per_list"]
    )
