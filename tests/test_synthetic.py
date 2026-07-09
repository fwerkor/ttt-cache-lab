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
