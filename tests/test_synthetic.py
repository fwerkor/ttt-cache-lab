from ttt_cache_lab.data.synthetic import SyntheticTaskFactory


def test_passkey_contains_answer() -> None:
    sample = SyntheticTaskFactory(1).passkey(context_length=128, answer_length=4)
    assert sample.answer in sample.prompt
    assert sample.metadata["task"] == "passkey"
