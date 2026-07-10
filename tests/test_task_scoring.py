from ttt_cache_lab.data.scoring import score_prediction
from ttt_cache_lab.data.synthetic import TaskSample


def _sample(*, scorer: str, answers: tuple[str, ...]) -> TaskSample:
    return TaskSample(
        prompt="prompt",
        answer=answers[0],
        metadata={"scorer": scorer, "answers": answers},
    )


def test_exact_match_accepts_any_reference() -> None:
    sample = _sample(scorer="exact_match", answers=("New York", "NYC"))
    assert score_prediction(sample, "nyc") == 1.0


def test_token_f1_returns_partial_credit() -> None:
    sample = _sample(scorer="token_f1", answers=("alpha beta gamma",))
    score = score_prediction(sample, "alpha gamma")
    assert 0.0 < score < 1.0


def test_rouge_l_rewards_ordered_subsequence() -> None:
    sample = _sample(scorer="rouge_l", answers=("a b c d",))
    assert score_prediction(sample, "a c d") > score_prediction(sample, "d c a")


def test_contains_supports_retrieval_answers() -> None:
    sample = _sample(scorer="contains", answers=("alpha-7",))
    assert score_prediction(sample, "The identifier is alpha-7.") == 1.0


def test_code_similarity_ignores_outer_whitespace() -> None:
    sample = _sample(scorer="code_similarity", answers=("function value() { return 1; }",))
    assert score_prediction(sample, "  function value() { return 1; }  ") == 1.0
