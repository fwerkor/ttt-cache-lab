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


def test_prefix_match_allows_explanation_but_not_late_instruction_echo() -> None:
    sample = _sample(scorer="prefix_match", answers=("NOT_FOUND",))
    assert score_prediction(sample, "NOT_FOUND because the record is absent.") == 1.0
    assert score_prediction(sample, "The code is 1234; the prompt said NOT_FOUND.") == 0.0


def test_code_similarity_ignores_outer_whitespace() -> None:
    sample = _sample(scorer="code_similarity", answers=("function value() { return 1; }",))
    assert score_prediction(sample, "  function value() { return 1; }  ") == 1.0


def test_multiple_choice_accepts_label_or_option_text() -> None:
    sample = TaskSample(
        prompt="prompt",
        answer="C",
        metadata={
            "scorer": "multiple_choice",
            "answers": ("C", "planner"),
            "choice_labels": ("A", "B", "C", "D"),
            "choices": ("cache", "runtime", "planner", "metrics"),
        },
    )
    assert score_prediction(sample, "The answer is C.") == 1.0
    assert score_prediction(sample, "planner") == 1.0
    assert score_prediction(sample, "B") == 0.0


def test_numeric_match_extracts_formatted_number() -> None:
    sample = _sample(scorer="numeric_match", answers=("1200",))
    assert score_prediction(sample, "There are 1,200 events.") == 1.0


def test_set_f1_is_order_invariant() -> None:
    sample = _sample(scorer="set_f1", answers=("alpha, beta, gamma",))
    assert score_prediction(sample, "gamma; alpha; beta") == 1.0
    assert 0.0 < score_prediction(sample, "alpha, gamma") < 1.0
