from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from ttt_cache_lab.data.synthetic import TaskSample


def score_prediction(sample: TaskSample, prediction: str) -> float:
    scorer = str(sample.metadata.get("scorer", "exact_match"))
    references = _references(sample)
    if scorer == "exact_match":
        return max(_exact_match(prediction, reference) for reference in references)
    if scorer == "contains":
        return max(_contains(prediction, reference) for reference in references)
    if scorer == "token_f1":
        return max(_token_f1(prediction, reference) for reference in references)
    if scorer == "rouge_l":
        return max(_rouge_l(prediction, reference) for reference in references)
    if scorer == "code_similarity":
        return max(_code_similarity(prediction, reference) for reference in references)
    if scorer == "multiple_choice":
        return _multiple_choice(sample, prediction)
    if scorer == "numeric_match":
        return max(_numeric_match(prediction, reference) for reference in references)
    if scorer == "set_f1":
        return max(_set_f1(prediction, reference) for reference in references)
    raise ValueError(f"Unsupported task scorer: {scorer}")


def _references(sample: TaskSample) -> tuple[str, ...]:
    raw: Any = sample.metadata.get("answers")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list | tuple):
        references = tuple(str(value) for value in raw if str(value).strip())
        if references:
            return references
    return (sample.answer,)



def _multiple_choice(sample: TaskSample, prediction: str) -> float:
    labels = tuple(str(value) for value in sample.metadata.get("choice_labels", ()))
    choices = tuple(str(value) for value in sample.metadata.get("choices", ()))
    references = _references(sample)
    correct_label = references[0].strip().upper()
    normalized_prediction = _normalize(prediction)
    for reference in references[1:]:
        if _normalize(reference) and _normalize(reference) == normalized_prediction:
            return 1.0
    if choices:
        for label, choice in zip(labels, choices, strict=True):
            if _normalize(choice) == normalized_prediction:
                return float(label.upper() == correct_label)
    candidates = re.findall(r"(?<![A-Za-z0-9])([A-Za-z])(?:[\s\).,:]|$)", prediction.strip())
    valid = [candidate.upper() for candidate in candidates if candidate.upper() in {label.upper() for label in labels}]
    if valid:
        return float(valid[-1] == correct_label)
    return 0.0


def _numeric_match(prediction: str, reference: str) -> float:
    predicted = _first_number(prediction)
    expected = _first_number(reference)
    if predicted is None or expected is None:
        return 0.0
    tolerance = 1e-9 * max(1.0, abs(expected))
    return float(abs(predicted - expected) <= tolerance)


def _first_number(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _set_f1(prediction: str, reference: str) -> float:
    predicted = _normalized_items(prediction)
    expected = _normalized_items(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = len(predicted & expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _normalized_items(text: str) -> set[str]:
    chunks = re.split(r"[,;\n]+", text)
    return {normalized for chunk in chunks if (normalized := _normalize(chunk))}

def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return " ".join(lowered.split())


def _exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def _contains(prediction: str, reference: str) -> float:
    normalized_reference = _normalize(reference)
    if not normalized_reference:
        return 0.0
    return 1.0 if normalized_reference in _normalize(prediction) else 0.0


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _normalize(prediction).split()
    reference_tokens = _normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, reference: str) -> float:
    prediction_tokens = _normalize(prediction).split()
    reference_tokens = _normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    lcs = _lcs_length(prediction_tokens, reference_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _code_similarity(prediction: str, reference: str) -> float:
    prediction_code = _normalize_code(prediction)
    reference_code = _normalize_code(reference)
    if not prediction_code or not reference_code:
        return float(prediction_code == reference_code)
    return SequenceMatcher(None, prediction_code, reference_code, autojunk=False).ratio()


def _normalize_code(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
