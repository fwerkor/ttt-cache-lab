from __future__ import annotations

import math

import numpy as np


def relative_error(reference: np.ndarray, candidate: np.ndarray, *, eps: float = 1e-12) -> float:
    numerator = float(np.linalg.norm(reference - candidate))
    denominator = float(np.linalg.norm(reference)) + eps
    return numerator / denominator


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray, *, eps: float = 1e-12) -> float:
    p = _softmax(reference_logits)
    q = _softmax(candidate_logits)
    value = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
    result = float(np.mean(value))
    if math.isnan(result):
        return float("inf")
    return result


def top1_agreement(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    ref = np.argmax(reference_logits, axis=-1)
    cand = np.argmax(candidate_logits, axis=-1)
    return float(np.mean(ref == cand))
