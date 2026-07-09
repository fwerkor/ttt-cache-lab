import numpy as np

from ttt_cache_lab.metrics.tensor import kl_divergence, relative_error, top1_agreement


def test_relative_error_zero_for_same_tensor() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert relative_error(x, x) == 0.0


def test_top1_agreement() -> None:
    a = np.array([[0.1, 0.9], [0.8, 0.2]])
    b = np.array([[0.2, 0.7], [0.1, 0.9]])
    assert top1_agreement(a, b) == 0.5


def test_kl_nonnegative() -> None:
    a = np.array([[0.1, 0.9]])
    b = np.array([[0.2, 0.8]])
    assert kl_divergence(a, b) >= 0.0
