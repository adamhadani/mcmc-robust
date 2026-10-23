"""Shared test fixtures and numerical gradient verification helpers."""

import numpy as np
import pytest

from mcmc_robust.core.target import TargetDistribution


def numerical_gradient(target: TargetDistribution, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Computes central finite-difference gradient of target.log_prob at x."""
    x_arr = np.asarray(x, dtype=np.float64)
    d = len(x_arr)
    num_grad = np.empty(d, dtype=np.float64)
    for i in range(d):
        e_i = np.zeros(d, dtype=np.float64)
        e_i[i] = h
        lp_plus = target.log_prob(x_arr + e_i)
        lp_minus = target.log_prob(x_arr - e_i)
        num_grad[i] = (lp_plus - lp_minus) / (2.0 * h)
    return num_grad


def max_relative_error(anal_grad: np.ndarray, num_grad: np.ndarray) -> float:
    """Computes maximum standardized relative error between analytical and numerical gradients."""
    diff = np.abs(anal_grad - num_grad)
    scale = np.maximum(np.abs(anal_grad), 1.0)
    return float(np.max(diff / scale))


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible random number generator fixture."""
    return np.random.default_rng(12345)
