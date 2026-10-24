"""Tests for the noise-aware tail-probability accuracy metric."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
from scipy import stats

from mcmc_robust.diagnostics.tail_accuracy import TailProbabilityAccuracy, tail_probability_accuracy

NU = 1.5
TAIL_PROBS = (0.01, 0.99)


def _t_quantiles(d: int) -> dict[float, np.ndarray]:
    return {p: np.full(d, stats.t.ppf(p, df=NU)) for p in TAIL_PROBS}


@pytest.mark.unit
def test_definition_matches_indicator_mean_and_arviz_mcse() -> None:
    """P_hat is the indicator mean, its MCSE is ArviZ's mean-MCSE of the indicator, z = (P_hat - p) / MCSE."""
    rng = np.random.default_rng(0)
    x = rng.standard_t(NU, size=(4, 1000, 6))
    q = _t_quantiles(6)
    res = tail_probability_accuracy(x, q)
    assert isinstance(res, TailProbabilityAccuracy)
    assert res.probs == TAIL_PROBS
    assert res.p_hat.shape == res.error.shape == res.mcse.shape == res.z.shape == (2, 6)
    for k, p in enumerate(TAIL_PROBS):
        ind = (x < q[p]).astype(np.float64)
        np.testing.assert_allclose(res.p_hat[k], ind.mean(axis=(0, 1)))
        np.testing.assert_allclose(res.error[k], ind.mean(axis=(0, 1)) - p)
        np.testing.assert_allclose(res.mcse[k], az.mcse(ind, method="mean"))
        np.testing.assert_allclose(res.z[k], (ind.mean(axis=(0, 1)) - p) / az.mcse(ind, method="mean"))
    assert not np.any(res.degenerate)
    s = res.summary()
    assert s["tail_prob_abs_err"] == pytest.approx(float(np.mean(np.abs(res.error))))
    assert s["tail_prob_abs_z"] == pytest.approx(float(np.mean(np.abs(res.z))))
    assert s["tail_prob_max_abs_z"] == pytest.approx(float(np.max(np.abs(res.z))))
    assert s["tail_prob_n_degenerate"] == 0


@pytest.mark.unit
def test_iid_draws_give_order_one_z_scores() -> None:
    """On exact iid draws |z| is O(1): its mean is near E|N(0,1)| = 0.80 and no |z| is extreme."""
    rng = np.random.default_rng(1)
    d = 20
    x = rng.standard_t(NU, size=(4, 1000, d))
    s = tail_probability_accuracy(x, _t_quantiles(d)).summary()
    assert 0.5 < s["tail_prob_abs_z"] < 1.15, s
    assert s["tail_prob_max_abs_z"] < 4.5, s
    # Binomial standard error of P_hat at p = 0.01 and n = 4000 is 0.0016.
    assert s["tail_prob_abs_err"] < 0.004, s


@pytest.mark.unit
def test_z_scores_are_calibrated_across_iid_replicates() -> None:
    """Across 200 iid replicates, z has mean square near 1 and about 5% of |z| exceed 1.96."""
    rng = np.random.default_rng(2)
    d = 5
    zs = np.concatenate(
        [
            tail_probability_accuracy(rng.standard_t(NU, size=(4, 1000, d)), _t_quantiles(d)).z.ravel()
            for _ in range(200)
        ]
    )
    assert 0.8 < float(np.mean(zs**2)) < 1.25
    assert 0.02 < float(np.mean(np.abs(zs) > 1.96)) < 0.09
    assert abs(float(np.mean(zs))) < 0.1


@pytest.mark.unit
def test_biased_tails_give_large_z_scores() -> None:
    """Draws whose tails are too light (scaled by 0.6) are flagged far beyond Monte Carlo noise."""
    rng = np.random.default_rng(3)
    d = 10
    x = 0.6 * rng.standard_t(NU, size=(4, 1000, d))
    s = tail_probability_accuracy(x, _t_quantiles(d)).summary()
    assert s["tail_prob_abs_z"] > 5.0, s


@pytest.mark.unit
def test_degenerate_indicators_use_the_binomial_floor_and_are_counted() -> None:
    """When no draw (or every draw) lies below q_p, ArviZ's MCSE is 0; z uses sqrt(p(1-p)/n) instead and stays finite."""
    rng = np.random.default_rng(4)
    d = 3
    x = rng.uniform(-1.0, 1.0, size=(4, 500, d))
    q = {0.01: np.full(d, -10.0), 0.99: np.full(d, 10.0)}
    res = tail_probability_accuracy(x, q)
    assert np.all(res.degenerate)
    n = 4 * 500
    floor = np.sqrt(0.01 * 0.99 / n)
    np.testing.assert_allclose(res.z[0], -0.01 / floor)
    np.testing.assert_allclose(res.z[1], 0.01 / floor)
    assert res.summary()["tail_prob_n_degenerate"] == 2 * d


@pytest.mark.unit
def test_input_shapes() -> None:
    """A single chain may be passed as (draws, d); mismatched quantile shapes are rejected."""
    rng = np.random.default_rng(5)
    x = rng.standard_t(NU, size=(800, 4))
    res = tail_probability_accuracy(x, _t_quantiles(4))
    assert res.z.shape == (2, 4)
    with pytest.raises(ValueError, match="quantile"):
        tail_probability_accuracy(x, _t_quantiles(3))
