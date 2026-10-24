"""Tests for the noise-aware tail-probability accuracy metric."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
from scipy import stats

from mcmc_robust.diagnostics.tail_accuracy import (
    ReferenceAgreement,
    TailProbabilityAccuracy,
    reference_agreement,
    tail_probability_accuracy,
)

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


# =============================================================================
# Agreement with reference draws (posteriordb), whose own Monte Carlo error counts
# =============================================================================

REF_PROBS = (0.01, 0.05, 0.95, 0.99)


@pytest.mark.unit
def test_reference_agreement_definition() -> None:
    """Thresholds are reference quantiles; errors are run-minus-reference; z divides by both MCSEs in quadrature."""
    rng = np.random.default_rng(10)
    x = rng.standard_t(3.0, size=(4, 1000, 3))
    ref = rng.standard_t(3.0, size=(10, 1000, 3))
    res = reference_agreement(x, ref, REF_PROBS)
    assert isinstance(res, ReferenceAgreement)
    assert res.probs == REF_PROBS
    assert res.error.shape == res.z.shape == (4, 3)
    assert res.mean_z.shape == (3,)
    for k, p in enumerate(REF_PROBS):
        q = np.quantile(ref.reshape(-1, 3), p, axis=0)
        np.testing.assert_allclose(res.thresholds[k], q)
        ind_x = (x < q).astype(np.float64)
        ind_r = (ref < q).astype(np.float64)
        err = ind_x.mean(axis=(0, 1)) - ind_r.mean(axis=(0, 1))
        se = np.sqrt(az.mcse(ind_x, method="mean") ** 2 + az.mcse(ind_r, method="mean") ** 2)
        np.testing.assert_allclose(res.error[k], err)
        np.testing.assert_allclose(res.z[k], err / se)
    mean_se = np.sqrt(az.mcse(x, method="mean") ** 2 + az.mcse(ref, method="mean") ** 2)
    np.testing.assert_allclose(res.mean_z, (x.mean(axis=(0, 1)) - ref.mean(axis=(0, 1))) / mean_se)
    s = res.summary()
    assert s["ref_tail_prob_abs_z"] == pytest.approx(float(np.mean(np.abs(res.z))))
    assert s["ref_tail_prob_max_abs_z"] == pytest.approx(float(np.max(np.abs(res.z))))
    assert s["ref_tail_prob_abs_err"] == pytest.approx(float(np.mean(np.abs(res.error))))
    assert s["ref_mean_abs_z"] == pytest.approx(float(np.mean(np.abs(res.mean_z))))
    assert s["ref_mean_max_abs_z"] == pytest.approx(float(np.max(np.abs(res.mean_z))))
    assert s["ref_tail_prob_n_degenerate"] == 0


@pytest.mark.unit
@pytest.mark.statistical
def test_reference_agreement_z_scores_are_calibrated_when_run_and_reference_agree() -> None:
    """Run and reference iid from the same law: tail z has mean square near 1 over 100 replicates.

    Without the reference's MCSE the mean square would be about 1.4 (4000 run draws, 10000 reference).
    """
    rng = np.random.default_rng(11)
    zs = np.concatenate(
        [
            reference_agreement(
                rng.standard_t(3.0, size=(4, 1000, 5)), rng.standard_t(3.0, size=(10, 1000, 5)), REF_PROBS
            ).z.ravel()
            for _ in range(100)
        ]
    )
    assert 0.8 < float(np.mean(zs**2)) < 1.25
    assert abs(float(np.mean(zs))) < 0.1


@pytest.mark.unit
def test_reference_agreement_flags_tails_that_are_too_light() -> None:
    """A run whose tails are too light (scaled by 0.7) is flagged far beyond Monte Carlo noise."""
    rng = np.random.default_rng(12)
    res = reference_agreement(0.7 * rng.standard_t(3.0, size=(4, 1000, 5)), rng.standard_t(3.0, size=(10, 1000, 5)))
    assert res.summary()["ref_tail_prob_abs_z"] > 5.0


@pytest.mark.unit
def test_reference_agreement_degenerate_indicators_use_the_binomial_floor() -> None:
    """A run with no draw below a reference quantile still gets a finite z, counted as degenerate."""
    rng = np.random.default_rng(13)
    ref = rng.normal(size=(10, 1000, 2))
    x = np.abs(rng.normal(size=(4, 500, 2))) + 5.0
    res = reference_agreement(x, ref, (0.01, 0.99))
    assert np.all(np.isfinite(res.z))
    assert res.summary()["ref_tail_prob_n_degenerate"] == 2 * 2
    p_ref = (ref < res.thresholds[0]).mean(axis=(0, 1))
    floor = np.sqrt(0.01 * 0.99 * (1 / 2000 + 1 / 10000))
    np.testing.assert_allclose(res.z[0], (0.0 - p_ref) / floor)


@pytest.mark.unit
def test_reference_agreement_input_validation() -> None:
    rng = np.random.default_rng(14)
    with pytest.raises(ValueError, match="parameters"):
        reference_agreement(rng.normal(size=(4, 100, 3)), rng.normal(size=(10, 100, 2)))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        reference_agreement(rng.normal(size=(4, 100, 2)), rng.normal(size=(10, 100, 2)), (0.0, 0.5))
    single = reference_agreement(rng.normal(size=(400, 2)), rng.normal(size=(1000, 2)))
    assert single.z.shape == (4, 2)
