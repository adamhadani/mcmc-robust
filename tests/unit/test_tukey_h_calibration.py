"""Tests for the Tukey-h calibration (`calibrate_jax_tukey_h_transform`) and `transform_kind="tukey_h"`.

Calibration: mu = median; (R, h) by maximum likelihood in the Tukey-h (Lambert W x Gaussian)
family x = mu + R z exp(h z^2 / 2), z ~ N(0, 1); h = 0 unless a likelihood-ratio test rejects it.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from mcmc_robust.integrations import JaxTukeyHTransform
from mcmc_robust.jax_unified_nuts import (
    TUKEY_H_MAX_H,
    TUKEY_H_TEST_LEVEL,
    TukeyHCalibration,
    _tukey_h_mean_nll_and_grad,
    calibrate_jax_tukey_h_transform,
)

#: Pooled Phase 1 draws the default pipeline calibrates from: 4 chains x (75 - 37) warm-up draws (tune = 500).
N_PHASE1 = 152


def _tukey_h_draws(rng: np.random.Generator, n: int, loc: float, radius: float, h: float) -> np.ndarray:
    z = rng.standard_normal(n)
    return loc + radius * z * np.exp(0.5 * h * z**2)


def _tukey_h_logpdf_scipy(x: np.ndarray, loc: float, radius: float, h: float) -> np.ndarray:
    """Independent Tukey-h log density via scipy's Lambert W: log phi(z) - log |dx/dz|."""
    u = (x - loc) / radius
    w = np.real(special.lambertw(h * u**2)) if h > 0 else np.zeros_like(u)
    z2 = u**2 * np.exp(-w)
    return stats.norm.logpdf(np.sqrt(z2)) - np.log(radius) - 0.5 * h * z2 - np.log1p(h * z2)


@pytest.mark.unit
def test_tukey_h_likelihood_gradient_matches_finite_differences() -> None:
    """The analytic gradient of the mean negative log-likelihood in (log R, h) matches central differences."""
    rng = np.random.default_rng(0)
    u = rng.standard_t(1.5, size=300) * 2.0
    eps = 1e-6

    def nll(th: np.ndarray) -> float:
        return _tukey_h_mean_nll_and_grad(th, u)[0]

    for theta in ([0.3, 0.7], [-1.0, 2.5], [2.0, 0.05]):
        th = np.asarray(theta, dtype=np.float64)
        fd = [(nll(th + eps * e) - nll(th - eps * e)) / (2 * eps) for e in np.eye(2)]
        np.testing.assert_allclose(_tukey_h_mean_nll_and_grad(th, u)[1], fd, rtol=1e-6, atol=1e-8)
    # At the boundary h = 0 (Gaussian model, a = u / R) the gradient is (1 - a^2, (3 a^2 - a^4) / 2) per draw.
    th0 = np.array([0.3, 0.0])
    a2 = (u / np.exp(0.3)) ** 2
    expected0 = [np.mean(1.0 - a2), np.mean(0.5 * (3.0 * a2 - a2**2))]
    np.testing.assert_allclose(_tukey_h_mean_nll_and_grad(th0, u)[1], expected0, rtol=1e-12)
    assert _tukey_h_mean_nll_and_grad(th0, u)[0] == pytest.approx(np.mean(0.5 * a2) + 0.3 + 0.5 * np.log(2 * np.pi))


@pytest.mark.unit
def test_calibration_recovers_tukey_h_parameters_and_maximises_the_likelihood() -> None:
    """On large exact Tukey-h samples the fit recovers (mu, R, h), and h = 0 survives only on the Gaussian column."""
    rng = np.random.default_rng(1)
    n = 20_000
    truth = [(3.0, 2.0, 0.5), (-1.0, 0.5, 1.0), (0.0, 1.0, 0.0), (10.0, 4.0, 2.0)]
    pilot = np.column_stack([_tukey_h_draws(rng, n, *p) for p in truth])
    cal = calibrate_jax_tukey_h_transform(pilot, min_radius=1e-3, max_radius=1e3)

    assert isinstance(cal, TukeyHCalibration)
    assert isinstance(cal.transform, JaxTukeyHTransform)
    loc, radius, h = (np.array(v) for v in zip(*truth))
    np.testing.assert_allclose(cal.location, np.median(pilot, axis=0), rtol=0, atol=0)
    assert np.all(np.abs(cal.location - loc) < 0.05 * radius), cal.location
    np.testing.assert_allclose(cal.radius_mle[h > 0], radius[h > 0], rtol=0.05)
    np.testing.assert_allclose(cal.h_mle[h > 0], h[h > 0], atol=0.05)
    np.testing.assert_array_equal(cal.heavy, h > 0)
    np.testing.assert_allclose(cal.radius_gaussian, np.sqrt(np.mean((pilot - cal.location) ** 2, axis=0)), rtol=1e-12)
    # The transform carries the fitted values: the Tukey-h fit where h = 0 is rejected, the Gaussian fit elsewhere.
    np.testing.assert_allclose(cal.h, np.where(cal.heavy, cal.h_mle, 0.0), rtol=1e-12)
    np.testing.assert_allclose(cal.radii, np.where(cal.heavy, cal.radius_mle, cal.radius_gaussian), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(cal.transform.h), cal.h)
    np.testing.assert_array_equal(np.asarray(cal.transform.radii), cal.radii)
    np.testing.assert_array_equal(np.asarray(cal.transform.location), cal.location)

    # The fitted (R, h) is a maximum of the likelihood: no nearby point of a grid does better.
    for j in np.flatnonzero(h > 0):
        u = pilot[:, j] - cal.location[j]
        best = _tukey_h_mean_nll_and_grad(np.array([np.log(cal.radius_mle[j]), cal.h_mle[j]]), u)[0]
        for dr in (-0.01, 0.0, 0.01):
            for dh in (-0.01, 0.0, 0.01):
                th = np.array([np.log(cal.radius_mle[j]) + dr, cal.h_mle[j] + dh])
                assert _tukey_h_mean_nll_and_grad(th, u)[0] >= best - 1e-12


@pytest.mark.unit
def test_calibration_likelihood_ratio_test_definition() -> None:
    """The statistic is 2 (max log-lik of Tukey-h - max log-lik with h = 0), with the boundary null 0.5 chi2_0 + 0.5 chi2_1."""
    rng = np.random.default_rng(2)
    pilot = np.column_stack(
        [
            rng.standard_t(1.0, N_PHASE1),
            rng.standard_t(4.0, N_PHASE1),
            rng.normal(size=N_PHASE1),
            rng.laplace(size=N_PHASE1),
        ]
    )
    cal = calibrate_jax_tukey_h_transform(pilot, level=0.05)
    assert cal.level == 0.05
    for j in range(pilot.shape[1]):
        x = pilot[:, j]
        ll1 = np.sum(_tukey_h_logpdf_scipy(x, cal.location[j], cal.radius_mle[j], cal.h_mle[j]))
        ll0 = np.sum(stats.norm.logpdf(x, loc=cal.location[j], scale=cal.radius_gaussian[j]))
        np.testing.assert_allclose(cal.lr_statistic[j], max(2.0 * (ll1 - ll0), 0.0), rtol=1e-8, atol=1e-8)
    assert np.all(cal.lr_statistic >= 0.0)
    expected_p = np.where(cal.lr_statistic > 0.0, 0.5 * stats.chi2.sf(cal.lr_statistic, df=1), 1.0)
    np.testing.assert_allclose(cal.p_value, expected_p, rtol=1e-12)
    np.testing.assert_array_equal(cal.heavy, cal.p_value < 0.05)
    assert cal.heavy[0] and not cal.heavy[2]


@pytest.mark.unit
def test_calibration_clips_mask_validation_and_degenerate_columns() -> None:
    """R is clipped with h rescaled so that h / R^2 (the map's shape) is unchanged; h_mle <= max_h; masks, constants and non-finite draws."""
    rng = np.random.default_rng(3)
    cauchy = rng.standard_cauchy(N_PHASE1)
    pilot = np.column_stack(
        [1e-4 * cauchy, 1e4 * cauchy, np.full(N_PHASE1, 7.5), cauchy, rng.standard_t(0.3, N_PHASE1)]
    )
    cal = calibrate_jax_tukey_h_transform(pilot, min_radius=0.1, max_radius=100.0, max_h=1.5)

    assert cal.heavy[0] and cal.heavy[1] and cal.heavy[3] and cal.heavy[4]
    np.testing.assert_allclose(cal.radii[:3], [0.1, 100.0, 0.1], rtol=1e-12)
    np.testing.assert_allclose(cal.radii, np.clip(np.where(cal.heavy, cal.radius_mle, 0.0), 0.1, 100.0), rtol=1e-12)
    np.testing.assert_allclose(
        cal.h / cal.radii**2, np.where(cal.heavy, cal.h_mle / cal.radius_mle**2, 0.0), rtol=1e-12
    )
    # The scaled Cauchy columns give the same standardised fit as the unscaled one.
    np.testing.assert_allclose(cal.h_mle[:2], cal.h_mle[3], rtol=1e-5)
    np.testing.assert_allclose(cal.radius_mle[:2] / [1e-4, 1e4], cal.radius_mle[3], rtol=1e-5)
    # The constant column: no test, h = 0, R at the lower clip.
    assert not cal.heavy[2] and cal.h[2] == 0.0 and cal.p_value[2] == 1.0 and cal.lr_statistic[2] == 0.0
    # t with 0.3 degrees of freedom wants h near 3; the fit stops at max_h.
    assert cal.h_mle[4] == pytest.approx(1.5)
    assert np.isnan(cal.h_mle[2]) and np.all(np.delete(cal.h_mle, 2) <= 1.5)

    # A masked-out coordinate is never warped, whatever its tails.
    mask = np.array([True, True, True, False, True])
    cal_m = calibrate_jax_tukey_h_transform(pilot, mask=mask, min_radius=0.1, max_radius=100.0)
    assert not cal_m.heavy[3] and cal_m.h[3] == 0.0
    np.testing.assert_array_equal(np.asarray(cal_m.transform.mask), mask)
    assert float(cal_m.transform.forward(jnp.asarray(pilot[0]))[3]) == pilot[0, 3]

    # Non-finite draws are dropped per coordinate.
    with_inf = pilot.copy()
    with_inf[5, 3] = np.inf
    with_inf[9, 3] = np.nan
    keep = np.isfinite(with_inf[:, 3])
    cal_inf = calibrate_jax_tukey_h_transform(with_inf[:, 3:4], min_radius=0.1, max_radius=100.0)
    cal_ref = calibrate_jax_tukey_h_transform(with_inf[keep, 3:4], min_radius=0.1, max_radius=100.0)
    np.testing.assert_array_equal(cal_inf.h, cal_ref.h)
    np.testing.assert_array_equal(cal_inf.radii, cal_ref.radii)

    # (chains, draws, d) input is pooled over chains.
    stacked = calibrate_jax_tukey_h_transform(pilot.reshape(4, N_PHASE1 // 4, 5), min_radius=0.1, max_radius=100.0)
    np.testing.assert_array_equal(stacked.h, calibrate_jax_tukey_h_transform(pilot, min_radius=0.1, max_radius=100.0).h)

    for bad in (
        {"level": 0.0},
        {"level": 1.0},
        {"max_h": 0.0},
        {"min_radius": 0.0},
        {"min_radius": 5.0, "max_radius": 1.0},
    ):
        with pytest.raises(ValueError):
            calibrate_jax_tukey_h_transform(pilot, **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="d="):
        calibrate_jax_tukey_h_transform(pilot, d=3)


@pytest.mark.unit
def test_calibration_detection_rates_on_exact_iid_draws() -> None:
    """At the default level and the default pipeline's Phase 1 sample size, t_1 and t_2 get h > 0 and Gaussians h = 0.

    Each column is an independent replicate of N_PHASE1 exact draws (fixed seed, so the rates are reproducible).
    """
    reps = 400
    rng = np.random.default_rng(4)
    rates = {}
    for name, draws in (
        ("t1", rng.standard_t(1.0, size=(N_PHASE1, reps))),
        ("t2", rng.standard_t(2.0, size=(N_PHASE1, reps))),
        ("gauss", 3.0 + 0.2 * rng.standard_normal(size=(N_PHASE1, reps))),
    ):
        cal = calibrate_jax_tukey_h_transform(draws, min_radius=1e-3, max_radius=1e3)
        assert cal.level == TUKEY_H_TEST_LEVEL
        assert np.array_equal(cal.heavy, cal.h > 0.0)
        rates[name] = float(np.mean(cal.heavy))
    assert rates["t1"] >= 0.95, rates
    assert rates["t2"] >= 0.95, rates
    assert 1.0 - rates["gauss"] >= 0.95, rates
    assert TUKEY_H_MAX_H > 0.0
