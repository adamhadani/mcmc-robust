"""Tests for the Tukey-h calibration (`calibrate_jax_tukey_h_transform`) and `transform_kind="tukey_h"`.

Calibration: mu = median; h = 0 (the affine map) unless the tail probe (`probe_tail_decay`) finds both sides of
the coordinate's log density polynomial; then (R, h) by maximum likelihood in the Tukey-h (Lambert W x Gaussian)
family x = mu + R z exp(h z^2 / 2), z ~ N(0, 1).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import inspect

import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from mcmc_robust.integrations import JaxTukeyHTransform
from mcmc_robust.jax_unified_nuts import (
    TAIL_PROBE_RADII,
    TRANSFORM_KINDS,
    TukeyHCalibration,
    _tukey_h_mean_nll_and_grad,
    calibrate_jax_tukey_h_transform,
    probe_tail_decay,
    sample_blackjax_unified_nuts,
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


def _tukey_h_product_logdensity(loc: np.ndarray, radius: np.ndarray, h: np.ndarray):  # type: ignore[no-untyped-def]
    """JAX log density of independent Tukey-h coordinates: log phi(z) - log |dx/dz| with z = forward(x)."""
    tr = JaxTukeyHTransform(radii=radius, location=loc, h=h, d=len(loc))

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        z = tr.forward(x)
        return -0.5 * jnp.sum(z**2) - tr.log_jacobian_det(z)

    return logdensity


def _scaled_t_logdensity(df: np.ndarray, loc: np.ndarray, scale: np.ndarray):  # type: ignore[no-untyped-def]
    """JAX log density of independent location-scale coordinates: t_df where df > 0, Gaussian where df == 0."""
    is_t = jnp.asarray(df) > 0.0
    nu = jnp.where(is_t, jnp.asarray(df, dtype=jnp.float64), 1.0)

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        r = (x - loc) / scale
        return jnp.sum(jnp.where(is_t, -0.5 * (nu + 1.0) * jnp.log1p(r**2 / nu), -0.5 * r**2))

    return logdensity


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
    """On large exact Tukey-h samples the fit recovers (mu, R, h); only the Gaussian column (h = 0) stays affine."""
    rng = np.random.default_rng(1)
    n = 20_000
    truth = [(3.0, 2.0, 0.5), (-1.0, 0.5, 1.0), (0.0, 1.0, 0.0), (10.0, 4.0, 2.0)]
    pilot = np.column_stack([_tukey_h_draws(rng, n, *p) for p in truth])
    loc, radius, h = (np.array(v) for v in zip(*truth))
    cal = calibrate_jax_tukey_h_transform(
        pilot, _tukey_h_product_logdensity(loc, radius, h), min_radius=1e-3, max_radius=1e3
    )

    assert isinstance(cal, TukeyHCalibration)
    assert isinstance(cal.transform, JaxTukeyHTransform)
    np.testing.assert_allclose(cal.location, np.median(pilot, axis=0), rtol=0, atol=0)
    assert np.all(np.abs(cal.location - loc) < 0.05 * radius), cal.location
    np.testing.assert_array_equal(cal.heavy, h > 0)
    np.testing.assert_allclose(cal.radius_mle[h > 0], radius[h > 0], rtol=0.05)
    np.testing.assert_allclose(cal.h_mle[h > 0], h[h > 0], atol=0.05)
    np.testing.assert_allclose(cal.radius_gaussian, np.sqrt(np.mean((pilot - cal.location) ** 2, axis=0)), rtol=1e-12)
    # The transform carries the fitted values: the Tukey-h fit where the guard warps, the Gaussian fit elsewhere.
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
    # The fit maximises the likelihood of scipy's independent Tukey-h density too.
    for j in np.flatnonzero(h > 0):
        x = pilot[:, j]
        best = np.sum(_tukey_h_logpdf_scipy(x, cal.location[j], cal.radius_mle[j], cal.h_mle[j]))
        assert best >= np.sum(_tukey_h_logpdf_scipy(x, cal.location[j], radius[j], h[j])) - 1e-6


@pytest.mark.unit
def test_the_guard_is_the_tail_probe_at_the_pilot_median_and_the_affine_scale() -> None:
    """The probe starts at mu with unit R_0 clipped to [R_min, R_max]; a coordinate is warped iff both sides are polynomial.

    Coordinates the guard leaves affine are not fitted (radius_mle = h_mle = NaN).
    """
    rng = np.random.default_rng(2)
    df = np.array([1.0, 4.0, 0.0, 0.0, 2.0])
    loc = np.array([1.0, -2.0, 0.0, 5.0, 0.0])
    scale = np.array([1.0, 0.5, 2.0, 1e-3, 1.0])
    pilot = np.column_stack(
        [
            loc[j] + scale[j] * (rng.standard_t(df[j], N_PHASE1) if df[j] else rng.standard_normal(N_PHASE1))
            for j in range(5)
        ]
    )

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        # Coordinate 4: t_2 on the right, Gaussian on the left (one polynomial side only).
        base = _scaled_t_logdensity(df, loc, scale)(x)
        r = x[4] - loc[4]
        return base + jnp.where(r < 0.0, -0.5 * r**2 + 1.5 * jnp.log1p(r**2 / 2.0), 0.0)

    cal = calibrate_jax_tukey_h_transform(pilot, logdensity, min_radius=0.1, max_radius=100.0)
    unit = np.clip(cal.radius_gaussian, 0.1, 100.0)
    assert unit[3] == 0.1  # the 1e-3-scale Gaussian probes in units of R_min
    expected = probe_tail_decay(logdensity, cal.location, unit)
    for field in ("potential", "slope_near", "slope_far", "polynomial"):
        np.testing.assert_array_equal(getattr(cal.probe, field), getattr(expected, field))
    np.testing.assert_array_equal(cal.probe.polynomial[4], [False, True])
    np.testing.assert_array_equal(cal.heavy, cal.probe.polynomial.all(axis=-1))
    np.testing.assert_array_equal(cal.heavy, [True, True, False, False, False])
    assert np.all(np.isfinite(cal.h_mle[cal.heavy])) and np.all(np.isnan(cal.h_mle[~cal.heavy]))
    assert np.all(np.isnan(cal.radius_mle[~cal.heavy]))


@pytest.mark.unit
def test_the_decision_comes_from_the_density_not_from_what_the_pilot_draws_look_like() -> None:
    """Heavy-looking draws of a Gaussian target (a transient) are not warped; Gaussian-looking draws of a Cauchy target are."""
    rng = np.random.default_rng(3)
    heavy_looking = 3.0 + rng.standard_cauchy((N_PHASE1, 1))
    gaussian_looking = rng.standard_normal((N_PHASE1, 1))
    gauss = _scaled_t_logdensity(np.array([0.0]), np.array([3.0]), np.array([1.0]))
    cauchy = _scaled_t_logdensity(np.array([1.0]), np.array([0.0]), np.array([1.0]))

    assert not calibrate_jax_tukey_h_transform(heavy_looking, gauss).heavy[0]
    assert calibrate_jax_tukey_h_transform(gaussian_looking, cauchy).heavy[0]


@pytest.mark.unit
def test_calibration_clips_mask_validation_and_degenerate_columns() -> None:
    """R is clipped with h rescaled so that h / R^2 (the map's shape) is unchanged; h_mle <= max_h; masks, constants and non-finite draws."""
    rng = np.random.default_rng(3)
    cauchy = rng.standard_cauchy(N_PHASE1)
    pilot = np.column_stack(
        [1e-4 * cauchy, 1e4 * cauchy, np.full(N_PHASE1, 7.5), cauchy, rng.standard_t(0.3, N_PHASE1)]
    )
    logdensity = _scaled_t_logdensity(
        np.array([1.0, 1.0, 1.0, 1.0, 0.3]), np.array([0.0, 0.0, 7.5, 0.0, 0.0]), np.array([1e-4, 1e4, 1.0, 1.0, 1.0])
    )
    cal = calibrate_jax_tukey_h_transform(pilot, logdensity, min_radius=0.1, max_radius=100.0, max_h=1.5)

    assert cal.probe is not None and cal.probe.polynomial.all()
    np.testing.assert_array_equal(cal.heavy, [True, True, False, True, True])
    np.testing.assert_allclose(cal.radii[:3], [0.1, 100.0, 0.1], rtol=1e-12)
    np.testing.assert_allclose(cal.radii, np.clip(np.where(cal.heavy, cal.radius_mle, 0.0), 0.1, 100.0), rtol=1e-12)
    np.testing.assert_allclose(
        cal.h / cal.radii**2, np.where(cal.heavy, cal.h_mle / cal.radius_mle**2, 0.0), rtol=1e-12
    )
    # The scaled Cauchy columns give the same standardised fit as the unscaled one.
    np.testing.assert_allclose(cal.h_mle[:2], cal.h_mle[3], rtol=1e-5)
    np.testing.assert_allclose(cal.radius_mle[:2] / [1e-4, 1e4], cal.radius_mle[3], rtol=1e-5)
    # The constant column has polynomial tails but no spread to fit: h = 0, R at the lower clip.
    assert cal.h[2] == 0.0 and np.isnan(cal.h_mle[2]) and np.isnan(cal.radius_mle[2])
    # t with 0.3 degrees of freedom wants h near 3; the fit stops at max_h.
    assert cal.h_mle[4] == pytest.approx(1.5)
    assert np.all(np.delete(cal.h_mle, 2) <= 1.5)

    # A masked-out coordinate is never warped, whatever its tails.
    mask = np.array([True, True, True, False, True])
    cal_m = calibrate_jax_tukey_h_transform(pilot, logdensity, mask=mask, min_radius=0.1, max_radius=100.0)
    assert not cal_m.heavy[3] and cal_m.h[3] == 0.0
    np.testing.assert_array_equal(np.asarray(cal_m.transform.mask), mask)
    assert float(cal_m.transform.forward(jnp.asarray(pilot[0]))[3]) == pilot[0, 3]

    # Non-finite draws are dropped per coordinate.
    with_inf = pilot.copy()
    with_inf[5, 3] = np.inf
    with_inf[9, 3] = np.nan
    keep = np.isfinite(with_inf[:, 3])
    cauchy1 = _scaled_t_logdensity(np.array([1.0]), np.array([0.0]), np.array([1.0]))
    cal_inf = calibrate_jax_tukey_h_transform(with_inf[:, 3:4], cauchy1, min_radius=0.1, max_radius=100.0)
    cal_ref = calibrate_jax_tukey_h_transform(with_inf[keep, 3:4], cauchy1, min_radius=0.1, max_radius=100.0)
    np.testing.assert_array_equal(cal_inf.h, cal_ref.h)
    np.testing.assert_array_equal(cal_inf.radii, cal_ref.radii)

    # (chains, draws, d) input is pooled over chains.
    stacked = calibrate_jax_tukey_h_transform(
        pilot.reshape(4, N_PHASE1 // 4, 5), logdensity, min_radius=0.1, max_radius=100.0
    )
    pooled = calibrate_jax_tukey_h_transform(pilot, logdensity, min_radius=0.1, max_radius=100.0)
    np.testing.assert_array_equal(stacked.h, pooled.h)

    # warp=False skips the probe and keeps h = 0 everywhere: the affine map (the h = 0 control arm of the benchmarks).
    never = calibrate_jax_tukey_h_transform(pilot, logdensity, warp=False, min_radius=0.1, max_radius=100.0, max_h=1.5)
    assert never.probe is None
    assert not never.heavy.any() and np.all(never.h == 0.0)
    assert np.all(np.isnan(never.h_mle)) and np.all(np.isnan(never.radius_mle))
    np.testing.assert_array_equal(never.radii, np.clip(never.radius_gaussian, 0.1, 100.0))
    np.testing.assert_array_equal(never.location, cal.location)

    for bad in (
        {"max_h": 0.0},
        {"min_radius": 0.0},
        {"min_radius": 5.0, "max_radius": 1.0},
    ):
        with pytest.raises(ValueError):
            calibrate_jax_tukey_h_transform(pilot, logdensity, **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="d="):
        calibrate_jax_tukey_h_transform(pilot, logdensity, d=3)


# =============================================================================
# The pipeline: sample_blackjax_unified_nuts(..., transform_kind="tukey_h")
# =============================================================================

#: Product target: two Cauchy, a t_2 and a t_5 coordinate, and two Gaussians, at assorted locations and scales.
PRODUCT_MARGINALS = (
    stats.t(df=1.0, loc=2.0, scale=0.5),
    stats.t(df=1.0, loc=-1.0, scale=3.0),
    stats.t(df=2.0, loc=0.0, scale=1.0),
    stats.t(df=5.0, loc=4.0, scale=2.0),
    stats.norm(loc=5.0, scale=0.3),
    stats.norm(loc=0.0, scale=4.0),
)
_DF = jnp.array([1.0, 1.0, 2.0, 5.0, 0.0, 0.0])
_LOC = jnp.array([m.mean() if m.dist.name == "norm" else m.kwds["loc"] for m in PRODUCT_MARGINALS])
_SCALE = jnp.array([m.std() if m.dist.name == "norm" else m.kwds["scale"] for m in PRODUCT_MARGINALS])
_IS_T = _DF > 0.0


def _product_logdensity(x: jnp.ndarray) -> jnp.ndarray:
    r = (x - _LOC) / _SCALE
    log_t = -0.5 * (_DF + 1.0) * jnp.log1p(r**2 / jnp.where(_IS_T, _DF, 1.0))
    return jnp.sum(jnp.where(_IS_T, log_t, -0.5 * r**2))


def _tail_prob_z(x: np.ndarray, quantiles: dict[float, np.ndarray]) -> np.ndarray:
    """z-scores (P_hat - p) / MCSE of the indicator 1{x_i < q_p}, with ArviZ's MCSE."""
    zs = []
    for p, q in quantiles.items():
        ind = (x < q).astype(np.float64)
        zs.append((ind.mean(axis=(0, 1)) - p) / az.mcse(ind, method="mean"))
    return np.concatenate(zs)


@pytest.mark.unit
def test_pipeline_tukey_h_kind_hits_exact_tail_probabilities_on_a_product_t_target() -> None:
    """transform_kind="tukey_h" samples a product of t and Gaussian marginals exactly, by tail probabilities.

    Tolerances are in Monte Carlo standard errors of the indicator means: with 30 z-scores the
    maximum |z| of a correct sampler exceeds 4.5 with probability below 1e-3.
    """
    d = len(PRODUCT_MARGINALS)
    x0 = np.random.default_rng(0).uniform(-2.0, 2.0, size=(4, d))
    res = sample_blackjax_unified_nuts(
        _product_logdensity, d=d, draws=2500, tune=1000, chains=4, seed=5, x0=x0, transform_kind="tukey_h"
    )
    tr = res.transform
    assert isinstance(tr, JaxTukeyHTransform)
    assert res.phase1_grad_evals > 0
    assert res.grad_evals == res.phase1_grad_evals + res.phase2_grad_evals + res.sampling_grad_evals
    np.testing.assert_allclose(
        np.asarray(jax.vmap(jax.vmap(tr.inverse))(jnp.asarray(res.z_samples))), res.samples, rtol=1e-12
    )
    # The guard warps the Cauchy and t_2 coordinates and leaves the Gaussians affine.
    h = np.asarray(tr.h)
    assert np.all(h[:3] > 0.0), h
    assert np.all(h[4:] == 0.0), h

    z = _tail_prob_z(
        res.samples, {p: np.array([m.ppf(p) for m in PRODUCT_MARGINALS]) for p in (0.01, 0.1, 0.5, 0.9, 0.99)}
    )
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z)) < 4.5, z
    assert np.mean(np.abs(z)) < 1.5, z
    assert res.divergences == 0


@pytest.mark.unit
def test_pipeline_tukey_h_without_phase1_or_warp_is_affine_and_the_probe_is_counted() -> None:
    """Without Phase 1 there is nothing to calibrate, so h = 0; `tukey_h_warp=False` keeps h = 0 after Phase 1.

    The probe's 8 d log-density evaluations are counted in Phase 1's gradient evaluations: with the same seed, the
    two runs share Phase 1 and differ by exactly that many.
    """
    d = len(PRODUCT_MARGINALS)
    res = sample_blackjax_unified_nuts(
        _product_logdensity, d=d, draws=50, tune=60, chains=2, seed=1, transform_kind="tukey_h", adapt_tail_scale=False
    )
    assert isinstance(res.transform, JaxTukeyHTransform)
    assert res.phase1_grad_evals == 0
    np.testing.assert_array_equal(np.asarray(res.transform.h), np.zeros(d))
    np.testing.assert_array_equal(np.asarray(res.transform.radii), np.full(d, 3.0))
    np.testing.assert_array_equal(np.asarray(res.transform.location), np.zeros(d))

    common = dict(d=d, draws=50, tune=200, chains=2, seed=1, transform_kind="tukey_h")
    warped = sample_blackjax_unified_nuts(_product_logdensity, **common)  # type: ignore[arg-type]
    affine = sample_blackjax_unified_nuts(_product_logdensity, tukey_h_warp=False, **common)  # type: ignore[arg-type]
    assert np.any(np.asarray(warped.transform.h) > 0.0)
    np.testing.assert_array_equal(np.asarray(affine.transform.h), np.zeros(d))
    np.testing.assert_array_equal(np.asarray(affine.transform.location), np.asarray(warped.transform.location))
    assert warped.phase1_grad_evals - affine.phase1_grad_evals == 2 * d * len(TAIL_PROBE_RADII)


@pytest.mark.unit
def test_tukey_h_is_a_transform_kind_and_arcsinh_stays_the_default() -> None:
    """ "tukey_h" is accepted; the default transform_kind is still "arcsinh"."""
    assert "tukey_h" in TRANSFORM_KINDS
    assert inspect.signature(sample_blackjax_unified_nuts).parameters["transform_kind"].default == "arcsinh"
