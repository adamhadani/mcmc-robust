"""Tests for `JaxAffineTransform` and the `transform_kind` option of the pure-JAX NUTS pipeline.

The affine transform z = (x - mu) / R is the ablation control for the arcsinh transform: it has
the same Phase 1 calibration of (mu, R) but no tail warping.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mcmc_robust.integrations import JaxAffineTransform, JaxArcsinhTransform
from mcmc_robust.jax_unified_nuts import (
    calibrate_jax_affine_transform,
    calibrate_jax_arcsinh_transform,
    sample_blackjax_unified_nuts,
)

LOCATION = np.array([2.0, -1.5, 0.25, 7.0, -3.0], dtype=np.float64)
RADII = np.array([0.3, 1.0, 4.0, 12.5, 0.07], dtype=np.float64)
MASK = np.array([True, True, False, True, True])


def _correlated_t_logdensity(x: jnp.ndarray) -> jnp.ndarray:
    """A non-separable, heavy-tailed test density so cross-terms in the gradient matter."""
    a = jnp.array([[1.0, 0.3, 0.0, 0.1, 0.0]] * 5) + jnp.eye(5)
    r = a @ (x - 1.0)
    return -2.5 * jnp.log1p(jnp.sum(r**2) / 3.0)


@pytest.mark.unit
def test_affine_transform_roundtrip_and_interface() -> None:
    """forward/inverse are mutual inverses, unmasked coordinates pass through, and the attributes match the arcsinh API."""
    tr = JaxAffineTransform(radii=RADII, location=LOCATION, mask=MASK)
    ref = JaxArcsinhTransform(radii=RADII, location=LOCATION, mask=MASK)
    for attr in ("d", "radius", "radii", "location", "mask"):
        np.testing.assert_array_equal(np.asarray(getattr(tr, attr)), np.asarray(getattr(ref, attr)))

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_cauchy(size=5) * 50.0)
    z = tr.forward(x)
    np.testing.assert_allclose(np.asarray(tr.inverse(z)), np.asarray(x), rtol=1e-13, atol=1e-12)
    expected_z = np.where(MASK, (np.asarray(x) - LOCATION) / RADII, np.asarray(x))
    np.testing.assert_allclose(np.asarray(z), expected_z, rtol=1e-14)

    # Scalar radius with an explicit dimension broadcasts like the arcsinh transform.
    tr_scalar = JaxAffineTransform(radius=2.0, dim=3)
    assert tr_scalar.d == 3
    np.testing.assert_array_equal(np.asarray(tr_scalar.radii), np.full(3, 2.0))


@pytest.mark.unit
def test_affine_log_jacobian_matches_autodiff() -> None:
    """log|det dx/dz| equals the log-determinant of the autodiff Jacobian of `inverse`."""
    tr = JaxAffineTransform(radii=RADII, location=LOCATION, mask=MASK)
    rng = np.random.default_rng(1)
    for _ in range(5):
        z = jnp.asarray(rng.normal(size=5) * 3.0)
        jac = jax.jacfwd(tr.inverse)(z)
        sign, logabsdet = jnp.linalg.slogdet(jac)
        assert float(sign) == 1.0
        np.testing.assert_allclose(float(tr.log_jacobian_det(z)), float(logabsdet), rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_affine_pullback_gradient_matches_autodiff() -> None:
    """`forward_grad` equals jax.grad of the wrapped (pulled-back) log density."""
    tr = JaxAffineTransform(radii=RADII, location=LOCATION, mask=MASK)
    wrapped = tr.wrap_logdensity(_correlated_t_logdensity)
    grad_x_fn = jax.grad(_correlated_t_logdensity)
    grad_z_fn = jax.grad(wrapped)
    rng = np.random.default_rng(2)
    for _ in range(5):
        x = jnp.asarray(rng.normal(size=5) * 4.0)
        z = tr.forward(x)
        analytic = tr.forward_grad(x, grad_x_fn(x))
        np.testing.assert_allclose(np.asarray(analytic), np.asarray(grad_z_fn(z)), rtol=1e-11, atol=1e-12)
        # The pulled-back density is the primal density plus the constant log-Jacobian.
        np.testing.assert_allclose(
            float(wrapped(z)),
            float(_correlated_t_logdensity(x)) + float(np.sum(np.log(RADII[MASK]))),
            rtol=1e-13,
        )


@pytest.mark.unit
def test_affine_calibration_uses_the_arcsinh_phase1_rule() -> None:
    """Affine and arcsinh calibration produce identical (mu, R) from the same pilot draws."""
    rng = np.random.default_rng(3)
    pilot = rng.standard_t(1.5, size=(3, 200, 4)) * np.array([0.05, 1.0, 10.0, 500.0]) + 3.0
    aff = calibrate_jax_affine_transform(pilot, min_radius=0.1, max_radius=100.0, scale_multiplier=1.5)
    asinh = calibrate_jax_arcsinh_transform(pilot, min_radius=0.1, max_radius=100.0, scale_multiplier=1.5)
    assert isinstance(aff, JaxAffineTransform)
    np.testing.assert_array_equal(np.asarray(aff.location), np.asarray(asinh.location))
    np.testing.assert_array_equal(np.asarray(aff.radii), np.asarray(asinh.radii))


def _gaussian_moment_checks(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    """Sample mean and variance agree with the truth within 5 Monte Carlo standard errors."""
    import arviz as az

    mean_mcse = az.mcse(samples, method="mean")
    sd_mcse = az.mcse(samples, method="sd")
    est_mean = samples.mean(axis=(0, 1))
    est_sd = samples.std(axis=(0, 1))
    assert np.all(np.abs(est_mean - mean) < 5.0 * mean_mcse), (est_mean, mean, mean_mcse)
    assert np.all(np.abs(est_sd - std) < 5.0 * sd_mcse), (est_sd, std, sd_mcse)


@pytest.mark.unit
def test_pipeline_affine_kind_recovers_gaussian_moments() -> None:
    """transform_kind='affine' samples a shifted anisotropic Gaussian correctly and calibrates (mu, R)."""
    mean = np.array([25.0, -15.0, 10.0, -5.0])
    std = np.array([0.5, 2.0, 4.0, 1.2])

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(((x - mean) / std) ** 2)

    x0 = np.random.default_rng(4).uniform(-2.0, 2.0, size=(4, 4))
    res = sample_blackjax_unified_nuts(
        logdensity, d=4, draws=1000, tune=500, chains=4, seed=11, x0=x0, transform_kind="affine"
    )
    assert isinstance(res.transform, JaxAffineTransform)
    assert res.phase1_grad_evals > 0
    assert res.grad_evals == res.phase1_grad_evals + res.phase2_grad_evals + res.sampling_grad_evals
    # Phase 1 calibration: median near the mean, R near 1.5 sigma (the normal-consistent IQR scale).
    np.testing.assert_allclose(np.asarray(res.transform.location), mean, atol=2.0 * std.max())
    np.testing.assert_allclose(np.asarray(res.transform.radii), 1.5 * std, rtol=0.5)
    # Under an affine map, z-space draws are exactly the rescaled x-space draws.
    np.testing.assert_allclose(
        res.z_samples,
        (res.samples - np.asarray(res.transform.location)) / np.asarray(res.transform.radii),
        rtol=1e-10,
        atol=1e-10,
    )
    _gaussian_moment_checks(res.samples, mean, std)


@pytest.mark.unit
def test_pipeline_identity_kind_is_stock_window_adapted_nuts() -> None:
    """transform_kind='identity' skips Phase 1 and samples in x-space (stock BlackJAX NUTS)."""
    mean = np.array([3.0, -1.0, 0.5])
    std = np.array([0.2, 1.0, 5.0])

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(((x - mean) / std) ** 2)

    x0 = np.random.default_rng(5).uniform(-2.0, 2.0, size=(4, 3))
    res = sample_blackjax_unified_nuts(
        logdensity, d=3, draws=1000, tune=500, chains=4, seed=12, x0=x0, transform_kind="identity"
    )
    assert res.phase1_grad_evals == 0
    assert res.grad_evals == res.phase2_grad_evals + res.sampling_grad_evals
    np.testing.assert_array_equal(res.z_samples, res.samples)
    # The adapted inverse mass matrix is the x-space variance (identity transform).
    np.testing.assert_allclose(np.median(res.inverse_mass_matrix, axis=0), std**2, rtol=0.5)
    _gaussian_moment_checks(res.samples, mean, std)


@pytest.mark.unit
def test_pipeline_rejects_unknown_transform_kind() -> None:
    """An unknown transform_kind fails loudly instead of silently falling back."""
    with pytest.raises(ValueError, match="transform_kind"):
        sample_blackjax_unified_nuts(lambda x: -0.5 * jnp.sum(x**2), d=2, draws=5, tune=0, transform_kind="sinh")
