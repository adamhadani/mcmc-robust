"""Unit tests for pure-JAX 3-phase adaptive UnifiedRobustNUTS (sample_blackjax_unified_nuts)."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from xarray import DataTree

from mcmc_robust.diagnostics.convergence import compute_split_rhat
from mcmc_robust.diagnostics.ess import compute_bulk_ess
from mcmc_robust.integrations import (
    JaxArcsinhTransform,
    JaxUnifiedNUTSResult,
    calibrate_jax_arcsinh_transform,
    sample_blackjax_unified_nuts,
)
from mcmc_robust.jax_unified_nuts import arviz_from_samples


# =============================================================================
# 1. Scale & Location Calibration Tests (calibrate_jax_arcsinh_transform)
# =============================================================================


@pytest.mark.unit
def test_calibrate_jax_arcsinh_transform_location_and_heterogeneous_scales() -> None:
    """Verify calibration recovers off-center location mu=[50, -30, 15] and scales sigma=[0.5, 5, 25]."""
    rng = np.random.default_rng(12345)
    mu_true = np.array([50.0, -30.0, 15.0], dtype=np.float64)
    sigma_true = np.array([0.5, 5.0, 25.0], dtype=np.float64)
    pilot_samples = rng.normal(loc=mu_true, scale=sigma_true, size=(4000, 3))

    scale_mult = 1.5
    transform = calibrate_jax_arcsinh_transform(
        pilot_samples=pilot_samples,
        d=3,
        min_radius=0.1,
        max_radius=100.0,
        scale_multiplier=scale_mult,
    )

    assert isinstance(transform, JaxArcsinhTransform)
    assert transform.d == 3
    assert transform.radii.shape == (3,)
    assert transform.location.shape == (3,)
    assert transform.mask.shape == (3,)
    assert bool(jnp.all(transform.mask))

    # Location recovery within sampling tolerance
    np.testing.assert_allclose(np.asarray(transform.location), mu_true, atol=0.8)

    # Scale recovery: expected R_i ~ scale_multiplier * sigma_i = [0.75, 7.5, 37.5]
    expected_radii = scale_mult * sigma_true
    np.testing.assert_allclose(np.asarray(transform.radii), expected_radii, rtol=0.15)

    # Verify exact roundtrip invertibility on arbitrary test points
    x_test = jnp.array([50.0, -100.0, 250.0], dtype=jnp.float64)
    z_test = transform.forward(x_test)
    x_rec = transform.inverse(z_test)
    assert jnp.allclose(x_test, x_rec, atol=1e-12)


@pytest.mark.unit
def test_calibrate_jax_arcsinh_transform_bounds_and_zero_variance() -> None:
    """Verify [min_radius, max_radius] clipping holds even on constant zero-variance and extreme inputs."""
    # Coordinate 0: constant (zero variance -> clipped to min_radius)
    # Coordinate 1: moderate dispersion
    # Coordinate 2: huge dispersion (-> clipped to max_radius)
    rng = np.random.default_rng(99)
    n = 500
    col0 = np.full(n, 42.0, dtype=np.float64)
    col1 = rng.normal(loc=-7.0, scale=2.0, size=n)
    col2 = rng.normal(loc=100.0, scale=500.0, size=n)
    samples = np.column_stack([col0, col1, col2])

    min_r = 0.25
    max_r = 30.0
    mask = np.array([True, True, False])
    transform = calibrate_jax_arcsinh_transform(
        pilot_samples=samples,
        d=3,
        mask=mask,
        min_radius=min_r,
        max_radius=max_r,
        scale_multiplier=1.5,
    )

    radii_np = np.asarray(transform.radii)
    loc_np = np.asarray(transform.location)

    assert np.all(radii_np >= min_r)
    assert np.all(radii_np <= max_r)
    assert np.isclose(radii_np[0], min_r, atol=1e-12)
    assert np.isclose(radii_np[2], max_r, atol=1e-12)
    assert np.isclose(loc_np[0], 42.0, atol=1e-12)
    assert np.array_equal(np.asarray(transform.mask), mask)

    # Untransformed coordinate (mask[2] == False) should be identity under forward/inverse
    x_pt = jnp.array([42.0, -7.0, 123.456], dtype=jnp.float64)
    z_pt = transform.forward(x_pt)
    assert jnp.isclose(z_pt[2], 123.456, atol=1e-12)
    assert jnp.allclose(transform.inverse(z_pt), x_pt, atol=1e-12)

    # Invalid radius bounds must raise ValueError
    with pytest.raises(ValueError, match="min_radius"):
        calibrate_jax_arcsinh_transform(samples, d=3, min_radius=0.0, max_radius=10.0)
    with pytest.raises(ValueError, match="min_radius"):
        calibrate_jax_arcsinh_transform(samples, d=3, min_radius=20.0, max_radius=10.0)


# =============================================================================
# 2. Off-Center Shifted Anisotropic Gaussian Target
# =============================================================================


@pytest.mark.unit
def test_sample_blackjax_unified_nuts_shifted_anisotropic_gaussian() -> None:
    """Verify R-hat < 1.05, posterior mean recovery, and location calibration on shifted Gaussian."""
    mu_true = jnp.array([25.0, -15.0, 10.0, -5.0], dtype=jnp.float64)
    sigma_true = jnp.array([0.5, 2.0, 4.0, 1.2], dtype=jnp.float64)

    def shifted_gaussian_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(((x - mu_true) / sigma_true) ** 2)

    res = sample_blackjax_unified_nuts(
        logdensity_fn=shifted_gaussian_logdensity,
        d=4,
        draws=400,
        tune=300,
        chains=4,
        seed=42,
    )

    assert isinstance(res, JaxUnifiedNUTSResult)
    assert res.samples.shape == (4, 400, 4)
    assert res.z_samples.shape == (4, 400, 4)
    assert np.all(np.isfinite(res.samples))

    # Convergence check: split R-hat < 1.05 across all coordinates
    max_rhat = float(compute_split_rhat(res.samples, reduction="max"))
    assert max_rhat < 1.05, f"Expected R-hat < 1.05 on shifted Gaussian, got {max_rhat:.4f}"

    # Posterior mean recovery
    post_mean = np.mean(res.samples, axis=(0, 1))
    np.testing.assert_allclose(post_mean, np.asarray(mu_true), atol=0.6)

    # Posterior std recovery
    post_std = np.std(res.samples, axis=(0, 1))
    np.testing.assert_allclose(post_std, np.asarray(sigma_true), rtol=0.20)

    # Phase 1 calibration must center transform.location near mu_true
    np.testing.assert_allclose(np.asarray(res.transform.location), np.asarray(mu_true), atol=3.5)
    assert res.divergences == 0
    assert 0.50 <= res.acceptance_rate <= 0.99


# =============================================================================
# 3. Heavy-Tailed Multivariate Cauchy (nu=1, d=6)
# =============================================================================


@pytest.mark.unit
def test_sample_blackjax_unified_nuts_multivariate_cauchy() -> None:
    """Verify R-hat < 1.05, zero divergences, and high ESS/grad on Multivariate Cauchy (nu=1, d=6)."""
    d = 6
    nu = 1.0

    def multivariate_cauchy_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * (nu + d) * jnp.log1p(jnp.sum(x**2) / nu)

    res = sample_blackjax_unified_nuts(
        logdensity_fn=multivariate_cauchy_logdensity,
        d=d,
        draws=500,
        tune=300,
        chains=4,
        seed=101,
    )

    assert res.samples.shape == (4, 500, d)
    assert np.all(np.isfinite(res.samples))

    # Zero post-warmup divergences
    assert res.divergences == 0, f"Expected 0 divergences on Cauchy, got {res.divergences}"

    # Split R-hat < 1.05
    max_rhat = float(compute_split_rhat(res.samples, reduction="max"))
    assert max_rhat < 1.05, f"Expected R-hat < 1.05 on Multivariate Cauchy, got {max_rhat:.4f}"

    # High ESS/grad (no tree-depth blowup in arcsinh-warped coordinates)
    mean_ess = float(compute_bulk_ess(res.samples, reduction="mean"))
    ess_per_grad = mean_ess / max(1, res.sampling_grad_evals)
    assert mean_ess > 250.0, f"Expected Bulk ESS > 250 on Cauchy, got {mean_ess:.1f}"
    assert ess_per_grad > 0.015, f"Expected ESS/grad > 0.015 on Cauchy, got {ess_per_grad:.4f}"


# =============================================================================
# 4. Centered Neal's Funnel (d=6)
# =============================================================================


@pytest.mark.unit
def test_sample_blackjax_unified_nuts_centered_neals_funnel() -> None:
    """Verify zero post-warmup divergences and deep funnel neck exploration on Neal's Funnel (d=6)."""
    d = 6
    sigma_v = 3.0

    def neals_funnel_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        v = x[0]
        rest = x[1:]
        lp_v = -0.5 * (v / sigma_v) ** 2
        lp_det = -0.5 * (d - 1) * v
        lp_quad = -0.5 * jnp.exp(-jnp.clip(v, -50.0, 50.0)) * jnp.sum(rest**2)
        return lp_v + lp_det + lp_quad

    res = sample_blackjax_unified_nuts(
        logdensity_fn=neals_funnel_logdensity,
        d=d,
        draws=500,
        tune=300,
        chains=4,
        seed=202,
    )

    assert res.samples.shape == (4, 500, d)
    assert np.all(np.isfinite(res.samples))

    # Zero post-warmup divergences
    assert res.divergences == 0, f"Expected 0 divergences on Neal's Funnel, got {res.divergences}"

    # Verify funnel neck exploration (v = x[..., 0] ~ N(0, 9), so min(v) < -3.5 and q05(v) < -2.5)
    v_draws = res.samples[:, :, 0]
    min_v = float(np.min(v_draws))
    q05_v = float(np.quantile(v_draws, 0.05))
    assert min_v < -3.5, f"Failed to explore funnel neck: min(v)={min_v:.2f}"
    assert q05_v < -2.5, f"Insufficient funnel neck mass: q05(v)={q05_v:.2f}"


# =============================================================================
# 5. Exact Gradient Accounting & ArviZ DataTree Output Contracts
# =============================================================================


@pytest.mark.unit
def test_sample_blackjax_unified_nuts_grad_accounting_and_arviz_datatree() -> None:
    """Verify exact gradient accounting across phases and ArviZ DataTree structure."""
    d = 3

    def iso_gauss_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(x**2)

    # 1. With 3-phase adaptation enabled (adapt_tail_scale=True, tune >= 40)
    res = sample_blackjax_unified_nuts(
        logdensity_fn=iso_gauss_logdensity,
        d=d,
        draws=100,
        tune=80,
        chains=2,
        seed=77,
        adapt_tail_scale=True,
    )

    assert res.phase1_grad_evals > 0
    assert res.phase2_grad_evals > 0
    assert res.sampling_grad_evals > 0
    assert res.grad_evals == res.phase1_grad_evals + res.phase2_grad_evals + res.sampling_grad_evals
    assert res.step_size > 0.0
    assert res.inverse_mass_matrix.shape == (d,)
    assert np.all(res.inverse_mass_matrix > 0.0)

    # Verify ArviZ DataTree output
    assert isinstance(res.idata, DataTree)
    assert hasattr(res.idata, "posterior")
    assert "x" in res.idata.posterior
    assert res.idata.posterior["x"].shape == (2, 100, d)

    # Verify standalone arviz_from_samples helper
    dt_direct = arviz_from_samples(res.samples)
    assert isinstance(dt_direct, DataTree)
    assert dt_direct.posterior["x"].shape == (2, 100, d)

    # 2. With adapt_tail_scale=False (skips Phase 1, phase1_grad_evals == 0)
    res_no_p1 = sample_blackjax_unified_nuts(
        logdensity_fn=iso_gauss_logdensity,
        d=d,
        draws=60,
        tune=50,
        chains=2,
        seed=88,
        adapt_tail_scale=False,
    )
    assert res_no_p1.phase1_grad_evals == 0
    assert res_no_p1.phase2_grad_evals > 0
    assert res_no_p1.sampling_grad_evals > 0
    assert res_no_p1.grad_evals == res_no_p1.phase2_grad_evals + res_no_p1.sampling_grad_evals
