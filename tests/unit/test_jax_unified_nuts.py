"""Unit tests for pure-JAX 3-phase adaptive UnifiedRobustNUTS and Vanilla-vs-JAX algorithmic parity."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import blackjax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from xarray import DataTree

from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.diagnostics.convergence import compute_split_rhat
from mcmc_robust.diagnostics.ess import compute_bulk_ess
from mcmc_robust.integrations import JaxArcsinhTransform
from mcmc_robust.jax_unified_nuts import (
    JaxUnifiedNUTSResult,
    arviz_from_samples,
    calibrate_jax_arcsinh_transform,
    sample_blackjax_unified_nuts,
)
from mcmc_robust.samplers.novel.nuts import UnifiedRobustNUTS
from mcmc_robust.targets.benign import DiagonalGaussian
from mcmc_robust.targets.heavy_tailed import MultivariateStudentTTarget
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform

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
    # Arcsinh warping stiffens light (Gaussian) tails in z-space, so rare divergences can occur.
    assert res.divergences <= 0.01 * res.samples.shape[0] * res.samples.shape[1]
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

    # A sampling test. Elliptical heavy tails are shared across coordinates, which the tail-alignment check
    # flags (it warned on seeds 0-9 of this budget), so the check is off here; test_tail_alignment.py tests it.
    res = sample_blackjax_unified_nuts(
        logdensity_fn=multivariate_cauchy_logdensity,
        d=d,
        draws=500,
        tune=300,
        chains=4,
        seed=101,
        check_tail_alignment=False,
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
    """Verify funnel neck exploration and honest divergence reporting on Neal's Funnel (d=6).

    NUTS on the centered funnel diverges in the neck; the count must be reported, not suppressed.
    """
    d = 6
    sigma_v = 3.0

    def neals_funnel_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        v = x[0]
        rest = x[1:]
        lp_v = -0.5 * (v / sigma_v) ** 2
        lp_det = -0.5 * (d - 1) * v
        lp_quad = -0.5 * jnp.exp(-jnp.clip(v, -50.0, 50.0)) * jnp.sum(rest**2)
        return lp_v + lp_det + lp_quad

    # A sampling test. The funnel's heavy tails are shared across coordinates through v, which the
    # tail-alignment check flags (it warned on seeds 0-9 of this budget), so the check is off here;
    # test_tail_alignment.py tests it.
    res = sample_blackjax_unified_nuts(
        logdensity_fn=neals_funnel_logdensity,
        d=d,
        draws=500,
        tune=300,
        chains=4,
        seed=202,
        check_tail_alignment=False,
    )

    assert res.samples.shape == (4, 500, d)
    assert np.all(np.isfinite(res.samples))

    # The reported count must equal the per-draw divergence flags.
    assert res.divergences == int(np.sum(res.idata.sample_stats["diverging"].values))

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

    # 1. With 3-phase adaptation enabled (adapt_tail_scale=True, tune >= 20)
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

    # Each chain samples with, and reports, its own adapted parameters.
    assert res.step_size.shape == (2,)
    assert np.all(res.step_size > 0.0)
    assert res.inverse_mass_matrix.shape == (2, d)
    assert np.all(res.inverse_mass_matrix > 0.0)
    np.testing.assert_array_equal(res.idata.sample_stats["step_size"].values[:, 0], res.step_size)
    assert np.all(res.idata.sample_stats["step_size"].values == res.step_size[:, np.newaxis])

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


# =============================================================================
# 6. Vanilla-vs-JAX Algorithmic Parity Tests
# =============================================================================


@pytest.mark.unit
def test_vanilla_vs_jax_transform_pointwise_and_pullback_grad_parity() -> None:
    """Verify JaxArcsinhTransform matches CoordinateWiseArcsinhTransform to < 1e-13 across bulk and tails."""
    d = 4
    radii = np.array([0.8, 2.5, 12.0, 4.0], dtype=np.float64)
    location = np.array([15.0, -20.0, 5.0, 0.0], dtype=np.float64)
    mask = np.array([True, True, False, True], dtype=bool)

    vanilla_tf = CoordinateWiseArcsinhTransform(d=d, radii=radii, location=location, mask=mask)
    jax_tf = JaxArcsinhTransform(radii=radii, location=location, mask=mask, d=d)

    # Student-t log-density for testing analytical pullback vs JAX autodiff
    nu = 2.5

    def student_t_logp_jax(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * (nu + d) * jnp.log1p(jnp.sum(((x - jnp.asarray(location)) / jnp.asarray(radii)) ** 2) / nu)

    wrapped_logp_jax = jax_tf.wrap_logdensity(student_t_logp_jax)
    autodiff_grad_z_fn = jax.grad(wrapped_logp_jax)
    autodiff_grad_x_fn = jax.grad(student_t_logp_jax)

    test_points = [
        np.array([15.0, -20.0, 5.0, 0.0], dtype=np.float64),  # exact center
        np.array([16.2, -18.5, 7.3, -1.1], dtype=np.float64),  # bulk
        np.array([250.0, -500.0, 100.0, 800.0], dtype=np.float64),  # far tail
        np.array([-1e5, 2e5, -3e4, -5e5], dtype=np.float64),  # extreme tail
    ]

    for x_np in test_points:
        x_jax = jnp.asarray(x_np, dtype=jnp.float64)

        # 1. Forward map z = phi(x)
        z_vanilla, log_det_fwd_vanilla = vanilla_tf.forward(x_np)
        z_jax = jax_tf.forward(x_jax)
        np.testing.assert_allclose(np.asarray(z_jax), z_vanilla, atol=1e-13, rtol=1e-13)

        # 2. Inverse map x = psi(z) and log |det J_psi(z)|
        x_rec_vanilla, log_det_inv_vanilla = vanilla_tf.inverse(z_vanilla)
        x_rec_jax = jax_tf.inverse(z_jax)
        log_det_inv_jax = float(jax_tf.log_jacobian_det(z_jax))
        np.testing.assert_allclose(np.asarray(x_rec_jax), x_rec_vanilla, atol=1e-12, rtol=1e-12)
        assert abs(log_det_inv_jax - log_det_inv_vanilla) < 1e-13
        assert abs(log_det_fwd_vanilla + log_det_inv_jax) < 1e-12

        # 3. Pullback gradient nabla_z log pi_Z(z) parity: vanilla vs JaxArcsinhTransform.forward_grad vs jax.grad
        grad_x_np = np.asarray(autodiff_grad_x_fn(x_jax), dtype=np.float64)
        grad_z_vanilla = vanilla_tf.forward_grad(x_np, grad_x_np)
        grad_z_jax_method = np.asarray(jax_tf.forward_grad(x_jax, grad_x_np), dtype=np.float64)
        grad_z_autodiff = np.asarray(autodiff_grad_z_fn(z_jax), dtype=np.float64)

        np.testing.assert_allclose(grad_z_jax_method, grad_z_vanilla, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(grad_z_autodiff, grad_z_vanilla, atol=1e-12, rtol=1e-12)


@pytest.mark.unit
def test_vanilla_vs_jax_phase1_calibration_exact_parity() -> None:
    """Verify calibrate_jax_arcsinh_transform matches CoordinateWiseArcsinhTransform & UnifiedRobustNUTS Phase 1."""
    rng = np.random.default_rng(2026)
    d = 4
    # Include off-center shifted coords, heavy-tailed coords, and a constant zero-variance coord
    s0 = rng.normal(loc=35.0, scale=0.8, size=120)
    s1 = -12.0 + 4.5 * rng.standard_cauchy(size=120)
    s2 = np.full(120, 7.25, dtype=np.float64)  # zero variance -> fallback & min_radius clip
    s3 = rng.normal(loc=100.0, scale=250.0, size=120)  # huge variance -> max_radius clip
    pilot_samples = np.column_stack([s0, s1, s2, s3])

    min_r = 0.15
    max_r = 80.0
    mult = 1.5

    # 1. Shared static helper on CoordinateWiseArcsinhTransform
    loc_vanilla, radii_vanilla = CoordinateWiseArcsinhTransform.compute_phase1_calibration(
        pilot_samples,
        min_radius=min_r,
        max_radius=max_r,
        scale_multiplier=mult,
    )

    # 2. Pure-JAX calibration function
    jax_tf = calibrate_jax_arcsinh_transform(
        pilot_samples=jnp.asarray(pilot_samples),
        d=d,
        min_radius=min_r,
        max_radius=max_r,
        scale_multiplier=mult,
    )

    np.testing.assert_allclose(np.asarray(jax_tf.location), loc_vanilla, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(jax_tf.radii), radii_vanilla, atol=1e-12, rtol=1e-12)

    # 3. Verify UnifiedRobustNUTS.adapt in Phase 1 produces the exact same transform
    nuts = UnifiedRobustNUTS(
        step_size=0.1,
        adapt_tail_scale=True,
        warmup_steps=800,  # W1 = max(10, int(0.15 * 800)) = 120 == len(pilot_samples)
        min_radius=min_r,
        max_radius=max_r,
    )
    for idx in range(120):
        dummy_state = ChainState(x=pilot_samples[idx], log_prob=0.0, grad_log_prob=np.zeros(d))
        dummy_res = StepResult(state=dummy_state, accepted=True, accept_prob=0.8, stats={})
        nuts.adapt(idx, dummy_res)

    assert nuts.transform is not None
    np.testing.assert_allclose(nuts.transform.location, np.asarray(jax_tf.location), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(nuts.transform.radii, np.asarray(jax_tf.radii), atol=1e-12, rtol=1e-12)


@pytest.mark.unit
def test_vanilla_vs_jax_single_leapfrog_step_parity() -> None:
    """Verify a single leapfrog step in z-space matches between UnifiedRobustNUTS and BlackJAX velocity_verlet."""
    d = 3
    target = MultivariateStudentTTarget(d=d, nu=2.5)
    radii = np.array([1.2, 2.8, 0.7], dtype=np.float64)
    location = np.array([3.0, -4.0, 1.5], dtype=np.float64)
    s_z = np.array([0.9, 1.4, 1.1], dtype=np.float64)
    epsilon = 0.08

    # Vanilla UnifiedRobustNUTS with gamma=0.0 (pure symplectic leapfrog in warped space)
    nuts = UnifiedRobustNUTS(step_size=epsilon, gamma=0.0, radius=radii)
    nuts._transform = CoordinateWiseArcsinhTransform(d=d, radii=radii, location=location)
    nuts.set_preconditioner(s_z)
    transform_vanilla = nuts._ensure_transform(d)

    # Pure-JAX transform & wrapped Student-t logdensity
    jax_tf = JaxArcsinhTransform(radii=radii, location=location, d=d)

    def student_t_logp_jax(x: jnp.ndarray) -> jnp.ndarray:
        diff = x - jnp.asarray(target.mu, dtype=jnp.float64)
        inv_cov = jnp.asarray(np.linalg.inv(target._cov), dtype=jnp.float64)
        quad = jnp.dot(diff, jnp.dot(inv_cov, diff))
        return jnp.asarray(target._log_const, dtype=jnp.float64) - 0.5 * (target.nu + d) * jnp.log1p(quad / target.nu)

    wrapped_logp_jax = jax_tf.wrap_logdensity(student_t_logp_jax)

    # Initial phase-space state (z0, p0)
    x0 = np.array([4.2, -2.5, 0.8], dtype=np.float64)
    p0 = np.array([0.6, -0.9, 0.35], dtype=np.float64)
    z0, _ = transform_vanilla.forward(x0)
    _, log_det_inv0 = transform_vanilla.inverse(z0)
    lp_x0, grad_x0 = target.log_prob_and_grad(x0)
    lp_z0 = float(lp_x0 + log_det_inv0)
    grad_z0 = transform_vanilla.forward_grad(x0, grad_x0)
    h0 = -lp_z0 + nuts._kinetic_energy(p0, s_z)

    # 1. Vanilla single leapfrog step via UnifiedRobustNUTS._build_tree(j=0, v=1)
    z_next_v, p_next_v, grad_z_next_v, _, _, _, cand_v, _, _, _, _, _, _ = nuts._build_tree(
        z=z0,
        p=p0,
        g_z=grad_z0,
        v=1,
        j=0,
        epsilon=epsilon,
        h0=h0,
        s_z=s_z,
        transform=transform_vanilla,
        target=target,
        rng=np.random.default_rng(0),
    )

    # 2. BlackJAX velocity_verlet single step with inverse_mass_matrix = s_z^2
    inv_mass_jax = jnp.asarray(s_z**2, dtype=jnp.float64)
    kinetic_energy_fn = lambda p: 0.5 * jnp.sum(inv_mass_jax * (p**2))  # noqa: E731
    vv_step = blackjax.mcmc.integrators.velocity_verlet(wrapped_logp_jax, kinetic_energy_fn)
    init_vv_state = blackjax.mcmc.integrators.IntegratorState(
        position=jnp.asarray(z0, dtype=jnp.float64),
        momentum=jnp.asarray(p0, dtype=jnp.float64),
        logdensity=jnp.asarray(lp_z0, dtype=jnp.float64),
        logdensity_grad=jnp.asarray(grad_z0, dtype=jnp.float64),
    )
    next_vv_state = vv_step(init_vv_state, epsilon)
    x_next_jax = jax_tf.inverse(next_vv_state.position)

    np.testing.assert_allclose(np.asarray(next_vv_state.position), z_next_v, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(next_vv_state.momentum), p_next_v, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(next_vv_state.logdensity_grad), grad_z_next_v, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(x_next_jax), cand_v.x, atol=1e-13, rtol=1e-13)
    assert abs(float(next_vv_state.logdensity) - cand_v.lp_z) < 1e-13


@pytest.mark.unit
def test_vanilla_vs_jax_posterior_agreement() -> None:
    """Verify UnifiedRobustNUTS and sample_blackjax_unified_nuts agree on posterior moments."""
    d = 3
    target = DiagonalGaussian(d=d, condition_number=16.0)
    shift = np.array([12.0, -8.0, 5.0], dtype=np.float64)
    target.mean = shift
    stds = target.stds

    # 1. Vanilla UnifiedRobustNUTS
    vanilla_nuts = UnifiedRobustNUTS(
        step_size=0.15,
        max_depth=8,
        adapt_step_size=True,
        adapt_preconditioner=True,
        adapt_tail_scale=True,
        warmup_steps=200,
    )
    vanilla_samples, _ = vanilla_nuts.sample(
        x0=np.zeros(d),
        target=target,
        n_steps=600,
        burn_in=200,
        rng=np.random.default_rng(42),
        adapt=True,
    )

    # 2. Pure-JAX sample_blackjax_unified_nuts
    mu_jax = jnp.asarray(shift, dtype=jnp.float64)
    std_jax = jnp.asarray(stds, dtype=jnp.float64)

    def diag_gauss_logp(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(((x - mu_jax) / std_jax) ** 2)

    jax_res = sample_blackjax_unified_nuts(
        logdensity_fn=diag_gauss_logp,
        d=d,
        draws=300,
        tune=200,
        chains=2,
        seed=42,
    )

    vanilla_mean = np.mean(vanilla_samples, axis=0)
    jax_mean = np.mean(jax_res.samples, axis=(0, 1))
    vanilla_std = np.std(vanilla_samples, axis=0)
    jax_std = np.std(jax_res.samples, axis=(0, 1))

    np.testing.assert_allclose(vanilla_mean, shift, atol=0.6)
    np.testing.assert_allclose(jax_mean, shift, atol=0.6)
    np.testing.assert_allclose(vanilla_mean, jax_mean, atol=0.7)
    np.testing.assert_allclose(vanilla_std, jax_std, rtol=0.25)


# =============================================================================
# 7. Exact-Reference Distribution Tests in Moderate Dimension
# =============================================================================
#
# Low-dimensional checks cannot tell whether the kernel targets the posterior itself: a modified
# density that only differs far below the posterior's typical log-density level is invisible at
# d <= 6. These tests compare against closed-form references where that difference shows up.


@pytest.mark.unit
def test_sample_blackjax_unified_nuts_standard_gaussian_second_moment_d50() -> None:
    """E[x_i^2] = 1 on a 50-d standard Gaussian; any inflated or flattened tail breaks it."""
    d = 50

    def std_gauss_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(x**2)

    res = sample_blackjax_unified_nuts(std_gauss_logdensity, d=d, draws=500, tune=400, chains=4, seed=0)
    draws = res.samples.reshape(-1, d)

    # Across seeds the coordinate-averaged second moment stays within 0.011 of 1 at this budget.
    second_moment = float(np.mean(draws**2))
    assert abs(second_moment - 1.0) < 0.05, f"E[x^2] = {second_moment:.3f}, expected 1"
    assert float(np.max(np.abs(np.mean(draws, axis=0)))) < 0.2


@pytest.mark.unit
def test_sample_blackjax_unified_nuts_student_t_radial_quantiles_d20() -> None:
    """Radius quantiles of a 20-d multivariate Student-t (nu = 3) match the exact F law."""
    d, nu = 20, 3.0

    def mvt_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * (nu + d) * jnp.log1p(jnp.sum(x**2) / nu)

    # A sampling test. Elliptical heavy tails are shared across coordinates, which the tail-alignment check
    # flags (on seeds 0-9 of this budget, though close to its threshold), so the check is off here;
    # test_tail_alignment.py tests it.
    res = sample_blackjax_unified_nuts(
        mvt_logdensity, d=d, draws=500, tune=400, chains=4, seed=1, check_tail_alignment=False
    )
    radius = np.linalg.norm(res.samples.reshape(-1, d), axis=1)

    # For an identity-scale multivariate t, ||x||^2 / d ~ F(d, nu). Across seeds the sampled
    # median and 90th percentile stay within 8% and 12% of the exact values at this budget.
    for prob, rel_tol in ((0.5, 0.15), (0.9, 0.25)):
        exact = float(np.sqrt(d * stats.f.ppf(prob, d, nu)))
        sampled = float(np.quantile(radius, prob))
        assert abs(sampled / exact - 1.0) < rel_tol, f"q{prob:.2f}(|x|) = {sampled:.2f}, exact {exact:.2f}"
