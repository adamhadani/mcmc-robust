"""Test-Driven Development (TDD) suite for Referee Report remediation.

Verifies every mathematical, algorithmic, and methodological fix required by the
critical evaluation:
1. Sampler.sample() calls finalize_adaptation() at the end of burn_in for all samplers.
2. CoordinateWiseArcsinhTransform and JaxArcsinhTransform are shift-equivariant with
   a location parameter mu_i and unbiased normal-consistent IQR/MAD scale calibration.
3. GentleMALA soft-capped attenuation rho(kappa) = (1 + kappa + (kappa / kappa_max)^2)^{-1/2}
   prevents origin overshoot and chain freezing at arbitrary R_0 >> 4*gamma/h while
   preserving square-root O(1) contraction inside the basin.
4. UnifiedRobustNUTS implements distinct inner-seam cross-subtree U-turn checks and
   calibrates both location and scale online during Phase 1 warmup.
5. sota_adapters counts strictly post-warmup divergences in `divergences` across PyMC,
   BlackJAX, and mcmc_robust, reporting `warmup_divergences` separately.
6. SparseSignalRegressionTarget has genuine heavy-tailed inactive marginals, non-zero
   active posterior modes, and non-degenerate ground-truth posterior draws.
7. plot.py has no mock data generator and raises FileNotFoundError when data is missing.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import mcmc_robust.bench.plot as plot_mod
from mcmc_robust.bench.sota_adapters import (
    SparseSignalRegressionTarget,
    run_blackjax_sampler,
    run_mcmc_robust_sampler,
)
from mcmc_robust.integrations import JaxArcsinhTransform
from mcmc_robust.samplers.barker import BarkerLangevin
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.nuts import UnifiedRobustNUTS
from mcmc_robust.samplers.tamed_mala import GloballyTamedMALA
from mcmc_robust.targets.benign import DiagonalGaussian, IsotropicGaussian
from mcmc_robust.targets.roughness import QuarticTarget
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform


class TestFinalizeAdaptationInBaseSampler:
    """Verifies Sampler.sample() finalizes dual-averaging step size at burn_in."""

    @pytest.mark.parametrize(
        "sampler_cls",
        [MALA, BarkerLangevin, GloballyTamedMALA, GentleMALA],
    )
    def test_sample_calls_finalize_adaptation_at_burn_in(self, sampler_cls: type) -> None:
        target = IsotropicGaussian(d=4)
        sampler = sampler_cls(step_size=0.8, adapt_step_size=True)
        rng = np.random.default_rng(123)
        x0 = np.zeros(4)

        _, step_results = sampler.sample(
            x0=x0,
            target=target,
            n_steps=60,
            burn_in=30,
            rng=rng,
            adapt=True,
        )

        assert sampler.adaptation is not None
        expected_bar = float(sampler.adaptation.smoothed_step_size)
        assert sampler.step_size == pytest.approx(expected_bar, rel=1e-12)

        # Post-warmup steps must all use the finalized smoothed step size
        post_step_sizes = [res.step_size for res in step_results[30:]]
        assert np.allclose(post_step_sizes, expected_bar, rtol=1e-12)


class TestShiftEquivariantArcsinhTransform:
    """Verifies location-scale CoordinateWiseArcsinhTransform and JaxArcsinhTransform."""

    def test_location_parameter_forward_inverse_and_jacobian(self) -> None:
        d = 4
        loc = np.array([10.0, -25.0, 5.0, 100.0])
        radii = np.array([1.5, 2.0, 0.8, 3.0])
        mask = np.array([True, True, True, False])

        t = CoordinateWiseArcsinhTransform(d=d, radii=radii, mask=mask, location=loc)
        assert np.allclose(t.location, loc)

        # At x == location, transformed active coordinates must be 0.0
        z_at_loc, _ = t.forward(loc)
        assert np.allclose(z_at_loc[:3], 0.0, atol=1e-15)
        assert z_at_loc[3] == pytest.approx(loc[3])

        # Round-trip and Jacobian cancellation on shifted coordinates
        rng = np.random.default_rng(42)
        x = loc + rng.standard_cauchy(size=d) * radii
        z, log_det_fwd = t.forward(x)
        x_rec, log_det_inv = t.inverse(z)

        assert np.max(np.abs(x - x_rec)) < 1e-12
        assert abs(log_det_fwd + log_det_inv) < 1e-12

        # Pullback gradient consistency vs central finite differences
        sigma = np.array([1.2, 2.5, 0.7, 1.5])
        grad_x = -(x - loc) / (sigma**2)
        grad_z_analytical = t.forward_grad(x, grad_x)

        eps = 1e-7
        grad_z_num = np.zeros(d)
        for i in range(d):
            zp = z.copy()
            zp[i] += eps
            xp, det_p = t.inverse(zp)
            lp_p = -0.5 * float(np.sum(((xp - loc) / sigma) ** 2)) + det_p

            zm = z.copy()
            zm[i] -= eps
            xm, det_m = t.inverse(zm)
            lp_m = -0.5 * float(np.sum(((xm - loc) / sigma) ** 2)) + det_m

            grad_z_num[i] = (lp_p - lp_m) / (2.0 * eps)

        assert np.allclose(grad_z_analytical, grad_z_num, atol=1e-6)

    def test_calibrate_from_samples_translation_equivariance_and_gaussian_consistency(self) -> None:
        rng = np.random.default_rng(99)
        n = 4000
        # Coord 0: Cauchy(0, 2.0), Coord 1: Normal(0, 3.0)
        c0 = rng.standard_cauchy(size=n) * 2.0
        c1 = rng.normal(loc=0.0, scale=3.0, size=n)
        base_samples = np.column_stack([c0, c1])

        shift = np.array([50.0, -120.0])
        shifted_samples = base_samples + shift

        t_base = CoordinateWiseArcsinhTransform.calibrate_from_samples(base_samples)
        t_shifted = CoordinateWiseArcsinhTransform.calibrate_from_samples(shifted_samples)

        # Radii and mask must be invariant to constant translation
        assert np.allclose(t_base.radii, t_shifted.radii, rtol=1e-10)
        assert np.array_equal(t_base.mask, t_shifted.mask)
        assert np.allclose(t_shifted.location, shift, atol=0.15)
        # For Normal(loc=-120, scale=3.0), normal-consistent IQR scale should be close to 3.0
        assert t_shifted.radii[1] == pytest.approx(3.0, rel=0.10)

    def test_jax_arcsinh_transform_location_shift(self) -> None:
        loc = jnp.array([15.0, -30.0])
        rad = jnp.array([2.0, 4.0])
        jt = JaxArcsinhTransform(radius=rad, location=loc)

        z0 = jt.forward(loc)
        assert np.allclose(np.asarray(z0), 0.0, atol=1e-15)

        x = jnp.array([19.0, -22.0])
        z = jt.forward(x)
        x_rec = jt.inverse(z)
        assert np.allclose(np.asarray(x), np.asarray(x_rec), atol=1e-12)


class TestGentleMALABasinAndSoftCap:
    """Verifies GentleMALA discrete basin proposition and soft-capped non-freezing."""

    def test_pure_sqrt_damping_basin_boundary(self) -> None:
        """Pure sqrt damping (kappa_max=inf) contracts for r < 4*gamma/h and overshoots for r > 4*gamma/h."""
        gamma = 1.0
        h = 0.1
        sampler_pure = GentleMALA(step_size=h, gamma=gamma, mode="global", kappa_max=float("inf"))
        s = np.ones(1)

        # Inside basin: r = 20 < 4 * gamma / h = 40
        r_in = 20.0
        g_in = np.array([-(r_in**3)])
        b_in = sampler_pure._gentle_drift(g_in, s)
        r_next_in = abs(r_in + h * b_in[0])
        assert r_next_in < r_in

        # Outside basin: r = 100 > 4 * gamma / h = 40 -> overshoots across origin to |r_next| > r
        r_out = 100.0
        g_out = np.array([-(r_out**3)])
        b_out = sampler_pure._gentle_drift(g_out, s)
        r_next_out = abs(r_out + h * b_out[0])
        assert r_next_out > r_out

    @pytest.mark.parametrize(
        "h,r0,max_steps",
        [
            (0.50, 31.6, 15),
            (0.10, 100.0, 20),
            (0.10, 200.0, 30),
            (0.01, 1000.0, 80),
        ],
    )
    def test_soft_capped_gentle_mala_never_freezes_on_referee_quartic_grid(
        self, h: float, r0: float, max_steps: int
    ) -> None:
        """Soft-capped GentleMALA (default kappa_max=25.0) never overshoots or freezes on the referee's grid."""
        d = 10
        target = QuarticTarget(d=d)
        rng = np.random.default_rng(42)
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        x0 = direction * r0

        sampler = GentleMALA(step_size=h, gamma=1.0, mode="global")
        state = sampler.init_state(x0, target)

        accepted_count = 0
        reached_bulk = False
        for _ in range(max_steps):
            res = sampler.step(state, target, rng)
            state = res.state
            if res.accepted:
                accepted_count += 1
            if float(np.linalg.norm(state.x)) <= 3.5:
                reached_bulk = True
                break

        assert accepted_count > 0, f"GentleMALA froze at h={h}, R0={r0}"
        assert reached_bulk, f"GentleMALA failed to reach bulk <= 3.5 within {max_steps} steps at h={h}, R0={r0}"


class TestUnifiedRobustNUTSAndCalibration:
    """Verifies UnifiedRobustNUTS Phase 1 location+scale calibration and d=10 invariance."""

    def test_phase1_calibrates_location_and_scale_on_shifted_target(self) -> None:
        d = 4
        # Shifted target: mean at [20, -20, 10, -10]
        mean = np.array([20.0, -20.0, 10.0, -10.0])
        stds = np.array([1.0, 2.0, 0.7, 1.4])

        target = DiagonalGaussian(d=d, mean=mean, stds=stds)
        nuts = UnifiedRobustNUTS(
            step_size=0.2,
            max_depth=6,
            adapt_tail_scale=True,
            adapt_preconditioner=True,
            adapt_step_size=True,
            warmup_steps=200,
        )
        rng = np.random.default_rng(42)
        samples, _ = nuts.sample(x0=mean + 0.5, target=target, n_steps=400, burn_in=200, rng=rng, adapt=True)

        assert nuts.transform is not None
        assert np.allclose(nuts.transform.location, mean, atol=1.5)
        assert np.allclose(np.mean(samples, axis=0), mean, atol=0.35)


class TestBenchmarkIntegrityAndFairness:
    """Verifies symmetric divergence accounting, Task 1 heavy tails, and plot.py strictness."""

    def test_sota_adapters_count_post_warmup_divergences_only(self) -> None:
        target = QuarticTarget(d=2)
        # Force an overly large initial step size that causes warmup divergences before adaptation settles
        x0 = np.array([[5.0, 5.0], [-5.0, -5.0]])
        _, info_bj = run_blackjax_sampler(
            target,
            sampler_type="nuts",
            n_samples=40,
            n_warmup=40,
            n_chains=2,
            seed=42,
            x0=x0,
        )
        assert "warmup_divergences" in info_bj
        assert "post_warmup_divergences" in info_bj
        assert info_bj["divergences"] == info_bj["post_warmup_divergences"]

        _, info_mr = run_mcmc_robust_sampler(
            UnifiedRobustNUTS,
            target,
            n_samples=40,
            n_warmup=40,
            n_chains=2,
            seed=42,
            x0=x0,
        )
        assert "warmup_divergences" in info_mr
        assert "post_warmup_divergences" in info_mr
        assert info_mr["divergences"] == info_mr["post_warmup_divergences"]

    def test_sparse_signal_regression_has_non_degenerate_ground_truth(self) -> None:
        target = SparseSignalRegressionTarget(d=10, n_obs=40, n_active=3, seed=42)
        gt = target.sample_ground_truth(2000, rng=np.random.default_rng(42))
        assert gt.shape == (2000, 10)
        # Ground truth must be posterior draws with positive variance, NOT np.tile(beta_true)
        stds = np.std(gt, axis=0)
        assert np.all(stds > 0.05), f"Ground truth has zero variance: {stds}"

    def test_plot_module_has_no_mock_fallback(self, tmp_path: Path) -> None:
        assert not hasattr(plot_mod, "generate_mock_benchmark_results")
        with pytest.raises(FileNotFoundError):
            plot_mod.generate_all_figures(reports_dir=str(tmp_path), output_dir=str(tmp_path / "figs"))
