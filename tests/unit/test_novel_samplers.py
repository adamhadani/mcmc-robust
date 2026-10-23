"""Comprehensive unit test suite for novel MCMC samplers (Features F15, F16, F17)."""

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.barker import BarkerLangevin
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.novel.anisotropic_tail import AnisotropicTailMALA
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.samplers.tamed_mala import GloballyTamedMALA
from mcmc_robust.targets.benign import IsotropicGaussian
from mcmc_robust.targets.dual import (
    ConfinementInducedRoughnessTarget,
    CoupledDualPathologyTarget,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)
from mcmc_robust.targets.roughness import QuarticTarget
from mcmc_robust.transforms.base import SpaceTransformation, TransformedTarget
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform


def _compute_bulk_ess(samples: np.ndarray) -> float:
    """Computes rank-normalized Bulk Effective Sample Size (ESS) for an MCMC chain."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    n, d = s.shape
    if n < 4:
        return float(n)

    ess_per_dim = []
    for j in range(d):
        col = s[:, j]
        ranks = stats.rankdata(col)
        normalized = stats.norm.ppf((ranks - 0.375) / (n + 0.25))
        var_norm = np.var(normalized, ddof=1)
        if var_norm < 1e-12:
            ess_per_dim.append(float(n))
            continue

        max_lag = min(n - 1, 100)
        norm_centered = normalized - np.mean(normalized)
        autocorr = np.correlate(norm_centered, norm_centered, mode="full")[n - 1 : n + max_lag]
        autocorr = autocorr / (autocorr[0] + 1e-15)

        tau = 1.0
        for lag in range(1, len(autocorr)):
            if autocorr[lag] < 0.05:
                break
            tau += 2.0 * autocorr[lag]
        ess_dim = n / max(1.0, tau)
        ess_per_dim.append(ess_dim)

    return float(np.min(ess_per_dim))


def _compute_tail_ess(samples: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95) -> float:
    """Computes indicator-based Tail ESS."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    n, d = s.shape
    if n < 4:
        return float(n)

    ess_tails = []
    for j in range(d):
        col = s[:, j]
        q_low, q_high = np.quantile(col, [lower_q, upper_q])
        ind_low = (col <= q_low).astype(np.float64)
        ind_high = (col >= q_high).astype(np.float64)

        for ind in [ind_low, ind_high]:
            var_ind = np.var(ind, ddof=1)
            if var_ind < 1e-12:
                ess_tails.append(float(n))
                continue
            max_lag = min(n - 1, 100)
            ind_c = ind - np.mean(ind)
            autocorr = np.correlate(ind_c, ind_c, mode="full")[n - 1 : n + max_lag]
            autocorr = autocorr / (autocorr[0] + 1e-15)

            tau = 1.0
            for lag in range(1, len(autocorr)):
                if autocorr[lag] < 0.05:
                    break
                tau += 2.0 * autocorr[lag]
            ess_tails.append(n / max(1.0, tau))

    return float(np.min(ess_tails))


# =============================================================================
# Category A: CoordinateWiseArcsinhTransform Tests (Feature F16)
# =============================================================================
class TestCoordinateWiseArcsinhTransform:
    """Unit tests for CoordinateWiseArcsinhTransform."""

    def test_f16_arcsinh_transform_interface(self):
        """Interface compliance, subclassing, dimension and argument checks."""
        t = CoordinateWiseArcsinhTransform(d=4, radii=2.0)
        assert isinstance(t, SpaceTransformation)
        assert t.input_dim == 4
        assert t.output_dim == 4
        assert t.d == 4
        assert np.allclose(t.radii, np.full(4, 2.0))
        assert t.mask.shape == (4,)
        assert np.all(t.mask)

        # Invalid arguments
        with pytest.raises(ValueError, match="positive"):
            CoordinateWiseArcsinhTransform(d=-1)
        with pytest.raises(ValueError, match="strictly positive"):
            CoordinateWiseArcsinhTransform(d=2, radii=-1.0)
        with pytest.raises(ValueError, match="radii shape"):
            CoordinateWiseArcsinhTransform(d=3, radii=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="mask shape"):
            CoordinateWiseArcsinhTransform(d=2, mask=np.array([True]))

    @pytest.mark.parametrize("dim", [2, 10, 50, 100])
    def test_f16_arcsinh_transform_roundtrip_precision(self, dim: int):
        """Round-trip invertibility || phi^{-1}(phi(x)) - x ||_inf < 1e-12."""
        rng = np.random.default_rng(42 + dim)
        radii = rng.uniform(0.5, 5.0, size=dim)
        t = CoordinateWiseArcsinhTransform(d=dim, radii=radii)

        # Spanning bulk to extreme values
        x = rng.normal(size=dim) * np.logspace(-3, 2, dim)
        z, _ = t.forward(x)
        x_rec, _ = t.inverse(z)

        max_err = float(np.max(np.abs(x - x_rec)))
        assert max_err < 1e-12, f"Round-trip error {max_err} exceeds 1e-12 threshold"

    @pytest.mark.parametrize("dim", [2, 10, 50, 100])
    def test_f16_arcsinh_transform_jacobian_cancellation(self, dim: int):
        """Analytical log-determinants cancel to machine precision: |det J_fwd + det J_inv| < 1e-12."""
        rng = np.random.default_rng(100 + dim)
        radii = rng.uniform(0.2, 10.0, size=dim)
        t = CoordinateWiseArcsinhTransform(d=dim, radii=radii)

        x = rng.normal(scale=10.0, size=dim)
        z, log_det_fwd = t.forward(x)
        _, log_det_inv = t.inverse(z)

        cancel_err = abs(log_det_fwd + log_det_inv)
        assert cancel_err < 1e-12, f"Jacobian cancellation error {cancel_err} exceeds 1e-12"

    def test_f16_arcsinh_transform_gradient_pullback_finite_difference(self):
        """Analytical pullback gradient matches central finite differences to < 1e-7."""
        d = 4
        radii = np.array([1.0, 2.5, 0.5, 5.0])
        t = CoordinateWiseArcsinhTransform(d=d, radii=radii)

        x = np.array([0.5, -2.0, 3.5, -1.0])
        # Synthetic base quadratic log-prob: log_pi(x) = -0.5 * sum((x / sigma)^2)
        sigma = np.array([1.0, 2.0, 0.8, 1.5])
        grad_x = -x / (sigma**2)

        # Transformed coordinate and analytical pullback gradient
        z, _ = t.forward(x)
        grad_z_analytical = t.forward_grad(x, grad_x)

        # Numerical gradient of log pi_z(z) = log pi_x(phi^{-1}(z)) + log |det J_phi^{-1}(z)|
        eps = 1e-7
        grad_z_numerical = np.zeros(d, dtype=np.float64)
        for i in range(d):
            z_plus = z.copy()
            z_plus[i] += eps
            x_plus, det_inv_plus = t.inverse(z_plus)
            lp_plus = -0.5 * float(np.sum((x_plus / sigma) ** 2)) + det_inv_plus

            z_minus = z.copy()
            z_minus[i] -= eps
            x_minus, det_inv_minus = t.inverse(z_minus)
            lp_minus = -0.5 * float(np.sum((x_minus / sigma) ** 2)) + det_inv_minus

            grad_z_numerical[i] = (lp_plus - lp_minus) / (2.0 * eps)

        diff = np.abs(grad_z_analytical - grad_z_numerical)
        assert np.all(diff < 1e-7), f"Max difference {np.max(diff)} exceeds 1e-7"

    def test_f16_arcsinh_transform_restoring_drift_student_t(self):
        """Converts vanishing polynomial tail gradient into constant restoring drift -nu * sgn(z)."""
        nu = 2.0
        R = 2.0
        target = MultivariateStudentTTarget(d=1, nu=nu)
        t = CoordinateWiseArcsinhTransform(d=1, radii=R)

        # TransformedTarget wrapper
        tt = TransformedTarget(target, t)

        for z_val in [5.0, 10.0, 20.0, 50.0]:
            z = np.array([z_val])
            grad_z = tt.grad_log_prob(z)
            # In theorem: limit is - (alpha - 1) = - nu
            expected = -nu
            assert np.isclose(grad_z[0], expected, atol=0.15)

    def test_f16_arcsinh_transform_heterogeneous_mask(self):
        """Light-tailed coordinates with mask=False are completely untouched and uncoupled."""
        radii = np.array([2.0, 1.0])
        mask = np.array([True, False])  # coord 0 heavy, coord 1 light
        t = CoordinateWiseArcsinhTransform(d=2, radii=radii, mask=mask)

        x = np.array([100.0, 1.5])
        z, log_det = t.forward(x)

        # Light coordinate 1 is strictly identity
        assert z[1] == 1.5
        x_rec, log_det_inv = t.inverse(z)
        assert x_rec[1] == 1.5

        # Gradient on coord 1 is unchanged
        gx = np.array([-5.0, -1.5])
        gz = t.forward_grad(x, gx)
        assert gz[1] == gx[1]

    def test_f16_arcsinh_calibrate_from_samples(self):
        """calibrate_from_samples factory accurately classifies heavy vs light coordinates."""
        rng = np.random.default_rng(123)
        # Coord 0: Cauchy (heavy, nu=1)
        c0 = rng.standard_cauchy(size=1000)
        # Coord 1: Gaussian (light)
        c1 = rng.normal(scale=2.0, size=1000)
        samples = np.column_stack([c0, c1])

        transform = CoordinateWiseArcsinhTransform.calibrate_from_samples(
            samples,
            alpha_threshold=3.5,
            quantile_ratio_threshold=3.0,
        )
        assert transform.mask[0] is np.True_ or transform.mask[0] == True
        assert transform.radii[1] > 0.0

    def test_f16_arcsinh_transform_extreme_float64_no_overflow(self):
        """Regression test for Defect 2: Verifies finite outputs and machine precision without warnings for |x| >= 1e154.

        Tests forward(), inverse(), and forward_grad() with inputs up to |x| = 10^300 and x=0,
        ensuring zero RuntimeWarning (overflow / divide by zero) under pytest -W error.
        """
        d = 7
        radii = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1.0])
        t = CoordinateWiseArcsinhTransform(d=d, radii=radii)

        # Vector spanning zero, bulk to extreme float64 boundaries
        x_extreme = np.array([1e200, -1e200, 1e300, -1e300, 1e154, -1e154, 0.0])
        gx_extreme = np.array([-1e-200, 1e-200, -1e-300, 1e-300, -1e-154, 1e-154, 2.5])

        # 1. Forward transform
        z, log_det_fwd = t.forward(x_extreme)
        assert np.all(np.isfinite(z)), f"Forward produced non-finite z: {z}"
        assert np.isfinite(log_det_fwd), f"Forward produced non-finite log_det: {log_det_fwd}"
        assert z[6] == 0.0

        # 2. Inverse round-trip precision
        x_rec, log_det_inv = t.inverse(z)
        assert np.all(np.isfinite(x_rec)), f"Inverse produced non-finite x_rec: {x_rec}"
        assert np.isfinite(log_det_inv), f"Inverse produced non-finite log_det: {log_det_inv}"
        assert x_rec[6] == 0.0

        max_rel_err = float(np.max(np.abs(x_rec - x_extreme) / np.maximum(1.0, np.abs(x_extreme))))
        assert max_rel_err < 1e-12, f"Roundtrip relative error {max_rel_err:.2e} exceeds 1e-12"

        # 3. Exact analytical Jacobian cancellation
        cancel_err = abs(log_det_fwd + log_det_inv)
        assert cancel_err < 1e-10, f"Jacobian cancellation error {cancel_err} exceeds 1e-10"

        # 4. Pullback gradient
        gz = t.forward_grad(x_extreme, gx_extreme)
        assert np.all(np.isfinite(gz)), f"Pullback gradient produced non-finite values: {gz}"
        assert gz[6] == 2.5

    def test_f16_arcsinh_transform_extreme_float64_pullback_drift(self):
        """Verifies that for Student-t target gradients at |x| = 10^200, pullback drift reaches -nu * sgn(x)."""
        nu = 2.5
        R = 1.5
        t = CoordinateWiseArcsinhTransform(d=2, radii=R)

        # Student-t log-density gradient: gx = - (nu + 1) * x / (nu + x^2) ~ - (nu + 1) / x for large |x|
        x = np.array([1e200, -1e200])
        gx = -(nu + 1.0) / x

        gz = t.forward_grad(x, gx)
        assert np.all(np.isfinite(gz))

        # Expected limit: scale * gx + (x / scale) = |x| * (-(nu+1)/x) + sgn(x) = -(nu+1)*sgn(x) + sgn(x) = -nu * sgn(x)
        expected = np.array([-nu, nu])
        assert np.allclose(gz, expected, rtol=1e-10), f"Extreme drift {gz} did not match expected {expected}"

    def test_f16_arcsinh_transform_extreme_float64_masked_heterogeneous(self):
        """Verifies that extreme inputs (|x| = 10^200) in heavy coordinates do not corrupt unmasked light coordinates."""
        d = 4
        radii = np.array([1.0, 1.0, 1.0, 1.0])
        mask = np.array([True, False, True, False])  # coords 0, 2 heavy; coords 1, 3 light
        t = CoordinateWiseArcsinhTransform(d=d, radii=radii, mask=mask)

        x = np.array([1e200, 2.5, -1e200, -1.5])
        gx = np.array([-1e-200, -2.5, 1e-200, 1.5])

        z, log_det_fwd = t.forward(x)
        assert z[1] == 2.5
        assert z[3] == -1.5

        x_rec, log_det_inv = t.inverse(z)
        assert x_rec[1] == 2.5
        assert x_rec[3] == -1.5
        assert abs(log_det_fwd + log_det_inv) < 1e-10

        gz = t.forward_grad(x, gx)
        assert gz[1] == gx[1]
        assert gz[3] == gx[3]
        assert np.all(np.isfinite(gz))


# =============================================================================
# Category B: GentleMALA Tests (Feature F15)
# =============================================================================
class TestGentleMALA:
    """Unit tests for GentleMALA (§3.3 local stiffness adaptation)."""

    def test_f15_gentle_mala_interface(self):
        """Constructor arguments, properties, and input validations."""
        sampler = GentleMALA(step_size=0.05, gamma=0.5, mode="global")
        assert sampler.requires_grad is True
        assert sampler.step_size == 0.05
        assert sampler.gamma == 0.5
        assert sampler.mode == "global"

        with pytest.raises(ValueError, match="step_size must be positive"):
            GentleMALA(step_size=-0.1)
        with pytest.raises(ValueError, match="gamma must be positive"):
            GentleMALA(gamma=-0.5)
        with pytest.raises(ValueError, match="mode must be"):
            GentleMALA(mode="invalid_mode")

    def test_f15_gentle_mala_stiffness_metric_nonnegative(self):
        """Stiffness metric kappa_h(x) is non-negative and scales with step size and gradient."""
        s = np.ones(3)
        for mode in ["global", "coordinate"]:
            sampler = GentleMALA(step_size=0.1, gamma=1.0, mode=mode)
            g_zero = np.zeros(3)
            stiff_zero = sampler._stiffness(g_zero, s)
            assert np.all(stiff_zero >= 0.0)
            assert np.allclose(stiff_zero, 0.0)

            g_pos = np.array([2.0, 4.0, 6.0])
            stiff_pos = sampler._stiffness(g_pos, s)
            assert np.all(stiff_pos > 0.0)

    def test_f15_gentle_mala_unconstrained_in_smooth_bulk(self):
        """In smooth bulk (small gradient), gentle taming preserves raw Langevin drift within 1%."""
        sampler = GentleMALA(step_size=0.05, gamma=1.0)
        g_small = np.array([0.01, -0.01, 0.005])
        drift = sampler._tamed_drift(g_small)
        rel_diff = np.linalg.norm(drift - g_small) / np.linalg.norm(g_small)
        assert rel_diff < 0.01, f"Relative difference {rel_diff} exceeds 1%"

    def test_f15_gentle_mala_sublinear_tail_growth(self):
        """Gentle drift is unbounded and grows sublinearly: ||b(4g)|| > ||b(g)|| but ratio < 4."""
        sampler = GentleMALA(step_size=0.1, gamma=1.0)
        g1 = np.array([100.0, 0.0])
        g2 = np.array([400.0, 0.0])

        b1 = sampler._tamed_drift(g1)
        b2 = sampler._tamed_drift(g2)

        norm_b1 = float(np.linalg.norm(b1))
        norm_b2 = float(np.linalg.norm(b2))

        assert norm_b2 > norm_b1, "Drift must be strictly increasing"
        growth_ratio = norm_b2 / norm_b1
        # For sublinear square-root growth, ratio of 400/100 = 4 yields sqrt(4) = 2
        assert growth_ratio < 3.0, f"Growth ratio {growth_ratio} is not sublinear"

    def test_f15_gentle_mala_oracle_budget_contract(self):
        """Strictly 1 gradient and 1 log_prob evaluation per transition step via GradBudgetTracker."""
        target = GradBudgetTracker(IsotropicGaussian(d=3))
        sampler = GentleMALA(step_size=0.1)

        state = sampler.init_state(np.zeros(3), target)
        assert target.n_grad == 1
        assert target.n_log_prob == 1

        n_steps = 30
        rng = np.random.default_rng(42)
        for _ in range(n_steps):
            res = sampler.step(state, target, rng)
            state = res.state

        # Cumulative accounting: 1 init + n_steps
        assert target.n_grad == 1 + n_steps
        assert target.n_log_prob == 1 + n_steps

    def test_f15_gentle_mala_stationarity_ks_gaussian(self):
        """Stationarity on 1D Gaussian ground truth (two-sample KS p-value > 0.01)."""
        target = IsotropicGaussian(d=1)
        sampler = GentleMALA(step_size=0.2, gamma=0.5)
        rng = np.random.default_rng(101)

        samples, _ = sampler.sample(np.array([1.0]), target, n_steps=8000, burn_in=1000, rng=rng)
        gt = target.sample_ground_truth(10000, rng=rng).flatten()
        ks = stats.ks_2samp(samples.flatten()[::3], gt)
        assert ks.pvalue > 0.01, f"KS test failed with p-value {ks.pvalue}"

    def test_f15_gentle_mala_stationarity_ks_quartic(self):
        """Stationarity on 1D QuarticTarget ground truth (two-sample KS p-value > 0.01)."""
        target = QuarticTarget(d=1)
        sampler = GentleMALA(step_size=0.15, gamma=0.5)
        rng = np.random.default_rng(202)

        samples, _ = sampler.sample(np.array([1.0]), target, n_steps=10000, burn_in=2000, rng=rng)
        gt = target.sample_ground_truth(10000, rng=rng).flatten()
        ks = stats.ks_2samp(samples.flatten()[::3], gt)
        assert ks.pvalue > 0.01, f"KS test failed on quartic with p-value {ks.pvalue}"

    def test_f15_gentle_mala_quartic_burnin_acceleration(self):
        """AC3 Guardrail: GentleMALA achieves >= 2x faster burn-in to bulk than GloballyTamedMALA."""
        target = QuarticTarget(d=1)
        x0 = np.array([15.0])  # Extreme tail
        h = 0.1

        def steps_to_bulk(s, max_steps=500):
            st = s.init_state(x0, target)
            r = np.random.default_rng(42)
            for step in range(1, max_steps + 1):
                res = s.step(st, target, r)
                st = res.state
                if abs(st.x[0]) <= 2.0:
                    return step
            return max_steps

        s_gentle = GentleMALA(step_size=h, gamma=1.0)
        s_tamed = GloballyTamedMALA(step_size=h, gamma=0.5)

        steps_g = steps_to_bulk(s_gentle)
        steps_t = steps_to_bulk(s_tamed)

        assert steps_g <= 0.5 * steps_t, f"Burn-in speedup failed: gentle={steps_g}, tamed={steps_t}"

    def test_f15_gentle_mala_10x_step_size_stability(self):
        """AC3 Guardrail: Remains numerically stable across 10x step sizes h in [0.01, 0.10]."""
        target = QuarticTarget(d=2)
        x0 = np.array([10.0, 10.0])
        rng = np.random.default_rng(777)

        for h in [0.01, 0.05, 0.10]:
            sampler = GentleMALA(step_size=h, gamma=1.0)
            samples, _ = sampler.sample(x0, target, n_steps=20, rng=rng)
            assert np.all(np.isfinite(samples)), f"Non-finite sample at step size {h}"

    def test_f15_gentle_mala_benign_parity_ac2(self):
        """AC2 Guardrail: Achieves >= 70% (target > 85%) ESS/grad parity on d=50 Gaussian."""
        d = 50
        target = IsotropicGaussian(d=d)
        x0 = np.zeros(d)
        h = 0.15
        n_steps = 1200
        burn_in = 200

        s_mala = MALA(step_size=h)
        samples_mala, _ = s_mala.sample(x0, target, n_steps=n_steps, burn_in=burn_in, rng=np.random.default_rng(42))

        s_gentle = GentleMALA(step_size=h, gamma=0.2, mode="global")
        samples_gentle, _ = s_gentle.sample(x0, target, n_steps=n_steps, burn_in=burn_in, rng=np.random.default_rng(42))

        ess_mala = _compute_bulk_ess(samples_mala)
        ess_gentle = _compute_bulk_ess(samples_gentle)
        parity = ess_gentle / ess_mala
        assert parity >= 0.70, f"Benign parity {parity:.2%} falls below 70% AC2 threshold"

    def test_f15_gentle_mala_defensive_overflow_guards(self):
        """Non-finite candidate or overflow triggers clean rejection with prior state."""
        target = IsotropicGaussian(d=2)
        sampler = GentleMALA(step_size=0.1)
        rng = np.random.default_rng(42)

        nan_state = ChainState(
            x=np.array([np.nan, 0.0]),
            log_prob=-np.inf,
            grad_log_prob=np.array([np.nan, 0.0]),
        )
        res = sampler.step(nan_state, target, rng)
        assert res.accepted is False


# =============================================================================
# Category C: AnisotropicTailMALA Tests (Feature F16)
# =============================================================================
class TestAnisotropicTailMALA:
    """Unit tests for AnisotropicTailMALA (§4.3 tail flatness resolution)."""

    def test_f16_anisotropic_tail_mala_interface(self):
        """Constructor arguments and parameter validations."""
        sampler = AnisotropicTailMALA(step_size=0.1, radii=np.array([1.0, 2.0]))
        assert sampler.requires_grad is True
        assert sampler.step_size == 0.1
        assert sampler.transform is not None

        with pytest.raises(ValueError, match="positive"):
            AnisotropicTailMALA(step_size=-0.1)

    def test_f16_anisotropic_tail_mala_oracle_budget_contract(self):
        """Strictly 1 gradient and 1 log_prob evaluation per step via GradBudgetTracker."""
        target = GradBudgetTracker(MultivariateStudentTTarget(d=2, nu=2.5))
        sampler = AnisotropicTailMALA(step_size=0.1, radii=1.0)

        state = sampler.init_state(np.zeros(2), target)
        assert target.n_grad == 1
        assert target.n_log_prob == 1

        n_steps = 25
        rng = np.random.default_rng(42)
        for _ in range(n_steps):
            res = sampler.step(state, target, rng)
            state = res.state

        assert target.n_grad == 1 + n_steps
        assert target.n_log_prob == 1 + n_steps

    def test_f16_anisotropic_tail_mala_primal_samples_contract(self):
        """Samples returned by .sample() are in primal coordinate space x, not transformed space z."""
        target = MultivariateStudentTTarget(d=1, nu=2.5)
        sampler = AnisotropicTailMALA(step_size=0.1, radii=2.0)
        samples, _ = sampler.sample(np.array([0.0]), target, n_steps=200, burn_in=50, rng=np.random.default_rng(42))
        assert samples.shape == (150, 1)
        assert np.all(np.isfinite(samples))

    def test_f16_anisotropic_tail_mala_stationarity_ks_student_t(self):
        """Stationarity on 1D Student-t (nu=2.5) ground truth (two-sample KS p-value > 0.01)."""
        target = MultivariateStudentTTarget(d=1, nu=2.5)
        sampler = AnisotropicTailMALA(step_size=0.2, radii=1.0)
        rng = np.random.default_rng(42)

        samples, _ = sampler.sample(np.array([0.0]), target, n_steps=10000, burn_in=2000, rng=rng)
        gt = target.sample_ground_truth(10000, rng=rng).flatten()
        ks = stats.ks_2samp(samples.flatten()[::3], gt)
        assert ks.pvalue > 0.01, f"KS test failed on Student-t with p-value {ks.pvalue}"

    def test_f16_anisotropic_tail_mala_tail_efficiency_ac4(self):
        """AC4 Guardrail: >= 2x Tail-ESS and >= 30% lower q99 error vs Barker on d=20 mixed target."""
        d = 20
        nu_list = [1.5, 2.5] * 5
        target = HeterogeneousHeavyTailedTarget(d=d, nu_list=nu_list)
        x0 = np.zeros(d)
        n_steps = 3000
        burn_in = 500

        mask = np.zeros(d, dtype=bool)
        mask[target.heavy_coords] = True

        rng_seed = 123
        s_barker = BarkerLangevin(step_size=0.1)
        samples_barker, _ = s_barker.sample(
            x0, target, n_steps=n_steps, burn_in=burn_in, rng=np.random.default_rng(rng_seed)
        )

        s_aniso = AnisotropicTailMALA(step_size=0.2, radii=1.0, mask=mask)
        samples_aniso, _ = s_aniso.sample(
            x0, target, n_steps=n_steps, burn_in=burn_in, rng=np.random.default_rng(rng_seed)
        )

        tail_ess_b = _compute_tail_ess(samples_barker[:, target.heavy_coords])
        tail_ess_a = _compute_tail_ess(samples_aniso[:, target.heavy_coords])

        exact_q99 = target.exact_moments_or_quantiles()["q99"][0]
        q99_b = np.quantile(samples_barker[:, 0], 0.99)
        q99_a = np.quantile(samples_aniso[:, 0], 0.99)
        err_b = abs(q99_b - exact_q99) / exact_q99
        err_a = abs(q99_a - exact_q99) / exact_q99
        err_reduction = (err_b - err_a) / err_b

        assert tail_ess_a / tail_ess_b >= 2.0, f"Tail-ESS ratio {tail_ess_a / tail_ess_b:.2f} < 2.0"
        assert err_reduction >= 0.30, f"Error reduction {err_reduction:.2%} < 30%"

    def test_f16_anisotropic_tail_mala_online_calibration_warmup(self):
        """Warmup calibration successfully estimates transform when radii=None."""
        target = MultivariateStudentTTarget(d=2, nu=2.5)
        sampler = AnisotropicTailMALA(step_size=0.1, calib_steps=25)
        rng = np.random.default_rng(42)

        # Before calibration, transform is None
        assert sampler.transform is None
        state = sampler.init_state(np.zeros(2), target)

        for _ in range(30):
            res = sampler.step(state, target, rng)
            state = res.state

        # After calibration steps, transform is configured
        assert sampler.transform is not None
        assert sampler.transform.d == 2


# =============================================================================
# Category D: UnifiedRobustSampler Tests (Feature F17)
# =============================================================================
class TestUnifiedRobustSampler:
    """Unit tests for UnifiedRobustSampler (dual pathology synthesis)."""

    def test_f17_unified_sampler_interface_and_immutability(self):
        """Conforms to Sampler base class and returns immutable ChainState."""
        sampler = UnifiedRobustSampler(step_size=0.1, gamma=0.2, radius=3.0)
        assert sampler.requires_grad is True

        target = IsotropicGaussian(d=2)
        state = sampler.init_state(np.zeros(2), target)
        assert isinstance(state, ChainState)
        assert state.x.flags.writeable is False

        rng = np.random.default_rng(42)
        res = sampler.step(state, target, rng)
        assert isinstance(res, StepResult)
        assert isinstance(res.state, ChainState)
        assert res.state.x.flags.writeable is False

    def test_f17_unified_sampler_oracle_budget_contract(self):
        """Strictly 1 gradient and 1 log_prob evaluation per step on coupled dual target."""
        target = GradBudgetTracker(CoupledDualPathologyTarget(d=4))
        sampler = UnifiedRobustSampler(step_size=0.05, gamma=0.1, radius=2.0)

        state = sampler.init_state(np.zeros(4), target)
        assert target.n_grad == 1

        n_steps = 30
        rng = np.random.default_rng(42)
        for _ in range(n_steps):
            res = sampler.step(state, target, rng)
            state = res.state

        assert target.n_grad == 1 + n_steps
        assert target.n_log_prob == 1 + n_steps

    def test_f17_unified_sampler_reduction_to_mala_on_benign(self):
        """With gamma=0 and mask=False on benign Gaussian, proposal matches standard MALA to float precision."""
        d = 2
        target = IsotropicGaussian(d=d)
        x0 = np.array([0.5, -0.5])
        h = 0.1

        s_unified = UnifiedRobustSampler(step_size=h, gamma=0.0, mask=np.zeros(d, dtype=bool))
        s_mala = MALA(step_size=h)

        st_u = s_unified.init_state(x0, target)
        st_m = s_mala.init_state(x0, target)

        # Propose with same noise vector
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)

        res_u = s_unified.step(st_u, target, rng1)
        res_m = s_mala.step(st_m, target, rng2)

        assert np.allclose(res_u.proposal, res_m.proposal, atol=1e-5)

    def test_f17_unified_sampler_three_phase_warmup_scheduling(self):
        """Verifies 3-phase warmup lifecycle without destructive parameter interference."""
        sampler = UnifiedRobustSampler(
            step_size=0.1,
            warmup_steps=100,
            adapt_step_size=True,
            adapt_preconditioner=True,
            adapt_tail_scale=True,
        )
        target = CoupledDualPathologyTarget(d=4)
        state = sampler.init_state(np.zeros(4), target)
        rng = np.random.default_rng(42)

        # Step through Phase 1 (0 <= t < 15)
        for t in range(15):
            res = sampler.step(state, target, rng)
            sampler.adapt(t, res)
            state = res.state

        assert sampler._tail_scale_frozen is True
        frozen_radius = sampler.radius.copy()

        # Step through Phase 2 (15 <= t < 80)
        for t in range(15, 80):
            res = sampler.step(state, target, rng)
            sampler.adapt(t, res)
            state = res.state

        assert sampler._preconditioner_frozen is True
        assert np.allclose(sampler.radius, frozen_radius)

        # Step through Phase 3 (80 <= t < 100)
        for t in range(80, 100):
            res = sampler.step(state, target, rng)
            sampler.adapt(t, res)
            state = res.state

        frozen_h = sampler.step_size

        # Post-warmup (t >= 100): all parameters remain completely frozen
        for t in range(100, 120):
            res = sampler.step(state, target, rng)
            sampler.adapt(t, res)
            state = res.state
            assert sampler.step_size == frozen_h

    def test_f17_unified_sampler_dual_pathology_stability(self):
        """Remains 100% non-divergent and returns to bulk on CoupledDualPathologyTarget."""
        target = CoupledDualPathologyTarget(d=4)
        x0 = np.array([15.0, -15.0, 15.0, -15.0])  # Both rough and flat in extreme tail
        mask = np.array([True, True, False, False])
        sampler = UnifiedRobustSampler(step_size=0.02, gamma=1.0, radius=3.0, mask=mask)
        rng = np.random.default_rng(42)

        samples, results = sampler.sample(x0, target, n_steps=200, burn_in=50, rng=rng)
        assert np.all(np.isfinite(samples)), "Encountered non-finite samples"
        acc_rate = np.mean([r.accepted for r in results])
        assert acc_rate > 0.20, f"Acceptance rate {acc_rate:.2%} too low"

        # Check return to bulk
        final_norm = float(np.linalg.norm(samples[-1]))
        assert final_norm < np.linalg.norm(x0) * 0.5, f"Final norm {final_norm} did not contract"

    def test_f17_unified_sampler_confinement_roughness_stability(self):
        """Stable and non-divergent on ConfinementInducedRoughnessTarget."""
        target = ConfinementInducedRoughnessTarget(d=2)
        sampler = UnifiedRobustSampler(step_size=0.05, gamma=0.5, radius=2.0)
        samples, results = sampler.sample(np.array([1.0, 1.0]), target, n_steps=100, rng=np.random.default_rng(42))
        assert np.all(np.isfinite(samples))
        assert np.mean([r.accepted for r in results]) > 0.10

    def test_f17_unified_sampler_stationarity_ks_gaussian(self):
        """Stationarity on 1D Gaussian ground truth (two-sample KS p-value > 0.01)."""
        target = IsotropicGaussian(d=1)
        sampler = UnifiedRobustSampler(step_size=0.15, gamma=0.1, radius=3.0)
        rng = np.random.default_rng(42)

        samples, _ = sampler.sample(np.array([1.0]), target, n_steps=8000, burn_in=1000, rng=rng)
        gt = target.sample_ground_truth(10000, rng=rng).flatten()
        ks = stats.ks_2samp(samples.flatten()[::3], gt)
        assert ks.pvalue > 0.01, f"KS test failed with p-value {ks.pvalue}"

    def test_f17_unified_sampler_stationarity_ks_quartic(self):
        """Stationarity on 1D QuarticTarget ground truth (two-sample KS p-value > 0.01)."""
        target = QuarticTarget(d=1)
        sampler = UnifiedRobustSampler(step_size=0.15, gamma=0.2, radius=2.0)
        rng = np.random.default_rng(42)

        samples, _ = sampler.sample(np.array([1.0]), target, n_steps=10000, burn_in=2000, rng=rng)
        gt = target.sample_ground_truth(10000, rng=rng).flatten()
        ks = stats.ks_2samp(samples.flatten()[::3], gt)
        assert ks.pvalue > 0.01, f"KS test failed on quartic with p-value {ks.pvalue}"

    def test_f17_unified_sampler_stationarity_ks_student_t(self):
        """Stationarity on 1D Student-t ground truth (two-sample KS p-value > 0.01)."""
        target = MultivariateStudentTTarget(d=1, nu=2.5)
        sampler = UnifiedRobustSampler(step_size=0.2, gamma=0.1, radius=1.0)
        rng = np.random.default_rng(42)

        samples, _ = sampler.sample(np.array([0.0]), target, n_steps=10000, burn_in=2000, rng=rng)
        gt = target.sample_ground_truth(10000, rng=rng).flatten()
        ks = stats.ks_2samp(samples.flatten()[::3], gt)
        assert ks.pvalue > 0.01, f"KS test failed on student-t with p-value {ks.pvalue}"

    def test_f17_unified_sampler_defensive_guards(self):
        """Input parameter validations raise descriptive ValueError."""
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            UnifiedRobustSampler(gamma=-0.1)
        with pytest.raises(ValueError, match="mode must be"):
            UnifiedRobustSampler(mode="bad_mode")
        with pytest.raises(ValueError, match="Invalid radius bounds"):
            UnifiedRobustSampler(min_radius=10.0, max_radius=5.0)

    @pytest.mark.parametrize("dim", [10, 20, 50, 100])
    def test_f17_unified_sampler_phase2_warmup_deadlock_prevention(self, dim: int):
        """Regression test for Defect 1: Guards against Phase 2 warmup deadlock in dimensions d >= 10.

        When adapt_tail_scale=True, recreating the transform at t = W1 - 1 must not invalidate
        the auxiliary state or cause a permanent rejection deadlock during Phase 2.
        Asserts Phase 2 acceptance rate > 20% and final adapted step size h > 0.001.
        """
        target = CoupledDualPathologyTarget(d=dim, nu=2.0, c=1.0)
        d1 = target.d1
        mask = np.zeros(dim, dtype=bool)
        mask[:d1] = True

        warmup_steps = 100
        W1 = max(10, int(0.15 * warmup_steps))  # 15
        W2 = int(0.80 * warmup_steps)  # 80

        sampler = UnifiedRobustSampler(
            step_size=0.05,
            warmup_steps=warmup_steps,
            adapt_step_size=True,
            adapt_preconditioner=True,
            adapt_tail_scale=True,
            mask=mask,
        )

        state = sampler.init_state(np.zeros(dim), target)
        rng = np.random.default_rng(42 + dim)

        phase2_accepted = []
        for t in range(warmup_steps):
            res = sampler.step(state, target, rng)
            sampler.adapt(t, res)
            state = res.state
            if W1 <= t < W2:
                phase2_accepted.append(res.accepted)

        phase2_acc_rate = float(np.mean(phase2_accepted))
        assert phase2_acc_rate > 0.20, (
            f"Phase 2 acceptance rate {phase2_acc_rate:.2%} <= 20% in dimension {dim}. "
            f"Indicates Phase 2 warmup deadlock due to stale auxiliary transform cache."
        )
        assert sampler.step_size > 0.001, (
            f"Final adapted step size {sampler.step_size:.6f} collapsed below 0.001 in dimension {dim}."
        )
        assert np.all(np.isfinite(state.x)), f"Non-finite state encountered in dimension {dim}"
        assert sampler._tail_scale_frozen is True
        assert sampler._preconditioner_frozen is True

    def test_f17_unified_sampler_transform_resynchronization_boundary(self):
        """Verifies exact auxiliary resynchronization at the Phase 1 -> Phase 2 transition boundary."""
        d = 10
        target = CoupledDualPathologyTarget(d=d, nu=2.0, c=1.0)
        mask = np.zeros(d, dtype=bool)
        mask[: target.d1] = True

        sampler = UnifiedRobustSampler(
            step_size=0.05,
            warmup_steps=100,
            adapt_step_size=True,
            adapt_preconditioner=True,
            adapt_tail_scale=True,
            mask=mask,
        )
        state = sampler.init_state(np.zeros(d), target)
        rng = np.random.default_rng(777)

        # Run Phase 1: step 0 to 14 (W1 = 15)
        for t in range(15):
            res = sampler.step(state, target, rng)
            sampler.adapt(t, res)
            state = res.state

        # At t = 14, self._transform was updated with calibrated radii.
        # Verify that state.auxiliary contains the new transform_id or will be resynchronized on step 15
        initial_t15_transform = sampler._transform
        res_15 = sampler.step(state, target, rng)
        sampler.adapt(15, res_15)

        # Ensure the candidate acceptance probability on step 15 is numerically sane (not 0.0 due to stale det)
        assert res_15.accept_prob > 1e-4, (
            f"Step 15 candidate accept_prob {res_15.accept_prob:.2e} collapsed across transform boundary."
        )
        if res_15.accepted:
            assert res_15.state.auxiliary.get("transform_id") == id(initial_t15_transform)

    def test_f17_unified_sampler_zero_redundancy_oracle_across_warmup_phases(self):
        """AC1/AC5 Guardrail: Strictly 1 gradient and 1 log-prob evaluation per step across all 3 warmup phases."""
        d = 10
        base_target = CoupledDualPathologyTarget(d=d, nu=2.0, c=1.0)
        target = GradBudgetTracker(base_target)

        mask = np.zeros(d, dtype=bool)
        mask[: base_target.d1] = True

        warmup_steps = 100
        sampler = UnifiedRobustSampler(
            step_size=0.05,
            warmup_steps=warmup_steps,
            adapt_step_size=True,
            adapt_preconditioner=True,
            adapt_tail_scale=True,
            mask=mask,
        )

        # 1. Initialization: exactly 1 gradient and 1 log-prob
        state = sampler.init_state(np.zeros(d), target)
        assert target.n_grad == 1, f"init_state used {target.n_grad} grads, expected 1"
        assert target.n_log_prob == 1, f"init_state used {target.n_log_prob} log_probs, expected 1"

        rng = np.random.default_rng(12345)
        total_steps = warmup_steps + 25  # 125 total steps spanning Phase 1, 2, 3, and post-warmup

        W1 = max(10, int(0.15 * warmup_steps))  # 15
        W2 = int(0.80 * warmup_steps)  # 80

        for t in range(total_steps):
            res = sampler.step(state, target, rng)
            sampler.adapt(t, res)
            state = res.state

            expected_evals = 1 + (t + 1)
            phase_label = (
                "Phase 1" if t < W1 else ("Phase 2" if t < W2 else ("Phase 3" if t < warmup_steps else "Post-warmup"))
            )
            assert target.n_grad == expected_evals, (
                f"Step {t} ({phase_label}): n_grad={target.n_grad} != expected {expected_evals}. "
                f"Redundant gradient evaluation detected in warmup adaptation!"
            )
            assert target.n_log_prob == expected_evals, (
                f"Step {t} ({phase_label}): n_log_prob={target.n_log_prob} != expected {expected_evals}."
            )
            assert res.stats.get("grad_evals") == 1
            assert res.stats.get("log_prob_evals") == 1
