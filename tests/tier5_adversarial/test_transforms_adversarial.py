"""Tier 5 Adversarial Tests: Transformations, Diffeomorphisms, and Jacobians.

Verifies round-trip precision (< 1e-10), log-determinant cancellation (< 1e-10),
pullback gradient correctness, boundary behavior, and numerical stabilization
under extreme inputs across all transformation classes.
"""

import numpy as np
import pytest

from mcmc_robust.targets.benign import IsotropicGaussian
from mcmc_robust.transforms.base import TransformedTarget
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform
from mcmc_robust.transforms.radial import RadialDiffeomorphicTransform
from mcmc_robust.transforms.stereographic import StereographicTransform


class TestTransformPrecisionAndJacobians:
    """Verifies round-trip invertibility and Jacobian cancellation to machine precision."""

    @pytest.mark.parametrize("d", [2, 10, 50, 100])
    @pytest.mark.parametrize("magnitude", [1e-6, 1.0, 10.0, 100.0, 1e4])
    def test_stereographic_roundtrip_and_logdet(self, d: int, magnitude: float) -> None:
        transform = StereographicTransform(d=d, scale=2.5)
        rng = np.random.default_rng(42 + d)
        x = rng.normal(size=d)
        x = (magnitude / np.linalg.norm(x)) * x

        z, log_det_fwd = transform.forward(x)
        assert z.shape == (d + 1,)
        assert np.isclose(np.linalg.norm(z), 1.0, atol=1e-12)

        x_rec, log_det_inv = transform.inverse(z)
        rel_inv_error = np.max(np.abs(x_rec - x)) / max(1.0, magnitude)
        assert rel_inv_error < 1e-10, f"Stereographic round-trip failed: {rel_inv_error}"

        cancellation_error = abs(log_det_fwd + log_det_inv)
        assert cancellation_error < 1e-10, f"Stereographic log-det failed: {cancellation_error}"

    @pytest.mark.parametrize("d", [2, 10, 50, 100])
    @pytest.mark.parametrize("magnitude", [0.5, 3.0, 3.0001, 10.0, 100.0, 1000.0])
    def test_radial_diffeo_roundtrip_and_logdet(self, d: int, magnitude: float) -> None:
        R = 3.0
        k = 0.5
        transform = RadialDiffeomorphicTransform(d=d, radius=R, stiffness=k)
        rng = np.random.default_rng(100 + d)
        x = rng.normal(size=d)
        x = (magnitude / np.linalg.norm(x)) * x

        y, log_det_fwd = transform.forward(x)
        assert y.shape == (d,)

        x_rec, log_det_inv = transform.inverse(y)
        rel_inv_error = np.max(np.abs(x_rec - x)) / max(1.0, magnitude)
        assert rel_inv_error < 1e-10, f"Radial round-trip failed: {rel_inv_error}"

        cancellation_error = abs(log_det_fwd + log_det_inv)
        assert cancellation_error < 1e-10, f"Radial log-det failed: {cancellation_error}"

    @pytest.mark.parametrize("d", [1, 2, 10, 50, 100])
    @pytest.mark.parametrize("magnitude", [1e-8, 1.0, 10.0, 1e4, 1e8, 1e15])
    def test_coordinate_arcsinh_roundtrip_and_logdet(self, d: int, magnitude: float) -> None:
        rng = np.random.default_rng(200 + d)
        radii = np.exp(rng.uniform(-2.0, 2.0, size=d)) if d > 1 else np.array([1.5])
        mask = rng.choice([True, False], size=d) if d > 1 else np.array([True])
        mask[0] = True  # ensure at least one coordinate transformed

        transform = CoordinateWiseArcsinhTransform(d=d, radii=radii, mask=mask)
        x = rng.uniform(-magnitude, magnitude, size=d)

        z, log_det_fwd = transform.forward(x)
        assert z.shape == (d,)

        # For magnitudes <= 1e8, check exact invertibility
        if magnitude <= 1e8:
            x_rec, log_det_inv = transform.inverse(z)
            rel_inv_error = np.max(np.abs(x_rec - x)) / max(1.0, magnitude)
            assert rel_inv_error < 1e-10, f"Coordinate arcsinh round-trip failed: {rel_inv_error}"

            cancellation_error = abs(log_det_fwd + log_det_inv)
            assert cancellation_error < 1e-10, f"Coordinate arcsinh log-det failed: {cancellation_error}"
        else:
            # Extreme magnitude forward stability (no NaN/Inf)
            assert np.all(np.isfinite(z))
            assert np.isfinite(log_det_fwd)


class TestTransformBoundariesAndGradients:
    """Verifies edge singularities, C2 transitions, and pullback gradients."""

    def test_stereographic_north_pole_singularity(self) -> None:
        d = 5
        st = StereographicTransform(d=d, scale=1.0)
        # Exact north pole z = [0, 0, 0, 0, 0, 1]
        z_north = np.zeros(d + 1)
        z_north[-1] = 1.0
        with pytest.raises(ValueError, match="singular under stereographic inverse"):
            st.inverse(z_north)

        # Near north pole: 1 - 1e-16
        z_near = np.zeros(d + 1)
        z_near[-1] = 1.0 - 1e-16
        with pytest.raises(ValueError, match="singular under stereographic inverse"):
            st.inverse(z_near)

    def test_radial_diffeo_boundary_c2_smoothness(self) -> None:
        R = 4.0
        k = 0.5
        rd = RadialDiffeomorphicTransform(d=3, radius=R, stiffness=k)

        # Inside bulk (r < R)
        v_in, d1_in, d2_in = rd._phi(R - 1e-8)
        assert np.isclose(v_in, R - 1e-8)
        assert d1_in == 1.0
        assert d2_in == 0.0

        # At boundary r = R
        v_at, d1_at, d2_at = rd._phi(R)
        assert v_at == R
        assert d1_at == 1.0
        assert d2_at == 0.0

        # Slightly outside (r = R + epsilon)
        eps = 1e-6
        v_out, d1_out, d2_out = rd._phi(R + eps)
        assert np.isclose(v_out, R + eps, atol=1e-9)
        assert np.isclose(d1_out, 1.0, atol=1e-5)
        assert np.isclose(d2_out, 0.0, atol=1e-4)

    def test_transformed_target_pullback_gradient_fd(self) -> None:
        """Verifies transformed gradient equals numerical finite differences."""
        d = 3
        base = IsotropicGaussian(d=d, std=1.5)
        transform = CoordinateWiseArcsinhTransform(d=d, radii=[1.0, 2.0, 3.0])
        tt = TransformedTarget(base, transform)

        rng = np.random.default_rng(999)
        y = rng.normal(size=d) * 1.5

        lp_ana, grad_ana = tt.log_prob_and_grad(y)

        # Central finite difference on transformed space y
        eps = 1e-7
        grad_num = np.empty(d)
        for i in range(d):
            y_pos = y.copy()
            y_neg = y.copy()
            y_pos[i] += eps
            y_neg[i] -= eps
            grad_num[i] = (tt.log_prob(y_pos) - tt.log_prob(y_neg)) / (2.0 * eps)

        assert np.allclose(grad_ana, grad_num, atol=1e-5, rtol=1e-5)

    def test_calibration_from_pathological_samples(self) -> None:
        """Verifies online calibration under degenerate sample matrices."""
        d = 10
        # 1. Constant samples (IQR = 0)
        constant_samples = np.full((30, d), 5.0)
        calib_const = CoordinateWiseArcsinhTransform.calibrate_from_samples(constant_samples)
        assert calib_const.d == d
        assert np.all(calib_const.radii >= 1e-4)
        assert np.all(np.isfinite(calib_const.radii))

        # 2. Too few samples (< 20) raises ValueError
        few_samples = np.ones((10, d))
        with pytest.raises(ValueError, match="Need at least 20 samples"):
            CoordinateWiseArcsinhTransform.calibrate_from_samples(few_samples)
