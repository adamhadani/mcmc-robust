"""Unit tests for space transformations and TransformedTarget (Feature F14)."""

import numpy as np
import pytest

from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.targets.benign import IsotropicGaussian
from mcmc_robust.targets.heavy_tailed import MultivariateStudentTTarget
from mcmc_robust.transforms.base import TransformedTarget
from mcmc_robust.transforms.radial import RadialDiffeomorphicTransform
from mcmc_robust.transforms.stereographic import StereographicTransform


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 10, 50, 100])
def test_stereographic_roundtrip_precision(d, rng):
    """Test F14.1: Stereographic projection round-trip inversion error < 1e-10."""
    stereo = StereographicTransform(d=d, scale=2.0)
    assert stereo.input_dim == d
    assert stereo.output_dim == d + 1

    for norm in [1e-4, 0.1, 1.0, 2.0, 10.0, 100.0, 1000.0]:
        x0 = rng.normal(size=d)
        x = (x0 / np.linalg.norm(x0)) * norm

        z, _ = stereo.forward(x)
        x_rec, _ = stereo.inverse(z)

        inv_err = float(np.max(np.abs(x - x_rec)))
        assert inv_err < 1e-10, f"Stereo inv error d={d}, norm={norm}: {inv_err}"

        # Sphere constraint ||z|| == 1
        z_norm_err = abs(np.linalg.norm(z) - 1.0)
        assert z_norm_err < 1e-12, f"Spherical constraint error: {z_norm_err}"


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 10, 50, 100])
def test_stereographic_jacobian_consistency(d, rng):
    """Test F14.2: Stereographic forward and inverse log-det Jacobians sum to 0 (< 1e-10)."""
    stereo = StereographicTransform(d=d, scale=1.5)

    for norm in [1e-3, 0.5, 1.5, 5.0, 50.0]:
        x0 = rng.normal(size=d)
        x = (x0 / np.linalg.norm(x0)) * norm

        z, fwd_det = stereo.forward(x)
        _, rev_det = stereo.inverse(z)

        det_err = float(abs(fwd_det + rev_det))
        assert det_err < 1e-10, f"Stereo det error d={d}, norm={norm}: {det_err}"


@pytest.mark.unit
def test_stereographic_tangent_gradient(rng):
    """Test F14.3: Stereographic forward_grad produces strictly tangent vectors on S^d."""
    d = 4
    stereo = StereographicTransform(d=d, scale=1.0)

    for _ in range(10):
        x = rng.normal(size=d) * 3.0
        gx = rng.normal(size=d)

        z, _ = stereo.forward(x)
        g_tan = stereo.forward_grad(x, gx)

        # Tangent property: <z, g_tan> == 0
        inner = float(np.dot(z, g_tan))
        assert abs(inner) < 1e-12, f"Tangent inner product non-zero: {inner}"


@pytest.mark.unit
def test_stereographic_north_pole_singularity():
    """Test F14.4: Inversion at the North pole raises ValueError."""
    d = 3
    stereo = StereographicTransform(d=d, scale=1.0)
    z_north = np.zeros(d + 1)
    z_north[-1] = 1.0

    with pytest.raises(ValueError, match="North pole"):
        stereo.inverse(z_north)


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 10, 50, 100])
def test_radial_diffeo_roundtrip_precision(d, rng):
    """Test F14.5: Radial diffeomorphic round-trip inversion error < 1e-10."""
    rad = RadialDiffeomorphicTransform(d=d, radius=3.0, stiffness=0.5)
    assert rad.input_dim == d
    assert rad.output_dim == d

    for norm in [0.05, 1.0, 2.999, 3.0, 3.001, 5.0, 20.0, 100.0, 1000.0]:
        x0 = rng.normal(size=d)
        x = (x0 / np.linalg.norm(x0)) * norm

        y, _ = rad.forward(x)
        x_rec, _ = rad.inverse(y)

        inv_err = float(np.max(np.abs(x - x_rec)))
        assert inv_err < 1e-10, f"Radial inv error d={d}, norm={norm}: {inv_err}"


@pytest.mark.unit
def test_radial_diffeo_bulk_identity():
    """Test F14.6: Radial transform inside bulk ||x|| <= R is strict identity."""
    d = 3
    R = 4.0
    rad = RadialDiffeomorphicTransform(d=d, radius=R, stiffness=0.5)

    x = np.array([1.0, -1.0, 0.5])  # norm = 1.5 <= 4.0
    y, log_det_fwd = rad.forward(x)
    assert np.allclose(y, x, atol=1e-15)
    assert log_det_fwd == 0.0

    x_rec, log_det_inv = rad.inverse(x)
    assert np.allclose(x_rec, x, atol=1e-15)
    assert log_det_inv == 0.0

    gx = np.array([0.2, -0.3, 0.4])
    gy = rad.forward_grad(x, gx)
    assert np.allclose(gy, gx, atol=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 10, 50, 100])
def test_radial_diffeo_jacobian_consistency(d, rng):
    """Test F14.7: Radial forward and inverse log-det Jacobians sum to 0 (< 1e-10)."""
    rad = RadialDiffeomorphicTransform(d=d, radius=2.5, stiffness=0.6)

    for norm in [0.5, 2.5, 3.0, 10.0, 100.0]:
        x0 = rng.normal(size=d)
        x = (x0 / np.linalg.norm(x0)) * norm

        y, fwd_det = rad.forward(x)
        _, rev_det = rad.inverse(y)

        det_err = float(abs(fwd_det + rev_det))
        assert det_err < 1e-10, f"Radial det error d={d}, norm={norm}: {det_err}"


@pytest.mark.unit
def test_radial_diffeo_c2_smoothness():
    """Test F14.8: C^2 smoothness at r = R (phi(R)=R, phi'(R)=1, phi''(R)=0)."""
    R = 3.0
    rad = RadialDiffeomorphicTransform(d=2, radius=R, stiffness=0.5)

    for dr in [1e-12, 1e-8, 1e-5]:
        val, d1, d2 = rad._phi(R + dr)
        assert abs(val - (R + dr)) < 1e-10
        assert abs(d1 - 1.0) < 1e-4
        assert abs(d2) < 1e-3


@pytest.mark.unit
def test_transformed_target_interface(rng):
    """Test F14.9: TransformedTarget fulfills TargetDistribution contracts."""
    d = 3
    base_target = IsotropicGaussian(d=d)
    transform = RadialDiffeomorphicTransform(d=d, radius=2.0, stiffness=0.5)
    transformed = TransformedTarget(base_target, transform)

    assert transformed.d == d
    assert transformed.target is base_target
    assert transformed.transform is transform

    y = np.array([1.0, 2.0, -1.0])
    lp = transformed.log_prob(y)
    grad = transformed.grad_log_prob(y)
    lp_joint, grad_joint = transformed.log_prob_and_grad(y)

    assert np.isclose(lp, lp_joint, atol=1e-12)
    assert np.allclose(grad, grad_joint, atol=1e-12)

    # Moments and ground truth sampling
    info = transformed.exact_moments_or_quantiles()
    assert info["type"] == "TransformedTarget"

    samples = transformed.sample_ground_truth(100, rng=rng)
    assert samples.shape == (100, d)


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 5, 10])
def test_transformed_target_finite_differences(d, rng):
    """Test F14.10: TransformedTarget analytical gradient matches finite differences (< 1e-5)."""
    base_target = IsotropicGaussian(d=d)
    transform = RadialDiffeomorphicTransform(d=d, radius=2.0, stiffness=0.4)
    transformed = TransformedTarget(base_target, transform)

    for norm in [1.5, 3.5, 12.0]:
        x0 = rng.normal(size=d)
        y = (x0 / np.linalg.norm(x0)) * norm

        _, grad_anal = transformed.log_prob_and_grad(y)

        # Central finite differences
        eps = 1e-6
        grad_fd = np.empty(d, dtype=np.float64)
        for i in range(d):
            y_plus = y.copy()
            y_plus[i] += eps
            y_minus = y.copy()
            y_minus[i] -= eps
            grad_fd[i] = (transformed.log_prob(y_plus) - transformed.log_prob(y_minus)) / (2.0 * eps)

        diff = float(np.max(np.abs(grad_anal - grad_fd)))
        rel_err = diff / float(max(1.0, np.max(np.abs(grad_anal))))
        assert rel_err < 1e-5, f"Gradient FD error d={d}, norm={norm}: {rel_err}"


@pytest.mark.unit
def test_transformed_target_mcmc_compatibility(rng):
    """Test F14.11: TransformedTarget works with GradBudgetTracker and MALA."""
    d = 2
    base_target = MultivariateStudentTTarget(d=d, nu=2.5)
    transform = RadialDiffeomorphicTransform(d=d, radius=2.0, stiffness=0.5)
    transformed = TransformedTarget(base_target, transform)

    tracker = GradBudgetTracker(transformed)
    sampler = MALA(step_size=0.1)

    state = sampler.init_state(np.array([0.0, 0.0]), tracker)
    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1

    n_steps = 20
    for _ in range(n_steps):
        res = sampler.step(state, tracker, rng)
        state = res.state

    assert tracker.n_grad == 1 + n_steps
    assert tracker.n_log_prob == 1 + n_steps
