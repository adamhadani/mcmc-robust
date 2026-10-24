"""Unit tests for HierarchicalArcsinhTransform and funnel-coupled Hamiltonian samplers."""

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.samplers.novel.nuts import UnifiedRobustNUTS
from mcmc_robust.samplers.novel.unified_hmc import UnifiedRobustHMC
from mcmc_robust.targets.benign import IsotropicGaussian
from mcmc_robust.targets.roughness import NealsFunnelTarget
from mcmc_robust.transforms.base import TransformedTarget
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform
from mcmc_robust.transforms.hierarchical_arcsinh import HierarchicalArcsinhTransform


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 10, 50])
@pytest.mark.parametrize("parent_idx", [0, 1])
def test_hierarchical_arcsinh_roundtrip_precision(d: int, parent_idx: int) -> None:
    """Verify forward-inverse roundtrip precision (< 1e-12) across d in {2, 10, 50}."""
    rng = np.random.default_rng(42 + d + parent_idx)
    location = rng.normal(loc=0.5, scale=1.5, size=d)
    radii = rng.uniform(0.5, 3.5, size=d)
    coupling = rng.uniform(-0.8, 0.8, size=d)

    transform = HierarchicalArcsinhTransform(
        d=d,
        radii=radii,
        location=location,
        coupling=coupling,
        parent_idx=parent_idx,
        parent_scale=2.5,
        max_log_scale=4.0,
    )

    assert transform.d == d
    assert transform.parent_idx == parent_idx
    assert transform.coupling[parent_idx] == 0.0

    for scale in [1e-3, 0.2, 1.0, 3.0, 10.0, 50.0]:
        x = location + rng.normal(size=d) * scale
        z, log_det_fwd = transform.forward(x)
        x_rec, log_det_inv = transform.inverse(z)
        z_rec, _ = transform.forward(x_rec)

        err_x = float(np.max(np.abs(x - x_rec)))
        err_z = float(np.max(np.abs(z - z_rec)))
        err_det = float(abs(log_det_fwd + log_det_inv))

        assert err_x < 1e-12, f"Roundtrip x error {err_x} >= 1e-12 for d={d}, scale={scale}"
        assert err_z < 1e-12, f"Roundtrip z error {err_z} >= 1e-12 for d={d}, scale={scale}"
        assert err_det < 1e-12, f"Log-det cancellation error {err_det} >= 1e-12 for d={d}"


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 5, 10])
@pytest.mark.parametrize("parent_idx", [0, 1])
def test_hierarchical_arcsinh_finite_difference_jacobian_and_grad(
    d: int,
    parent_idx: int,
) -> None:
    """Verify log_det_jacobian and forward_grad against central finite differences (< 1e-7)."""
    rng = np.random.default_rng(123 + d + parent_idx)
    location = rng.normal(loc=-0.3, scale=0.8, size=d)
    radii = rng.uniform(0.8, 2.5, size=d)
    coupling = rng.uniform(-0.6, 0.6, size=d)

    transform = HierarchicalArcsinhTransform(
        d=d,
        radii=radii,
        location=location,
        coupling=coupling,
        parent_idx=parent_idx,
        parent_scale=2.0,
        max_log_scale=3.5,
    )

    target = NealsFunnelTarget(d=d, sigma_v=3.0)
    eps = 1e-6

    for _ in range(5):
        z = rng.normal(scale=0.8, size=d)
        x, log_det_inv = transform.inverse(z)
        _, log_det_fwd = transform.forward(x)

        # 1. Numerical Jacobian matrix of forward and inverse maps
        j_fwd = np.empty((d, d), dtype=np.float64)
        j_inv = np.empty((d, d), dtype=np.float64)
        for col in range(d):
            xp = x.copy()
            xm = x.copy()
            xp[col] += eps
            xm[col] -= eps
            zp_f, _ = transform.forward(xp)
            zm_f, _ = transform.forward(xm)
            j_fwd[:, col] = (zp_f - zm_f) / (2.0 * eps)

            zp = z.copy()
            zm = z.copy()
            zp[col] += eps
            zm[col] -= eps
            xp_i, _ = transform.inverse(zp)
            xm_i, _ = transform.inverse(zm)
            j_inv[:, col] = (xp_i - xm_i) / (2.0 * eps)

        _, num_log_det_fwd = np.linalg.slogdet(j_fwd)
        _, num_log_det_inv = np.linalg.slogdet(j_inv)

        rel_err_fwd = abs(log_det_fwd - float(num_log_det_fwd)) / max(1.0, abs(float(num_log_det_fwd)))
        rel_err_inv = abs(log_det_inv - float(num_log_det_inv)) / max(1.0, abs(float(num_log_det_inv)))
        assert rel_err_fwd < 1e-7, f"Forward log_det relative error {rel_err_fwd} >= 1e-7"
        assert rel_err_inv < 1e-7, f"Inverse log_det relative error {rel_err_inv} >= 1e-7"

        # 2. Numerical gradient of log \tilde{\pi}(z) = log \pi(\Phi^{-1}(z)) + log|det D\Phi^{-1}(z)|
        def log_prob_z(z_vec: np.ndarray) -> float:
            x_vec, ld_inv = transform.inverse(z_vec)
            return float(target.log_prob(x_vec) + ld_inv)

        grad_x = target.grad_log_prob(x)
        anal_grad_z = transform.forward_grad(x, grad_x)

        num_grad_z = np.empty(d, dtype=np.float64)
        for i in range(d):
            zp = z.copy()
            zm = z.copy()
            zp[i] += eps
            zm[i] -= eps
            num_grad_z[i] = (log_prob_z(zp) - log_prob_z(zm)) / (2.0 * eps)

        rel_grad_err = float(np.linalg.norm(anal_grad_z - num_grad_z) / max(1.0, float(np.linalg.norm(num_grad_z))))
        assert rel_grad_err < 1e-7, f"forward_grad relative error {rel_grad_err} >= 1e-7"


@pytest.mark.unit
@pytest.mark.parametrize("d", [2, 10, 50])
def test_hierarchical_arcsinh_zero_coupling_matches_coordinate_wise(d: int) -> None:
    """Verify exact reduction to CoordinateWiseArcsinhTransform when coupling = 0."""
    rng = np.random.default_rng(999 + d)
    location = rng.normal(size=d)
    radii = rng.uniform(0.5, 4.0, size=d)
    mask = rng.choice([True, False], size=d, p=[0.8, 0.2])
    mask[0] = True

    coord_tf = CoordinateWiseArcsinhTransform(
        d=d,
        radii=radii,
        location=location,
        mask=mask,
    )
    hier_tf = HierarchicalArcsinhTransform(
        d=d,
        radii=radii,
        location=location,
        coupling=0.0,
        mask=mask,
    )

    x = rng.normal(scale=3.0, size=d)
    grad_x = rng.normal(size=d)

    z_c, ld_fwd_c = coord_tf.forward(x)
    z_h, ld_fwd_h = hier_tf.forward(x)
    assert np.allclose(z_c, z_h, atol=1e-14, rtol=1e-14)
    assert abs(ld_fwd_c - ld_fwd_h) < 1e-13

    x_c, ld_inv_c = coord_tf.inverse(z_c)
    x_h, ld_inv_h = hier_tf.inverse(z_h)
    assert np.allclose(x_c, x_h, atol=1e-14, rtol=1e-14)
    assert abs(ld_inv_c - ld_inv_h) < 1e-13

    gz_c = coord_tf.forward_grad(x, grad_x)
    gz_h = hier_tf.forward_grad(x, grad_x)
    assert np.allclose(gz_c, gz_h, atol=1e-14, rtol=1e-14)


@pytest.mark.unit
def test_calibrate_from_samples_funnel_vs_gaussian() -> None:
    """Verify calibrate_from_samples detects positive coupling on Neal's Funnel and 0 on Gaussian."""
    d = 10
    rng = np.random.default_rng(2026)

    # 1. Neal's Funnel ground-truth draws
    funnel = NealsFunnelTarget(d=d, sigma_v=3.0)
    funnel_samples = funnel.sample_ground_truth(2000, rng=rng)
    tf_funnel = HierarchicalArcsinhTransform.calibrate_from_samples(funnel_samples, parent_idx=0)

    assert tf_funnel.coupling[0] == 0.0
    assert np.all(tf_funnel.coupling[1:] > 0.2), (
        f"Expected positive funnel coupling > 0.2 on child coordinates, got {tf_funnel.coupling[1:]}"
    )

    # 2. Isotropic Gaussian ground-truth draws
    gauss = IsotropicGaussian(d=d, std=1.5)
    gauss_samples = gauss.sample_ground_truth(2000, rng=rng)
    tf_gauss = HierarchicalArcsinhTransform.calibrate_from_samples(gauss_samples, parent_idx=0)

    assert np.all(tf_gauss.coupling == 0.0), (
        f"Expected zero coupling on IsotropicGaussian draws, got {tf_gauss.coupling}"
    )


@pytest.mark.unit
def test_unified_robust_nuts_funnel_coupling_neals_funnel() -> None:
    """Verify UnifiedRobustNUTS(funnel_coupling=True) on NealsFunnelTarget(d=10).

    Achieves zero post-warmup divergences and accurate deep-neck exploration P(v < -3).
    """
    d = 10
    target = NealsFunnelTarget(d=d, sigma_v=3.0)
    rng = np.random.default_rng(31415)

    sampler = UnifiedRobustNUTS(
        step_size=0.1,
        max_depth=8,
        adapt_step_size=True,
        adapt_preconditioner=True,
        adapt_tail_scale=True,
        funnel_coupling=True,
        funnel_parent_idx=0,
        warmup_steps=500,
    )

    x0 = np.zeros(d, dtype=np.float64)
    samples, results = sampler.sample(
        x0=x0,
        target=target,
        n_steps=1500,
        burn_in=500,
        rng=rng,
        adapt=True,
    )

    assert isinstance(sampler.transform, HierarchicalArcsinhTransform)
    assert np.all(sampler.transform.coupling[1:] > 0.1)

    post_results = results[500:]
    n_divergent = sum(int(r.stats["divergent"] > 0.0) for r in post_results)
    assert n_divergent == 0, f"Expected 0 post-warmup divergences, got {n_divergent}"

    v_samples = samples[:, 0]
    true_prob_neck = float(stats.norm.cdf(-3.0, loc=0.0, scale=3.0))  # ~0.158655
    emp_prob_neck = float(np.mean(v_samples < -3.0))
    assert 0.07 < emp_prob_neck < 0.28, (
        f"Deep-neck probability P(v < -3) = {emp_prob_neck:.3f} far from true {true_prob_neck:.3f}"
    )
    assert abs(float(np.mean(v_samples))) < 0.8
    assert 2.0 < float(np.std(v_samples)) < 4.0


@pytest.mark.unit
def test_unified_robust_hmc_funnel_coupling_neals_funnel() -> None:
    """Verify UnifiedRobustHMC(funnel_coupling=True) calibrates HierarchicalArcsinhTransform."""
    d = 10
    target = NealsFunnelTarget(d=d, sigma_v=3.0)
    rng = np.random.default_rng(27182)

    sampler = UnifiedRobustHMC(
        step_size=0.3,
        n_leapfrog=8,
        adapt_step_size=True,
        adapt_preconditioner=True,
        adapt_tail_scale=True,
        funnel_coupling=True,
        funnel_parent_idx=0,
        warmup_steps=400,
    )

    samples, results = sampler.sample(
        x0=np.zeros(d, dtype=np.float64),
        target=target,
        n_steps=1000,
        burn_in=400,
        rng=rng,
        adapt=True,
    )

    assert isinstance(sampler.transform, HierarchicalArcsinhTransform)
    assert np.all(sampler.transform.coupling[1:] > 0.1)
    post_results = results[400:]
    n_divergent = sum(int(r.stats["divergent"] > 0.0) for r in post_results)
    assert n_divergent == 0
    assert float(np.mean(samples[:, 0] < -3.0)) > 0.05


@pytest.mark.unit
def test_hierarchical_arcsinh_validation_and_transformed_target() -> None:
    """Verify input validation guards and TransformedTarget integration."""
    with pytest.raises(ValueError, match="parent_idx"):
        HierarchicalArcsinhTransform(d=3, parent_idx=3)
    with pytest.raises(ValueError, match="parent_scale"):
        HierarchicalArcsinhTransform(d=3, parent_scale=0.0)
    with pytest.raises(ValueError, match="max_log_scale"):
        HierarchicalArcsinhTransform(d=3, max_log_scale=-1.0)
    with pytest.raises(ValueError, match="radii"):
        HierarchicalArcsinhTransform(d=3, radii=np.array([1.0, -1.0, 1.0]))

    target = NealsFunnelTarget(d=3, sigma_v=3.0)
    tf = HierarchicalArcsinhTransform(d=3, coupling=[0.0, 0.4, 0.4], parent_scale=3.0)
    tt = TransformedTarget(target, tf)

    z = np.array([-0.5, 0.3, -0.2], dtype=np.float64)
    lp, grad = tt.log_prob_and_grad(z)
    assert np.isfinite(lp)
    assert np.all(np.isfinite(grad))
    assert np.isclose(lp, tt.log_prob(z))
    assert np.allclose(grad, tt.grad_log_prob(z))
