"""Unit tests for target distributions across all five regimes."""

import time

import numpy as np
import pytest
from scipy import special, stats
from scipy.integrate import quad

from mcmc_robust.targets.benign import (
    DiagonalGaussian,
    FullCovarianceGaussian,
    IsotropicGaussian,
)
from mcmc_robust.targets.dual import (
    ConfinementInducedRoughnessTarget,
    ConfinementRoughTarget,
    CoupledDualPathologyTarget,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)
from mcmc_robust.targets.roughness import (
    L0L1SmoothTarget,
    NealsFunnelTarget,
    QuarticTarget,
    RippledGaussianTarget,
    SuperlinearPolynomialTarget,
)
from mcmc_robust.targets.suite import get_target, list_targets
from tests.conftest import max_relative_error, numerical_gradient


@pytest.mark.unit
def test_gaussian_target_finite_difference():
    """Test 1: Finite-difference gradient relative error < 10^-5 across Gaussians."""
    # Isotropic
    iso = IsotropicGaussian(d=3, mean=np.array([1.0, -1.0, 0.5]), std=1.5)
    x_test = np.array([2.0, 0.5, -1.0])
    anal_grad = iso.grad_log_prob(x_test)
    num_grad = numerical_gradient(iso, x_test)
    assert max_relative_error(anal_grad, num_grad) < 1e-5

    # Diagonal anisotropic across condition numbers
    for kappa in [1.0, 10.0, 100.0]:
        diag_target = DiagonalGaussian(d=4, condition_number=kappa)
        x = np.array([0.5, -1.2, 2.0, -0.8])
        assert max_relative_error(diag_target.grad_log_prob(x), numerical_gradient(diag_target, x)) < 1e-5

    # Full covariance
    full_target = FullCovarianceGaussian(d=3, condition_number=20.0)
    x = np.array([1.5, -0.5, 0.8])
    assert max_relative_error(full_target.grad_log_prob(x), numerical_gradient(full_target, x)) < 1e-5


@pytest.mark.unit
def test_gaussian_target_analytical_moments(rng):
    """Test 2: Empirical sample mean and covariance match analytical ground truth."""
    d = 3
    mean = np.array([0.5, -0.5, 1.0])
    std = 1.2
    target = IsotropicGaussian(d=d, mean=mean, std=std)

    n_samples = 5000
    samples = target.sample_ground_truth(n_samples, rng=rng)

    emp_mean = np.mean(samples, axis=0)
    emp_cov = np.cov(samples, rowvar=False)

    moments = target.exact_moments_or_quantiles()
    assert np.allclose(emp_mean, moments["mean"], atol=3.5 * std / np.sqrt(n_samples))
    assert np.allclose(emp_cov, moments["cov"], atol=0.15)


@pytest.mark.unit
def test_gaussian_target_ground_truth_ks(rng):
    """Test 3: Ground truth 1D marginals pass Kolmogorov-Smirnov test (p > 0.05)."""
    d = 3
    mean = np.array([1.0, 0.0, -1.0])
    stds = np.array([0.5, 1.0, 2.0])
    target = DiagonalGaussian(d=d, mean=mean, stds=stds)

    samples = target.sample_ground_truth(3000, rng=rng)
    for j in range(d):
        cdf = stats.norm(loc=mean[j], scale=stds[j]).cdf
        res = stats.kstest(samples[:, j], cdf)
        assert res.pvalue > 0.05, f"Coordinate {j} failed KS test with p={res.pvalue}"


@pytest.mark.unit
def test_quartic_target_finite_difference():
    """Test 4: Superlinear potential gradient relative error < 10^-5 at mode, bulk, tail."""
    target_p4 = QuarticTarget(d=2, p=4.0)
    target_p6 = SuperlinearPolynomialTarget(d=2, p=6.0)

    for target in [target_p4, target_p6]:
        # 1. Mode
        x_mode = np.zeros(2)
        assert np.allclose(target.grad_log_prob(x_mode), np.zeros(2))

        # 2. Bulk
        x_bulk = np.array([0.8, -0.6])
        assert max_relative_error(target.grad_log_prob(x_bulk), numerical_gradient(target, x_bulk)) < 1e-5

        # 3. Tail
        x_tail = np.array([5.0, -5.0])
        assert max_relative_error(target.grad_log_prob(x_tail), numerical_gradient(target, x_tail)) < 1e-5


@pytest.mark.unit
def test_quartic_target_exact_sampling_ks(rng):
    """Test 5: QuarticTarget ground truth samples pass 1D and multi-D radial KS tests."""
    # 1D Quartic test against exact incomplete gamma CDF
    target_1d = QuarticTarget(d=1, p=4.0)
    samples_1d = target_1d.sample_ground_truth(3000, rng=rng).ravel()

    def quartic_cdf(x):
        return 0.5 + 0.5 * np.sign(x) * special.gammainc(0.25, (x**4) / 4.0)

    res_1d = stats.kstest(samples_1d, quartic_cdf)
    assert res_1d.pvalue > 0.05, f"1D Quartic KS test failed: p={res_1d.pvalue}"

    # Multi-D Radial distribution test: ||X||^4 / 4 ~ Gamma(shape=d/4, scale=1)
    d = 4
    target_nd = QuarticTarget(d=d, p=4.0)
    samples_nd = target_nd.sample_ground_truth(3000, rng=rng)
    r4_over_4 = (np.linalg.norm(samples_nd, axis=1) ** 4) / 4.0
    res_nd = stats.kstest(r4_over_4, stats.gamma(a=d / 4.0, scale=1.0).cdf)
    assert res_nd.pvalue > 0.05, f"Multi-D Quartic Radial KS test failed: p={res_nd.pvalue}"


@pytest.mark.unit
def test_l0l1_smooth_target_finite_difference():
    """Test 6: (L0, L1)-smooth target gradient rel err < 10^-5 and curvature bound."""
    target = L0L1SmoothTarget(d=3, L0=1.5, lambda_param=2.0)
    x = np.array([0.5, -1.0, 1.2])

    anal_grad = target.grad_log_prob(x)
    num_grad = numerical_gradient(target, x)
    assert max_relative_error(anal_grad, num_grad) < 1e-5

    # Curvature bound: ||grad^2 U(x)||_2 = L0 + 3 * lambda * ||x||^2 <= L0 + L1 * ||grad U(x)||
    norm_x = np.linalg.norm(x)
    hess_norm = target.L0 + 3.0 * target.lambda_param * (norm_x**2)
    grad_norm = np.linalg.norm(anal_grad)
    assert hess_norm <= target.L0 + target.L1 * grad_norm + 1e-9


@pytest.mark.unit
def test_neals_funnel_target_finite_difference():
    """Test 7: Neal's Funnel gradient relative error < 10^-5 at neck and mouth."""
    target = NealsFunnelTarget(d=3, sigma_v=3.0)

    # 1. Funnel neck (v = -3.0)
    x_neck = np.array([-3.0, 0.2, -0.3])
    assert max_relative_error(target.grad_log_prob(x_neck), numerical_gradient(target, x_neck)) < 1e-5

    # 2. Funnel mouth (v = +3.0)
    x_mouth = np.array([3.0, 10.0, -15.0])
    assert max_relative_error(target.grad_log_prob(x_mouth), numerical_gradient(target, x_mouth)) < 1e-5


@pytest.mark.unit
def test_neals_funnel_ground_truth_ks(rng):
    """Test 8: Neal's funnel ancestral draws pass marginal and conditional KS tests."""
    target = NealsFunnelTarget(d=3, sigma_v=3.0)
    samples = target.sample_ground_truth(3000, rng=rng)

    # Marginal v ~ N(0, 3^2)
    cdf_v = stats.norm(loc=0.0, scale=3.0).cdf
    res_v = stats.kstest(samples[:, 0], cdf_v)
    assert res_v.pvalue > 0.05, f"Funnel v marginal failed KS test: p={res_v.pvalue}"

    # Conditional standardized residuals: x_j / exp(v / 2) ~ N(0, 1)
    std_res_1 = samples[:, 1] / np.exp(0.5 * samples[:, 0])
    std_res_2 = samples[:, 2] / np.exp(0.5 * samples[:, 0])

    assert stats.kstest(std_res_1, stats.norm.cdf).pvalue > 0.05
    assert stats.kstest(std_res_2, stats.norm.cdf).pvalue > 0.05


@pytest.mark.unit
def test_oscillating_target_finite_difference():
    """Test 9: Spatially oscillating potential gradient relative error < 10^-5."""
    target = RippledGaussianTarget(d=2, sigma=1.2, amplitude=0.6, omega=8.0)
    x = np.array([0.45, -1.25])
    assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


@pytest.mark.unit
def test_student_t_target_finite_difference():
    """Test 10: Student-t gradient rel err < 10^-5 across degrees of freedom and dimensions."""
    for nu in [1.5, 2.5, 4.0]:
        for d in [2, 10]:
            target = MultivariateStudentTTarget(d=d, nu=nu)
            x = np.ones(d) * 1.5
            assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


@pytest.mark.unit
def test_student_t_ground_truth_ks(rng):
    """Test 11: Multivariate Student-t marginal ground truth draws pass KS test (p > 0.05)."""
    target = MultivariateStudentTTarget(d=2, nu=2.5)
    samples = target.sample_ground_truth(3000, rng=rng)

    for j in range(2):
        cdf = stats.t(df=2.5).cdf
        res = stats.kstest(samples[:, j], cdf)
        assert res.pvalue > 0.05, f"Student-t coordinate {j} failed KS test: p={res.pvalue}"


@pytest.mark.unit
def test_heterogeneous_target_finite_difference():
    """Test 12: Heterogeneous heavy/light target gradient relative error < 10^-5."""
    target = HeterogeneousHeavyTailedTarget(d=4, heavy_coords=[0, 1], nu_list=[1.8, 2.3])
    x = np.array([2.0, -1.5, 0.7, -0.4])
    assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


@pytest.mark.unit
def test_heterogeneous_target_ground_truth_ks():
    """Test 13: Heterogeneous target heavy coordinates pass t-test and light pass normal KS test."""
    local_rng = np.random.default_rng(2026)
    target = HeterogeneousHeavyTailedTarget(d=3, heavy_coords=[0], nu_list=[2.2])
    samples = target.sample_ground_truth(3000, rng=local_rng)

    # Coord 0 is Student-t(nu=2.2)
    assert stats.kstest(samples[:, 0], stats.t(df=2.2).cdf).pvalue > 0.05
    # Coord 1, 2 are standard normal
    assert stats.kstest(samples[:, 1], stats.norm.cdf).pvalue > 0.05
    assert stats.kstest(samples[:, 2], stats.norm.cdf).pvalue > 0.05


@pytest.mark.unit
def test_dual_rough_flat_target_finite_difference():
    """Test 14: Coupled dual rough-flat target gradient relative error < 10^-5."""
    target = CoupledDualPathologyTarget(d=4, nu=2.0, c=1.5)
    x = np.array([1.2, -0.8, 0.9, -1.1])
    assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


@pytest.mark.unit
def test_confinement_rough_target_finite_difference():
    """Test 15: Confinement target gradients relative error < 10^-5."""
    target1 = ConfinementInducedRoughnessTarget(d=2, base_scale=1.0)
    x1 = np.array([1.2, -0.7])
    assert max_relative_error(target1.grad_log_prob(x1), numerical_gradient(target1, x1)) < 1e-5

    target2 = ConfinementRoughTarget(d=2, nu=2.0, lambda_param=1.0, beta=2.0, radius=2.0)
    x2 = np.array([2.5, -1.8])
    assert max_relative_error(target2.grad_log_prob(x2), numerical_gradient(target2, x2)) < 1e-5


@pytest.mark.unit
def test_target_suite_registry_factory():
    """Test 16: get_target() instantiates all registry targets across dimensions {2, 10, 50, 100}."""
    test_dims = [2, 10, 50, 100]
    canonical_names = list_targets()
    assert len(canonical_names) >= 14

    for name in canonical_names:
        for d in test_dims:
            target = get_target(name, dim=d)
            assert target.d == d
            assert target.dim == d

    # Unknown target raises ValueError
    with pytest.raises(ValueError, match="Unknown target"):
        get_target("non_existent_target", dim=2)


@pytest.mark.unit
def test_target_dimension_scaling_and_shapes():
    """Test 17: Log-prob is scalar float and grad is 1D ndarray of shape (d,) across targets."""
    for d in [2, 10]:
        target = get_target("gaussian_anisotropic", dim=d)
        x = np.ones(d)
        lp = target.log_prob(x)
        grad = target.grad_log_prob(x)

        assert isinstance(lp, float)
        assert isinstance(grad, np.ndarray)
        assert grad.shape == (d,)
        assert grad.dtype == np.float64


@pytest.mark.unit
def test_joint_evaluation_consistency():
    """Test 18: log_prob_and_grad(x) identically matches (log_prob(x), grad_log_prob(x))."""
    targets = [
        IsotropicGaussian(d=3),
        DiagonalGaussian(d=3),
        FullCovarianceGaussian(d=3),
        QuarticTarget(d=3),
        NealsFunnelTarget(d=3),
        RippledGaussianTarget(d=3),
        MultivariateStudentTTarget(d=3),
        HeterogeneousHeavyTailedTarget(d=3),
        CoupledDualPathologyTarget(d=3),
        ConfinementInducedRoughnessTarget(d=3),
    ]

    x = np.array([0.5, -0.3, 0.8])
    for target in targets:
        joint_lp, joint_grad = target.log_prob_and_grad(x)
        sep_lp = target.log_prob(x)
        sep_grad = target.grad_log_prob(x)

        assert np.isclose(joint_lp, sep_lp, atol=1e-12)
        assert np.allclose(joint_grad, sep_grad, atol=1e-12)


@pytest.mark.unit
def test_l0l1_smooth_target_sampling_high_dim():
    """Test 19: L0L1SmoothTarget sampling runs in < 0.1s for d in {50, 100} and passes radial KS test."""
    for d in [50, 100]:
        target = L0L1SmoothTarget(d=d, L0=1.0, lambda_param=1.0)

        # 1. Latency check: 10 draws must complete in < 0.1s
        t0 = time.perf_counter()
        samples_10 = target.sample_ground_truth(10)
        dt = time.perf_counter() - t0
        assert dt < 0.1, f"d={d} sample_ground_truth(10) took {dt:.4f}s >= 0.1s"
        assert samples_10.shape == (10, d)
        assert np.all(np.isfinite(samples_10))

        # 2. Radial distribution KS test on R = ||X||^2 with N=2000 draws
        rng = np.random.default_rng(2026 + d)
        samples = target.sample_ground_truth(2000, rng=rng)
        s_vals = np.sum(samples**2, axis=1)

        alpha = 0.5 * (d - 2)
        w_star = (-target.L0 + np.sqrt(target.L0**2 + 8.0 * target.lambda_param * alpha)) / (2.0 * target.lambda_param)

        def unnorm(s: float) -> float:
            if s <= 0.0:
                return 0.0
            log_p = alpha * np.log(s) - 0.5 * target.L0 * s - 0.25 * target.lambda_param * (s**2)
            log_p_star = alpha * np.log(w_star) - 0.5 * target.L0 * w_star - 0.25 * target.lambda_param * (w_star**2)
            return float(np.exp(log_p - log_p_star))

        s_low = max(0.0, w_star - 10.0)
        s_high = w_star + 10.0
        Z, _ = quad(unnorm, s_low, s_high)

        def cdf_scalar(s: float) -> float:
            if s <= s_low:
                return 0.0
            if s >= s_high:
                return 1.0
            val, _ = quad(unnorm, s_low, s)
            return float(val / Z)

        cdf_vec = np.vectorize(cdf_scalar)
        res = stats.ks_1samp(s_vals, cdf_vec)
        assert res.pvalue > 0.05, f"d={d} failed radial KS test: stat={res.statistic:.4f}, p={res.pvalue:.4f}"


@pytest.mark.unit
def test_neals_funnel_tail_gradient_consistency():
    """Test 20: Verify exact gradient derivative consistency in Neal's funnel far tails."""
    tf = NealsFunnelTarget(d=2, sigma_v=3.0)
    h = 1e-6

    # 1. Challenger point v = -65, x_1 = 10 (was 100% error in Iteration 1)
    x_c = np.array([-65.0, 10.0])
    anal_grad = tf.grad_log_prob(x_c)
    num_grad = numerical_gradient(tf, x_c, h=h)
    rel_err = max_relative_error(anal_grad, num_grad)
    assert rel_err < 1e-5, f"Neal's Funnel tail gradient relative error {rel_err:.4e} >= 1e-5"

    # 2. Deep neck tail v = -200, x_1 = 2
    x_deep = np.array([-200.0, 2.0])
    anal_deep = tf.grad_log_prob(x_deep)
    num_deep = numerical_gradient(tf, x_deep, h=h)
    rel_err_deep = max_relative_error(anal_deep, num_deep)
    assert rel_err_deep < 1e-5, f"Deep tail gradient relative error {rel_err_deep:.4e} >= 1e-5"

    # 3. Extreme input stability: finite restoring gradients without NaN
    x_ext_pos = np.ones(2) * 1e4 / np.sqrt(2)
    x_ext_neg = -np.ones(2) * 1e4 / np.sqrt(2)
    assert np.all(np.isfinite(tf.grad_log_prob(x_ext_pos)))
    assert np.all(np.isfinite(tf.grad_log_prob(x_ext_neg)))
    assert tf.log_prob(x_ext_neg) == -np.inf


@pytest.mark.unit
def test_confinement_rough_tail_gradient_consistency():
    """Test 21: Verify exact gradient derivative consistency in ConfinementInducedRoughness far tails."""
    tc = ConfinementInducedRoughnessTarget(d=2, base_scale=1.0)
    h = 1e-6

    # 1. Challenger point y = [60, 0] (was 100% error in Iteration 1)
    y_c = np.array([60.0, 0.0])
    anal_grad = tc.grad_log_prob(y_c)
    num_grad = numerical_gradient(tc, y_c, h=h)
    rel_err = max_relative_error(anal_grad, num_grad)
    assert rel_err < 1e-5, f"Confinement Induced Roughness tail gradient relative error {rel_err:.4e} >= 1e-5"

    # 2. Deep tail y = [150, 0]
    y_deep = np.array([150.0, 0.0])
    anal_deep = tc.grad_log_prob(y_deep)
    num_deep = numerical_gradient(tc, y_deep, h=h)
    rel_err_deep = max_relative_error(anal_deep, num_deep)
    assert rel_err_deep < 1e-5, f"Deep tail gradient relative error {rel_err_deep:.4e} >= 1e-5"

    # 3. Symmetric multi-D tail points across dimensions
    for d in [2, 10, 50, 100]:
        t_d = ConfinementInducedRoughnessTarget(d=d)
        y_sym = np.full(d, 60.0 / np.sqrt(d))
        anal_g = t_d.grad_log_prob(y_sym)
        num_g = numerical_gradient(t_d, y_sym, h=h)
        assert max_relative_error(anal_g, num_g) < 1e-5

    # 4. Extreme input stability: finite restoring gradients without NaN
    assert np.all(np.isfinite(tc.grad_log_prob(np.ones(2) * 1e4)))
    assert np.all(np.isfinite(tc.grad_log_prob(-np.ones(2) * 1e4)))
    assert tc.log_prob(np.ones(2) * 1e4) == -np.inf


@pytest.mark.unit
def test_confinement_rough_target_sampling_high_dim():
    """Test 22: ConfinementRoughTarget sampling at d=100 terminates in < 20ms with bounded support."""
    t = ConfinementRoughTarget(d=100)
    for seed in [0, 8, 42]:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        samples = t.sample_ground_truth(100, rng=rng)
        dt = time.perf_counter() - t0

        assert dt < 0.020, f"Seed {seed} sampling took {dt * 1000:.2f}ms >= 20ms"
        assert samples.shape == (100, 100)
        assert np.all(np.isfinite(samples))
        assert np.max(np.abs(samples)) < 5.0, f"Seed {seed} max coordinate {np.max(np.abs(samples))} >= 5.0"


@pytest.mark.unit
def test_coupled_dual_target_sampling_termination():
    """Test 23: CoupledDualPathologyTarget sampling terminates in < 0.5s at d in {50, 100} with non-zero coupling."""
    for d in [50, 100]:
        t = CoupledDualPathologyTarget(d=d, c=1.0)
        for seed in [0, 6, 42]:
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            s = t.sample_ground_truth(10, rng=rng)
            dt = time.perf_counter() - t0

            assert dt < 0.5, f"d={d} seed={seed} took {dt:.4f}s >= 0.5s"
            assert s.shape == (10, d)
            assert np.all(np.isfinite(s))

        # Check positive coupling correlation between heavy coordinate 0 and coupled rough coordinate d1
        s_corr = t.sample_ground_truth(500, rng=np.random.default_rng(2026))
        corr = np.corrcoef(s_corr[:, 0], s_corr[:, t.d1])[0, 1]
        assert corr > 0.15, f"d={d} coupling correlation {corr:.3f} <= 0.15"
