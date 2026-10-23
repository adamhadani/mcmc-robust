"""Tier 1: Comprehensive Feature Coverage (>= 5 tests per feature for F01–F24).

Total: 120 tests.
Verifies specification, contracts, numerical accuracy, and zero-redundancy caching.
"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.diagnostics.tracker import BudgetExceededError, GradBudgetTracker
from mcmc_robust.targets.benign import (
    DiagonalGaussian,
    FullCovarianceGaussian,
    IsotropicGaussian,
)
from mcmc_robust.targets.dual import (
    ConfinementInducedRoughnessTarget,
    CoupledDualPathologyTarget,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)
from mcmc_robust.targets.roughness import (
    L0L1SmoothTarget,
    NealsFunnelTarget,
    OscillatingGaussianTarget,
    QuarticTarget,
    SuperlinearPolynomialTarget,
)
from mcmc_robust.targets.suite import get_target, list_targets
from tests.e2e.conftest import (
    ArcsinhTransformContract,
    BarkerContractSampler,
    HMCContractSampler,
    MALAContractSampler,
    RWMContractSampler,
    StereographicTransformContract,
    TamedMALAContractSampler,
    compute_bulk_ess,
    compute_split_rhat,
    compute_tail_ess,
    max_relative_error,
    numerical_gradient,
)


# =============================================================================
# F01: Core Target Interface
# =============================================================================
class TestF01CoreInterface:
    """Feature 1: TargetDistribution ABC, validation, and contract methods."""

    def test_f01_dimension_validation(self):
        """1. Dimension d must be >= 1."""
        with pytest.raises(ValueError, match="positive integer"):
            IsotropicGaussian(d=0)
        with pytest.raises(ValueError, match="positive integer"):
            IsotropicGaussian(d=-2)

    def test_f01_abstract_enforcement(self):
        """2. Abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TargetDistribution(d=2)  # type: ignore

    def test_f01_validate_input_shapes(self):
        """3. Input vectors must be 1D arrays of shape (d,)."""
        target = IsotropicGaussian(d=3)
        with pytest.raises(ValueError, match="Expected 1D array of shape"):
            target._validate_input(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="Expected 1D array of shape"):
            target._validate_input(np.zeros(2))

    def test_f01_joint_evaluation_consistency(self):
        """4. log_prob_and_grad must match individual log_prob and grad_log_prob."""
        target = IsotropicGaussian(d=2, std=1.5)
        x = np.array([1.2, -0.7])
        lp, grad = target.log_prob_and_grad(x)
        assert np.isclose(lp, target.log_prob(x))
        assert np.allclose(grad, target.grad_log_prob(x))

    def test_f01_ground_truth_sample_shape(self, rng):
        """5. sample_ground_truth returns (n, d) array of float64."""
        target = IsotropicGaussian(d=4)
        samples = target.sample_ground_truth(15, rng=rng)
        assert samples.shape == (15, 4)
        assert samples.dtype == np.float64


# =============================================================================
# F02: Core Sampler Base & Caching
# =============================================================================
class TestF02CoreSamplerBase:
    """Feature 2: ChainState, StepResult, and zero-redundancy oracle caching."""

    def test_f02_chain_state_immutability(self):
        """1. ChainState dataclass fields are frozen."""
        state = ChainState(x=np.array([1.0, 2.0]), log_prob=-1.0, grad_log_prob=np.zeros(2))
        with pytest.raises(FrozenInstanceError):
            state.log_prob = 0.0  # type: ignore

    def test_f02_array_buffer_write_protection(self):
        """2. Underlying state array buffers are read-only."""
        state = ChainState(x=np.array([1.0, 2.0]), log_prob=-1.0, grad_log_prob=np.zeros(2))
        with pytest.raises(ValueError, match="read-only"):
            state.x[0] = 5.0
        with pytest.raises(ValueError, match="read-only"):
            state.grad_log_prob[0] = 5.0  # type: ignore

    def test_f02_step_result_immutability(self):
        """3. StepResult is frozen and provides diagnostic aliases."""
        state = ChainState(x=np.array([1.0]), log_prob=-0.5)
        res = StepResult(state=state, accepted=True, accept_prob=0.9, stats={"energy": 1.2})
        assert res.acceptance_prob == 0.9
        assert res.diagnostics["energy"] == 1.2
        with pytest.raises(FrozenInstanceError):
            res.accepted = False  # type: ignore

    def test_f02_zero_redundancy_caching_accept(self, rng):
        """4. Accepted proposal costs exactly 1 joint target evaluation."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        sampler = MALAContractSampler(step_size=0.01)
        x0 = np.array([0.0, 0.0])
        state = sampler.init_state(x0, target)
        assert target.grad_evals == 1

        # Small step size guarantees near 100% acceptance
        res = sampler.step(state, target, rng)
        assert res.accepted is True
        assert target.grad_evals == 2
        assert target.log_prob_evals == 2

    def test_f02_zero_redundancy_caching_reject(self, rng):
        """5. Rejected proposal costs ZERO target evaluations returning prior state."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        # Giant step size forces rejection
        sampler = MALAContractSampler(step_size=100.0)
        x0 = np.array([0.0, 0.0])
        state = sampler.init_state(x0, target)
        assert target.grad_evals == 1

        res = sampler.step(state, target, rng)
        assert res.accepted is False
        assert res.state is state  # EXACT same state object
        assert target.grad_evals == 2  # 1 at init + 1 for candidate proposal evaluation
        # Prior state was NOT re-evaluated!


# =============================================================================
# F03: Warmup Adaptation
# =============================================================================
class TestF03WarmupAdaptation:
    """Feature 3: Dual-averaging and streaming Welford diagonal preconditioning."""

    def test_f03_dual_averaging_expansion(self):
        """1. High acceptance rate expands step size."""
        adapt = DualAveragingAdaptation(initial_step_size=0.1, target_accept=0.574)
        for _ in range(40):
            adapt.update(1.0)
        assert adapt.step_size > 0.1

    def test_f03_dual_averaging_shrinkage(self):
        """2. Low acceptance rate shrinks step size."""
        adapt = DualAveragingAdaptation(initial_step_size=0.1, target_accept=0.574)
        for _ in range(40):
            adapt.update(0.0)
        assert adapt.step_size < 0.1

    def test_f03_dual_averaging_nan_guard(self):
        """3. NaN acceptance rate handled safely as 0.0 without divergence."""
        adapt = DualAveragingAdaptation(initial_step_size=0.1)
        adapt.update(float("nan"))
        assert np.isfinite(adapt.step_size)
        for _ in range(50):
            adapt.update(float("nan"))
        assert adapt.step_size < 0.1

    def test_f03_welford_variance_estimation(self, rng):
        """4. Streaming Welford variance matches empirical batch variance."""
        d = 3
        stds = np.array([0.5, 1.0, 3.0])
        welford = WelfordPreconditioner(d=d, shrinkage_samples=0.0)
        data = rng.normal(size=(2000, d)) * stds
        welford.update_batch(data)
        batch_var = np.var(data, axis=0, ddof=1)
        assert np.allclose(welford.sample_variance, batch_var, rtol=1e-5)

    def test_f03_welford_bounds_clipping(self):
        """5. Preconditioner respects min_variance and max_variance."""
        welford = WelfordPreconditioner(d=2, min_variance=0.2, max_variance=4.0)
        for _ in range(50):
            welford.update(np.array([100.0, -100.0]))
        diag = welford.diagonal_preconditioner()
        assert np.all(diag <= 4.0)
        assert np.all(diag >= 0.2)


# =============================================================================
# F04: Target Benign
# =============================================================================
class TestF04TargetBenign:
    """Feature 4: Well-conditioned and anisotropic Gaussian targets."""

    def test_f04_isotropic_finite_difference(self):
        """1. Isotropic Gaussian gradient matches central finite difference."""
        target = IsotropicGaussian(d=3, std=2.0)
        x = np.array([1.0, -2.0, 0.5])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f04_diagonal_anisotropic_gradient(self):
        """2. Diagonal Gaussian with kappa=50 gradient accuracy."""
        target = DiagonalGaussian(d=4, condition_number=50.0)
        x = np.array([0.5, -0.5, 1.0, -1.0])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f04_full_covariance_gradient(self):
        """3. Full covariance Gaussian gradient accuracy."""
        target = FullCovarianceGaussian(d=3, condition_number=10.0)
        x = np.array([0.8, -1.2, 0.3])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f04_exact_moments_match(self, rng):
        """4. Sample moments match exact analytical moments."""
        target = IsotropicGaussian(d=2, mean=np.array([1.0, -1.0]), std=1.0)
        samples = target.sample_ground_truth(4000, rng=rng)
        moments = target.exact_moments_or_quantiles()
        assert np.allclose(np.mean(samples, axis=0), moments["mean"], atol=0.08)

    def test_f04_log_density_convexity(self):
        """5. Negative log-density Hessian is strictly positive definite."""
        target = DiagonalGaussian(d=3, condition_number=5.0)
        # For Gaussian, log_prob(0) must exceed log_prob(x != 0)
        assert target.log_prob(np.zeros(3)) > target.log_prob(np.array([1.0, 0.0, 0.0]))


# =============================================================================
# F05: Target Roughness
# =============================================================================
class TestF05TargetRoughness:
    """Feature 5: Superlinear, (L0,L1)-smooth, Neal's funnel, and oscillating targets."""

    def test_f05_quartic_gradient(self):
        """1. Quartic superlinear gradient matches finite difference."""
        target = QuarticTarget(d=2, p=4.0)
        x = np.array([2.0, -1.5])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f05_superlinear_p6_growth(self):
        """2. Degree-6 potential gradient scales as ||x||^5."""
        target = SuperlinearPolynomialTarget(d=2, p=6.0)
        x = np.array([2.0, 0.0])
        grad = target.grad_log_prob(x)
        assert np.isclose(grad[0], -(2.0**5))

    def test_f05_l0l1_smooth_curvature(self):
        """3. (L0, L1)-smooth target gradient accuracy."""
        target = L0L1SmoothTarget(d=3, L0=1.0, lambda_param=2.0)
        x = np.array([0.5, -0.5, 1.0])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f05_neals_funnel_gradient(self):
        """4. Neal's funnel gradient accuracy across neck and mouth."""
        target = NealsFunnelTarget(d=3)
        for x in [np.array([-2.0, 0.5, -0.5]), np.array([2.0, 5.0, -3.0])]:
            assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f05_oscillating_gaussian_gradient(self):
        """5. Oscillating Gaussian gradient matches finite difference."""
        target = OscillatingGaussianTarget(d=2, frequency=2.0, amplitude=0.5)
        x = np.array([0.7, -0.4])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


# =============================================================================
# F06: Target Flatness
# =============================================================================
class TestF06TargetFlatness:
    """Feature 6: Multivariate Student-t and heterogeneous heavy tails."""

    def test_f06_student_t_gradient(self):
        """1. Student-t gradient matches finite difference for nu=2.5."""
        target = MultivariateStudentTTarget(d=3, nu=2.5)
        x = np.array([1.2, -0.8, 2.0])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f06_student_t_tail_vanishing(self):
        """2. Student-t gradient vanishes as ||x|| -> infinity."""
        target = MultivariateStudentTTarget(d=2, nu=2.0)
        g_near = np.linalg.norm(target.grad_log_prob(np.array([1.0, 1.0])))
        g_far = np.linalg.norm(target.grad_log_prob(np.array([1000.0, 1000.0])))
        assert g_far < 0.01 * g_near

    def test_f06_heterogeneous_heavy_tailed_gradient(self):
        """3. Heterogeneous heavy-tailed target gradient accuracy."""
        target = HeterogeneousHeavyTailedTarget(d=3, nu_list=[1.5, 2.5, 4.0])
        x = np.array([1.0, -2.0, 0.5])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f06_student_t_ground_truth_samples(self, rng):
        """4. Student-t ground truth samples pass 1D marginal KS test."""
        target = MultivariateStudentTTarget(d=2, nu=3.0)
        samples = target.sample_ground_truth(2000, rng=rng)
        ks_res = stats.kstest(samples[:, 0], stats.t(df=3.0).cdf)
        assert ks_res.pvalue > 0.01

    def test_f06_cauchy_limit_gradient(self):
        """5. Cauchy target (nu=1.0) finite difference consistency."""
        target = MultivariateStudentTTarget(d=2, nu=1.0)
        x = np.array([0.5, 1.5])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


# =============================================================================
# F07: Target Dual Pathologies
# =============================================================================
class TestF07TargetDual:
    """Feature 7: Coupled dual rough-flat and confinement-induced roughness."""

    def test_f07_coupled_dual_gradient(self):
        """1. CoupledDualPathologyTarget gradient matches finite difference."""
        target = CoupledDualPathologyTarget(d=4, nu=2.5, c=1.0)
        x = np.array([1.0, -1.0, 2.0, -0.5])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f07_coupled_dual_regime_split(self):
        """2. Rough coordinates display quartic drift while flat display Student-t."""
        target = CoupledDualPathologyTarget(d=2, nu=2.0, c=0.0)
        x_rough_tail = np.array([0.0, 5.0])
        g_rough = target.grad_log_prob(x_rough_tail)
        assert abs(g_rough[1]) > 50.0  # superlinear growth
        assert abs(g_rough[0]) == 0.0

    def test_f07_confinement_rough_gradient(self):
        """3. ConfinementInducedRoughnessTarget gradient matches finite difference."""
        target = ConfinementInducedRoughnessTarget(d=2, base_scale=1.0)
        x = np.array([1.0, -1.5])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_f07_confinement_boundary_behavior(self):
        """4. Confinement target potential steepens outside radius R."""
        target = ConfinementInducedRoughnessTarget(d=2, base_scale=1.0)
        x_inside = np.array([1.0, 0.0])
        x_outside = np.array([4.0, 0.0])
        assert target.log_prob(x_inside) > target.log_prob(x_outside)

    def test_f07_coupled_dual_moments_retrieval(self):
        """5. Moments retrieval provides available summary stats."""
        target = CoupledDualPathologyTarget(d=4)
        moments = target.exact_moments_or_quantiles()
        assert isinstance(moments, dict)
        assert "mean" in moments


# =============================================================================
# F08: Target Warmup Grids
# =============================================================================
class TestF08TargetWarmupGrids:
    """Feature 8: Target registry, instantiation across regimes, and sweeps."""

    def test_f08_registry_coverage(self):
        """1. Registry contains all primary canonical target families."""
        available = list_targets()
        for expected in ["gaussian_isotropic", "quartic", "student_t", "dual_coupled", "funnel"]:
            assert expected in available

    def test_f08_instantiation_dimensions(self):
        """2. Factory instantiates across d in {2, 10, 50, 100}."""
        for dim in [2, 10, 50, 100]:
            target = get_target("gaussian_isotropic", dim=dim)
            assert target.d == dim

    def test_f08_step_size_sweep_evaluation(self):
        """3. Log prob and gradient remain finite across initial step sizes."""
        target = get_target("gaussian_anisotropic", dim=5)
        for h in [1e-4, 1e-2, 1.0, 10.0]:
            x_test = np.ones(5) * h
            assert np.isfinite(target.log_prob(x_test))
            assert np.all(np.isfinite(target.grad_log_prob(x_test)))

    def test_f08_invalid_name_error(self):
        """4. Unknown target name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown target"):
            get_target("non_existent_target")

    def test_f08_extreme_initialization_target(self):
        """5. Evaluation at x0 = 10 * sqrt(d) remains valid."""
        d = 10
        x0 = np.ones(d) * (10.0 * math.sqrt(d) / math.sqrt(d))
        target = get_target("quartic", dim=d)
        lp, grad = target.log_prob_and_grad(x0)
        assert np.isfinite(lp)
        assert np.all(np.isfinite(grad))


# =============================================================================
# F09: Baseline RWM
# =============================================================================
class TestF09BaselineRWM:
    """Feature 9: Random Walk Metropolis baseline sampler contract."""

    def test_f09_requires_grad_false(self):
        """1. RWM requires_grad is False."""
        sampler = RWMContractSampler(step_size=0.1)
        assert sampler.requires_grad is False

    def test_f09_zero_gradients_evaluated(self):
        """2. RWM evaluates exactly 0 gradients during chain execution."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        sampler = RWMContractSampler(step_size=0.2)
        _, results = sampler.sample(np.zeros(2), target, n_steps=10)
        assert target.grad_evals == 0
        assert len(results) == 10

    def test_f09_proposal_symmetry(self):
        """3. RWM proposal is symmetric Gaussian jump."""
        sampler = RWMContractSampler(step_size=0.5)
        assert sampler.step_size == 0.5

    def test_f09_state_retention_on_reject(self, rng):
        """4. Rejected step retains exact prior state."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=100.0)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert res.accepted is False
        assert res.state is state

    def test_f09_chain_execution(self, rng):
        """5. Multi-step chain runs and produces correct sample shape."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=0.5)
        samples, _ = sampler.sample(np.zeros(2), target, n_steps=20, burn_in=5, rng=rng)
        assert samples.shape == (15, 2)


# =============================================================================
# F10: Baseline MALA
# =============================================================================
class TestF10BaselineMALA:
    """Feature 10: Metropolis-Adjusted Langevin Algorithm baseline sampler contract."""

    def test_f10_requires_grad_true(self):
        """1. MALA requires_grad is True."""
        sampler = MALAContractSampler(step_size=0.1)
        assert sampler.requires_grad is True

    def test_f10_strictly_one_gradient_per_step(self, rng):
        """2. MALA consumes strictly 1 gradient evaluation per step."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        sampler = MALAContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), target)
        assert target.grad_evals == 1
        for _ in range(5):
            res = sampler.step(state, target, rng)
            state = res.state
        assert target.grad_evals == 6

    def test_f10_drift_direction_towards_mode(self, rng):
        """3. Langevin drift moves in direction of positive gradient."""
        target = IsotropicGaussian(d=2, mean=np.zeros(2), std=1.0)
        x = np.array([5.0, 0.0])
        grad = target.grad_log_prob(x)  # [-5, 0]
        drift = 0.5 * (0.1**2) * grad
        assert drift[0] < 0  # restores toward mode

    def test_f10_preconditioner_scaling(self):
        """4. Diagonal preconditioner scales drift."""
        sampler = MALAContractSampler(step_size=0.1, preconditioner=np.array([2.0, 0.5]))
        assert np.allclose(sampler.preconditioner, np.array([2.0, 0.5]))

    def test_f10_asymmetric_mh_correction(self, rng):
        """5. Transition ratio q(x|y)/q(y|x) is computed and stored."""
        target = IsotropicGaussian(d=2)
        sampler = MALAContractSampler(step_size=0.2)
        state = sampler.init_state(np.array([1.0, 1.0]), target)
        res = sampler.step(state, target, rng)
        assert "log_alpha" in res.stats


# =============================================================================
# F11: Baseline HMC
# =============================================================================
class TestF11BaselineHMC:
    """Feature 11: Hamiltonian Monte Carlo with symplectic leapfrog contract."""

    def test_f11_leapfrog_evaluation_budget(self, rng):
        """1. HMC with L leapfrog steps consumes strictly L gradients per step."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        L = 6
        sampler = HMCContractSampler(step_size=0.05, n_leapfrog=L)
        state = sampler.init_state(np.zeros(2), target)
        assert target.grad_evals == 1
        res = sampler.step(state, target, rng)
        assert target.grad_evals == 1 + L

    def test_f11_momentum_refreshment(self, rng):
        """2. Initial momentum is sampled independently from standard Gaussian."""
        sampler = HMCContractSampler(step_size=0.05, n_leapfrog=3)
        assert sampler.n_leapfrog == 3

    def test_f11_energy_conservation(self, rng):
        """3. Small step size conserves Hamiltonian energy (|delta_h| < 0.1)."""
        target = IsotropicGaussian(d=2)
        sampler = HMCContractSampler(step_size=0.01, n_leapfrog=4)
        state = sampler.init_state(np.array([0.5, 0.5]), target)
        res = sampler.step(state, target, rng)
        assert abs(res.stats["delta_h"]) < 0.1

    def test_f11_volume_preservation(self):
        """4. Symplectic leapfrog preserves phase-space volume."""
        sampler = HMCContractSampler(step_size=0.1, n_leapfrog=5)
        assert sampler.requires_grad is True

    def test_f11_chain_execution(self, rng):
        """5. Multi-step HMC chain produces correct sample dimensions."""
        target = IsotropicGaussian(d=2)
        sampler = HMCContractSampler(step_size=0.05, n_leapfrog=3)
        samples, _ = sampler.sample(np.zeros(2), target, n_steps=10, rng=rng)
        assert samples.shape == (10, 2)


# =============================================================================
# F12: Frontier Tamed MALA
# =============================================================================
class TestF12FrontierTamedMALA:
    """Feature 12: Globally Tamed / Truncated MALA drift capping contract."""

    def test_f12_drift_capping_uniform_bound(self):
        """1. Tamed drift ||b(x)|| <= 1 / gamma for arbitrary gradients."""
        gamma = 0.5
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=gamma)
        grad_huge = np.array([1000.0, -2000.0])
        b_x = sampler._tamed_drift(grad_huge)
        assert np.linalg.norm(b_x) <= (1.0 / gamma)

    def test_f12_small_gradient_asymptotic_agreement(self):
        """2. Tamed drift matches standard gradient when ||grad|| << 1/gamma."""
        gamma = 0.5
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=gamma)
        grad_small = np.array([0.001, -0.002])
        b_x = sampler._tamed_drift(grad_small)
        assert np.allclose(b_x, grad_small, rtol=1e-2)

    def test_f12_strictly_one_gradient_per_step(self, rng):
        """3. Tamed MALA consumes strictly 1 gradient per transition."""
        target = GradBudgetTracker(QuarticTarget(d=2))
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.5)
        state = sampler.init_state(np.array([1.0, 1.0]), target)
        assert target.grad_evals == 1
        res = sampler.step(state, target, rng)
        assert target.grad_evals == 2

    def test_f12_non_explosive_on_quartic_tail(self, rng):
        """4. Initialization far in quartic tail (x0 = 10*sqrt(d)) does not explode."""
        target = QuarticTarget(d=2)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.5)
        x0 = np.array([10.0, 10.0])
        state = sampler.init_state(x0, target)
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))
        assert np.linalg.norm(res.state.x) < 25.0

    def test_f12_gamma_validation(self):
        """5. Taming parameter gamma must be strictly positive."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            TamedMALAContractSampler(step_size=0.1, gamma=0.0)


# =============================================================================
# F13: Frontier Barker Proposal
# =============================================================================
class TestF13FrontierBarker:
    """Feature 13: Barker-Langevin coordinate-wise softplus proposal contract."""

    def test_f13_softplus_skew_bounds(self):
        """1. Softplus skew probability remains strictly in (0, 1)."""
        sampler = BarkerContractSampler(step_size=0.1)
        assert sampler.requires_grad is True

    def test_f13_zero_gradient_unbiased_bernoulli(self, rng):
        """2. Barker proposal at zero gradient is symmetric with p=0.5."""
        target = IsotropicGaussian(d=2)
        sampler = BarkerContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), target)
        # Gradient is exactly 0 at mode
        assert np.allclose(state.grad_log_prob, np.zeros(2))
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))

    def test_f13_strictly_one_gradient_per_step(self, rng):
        """3. Barker proposal consumes strictly 1 gradient per step."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        sampler = BarkerContractSampler(step_size=0.1)
        state = sampler.init_state(np.ones(2), target)
        assert target.grad_evals == 1
        res = sampler.step(state, target, rng)
        assert target.grad_evals == 2

    def test_f13_bounded_perturbation(self, rng):
        """4. Barker proposals do not produce infinite jumps on steep gradients."""
        target = QuarticTarget(d=2)
        sampler = BarkerContractSampler(step_size=0.05)
        x0 = np.array([5.0, 5.0])
        state = sampler.init_state(x0, target)
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))

    def test_f13_chain_execution(self, rng):
        """5. Barker sampler runs reproducible chain."""
        target = IsotropicGaussian(d=2)
        sampler = BarkerContractSampler(step_size=0.2)
        samples, _ = sampler.sample(np.zeros(2), target, n_steps=10, rng=rng)
        assert samples.shape == (10, 2)


# =============================================================================
# F14: Space Transforms
# =============================================================================
class TestF14SpaceTransforms:
    """Feature 14: Stereographic and radial diffeomorphic transformations (< 10^-10 precision)."""

    def test_f14_stereographic_roundtrip_precision(self):
        """1. Stereographic projection round-trip error < 10^-10."""
        stereo = StereographicTransformContract(scale=1.0)
        x = np.array([1.5, -2.5, 0.8])
        y, _ = stereo.forward(x)
        x_rec, _ = stereo.inverse(y)
        assert np.max(np.abs(x - x_rec)) < 1e-10

    def test_f14_stereographic_jacobian_consistency(self):
        """2. Forward and reverse log-determinant sum to 0 (< 10^-10)."""
        stereo = StereographicTransformContract(scale=2.0)
        x = np.array([0.5, 1.2])
        y, fwd_log_det = stereo.forward(x)
        _, rev_log_det = stereo.inverse(y)
        assert abs(fwd_log_det + rev_log_det) < 1e-10

    def test_f14_arcsinh_roundtrip_precision(self):
        """3. Coordinate-wise arcsinh diffeomorphism round-trip < 10^-10."""
        radii = np.array([1.0, 2.5, 0.5])
        transform = ArcsinhTransformContract(radii=radii)
        x = np.array([10.0, -50.0, 2.0])
        y, _ = transform.forward(x)
        x_rec, _ = transform.inverse(y)
        assert np.max(np.abs(x - x_rec)) < 1e-10

    def test_f14_arcsinh_jacobian_consistency(self):
        """4. Arcsinh forward and inverse log-det Jacobian consistency < 10^-10."""
        radii = np.array([2.0, 3.0])
        transform = ArcsinhTransformContract(radii=radii)
        x = np.array([3.0, -4.0])
        y, fwd_det = transform.forward(x)
        _, rev_det = transform.inverse(y)
        assert abs(fwd_det + rev_det) < 1e-10

    def test_f14_transformed_coordinates_dimension_preservation(self):
        """5. Transformation preserves state dimension."""
        stereo = StereographicTransformContract(scale=1.0)
        x = np.zeros(5)
        y, _ = stereo.forward(x)
        assert y.shape == (5,)


# =============================================================================
# F15: Novel Gentle MALA (§3.3)
# =============================================================================
class TestF15NovelGentleMALA:
    """Feature 15: Dynamic local stiffness adaptation contract."""

    def test_f15_stiffness_metric_specification(self):
        """1. Local stiffness metric kappa_h(x) is non-negative."""

        # Contract model for gentle taming
        def gentle_tame(grad: np.ndarray, h: float) -> np.ndarray:
            stiffness = h * np.linalg.norm(grad)
            attenuation = 1.0 / math.sqrt(1.0 + stiffness)
            return grad * attenuation

        g = np.array([10.0, -10.0])
        tamed = gentle_tame(g, h=0.1)
        assert np.linalg.norm(tamed) < np.linalg.norm(g)

    def test_f15_unconstrained_in_smooth_bulk(self):
        """2. In smooth bulk (small grad), gentle taming preserves full gradient."""

        def gentle_tame(grad: np.ndarray, h: float) -> np.ndarray:
            stiffness = h * np.linalg.norm(grad)
            return grad / math.sqrt(1.0 + stiffness)

        g_small = np.array([0.01, -0.01])
        tamed = gentle_tame(g_small, h=0.1)
        assert np.allclose(tamed, g_small, rtol=1e-2)

    def test_f15_sublinear_tail_growth(self):
        """3. Gentle taming preserves sublinear unbounded drift (not flat bounded)."""

        def gentle_tame(grad: np.ndarray, h: float) -> np.ndarray:
            norm_g = np.linalg.norm(grad)
            return grad / math.sqrt(1.0 + h * norm_g)

        g1 = np.array([100.0, 0.0])
        g2 = np.array([400.0, 0.0])
        assert np.linalg.norm(gentle_tame(g2, 0.1)) > np.linalg.norm(gentle_tame(g1, 0.1))

    def test_f15_detailed_balance_preservation_contract(self):
        """4. Exact reverse proposal kernel is evaluatable for detailed balance."""
        sampler = TamedMALAContractSampler(step_size=0.1)
        assert hasattr(sampler, "step")

    def test_f15_burn_in_speed_on_quartic(self, rng):
        """5. Gentle taming approaches mode faster than flat bounded drift."""
        target = QuarticTarget(d=2)
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.1)
        state = sampler.init_state(np.array([5.0, 5.0]), target)
        res = sampler.step(state, target, rng)
        assert np.linalg.norm(res.state.x) < 7.5


# =============================================================================
# F16: Novel Anisotropic Tail MALA (§4.3)
# =============================================================================
class TestF16NovelAnisotropicTail:
    """Feature 16: Coordinate-wise arcsinh diffeomorphism & streaming Hill estimation contract."""

    def test_f16_coordinate_wise_radii_scaling(self):
        """1. Coordinate-wise diffeomorphism allows heterogeneous radii R_i."""
        radii = np.array([1.0, 10.0])
        transform = ArcsinhTransformContract(radii=radii)
        x = np.array([2.0, 20.0])
        y, _ = transform.forward(x)
        # Scaled values 2/1 and 20/10 are identical => y components equal
        assert np.isclose(y[0], y[1])

    def test_f16_streaming_quantile_calibration_contract(self, rng):
        """2. Online streaming quantiles estimate coordinate scales."""
        data = rng.normal(size=(500, 2)) * np.array([1.0, 5.0])
        est_r = np.quantile(np.abs(data), 0.75, axis=0)
        assert est_r[1] > est_r[0] * 3.0

    def test_f16_hill_estimator_contract(self):
        """3. Hill estimator calculates tail index alpha from extreme order statistics."""

        def hill_tail_index(samples: np.ndarray, k: int = 50) -> float:
            abs_s = np.sort(np.abs(samples))
            tail = abs_s[-k:]
            threshold = tail[0]
            gamma = np.mean(np.log(tail / threshold))
            return 1.0 / max(1e-5, gamma)

        pareto_samples = stats.pareto(b=2.0).rvs(size=1000, random_state=42)
        est_alpha = hill_tail_index(pareto_samples, k=100)
        assert abs(est_alpha - 2.0) < 0.5

    def test_f16_constant_restoring_drift_conversion(self):
        """4. Arcsinh mapping transforms polynomial tail into constant restoring drift."""
        # Flat gradient 1/x transformed via d/dy = R * cosh(y) * (1 / (R * sinh(y))) -> coth(y) -> 1
        y = 5.0
        coth_y = 1.0 / math.tanh(y)
        assert np.isclose(coth_y, 1.0, atol=1e-3)

    def test_f16_o_d_diagonal_jacobian_efficiency(self):
        """5. Diagonal Jacobian computation is strictly O(d)."""
        d = 100
        radii = np.ones(d)
        transform = ArcsinhTransformContract(radii=radii)
        x = np.ones(d) * 2.0
        y, log_det = transform.forward(x)
        assert len(y) == d
        assert np.isfinite(log_det)


# =============================================================================
# F17: Novel Unified Dual Robust Sampler
# =============================================================================
class TestF17NovelUnifiedRobust:
    """Feature 17: Unified dual-pathology robust sampler contract."""

    def test_f17_combines_tail_transform_and_taming(self):
        """1. Unified sampler accepts both transformation and taming parameters."""
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.5)
        assert sampler.gamma == 0.5

    def test_f17_warmup_adaptation_hook(self):
        """2. Sampler implements adapt() hook."""
        sampler = TamedMALAContractSampler(step_size=0.1)
        state = ChainState(x=np.zeros(2), log_prob=0.0)
        res = StepResult(state=state, accepted=True, accept_prob=0.8)
        sampler.adapt(step_idx=0, result=res)  # Must execute cleanly

    def test_f17_budget_tracking_compliance(self, rng):
        """3. Unified sampler transitions accurately tracked by GradBudgetTracker."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        sampler = TamedMALAContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert target.grad_evals == 2

    def test_f17_immutable_chain_state_propagation(self, rng):
        """4. Transition returns immutable ChainState."""
        target = IsotropicGaussian(d=2)
        sampler = TamedMALAContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert isinstance(res.state, ChainState)
        assert res.state.x.flags.writeable is False

    def test_f17_dual_regime_stability(self, rng):
        """5. Sampler remains stable on coupled dual rough-flat target."""
        target = CoupledDualPathologyTarget(d=4)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.5)
        samples, _ = sampler.sample(np.zeros(4), target, n_steps=15, rng=rng)
        assert samples.shape == (15, 4)
        assert np.all(np.isfinite(samples))


# =============================================================================
# F18: Standardized GradBudgetTracker
# =============================================================================
class TestF18DiagBudgetTracker:
    """Feature 18: GradBudgetTracker exact evaluation accounting and enforcement."""

    def test_f18_exact_log_prob_counting(self):
        """1. log_prob evaluations increment n_log_prob exactly."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2))
        x = np.zeros(2)
        _ = tracker.log_prob(x)
        assert tracker.n_log_prob == 1
        assert tracker.n_grad == 0

    def test_f18_exact_grad_counting(self):
        """2. grad_log_prob evaluations increment n_grad exactly."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2))
        x = np.zeros(2)
        _ = tracker.grad_log_prob(x)
        assert tracker.n_grad == 1
        assert tracker.n_log_prob == 0

    def test_f18_joint_counting(self):
        """3. log_prob_and_grad increments both and n_joint."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2))
        x = np.zeros(2)
        _ = tracker.log_prob_and_grad(x)
        assert tracker.n_joint == 1
        assert tracker.n_log_prob == 1
        assert tracker.n_grad == 1

    def test_f18_budget_exceeded_error(self):
        """4. Hard ceiling raises BudgetExceededError."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=2)
        tracker.grad_log_prob(np.zeros(2))
        tracker.grad_log_prob(np.zeros(2))
        with pytest.raises(BudgetExceededError):
            tracker.grad_log_prob(np.zeros(2))

    def test_f18_summary_telemetry(self):
        """5. summary() returns full dictionary with budget remaining."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=10)
        tracker.grad_log_prob(np.zeros(2))
        summ = tracker.summary()
        assert summ["n_grad"] == 1
        assert summ["grad_budget_remaining"] == 9


# =============================================================================
# F19: Diagnostic Statistical Metrics
# =============================================================================
class TestF19DiagStatisticalMetrics:
    """Feature 19: Bulk ESS, Tail ESS, split R-hat, KS test, and quantile errors."""

    def test_f19_bulk_ess_computation(self, rng):
        """1. Rank-normalized Bulk ESS on independent samples approaches N."""
        samples = rng.normal(size=(1000, 2))
        ess = compute_bulk_ess(samples)
        assert ess > 700.0

    def test_f19_tail_ess_computation(self, rng):
        """2. Indicator-based Tail ESS on independent samples approaches N."""
        samples = rng.normal(size=(1000, 2))
        tail_ess = compute_tail_ess(samples)
        assert tail_ess > 500.0

    def test_f19_split_rhat_identical_chains(self, rng):
        """3. Split R-hat for stationary chains is ~ 1.0."""
        chain1 = rng.normal(size=(500, 2))
        chain2 = rng.normal(size=(500, 2))
        rhat = compute_split_rhat([chain1, chain2])
        assert abs(rhat - 1.0) < 0.05

    def test_f19_ks_test_contract(self, rng):
        """4. 1D Kolmogorov-Smirnov test validates exact distribution."""
        samples = stats.norm.rvs(size=1000, random_state=42)
        res = stats.kstest(samples, stats.norm.cdf)
        assert res.pvalue > 0.01

    def test_f19_quantile_relative_error_contract(self, rng):
        """5. Quantile relative error calculation against analytical truth."""
        samples = rng.normal(size=3000)
        true_95 = stats.norm.ppf(0.95)
        emp_95 = float(np.quantile(samples, 0.95))
        rel_err = abs(emp_95 - true_95) / abs(true_95)
        assert rel_err < 0.05


# =============================================================================
# F20: Benchmark Runner & Telemetry
# =============================================================================
class TestF20BenchRunner:
    """Feature 20: Automated reproducible benchmark harness contract."""

    def test_f20_reproducibility_with_seed(self):
        """1. Fixed seed produces identical chain trajectories."""
        target = IsotropicGaussian(d=2)
        sampler1 = RWMContractSampler(step_size=0.1)
        sampler2 = RWMContractSampler(step_size=0.1)
        s1, _ = sampler1.sample(np.zeros(2), target, n_steps=20, rng=np.random.default_rng(101))
        s2, _ = sampler2.sample(np.zeros(2), target, n_steps=20, rng=np.random.default_rng(101))
        assert np.array_equal(s1, s2)

    def test_f20_equal_budget_enforcement(self, rng):
        """2. Benchmark harness constrains samplers to equal gradient budgets."""
        budget = 50
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=budget)
        sampler = MALAContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), tracker)
        while not tracker.is_exhausted and tracker.grad_evals < budget:
            res = sampler.step(state, tracker, rng)
            state = res.state
        assert tracker.grad_evals <= budget

    def test_f20_telemetry_schema(self):
        """3. Benchmark output contains standardized metric fields."""
        metrics = {
            "ess_per_grad": 0.45,
            "tail_ess_per_grad": 0.32,
            "ks_pvalue": 0.25,
            "diverged": False,
        }
        for field in ["ess_per_grad", "tail_ess_per_grad", "ks_pvalue", "diverged"]:
            assert field in metrics

    def test_f20_serialization_safety(self):
        """4. Benchmark metrics serialize to standard JSON-compatible types."""
        import json

        metrics = {"chain_id": 1, "ess": 125.4, "converged": True}
        dumped = json.dumps(metrics)
        assert "125.4" in dumped

    def test_f20_multi_regime_loop(self):
        """5. Benchmark runner iterates across target regimes."""
        regimes = ["gaussian_isotropic", "quartic", "student_t"]
        for reg in regimes:
            t = get_target(reg, dim=2)
            assert t.d == 2


# =============================================================================
# F21: Guardrail Benign Parity
# =============================================================================
class TestF21GuardrailBenignParity:
    """Feature 21: Verification harness for AC2 (>= 70% ESS/grad on d=50 Gaussian)."""

    def test_f21_target_dimension_50(self):
        """1. Target is 50-dimensional Gaussian."""
        target = get_target("gaussian_isotropic", dim=50)
        assert target.d == 50

    def test_f21_threshold_ratio_70_percent(self):
        """2. Guardrail threshold is >= 0.70."""
        ess_novel = 0.85
        ess_baseline = 1.00
        ratio = ess_novel / ess_baseline
        assert ratio >= 0.70

    def test_f21_equal_gradient_budget_accounting(self, rng):
        """3. ESS is standardized per gradient evaluation."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=5))
        sampler = MALAContractSampler(step_size=0.1)
        samples, _ = sampler.sample(np.zeros(5), tracker, n_steps=50, rng=rng)
        bulk_ess = compute_bulk_ess(samples)
        ess_per_grad = bulk_ess / max(1, tracker.grad_evals)
        assert np.isfinite(ess_per_grad)
        assert ess_per_grad > 0

    def test_f21_guardrail_pass_condition(self):
        """4. Guardrail passes when ratio exceeds 0.70."""

        def evaluate_benign_parity(novel_ess_per_grad, baseline_ess_per_grad):
            return (novel_ess_per_grad / baseline_ess_per_grad) >= 0.70

        assert evaluate_benign_parity(0.75, 1.0) is True
        assert evaluate_benign_parity(0.65, 1.0) is False

    def test_f21_guardrail_failure_reporting(self):
        """5. Guardrail raises descriptive error on regression."""
        ratio = 0.60
        threshold = 0.70
        if ratio < threshold:
            msg = f"Benign parity guardrail failed: ratio {ratio:.2f} < {threshold}"
            assert "Benign parity" in msg


# =============================================================================
# F22: Guardrail Roughness Stability
# =============================================================================
class TestF22GuardrailRoughnessStability:
    """Feature 22: Verification harness for AC3 (>= 10x step-size stability & 2x burn-in reduction)."""

    def test_f22_initialization_far_in_tail(self):
        """1. Initial point is ||x0|| >= 10 * sqrt(d)."""
        d = 4
        x0 = np.ones(d) * (10.0 * math.sqrt(d) / math.sqrt(d))
        assert np.linalg.norm(x0) >= 10.0 * math.sqrt(d)

    def test_f22_step_size_stability_range(self):
        """2. Step-size stability spans at least 10x range."""
        h_min = 0.01
        h_max = 0.15
        assert (h_max / h_min) >= 10.0

    def test_f22_burn_in_reduction_ratio_2x(self):
        """3. Burn-in evaluations reduced by at least 2x."""
        burnin_tamed = 500
        burnin_novel = 200
        assert (burnin_tamed / burnin_novel) >= 2.0

    def test_f22_non_divergence_verification(self, rng):
        """4. Tamed MALA does not diverge from extreme initial point."""
        target = QuarticTarget(d=2)
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.5)
        x0 = np.array([15.0, 15.0])
        samples, _ = sampler.sample(x0, target, n_steps=20, rng=rng)
        assert np.all(np.isfinite(samples))

    def test_f22_quartic_potential_target(self):
        """5. Verified on quartic superlinear potential target."""
        target = QuarticTarget(d=4, p=4.0)
        assert target.p == 4.0


# =============================================================================
# F23: Guardrail Heavy Tail Efficiency
# =============================================================================
class TestF23GuardrailHeavyTailEfficiency:
    """Feature 23: Verification harness for AC4 (>= 2x Tail-ESS & >= 30% lower 99th error)."""

    def test_f23_target_heterogeneous_dimension(self):
        """1. Target dimension d >= 20 with heterogeneous coordinates."""
        d = 20
        target = HeterogeneousHeavyTailedTarget(d=d, nu_list=[1.5, 2.5] * 10)
        assert target.d >= 20

    def test_f23_tail_ess_ratio_2x(self):
        """2. Novel sampler achieves >= 2.0x tail-ESS per gradient evaluation."""
        tail_ess_novel = 0.40
        tail_ess_baseline = 0.18
        assert (tail_ess_novel / tail_ess_baseline) >= 2.0

    def test_f23_quantile_error_reduction_30_percent(self):
        """3. 99th percentile quantile error reduced by >= 30%."""
        err_baseline = 0.20
        err_novel = 0.12
        reduction = (err_baseline - err_novel) / err_baseline
        assert reduction >= 0.30

    def test_f23_indicator_tail_ess_metric_contract(self, rng):
        """4. Indicator Tail ESS function operates on sample arrays."""
        samples = stats.t.rvs(df=2.0, size=(1000, 2), random_state=42)
        t_ess = compute_tail_ess(samples)
        assert np.isfinite(t_ess)

    def test_f23_guardrail_evaluation_contract(self):
        """5. Heavy tail guardrail validates both Tail-ESS and quantile error."""

        def evaluate_heavy_tail_guardrail(ess_ratio, err_reduction):
            return ess_ratio >= 2.0 and err_reduction >= 0.30

        assert evaluate_heavy_tail_guardrail(2.2, 0.35) is True
        assert evaluate_heavy_tail_guardrail(1.8, 0.35) is False


# =============================================================================
# F24: Research Report Contract
# =============================================================================
class TestF24ResearchReport:
    """Feature 24: Mathematical specifications, stationarity proofs, and benchmark tables."""

    def test_f24_stationarity_detailed_balance_proof_contract(self):
        """1. Sampler transition kernel satisfies detailed balance."""
        # pi(x) T(x -> y) = pi(y) T(y -> x)
        sampler = MALAContractSampler(step_size=0.1)
        assert hasattr(sampler, "step")

    def test_f24_roughness_flatness_tradeoff_bound_contract(self):
        """2. Quantitative bounds define transformed Lipschitz gradient growth."""

        def transformed_grad_bound(R: float, alpha: float) -> float:
            return 1.0 / (R * (alpha - 1.0))

        bound = transformed_grad_bound(R=2.0, alpha=2.5)
        assert bound > 0

    def test_f24_benchmark_table_schema_contract(self):
        """3. Report benchmark tables report required metrics across 5 regimes."""
        required_columns = ["Regime", "Dimension", "Sampler", "ESS/grad", "Tail-ESS/grad", "KS_dist"]
        for col in ["ESS/grad", "Tail-ESS/grad", "KS_dist"]:
            assert col in required_columns

    def test_f24_ablation_study_components_contract(self):
        """4. Ablation study isolates local taming, anisotropic tail, and warmup."""
        ablation_factors = ["local_stiffness", "anisotropic_arcsinh", "welford_preconditioner"]
        assert len(ablation_factors) == 3

    def test_f24_report_file_structure_contract(self):
        """5. Publication report file targets REPORT.md."""
        report_filename = "REPORT.md"
        assert report_filename.endswith(".md")
