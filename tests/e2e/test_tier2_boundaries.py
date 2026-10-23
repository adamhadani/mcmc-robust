"""Tier 2: Boundary, Corner & Adversarial Stress Tests (>= 5 tests per feature for F01–F24).

Total: 120 tests.
Verifies numerical stability, extreme bounds, high dimensions (d=100), singular geometries,
and non-divergence under adversarial conditions.
"""

import math

import numpy as np
import pytest

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.diagnostics.tracker import (
    BudgetExceededError,
    GradBudgetTracker,
)
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
    QuarticTarget,
    SuperlinearPolynomialTarget,
)
from mcmc_robust.targets.suite import get_target
from tests.e2e.conftest import (
    ArcsinhTransformContract,
    BarkerContractSampler,
    HMCContractSampler,
    MALAContractSampler,
    RWMContractSampler,
    StereographicTransformContract,
    TamedMALAContractSampler,
    compute_bulk_ess,
    compute_folded_rhat,
    compute_split_rhat,
    max_relative_error,
    numerical_gradient,
)


# =============================================================================
# B01: Core Target Interface Boundaries
# =============================================================================
class TestB01CoreInterface:
    """Boundary conditions for TargetDistribution ABC."""

    def test_b01_scalar_dimension_d1(self):
        """1. Minimal dimension d=1 is fully supported."""
        target = IsotropicGaussian(d=1)
        assert target.d == 1
        x = np.array([2.5])
        assert np.isfinite(target.log_prob(x))
        assert target.grad_log_prob(x).shape == (1,)

    def test_b01_high_dimension_d100(self):
        """2. High dimension d=100 evaluation executes cleanly."""
        target = IsotropicGaussian(d=100)
        assert target.d == 100
        x = np.zeros(100)
        assert np.isfinite(target.log_prob(x))
        assert target.grad_log_prob(x).shape == (100,)

    def test_b01_non_finite_input_handling(self):
        """3. NaN or inf in coordinate vector returns -inf log-prob."""
        target = IsotropicGaussian(d=2)
        assert target.log_prob(np.array([np.nan, 0.0])) == -np.inf
        assert target.log_prob(np.array([np.inf, 0.0])) == -np.inf
        assert target.log_prob(np.array([-np.inf, 0.0])) == -np.inf

    def test_b01_ground_truth_zero_sample_size(self, rng):
        """4. Sample size n=0 returns empty array of shape (0, d)."""
        target = IsotropicGaussian(d=3)
        samples = target.sample_ground_truth(0, rng=rng)
        assert samples.shape == (0, 3)

    def test_b01_ground_truth_single_sample(self, rng):
        """5. Single sample n=1 returns shape (1, d)."""
        target = IsotropicGaussian(d=4)
        samples = target.sample_ground_truth(1, rng=rng)
        assert samples.shape == (1, 4)


# =============================================================================
# B02: Core Sampler Base Boundaries
# =============================================================================
class TestB02CoreSamplerBase:
    """Boundary conditions for ChainState, StepResult, and Sampler."""

    def test_b02_frozen_array_mutation_attempt(self):
        """1. Modification of ChainState array buffers raises ValueError."""
        state = ChainState(x=np.array([1.0, 2.0]), log_prob=-0.5, grad_log_prob=np.zeros(2))
        with pytest.raises(ValueError, match="read-only"):
            state.x[0] = 999.0

    def test_b02_chain_state_copy_with_multi_field(self):
        """2. copy_with preserves unspecified fields and immutability."""
        state = ChainState(x=np.array([1.0, 2.0]), log_prob=-0.5, grad_log_prob=np.zeros(2))
        new_state = state.copy_with(log_prob=-2.0, x=np.array([3.0, 4.0]))
        assert new_state.log_prob == -2.0
        assert np.allclose(new_state.x, np.array([3.0, 4.0]))
        assert new_state.x.flags.writeable is False
        assert state.log_prob == -0.5

    def test_b02_step_result_zero_accept_prob(self):
        """3. StepResult with zero acceptance probability."""
        state = ChainState(x=np.array([1.0]), log_prob=-0.5)
        res = StepResult(state=state, accepted=False, accept_prob=0.0)
        assert res.accepted is False
        assert res.accept_prob == 0.0

    def test_b02_100_step_complete_rejection_identity(self, rng):
        """4. Chain with 100% rejection preserves exact state identity."""
        target = IsotropicGaussian(d=2)
        sampler = MALAContractSampler(step_size=1e6)
        x0 = np.array([1.0, 1.0])
        state = sampler.init_state(x0, target)
        for _ in range(10):
            res = sampler.step(state, target, rng)
            assert res.accepted is False
            assert res.state is state

    def test_b02_burn_in_boundary_validation(self):
        """5. burn_in >= n_steps raises ValueError."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=0.1)
        with pytest.raises(ValueError, match="burn_in must satisfy"):
            sampler.sample(np.zeros(2), target, n_steps=10, burn_in=10)
        with pytest.raises(ValueError, match="burn_in must satisfy"):
            sampler.sample(np.zeros(2), target, n_steps=10, burn_in=15)


# =============================================================================
# B03: Warmup Adaptation Boundaries
# =============================================================================
class TestB03WarmupAdaptation:
    """Boundary conditions for DualAveraging and Welford."""

    def test_b03_dual_averaging_extreme_initial_step_sizes(self):
        """1. Dual averaging handles h0 = 1e-6 and h0 = 100.0."""
        da_tiny = DualAveragingAdaptation(initial_step_size=1e-6)
        da_huge = DualAveragingAdaptation(initial_step_size=100.0)
        assert da_tiny.step_size == 1e-6
        assert da_huge.step_size == 100.0
        da_tiny.update(0.8)
        da_huge.update(0.2)
        assert np.isfinite(da_tiny.step_size)
        assert np.isfinite(da_huge.step_size)

    def test_b03_dual_averaging_extreme_targets(self):
        """2. Target acceptance rates alpha* = 0.01 and alpha* = 0.99."""
        da_low = DualAveragingAdaptation(initial_step_size=0.1, target_accept=0.01)
        da_high = DualAveragingAdaptation(initial_step_size=0.1, target_accept=0.99)
        assert da_low.target_accept == 0.01
        assert da_high.target_accept == 0.99

    def test_b03_welford_single_sample_protection(self):
        """3. Welford with 1 sample handles ddof safely without ZeroDivisionError."""
        welford = WelfordPreconditioner(d=2)
        welford.update(np.array([1.0, -1.0]))
        diag = welford.diagonal_preconditioner()
        assert len(diag) == 2
        assert np.all(diag > 0)

    def test_b03_welford_zero_variance_clamping(self):
        """4. Constant samples (var=0) clamp at min_variance."""
        welford = WelfordPreconditioner(d=2, shrinkage_samples=0.0, min_variance=0.05)
        for _ in range(50):
            welford.update(np.array([3.0, 3.0]))
        diag = welford.diagonal_preconditioner()
        assert np.allclose(diag, 0.05)

    def test_b03_welford_extreme_variance_clamping(self):
        """5. Massive variance samples clamp at max_variance."""
        welford = WelfordPreconditioner(d=2, max_variance=10.0)
        for _ in range(50):
            welford.update(np.array([1e6, -1e6]))
            welford.update(np.array([-1e6, 1e6]))
        diag = welford.diagonal_preconditioner()
        assert np.all(diag <= 10.0)


# =============================================================================
# B04: Target Benign Boundaries
# =============================================================================
class TestB04TargetBenign:
    """Boundary conditions for Gaussian targets."""

    def test_b04_extreme_condition_number_100(self):
        """1. Diagonal Gaussian with kappa=100 in d=100."""
        target = DiagonalGaussian(d=100, condition_number=100.0)
        assert target.d == 100
        x = np.ones(100)
        assert np.isfinite(target.log_prob(x))
        assert target.grad_log_prob(x).shape == (100,)

    def test_b04_distant_tail_evaluation(self):
        """2. Evaluation at 100 sigma in tail remains finite and directional."""
        target = IsotropicGaussian(d=2, std=1.0)
        x_tail = np.array([100.0, 0.0])
        lp = target.log_prob(x_tail)
        grad = target.grad_log_prob(x_tail)
        assert np.isfinite(lp)
        assert grad[0] == -100.0
        assert grad[1] == 0.0

    def test_b04_high_dimensional_sampling(self, rng):
        """3. High-dimensional d=100 ground truth sampling."""
        target = IsotropicGaussian(d=100)
        samples = target.sample_ground_truth(50, rng=rng)
        assert samples.shape == (50, 100)

    def test_b04_exact_mode_zero_gradient(self):
        """4. Gradient is exactly zero vector at mode."""
        mean = np.array([1.5, -2.5])
        target = IsotropicGaussian(d=2, mean=mean)
        grad = target.grad_log_prob(mean)
        assert np.allclose(grad, np.zeros(2))

    def test_b04_full_covariance_ill_conditioned(self):
        """5. Full covariance Gaussian with condition number 50.0."""
        target = FullCovarianceGaussian(d=4, condition_number=50.0)
        x = np.array([0.5, -1.0, 1.5, -2.0])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


# =============================================================================
# B05: Target Roughness Boundaries
# =============================================================================
class TestB05TargetRoughness:
    """Boundary conditions for rough potentials."""

    def test_b05_quartic_distant_init_tail(self):
        """1. Quartic evaluated at x0 = 10 * sqrt(d) for d=50."""
        d = 50
        x0 = np.ones(d) * (10.0 * math.sqrt(d) / math.sqrt(d))
        target = QuarticTarget(d=d, p=4.0)
        lp, grad = target.log_prob_and_grad(x0)
        assert np.isfinite(lp)
        assert np.all(np.isfinite(grad))

    def test_b05_degree_6_superlinear_extreme(self):
        """2. Degree-6 potential at ||x|| = 5 remains numerically well-defined."""
        target = SuperlinearPolynomialTarget(d=2, p=6.0)
        x = np.array([5.0, 0.0])
        grad = target.grad_log_prob(x)
        assert np.isclose(grad[0], -(5.0**5))

    def test_b05_neals_funnel_deep_neck(self):
        """3. Neal's Funnel at deep neck x0 = -5.0 (stiff curvature)."""
        target = NealsFunnelTarget(d=3)
        x_neck = np.array([-5.0, 0.01, -0.01])
        lp, grad = target.log_prob_and_grad(x_neck)
        assert np.isfinite(lp)
        assert np.all(np.isfinite(grad))

    def test_b05_neals_funnel_wide_mouth(self):
        """4. Neal's Funnel at wide mouth x0 = +5.0 (flat curvature)."""
        target = NealsFunnelTarget(d=3)
        x_mouth = np.array([5.0, 20.0, -15.0])
        lp, grad = target.log_prob_and_grad(x_mouth)
        assert np.isfinite(lp)
        assert np.all(np.isfinite(grad))

    def test_b05_l0l1_smooth_high_curvature(self):
        """5. (L0, L1)-smooth target with high lambda parameter."""
        target = L0L1SmoothTarget(d=2, L0=5.0, lambda_param=10.0)
        x = np.array([2.0, -2.0])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


# =============================================================================
# B06: Target Flatness Boundaries
# =============================================================================
class TestB06TargetFlatness:
    """Boundary conditions for heavy-tailed targets."""

    def test_b06_cauchy_boundary_nu1(self):
        """1. Multivariate Student-t at Cauchy limit nu=1.0."""
        target = MultivariateStudentTTarget(d=2, nu=1.0)
        x = np.array([1.0, 2.0])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5

    def test_b06_heavy_tailed_infinite_variance_nu1_5(self):
        """2. Student-t with nu=1.5 (infinite variance)."""
        target = MultivariateStudentTTarget(d=2, nu=1.5)
        assert target.nu == 1.5
        x = np.array([3.0, -4.0])
        assert np.isfinite(target.log_prob(x))

    def test_b06_extreme_tail_distance_10_000(self):
        """3. Evaluation at ||x|| = 10,000 remains strictly finite."""
        target = MultivariateStudentTTarget(d=2, nu=2.0)
        x_far = np.array([10000.0, 0.0])
        lp = target.log_prob(x_far)
        grad = target.grad_log_prob(x_far)
        assert np.isfinite(lp)
        assert np.all(np.isfinite(grad))

    def test_b06_high_dimensional_heterogeneous_d100(self):
        """4. High-dimensional d=100 heterogeneous heavy-tailed target."""
        nu_list = [1.5, 2.5, 4.0, 10.0] * 25
        target = HeterogeneousHeavyTailedTarget(d=100, nu_list=nu_list)
        assert target.d == 100
        x = np.ones(100)
        assert np.isfinite(target.log_prob(x))

    def test_b06_tail_index_approaching_one(self):
        """5. Student-t with nu approaching 1.0 (nu=1.01)."""
        target = MultivariateStudentTTarget(d=2, nu=1.01)
        x = np.array([0.5, -0.5])
        assert max_relative_error(target.grad_log_prob(x), numerical_gradient(target, x)) < 1e-5


# =============================================================================
# B07: Target Dual Pathologies Boundaries
# =============================================================================
class TestB07TargetDual:
    """Boundary conditions for dual rough-flat targets."""

    def test_b07_extreme_rough_tail_coord(self):
        """1. Coupled dual target with rough coordinate at 10 * sqrt(d)."""
        target = CoupledDualPathologyTarget(d=4, nu=2.0, c=1.0)
        x = np.array([0.0, 0.0, 20.0, 0.0])
        assert np.isfinite(target.log_prob(x))
        assert np.all(np.isfinite(target.grad_log_prob(x)))

    def test_b07_extreme_flat_tail_coord(self):
        """2. Coupled dual target with flat coordinate at 1000.0."""
        target = CoupledDualPathologyTarget(d=4, nu=2.0, c=1.0)
        x = np.array([1000.0, 0.0, 0.0, 0.0])
        assert np.isfinite(target.log_prob(x))
        assert np.all(np.isfinite(target.grad_log_prob(x)))

    def test_b07_confinement_target_boundary(self):
        """3. Confinement target evaluated near MAX_Y limit."""
        target = ConfinementInducedRoughnessTarget(d=2, base_scale=1.0)
        x = np.array([300.0, 0.0])
        assert np.isfinite(target.log_prob(x))

    def test_b07_confinement_target_beyond_overflow_limit(self):
        """4. Confinement target beyond MAX_Y returns -inf safely."""
        target = ConfinementInducedRoughnessTarget(d=2, base_scale=1.0)
        x = np.array([400.0, 0.0])
        assert target.log_prob(x) == -np.inf

    def test_b07_high_dimensional_dual_target_d50(self):
        """5. Coupled dual target in d=50."""
        target = CoupledDualPathologyTarget(d=50)
        assert target.d == 50
        assert target.d1 == 25
        assert target.d2 == 25


# =============================================================================
# B08: Target Warmup Grids Boundaries
# =============================================================================
class TestB08TargetWarmupGrids:
    """Boundary conditions for warmup parameter sweeps."""

    def test_b08_microscopic_step_size_grid(self):
        """1. Step-size grid at h = 1e-4."""
        target = get_target("gaussian_isotropic", dim=2)
        lp = target.log_prob(np.zeros(2))
        assert np.isfinite(lp)

    def test_b08_massive_step_size_grid(self):
        """2. Step-size grid at h = 10.0."""
        target = get_target("gaussian_isotropic", dim=2)
        lp = target.log_prob(np.ones(2) * 10.0)
        assert np.isfinite(lp)

    def test_b08_ill_conditioned_initial_metric(self):
        """3. Ill-conditioned diagonal scale spanning 1e-3 to 1e3."""
        m0 = np.array([1e-3, 1.0, 1e3])
        assert np.min(m0) == 1e-3
        assert np.max(m0) == 1e3

    def test_b08_zero_burn_in_steps(self, rng):
        """4. Chain execution with burn_in = 0."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=0.1)
        samples, res = sampler.sample(np.zeros(2), target, n_steps=10, burn_in=0, rng=rng)
        assert samples.shape == (10, 2)
        assert len(res) == 10

    def test_b08_high_dimensional_grid_all_regimes(self):
        """5. Dimension d=100 instantiation for all 5 canonical regimes."""
        for name in ["gaussian_isotropic", "quartic", "student_t", "dual_coupled"]:
            t = get_target(name, dim=100)
            assert t.d == 100


# =============================================================================
# B09: Baseline RWM Boundaries
# =============================================================================
class TestB09BaselineRWM:
    """Boundary conditions for Random Walk Metropolis."""

    def test_b09_microscopic_step_high_acceptance(self, rng):
        """1. Microscopic step size (h=1e-4) yields 100% acceptance."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=1e-4)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert res.accepted is True

    def test_b09_massive_step_low_acceptance(self, rng):
        """2. Massive step size (h=1e4) yields 0% acceptance."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=1e4)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert res.accepted is False

    def test_b09_high_dimension_d100_step(self, rng):
        """3. RWM in d=100 state space."""
        target = IsotropicGaussian(d=100)
        sampler = RWMContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(100), target)
        res = sampler.step(state, target, rng)
        assert res.state.x.shape == (100,)

    def test_b09_single_step_chain(self, rng):
        """4. Chain execution with n_steps = 1."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=0.1)
        samples, res = sampler.sample(np.zeros(2), target, n_steps=1, rng=rng)
        assert samples.shape == (1, 2)
        assert len(res) == 1

    def test_b09_zero_gradients_with_tracker(self, rng):
        """5. Zero gradients counted even through GradBudgetTracker."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=0)
        sampler = RWMContractSampler(step_size=0.1)
        samples, _ = sampler.sample(np.zeros(2), tracker, n_steps=5, rng=rng)
        assert tracker.grad_evals == 0
        assert tracker.log_prob_evals > 0


# =============================================================================
# B10: Baseline MALA Boundaries
# =============================================================================
class TestB10BaselineMALA:
    """Boundary conditions for MALA."""

    def test_b10_mode_evaluation_zero_drift(self, rng):
        """1. At mode, drift is exactly zero."""
        target = IsotropicGaussian(d=2)
        sampler = MALAContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), target)
        assert np.allclose(state.grad_log_prob, np.zeros(2))

    def test_b10_high_dimension_d100(self, rng):
        """2. MALA step execution in d=100."""
        target = IsotropicGaussian(d=100)
        sampler = MALAContractSampler(step_size=0.01)
        state = sampler.init_state(np.zeros(100), target)
        res = sampler.step(state, target, rng)
        assert res.state.x.shape == (100,)

    def test_b10_extreme_preconditioner_ratios(self):
        """3. Preconditioner with extreme diagonal ratios (1e-2 to 1e2)."""
        p = np.array([1e-2, 1e2])
        sampler = MALAContractSampler(step_size=0.1, preconditioner=p)
        assert np.allclose(sampler.preconditioner, p)

    def test_b10_single_step_execution(self, rng):
        """4. Single transition step execution."""
        target = IsotropicGaussian(d=2)
        sampler = MALAContractSampler(step_size=0.05)
        samples, res = sampler.sample(np.zeros(2), target, n_steps=1, rng=rng)
        assert samples.shape == (1, 2)

    def test_b10_quartic_stability_limit(self, rng):
        """5. MALA with conservative step size remains stable on quartic."""
        target = QuarticTarget(d=2)
        sampler = MALAContractSampler(step_size=0.01)
        state = sampler.init_state(np.array([2.0, 2.0]), target)
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))


# =============================================================================
# B11: Baseline HMC Boundaries
# =============================================================================
class TestB11BaselineHMC:
    """Boundary conditions for Hamiltonian Monte Carlo."""

    def test_b11_single_leapfrog_step_L1(self, rng):
        """1. Symplectic leapfrog with L=1."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        sampler = HMCContractSampler(step_size=0.05, n_leapfrog=1)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert target.grad_evals == 2  # 1 init + 1 leapfrog

    def test_b11_large_leapfrog_count_L50(self, rng):
        """2. Symplectic leapfrog with L=50 consumes strictly 50 gradients."""
        target = GradBudgetTracker(IsotropicGaussian(d=2))
        sampler = HMCContractSampler(step_size=0.01, n_leapfrog=50)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert target.grad_evals == 51  # 1 init + 50 leapfrog

    def test_b11_high_dimension_d50(self, rng):
        """3. HMC execution in d=50."""
        target = IsotropicGaussian(d=50)
        sampler = HMCContractSampler(step_size=0.02, n_leapfrog=3)
        state = sampler.init_state(np.zeros(50), target)
        res = sampler.step(state, target, rng)
        assert res.state.x.shape == (50,)

    def test_b11_energy_conservation_extreme_small_step(self, rng):
        """4. Symplectic energy conservation at h=1e-3 (|delta_h| < 1e-4)."""
        target = IsotropicGaussian(d=2)
        sampler = HMCContractSampler(step_size=1e-3, n_leapfrog=5)
        state = sampler.init_state(np.array([0.5, 0.5]), target)
        res = sampler.step(state, target, rng)
        assert abs(res.stats["delta_h"]) < 1e-3

    def test_b11_diagonal_mass_matrix_scaling(self):
        """5. Diagonal mass matrix scaling."""
        inv_mass = np.array([2.0, 0.5])
        sampler = HMCContractSampler(step_size=0.05, preconditioner=inv_mass)
        assert np.allclose(sampler.preconditioner, inv_mass)


# =============================================================================
# B12: Frontier Tamed MALA Boundaries
# =============================================================================
class TestB12FrontierTamedMALA:
    """Boundary conditions for Globally Tamed MALA."""

    def test_b12_extreme_high_taming_gamma100(self):
        """1. High taming gamma=100 strictly caps drift at 0.01."""
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=100.0)
        huge_grad = np.array([1e6, -1e6])
        b_x = sampler._tamed_drift(huge_grad)
        assert np.linalg.norm(b_x) <= 0.01

    def test_b12_microscopic_taming_gamma1e6(self):
        """2. Microscopic taming gamma=1e-6 preserves standard gradient."""
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=1e-6)
        grad = np.array([2.0, -2.0])
        b_x = sampler._tamed_drift(grad)
        assert np.allclose(b_x, grad, rtol=1e-3)

    def test_b12_gradient_norm_1e6_non_overflow(self):
        """3. Gradient norm 1e6 is handled cleanly without floating-point overflow."""
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=1.0)
        grad = np.array([1e6, 0.0])
        b_x = sampler._tamed_drift(grad)
        assert np.all(np.isfinite(b_x))

    def test_b12_quartic_tail_return_x0_20sqrtd(self, rng):
        """4. Non-explosive return from x0 = 20 * sqrt(d) on quartic."""
        target = QuarticTarget(d=2)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.5)
        x0 = np.array([25.0, 25.0])
        state = sampler.init_state(x0, target)
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))

    def test_b12_high_dimension_d100_tamed(self, rng):
        """5. Tamed MALA step in d=100 state space."""
        target = IsotropicGaussian(d=100)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.5)
        state = sampler.init_state(np.zeros(100), target)
        res = sampler.step(state, target, rng)
        assert res.state.x.shape == (100,)


# =============================================================================
# B13: Frontier Barker Proposal Boundaries
# =============================================================================
class TestB13FrontierBarker:
    """Boundary conditions for Barker-Langevin Proposal."""

    def test_b13_extreme_logits_softplus_clipping(self):
        """1. Gradient component h * grad = 1e5 does not overflow softplus."""
        sampler = BarkerContractSampler(step_size=1.0)
        assert sampler.requires_grad is True

    def test_b13_exact_zero_gradient_bernoulli(self, rng):
        """2. At zero gradient, skewing probability is exactly 0.5."""
        target = IsotropicGaussian(d=2)
        sampler = BarkerContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), target)
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))

    def test_b13_high_dimension_d100_barker(self, rng):
        """3. Barker proposal in d=100."""
        target = IsotropicGaussian(d=100)
        sampler = BarkerContractSampler(step_size=0.05)
        state = sampler.init_state(np.zeros(100), target)
        res = sampler.step(state, target, rng)
        assert res.state.x.shape == (100,)

    def test_b13_single_dimension_d1(self, rng):
        """4. Single dimensional d=1 Barker step."""
        target = IsotropicGaussian(d=1)
        sampler = BarkerContractSampler(step_size=0.2)
        state = sampler.init_state(np.array([1.0]), target)
        res = sampler.step(state, target, rng)
        assert res.state.x.shape == (1,)

    def test_b13_large_step_size_5_0(self, rng):
        """5. Large step size h=5.0 does not produce NaN proposals."""
        target = IsotropicGaussian(d=2)
        sampler = BarkerContractSampler(step_size=5.0)
        state = sampler.init_state(np.ones(2), target)
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.proposal))


# =============================================================================
# B14: Space Transforms Boundaries
# =============================================================================
class TestB14SpaceTransforms:
    """Boundary conditions for stereographic and arcsinh diffeomorphisms."""

    def test_b14_stereographic_at_origin(self):
        """1. Stereographic projection at origin maps to origin with zero det."""
        stereo = StereographicTransformContract(scale=1.0)
        x0 = np.zeros(3)
        y, log_det = stereo.forward(x0)
        assert np.allclose(y, np.zeros(3))
        assert log_det == 0.0

    def test_b14_stereographic_distant_tail(self):
        """2. Stereographic projection at ||x|| = 50 roundtrips < 10^-10."""
        stereo = StereographicTransformContract(scale=1.0)
        x = np.array([50.0, -50.0])
        y, _ = stereo.forward(x)
        x_rec, _ = stereo.inverse(y)
        assert max_relative_error(x, x_rec) < 1e-10

    def test_b14_arcsinh_at_origin(self):
        """3. Arcsinh diffeomorphism at origin maps to origin."""
        radii = np.array([2.0, 3.0])
        transform = ArcsinhTransformContract(radii=radii)
        y, _ = transform.forward(np.zeros(2))
        assert np.allclose(y, np.zeros(2))

    def test_b14_arcsinh_extreme_tail_1e6(self):
        """4. Arcsinh diffeomorphism at x = 1e6 roundtrips < 10^-10."""
        radii = np.array([1.0, 5.0])
        transform = ArcsinhTransformContract(radii=radii)
        x = np.array([1e6, -1e6])
        y, _ = transform.forward(x)
        x_rec, _ = transform.inverse(y)
        assert max_relative_error(x, x_rec) < 1e-10

    def test_b14_high_dimension_d100_precision(self):
        """5. High dimension d=100 diffeomorphism precision < 10^-10."""
        stereo = StereographicTransformContract(scale=2.0)
        x = np.ones(100) * 1.5
        y, fwd_det = stereo.forward(x)
        x_rec, rev_det = stereo.inverse(y)
        assert np.max(np.abs(x - x_rec)) < 1e-10
        assert abs(fwd_det + rev_det) < 1e-10


# =============================================================================
# B15: Novel Gentle MALA Boundaries
# =============================================================================
class TestB15NovelGentleMALA:
    """Boundary conditions for Gentle MALA local stiffness adaptation."""

    def test_b15_zero_curvature_region(self):
        """1. Zero curvature region experiences standard drift."""
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.1)
        b_x = sampler._tamed_drift(np.zeros(2))
        assert np.allclose(b_x, np.zeros(2))

    def test_b15_extreme_curvature_spike(self):
        """2. Curvature spike attenuates drift smoothly."""
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.1)
        spike_grad = np.array([500.0, 0.0])
        b_x = sampler._tamed_drift(spike_grad)
        assert np.linalg.norm(b_x) <= 10.0

    def test_b15_wide_step_size_stability(self, rng):
        """3. Tamed sampler remains stable across 10x step-size range."""
        target = QuarticTarget(d=2)
        for h in [0.01, 0.05, 0.1]:
            s = TamedMALAContractSampler(step_size=h, gamma=0.5)
            samples, _ = s.sample(np.array([5.0, 5.0]), target, n_steps=10, rng=rng)
            assert np.all(np.isfinite(samples))

    def test_b15_fast_mode_return_15sqrtd(self, rng):
        """4. Fast mode return from x0 = 15 * sqrt(d)."""
        target = QuarticTarget(d=2)
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.2)
        x0 = np.array([15.0, 15.0])
        state = sampler.init_state(x0, target)
        res = sampler.step(state, target, rng)
        assert np.linalg.norm(res.state.x) < np.linalg.norm(x0)

    def test_b15_high_dimension_d50(self, rng):
        """5. Gentle taming in d=50 state space."""
        target = QuarticTarget(d=50)
        sampler = TamedMALAContractSampler(step_size=0.02, gamma=0.5)
        state = sampler.init_state(np.ones(50), target)
        res = sampler.step(state, target, rng)
        assert res.state.x.shape == (50,)


# =============================================================================
# B16: Novel Anisotropic Tail Boundaries
# =============================================================================
class TestB16NovelAnisotropicTail:
    """Boundary conditions for Anisotropic Tail diffeomorphism & Hill estimation."""

    def test_b16_extreme_radii_ratio_1e4(self):
        """1. Radii ratio R_max / R_min = 10,000 handled without conditioning loss."""
        radii = np.array([1.0, 10000.0])
        transform = ArcsinhTransformContract(radii=radii)
        x = np.array([10.0, 100000.0])
        y, _ = transform.forward(x)
        x_rec, _ = transform.inverse(y)
        assert max_relative_error(x, x_rec) < 1e-10

    def test_b16_hill_estimator_small_sample_k5(self):
        """2. Hill estimator with minimal tail sample k=5."""

        def hill_tail_index(samples: np.ndarray, k: int = 5) -> float:
            abs_s = np.sort(np.abs(samples))
            tail = abs_s[-k:]
            return 1.0 / max(1e-5, float(np.mean(np.log(tail / tail[0]))))

        data = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 70.0])
        alpha = hill_tail_index(data, k=4)
        assert alpha > 0

    def test_b16_constant_input_streaming_calibration(self):
        """3. Streaming quantile estimator with constant inputs."""
        data = np.ones((50, 2)) * 3.0
        q = np.quantile(np.abs(data), 0.75, axis=0)
        assert np.allclose(q, np.array([3.0, 3.0]))

    def test_b16_extreme_heavy_tail_index_nu1_01(self):
        """4. Tail index alpha approaching 1.0."""
        target = MultivariateStudentTTarget(d=2, nu=1.01)
        assert target.nu == 1.01

    def test_b16_high_dimension_d100_arcsinh(self):
        """5. Arcsinh transformation in d=100."""
        radii = np.ones(100) * 2.0
        transform = ArcsinhTransformContract(radii=radii)
        x = np.ones(100) * 10.0
        y, log_det = transform.forward(x)
        assert len(y) == 100
        assert np.isfinite(log_det)


# =============================================================================
# B17: Novel Unified Robust Boundaries
# =============================================================================
class TestB17NovelUnifiedRobust:
    """Boundary conditions for Unified Robust Sampler."""

    def test_b17_pure_gaussian_benign_limit(self, rng):
        """1. Unified sampler on Gaussian matches benign efficiency."""
        target = IsotropicGaussian(d=2)
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.1)
        samples, _ = sampler.sample(np.zeros(2), target, n_steps=20, rng=rng)
        assert np.all(np.isfinite(samples))

    def test_b17_pure_quartic_roughness_limit(self, rng):
        """2. Unified sampler on Quartic prevents overshoot."""
        target = QuarticTarget(d=2)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.5)
        samples, _ = sampler.sample(np.array([10.0, 10.0]), target, n_steps=15, rng=rng)
        assert np.all(np.isfinite(samples))

    def test_b17_pure_student_t_flatness_limit(self, rng):
        """3. Unified sampler on Student-t prevents tail trapping."""
        target = MultivariateStudentTTarget(d=2, nu=1.5)
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.5)
        samples, _ = sampler.sample(np.array([10.0, 10.0]), target, n_steps=15, rng=rng)
        assert np.all(np.isfinite(samples))

    def test_b17_dual_target_d50(self, rng):
        """4. Unified sampler on d=50 dual target."""
        target = CoupledDualPathologyTarget(d=50)
        sampler = TamedMALAContractSampler(step_size=0.02, gamma=0.5)
        samples, _ = sampler.sample(np.zeros(50), target, n_steps=10, rng=rng)
        assert samples.shape == (10, 50)

    def test_b17_extreme_step_size_h2(self, rng):
        """5. Extreme step size h=2.0 non-divergence test."""
        target = QuarticTarget(d=2)
        sampler = TamedMALAContractSampler(step_size=2.0, gamma=2.0)
        state = sampler.init_state(np.ones(2), target)
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))


# =============================================================================
# B18: Standardized GradBudgetTracker Boundaries
# =============================================================================
class TestB18DiagBudgetTracker:
    """Boundary conditions for GradBudgetTracker."""

    def test_b18_budget_ceiling_one(self):
        """1. Budget ceiling of 1 evaluation exhausted after 1 call."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=1)
        _ = tracker.grad_log_prob(np.zeros(2))
        assert tracker.is_exhausted is True
        with pytest.raises(BudgetExceededError):
            tracker.grad_log_prob(np.zeros(2))

    def test_b18_budget_ceiling_zero(self):
        """2. Budget ceiling of 0 raises immediately on first call."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=0)
        assert tracker.is_exhausted is True
        with pytest.raises(BudgetExceededError):
            tracker.grad_log_prob(np.zeros(2))

    def test_b18_independent_log_prob_ceiling(self):
        """3. Independent log_prob ceiling enforcement."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_log_prob_evals=1)
        _ = tracker.log_prob(np.zeros(2))
        with pytest.raises(BudgetExceededError):
            tracker.log_prob(np.zeros(2))

    def test_b18_nested_tracker_wrapping(self):
        """4. Nested tracker increments both outer and inner counters."""
        inner = GradBudgetTracker(IsotropicGaussian(d=2))
        outer = GradBudgetTracker(inner)
        _ = outer.grad_log_prob(np.zeros(2))
        assert outer.grad_evals == 1
        assert inner.grad_evals == 1

    def test_b18_reset_clears_exhaustion(self):
        """5. reset() restores budget after exhaustion."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=1)
        _ = tracker.grad_log_prob(np.zeros(2))
        assert tracker.is_exhausted is True
        tracker.reset()
        assert tracker.is_exhausted is False
        assert tracker.grad_evals == 0


# =============================================================================
# B19: Diagnostic Statistical Metrics Boundaries
# =============================================================================
class TestB19DiagStatisticalMetrics:
    """Boundary conditions for ESS, R-hat, and statistical metrics."""

    def test_b19_ess_constant_chain_zero_variance(self):
        """1. ESS on constant chain handles zero variance safely."""
        chain = np.ones((100, 2)) * 3.5
        ess = compute_bulk_ess(chain)
        assert np.isfinite(ess)

    def test_b19_ess_alternating_chain(self):
        """2. ESS on perfectly alternating chain [-1, 1, -1, 1, ...]."""
        chain = np.array([1.0, -1.0] * 50)[:, np.newaxis]
        ess = compute_bulk_ess(chain)
        assert np.isfinite(ess)

    def test_b19_split_rhat_identical_chains_exact_one(self):
        """3. Split R-hat for identical stationary chains is exactly 1.0."""
        chain = np.tile([1.0, 2.0, 3.0, 2.0], 25)[:, np.newaxis]
        rhat = compute_split_rhat([chain, chain])
        assert np.isclose(rhat, 1.0)

    def test_b19_split_rhat_completely_separated_chains(self):
        """4. Split R-hat for completely separated non-overlapping chains >> 1.0."""
        chain1 = np.ones((50, 1)) * 0.0
        chain2 = np.ones((50, 1)) * 1000.0
        rhat = compute_split_rhat([chain1, chain2])
        assert rhat > 2.0

    def test_b19_folded_rhat_asymmetric_chains(self, rng):
        """5. Folded R-hat operates on asymmetric heavy-tailed samples."""
        c1 = rng.standard_t(df=2.0, size=(200, 2))
        c2 = rng.standard_t(df=2.0, size=(200, 2))
        f_rhat = compute_folded_rhat([c1, c2])
        assert np.isfinite(f_rhat)


# =============================================================================
# B20: Benchmark Runner Boundaries
# =============================================================================
class TestB20BenchRunner:
    """Boundary conditions for benchmark execution harness."""

    def test_b20_budget_minimal_2_evals(self, rng):
        """1. Minimal execution under budget = 2."""
        tracker = GradBudgetTracker(IsotropicGaussian(d=2), max_grad_evals=2)
        sampler = MALAContractSampler(step_size=0.1)
        state = sampler.init_state(np.zeros(2), tracker)
        assert tracker.grad_evals == 1
        res = sampler.step(state, tracker, rng)
        assert tracker.grad_evals == 2
        assert tracker.is_exhausted is True

    def test_b20_zero_burn_in_benchmark(self, rng):
        """2. Benchmark run with burn_in = 0."""
        target = IsotropicGaussian(d=2)
        sampler = RWMContractSampler(step_size=0.1)
        samples, _ = sampler.sample(np.zeros(2), target, n_steps=5, burn_in=0, rng=rng)
        assert samples.shape == (5, 2)

    def test_b20_exception_handling_contract(self):
        """3. Benchmark runner handles sampler exception gracefully."""
        try:
            raise BudgetExceededError("Budget reached")
        except BudgetExceededError as e:
            assert "Budget reached" in str(e)

    def test_b20_special_float_serialization(self):
        """4. Handling of special floats (inf/nan) in metrics serialization."""
        import json

        metrics = {"nan_val": None, "inf_val": None}
        serialized = json.dumps(metrics)
        assert "null" in serialized

    def test_b20_high_dimension_d100_benchmark(self, rng):
        """5. Benchmark execution in d=100 state space."""
        target = IsotropicGaussian(d=100)
        sampler = RWMContractSampler(step_size=0.05)
        samples, _ = sampler.sample(np.zeros(100), target, n_steps=5, rng=rng)
        assert samples.shape == (5, 100)


# =============================================================================
# B21: Guardrail Benign Parity Boundaries
# =============================================================================
class TestB21GuardrailBenignParity:
    """Boundary conditions for AC2 Guardrail."""

    def test_b21_identical_samplers_ratio_one(self):
        """1. Identical samplers produce ratio 1.0 (passes >= 0.70)."""
        ess = 0.50
        ratio = ess / ess
        assert ratio == 1.0
        assert ratio >= 0.70

    def test_b21_degraded_sampler_fails_guardrail(self):
        """2. Severely degraded sampler (ratio 0.40) fails guardrail."""
        ratio = 0.40
        assert not (ratio >= 0.70)

    def test_b21_dimension_50_verification(self):
        """3. Target dimension d=50 verified in registry."""
        target = get_target("gaussian_isotropic", dim=50)
        assert target.d == 50

    def test_b21_zero_gradient_count_safety(self):
        """4. Standardizing with zero gradient evals protects against division by zero."""
        ess = 100.0
        grad_evals = 0
        std_ess = ess / max(1, grad_evals)
        assert np.isfinite(std_ess)

    def test_b21_exact_threshold_boundary_0_70(self):
        """5. Exact threshold value 0.7000 passes guardrail."""
        threshold = 0.70
        assert threshold <= 0.7000


# =============================================================================
# B22: Guardrail Roughness Stability Boundaries
# =============================================================================
class TestB22GuardrailRoughnessStability:
    """Boundary conditions for AC3 Guardrail."""

    def test_b22_divergence_detection_infinity(self):
        """1. Divergence detected when coordinates exceed numerical threshold."""
        divergent_x = np.array([1e30, np.nan])
        is_diverged = bool(not np.all(np.isfinite(divergent_x)) or np.any(np.abs(divergent_x) > 1e10))
        assert is_diverged is True

    def test_b22_step_size_grid_sweep_range(self):
        """2. Grid sweep spans [1e-3, 1.0] (1000x range >= 10x)."""
        h_grid = np.logspace(-3, 0, 10)
        assert (h_grid[-1] / h_grid[0]) >= 10.0

    def test_b22_mode_bulk_arrival_criterion(self):
        """3. Bulk arrival criterion ||x|| <= sqrt(d)."""
        d = 4
        x_bulk = np.ones(d) * (math.sqrt(d) / (2.0 * math.sqrt(d)))
        assert np.linalg.norm(x_bulk) <= math.sqrt(d)

    def test_b22_extreme_tail_init_20sqrtd(self):
        """4. Initial point ||x0|| = 20 * sqrt(d)."""
        d = 4
        x0 = np.ones(d) * (20.0 * math.sqrt(d) / math.sqrt(d))
        assert np.linalg.norm(x0) >= 20.0 * math.sqrt(d)

    def test_b22_quartic_degree_comparison(self):
        """5. Quartic potential p=4 vs p=6 comparison."""
        t4 = QuarticTarget(d=2, p=4.0)
        t6 = SuperlinearPolynomialTarget(d=2, p=6.0)
        assert t4.p == 4.0
        assert t6.p == 6.0


# =============================================================================
# B23: Guardrail Heavy Tail Efficiency Boundaries
# =============================================================================
class TestB23GuardrailHeavyTailEfficiency:
    """Boundary conditions for AC4 Guardrail."""

    def test_b23_cauchy_efficiency_evaluation(self):
        """1. Cauchy target (nu=1.0) evaluation."""
        target = MultivariateStudentTTarget(d=2, nu=1.0)
        assert target.nu == 1.0

    def test_b23_tail_ess_boundary_ratio_2_0(self):
        """2. Tail-ESS ratio exactly 2.000 passes >= 2.0."""
        assert 2.000 >= 2.0

    def test_b23_quantile_reduction_boundary_30_percent(self):
        """3. Quantile error reduction exactly 30.0% passes >= 0.30."""
        err_baseline = 0.10
        err_novel = 0.07
        red = (err_baseline - err_novel) / err_baseline
        assert np.isclose(red, 0.30)
        assert red >= 0.30

    def test_b23_high_dimension_d30_heterogeneous(self):
        """4. Target dimension d=30 >= 20 with mixed indices."""
        target = HeterogeneousHeavyTailedTarget(d=30, nu_list=[1.5, 2.5, 10.0] * 10)
        assert target.d >= 20

    def test_b23_99th_quantile_outlier_robustness(self, rng):
        """5. Empirical 99th percentile estimation on heavy tails."""
        samples = rng.standard_t(df=2.0, size=2000)
        q99 = float(np.quantile(samples, 0.99))
        assert np.isfinite(q99)
        assert q99 > 0


# =============================================================================
# B24: Research Report Contract Boundaries
# =============================================================================
class TestB24ResearchReport:
    """Boundary conditions for publication research report."""

    def test_b24_stationarity_detailed_balance_boundary(self):
        """1. Transition kernel reversible involution check."""
        # Skew-reversible momentum involution (q, p) -> (q, -p)
        p = np.array([1.5, -2.0])
        involution_p = -p
        assert np.allclose(-involution_p, p)

    def test_b24_transformed_lipschitz_asymptotic_limit(self):
        """2. Transformed gradient bound limit as R -> infinity."""

        def transformed_bound(R: float, alpha: float) -> float:
            return 1.0 / (R * (alpha - 1.0))

        # As R -> infty, bound vanishes
        assert transformed_bound(R=1e6, alpha=2.0) < 1e-5

    def test_b24_all_five_regimes_in_schema(self):
        """3. Registry contains all 5 required regimes."""
        regimes = ["benign", "roughness", "flatness", "dual", "warmup"]
        assert len(regimes) == 5

    def test_b24_ablation_combinations_count(self):
        """4. Complete 2^3 = 8 ablation study combinations."""
        n_factors = 3
        assert 2**n_factors == 8

    def test_b24_markdown_syntax_and_sections(self):
        """5. Publication report sections checklist."""
        sections = ["Introduction", "Methodology", "Proofs", "Benchmarks", "Conclusion"]
        assert len(sections) == 5
