"""Unit tests for core architecture, state containers, samplers, trackers, and adaptation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.diagnostics.tracker import (
    BudgetExceededError,
    BudgetExhaustedError,
    GradBudgetTracker,
)
from mcmc_robust.targets.benign import IsotropicGaussian


# Mock Sampler for testing zero-redundancy caching protocol
class MockCachingSampler(Sampler):
    """Deterministic mock sampler verifying zero-redundancy oracle caching."""

    def __init__(self, accept_sequence: list[bool]) -> None:
        super().__init__(step_size=0.1)
        self.accept_sequence = accept_sequence
        self.step_idx = 0

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        lp, grad = target.log_prob_and_grad(x0)
        return ChainState(x=x0.copy(), log_prob=lp, grad_log_prob=grad)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        should_accept = self.accept_sequence[self.step_idx % len(self.accept_sequence)]
        self.step_idx += 1
        # Proposal generated using state.grad_log_prob without target re-eval
        assert state.grad_log_prob is not None
        y = state.x + 0.05 * state.grad_log_prob

        if should_accept:
            # On accept: evaluate candidate gradient and log_prob exactly once
            lp_y, grad_y = target.log_prob_and_grad(y)
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=grad_y)
            return StepResult(
                state=new_state,
                accepted=True,
                accept_prob=1.0,
                stats={"step": self.step_idx},
                proposal=y,
                step_size=self.step_size,
            )
        else:
            # On reject: return previous state with ZERO target evaluations
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=0.0,
                stats={"step": self.step_idx},
                proposal=y,
                step_size=self.step_size,
            )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        pass


@pytest.mark.unit
def test_chain_state_immutability():
    """Test 1: ChainState is frozen and underlying numpy arrays are write-protected."""
    x = np.array([1.0, 2.0], dtype=np.float64)
    g = np.array([-0.5, -1.0], dtype=np.float64)
    state = ChainState(x=x, log_prob=-2.5, grad_log_prob=g, auxiliary={"tag": "init"})

    # Dataclass attribute immutability
    with pytest.raises(FrozenInstanceError):
        state.log_prob = 0.0  # type: ignore

    with pytest.raises(FrozenInstanceError):
        state.x = np.array([3.0, 4.0])  # type: ignore

    # Underlying array buffer immutability
    with pytest.raises(ValueError, match="read-only"):
        state.x[0] = 999.0

    assert state.grad_log_prob is not None
    with pytest.raises(ValueError, match="read-only"):
        state.grad_log_prob[0] = 999.0

    # Test copy_with
    new_state = state.copy_with(log_prob=-1.0)
    assert new_state.log_prob == -1.0
    assert state.log_prob == -2.5
    assert np.array_equal(new_state.x, state.x)
    assert new_state.aux["tag"] == "init"


@pytest.mark.unit
def test_step_result_dataclass_contract():
    """Test 2: StepResult dataclass fields, immutability, and alias properties."""
    state = ChainState(x=np.array([1.0]), log_prob=-0.5, grad_log_prob=np.array([-1.0]))
    res = StepResult(
        state=state,
        accepted=True,
        accept_prob=0.85,
        stats={"energy": 0.5},
        proposal=np.array([1.1]),
        step_size=0.05,
    )

    with pytest.raises(FrozenInstanceError):
        res.accepted = False  # type: ignore

    assert res.accepted is True
    assert res.accept_prob == 0.85
    assert res.acceptance_prob == 0.85
    assert res.diagnostics["energy"] == 0.5
    assert res.step_size == 0.05


@pytest.mark.unit
def test_target_distribution_abstract_enforcement():
    """Test 3: TargetDistribution abstract base class enforcement and validation."""
    # Direct instantiation must fail
    with pytest.raises(TypeError):
        TargetDistribution(d=2)  # type: ignore

    # Subclass missing grad_log_prob must fail
    class IncompleteTarget(TargetDistribution):
        def log_prob(self, x: np.ndarray) -> float:
            return 0.0

        def sample_ground_truth(self, n: int, rng=None) -> np.ndarray:
            return np.zeros((n, self.d))

        def exact_moments_or_quantiles(self) -> dict:
            return {}

    with pytest.raises(TypeError):
        IncompleteTarget(d=2)  # type: ignore

    # Invalid dimension validation
    class ValidTarget(TargetDistribution):
        def log_prob(self, x: np.ndarray) -> float:
            return 0.0

        def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(self.d)

        def sample_ground_truth(self, n: int, rng=None) -> np.ndarray:
            return np.zeros((n, self.d))

        def exact_moments_or_quantiles(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="Dimension d must be a positive integer"):
        ValidTarget(d=0)


@pytest.mark.unit
def test_grad_budget_tracker_call_interception():
    """Test 4: GradBudgetTracker strictly increments counters on every oracle call."""
    base_target = IsotropicGaussian(d=2, std=1.0)
    tracker = GradBudgetTracker(base_target)

    x = np.array([0.5, -0.5])
    assert tracker.n_log_prob == 0
    assert tracker.n_grad == 0
    assert tracker.n_joint == 0

    # 1. log_prob call
    _ = tracker.log_prob(x)
    assert tracker.n_log_prob == 1
    assert tracker.log_prob_evals == 1
    assert tracker.n_grad == 0

    # 2. grad_log_prob call
    _ = tracker.grad_log_prob(x)
    assert tracker.n_log_prob == 1
    assert tracker.n_grad == 1
    assert tracker.grad_evals == 1

    # 3. joint log_prob_and_grad call
    _ = tracker.log_prob_and_grad(x)
    assert tracker.n_log_prob == 2
    assert tracker.n_grad == 2
    assert tracker.n_joint == 1
    assert tracker.total_evals == 4

    # 4. Non-evaluating methods
    _ = tracker.sample_ground_truth(5)
    _ = tracker.exact_moments_or_quantiles()
    assert tracker.n_log_prob == 2
    assert tracker.n_grad == 2


@pytest.mark.unit
def test_grad_budget_tracker_budget_exhaustion():
    """Test 5: Exceeding configured budget raises BudgetExceededError immediately."""
    base_target = IsotropicGaussian(d=2, std=1.0)
    tracker = GradBudgetTracker(base_target, max_grad_evals=5)

    x = np.array([0.1, 0.2])
    for _ in range(5):
        _ = tracker.grad_log_prob(x)

    assert tracker.n_grad == 5
    assert tracker.is_exhausted is True

    # 6th evaluation must raise BudgetExceededError (or BudgetExhaustedError)
    with pytest.raises((BudgetExceededError, BudgetExhaustedError)):
        _ = tracker.grad_log_prob(x)

    # Counter should remain 5
    assert tracker.n_grad == 5


@pytest.mark.unit
def test_grad_budget_tracker_numerical_transparency():
    """Test 6: Tracker outputs match underlying target exactly without distortion."""
    base_target = IsotropicGaussian(d=2, mean=np.array([1.0, -1.0]), std=2.0)
    tracker = GradBudgetTracker(base_target)

    x = np.array([3.0, 1.0])
    raw_lp = base_target.log_prob(x)
    raw_grad = base_target.grad_log_prob(x)

    tracked_lp = tracker.log_prob(x)
    tracked_grad = tracker.grad_log_prob(x)

    assert tracked_lp == raw_lp
    assert np.allclose(tracked_grad, raw_grad, atol=1e-15)

    joint_lp, joint_grad = tracker.log_prob_and_grad(x)
    assert joint_lp == raw_lp
    assert np.allclose(joint_grad, raw_grad, atol=1e-15)


@pytest.mark.unit
def test_zero_redundancy_caching_mock_sampler(rng):
    """Test 7: Zero-redundancy oracle caching protocol under accept/reject sequence."""
    target = IsotropicGaussian(d=2)
    tracker = GradBudgetTracker(target)

    # 8-step sequence with 3 accepts and 5 rejects
    accept_sequence = [True, False, False, True, False, False, False, True]
    sampler = MockCachingSampler(accept_sequence=accept_sequence)

    x0 = np.array([1.0, 1.0])
    state = sampler.init_state(x0, tracker)

    # Initialization: exactly 1 grad, 1 log_prob
    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1

    for step_idx in range(len(accept_sequence)):
        prior_grads = tracker.n_grad
        prior_lps = tracker.n_log_prob

        res = sampler.step(state, tracker, rng)
        expected_accept = accept_sequence[step_idx]

        assert res.accepted == expected_accept

        if expected_accept:
            # Accepted: exactly 1 new grad and 1 new log_prob
            assert tracker.n_grad == prior_grads + 1
            assert tracker.n_log_prob == prior_lps + 1
            assert res.state is not state
        else:
            # Rejected: exactly ZERO new evals!
            assert tracker.n_grad == prior_grads
            assert tracker.n_log_prob == prior_lps
            assert res.state is state

        state = res.state

    # Total: 1 (init) + 3 (accepts) = 4 evals total!
    assert tracker.n_grad == 4
    assert tracker.n_log_prob == 4


@pytest.mark.unit
def test_sampler_base_interface_contract():
    """Test 8: Sampler abstract base class interface contracts and preconditioner."""

    # Incomplete sampler missing step
    class IncompleteSampler(Sampler):
        def init_state(self, x0, target):
            return super().init_state(x0, target)

        def adapt(self, step_idx, result):
            pass

    with pytest.raises(TypeError):
        IncompleteSampler()  # type: ignore

    # Concrete mock sampler
    sampler = MockCachingSampler([True])
    assert sampler.requires_grad is True

    # Preconditioner validation
    with pytest.raises(ValueError, match="strictly positive"):
        sampler.set_preconditioner(np.array([1.0, -0.5]))

    sampler.set_preconditioner(np.array([2.0, 3.0]))
    assert sampler.preconditioner is not None
    assert np.allclose(sampler.preconditioner, np.array([2.0, 3.0]))


@pytest.mark.unit
def test_dual_averaging_adaptation_updates():
    """Test 9: DualAveragingAdaptation dynamic step size expansion and contraction."""
    adapt = DualAveragingAdaptation(initial_step_size=1.0, target_accept=0.574)
    h0 = adapt.step_size

    # High acceptance rate (1.0 > 0.574) must expand step size
    for _ in range(50):
        adapt.update(1.0)
    assert adapt.step_size > h0
    assert adapt.final_step_size > h0

    # Reset
    adapt.reset(initial_step_size=1.0)
    assert adapt.step_size == 1.0

    # Low acceptance rate (0.0 < 0.574) must shrink step size
    for _ in range(50):
        adapt.update(0.0)
    assert adapt.step_size < 1.0
    assert adapt.final_step_size < 1.0


@pytest.mark.unit
def test_welford_preconditioner_streaming_variance(rng):
    """Test 10: WelfordPreconditioner online variance matches batch sample variance."""
    d = 4
    true_stds = np.array([0.5, 1.0, 2.0, 4.0])
    welford = WelfordPreconditioner(d=d, shrinkage_samples=0.0)

    n_samples = 5000
    samples = rng.normal(size=(n_samples, d)) * true_stds
    welford.update_batch(samples)

    batch_var = np.var(samples, axis=0, ddof=1)
    # Online Welford variance matches batch sample variance to high precision
    assert np.allclose(welford.sample_variance, batch_var, rtol=1e-6)

    # Matches true diagonal variance within sampling error (< 10%)
    assert np.allclose(welford.sample_variance, true_stds**2, rtol=0.10)


@pytest.mark.unit
def test_welford_preconditioner_clipping():
    """Test 11: WelfordPreconditioner variance clipping bounds."""
    d = 2
    welford = WelfordPreconditioner(
        d=d,
        shrinkage_samples=0.0,
        min_variance=0.1,
        max_variance=5.0,
    )

    # Feed nearly constant data (var ~ 0)
    for _ in range(100):
        welford.update(np.array([1.0, 1.0]))

    # Must be clamped at min_variance = 0.1
    diag = welford.diagonal_preconditioner()
    assert np.all(diag >= 0.1)

    # Feed extreme variance data
    for _ in range(50):
        welford.update(np.array([1000.0, -1000.0]))
        welford.update(np.array([-1000.0, 1000.0]))

    # Must be clamped at max_variance = 5.0
    diag = welford.diagonal_preconditioner()
    assert np.all(diag <= 5.0)


@pytest.mark.unit
def test_chain_state_defensive_copy():
    """Test 12: ChainState makes defensive copies of x and grad_log_prob preventing caller mutation."""
    orig_x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    orig_g = np.array([0.1, -0.2, 0.3], dtype=np.float64)

    state = ChainState(x=orig_x, log_prob=-2.5, grad_log_prob=orig_g)

    # State arrays must be read-only
    assert state.x.flags.writeable is False
    assert state.grad_log_prob is not None
    assert state.grad_log_prob.flags.writeable is False

    # Modifying caller arrays must NOT affect ChainState
    orig_x[0] = 999.0
    orig_g[0] = -888.0

    assert state.x[0] == 1.0
    assert state.grad_log_prob[0] == 0.1


@pytest.mark.unit
def test_dual_averaging_nan_guard():
    """Test 13: DualAveragingAdaptation handles non-finite / NaN acceptance probabilities gracefully."""
    da_nan = DualAveragingAdaptation(initial_step_size=0.1, target_accept=0.574)
    da_zero = DualAveragingAdaptation(initial_step_size=0.1, target_accept=0.574)

    # Non-finite values (NaN, +inf, -inf) must be treated identically to 0.0 acceptance
    da_nan.update(np.nan)
    da_zero.update(0.0)

    assert np.isclose(da_nan.step_size, da_zero.step_size)
    assert np.isclose(da_nan.final_step_size, da_zero.final_step_size)
    assert np.isfinite(da_nan.step_size)
    assert np.isfinite(da_nan.final_step_size)

    # +inf and -inf also treated safely as 0.0
    da_nan.update(np.inf)
    assert np.isfinite(da_nan.step_size)

    da_nan.update(-np.inf)
    assert np.isfinite(da_nan.step_size)

    # Over 50 NaN updates, step size shrinks strictly below initial step size
    for _ in range(50):
        da_nan.update(np.nan)
    assert da_nan.step_size < 0.1


@pytest.mark.unit
def test_welford_parameter_validation():
    """Test 14: WelfordPreconditioner validates variance and shrinkage parameters."""
    # min_variance must be positive
    with pytest.raises(ValueError, match="min_variance must be positive"):
        WelfordPreconditioner(d=2, min_variance=0.0)

    with pytest.raises(ValueError, match="min_variance must be positive"):
        WelfordPreconditioner(d=2, min_variance=-1.0)

    # prior_variance must be positive
    with pytest.raises(ValueError, match="prior_variance must be positive"):
        WelfordPreconditioner(d=2, prior_variance=0.0)

    with pytest.raises(ValueError, match="prior_variance must be positive"):
        WelfordPreconditioner(d=2, prior_variance=-0.5)

    # min_variance must be strictly less than max_variance
    with pytest.raises(ValueError, match="strictly less than max_variance"):
        WelfordPreconditioner(d=2, min_variance=10.0, max_variance=5.0)

    with pytest.raises(ValueError, match="strictly less than max_variance"):
        WelfordPreconditioner(d=2, min_variance=5.0, max_variance=5.0)

    # shrinkage_samples must be non-negative
    with pytest.raises(ValueError, match="shrinkage_samples must be non-negative"):
        WelfordPreconditioner(d=2, shrinkage_samples=-1.0)


@pytest.mark.unit
def test_grad_budget_tracker_immutable_setters():
    """Test 15: GradBudgetTracker public setters for counters are removed and immutable."""
    target = IsotropicGaussian(d=2)
    tracker = GradBudgetTracker(target=target)

    assert tracker.grad_evals == 0
    assert tracker.log_prob_evals == 0

    with pytest.raises(AttributeError):
        tracker.grad_evals = 10

    with pytest.raises(AttributeError):
        tracker.log_prob_evals = 10

    # Read-only counters remain intact
    assert tracker.grad_evals == 0
    assert tracker.log_prob_evals == 0
