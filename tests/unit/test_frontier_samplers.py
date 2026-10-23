"""Unit tests for frontier robust samplers: GloballyTamedMALA and BarkerLangevin."""

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.barker import BarkerLangevin
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.tamed_mala import GloballyTamedMALA
from mcmc_robust.targets.benign import DiagonalGaussian, IsotropicGaussian
from mcmc_robust.targets.roughness import QuarticTarget


@pytest.mark.unit
def test_tamed_mala_drift_capping():
    """Test F12.1: Tamed drift is strictly bounded by 1/gamma for both global and coordinate modes."""
    gamma = 0.2
    bound = 1.0 / gamma

    sampler_global = GloballyTamedMALA(gamma=gamma, mode="global")
    sampler_coord = GloballyTamedMALA(gamma=gamma, mode="coordinate")

    for grad_norm in [1e-4, 1.0, 10.0, 1e3, 1e8, 1e16]:
        grad = np.array([grad_norm, -grad_norm * 0.5, grad_norm * 0.2])

        # Global mode: ||b(x)|| < 1/gamma
        b_global = sampler_global._tamed_drift(grad)
        assert np.linalg.norm(b_global) < bound

        # Coordinate mode: |b_i(x)| < 1/gamma for each i
        b_coord = sampler_coord._tamed_drift(grad)
        assert np.all(np.abs(b_coord) < bound)


@pytest.mark.unit
def test_tamed_mala_bulk_consistency():
    """Test F12.2: Tamed drift matches raw gradient in bulk when ||grad|| << 1/gamma."""
    gamma = 0.1
    sampler = GloballyTamedMALA(gamma=gamma, mode="global")

    # Small gradient
    grad = np.array([1e-5, -2e-5, 3e-5])
    b = sampler._tamed_drift(grad)
    diff = np.linalg.norm(b - grad)
    # Order of difference is O(gamma * ||grad||^2)
    assert diff < gamma * (np.linalg.norm(grad) ** 2) * 1.5


@pytest.mark.unit
def test_tamed_mala_oracle_budget_contract(rng):
    """Test F12.3: GloballyTamedMALA strictly evaluates 1 gradient and 1 log_prob per step."""
    target = IsotropicGaussian(d=3)
    tracker = GradBudgetTracker(target)
    sampler = GloballyTamedMALA(step_size=0.1, gamma=0.1)

    x0 = np.array([0.5, -0.5, 1.0])
    state = sampler.init_state(x0, tracker)

    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1

    n_steps = 30
    for _ in range(n_steps):
        prior_grad = tracker.n_grad
        prior_lp = tracker.n_log_prob
        res = sampler.step(state, tracker, rng)

        assert tracker.n_grad == prior_grad + 1, "Tamed MALA must consume strictly 1 grad per step"
        assert tracker.n_log_prob == prior_lp + 1, "Tamed MALA must consume strictly 1 log_prob per step"
        assert res.stats["grad_evals"] == 1

        if res.accepted:
            assert res.state is not state
            assert np.allclose(res.state.grad_log_prob, target.grad_log_prob(res.state.x), atol=1e-12)
        else:
            assert res.state is state

        state = res.state

    assert tracker.n_grad == 1 + n_steps
    assert tracker.n_log_prob == 1 + n_steps


@pytest.mark.unit
def test_barker_displacement_boundedness(rng):
    """Test F13.1: Barker displacement |y_i - x_i| is bounded by sqrt(h) s_i |sigma_i| regardless of grad."""
    h = 0.2
    sampler = BarkerLangevin(step_size=h)

    # Mock target with massive gradients
    class ExplosiveGradTarget(IsotropicGaussian):
        def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
            return np.array([1e15, -1e15])

        def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
            return self.log_prob(x), self.grad_log_prob(x)

    target = ExplosiveGradTarget(d=2)
    state = sampler.init_state(np.array([0.0, 0.0]), target)

    for _ in range(20):
        res = sampler.step(state, target, rng)
        # Check proposal displacement norm: for h=0.2, displacement is bounded by Gaussian draws
        disp = res.proposal - state.x
        assert np.all(np.isfinite(disp))
        # Even with gradient 1e15, displacement magnitude is strictly finite and not 1e15
        assert np.linalg.norm(disp) < 15.0 * np.sqrt(h)


@pytest.mark.unit
def test_barker_numerical_stability_extreme_gradients(rng):
    """Test F13.2: Barker log transition density difference avoids overflow via softplus."""
    sampler = BarkerLangevin(step_size=0.1)

    class ExtremeTarget(IsotropicGaussian):
        def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
            return np.array([1e12, -1e12])

        def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
            return self.log_prob(x), self.grad_log_prob(x)

    target = ExtremeTarget(d=2)
    state = sampler.init_state(np.array([0.0, 0.0]), target)
    res = sampler.step(state, target, rng)

    assert np.isfinite(res.accept_prob)
    assert 0.0 <= res.accept_prob <= 1.0


@pytest.mark.unit
def test_barker_oracle_budget_contract(rng):
    """Test F13.3: BarkerLangevin strictly evaluates 1 gradient and 1 log_prob per step."""
    target = IsotropicGaussian(d=3)
    tracker = GradBudgetTracker(target)
    sampler = BarkerLangevin(step_size=0.1)

    x0 = np.array([0.5, -0.5, 1.0])
    state = sampler.init_state(x0, tracker)

    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1

    n_steps = 30
    for _ in range(n_steps):
        prior_grad = tracker.n_grad
        prior_lp = tracker.n_log_prob
        res = sampler.step(state, tracker, rng)

        assert tracker.n_grad == prior_grad + 1, "Barker must consume strictly 1 grad per step"
        assert tracker.n_log_prob == prior_lp + 1, "Barker must consume strictly 1 log_prob per step"
        assert res.stats["grad_evals"] == 1

        if res.accepted:
            assert res.state is not state
            assert np.allclose(res.state.grad_log_prob, target.grad_log_prob(res.state.x), atol=1e-12)
        else:
            assert res.state is state

        state = res.state

    assert tracker.n_grad == 1 + n_steps
    assert tracker.n_log_prob == 1 + n_steps


@pytest.mark.unit
@pytest.mark.parametrize(
    "sampler_cls, kwargs",
    [
        (GloballyTamedMALA, {"step_size": 0.3, "gamma": 0.1}),
        (BarkerLangevin, {"step_size": 0.3}),
    ],
)
def test_frontier_stationarity_ks_gaussian(sampler_cls, kwargs):
    """Test F12-F13: Stationarity on 1D Gaussian verified by KS test (p > 0.01)."""
    sampler = sampler_cls(**kwargs)
    target = IsotropicGaussian(d=1, std=1.5)
    exact_samples = target.sample_ground_truth(10000, np.random.default_rng(1234)).flatten()

    samples, _ = sampler.sample(
        x0=np.array([0.0]),
        target=target,
        n_steps=12000,
        burn_in=2000,
        rng=np.random.default_rng(42),
    )
    thinned = samples.flatten()[::5]
    res = stats.ks_2samp(thinned, exact_samples)
    assert res.pvalue > 0.01, f"{sampler_cls.__name__} failed KS stationarity test (p={res.pvalue})"


@pytest.mark.unit
@pytest.mark.parametrize(
    "sampler_cls, kwargs",
    [
        (GloballyTamedMALA, {"step_size": 0.2, "gamma": 0.1}),
        (BarkerLangevin, {"step_size": 0.2}),
    ],
)
def test_frontier_stationarity_ks_quartic(sampler_cls, kwargs):
    """Test F12-F13: Stationarity on 1D Quartic target verified by KS test (p > 0.01)."""
    sampler = sampler_cls(**kwargs)
    target = QuarticTarget(d=1)
    exact_samples = target.sample_ground_truth(10000, np.random.default_rng(5678)).flatten()

    samples, _ = sampler.sample(
        x0=np.array([0.0]),
        target=target,
        n_steps=12000,
        burn_in=2000,
        rng=np.random.default_rng(101),
    )
    thinned = samples.flatten()[::5]
    res = stats.ks_2samp(thinned, exact_samples)
    assert res.pvalue > 0.01, f"{sampler_cls.__name__} failed Quartic KS test (p={res.pvalue})"


@pytest.mark.unit
def test_frontier_dual_averaging_adaptation():
    """Test F12-F13: Dual Averaging step-size adaptation expands from tiny initial step size."""
    target = IsotropicGaussian(d=2)

    for sampler in [
        GloballyTamedMALA(step_size=0.01, adapt_step_size=True),
        BarkerLangevin(step_size=0.01, adapt_step_size=True),
    ]:
        h_init = sampler.step_size
        _, _ = sampler.sample(
            x0=np.array([0.0, 0.0]),
            target=target,
            n_steps=200,
            burn_in=100,
            rng=np.random.default_rng(777),
            adapt=True,
        )
        assert sampler.step_size > h_init, f"Step size failed to expand for {sampler.__class__.__name__}"


@pytest.mark.unit
def test_frontier_preconditioning():
    """Test F12-F13: Diagonal preconditioning restores high acceptance on anisotropic target."""
    target = DiagonalGaussian(d=2, condition_number=100.0)
    scales = target.stds  # [1.0, 0.1]

    for sampler in [
        GloballyTamedMALA(step_size=0.1, preconditioner=scales),
        BarkerLangevin(step_size=0.1, preconditioner=scales),
    ]:
        _, results = sampler.sample(
            x0=np.array([0.0, 0.0]),
            target=target,
            n_steps=500,
            burn_in=50,
            rng=np.random.default_rng(888),
        )
        acc_rate = np.mean([r.accepted for r in results])
        assert acc_rate > 0.3, f"{sampler.__class__.__name__} preconditioned acc rate too low ({acc_rate})"


@pytest.mark.unit
def test_quartic_stability_frontier_vs_mala_freezing():
    """Test F12/F13 vs F10: On Quartic tail ||x0|| = 14.14, MALA freezes while Tamed MALA and Barker move."""
    target = QuarticTarget(d=2)
    x0 = np.array([10.0, 10.0])  # ||x0|| = 14.14, extreme tail
    h = 0.1

    mala = MALA(step_size=h)
    tamed = GloballyTamedMALA(step_size=h, gamma=0.2)
    barker = BarkerLangevin(step_size=h)

    n_steps = 100

    # 1. Standard MALA catastrophically freezes
    _, mala_results = mala.sample(x0=x0, target=target, n_steps=n_steps, rng=np.random.default_rng(123))
    mala_acc = np.mean([r.accepted for r in mala_results])
    assert mala_acc == 0.0, f"Expected standard MALA to freeze (0% acceptance), got {mala_acc}"

    # 2. GloballyTamedMALA remains active and moves toward bulk
    tamed_samples, tamed_results = tamed.sample(x0=x0, target=target, n_steps=n_steps, rng=np.random.default_rng(123))
    tamed_acc = np.mean([r.accepted for r in tamed_results])
    assert tamed_acc > 0.5, f"Expected Tamed MALA acceptance > 0.5, got {tamed_acc}"
    # Final position must contract toward bulk (||x|| < ||x0||)
    final_norm_tamed = np.linalg.norm(tamed_samples[-1])
    assert final_norm_tamed < np.linalg.norm(x0) * 0.5, f"Tamed MALA failed to contract: {final_norm_tamed}"

    # 3. BarkerLangevin remains active and moves toward bulk
    barker_samples, barker_results = barker.sample(
        x0=x0, target=target, n_steps=n_steps, rng=np.random.default_rng(123)
    )
    barker_acc = np.mean([r.accepted for r in barker_results])
    assert barker_acc > 0.5, f"Expected Barker acceptance > 0.5, got {barker_acc}"
    final_norm_barker = np.linalg.norm(barker_samples[-1])
    assert final_norm_barker < np.linalg.norm(x0) * 0.5, f"Barker failed to contract: {final_norm_barker}"
