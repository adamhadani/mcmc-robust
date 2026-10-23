"""Unit and feature tests for baseline samplers (RWM, MALA, HMC)."""

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.core.state import ChainState
from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.hmc import HMC
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.rwm import RandomWalkMetropolis
from mcmc_robust.targets.benign import DiagonalGaussian, IsotropicGaussian
from mcmc_robust.targets.roughness import QuarticTarget


@pytest.mark.unit
def test_rwm_oracle_budget_contract(rng):
    """Test F09.1: RWM strictly evaluates 0 gradients and 1 log_prob per step."""
    target = IsotropicGaussian(d=3)
    tracker = GradBudgetTracker(target)
    sampler = RandomWalkMetropolis(step_size=0.15)

    x0 = np.array([1.0, -1.0, 0.5])
    state = sampler.init_state(x0, tracker)

    # Init contract: 0 grad, 1 log_prob
    assert tracker.n_grad == 0
    assert tracker.n_log_prob == 1
    assert state.grad_log_prob is None
    assert sampler.requires_grad is False

    n_steps = 25
    for _ in range(n_steps):
        prior_grad = tracker.n_grad
        prior_lp = tracker.n_log_prob
        res = sampler.step(state, tracker, rng)

        assert tracker.n_grad == prior_grad, "RWM must never evaluate gradients"
        assert tracker.n_log_prob == prior_lp + 1
        assert res.stats["grad_evals"] == 0
        assert res.stats["log_prob_evals"] == 1

        if not res.accepted:
            assert res.state is state, "On rejection, previous ChainState must be returned"
        else:
            assert res.state is not state

        state = res.state

    assert tracker.n_grad == 0
    assert tracker.n_log_prob == 1 + n_steps


@pytest.mark.unit
def test_mala_oracle_budget_contract(rng):
    """Test F10.1: MALA strictly evaluates 1 gradient and 1 log_prob per step."""
    target = IsotropicGaussian(d=2)
    tracker = GradBudgetTracker(target)
    sampler = MALA(step_size=0.1)

    x0 = np.array([0.5, -0.5])
    state = sampler.init_state(x0, tracker)

    # Init contract: 1 grad, 1 log_prob
    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1
    assert state.grad_log_prob is not None
    assert sampler.requires_grad is True

    n_steps = 30
    for _ in range(n_steps):
        prior_grad = tracker.n_grad
        prior_lp = tracker.n_log_prob
        res = sampler.step(state, tracker, rng)

        assert tracker.n_grad == prior_grad + 1, "MALA must consume strictly 1 grad per step"
        assert tracker.n_log_prob == prior_lp + 1, "MALA must consume strictly 1 log_prob per step"
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
def test_hmc_oracle_budget_contract(rng):
    """Test F11.1: HMC with L leapfrog steps strictly evaluates L gradients per step."""
    target = IsotropicGaussian(d=2)
    tracker = GradBudgetTracker(target)
    L = 6
    sampler = HMC(step_size=0.3, n_leapfrog=L)

    x0 = np.array([0.2, -0.8])
    state = sampler.init_state(x0, tracker)

    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1

    n_steps = 20
    for _ in range(n_steps):
        prior_grad = tracker.n_grad
        prior_lp = tracker.n_log_prob
        res = sampler.step(state, tracker, rng)

        assert tracker.n_grad == prior_grad + L, f"HMC must consume strictly L={L} grads per step"
        assert tracker.n_log_prob == prior_lp + 1, "HMC must consume strictly 1 log_prob per step"
        assert res.stats["grad_evals"] == L
        assert "hamiltonian_error" in res.stats

        if res.accepted:
            assert res.state is not state
            assert np.allclose(res.state.grad_log_prob, target.grad_log_prob(res.state.x), atol=1e-12)
        else:
            assert res.state is state

        state = res.state

    assert tracker.n_grad == 1 + n_steps * L
    assert tracker.n_log_prob == 1 + n_steps


@pytest.mark.unit
def test_hmc_symplectic_energy_scaling():
    """Test F11.2: HMC Hamiltonian error scales as O(epsilon^2) for fixed trajectory time."""
    target = IsotropicGaussian(d=2)
    state0 = ChainState(
        x=np.array([1.0, 1.0]),
        log_prob=target.log_prob(np.array([1.0, 1.0])),
        grad_log_prob=target.grad_log_prob(np.array([1.0, 1.0])),
    )

    s1 = HMC(step_size=0.4, n_leapfrog=10)  # eps = 0.04
    s2 = HMC(step_size=0.4, n_leapfrog=20)  # eps = 0.02

    errs1 = []
    errs2 = []
    rng = np.random.default_rng(999)

    for _ in range(50):
        seed = rng.integers(0, 1000000)
        r1 = s1.step(state0, target, np.random.default_rng(seed))
        r2 = s2.step(state0, target, np.random.default_rng(seed))
        errs1.append(abs(r1.stats["hamiltonian_error"]))
        errs2.append(abs(r2.stats["hamiltonian_error"]))

    ratio = np.mean(errs1) / max(1e-12, np.mean(errs2))
    # Halving step size in a 2nd-order symplectic integrator quarters the error (ratio ~ 4.0)
    assert 2.8 <= ratio <= 5.5, f"Expected second-order scaling ratio ~ 4.0, got {ratio}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "sampler_cls, kwargs",
    [
        (RandomWalkMetropolis, {"step_size": 0.5}),
        (MALA, {"step_size": 0.3}),
        (HMC, {"step_size": 0.5, "n_leapfrog": 5}),
    ],
)
def test_baseline_stationarity_ks_gaussian(sampler_cls, kwargs):
    """Test F09-F11: Stationarity on 1D Gaussian verified by Kolmogorov-Smirnov test (p > 0.01)."""
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
        (RandomWalkMetropolis, {"step_size": 0.3}),
        (MALA, {"step_size": 0.15}),
        (HMC, {"step_size": 0.4, "n_leapfrog": 6}),
    ],
)
def test_baseline_stationarity_ks_quartic(sampler_cls, kwargs):
    """Test F09-F11: Stationarity on 1D Quartic target verified by KS test (p > 0.01)."""
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
def test_dual_averaging_adaptation_warmup():
    """Test F03/F09-F11: Dual Averaging step-size adaptation expands from tiny initial step size."""
    target = IsotropicGaussian(d=2)

    for sampler in [
        RandomWalkMetropolis(step_size=0.01, adapt_step_size=True),
        MALA(step_size=0.01, adapt_step_size=True),
        HMC(step_size=0.01, n_leapfrog=5, adapt_step_size=True),
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
        assert sampler.step_size > h_init, f"Step size failed to expand during warmup for {sampler.__class__.__name__}"


@pytest.mark.unit
def test_preconditioning_scaling():
    """Test F09-F11: Diagonal preconditioning restores high acceptance on anisotropic target."""
    target = DiagonalGaussian(d=2, condition_number=100.0)
    scales = target.stds  # [1.0, 0.1]

    for sampler in [
        RandomWalkMetropolis(step_size=0.1, preconditioner=scales),
        MALA(step_size=0.1, preconditioner=scales),
        HMC(step_size=0.2, n_leapfrog=5, preconditioner=scales),
    ]:
        _, results = sampler.sample(
            x0=np.array([0.0, 0.0]),
            target=target,
            n_steps=500,
            burn_in=50,
            rng=np.random.default_rng(888),
        )
        acc_rate = np.mean([r.accepted for r in results])
        assert acc_rate > 0.3, f"{sampler.__class__.__name__} preconditioned acceptance rate too low ({acc_rate})"


@pytest.mark.unit
def test_numerical_singularity_rejection(rng):
    """Test: When proposals land in non-finite/NaN regions, proposal is rejected safely."""

    class SingularTarget(IsotropicGaussian):
        def log_prob(self, x: np.ndarray) -> float:
            if np.allclose(x, 0.0):
                return super().log_prob(x)
            return -np.inf

        def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
            if np.allclose(x, 0.0):
                return super().grad_log_prob(x)
            return np.full(self.d, np.nan)

        def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
            return self.log_prob(x), self.grad_log_prob(x)

    target = SingularTarget(d=2)

    for sampler in [
        RandomWalkMetropolis(step_size=1.0),
        MALA(step_size=1.0),
        HMC(step_size=1.0, n_leapfrog=5),
    ]:
        state = sampler.init_state(np.array([0.0, 0.0]), target)
        res = sampler.step(state, target, rng)
        # Any step away from origin encounters singular values, must cleanly reject
        assert not res.accepted
        assert res.accept_prob == 0.0
        assert res.state is state
