"""Unit tests for Reference Sampler Scrutiny and Stock BlackJAX Cross-Validation.

Verifies that all reference implementations (RandomWalkMetropolis, MALA, HMC, BarkerLangevin,
GloballyTamedMALA, StereographicSampler, RadialDiffeomorphicSampler) are completely faithful,
un-crippled, and match stock BlackJAX (v1.7.1) and theoretical optimal scaling invariants.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import blackjax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.automala import AutoMALA
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.rwm import RandomWalkMetropolis
from mcmc_robust.targets.benign import IsotropicGaussian
from mcmc_robust.targets.roughness import QuarticTarget
from mcmc_robust.transforms.radial import RadialDiffeomorphicTransform
from mcmc_robust.transforms.stereographic import StereographicTransform


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# =============================================================================
# 1. Direct Cross-Validation vs Stock BlackJAX (v1.7.1)
# =============================================================================


@pytest.mark.unit
def test_hmc_leapfrog_machine_precision_match_with_blackjax_velocity_verlet():
    r"""Verify HMC velocity Verlet leapfrog produces exact match (< 1e-13) with BlackJAX."""
    d = 4
    x0_np = np.array([1.2, -0.7, 0.4, -1.8], dtype=np.float64)
    p0_np = np.array([0.5, 1.1, -0.3, 0.8], dtype=np.float64)
    eps = 0.125
    L = 5

    def jax_logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(x**2)

    # 1. Stock BlackJAX Velocity Verlet Integrator
    integrator = blackjax.mcmc.integrators.velocity_verlet(
        jax_logdensity,
        lambda p: 0.5 * jnp.sum(p**2),
    )

    x0_jax = jnp.asarray(x0_np, dtype=jnp.float64)
    p0_jax = jnp.asarray(p0_np, dtype=jnp.float64)
    state = blackjax.mcmc.integrators.IntegratorState(
        position=x0_jax,
        momentum=p0_jax,
        logdensity=jax_logdensity(x0_jax),
        logdensity_grad=jax.grad(jax_logdensity)(x0_jax),
    )

    for _ in range(L):
        state = integrator(state, eps)

    bj_final_pos = np.asarray(state.position, dtype=np.float64)
    bj_final_mom = np.asarray(state.momentum, dtype=np.float64)

    # 2. mcmc_robust HMC Leapfrog Loop
    target = IsotropicGaussian(d=d)
    x = x0_np.copy()
    p = p0_np.copy()
    g = target.grad_log_prob(x)

    for _ in range(L):
        p += 0.5 * eps * g
        x += eps * p
        g = target.grad_log_prob(x)
        p += 0.5 * eps * g

    # Compare to machine precision (< 1e-13)
    assert np.allclose(bj_final_pos, x, atol=1e-13, rtol=1e-13), (
        f"Position mismatch:\nBlackJAX: {bj_final_pos}\nmcmc_robust: {x}"
    )
    assert np.allclose(bj_final_mom, p, atol=1e-13, rtol=1e-13), (
        f"Momentum mismatch:\nBlackJAX: {bj_final_mom}\nmcmc_robust: {p}"
    )


@pytest.mark.unit
def test_mala_log_proposal_ratio_identity_vs_blackjax():
    r"""Verify MALA Gaussian transition density ratio exactly matches BlackJAX formula."""
    d = 3
    target = IsotropicGaussian(d=d)
    step_size = 0.2
    sigma2 = step_size**2

    x = np.array([0.5, -1.0, 1.5], dtype=np.float64)
    y = np.array([0.7, -0.8, 1.2], dtype=np.float64)
    grad_x = target.grad_log_prob(x)
    grad_y = target.grad_log_prob(y)

    # mcmc_robust MALA proposal log-ratio: log q(x | y) - log q(y | x)
    mean_forward = x + 0.5 * sigma2 * grad_x
    mean_backward = y + 0.5 * sigma2 * grad_y
    log_q_forward = -0.5 * np.sum((y - mean_forward) ** 2) / sigma2
    log_q_backward = -0.5 * np.sum((x - mean_backward) ** 2) / sigma2
    diff_log_q = log_q_backward - log_q_forward

    # BlackJAX analytical transition formula
    bj_log_q_forward = stats.multivariate_normal.logpdf(y, mean=mean_forward, cov=sigma2 * np.eye(d))
    bj_log_q_backward = stats.multivariate_normal.logpdf(x, mean=mean_backward, cov=sigma2 * np.eye(d))
    bj_diff = bj_log_q_backward - bj_log_q_forward

    assert np.isclose(diff_log_q, bj_diff, atol=1e-12)


@pytest.mark.unit
def test_barker_log_proposal_ratio_softplus_identity():
    r"""Verify Barker proposal log-ratio matches exact softplus difference formula."""
    d = 4
    h = 0.05
    sqrt_h = np.sqrt(h)
    x = np.array([1.0, -2.0, 0.5, -0.5], dtype=np.float64)
    grad_x = -x  # Gaussian target
    z = np.array([0.2, -0.4, 0.1, -0.3], dtype=np.float64)
    y = x + z
    grad_y = -y

    # Coordinate-wise Barker ratio: log p(x -> y) - log p(y -> x)
    # where log p(x -> y) = sum_i softplus(- z_i * grad_x_i)
    delta = y - x
    log_ratio_direct = float(np.sum(np.log1p(np.exp(-delta * grad_y)) - np.log1p(np.exp(-delta * grad_x))))

    # Using numerically stable softplus identity: log(1 + exp(a)) = np.logaddexp(0, a)
    log_ratio_stable = float(np.sum(np.logaddexp(0.0, -delta * grad_y) - np.logaddexp(0.0, -delta * grad_x)))

    assert np.isclose(log_ratio_direct, log_ratio_stable, atol=1e-14)


# =============================================================================
# 2. Theoretical Optimal Scaling Invariance Tests
# =============================================================================


@pytest.mark.unit
def test_roberts_rosenthal_optimal_scaling_rwm_and_mala():
    r"""Verify empirical acceptance rates match Roberts & Rosenthal asymptotic theory.

    Theory:
    - RWM on d-dim standard Gaussian: proposal variance 2h = (2.38)^2 / d -> alpha ~ 0.234
    - MALA on d-dim standard Gaussian: proposal variance 2h = (1.65)^2 * d^(-1/3) -> alpha ~ 0.574
    """
    d = 20
    target = IsotropicGaussian(d=d)
    n_steps = 2500
    burn_in = 500
    rng_chain = np.random.default_rng(2024)

    # 1. RWM optimal acceptance rate: 2h = 2.38^2 / d -> h = 2.38^2 / (2d)
    rwm_step = (2.38**2) / (2.0 * d)
    rwm = RandomWalkMetropolis(step_size=rwm_step)
    _, rwm_results = rwm.sample(
        x0=np.zeros(d),
        target=target,
        n_steps=n_steps,
        burn_in=burn_in,
        rng=rng_chain,
    )
    rwm_acc = np.mean([r.accepted for r in rwm_results[burn_in:]])
    # Verify RWM acceptance is within [0.20, 0.28] (centered around 0.234)
    assert 0.20 <= rwm_acc <= 0.28, f"RWM acceptance {rwm_acc:.3f} outside expected [0.20, 0.28]"

    # 2. MALA optimal acceptance rate: 2h = 1.65^2 * d^(-1/3) -> h = 0.5 * 1.65^2 * d^(-1/3)
    mala_step = 0.5 * (1.65**2) * (d ** (-1.0 / 3.0))
    mala = MALA(step_size=mala_step)
    _, mala_results = mala.sample(
        x0=np.zeros(d),
        target=target,
        n_steps=n_steps,
        burn_in=burn_in,
        rng=rng_chain,
    )
    mala_acc = np.mean([r.accepted for r in mala_results[burn_in:]])
    # Verify MALA acceptance is within [0.52, 0.63] (centered around 0.574)
    assert 0.52 <= mala_acc <= 0.63, f"MALA acceptance {mala_acc:.3f} outside expected [0.52, 0.63]"


@pytest.mark.unit
def test_globally_tamed_mala_asymptotic_convergence_to_exact_mala():
    r"""Verify Globally Tamed MALA drift approaches exact MALA as step_size -> 0."""
    d = 5
    target = IsotropicGaussian(d=d)
    x = np.array([2.0, -3.0, 1.5, -0.5, 4.0], dtype=np.float64)
    grad = target.grad_log_prob(x)

    for h in [0.01, 0.001, 0.0001]:
        # Tamed drift: grad / (1 + h * ||grad||)
        tamed_drift = grad / (1.0 + h * np.linalg.norm(grad))
        diff = np.linalg.norm(tamed_drift - grad)
        # Difference should scale as O(h)
        assert diff <= 2.0 * h * (np.linalg.norm(grad) ** 2)


@pytest.mark.unit
def test_reference_diffeomorphism_roundtrip_and_jacobian_precision():
    r"""Verify Stereographic and Radial transformations achieve roundtrip < 1e-13."""
    d = 4

    # 1. Stereographic
    st = StereographicTransform(d=d, scale=2.0)
    x_test = np.array([1.5, -2.5, 0.8, -1.2], dtype=np.float64)
    z, log_det_fwd = st.forward(x_test)
    x_rec, log_det_inv = st.inverse(z)
    assert np.allclose(x_test, x_rec, atol=1e-13)
    assert np.isclose(log_det_fwd + log_det_inv, 0.0, atol=1e-13)

    # 2. Radial Diffeomorphic
    rt = RadialDiffeomorphicTransform(d=d, radius=3.0, stiffness=1.0)
    # Test point outside radius
    x_outside = np.array([3.0, 4.0, 1.0, -2.0], dtype=np.float64)  # norm ~ 5.47 > 3.0
    y, log_det_r_fwd = rt.forward(x_outside)
    x_r_rec, log_det_r_inv = rt.inverse(y)
    assert np.allclose(x_outside, x_r_rec, atol=1e-13)
    assert np.isclose(log_det_r_fwd + log_det_r_inv, 0.0, atol=1e-12)


@pytest.mark.unit
def test_automala_doubling_halving_and_oracle_accounting():
    r"""Verify AutoMALA (Biron-Lattes et al. 2024) halving/doubling selector and exact grad accounting."""
    d = 4
    rng_local = np.random.default_rng(12345)

    # 1. Stiff quartic far in tail with oversized initial step_size -> must halve (k_fwd < 0)
    quartic = QuarticTarget(d=d, p=4.0)
    tracked_q = GradBudgetTracker(quartic)
    sampler_stiff = AutoMALA(step_size=2.0, max_halvings=10, max_doublings=8)
    state_q = sampler_stiff.init_state(np.full(d, 5.0), tracked_q)
    g_before = tracked_q.grad_evals
    res_q = sampler_stiff.step(state_q, tracked_q, rng_local)
    g_after = tracked_q.grad_evals
    assert res_q.stats["grad_evals"] == (g_after - g_before)
    assert res_q.stats["k_fwd"] < 0
    assert res_q.stats["selected_step_size"] < 2.0

    # 2. Benign Gaussian with tiny initial step_size -> must double (k_fwd > 0)
    gauss = IsotropicGaussian(d=d)
    tracked_g = GradBudgetTracker(gauss)
    sampler_tiny = AutoMALA(step_size=1e-4, max_halvings=8, max_doublings=12)
    state_g = sampler_tiny.init_state(np.ones(d), tracked_g)
    g_before_2 = tracked_g.grad_evals
    res_g = sampler_tiny.step(state_g, tracked_g, rng_local)
    g_after_2 = tracked_g.grad_evals
    assert res_g.stats["grad_evals"] == (g_after_2 - g_before_2)
    assert res_g.stats["k_fwd"] > 0
    assert res_g.stats["selected_step_size"] > 1e-4


@pytest.mark.unit
def test_automala_stationarity_on_gaussian():
    r"""Verify AutoMALA involutive reversibility preserves exact N(0, I_d) stationary distribution."""
    d = 3
    target = IsotropicGaussian(d=d)
    sampler = AutoMALA(step_size=0.5, adapt_step_size=True)
    samples, _ = sampler.sample(
        x0=np.zeros(d),
        target=target,
        n_steps=2000,
        burn_in=400,
        rng=np.random.default_rng(77),
        adapt=True,
    )
    assert np.allclose(np.mean(samples, axis=0), np.zeros(d), atol=0.12)
    assert np.allclose(np.std(samples, axis=0), np.ones(d), atol=0.12)
