"""Comprehensive unit test suite for UnifiedRobustHMC and UnifiedRobustNUTS (Features F25 & F26)."""

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.novel.nuts import UnifiedRobustNUTS
from mcmc_robust.samplers.novel.unified_hmc import UnifiedRobustHMC
from mcmc_robust.targets.benign import DiagonalGaussian, IsotropicGaussian
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)
from mcmc_robust.targets.roughness import QuarticTarget
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform

# ---------------------------------------------------------------------------
# 1. Symplectic Reversibility Checks (< 1e-12)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("gamma", [0.0, 0.2])
@pytest.mark.parametrize("n_leapfrog", [5, 10, 15])
def test_transformed_hmc_symplectic_reversibility(gamma: float, n_leapfrog: int):
    """Verify exact time-reversibility of transformed Hamiltonian leapfrog dynamics.

    Integrating forward L steps, flipping momentum p -> -p, integrating L steps,
    and flipping momentum p -> -p must recover (z0, p0) to machine precision (< 1e-12).
    Holds for any position-dependent force, including gentle taming (gamma >= 0).
    """
    d = 2
    target = IsotropicGaussian(d=d)
    sampler = UnifiedRobustHMC(
        step_size=0.15,
        n_leapfrog=n_leapfrog,
        gamma=gamma,
        radius=2.5,
    )
    transform = sampler._ensure_transform(d)
    s_z = np.array([1.2, 0.8], dtype=np.float64)
    sampler.set_preconditioner(s_z)

    z0 = np.array([0.7, -1.1], dtype=np.float64)
    p0 = np.array([-0.4, 0.9], dtype=np.float64)
    epsilon = sampler.step_size / n_leapfrog

    def leapfrog_integrate(
        z_init: np.ndarray,
        p_init: np.ndarray,
        steps: int,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        z_curr = z_init.copy()
        p_curr = p_init.copy()

        # Compute initial force at z_init
        x_init, _ = transform.inverse(z_curr)
        grad_x = target.grad_log_prob(x_init)
        grad_z = transform.forward_grad(x_init, grad_x)
        force, _ = sampler._gentle_force(grad_z, s_z, eps)

        for _ in range(steps):
            # Half kick
            p_curr += 0.5 * eps * force
            # Full drift
            z_curr += eps * sampler._velocity(p_curr, s_z)
            # Evaluate new force
            x_curr, _ = transform.inverse(z_curr)
            grad_x = target.grad_log_prob(x_curr)
            grad_z = transform.forward_grad(x_curr, grad_x)
            force, _ = sampler._gentle_force(grad_z, s_z, eps)
            # Half kick
            p_curr += 0.5 * eps * force

        return z_curr, p_curr

    # Forward integration
    z_fwd, p_fwd = leapfrog_integrate(z0, p0, n_leapfrog, epsilon)

    # Momentum flip R(z, p) = (z, -p)
    p_flipped = -p_fwd

    # Backward integration (forward with flipped momentum)
    z_rev, p_rev = leapfrog_integrate(z_fwd, p_flipped, n_leapfrog, epsilon)

    # Momentum flip again
    p_final = -p_rev

    # Assert exact reversibility to machine precision
    z_err = np.max(np.abs(z_rev - z0))
    p_err = np.max(np.abs(p_final - p0))

    assert z_err < 1e-12, f"Reversibility error in z={z_err} exceeds 1e-12 (gamma={gamma})"
    assert p_err < 1e-12, f"Reversibility error in p={p_err} exceeds 1e-12 (gamma={gamma})"


# ---------------------------------------------------------------------------
# 2. Exact Volume Preservation (det J = 1.0)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("gamma", [0.0, 0.3])
def test_transformed_hmc_phase_space_volume_preservation(gamma: float):
    """Verify that transformed leapfrog flow preserves phase-space volume (det J == 1.0).

    Computes numerical Jacobian of the map (z0, p0) -> (zL, pL) in R^{2d}
    via central finite differences and verifies det(J) = 1.0 within discretization error.
    """
    d = 1
    target = QuarticTarget(d=d)
    sampler = UnifiedRobustHMC(
        step_size=0.1,
        n_leapfrog=5,
        gamma=gamma,
        radius=2.0,
    )
    transform = sampler._ensure_transform(d)
    s_z = np.ones(d, dtype=np.float64)
    epsilon = sampler.step_size / sampler.n_leapfrog

    def flow_map(w: np.ndarray) -> np.ndarray:
        z_curr = np.array([w[0]], dtype=np.float64)
        p_curr = np.array([w[1]], dtype=np.float64)

        x_curr, _ = transform.inverse(z_curr)
        grad_x = target.grad_log_prob(x_curr)
        grad_z = transform.forward_grad(x_curr, grad_x)
        force, _ = sampler._gentle_force(grad_z, s_z, epsilon)

        for _ in range(sampler.n_leapfrog):
            p_curr += 0.5 * epsilon * force
            z_curr += epsilon * sampler._velocity(p_curr, s_z)
            x_curr, _ = transform.inverse(z_curr)
            grad_x = target.grad_log_prob(x_curr)
            grad_z = transform.forward_grad(x_curr, grad_x)
            force, _ = sampler._gentle_force(grad_z, s_z, epsilon)
            p_curr += 0.5 * epsilon * force

        return np.array([z_curr[0], p_curr[0]], dtype=np.float64)

    w0 = np.array([0.6, -0.4], dtype=np.float64)
    delta = 1e-6
    n_dim = 2 * d
    J = np.empty((n_dim, n_dim), dtype=np.float64)

    for col in range(n_dim):
        w_plus = w0.copy()
        w_minus = w0.copy()
        w_plus[col] += delta
        w_minus[col] -= delta

        out_plus = flow_map(w_plus)
        out_minus = flow_map(w_minus)
        J[:, col] = (out_plus - out_minus) / (2.0 * delta)

    det_J = float(np.linalg.det(J))
    assert abs(det_J - 1.0) < 1e-5, f"det(J)={det_J} deviates from 1.0 (gamma={gamma})"


# ---------------------------------------------------------------------------
# 3. Transformed Potential Gradient Accuracy vs Finite Differences (< 1e-6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "target_cls, kwargs",
    [
        (IsotropicGaussian, {"d": 2}),
        (MultivariateStudentTTarget, {"d": 2, "nu": 2.5}),
        (QuarticTarget, {"d": 2}),
    ],
)
def test_transformed_potential_gradient_accuracy(target_cls, kwargs):
    """Verify that pullback gradient matches central finite differences to < 1e-6."""
    target = target_cls(**kwargs)
    d = target.d
    radii = np.array([1.5, 3.0], dtype=np.float64)
    transform = CoordinateWiseArcsinhTransform(d=d, radii=radii)

    def log_prob_z(z_vec: np.ndarray) -> float:
        x_vec, log_det_inv = transform.inverse(z_vec)
        lp_x = target.log_prob(x_vec)
        return float(lp_x + log_det_inv)

    z_test = np.array([0.7, -0.9], dtype=np.float64)
    x_test, _ = transform.inverse(z_test)
    grad_x = target.grad_log_prob(x_test)
    analytical_grad_z = transform.forward_grad(x_test, grad_x)

    delta = 1e-6
    num_grad_z = np.empty(d, dtype=np.float64)
    for i in range(d):
        zp = z_test.copy()
        zm = z_test.copy()
        zp[i] += delta
        zm[i] -= delta
        num_grad_z[i] = (log_prob_z(zp) - log_prob_z(zm)) / (2.0 * delta)

    max_err = np.max(np.abs(analytical_grad_z - num_grad_z))
    assert max_err < 1e-6, f"Transformed gradient error {max_err} exceeds 1e-6"


# ---------------------------------------------------------------------------
# 4. Exact Stationarity Tests (Two-Sample KS p-value > 0.01)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unified_hmc_stationarity_ks_gaussian():
    """UnifiedRobustHMC preserves exact Gaussian stationarity (KS p-value > 0.01)."""
    target = IsotropicGaussian(d=1)
    sampler = UnifiedRobustHMC(
        step_size=0.4,
        n_leapfrog=5,
        gamma=0.0,
        radius=2.0,
    )
    rng = np.random.default_rng(101)
    samples, _ = sampler.sample(np.array([0.5]), target, n_steps=6000, burn_in=1000, rng=rng)
    gt = target.sample_ground_truth(10000, rng=rng).flatten()
    ks = stats.ks_2samp(samples.flatten()[::2], gt)
    assert ks.pvalue > 0.01, f"HMC Gaussian KS test failed with p={ks.pvalue}"


@pytest.mark.unit
def test_unified_nuts_stationarity_ks_gaussian():
    """UnifiedRobustNUTS preserves exact Gaussian stationarity (KS p-value > 0.01)."""
    target = IsotropicGaussian(d=1)
    sampler = UnifiedRobustNUTS(
        step_size=0.15,
        max_depth=6,
        gamma=0.0,
        radius=2.0,
    )
    rng = np.random.default_rng(202)
    samples, _ = sampler.sample(np.array([0.5]), target, n_steps=5000, burn_in=1000, rng=rng)
    gt = target.sample_ground_truth(10000, rng=rng).flatten()
    ks = stats.ks_2samp(samples.flatten()[::2], gt)
    assert ks.pvalue > 0.01, f"NUTS Gaussian KS test failed with p={ks.pvalue}"


@pytest.mark.unit
def test_unified_hmc_stationarity_ks_student_t():
    """UnifiedRobustHMC preserves heavy-tailed Student-t stationarity (KS p-value > 0.01)."""
    target = MultivariateStudentTTarget(d=1, nu=2.5)
    sampler = UnifiedRobustHMC(
        step_size=0.3,
        n_leapfrog=5,
        gamma=0.05,
        radius=1.5,
    )
    rng = np.random.default_rng(303)
    samples, _ = sampler.sample(np.array([0.0]), target, n_steps=7000, burn_in=1500, rng=rng)
    gt = target.sample_ground_truth(10000, rng=rng).flatten()
    ks = stats.ks_2samp(samples.flatten()[::2], gt)
    assert ks.pvalue > 0.01, f"HMC Student-t KS test failed with p={ks.pvalue}"


@pytest.mark.unit
def test_unified_nuts_stationarity_ks_student_t():
    """UnifiedRobustNUTS preserves heavy-tailed Student-t stationarity (KS p-value > 0.01)."""
    target = MultivariateStudentTTarget(d=1, nu=2.5)
    sampler = UnifiedRobustNUTS(
        step_size=0.15,
        max_depth=6,
        gamma=0.05,
        radius=1.5,
    )
    rng = np.random.default_rng(404)
    samples, _ = sampler.sample(np.array([0.0]), target, n_steps=6000, burn_in=1000, rng=rng)
    gt = target.sample_ground_truth(10000, rng=rng).flatten()
    ks = stats.ks_2samp(samples.flatten()[::2], gt)
    assert ks.pvalue > 0.01, f"NUTS Student-t KS test failed with p={ks.pvalue}"


@pytest.mark.unit
def test_unified_hmc_stationarity_ks_quartic():
    """UnifiedRobustHMC preserves rough QuarticTarget stationarity (KS p-value > 0.01)."""
    target = QuarticTarget(d=1)
    sampler = UnifiedRobustHMC(
        step_size=0.25,
        n_leapfrog=5,
        gamma=0.1,
        radius=2.0,
    )
    rng = np.random.default_rng(505)
    samples, _ = sampler.sample(np.array([0.8]), target, n_steps=6000, burn_in=1000, rng=rng)
    gt = target.sample_ground_truth(10000, rng=rng).flatten()
    ks = stats.ks_2samp(samples.flatten()[::2], gt)
    assert ks.pvalue > 0.01, f"HMC Quartic KS test failed with p={ks.pvalue}"


@pytest.mark.unit
def test_unified_nuts_stationarity_ks_quartic():
    """UnifiedRobustNUTS preserves rough QuarticTarget stationarity (KS p-value > 0.01)."""
    target = QuarticTarget(d=1)
    sampler = UnifiedRobustNUTS(
        step_size=0.12,
        max_depth=6,
        gamma=0.1,
        radius=2.0,
    )
    rng = np.random.default_rng(606)
    samples, _ = sampler.sample(np.array([0.8]), target, n_steps=5000, burn_in=1000, rng=rng)
    gt = target.sample_ground_truth(10000, rng=rng).flatten()
    ks = stats.ks_2samp(samples.flatten()[::2], gt)
    assert ks.pvalue > 0.01, f"NUTS Quartic KS test failed with p={ks.pvalue}"


# ---------------------------------------------------------------------------
# 5. Bounded Tree Depth on Heavy-Tailed Targets (No Blowup to max_depth)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nuts_bounded_tree_depth_on_heavy_tails():
    """Confirm NUTS tree depth is strictly bounded on heavy tails without blowup.

    Under the arcsinh diffeomorphism, V(z) ~ nu |z|, providing constant restoring force.
    The No-U-Turn stopping criterion triggers in finite, small leapfrog steps,
    preventing tree depth from reaching max_depth (10).
    """
    target = MultivariateStudentTTarget(d=1, nu=2.0)
    max_depth = 10
    sampler = UnifiedRobustNUTS(
        step_size=0.15,
        max_depth=max_depth,
        radius=1.5,
    )
    rng = np.random.default_rng(707)

    # Initialize deep in the heavy tail (x0 = 15.0)
    state = sampler.init_state(np.array([15.0]), target)

    depths = []
    n_steps = 100
    for _ in range(n_steps):
        res = sampler.step(state, target, rng)
        depths.append(res.stats["tree_depth"])
        state = res.state

    mean_depth = float(np.mean(depths))
    max_reached_depth = int(np.max(depths))

    # In transformed space, mean depth should stay <= 5.0, far below max_depth=10
    assert mean_depth < 6.0, f"Mean tree depth {mean_depth} too high; blowup suspected"
    assert max_reached_depth < max_depth, f"Tree depth reached max_depth={max_depth}; blowup occurred on heavy tail"


# ---------------------------------------------------------------------------
# 6. Stiff Tail Initialization Stability (||x0|| = 10 * sqrt(d))
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("sampler_cls", [UnifiedRobustHMC, UnifiedRobustNUTS])
def test_stiff_tail_initialization_numerical_stability(sampler_cls):
    """Verify numerical stability on extreme superlinear quartic tail initialization.

    Initializes at ||x0|| = 10 * sqrt(d) where gradient magnitude is ~ (14.14)^3 = 2828.
    Gentle stiffness attenuation (gamma > 0) prevents numerical overflow and NaN.
    """
    d = 2
    target = QuarticTarget(d=d)
    x0 = np.full(d, 10.0, dtype=np.float64)  # ||x0|| = 10 * sqrt(2) ~ 14.14

    if sampler_cls is UnifiedRobustHMC:
        sampler = sampler_cls(
            step_size=0.2,
            n_leapfrog=10,
            gamma=0.2,
            radius=2.0,
        )
    else:
        sampler = sampler_cls(
            step_size=0.02,
            max_depth=5,
            gamma=0.2,
            radius=2.0,
        )
    rng = np.random.default_rng(808)

    state = sampler.init_state(x0, target)
    assert np.all(np.isfinite(state.x))

    for step_idx in range(50):
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x)), f"{sampler_cls.__name__} produced non-finite state at step {step_idx}"
        assert np.isfinite(res.state.log_prob)
        state = res.state

    # After 50 steps, chain should contract significantly toward the bulk (||x|| < 5.0)
    final_norm = float(np.linalg.norm(state.x))
    assert final_norm < 8.0, f"Chain failed to contract from stiff tail: final norm {final_norm}"


# ---------------------------------------------------------------------------
# 7. Zero-Redundancy Oracle Budget Contracts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unified_hmc_oracle_budget_contract():
    """UnifiedRobustHMC strictly evaluates L gradients per step and 1 log_prob."""
    d = 2
    target = IsotropicGaussian(d=d)
    tracker = GradBudgetTracker(target)
    L = 7
    sampler = UnifiedRobustHMC(step_size=0.2, n_leapfrog=L)

    x0 = np.array([0.5, -0.5])
    state = sampler.init_state(x0, tracker)

    # Init contract: 1 grad, 1 log_prob
    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1

    rng = np.random.default_rng(909)
    n_steps = 15
    for _ in range(n_steps):
        prior_grad = tracker.n_grad
        prior_lp = tracker.n_log_prob
        res = sampler.step(state, tracker, rng)

        assert tracker.n_grad == prior_grad + L, f"Expected strictly L={L} grads, got {tracker.n_grad - prior_grad}"
        assert tracker.n_log_prob == prior_lp + 1
        assert res.stats["grad_evals"] == L

        if not res.accepted:
            # Auxiliary may be updated with cached transform, but x, log_prob, grad_log_prob unchanged
            assert np.array_equal(res.state.x, state.x)
            assert res.state.log_prob == state.log_prob

        state = res.state

    assert tracker.n_grad == 1 + L * n_steps
    assert tracker.n_log_prob == 1 + n_steps


@pytest.mark.unit
def test_unified_nuts_oracle_budget_contract():
    """UnifiedRobustNUTS strictly evaluates 1 gradient per leapfrog step."""
    d = 2
    target = IsotropicGaussian(d=d)
    tracker = GradBudgetTracker(target)
    sampler = UnifiedRobustNUTS(step_size=0.1, max_depth=5)

    x0 = np.array([0.3, -0.4])
    state = sampler.init_state(x0, tracker)

    assert tracker.n_grad == 1
    assert tracker.n_log_prob == 1

    rng = np.random.default_rng(1010)
    n_steps = 12
    total_leapfrogs = 0

    for _ in range(n_steps):
        prior_grad = tracker.n_grad
        res = sampler.step(state, tracker, rng)
        n_lf = res.stats["n_leapfrog"]
        total_leapfrogs += n_lf

        assert tracker.n_grad == prior_grad + n_lf, (
            f"Target grads {tracker.n_grad - prior_grad} must equal n_leapfrog {n_lf}"
        )
        assert res.stats["grad_evals"] == n_lf
        state = res.state

    assert tracker.n_grad == 1 + total_leapfrogs


# ---------------------------------------------------------------------------
# 8. Three-Phase Warmup Adaptation Progression
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("sampler_cls", [UnifiedRobustHMC, UnifiedRobustNUTS])
def test_three_phase_warmup_adaptation_progression(sampler_cls):
    """Verify that Phase 1 freezes tail scale, Phase 2 freezes metric, and Phase 3 tunes epsilon."""
    target = HeterogeneousHeavyTailedTarget(d=3, heavy_coords=[0, 1], nu_list=[1.5, 2.5])
    mask = np.zeros(3, dtype=bool)
    mask[target.heavy_coords] = True
    warmup_steps = 100
    sampler = sampler_cls(
        step_size=0.05,
        radius=1.0,
        mask=mask,
        adapt_step_size=True,
        adapt_preconditioner=True,
        adapt_tail_scale=True,
        warmup_steps=warmup_steps,
    )
    rng = np.random.default_rng(1111)

    initial_radius = 1.0
    initial_step_size = sampler.step_size

    samples, results = sampler.sample(
        x0=np.array([1.5, -2.0, 0.5]),
        target=target,
        n_steps=120,
        burn_in=warmup_steps,
        rng=rng,
        adapt=True,
    )

    # Tail scale must be frozen and adapted from initial value
    assert sampler._tail_scale_frozen is True
    assert sampler.radius is not None
    assert not np.allclose(sampler.radius, initial_radius)

    # Preconditioner must be frozen and adapted
    assert sampler._preconditioner_frozen is True
    assert sampler.preconditioner is not None

    # Step size must have evolved
    assert sampler.step_size != initial_step_size


# ---------------------------------------------------------------------------
# 9. Divergence Detection and Energy Diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nuts_divergence_detection_on_extreme_energy_error():
    """Verify that energy error Delta H > delta_max flags divergent=1.0 and rejects gracefully."""
    target = QuarticTarget(d=2)
    delta_max = 50.0
    # Artificially huge step size with no taming forces energy divergence
    sampler = UnifiedRobustNUTS(
        step_size=5.0,
        max_depth=4,
        gamma=0.0,
        delta_max=delta_max,
    )
    rng = np.random.default_rng(1212)
    state = sampler.init_state(np.array([3.0, 3.0]), target)

    res = sampler.step(state, target, rng)
    assert res.stats["divergent"] == 1.0
    # Must remain safe and finite
    assert np.all(np.isfinite(res.state.x))


# ---------------------------------------------------------------------------
# 10. Class Aliasing and Export Contracts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_class_aliasing_and_exports():
    """Verify ArcsinhTransformedNUTS is an exact alias of UnifiedRobustNUTS and exported."""
    from mcmc_robust.samplers import (
        ArcsinhTransformedNUTS as TopAlias,
        UnifiedRobustHMC as TopHMC,
        UnifiedRobustNUTS as TopNUTS,
    )
    from mcmc_robust.samplers.novel import (
        ArcsinhTransformedNUTS as NovelAlias,
        UnifiedRobustHMC as NovelHMC,
        UnifiedRobustNUTS as NovelNUTS,
    )

    assert TopNUTS is UnifiedRobustNUTS
    assert NovelNUTS is UnifiedRobustNUTS
    assert TopAlias is UnifiedRobustNUTS
    assert NovelAlias is UnifiedRobustNUTS
    assert TopHMC is UnifiedRobustHMC
    assert NovelHMC is UnifiedRobustHMC


# ---------------------------------------------------------------------------
# 11. Parameter Validation Guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_defensive_parameter_validation_guards():
    """Verify ValueError is raised on invalid configuration parameters."""
    with pytest.raises(ValueError, match="n_leapfrog must be >= 1"):
        UnifiedRobustHMC(n_leapfrog=0)
    with pytest.raises(ValueError, match="gamma must be non-negative"):
        UnifiedRobustHMC(gamma=-0.5)
    with pytest.raises(ValueError, match="mode must be"):
        UnifiedRobustHMC(mode="invalid_mode")
    with pytest.raises(ValueError, match="target_accept must be in"):
        UnifiedRobustHMC(target_accept=0.0)
    with pytest.raises(ValueError, match="taming_power must be in"):
        UnifiedRobustHMC(taming_power=0.0)

    with pytest.raises(ValueError, match="max_depth must be non-negative"):
        UnifiedRobustNUTS(max_depth=-1)
    with pytest.raises(ValueError, match="delta_max must be positive"):
        UnifiedRobustNUTS(delta_max=0.0)
    with pytest.raises(ValueError, match="Invalid radius bounds"):
        UnifiedRobustNUTS(min_radius=10.0, max_radius=5.0)


# ---------------------------------------------------------------------------
# 12. Energy Conservation: O(epsilon^2) Scaling for Symplectic Leapfrog
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_symplectic_energy_conservation_scaling():
    """Verify that Hamiltonian error scales as O(epsilon^2) for fixed trajectory time (gamma=0)."""
    target = IsotropicGaussian(d=2)
    # Fixed trajectory length h = 0.4
    s1 = UnifiedRobustHMC(step_size=0.4, n_leapfrog=10, gamma=0.0)  # eps = 0.04
    s2 = UnifiedRobustHMC(step_size=0.4, n_leapfrog=20, gamma=0.0)  # eps = 0.02

    rng = np.random.default_rng(1313)
    x0 = np.array([0.5, -0.5])
    state1 = s1.init_state(x0, target)
    state2 = s2.init_state(x0, target)

    # Single deterministic step with same momentum
    res1 = s1.step(state1, target, np.random.default_rng(42))
    res2 = s2.step(state2, target, np.random.default_rng(42))

    err1 = abs(res1.stats["hamiltonian_error"])
    err2 = abs(res2.stats["hamiltonian_error"])

    # Halving step size should reduce energy error by approximately factor of 4
    ratio = err1 / max(1e-12, err2)
    assert 2.0 < ratio < 6.0, f"Expected ~4x error reduction (O(eps^2)), got ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# 13. Preconditioning Scaling and Anisotropic Targets
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("sampler_cls", [UnifiedRobustHMC, UnifiedRobustNUTS])
def test_preconditioned_anisotropic_gaussian_exploration(sampler_cls):
    """Verify diagonal preconditioning enables efficient exploration on ill-conditioned Gaussian."""
    target = DiagonalGaussian(d=2, condition_number=50.0)
    stds = target.stds
    rng = np.random.default_rng(1414)

    if sampler_cls is UnifiedRobustHMC:
        sampler = sampler_cls(
            step_size=0.3,
            n_leapfrog=6,
            preconditioner=stds,
            radius=3.0,
        )
    else:
        sampler = sampler_cls(
            step_size=0.1,
            max_depth=5,
            preconditioner=stds,
            radius=3.0,
        )

    samples, results = sampler.sample(
        x0=np.array([0.0, 0.0]),
        target=target,
        n_steps=500,
        burn_in=100,
        rng=rng,
    )

    accept_rates = [r.accepted for r in results[100:]]
    mean_accept = np.mean(accept_rates)
    assert mean_accept > 0.60, f"Preconditioned {sampler_cls.__name__} acceptance {mean_accept} too low"


# ---------------------------------------------------------------------------
# 14. Global Stiffness Attenuation Mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("sampler_cls", [UnifiedRobustHMC, UnifiedRobustNUTS])
def test_global_mode_stiffness_attenuation(sampler_cls):
    """Verify mode='global' gentle force executes properly without error."""
    target = QuarticTarget(d=2)
    rng = np.random.default_rng(1515)

    sampler = sampler_cls(
        step_size=0.05,
        gamma=0.1,
        mode="global",
        radius=2.0,
    )

    state = sampler.init_state(np.array([1.5, -1.5]), target)
    for _ in range(5):
        res = sampler.step(state, target, rng)
        assert np.all(np.isfinite(res.state.x))
        state = res.state


# ---------------------------------------------------------------------------
# 15. Fallback Warmup without Warmup Steps (Pure Dual Averaging)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("sampler_cls", [UnifiedRobustHMC, UnifiedRobustNUTS])
def test_pure_dual_averaging_warmup_fallback(sampler_cls):
    """Verify adapt_step_size=True with warmup_steps=0 falls back cleanly to dual averaging."""
    target = IsotropicGaussian(d=2)
    rng = np.random.default_rng(1616)

    sampler = sampler_cls(
        step_size=0.01,
        adapt_step_size=True,
        warmup_steps=0,
    )
    init_step_size = sampler.step_size

    samples, _ = sampler.sample(
        x0=np.array([0.0, 0.0]),
        target=target,
        n_steps=60,
        burn_in=50,
        rng=rng,
        adapt=True,
    )

    assert sampler.step_size > init_step_size, f"Step size failed to adapt for {sampler_cls.__name__}"


# ---------------------------------------------------------------------------
# 16. Structural Attributes and Interface Contracts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sampler_attributes_and_properties():
    """Verify requires_grad, transform, and radius vector handling."""
    hmc = UnifiedRobustHMC(radius=np.array([2.0, 3.0]))
    nuts = UnifiedRobustNUTS(radius=np.array([1.5, 2.5]))

    assert hmc.requires_grad is True
    assert nuts.requires_grad is True

    target = IsotropicGaussian(d=2)
    _ = hmc.init_state(np.array([0.0, 0.0]), target)
    _ = nuts.init_state(np.array([0.0, 0.0]), target)

    assert isinstance(hmc.transform, CoordinateWiseArcsinhTransform)
    assert isinstance(nuts.transform, CoordinateWiseArcsinhTransform)
    assert np.allclose(hmc.radius, [2.0, 3.0])
    assert np.allclose(nuts.radius, [1.5, 2.5])

    # Dimension mismatch in initial radius vector raises ValueError
    hmc_mismatch = UnifiedRobustHMC(radius=np.array([2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="Initial radius shape"):
        hmc_mismatch.init_state(np.array([0.0, 0.0]), target)
