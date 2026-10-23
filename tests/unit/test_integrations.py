"""Unit tests for PyMC and BlackJAX framework integrations."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from mcmc_robust.integrations import (
    BlackJAXAlgorithm,
    JaxArcsinhTransform,
    PyMCModelTarget,
    as_blackjax_algorithm,
    sample_pymc_model,
)
from mcmc_robust.samplers.novel.nuts import UnifiedRobustNUTS
from mcmc_robust.targets.benign import IsotropicGaussian


@pytest.fixture
def simple_pymc_model() -> pm.Model:
    """Builds a simple 2-parameter continuous PyMC model."""
    with pm.Model() as model:
        a = pm.Normal("a", mu=2.0, sigma=1.0)
        b = pm.Normal("b", mu=-1.0, sigma=2.0)
    return model


@pytest.fixture
def hierarchical_pymc_model() -> pm.Model:
    """Builds a hierarchical PyMC model with vector parameters."""
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=5.0)
        theta = pm.Normal("theta", mu=mu, sigma=1.0, shape=3)
    return model


# =============================================================================
# 1. PyMCModelTarget Tests
# =============================================================================


@pytest.mark.unit
def test_pymc_model_target_basic(simple_pymc_model: pm.Model):
    """Test PyMCModelTarget dimension, variable mapping, and initial point."""
    target = PyMCModelTarget(simple_pymc_model)
    assert target.d == 2
    assert "a" in target.variable_names
    assert "b" in target.variable_names
    assert target.initial_point_array.shape == (2,)

    # Unflattening roundtrip
    x = np.array([2.5, -0.5], dtype=np.float64)
    pt = target.unflatten(x)
    assert "a" in pt
    assert "b" in pt
    assert np.isclose(pt["a"], 2.5)
    assert np.isclose(pt["b"], -0.5)


@pytest.mark.unit
def test_pymc_model_target_logp_and_grad(simple_pymc_model: pm.Model):
    """Test logp and gradient computation against analytical normal densities."""
    target = PyMCModelTarget(simple_pymc_model)

    x = np.array([2.0, -1.0], dtype=np.float64)  # At the mean
    lp = target.log_prob(x)
    assert np.isfinite(lp)

    grad = target.grad_log_prob(x)
    # At the mean of N(2, 1) and N(-1, 2^2), grad should be ~ [0, 0]
    assert np.allclose(grad, [0.0, 0.0], atol=1e-5)

    lp2, grad2 = target.log_prob_and_grad(x)
    assert np.isclose(lp, lp2)
    assert np.allclose(grad, grad2)


@pytest.mark.unit
def test_pymc_model_target_vector_variables(hierarchical_pymc_model: pm.Model):
    """Test PyMCModelTarget with vector parameters."""
    target = PyMCModelTarget(hierarchical_pymc_model)
    assert target.d == 4  # mu (1) + theta (3) = 4

    x = np.array([1.0, 0.5, 1.5, 2.0], dtype=np.float64)
    pt = target.unflatten(x)
    assert pt["mu"].shape == ()
    assert pt["theta"].shape == (3,)
    assert np.isclose(pt["mu"], 1.0)
    assert np.allclose(pt["theta"], [0.5, 1.5, 2.0])

    lp, grad = target.log_prob_and_grad(x)
    assert np.isfinite(lp)
    assert grad.shape == (4,)
    assert np.all(np.isfinite(grad))


# =============================================================================
# 2. sample_pymc_model Drop-In API Tests
# =============================================================================


@pytest.mark.unit
def test_sample_pymc_model_returns_datatree(simple_pymc_model: pm.Model):
    """Test sample_pymc_model returns a valid ArviZ DataTree with correct groups."""
    dt = sample_pymc_model(
        model=simple_pymc_model,
        sampler="unified_nuts",
        draws=30,
        tune=15,
        chains=2,
        random_seed=123,
    )

    from xarray import DataTree

    assert isinstance(dt, DataTree)
    assert hasattr(dt, "posterior")
    assert hasattr(dt, "sample_stats")

    # Check posterior parameters
    assert "a" in dt.posterior
    assert "b" in dt.posterior
    assert dt.posterior["a"].shape == (2, 30)
    assert dt.posterior["b"].shape == (2, 30)

    # Check sample_stats group
    assert "diverging" in dt.sample_stats
    assert "step_size" in dt.sample_stats
    assert "n_steps" in dt.sample_stats
    assert "energy" in dt.sample_stats
    assert "acceptance_rate" in dt.sample_stats

    assert dt.sample_stats["diverging"].shape == (2, 30)
    assert dt.sample_stats["n_steps"].shape == (2, 30)


@pytest.mark.unit
def test_sample_pymc_model_with_langevin_sampler(simple_pymc_model: pm.Model):
    """Test sample_pymc_model works with UnifiedRobustSampler string identifier."""
    dt = sample_pymc_model(
        model=simple_pymc_model,
        sampler="unified",
        draws=25,
        tune=10,
        chains=2,
        random_seed=456,
    )
    assert dt.posterior["a"].shape == (2, 25)
    assert dt.posterior["b"].shape == (2, 25)


# =============================================================================
# 3. JaxArcsinhTransform Tests
# =============================================================================


@pytest.mark.unit
def test_jax_arcsinh_transform_roundtrip():
    """Verify exact roundtrip invertibility of JaxArcsinhTransform to machine precision."""
    transform = JaxArcsinhTransform(radius=np.array([1.5, 3.0, 5.0]))
    x = jnp.array([0.0, 5.0, -10.0], dtype=jnp.float64)

    z = transform.forward(x)
    x_rec = transform.inverse(z)

    assert jnp.allclose(x, x_rec, atol=1e-12)


@pytest.mark.unit
def test_jax_arcsinh_transform_jacobian_log_det():
    """Verify log-Jacobian determinant matches analytical sum of log(R * cosh(z))."""
    radius = jnp.array([2.0, 4.0], dtype=jnp.float64)
    transform = JaxArcsinhTransform(radius=radius)

    z = jnp.array([0.5, -1.2], dtype=jnp.float64)
    log_det = transform.log_jacobian_det(z)

    expected = float(jnp.sum(jnp.log(radius * jnp.cosh(z))))
    assert np.isclose(float(log_det), expected, atol=1e-12)


@pytest.mark.unit
def test_jax_arcsinh_transform_wrap_logdensity():
    """Verify transformed log-density preserves density transformation law."""
    transform = JaxArcsinhTransform(radius=3.0, dim=2)

    def standard_normal_logp(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(x**2)

    wrapped_logp = transform.wrap_logdensity(standard_normal_logp)

    z = jnp.array([1.0, -0.5], dtype=jnp.float64)
    x = transform.inverse(z)
    expected = standard_normal_logp(x) + transform.log_jacobian_det(z)

    actual = wrapped_logp(z)
    assert jnp.isclose(actual, expected)


@pytest.mark.unit
def test_jax_arcsinh_transform_calibrate_from_samples():
    """Verify empirical calibration from samples computes reasonable coordinate scales."""
    rng = np.random.default_rng(42)
    # 1st coord standard normal (IQR ~ 1.35), 2nd coord Cauchy (wide IQR)
    samples = np.column_stack(
        [
            rng.normal(0, 1.0, size=500),
            rng.standard_cauchy(size=500),
        ]
    )

    transform = JaxArcsinhTransform.calibrate_from_samples(samples, min_scale=0.2, max_scale=50.0)
    assert transform.d == 2
    assert transform.radius.shape == (2,)
    assert 0.2 <= float(transform.radius[0]) <= 2.0
    assert float(transform.radius[1]) >= 0.2


# =============================================================================
# 4. as_blackjax_algorithm Functional API Tests
# =============================================================================


@pytest.mark.unit
def test_as_blackjax_algorithm_interface():
    """Verify as_blackjax_algorithm provides standard init/step interface."""
    target = IsotropicGaussian(d=3)
    sampler = UnifiedRobustNUTS(step_size=0.1)

    algo = as_blackjax_algorithm(sampler, target)
    assert isinstance(algo, BlackJAXAlgorithm)

    init_pos = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    state = algo.init(init_pos)

    assert np.allclose(state.position, init_pos)
    assert np.isfinite(state.log_p)
    assert state.step_num == 0

    # Execute one transition
    key = jax.random.PRNGKey(99)
    next_state, info = algo.step(key, state)

    assert next_state.position.shape == (3,)
    assert np.all(np.isfinite(next_state.position))
    assert next_state.step_num == 1
    assert "step_size" in info or "divergent" in info or "hamiltonian_error" in info


# =============================================================================
# 4. README Quickstart Snippet Validation Tests
# =============================================================================


@pytest.mark.unit
def test_readme_quickstart_snippet_1_native() -> None:
    """Verifies that README Snippet 1 (Native API) executes cleanly."""
    from mcmc_robust import UnifiedRobustNUTS
    from mcmc_robust.targets import NealsFunnelTarget

    target = NealsFunnelTarget(d=10)
    sampler = UnifiedRobustNUTS(step_size=0.05, warmup_steps=50, max_depth=6)
    x0 = np.zeros(10)
    samples, results = sampler.sample(x0=x0, target=target, n_steps=100, burn_in=50)

    assert samples.shape == (50, 10)
    assert np.all(np.isfinite(samples))
    assert len(results) == 100
    divs = sum(int(r.stats.get("divergent", 0)) for r in results)
    assert divs >= 0


@pytest.mark.unit
def test_readme_quickstart_snippet_2_pymc() -> None:
    """Verifies that README Snippet 2 (PyMC model) executes cleanly."""
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, sigma=10)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal(
            "y",
            mu=theta,
            sigma=[15, 10, 16, 11, 9, 11, 10, 18],
            observed=[28, 8, -3, 7, -1, 1, 18, 12],
        )

    idata = sample_pymc_model(
        model,
        sampler="unified_nuts",
        draws=20,
        tune=10,
        chains=2,
        random_seed=42,
    )
    assert hasattr(idata, "posterior")
    assert "mu" in idata.posterior


@pytest.mark.unit
def test_readme_quickstart_snippet_3_jax() -> None:
    """Verifies that README Snippet 3 (Pure JAX / BlackJAX) executes cleanly."""
    import blackjax

    def logdensity_fn(x: jnp.ndarray) -> jnp.ndarray:
        v = x[0]
        return -0.5 * (v / 3.0) ** 2 - 0.5 * jnp.sum(x[1:] ** 2) * jnp.exp(-v) - 0.5 * 9.0 * v

    # Verify both radius and scales arguments work
    transform_r = JaxArcsinhTransform(radius=jnp.ones(10))
    transform_s = JaxArcsinhTransform(scales=jnp.ones(10))
    assert jnp.allclose(transform_r.radius, transform_s.radius)

    warped_logdensity = transform_r.wrap_logdensity(logdensity_fn)
    sampler = blackjax.nuts(warped_logdensity, step_size=0.05, inverse_mass_matrix=jnp.ones(10))
    state = sampler.init(jnp.zeros(10))
    step_fn = jax.jit(sampler.step)

    def body_fn(state: Any, key: Any) -> tuple[Any, jnp.ndarray]:
        state, info = step_fn(key, state)
        return state, transform_r.inverse(state.position)

    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    final_state, samples = jax.lax.scan(body_fn, state, keys)
    assert samples.shape == (10, 10)
    assert np.all(np.isfinite(np.asarray(samples)))
