"""Tests for the BridgeStan-to-JAX log density adapter (`mcmc_robust.stan_bridge`).

Most tests use `GaussianModel`, a NumPy object with the two `bridgestan.StanModel` methods the adapter
calls, so they run without Stan. The last test compiles a real Stan model and is skipped unless the
`posteriordb` extra (BridgeStan) is installed and a C++ toolchain is available.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mcmc_robust.jax_unified_nuts import sample_blackjax_unified_nuts
from mcmc_robust.stan_bridge import bridgestan_logdensity_fn

SCALES = np.array([0.5, 1.0, 3.0])


class GaussianModel:
    """N(0, diag(SCALES^2)) on the unconstrained space; raises like BridgeStan outside |theta_0| < `bound`."""

    def __init__(self, bound: float = np.inf) -> None:
        self.bound = bound
        self.calls = 0

    def param_unc_num(self) -> int:
        return SCALES.size

    def _check(self, theta: np.ndarray) -> None:
        self.calls += 1
        if abs(theta[0]) >= self.bound:
            raise RuntimeError("log_density() failed with exception: domain error")

    # Stan's C++ overflows to inf silently; NumPy would warn, and tests run with -W error. Step-size search
    # through the arcsinh map does propose points whose x overflows.
    def log_density(self, theta: np.ndarray, *, propto: bool = True, jacobian: bool = True) -> float:
        self._check(theta)
        with np.errstate(over="ignore"):
            return float(-0.5 * np.sum((theta / SCALES) ** 2))

    def log_density_gradient(
        self, theta: np.ndarray, *, propto: bool = True, jacobian: bool = True
    ) -> tuple[float, np.ndarray]:
        self._check(theta)
        with np.errstate(over="ignore"):
            return float(-0.5 * np.sum((theta / SCALES) ** 2)), -theta / SCALES**2


@pytest.mark.unit
def test_value_and_gradient_match_the_model() -> None:
    model = GaussianModel()
    logdensity = bridgestan_logdensity_fn(model)
    theta = np.array([0.3, -1.2, 2.0])
    value, grad = jax.value_and_grad(logdensity)(jnp.asarray(theta))
    expected_value, expected_grad = model.log_density_gradient(theta)
    assert float(value) == pytest.approx(expected_value, rel=1e-15)
    np.testing.assert_allclose(np.asarray(grad), expected_grad, rtol=1e-15)


@pytest.mark.unit
def test_works_under_jit_and_vmap() -> None:
    model = GaussianModel()
    logdensity = bridgestan_logdensity_fn(model)
    thetas = np.random.default_rng(0).normal(size=(5, 3))
    values, grads = jax.jit(jax.vmap(jax.value_and_grad(logdensity)))(jnp.asarray(thetas))
    for theta, value, grad in zip(thetas, np.asarray(values), np.asarray(grads)):
        expected_value, expected_grad = model.log_density_gradient(theta)
        assert value == pytest.approx(expected_value, rel=1e-15)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-15)


@pytest.mark.unit
def test_cotangent_scales_the_gradient() -> None:
    logdensity = bridgestan_logdensity_fn(GaussianModel())
    theta = jnp.array([0.3, -1.2, 2.0])
    np.testing.assert_allclose(
        np.asarray(jax.grad(lambda t: 2.5 * logdensity(t))(theta)),
        2.5 * np.asarray(jax.grad(logdensity)(theta)),
        rtol=1e-15,
    )


@pytest.mark.unit
def test_a_model_error_is_minus_infinity_with_a_zero_gradient() -> None:
    """A Stan exception (outside the support, a failed solver) is a point of zero density, which NUTS rejects."""
    logdensity = bridgestan_logdensity_fn(GaussianModel(bound=1.0))
    value, grad = jax.value_and_grad(logdensity)(jnp.array([1.5, 0.0, 0.0]))
    assert float(value) == -np.inf
    np.testing.assert_array_equal(np.asarray(grad), np.zeros(3))
    assert float(logdensity(jnp.array([1.5, 0.0, 0.0]))) == -np.inf


@pytest.mark.unit
def test_primal_evaluation_does_not_compute_a_gradient() -> None:
    """Without differentiation the adapter calls log_density, not log_density_gradient."""

    class NoGradient(GaussianModel):
        def log_density_gradient(self, theta, **kwargs):  # type: ignore[no-untyped-def,override]
            raise AssertionError("gradient requested for a primal evaluation")

    assert float(bridgestan_logdensity_fn(NoGradient())(jnp.zeros(3))) == 0.0


@pytest.mark.unit
@pytest.mark.statistical
def test_the_pipeline_samples_through_the_adapter() -> None:
    """sample_blackjax_unified_nuts on the adapted Gaussian recovers its scales (4 x 1000 draws)."""
    res = sample_blackjax_unified_nuts(
        bridgestan_logdensity_fn(GaussianModel()), d=3, draws=1000, tune=500, chains=4, seed=0
    )
    draws = res.samples.reshape(-1, 3)
    # Monte Carlo error of the sd at ESS ~ 1000 is ~2%; the tolerance is 10%.
    np.testing.assert_allclose(np.std(draws, axis=0), SCALES, rtol=0.10)
    assert np.all(np.abs(np.mean(draws, axis=0)) < 0.15 * SCALES)


requires_bridgestan = pytest.mark.skipif(
    importlib.util.find_spec("bridgestan") is None or shutil.which("make") is None,
    reason="needs BridgeStan (posteriordb extra) and make",
)

STAN_CODE = """
parameters {
  real<lower=0> sigma;
  vector[2] x;
}
model {
  sigma ~ lognormal(0, 1);
  x ~ normal(0, sigma);
}
"""


@pytest.mark.unit
@requires_bridgestan
def test_a_compiled_stan_model_round_trips(tmp_path: Path) -> None:
    """The adapter reproduces BridgeStan's log density and gradient, Jacobian included, on a compiled model."""
    import bridgestan as bs

    stan_file = tmp_path / "scale.stan"
    stan_file.write_text(STAN_CODE)
    model = bs.StanModel(str(stan_file))
    logdensity = bridgestan_logdensity_fn(model)
    theta = np.array([0.4, -0.7, 1.1])
    value, grad = jax.value_and_grad(logdensity)(jnp.asarray(theta))
    expected_value, expected_grad = model.log_density_gradient(theta, propto=True, jacobian=True)
    assert float(value) == pytest.approx(expected_value, rel=1e-14)
    np.testing.assert_allclose(np.asarray(grad), expected_grad, rtol=1e-14)
