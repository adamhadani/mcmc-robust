"""Unit tests for the direct-competitor sampler adapters (bench/competitors.py).

Tests that need nutpie (or, for the flow arm, flowjax) skip cleanly when the optional
``competitors`` extra is not installed (``uv sync --extra competitors``).
"""

from __future__ import annotations

import sys

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest

from mcmc_robust.bench.competitors import (
    STANDARD_TELEMETRY_KEYS,
    CompetitorUnavailableError,
    InitPositionQueue,
    JaxLogdensityTarget,
    nutpie_available,
    nutpie_flow_available,
    run_arcsinh_nuts,
    run_nutpie,
    run_nutpie_diag,
    run_nutpie_flow,
    run_stock_nuts,
)
from mcmc_robust.targets.benign import DiagonalGaussian

requires_nutpie = pytest.mark.skipif(not nutpie_available(), reason="nutpie not installed (competitors extra)")
requires_flow = pytest.mark.skipif(
    not nutpie_flow_available(), reason="nutpie flow adaptation needs nutpie, flowjax and equinox (competitors extra)"
)

SCALES = np.array([1.0, 2.0, 0.5], dtype=np.float64)


def _gaussian_logdensity(x: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.sum((x / jnp.asarray(SCALES)) ** 2)


def _init(chains: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-2.0, 2.0, size=(chains, d))


def _assert_standard_telemetry(telemetry: dict, chains: int, draws: int) -> None:
    for key in STANDARD_TELEMETRY_KEYS:
        assert key in telemetry, f"missing telemetry key {key}"
    assert telemetry["total_grad_evals"] == telemetry["warmup_grad_evals"] + telemetry["draw_grad_evals"]
    # Every retained draw needs at least one leapfrog step, so at least one gradient.
    assert telemetry["draw_grad_evals"] >= chains * draws
    assert telemetry["warmup_grad_evals"] > 0
    assert telemetry["divergences"] >= 0
    # None means the sampler does not expose warm-up divergences (the arcsinh pipeline).
    assert telemetry["warmup_divergences"] is None or telemetry["warmup_divergences"] >= 0
    assert telemetry["wall_clock_time"] > 0.0
    assert isinstance(telemetry["grad_eval_accounting"], str) and telemetry["grad_eval_accounting"]


def _assert_gaussian_moments_within_mcse(samples: np.ndarray, scales: np.ndarray, n_mcse: float = 5.0) -> None:
    """Mean and variance of every coordinate agree with the truth to within ``n_mcse`` MCSEs."""
    dt = az.from_dict({"posterior": {"x": samples, "x2": samples**2}})
    mcse = az.mcse(dt, method="mean")
    mean_hat = samples.reshape(-1, samples.shape[-1]).mean(axis=0)
    var_hat = (samples**2).reshape(-1, samples.shape[-1]).mean(axis=0)
    mcse_mean = np.asarray(mcse["x"].values)
    mcse_var = np.asarray(mcse["x2"].values)
    assert np.all(np.isfinite(mcse_mean)) and np.all(mcse_mean > 0)
    assert np.all(np.abs(mean_hat) <= n_mcse * mcse_mean), (mean_hat, mcse_mean)
    assert np.all(np.abs(var_hat - scales**2) <= n_mcse * mcse_var), (var_hat, scales**2, mcse_var)


# =============================================================================
# 1. Helpers that do not need the optional extra
# =============================================================================


@pytest.mark.unit
def test_init_position_queue_returns_rows_in_order_then_raises() -> None:
    rows = _init(3, 2)
    queue = InitPositionQueue(rows)
    for expected in rows:
        got = queue(123)
        assert got.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(got, expected)
    assert queue.n_served == 3
    with pytest.raises(RuntimeError, match="initial positions"):
        queue(0)


@pytest.mark.unit
def test_jax_logdensity_target_wraps_callable() -> None:
    target = JaxLogdensityTarget(_gaussian_logdensity, d=3)
    x = np.array([1.0, -2.0, 0.5])
    assert target.d == 3
    np.testing.assert_allclose(target.log_prob(x), float(_gaussian_logdensity(jnp.asarray(x))))
    np.testing.assert_allclose(target.grad_log_prob(x), -x / SCALES**2)
    assert target.jax_log_prob is _gaussian_logdensity


@pytest.mark.unit
def test_run_nutpie_without_extra_raises_informative_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes `import nutpie` raise ImportError, as if not installed.
    monkeypatch.setitem(sys.modules, "nutpie", None)
    assert not nutpie_available()
    with pytest.raises(CompetitorUnavailableError, match="competitors"):
        run_nutpie(_gaussian_logdensity, d=3, seed=0, tune=10, draws=10, chains=1, init=_init(1, 3))


@pytest.mark.unit
def test_run_nutpie_rejects_unknown_adaptation() -> None:
    with pytest.raises(ValueError, match="adaptation"):
        run_nutpie(_gaussian_logdensity, d=3, seed=0, tune=10, draws=10, chains=1, init=None, adaptation="dense")


@pytest.mark.unit
def test_init_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="init"):
        run_stock_nuts(_gaussian_logdensity, d=3, seed=0, tune=10, draws=10, chains=2, init=np.zeros((3, 3)))


# =============================================================================
# 2. BlackJAX reference arms (stock NUTS and the repo's arcsinh NUTS)
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("adapter", [run_stock_nuts, run_arcsinh_nuts])
def test_blackjax_arm_smoke_accepts_target_and_callable(adapter) -> None:
    chains, draws = 2, 30
    for logdensity in (DiagonalGaussian(d=3, stds=SCALES), _gaussian_logdensity):
        samples, telemetry = adapter(
            logdensity, d=3, seed=1, tune=40, draws=draws, chains=chains, init=_init(chains, 3)
        )
        assert samples.shape == (chains, draws, 3)
        assert np.all(np.isfinite(samples))
        _assert_standard_telemetry(telemetry, chains, draws)


@pytest.mark.unit
@pytest.mark.statistical
@pytest.mark.parametrize("adapter", [run_stock_nuts, run_arcsinh_nuts])
def test_blackjax_arm_gaussian_moments_within_mcse(adapter) -> None:
    samples, _ = adapter(_gaussian_logdensity, d=3, seed=3, tune=300, draws=1000, chains=4, init=_init(4, 3, seed=3))
    _assert_gaussian_moments_within_mcse(samples, SCALES)


# =============================================================================
# 3. nutpie arms (optional extra)
# =============================================================================


@requires_nutpie
@pytest.mark.unit
def test_nutpie_diag_smoke_and_exact_gradient_count() -> None:
    chains, draws = 2, 40
    samples, telemetry = run_nutpie_diag(
        _gaussian_logdensity, d=3, seed=5, tune=60, draws=draws, chains=chains, init=_init(chains, 3)
    )
    assert samples.shape == (chains, draws, 3)
    _assert_standard_telemetry(telemetry, chains, draws)
    assert telemetry["adaptation"] == "diag"
    assert telemetry["nutpie_version"]
    leapfrog = telemetry["leapfrog_steps_warmup"] + telemetry["leapfrog_steps_draws"]
    # Every leapfrog step evaluates the gradient once; the counted total adds the initial points
    # and step-size initialisation, a few evaluations per chain.
    assert leapfrog <= telemetry["total_grad_evals"] <= leapfrog + 20 * chains
    assert telemetry["draw_grad_evals"] == telemetry["leapfrog_steps_draws"]
    assert telemetry["flow_adapter_evals"] == 0


@requires_nutpie
@pytest.mark.unit
def test_nutpie_accepts_target_distribution() -> None:
    target = DiagonalGaussian(d=3, stds=SCALES)
    samples, telemetry = run_nutpie_diag(target, d=None, seed=2, tune=40, draws=20, chains=1, init=_init(1, 3))
    assert samples.shape == (1, 20, 3)
    assert telemetry["total_grad_evals"] > 0


@requires_nutpie
@pytest.mark.unit
def test_nutpie_starts_chains_from_given_positions() -> None:
    # The density is finite only within radius 10 of (50, 50, 50). nutpie's own initialisation
    # draws from Uniform(-2, 2), where the density is zero, so sampling can only succeed if the
    # chains start from the positions we hand over.
    def far_ball(x: jnp.ndarray) -> jnp.ndarray:
        r2 = jnp.sum((x - 50.0) ** 2)
        return jnp.where(r2 < 100.0, -0.5 * r2, -jnp.inf)

    init = 50.0 + _init(2, 3)
    samples, telemetry = run_nutpie_diag(far_ball, d=3, seed=0, tune=50, draws=20, chains=2, init=init)
    assert telemetry["init_positions_served"] == 2
    assert np.all(np.sum((samples - 50.0) ** 2, axis=-1) < 100.0)


@requires_nutpie
@pytest.mark.unit
@pytest.mark.statistical
def test_nutpie_diag_gaussian_moments_within_mcse() -> None:
    samples, telemetry = run_nutpie_diag(
        _gaussian_logdensity, d=3, seed=7, tune=300, draws=1000, chains=4, init=_init(4, 3, seed=7)
    )
    assert telemetry["divergences"] == 0
    _assert_gaussian_moments_within_mcse(samples, SCALES)


@requires_flow
@pytest.mark.unit
def test_nutpie_flow_counts_adapter_gradient_evaluations() -> None:
    chains, draws = 1, 40
    samples, telemetry = run_nutpie_flow(
        _gaussian_logdensity, d=3, seed=11, tune=150, draws=draws, chains=chains, init=_init(chains, 3)
    )
    assert samples.shape == (chains, draws, 3)
    _assert_standard_telemetry(telemetry, chains, draws)
    assert telemetry["adaptation"] == "flow"
    # Under flow adaptation nutpie evaluates the target inside the Python transform adapter, so
    # almost every gradient is counted there rather than in the Rust-side logp callback.
    assert telemetry["flow_adapter_evals"] > telemetry["logp_callback_evals"]
    leapfrog = telemetry["leapfrog_steps_warmup"] + telemetry["leapfrog_steps_draws"]
    assert leapfrog <= telemetry["total_grad_evals"] <= leapfrog + 50 * chains
    assert telemetry["flow_updates"] > 0
    assert telemetry["flow_training_seconds"] >= 0.0


@requires_flow
@pytest.mark.unit
@pytest.mark.statistical
def test_nutpie_flow_gaussian_moments_within_mcse() -> None:
    samples, _ = run_nutpie_flow(
        _gaussian_logdensity, d=3, seed=13, tune=300, draws=600, chains=2, init=_init(2, 3, seed=13)
    )
    _assert_gaussian_moments_within_mcse(samples, SCALES)
