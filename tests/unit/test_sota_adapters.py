"""Unit tests for External SOTA Adapters and Benchmark Runner (PyMC, BlackJAX, ArviZ 1.3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mcmc_robust.bench.sota_adapters import (
    EightSchoolsTarget,
    RobustOutlierRegressionTarget,
    SOTATargetAdapter,
    SparseSignalRegressionTarget,
    evaluate_chains_with_arviz,
    run_blackjax_sampler,
    run_mcmc_robust_sampler,
    run_pymc_sampler,
)
from mcmc_robust.bench.sota_runner import (
    SOTABenchmarkConfig,
    SOTABenchmarkRunner,
)
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.targets.roughness import QuarticTarget


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# =============================================================================
# 1. Target Adapters & Analytical Gradient Verification vs JAX Autodiff
# =============================================================================


@pytest.mark.unit
def test_eight_schools_target_dimension_and_data():
    """Verify EightSchoolsTarget parameter dimension and data fixtures."""
    target = EightSchoolsTarget()
    assert target.d == 10
    assert len(target.y) == 8
    assert len(target.sigma) == 8
    assert np.allclose(target.y, [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])

    # Log prob at nominal parameters
    x0 = np.zeros(10, dtype=np.float64)
    lp0 = target.log_prob(x0)
    assert np.isfinite(lp0)

    # Gradient matches finite values
    grad0 = target.grad_log_prob(x0)
    assert grad0.shape == (10,)
    assert np.all(np.isfinite(grad0))


@pytest.mark.unit
def test_eight_schools_gradient_autodiff_agreement(rng):
    """Verify EightSchoolsTarget analytical gradients agree with JAX autodiff."""
    target = EightSchoolsTarget()
    adapter = SOTATargetAdapter(target)
    jax_logp_fn = adapter.jax_logdensity_fn()
    jax_grad_fn = jax.grad(jax_logp_fn)

    # Test at multiple points
    test_points = [
        np.zeros(10, dtype=np.float64),
        np.array([5.0, 1.0] + [2.0] * 8, dtype=np.float64),
        rng.normal(0.0, 1.5, size=10).astype(np.float64),
        rng.normal(0.0, 2.0, size=10).astype(np.float64),
    ]

    for pt in test_points:
        analytical_grad = target.grad_log_prob(pt)
        jax_grad = np.array(jax_grad_fn(jnp.asarray(pt)), dtype=np.float64)
        assert np.allclose(analytical_grad, jax_grad, atol=1e-6, rtol=1e-5), (
            f"Gradient mismatch at {pt}:\nAnalytical: {analytical_grad}\nJAX autodiff: {jax_grad}"
        )


@pytest.mark.unit
def test_eight_schools_pymc_model_compilation():
    """Verify EightSchoolsTarget PyMC model builds valid graph with 'x'."""
    target = EightSchoolsTarget()
    adapter = SOTATargetAdapter(target)
    model = adapter.pymc_model()
    assert "mu" in model.named_vars
    assert "tau" in model.named_vars
    assert "theta" in model.named_vars
    assert "x" in model.named_vars


@pytest.mark.unit
def test_sparse_signal_target_gradient_and_extreme_stability(rng):
    """Verify SparseSignalRegressionTarget gradients and stability in extreme tails."""
    target = SparseSignalRegressionTarget(n_obs=30, d=5, noise_std=1.0, prior_scale=1.0, seed=123)
    assert target.d == 5

    adapter = SOTATargetAdapter(target)
    jax_grad_fn = jax.grad(adapter.jax_logdensity_fn())

    test_points = [
        np.zeros(5, dtype=np.float64),
        rng.normal(0.0, 1.0, size=5).astype(np.float64),
        np.array([10.0, -5.0, 0.5, -2.0, 3.0], dtype=np.float64),
    ]

    for pt in test_points:
        analytical_grad = target.grad_log_prob(pt)
        jax_grad = np.array(jax_grad_fn(jnp.asarray(pt)), dtype=np.float64)
        assert np.allclose(analytical_grad, jax_grad, atol=1e-6, rtol=1e-5)

    # Test extreme tail stability without overflow warnings or NaNs
    extreme_pt = np.array([1e150, -1e150, 0.0, 1e50, -1e50], dtype=np.float64)
    lp_ext = target.log_prob(extreme_pt)
    assert np.isfinite(lp_ext)
    grad_ext = target.grad_log_prob(extreme_pt)
    assert np.all(np.isfinite(grad_ext))


@pytest.mark.unit
def test_robust_outlier_target_gradient_and_extreme_stability(rng):
    """Verify RobustOutlierRegressionTarget gradients and stability in extreme tails."""
    target = RobustOutlierRegressionTarget(
        n_obs=40, d=5, nu=4.0, scale=1.0, outlier_fraction=0.1, prior_scale=1.0, seed=456
    )
    assert target.d == 5

    adapter = SOTATargetAdapter(target)
    jax_grad_fn = jax.grad(adapter.jax_logdensity_fn())

    test_points = [
        np.zeros(5, dtype=np.float64),
        rng.normal(0.0, 1.0, size=5).astype(np.float64),
        np.array([2.0, -1.0, 0.0, 4.0, -3.0], dtype=np.float64),
    ]

    for pt in test_points:
        analytical_grad = target.grad_log_prob(pt)
        jax_grad = np.array(jax_grad_fn(jnp.asarray(pt)), dtype=np.float64)
        assert np.allclose(analytical_grad, jax_grad, atol=1e-6, rtol=1e-5)

    # Extreme tail test
    extreme_pt = np.array([1e150, -1e150, 1.0, -1.0, 1e60], dtype=np.float64)
    lp_ext = target.log_prob(extreme_pt)
    assert np.isfinite(lp_ext)
    grad_ext = target.grad_log_prob(extreme_pt)
    assert np.all(np.isfinite(grad_ext))


# =============================================================================
# 2. Sampler Runner Execution & Gradient Counting
# =============================================================================


@pytest.mark.unit
def test_run_blackjax_nuts_and_hmc():
    """Verify BlackJAX NUTS and HMC multi-chain runners and gradient tracking."""
    target = QuarticTarget(d=3)

    # 1. BlackJAX NUTS
    samples_nuts, info_nuts = run_blackjax_sampler(
        target=target,
        sampler_type="nuts",
        n_chains=2,
        n_samples=20,
        n_warmup=10,
        step_size=0.1,
        seed=10,
    )
    assert samples_nuts.shape == (2, 20, 3)
    assert info_nuts["total_grad_evals"] > 0
    assert info_nuts["divergences"] >= 0
    assert info_nuts["wall_clock_time"] > 0.0
    assert np.all(np.isfinite(samples_nuts))

    # 2. BlackJAX HMC
    samples_hmc, info_hmc = run_blackjax_sampler(
        target=target,
        sampler_type="hmc",
        n_chains=2,
        n_samples=20,
        n_warmup=10,
        step_size=0.1,
        seed=11,
        num_integration_steps=5,
    )
    assert samples_hmc.shape == (2, 20, 3)
    # HMC grads: 2 chains * (20 + 10) steps * 5 leapfrog = 300
    assert info_hmc["total_grad_evals"] == 2 * 30 * 5
    assert info_hmc["divergences"] >= 0
    assert info_hmc["wall_clock_time"] > 0.0


@pytest.mark.unit
def test_run_blackjax_invalid_algorithm():
    """Verify invalid algorithm raises ValueError."""
    target = QuarticTarget(d=2)
    with pytest.raises(ValueError, match="Unsupported BlackJAX sampler_type"):
        run_blackjax_sampler(
            target=target,
            sampler_type="unsupported_algo",
            n_chains=1,
            n_samples=10,
            n_warmup=5,
        )


@pytest.mark.unit
def test_run_pymc_sampler():
    """Verify PyMC NUTS runner and gradient accounting."""
    target = SparseSignalRegressionTarget(n_obs=20, d=3, seed=77)

    samples, info = run_pymc_sampler(
        target=target,
        n_chains=2,
        n_samples=20,
        n_warmup=10,
        seed=42,
    )
    assert samples.shape == (2, 20, 3)
    assert info["total_grad_evals"] > 0
    assert info["divergences"] >= 0
    assert info["wall_clock_time"] > 0.0
    assert np.all(np.isfinite(samples))


@pytest.mark.unit
def test_run_mcmc_robust_sampler():
    """Verify mcmc_robust sampler runner and GradBudgetTracker accounting."""
    target = QuarticTarget(d=3)
    init_states = [np.ones(3) * 0.5, -np.ones(3) * 0.5]

    samples, info = run_mcmc_robust_sampler(
        sampler_cls=GentleMALA,
        target=target,
        n_chains=2,
        n_samples=25,
        n_warmup=10,
        x0=init_states,
        sampler_kwargs={"step_size": 0.1},
        seed=99,
    )
    assert samples.shape == (2, 25, 3)
    # 2 chains * (25 + 10) proposals + 2 initial evals = 72
    assert info["total_grad_evals"] == 2 * (25 + 10) + 2
    assert info["divergences"] == 0
    assert info["wall_clock_time"] > 0.0
    assert np.all(np.isfinite(samples))


# =============================================================================
# 3. ArviZ 1.3 Metric Evaluation
# =============================================================================


@pytest.mark.unit
def test_evaluate_chains_with_arviz(rng):
    """Verify ArviZ 1.3 metrics extraction (bulk-ESS, tail-ESS, split-Rhat)."""
    # 4 chains of 100 samples from N(0, 1) in 3 dimensions
    samples = rng.normal(0.0, 1.0, size=(4, 100, 3)).astype(np.float64)
    target = QuarticTarget(d=3)

    metrics = evaluate_chains_with_arviz(
        samples=samples,
        total_grad_evals=1000,
        target=target,
        divergences=0,
        wall_clock_time=0.5,
    )

    assert "bulk_ess" in metrics
    assert "tail_ess" in metrics
    assert "split_rhat" in metrics
    assert "ess_per_grad" in metrics
    assert "tail_ess_per_grad" in metrics
    assert "q99_rel_error" in metrics
    assert "divergence_rate" in metrics

    assert metrics["bulk_ess"] > 50.0
    assert metrics["tail_ess"] > 50.0
    assert 0.9 <= metrics["split_rhat"] <= 1.2
    assert metrics["ess_per_grad"] > 0.0
    assert metrics["tail_ess_per_grad"] > 0.0
    assert metrics["divergence_rate"] == 0.0
    assert isinstance(metrics["q99_rel_error"], float)


# =============================================================================
# 4. SOTA Benchmark Runner End-to-End Smoke Test
# =============================================================================


@pytest.mark.unit
def test_sota_benchmark_runner_quick_smoke(tmp_path: Path):
    """Verify SOTABenchmarkRunner executes a quick smoke test and saves JSON/CSV."""
    config = SOTABenchmarkConfig(
        n_samples=20,
        n_warmup=15,
        n_chains=2,
        seed=101,
        output_dir=str(tmp_path),
        quick=True,
        tasks=["task1_heavy_tailed_regression"],
    )

    runner = SOTABenchmarkRunner(config=config)
    res = runner.run_all()

    assert "summary" in res
    assert "records" in res
    assert len(res["records"]) > 0

    json_path = tmp_path / "sota_comparison_results.json"
    csv_path = tmp_path / "sota_comparison_results.csv"

    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        assert "guardrails" in payload["summary"]
        assert len(payload["results"]) == len(res["records"])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == len(res["records"])
        assert "task_id" in rows[0]
        assert "sampler_name" in rows[0]
        assert "bulk_ess" in rows[0]
        assert "tail_ess" in rows[0]
