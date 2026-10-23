"""Tier 5 Adversarial Tests: Diagnostics, Oracle Tracking, and Benchmark Serialization.

Verifies statistical diagnostics on pathological chains (zero variance, NaNs, Infs,
short chains), budget enforcement guarantees, and runner serialization.
"""

import numpy as np
import pytest

from mcmc_robust.bench.runner import BenchmarkConfig, BenchmarkRunner
from mcmc_robust.bench.sweep import StepSizeSweep, SweepConfig
from mcmc_robust.diagnostics.convergence import compute_folded_rhat, compute_split_rhat
from mcmc_robust.diagnostics.ess import compute_bulk_ess, compute_tail_ess
from mcmc_robust.diagnostics.metrics import (
    compute_quantile_relative_error,
)
from mcmc_robust.diagnostics.tracker import BudgetExceededError, GradBudgetTracker
from mcmc_robust.targets.benign import IsotropicGaussian


class TestDiagnosticsDegenerateChains:
    """Verifies that diagnostic functions do not crash on pathological MCMC outputs."""

    def test_bulk_ess_and_tail_ess_constant_chain(self) -> None:
        # Constant chains with zero variance
        const_chain = np.full((4, 100, 3), 2.5)
        bulk = compute_bulk_ess(const_chain)
        assert np.all(bulk == 0.0)

        tail = compute_tail_ess(const_chain)
        assert np.all(tail == 0.0)

    def test_bulk_ess_and_tail_ess_nan_chain(self) -> None:
        nan_chain = np.full((2, 50, 2), np.nan)
        bulk = compute_bulk_ess(nan_chain)
        assert np.all(np.isnan(bulk))

        tail = compute_tail_ess(nan_chain)
        assert np.all(np.isnan(tail))

    def test_split_rhat_and_folded_rhat_extremes(self) -> None:
        # 1. Single chain / short chain fallback
        single_chain = np.ones((1, 5, 2))
        rhat_single = compute_split_rhat(single_chain)
        assert np.all(rhat_single == 1.0)

        # 2. Identical constant chains across chains
        ident_chains = np.full((4, 50, 2), 3.0)
        rhat_ident = compute_split_rhat(ident_chains)
        assert np.all(rhat_ident == 1.0)

        # 3. Separated constant chains (stuck chains)
        sep_chains = np.zeros((2, 50, 2))
        sep_chains[0, :, :] = 1.0
        sep_chains[1, :, :] = 10.0
        rhat_sep = compute_split_rhat(sep_chains)
        assert np.all(np.isinf(rhat_sep))

        # 4. Folded R-hat on divergent scale
        folded = compute_folded_rhat(sep_chains)
        assert np.all(np.isfinite(folded) | np.isinf(folded))

    def test_metrics_zero_quantile_guard(self) -> None:
        """Verifies that relative quantile error does not divide by zero for median=0."""
        d = 2
        target = IsotropicGaussian(d=d)
        rng = np.random.default_rng(42)
        samples = target.sample_ground_truth(1000, rng=rng)

        # 50th percentile (median) is exactly 0.0 for zero-mean Gaussian
        errs = compute_quantile_relative_error(samples, target, quantiles=[0.50])
        assert 0.50 in errs
        assert np.all(np.isfinite(errs[0.50]))
        # Value should be small absolute error, not inf or nan
        assert np.all(errs[0.50] < 0.2)


class TestBudgetTrackerAdversarial:
    """Verifies strict budget ceiling enforcement."""

    def test_budget_exceeded_immediately(self) -> None:
        target = IsotropicGaussian(d=2)
        tracker = GradBudgetTracker(target, max_grad_evals=1)

        x0 = np.array([0.0, 0.0])
        # First grad eval consumes budget
        tracker.grad_log_prob(x0)
        assert tracker.is_exhausted

        # Second grad eval must raise BudgetExceededError
        with pytest.raises(BudgetExceededError, match="Gradient budget exceeded"):
            tracker.grad_log_prob(x0)

    def test_joint_eval_budget_accounting(self) -> None:
        target = IsotropicGaussian(d=2)
        tracker = GradBudgetTracker(target, max_grad_evals=2, max_log_prob_evals=2)

        x0 = np.array([0.0, 0.0])
        lp, g = tracker.log_prob_and_grad(x0)
        assert tracker.n_grad == 1
        assert tracker.n_log_prob == 1
        assert tracker.n_joint == 1

        tracker.log_prob_and_grad(x0)
        assert tracker.is_exhausted

        with pytest.raises(BudgetExceededError):
            tracker.log_prob_and_grad(x0)


class TestBenchmarkExecutionAdversarial:
    """Verifies BenchmarkRunner and StepSizeSweep on extreme configurations."""

    def test_benchmark_runner_quick_smoke(self, tmp_path) -> None:
        config = BenchmarkConfig(
            regimes=("benign",),
            dimensions=(2,),
            samplers=("mala", "unified_robust"),
            budget=50,
            n_chains=2,
            output_dir=str(tmp_path),
            quick=True,
        )
        runner = BenchmarkRunner(config)
        res = runner.run_all()

        assert res.summary["total_runs"] > 0
        assert res.summary["completed_runs"] > 0
        assert (tmp_path / "benchmark_results.json").exists()
        assert (tmp_path / "benchmark_results.csv").exists()

    def test_step_size_sweep_quick_smoke(self, tmp_path) -> None:
        config = SweepConfig(
            targets=("quartic",),
            dimensions=(2,),
            samplers=("gentle_mala", "unified_robust"),
            n_grid=3,
            steps_per_run=20,
            output_dir=str(tmp_path),
            quick=True,
        )
        sweep = StepSizeSweep(config)
        res = sweep.run_sweep()

        assert "summary" in res
        assert (tmp_path / "sweep_results.json").exists()
        assert (tmp_path / "sweep_results.csv").exists()
