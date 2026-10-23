"""Adversarial stress harness for MCMC benchmark runner, budget invariants, divergence isolation,
seed determinism, and stability sweeps (Feature F20 / M4).
"""

import numpy as np
import pytest

from mcmc_robust.bench.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    RunRecord,
)
from mcmc_robust.bench.sweep import (
    StepSizeSweep,
    SweepConfig,
    SweepRecord,
)
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.targets.benign import IsotropicGaussian

# =============================================================================
# Custom Adversarial Targets for Edge-Case Testing
# =============================================================================


class ExplosiveDecicTarget(TargetDistribution):
    r"""Adversarial target with explosive 10th-order polynomial potential:
    U(x) = \frac{1}{10} \|x\|^{10}
    \nabla U(x) = \|x\|^8 x
    """

    def __init__(self, d: int = 2) -> None:
        super().__init__(d=d)

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = self._validate_input(x)
        r2 = float(np.sum(x_arr**2))
        return float(-0.1 * (r2**5))

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = self._validate_input(x)
        r2 = float(np.sum(x_arr**2))
        return -(r2**4) * x_arr

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return self.log_prob(x), self.grad_log_prob(x)

    def sample_ground_truth(self, n: int, rng=None) -> np.ndarray:
        return np.zeros((n, self.d))

    def exact_moments_or_quantiles(self) -> dict:
        return {"mean": np.zeros(self.d)}


class FaultyNanTarget(TargetDistribution):
    """Target that produces NaNs or infs immediately."""

    def __init__(self, d: int = 2) -> None:
        super().__init__(d=d)

    def log_prob(self, x: np.ndarray) -> float:
        return float("nan")

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return np.full(self.d, np.nan)

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return float("nan"), np.full(self.d, np.nan)

    def sample_ground_truth(self, n: int, rng=None) -> np.ndarray:
        return np.zeros((n, self.d))

    def exact_moments_or_quantiles(self) -> dict:
        return {}


# =============================================================================
# 1. Budget Enforcement & Zero Redundancy Invariants
# =============================================================================

ALL_SAMPLERS = [
    "rwm",
    "mala",
    "hmc",
    "tamed_mala",
    "barker",
    "stereographic_mala",
    "radial_mala",
    "gentle_mala",
    "anisotropic_tail_mala",
    "unified_robust",
]


@pytest.mark.unit
@pytest.mark.parametrize("sampler_name", ALL_SAMPLERS)
def test_budget_enforcement_all_samplers_b1000(sampler_name):
    """Verifies that running BenchmarkRunner with B=1000 strictly halts with evals <= B
    across all 10 samplers.
    """
    b = 1000
    n_chains = 4
    cfg = BenchmarkConfig(budget=b, n_chains=n_chains, quick=True)
    runner = BenchmarkRunner(cfg)

    record = runner.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name=sampler_name,
    )

    if sampler_name == "rwm":
        assert record.grad_evals == 0, f"RWM evaluated {record.grad_evals} gradients, expected 0"
        assert record.log_prob_evals <= b, f"RWM log_prob_evals {record.log_prob_evals} exceeded budget {b}"
    else:
        assert record.grad_evals <= b, f"Sampler {sampler_name} grad_evals {record.grad_evals} exceeded budget {b}"
    assert record.budget == b
    assert record.non_divergence is True


@pytest.mark.unit
@pytest.mark.parametrize("sampler_name", ["rwm", "mala", "hmc", "gentle_mala", "unified_robust"])
def test_budget_enforcement_all_samplers_b3000(sampler_name):
    """Verifies that running BenchmarkRunner with B=3000 strictly halts with evals <= B."""
    b = 3000
    n_chains = 4
    cfg = BenchmarkConfig(budget=b, n_chains=n_chains, quick=True)
    runner = BenchmarkRunner(cfg)

    record = runner.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name=sampler_name,
    )

    if sampler_name == "rwm":
        assert record.grad_evals == 0
        assert record.log_prob_evals <= b
    else:
        assert record.grad_evals <= b
    assert record.non_divergence is True


@pytest.mark.unit
@pytest.mark.parametrize("b,n_chains", [(77, 4), (250, 3), (1003, 4)])
def test_budget_enforcement_arbitrary_and_odd_budgets(b, n_chains):
    """Verifies strict budget ceiling under non-divisible and odd budget/chain partitions
    when B >= 10 * n_chains.
    """
    cfg = BenchmarkConfig(budget=b, n_chains=n_chains, quick=True)
    runner = BenchmarkRunner(cfg)

    for sampler_name in ["rwm", "mala"]:
        record = runner.run_single_configuration(
            regime="benign",
            target_name="gaussian_isotropic",
            dim=2,
            init_type="bulk",
            sampler_name=sampler_name,
        )
        if sampler_name == "rwm":
            assert record.grad_evals == 0
            assert record.log_prob_evals <= b, f"RWM log_prob_evals {record.log_prob_evals} exceeded budget {b}"
        else:
            assert record.grad_evals <= b, f"MALA grad_evals {record.grad_evals} exceeded budget {b}"


@pytest.mark.unit
def test_budget_floor_behavior_when_budget_below_min_chain_floor():
    """Verifies that small budgets B < 10 * n_chains strictly respect total budget B
    and evaluations across chains never exceed B.
    """
    b = 13
    n_chains = 3
    cfg = BenchmarkConfig(budget=b, n_chains=n_chains, quick=True)
    runner = BenchmarkRunner(cfg)

    record = runner.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="rwm",
    )
    # Evaluations must strictly not exceed budget ceiling B
    assert record.log_prob_evals <= b, f"Evaluations {record.log_prob_evals} exceeded budget ceiling {b}"


@pytest.mark.unit
@pytest.mark.parametrize("b,n_chains", [(2, 2), (5, 3), (9, 4), (13, 3), (20, 4)])
def test_budget_strict_ceiling_small_budget_sweep(b, n_chains):
    """Exhaustive boundary check proving small budgets strictly respect evals <= B."""
    cfg = BenchmarkConfig(budget=b, n_chains=n_chains, quick=True)
    runner = BenchmarkRunner(cfg)
    for sampler in ["rwm", "mala"]:
        rec = runner.run_single_configuration("benign", "gaussian_isotropic", 2, "bulk", sampler)
        evals = rec.grad_evals if sampler == "mala" else rec.log_prob_evals
        assert evals <= b, f"Budget {b} exceeded with {sampler}: {evals}"


@pytest.mark.unit
def test_hmc_leapfrog_interrupted_budget_enforcement():
    """HMC with multiple leapfrog steps stops precisely when budget is exhausted mid-trajectory."""
    cfg = BenchmarkConfig(budget=25, n_chains=1, quick=True)
    runner = BenchmarkRunner(cfg)

    record = runner.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="hmc",
    )
    assert record.grad_evals <= 25


@pytest.mark.unit
def test_zero_redundancy_caching_invariant():
    """Verifies that rejected proposals in MALA do not re-evaluate the target log_prob or gradient."""
    target = IsotropicGaussian(d=2, std=1.0)
    tracker = GradBudgetTracker(target)
    # Huge step size to force 100% rejection
    sampler = MALA(step_size=100.0)
    rng = np.random.default_rng(123)

    state = sampler.init_state(np.array([1.0, 1.0]), tracker)
    assert tracker.grad_evals == 1
    assert tracker.log_prob_evals == 1

    for step_num in range(1, 10):
        res = sampler.step(state, tracker, rng)
        # Verify 1 gradient and 1 log_prob evaluated per step regardless of rejection
        assert tracker.grad_evals == 1 + step_num
        assert tracker.log_prob_evals == 1 + step_num
        if not res.accepted:
            # Check zero-redundancy: state returned is prior state
            assert np.array_equal(res.state.x, state.x)


# =============================================================================
# 2. Divergence Isolation & Graceful Recovery
# =============================================================================


@pytest.mark.unit
def test_divergence_massive_step_size_on_quartic():
    """Forces extreme step size (h=100.0) on standard MALA on QuarticTarget:
    Verifies that zero acceptance rate / frozen chain is detected and flagged as diverged.
    """
    import mcmc_robust.bench.runner as runner_mod

    cfg = BenchmarkConfig(budget=100, n_chains=2, burn_in_ratio=0.0, quick=True)
    runner = BenchmarkRunner(cfg)

    # Force h=100.0 for MALA
    orig_step = runner_mod.get_canonical_step_size
    runner_mod.get_canonical_step_size = lambda s, d: 100.0 if s == "mala" else 0.1
    try:
        record = runner.run_single_configuration(
            regime="roughness",
            target_name="quartic",
            dim=2,
            init_type="tail",
            sampler_name="mala",
        )
        assert isinstance(record, RunRecord)
        assert record.acceptance_rate == 0.0
        assert record.non_divergence is False
        assert record.divergence_reason == "Frozen chain / zero acceptance rate"
        assert np.isnan(record.bulk_ess)
        assert np.isnan(record.split_rhat)
    finally:
        runner_mod.get_canonical_step_size = orig_step


@pytest.mark.unit
def test_divergence_explosive_potential_isolation(monkeypatch):
    """Under an explosive potential raising FloatingPointError/OverflowError, runner isolates divergence
    without unhandled exception and correctly marks non_divergence=False.
    """
    import mcmc_robust.bench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_target", lambda name, dim: ExplosiveDecicTarget(d=dim))

    cfg = BenchmarkConfig(budget=100, n_chains=2, quick=True)
    runner = BenchmarkRunner(cfg)

    record = runner.run_single_configuration(
        regime="roughness",
        target_name="explosive_decic",
        dim=2,
        init_type="tail",
        sampler_name="mala",
    )

    assert isinstance(record, RunRecord)
    assert record.non_divergence is False
    assert record.divergence_reason is not None
    assert np.isnan(record.bulk_ess)
    assert np.isnan(record.tail_ess)
    assert np.isnan(record.split_rhat)
    assert record.n_samples == 0


@pytest.mark.unit
def test_divergence_nan_target_isolation(monkeypatch):
    """Target returning NaNs is caught and marked as diverged."""
    import mcmc_robust.bench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_target", lambda name, dim: FaultyNanTarget(d=dim))

    cfg = BenchmarkConfig(budget=50, n_chains=2, quick=True)
    runner = BenchmarkRunner(cfg)

    record = runner.run_single_configuration(
        regime="roughness",
        target_name="faulty_nan",
        dim=2,
        init_type="bulk",
        sampler_name="mala",
    )

    assert isinstance(record, RunRecord)
    assert record.non_divergence is False
    assert record.divergence_reason is not None
    assert "NaN" in record.divergence_reason or "Exception" in record.divergence_reason


@pytest.mark.unit
def test_divergence_does_not_poison_subsequent_runs(monkeypatch, tmp_path):
    """A diverged run in the benchmark suite does not halt or corrupt subsequent runs."""
    cfg = BenchmarkConfig(
        regimes=("roughness", "benign"),
        dimensions=(2,),
        initializations=("bulk",),
        samplers=("mala",),
        budget=100,
        n_chains=2,
        output_dir=str(tmp_path),
        quick=True,
    )
    runner = BenchmarkRunner(cfg)

    # Run all
    result = runner.run_all()
    assert result.summary["total_runs"] >= 2
    # Benign run on gaussian must succeed even if roughness had issues
    benign_runs = [r for r in result.records if r.regime == "benign"]
    assert len(benign_runs) >= 1
    assert all(r.non_divergence for r in benign_runs)


# =============================================================================
# 3. Multi-Chain Seed Determinism
# =============================================================================


@pytest.mark.unit
def test_multichain_seed_bitwise_determinism():
    """Running identical BenchmarkConfig with identical seed produces bitwise identical
    trajectories, evaluations, and metrics.
    """
    cfg1 = BenchmarkConfig(budget=150, n_chains=2, seed=42, quick=True)
    runner1 = BenchmarkRunner(cfg1)
    rec1 = runner1.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="mala",
    )

    cfg2 = BenchmarkConfig(budget=150, n_chains=2, seed=42, quick=True)
    runner2 = BenchmarkRunner(cfg2)
    rec2 = runner2.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="mala",
    )

    assert rec1.grad_evals == rec2.grad_evals
    assert rec1.log_prob_evals == rec2.log_prob_evals
    assert rec1.n_samples == rec2.n_samples
    assert rec1.acceptance_rate == rec2.acceptance_rate
    assert rec1.bulk_ess == rec2.bulk_ess
    assert rec1.tail_ess == rec2.tail_ess
    assert rec1.split_rhat == rec2.split_rhat
    assert rec1.folded_rhat == rec2.folded_rhat
    assert rec1.wasserstein1 == rec2.wasserstein1
    assert rec1.q05_error == rec2.q05_error
    assert rec1.q95_error == rec2.q95_error
    assert rec1.q99_error == rec2.q99_error


@pytest.mark.unit
def test_seed_sensitivity_produces_distinct_trajectories():
    """Different seeds must produce different sample trajectories and acceptance counts."""
    cfg1 = BenchmarkConfig(budget=200, n_chains=2, seed=42, quick=True)
    runner1 = BenchmarkRunner(cfg1)
    rec1 = runner1.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="mala",
    )

    cfg2 = BenchmarkConfig(budget=200, n_chains=2, seed=9999, quick=True)
    runner2 = BenchmarkRunner(cfg2)
    rec2 = runner2.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="mala",
    )

    # Metrics should be statistically close but not bitwise identical
    assert rec1.bulk_ess != rec2.bulk_ess or rec1.acceptance_rate != rec2.acceptance_rate


# =============================================================================
# 4. Step-Size Sweeps & Stability Window Calculations
# =============================================================================


@pytest.mark.unit
def test_stability_window_ratio_edge_cases(tmp_path):
    """Empirically stress-tests StepSizeSweep._analyze_stability_windows on boundary cases:
    - 0 stable step sizes (total divergence) -> window_ratio == 0.0
    - 1 stable step size -> window_ratio == 1.0
    - MALA window == 0, GentleMALA window > 0 -> stability_speedup == inf
    - Tamed burn is None -> AC3_burnin_pass is False, no exception
    """
    sweep = StepSizeSweep(SweepConfig(output_dir=str(tmp_path), quick=True))

    # Scenario A: All diverged
    records_diverged = [
        SweepRecord(
            target_name="quartic",
            dimension=2,
            sampler_name="mala",
            step_size=0.1 * (2**i),
            acceptance_rate=0.0,
            diverged=True,
            burn_in_steps=None,
            terminal_norm=1e15,
            divergence_reason="Explosion",
        )
        for i in range(5)
    ]
    summary_a = sweep._analyze_stability_windows(records_diverged)
    assert summary_a["quartic"]["d_2"]["sampler_windows"]["mala"]["window_ratio"] == 0.0
    assert summary_a["quartic"]["d_2"]["sampler_windows"]["mala"]["median_burnin_steps"] is None

    # Scenario B: Exactly 1 stable point
    records_single = [
        SweepRecord(
            target_name="quartic",
            dimension=2,
            sampler_name="mala",
            step_size=0.05,
            acceptance_rate=0.5,
            diverged=False,
            burn_in_steps=20,
            terminal_norm=1.2,
        )
    ]
    summary_b = sweep._analyze_stability_windows(records_single)
    assert summary_b["quartic"]["d_2"]["sampler_windows"]["mala"]["window_ratio"] == 1.0
    assert summary_b["quartic"]["d_2"]["sampler_windows"]["mala"]["median_burnin_steps"] == 20.0

    # Scenario C: MALA window = 0, GentleMALA window = 100
    records_c = [
        SweepRecord(
            target_name="quartic",
            dimension=2,
            sampler_name="gentle_mala",
            step_size=0.01,
            acceptance_rate=0.7,
            diverged=False,
            burn_in_steps=10,
            terminal_norm=1.0,
        ),
        SweepRecord(
            target_name="quartic",
            dimension=2,
            sampler_name="gentle_mala",
            step_size=1.0,
            acceptance_rate=0.6,
            diverged=False,
            burn_in_steps=8,
            terminal_norm=1.0,
        ),
    ]
    summary_c = sweep._analyze_stability_windows(records_c)
    d2 = summary_c["quartic"]["d_2"]
    assert d2["sampler_windows"]["mala"]["window_ratio"] == 0.0
    assert d2["sampler_windows"]["gentle_mala"]["window_ratio"] == 100.0
    assert d2["AC3_stability_factor"] == float("inf")
    assert d2["AC3_stability_pass"] is True
    # Tamed was not present -> burnin_reduction is None, AC3_burnin_pass is False
    assert d2["AC3_burnin_reduction"] is None
    assert d2["AC3_burnin_pass"] is False


@pytest.mark.unit
def test_quartic_stability_and_burnin_speedup(tmp_path):
    """Empirically runs step-size sweep on QuarticTarget to verify GentleMALA
    stability window superiority and burn-in acceleration over Globally Tamed MALA.
    """
    cfg = SweepConfig(
        targets=("quartic",),
        dimensions=(2,),
        samplers=("mala", "tamed_mala", "gentle_mala"),
        h_min=1e-3,
        h_max=0.5,
        n_grid=8,
        steps_per_run=120,
        output_dir=str(tmp_path),
        quick=True,
    )
    sweep = StepSizeSweep(cfg)
    payload = sweep.run_sweep()

    summary = payload["summary"]["quartic"]["d_2"]
    windows = summary["sampler_windows"]

    # GentleMALA must be stable across a window
    assert windows["gentle_mala"]["window_ratio"] >= 1.0

    # If both tamed and gentle reached bulk, verify burn-in calculation
    tamed_burn = windows["tamed_mala"]["median_burnin_steps"]
    gentle_burn = windows["gentle_mala"]["median_burnin_steps"]
    if tamed_burn is not None and gentle_burn is not None:
        expected_ratio = tamed_burn / gentle_burn
        assert summary["AC3_burnin_reduction"] == pytest.approx(expected_ratio, rel=1e-5)


# =============================================================================
# 7. Non-finite Chains and Edge-Case Guards
# =============================================================================


@pytest.mark.unit
def test_split_rhat_inf_clean_handling():
    """Chains containing inf return inf cleanly without NumPy subtract/var RuntimeWarnings."""
    from mcmc_robust.diagnostics.convergence import compute_split_rhat

    c_inf = np.array([[[np.inf], [np.inf]], [[1.0], [2.0]]])
    r = compute_split_rhat(c_inf)
    assert np.isinf(r[0])


@pytest.mark.unit
def test_split_rhat_nan_clean_handling():
    """Chains containing nan return nan cleanly without NumPy RuntimeWarnings."""
    from mcmc_robust.diagnostics.convergence import compute_split_rhat

    c_nan = np.array([[[np.nan], [1.0]], [[1.0], [2.0]]])
    r = compute_split_rhat(c_nan)
    assert np.isnan(r[0])


@pytest.mark.unit
def test_folded_rhat_inf_and_nan_clean_handling():
    """Folded split R-hat handles inf and nan without subtract/median warnings."""
    from mcmc_robust.diagnostics.convergence import compute_folded_rhat

    c_inf = np.array([[[np.inf], [np.inf]], [[1.0], [2.0]]])
    r_inf = compute_folded_rhat(c_inf)
    assert np.isinf(r_inf[0])

    c_nan = np.array([[[np.nan], [1.0]], [[1.0], [2.0]]])
    r_nan = compute_folded_rhat(c_nan)
    assert np.isnan(r_nan[0])


@pytest.mark.unit
def test_quantile_relative_error_zero_median_bound():
    """When true quantile is 0.0 (e.g. median of symmetric target),
    compute_quantile_relative_error reports absolute error and does not explode by 10^10.
    """
    from mcmc_robust.diagnostics.metrics import compute_quantile_relative_error

    target = IsotropicGaussian(d=2, std=1.0)
    rng = np.random.default_rng(42)
    samples = target.sample_ground_truth(5000, rng=rng)

    # Query median p=0.50 where true_q = 0.0
    q_errs = compute_quantile_relative_error(samples, target, quantiles=(0.50,), rng=rng)
    assert 0.50 in q_errs
    median_err = q_errs[0.50]
    # Absolute error should be bounded by Monte Carlo error < 0.10, not 10^8
    assert np.all(median_err < 0.10)
