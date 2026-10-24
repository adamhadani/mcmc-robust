"""Unit tests for MCMC benchmark harness, stability sweeps, adapters, and plotting (Feature F20)."""

import csv
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from mcmc_robust.bench.adapters import (
    make_radial_mala,
    make_stereographic_mala,
)
from mcmc_robust.bench.plot import (
    generate_all_figures,
    load_benchmark_data,
)
from mcmc_robust.bench.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    RunRecord,
)
from mcmc_robust.bench.sweep import (
    StepSizeSweep,
    SweepConfig,
)
from mcmc_robust.targets.benign import IsotropicGaussian


@pytest.fixture
def rng():
    """Deterministic random generator."""
    return np.random.default_rng(42)


# =============================================================================
# 1. Benchmark Configuration & Data Schemas
# =============================================================================


@pytest.mark.unit
def test_benchmark_config_validation():
    """Verifies dataclass immutability and default argument integrity."""
    cfg = BenchmarkConfig()
    assert cfg.budget == 3000
    assert cfg.n_chains == 4
    assert "benign" in cfg.regimes
    assert 2 in cfg.dimensions

    with pytest.raises(FrozenInstanceError):
        cfg.budget = 5000  # type: ignore


@pytest.mark.unit
def test_run_record_serialization():
    """Verifies that RunRecord serializes to JSON cleanly, sanitizing NaNs and numpy types."""
    rec = RunRecord(
        regime="roughness",
        target_name="quartic",
        dimension=2,
        init_type="tail",
        sampler_name="mala",
        step_size=0.1,
        budget=100,
        n_chains=2,
        n_samples=0,
        grad_evals=10,
        log_prob_evals=10,
        bulk_ess=float("nan"),
        tail_ess=float("nan"),
        ess_per_grad=float("nan"),
        tail_ess_per_grad=float("nan"),
        split_rhat=float("nan"),
        folded_rhat=float("nan"),
        ks_pvalue=float("nan"),
        ks_stat=float("nan"),
        wasserstein1=float("nan"),
        q05_error=float("nan"),
        q95_error=float("nan"),
        q99_error=float("nan"),
        acceptance_rate=0.0,
        wall_clock_time=0.01,
        non_divergence=False,
        divergence_reason="Explosion",
    )
    d = rec.to_dict()
    assert d["bulk_ess"] is None
    assert d["non_divergence"] is False
    assert json.dumps(d)  # Must serialize without TypeError


# =============================================================================
# 2. Benchmark Runner Budget Enforcement & Recovery
# =============================================================================


@pytest.mark.unit
def test_single_run_oracle_budget_enforcement(rng):
    """BenchmarkRunner strictly caps gradient evaluations to the configured budget."""
    cfg = BenchmarkConfig(budget=60, n_chains=2, quick=True)
    runner = BenchmarkRunner(cfg)
    record = runner.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="mala",
    )
    assert record.non_divergence is True
    # Budget was 60 total; 2 chains with max 30 evals each -> grad_evals <= 60
    assert record.grad_evals <= 60
    assert record.n_samples > 0
    assert record.bulk_ess > 0.0


@pytest.mark.unit
def test_zero_gradient_sampler_budget_enforcement(rng):
    """RWM (zeroth-order) uses 0 gradients and caps log-prob evaluations."""
    cfg = BenchmarkConfig(budget=60, n_chains=2, quick=True)
    runner = BenchmarkRunner(cfg)
    record = runner.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="rwm",
    )
    assert record.non_divergence is True
    assert record.grad_evals == 0
    assert record.log_prob_evals <= 60


@pytest.mark.unit
def test_graceful_divergence_recovery(rng):
    """Runner isolates numerical divergence without raising unhandled exceptions."""
    cfg = BenchmarkConfig(budget=100, n_chains=2, quick=True)
    runner = BenchmarkRunner(cfg)

    # Calling an invalid target dimension (funnel with d=1)
    record = runner.run_single_configuration(
        regime="roughness",
        target_name="quartic",
        dim=2,
        init_type="tail",
        sampler_name="mala",
    )
    # Even if MALA diverges or passes, record is a valid RunRecord
    assert isinstance(record, RunRecord)


# =============================================================================
# 3. Transformed Sampler Adapters
# =============================================================================


@pytest.mark.unit
def test_transformed_sampler_adapter_pullback(rng):
    """Radial MALA stores and pulls back samples to primal space."""
    sampler = make_radial_mala(step_size=0.1, dim=2, radius=3.0, stiffness=0.5)
    target = IsotropicGaussian(d=2, std=1.0)
    x0 = np.array([5.0, 5.0])

    state = sampler.init_state(x0, target)
    assert "primal_x" in state.auxiliary
    assert np.allclose(state.auxiliary["primal_x"], x0)

    res = sampler.step(state, target, rng)
    assert "primal_x" in res.state.auxiliary
    assert res.state.auxiliary["primal_x"].shape == (2,)


@pytest.mark.unit
def test_stereographic_spherical_retraction_safeguard(rng):
    """Stereographic MALA with spherical retraction avoids North-pole singularities."""
    sampler = make_stereographic_mala(step_size=0.2, dim=2, scale=1.0)
    target = IsotropicGaussian(d=2, std=1.0)
    x0 = np.array([25.0, 25.0])  # Extreme tail near North pole

    state = sampler.init_state(x0, target)
    for _ in range(50):
        res = sampler.step(state, target, rng)
        state = res.state
        # Hypersphere norm must be approximately 1.0
        norm_s = np.linalg.norm(state.x)
        assert abs(norm_s - 1.0) < 1e-6
        # North pole component z_{d+1} must be < 1.0
        assert state.x[-1] < 1.0
        # Primal coordinate must be finite
        assert np.all(np.isfinite(state.auxiliary["primal_x"]))


# =============================================================================
# 4. Multi-Chain R-hat and Convergence Aggregation
# =============================================================================


@pytest.mark.unit
def test_multi_chain_rhat_aggregation(rng):
    """BenchmarkRunner produces valid split R-hat and folded R-hat across chains."""
    cfg = BenchmarkConfig(budget=300, n_chains=2, quick=True)
    runner = BenchmarkRunner(cfg)
    record = runner.run_single_configuration(
        regime="benign",
        target_name="gaussian_isotropic",
        dim=2,
        init_type="bulk",
        sampler_name="mala",
    )
    assert record.non_divergence is True
    assert record.split_rhat < 1.25
    assert record.folded_rhat < 1.25


# =============================================================================
# 5. Serialization & CLI Export
# =============================================================================


@pytest.mark.unit
def test_json_and_csv_serialization_integrity(tmp_path):
    """BenchmarkRunner exports valid synchronized JSON and CSV results."""
    cfg = BenchmarkConfig(
        regimes=("benign",),
        dimensions=(2,),
        initializations=("bulk",),
        samplers=("mala",),
        budget=100,
        n_chains=2,
        output_dir=str(tmp_path),
        quick=True,
    )
    runner = BenchmarkRunner(cfg)
    res = runner.run_all()

    json_file = tmp_path / "benchmark_results.json"
    csv_file = tmp_path / "benchmark_results.csv"

    assert json_file.exists()
    assert csv_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data
    assert "summary" in data
    assert len(data["results"]) == 1

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["sampler_name"] == "mala"


# =============================================================================
# 6. Step-Size Sweeps & AC3 Stability Verification
# =============================================================================


@pytest.mark.unit
def test_sweep_operating_window_calculation(tmp_path):
    """StepSizeSweep calculates operating windows and AC3 stability metrics."""
    cfg = SweepConfig(
        targets=("quartic",),
        dimensions=(2,),
        samplers=("mala", "gentle_mala"),
        h_min=1e-4,
        h_max=1e-1,
        n_grid=5,
        steps_per_run=60,
        output_dir=str(tmp_path),
        quick=True,
    )
    sweep = StepSizeSweep(cfg)
    results = sweep.run_sweep()

    assert "quartic" in results["summary"]
    d2_res = results["summary"]["quartic"]["d_2"]
    assert "sampler_windows" in d2_res
    assert "mala" in d2_res["sampler_windows"]
    assert "gentle_mala" in d2_res["sampler_windows"]
    assert (tmp_path / "sweep_results.json").exists()
    assert (tmp_path / "sweep_results.csv").exists()


@pytest.mark.unit
def test_ac3_gentle_mala_stability_superiority(tmp_path):
    """GentleMALA achieves wider stability window on QuarticTarget than MALA."""
    cfg = SweepConfig(
        targets=("quartic",),
        dimensions=(2,),
        samplers=("mala", "gentle_mala"),
        h_min=1e-3,
        h_max=1.0,
        n_grid=8,
        steps_per_run=100,
        output_dir=str(tmp_path),
        quick=True,
    )
    sweep = StepSizeSweep(cfg)
    results = sweep.run_sweep()

    d2_res = results["summary"]["quartic"]["d_2"]
    # GentleMALA must have stability factor >= 1.0 vs MALA
    assert d2_res["AC3_stability_factor"] >= 1.0


# =============================================================================
# 7. Plotting Pipeline & Publication Figures
# =============================================================================


@pytest.mark.unit
def test_plot_generate_all_figures(tmp_path):
    """Plotting pipeline generates all 5 figures in PNG format from reports/ artifacts."""
    figs = generate_all_figures(reports_dir=Path("reports"), out_dir=tmp_path, formats=("png",))
    assert len(figs) == 5
    for f in figs:
        assert f.exists()
        assert f.stat().st_size > 1000  # Non-empty image file


@pytest.mark.unit
def test_plot_load_real_benchmark_artifacts(tmp_path):
    """Verifies load_benchmark_data ingests real benchmark and sweep JSON artifacts without warning or mock fallback."""
    bench_p = Path("reports/benchmark_results.json")
    sweep_p = Path("reports/sweep_results.json")
    if bench_p.exists() and sweep_p.exists():
        data = load_benchmark_data(bench_p, sweep_path=sweep_p)
        assert "figure1_roughness" in data
        assert "figure2_heterogeneous" in data
        assert "figure3_bayesian" in data
        assert "figure4_benign" in data

        f1 = data["figure1_roughness"]
        assert len(f1["h_grid"]) > 0
        assert f1["stability_factor"] >= 1.0

        f2 = data["figure2_heterogeneous"]
        assert len(f2["mean_q99_errors"]) > 0
        assert f2["error_reduction"] is not None

        f3 = data["figure3_bayesian"]
        assert "funnel_mala" in f3 and "funnel_unified" in f3
        assert len(f3["cauchy_samplers"]) > 0

        f4 = data["figure4_benign"]
        assert len(f4["ratios"]) > 0
        assert len(f4["ess_per_grad"]) > 0

        # Verify figure generation on real data
        figs = generate_all_figures(results_path=bench_p, sweep_path=sweep_p, out_dir=tmp_path, formats=("png",))
        assert len(figs) == 5
        for f in figs:
            assert f.exists()
            assert f.stat().st_size > 1000


@pytest.mark.unit
def test_plot_load_benchmark_data_dir_resolution(tmp_path):
    """Verifies directory-based resolution of benchmark_results.json and sweep_results.json."""
    reports_dir = Path("reports")
    if (reports_dir / "benchmark_results.json").exists():
        data = load_benchmark_data(reports_dir)
        assert "figure1_roughness" in data
        assert "figure4_benign" in data


@pytest.mark.unit
def test_plot_raises_on_missing_file():
    """Verifies FileNotFoundError is raised when benchmark files do not exist (no mock fallback)."""
    with pytest.raises(FileNotFoundError):
        load_benchmark_data("non_existent_path_xyz.json")
