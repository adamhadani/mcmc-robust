"""Unified 5-Benchmark Suite CLI Runner (Feature F29 / Deliverable R3).

Provides a single CLI interface to execute:
- Benchmark 1: Hyperparameter Sweeps (radius, damping, step size) -> sweep_results.{json,csv}
- Benchmark 2: Heavy-Tailed Target Regime -> benchmark_results.{json,csv}
- Benchmark 3: Hierarchical Funnel & Roughness Regime -> benchmark_results.{json,csv}
- Benchmark 4: Dual Pathology Regime (roughness + flatness) -> benchmark_results.{json,csv}
- Benchmark 5: External SOTA Head-to-Head (PyMC & BlackJAX) -> sota_comparison_results.{json,csv}
- All: Runs all 5 benchmarks sequentially.

Usage:
    python -m mcmc_robust.bench --benchmark {1,2,3,4,5,all} [--quick] [--plot] [--output-dir reports/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcmc_robust.bench.plot import generate_all_figures, plot_figure5_sota_nuts_comparison
from mcmc_robust.bench.runner import BenchmarkConfig, BenchmarkRunner
from mcmc_robust.bench.sota_runner import SOTABenchmarkConfig, SOTABenchmarkRunner
from mcmc_robust.bench.sweep import StepSizeSweep, SweepConfig


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the unified command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcmc_robust.bench",
        description="Unified 5-Benchmark Suite Runner for MCMC Robustness Research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Benchmark to execute: 1 (Sweeps), 2 (Heavy Tails), 3 (Funnels), 4 (Dual Pathology), 5 (SOTA Head-to-Head), or 'all'",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test mode with reduced samples and chains",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate publication-grade figures (Figures 1-5 in PNG and PDF formats)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for CSV, JSON, and figure artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility",
    )
    return parser


def run_benchmark_1(output_dir: Path, quick: bool, seed: int) -> None:
    """Executes Benchmark 1: Parameter Sweeps."""
    print("\n" + "=" * 70)
    print("Executing Benchmark 1: Hyperparameter Sweeps")
    print("=" * 70)
    targets = ("quartic",) if quick else ("quartic", "neals_funnel")
    dimensions = (2,) if quick else (2, 10, 50)
    samplers = ("mala", "gentle_mala") if quick else ("mala", "tamed_mala", "gentle_mala", "barker", "unified_robust")
    n_grid = 5 if quick else 25
    steps = 100 if quick else 500

    config = SweepConfig(
        targets=targets,
        dimensions=dimensions,
        samplers=samplers,
        n_grid=n_grid,
        steps_per_run=steps,
        seed=seed,
        output_dir=str(output_dir),
        quick=quick,
    )
    runner = StepSizeSweep(config)
    runner.run_sweep()
    print(f"Benchmark 1 complete. Saved to {output_dir / 'sweep_results.csv'}")


def run_regime_benchmark(
    regime: str,
    output_dir: Path,
    quick: bool,
    seed: int,
    benchmark_num: int,
) -> None:
    """Executes Benchmark 2, 3, or 4 (Target Regimes)."""
    print("\n" + "=" * 70)
    print(f"Executing Benchmark {benchmark_num}: Regime '{regime}'")
    print("=" * 70)

    dims = (2, 10) if quick else (2, 10, 50)
    budget = 150 if quick else 2000
    n_chains = 2 if quick else 4
    samplers = ("mala", "gentle_mala", "unified_robust") if quick else BenchmarkConfig.samplers

    config = BenchmarkConfig(
        regimes=(regime,),
        dimensions=dims,
        samplers=samplers,
        budget=budget,
        n_chains=n_chains,
        seed=seed,
        output_dir=str(output_dir),
        quick=quick,
    )
    runner = BenchmarkRunner(config)
    result = runner.run_all()
    print(
        f"Benchmark {benchmark_num} complete. Completed runs: {result.summary['completed_runs']}/{result.summary['total_runs']}"
    )


def run_benchmark_5(output_dir: Path, quick: bool, seed: int) -> None:
    """Executes Benchmark 5: External SOTA Head-to-Head."""
    print("\n" + "=" * 70)
    print("Executing Benchmark 5: External SOTA Head-to-Head (PyMC & BlackJAX)")
    print("=" * 70)

    n_samples = 25 if quick else 200
    n_warmup = 15 if quick else 100
    n_chains = 2

    config = SOTABenchmarkConfig(
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_chains=n_chains,
        seed=seed,
        output_dir=str(output_dir),
        quick=quick,
    )
    runner = SOTABenchmarkRunner(config=config)
    runner.run_all()
    print(f"Benchmark 5 complete. Saved to {output_dir / 'sota_comparison_results.csv'}")


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution logic."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    choice = args.benchmark
    quick = args.quick
    seed = args.seed

    if choice in ("1", "all"):
        run_benchmark_1(output_dir=out_path, quick=quick, seed=seed)

    if choice in ("2", "all"):
        run_regime_benchmark("heavy_tails", output_dir=out_path, quick=quick, seed=seed, benchmark_num=2)

    if choice in ("3", "all"):
        run_regime_benchmark("roughness", output_dir=out_path, quick=quick, seed=seed, benchmark_num=3)

    if choice in ("4", "all"):
        run_regime_benchmark("dual_pathology", output_dir=out_path, quick=quick, seed=seed, benchmark_num=4)

    if choice in ("5", "all"):
        run_benchmark_5(output_dir=out_path, quick=quick, seed=seed)

    if args.plot:
        print("\n" + "=" * 70)
        print("Generating Publication Figures (Figures 1-5)")
        print("=" * 70)
        fig_dir = out_path / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        generate_all_figures(
            results_path=out_path / "benchmark_results.json"
            if (out_path / "benchmark_results.json").exists()
            else None,
            sweep_path=out_path / "sweep_results.json" if (out_path / "sweep_results.json").exists() else None,
            out_dir=fig_dir,
            formats=("png", "pdf"),
        )
        # Ensure Figure 5 is explicitly rendered
        plot_figure5_sota_nuts_comparison(
            sota_path=out_path / "sota_comparison_results.csv"
            if (out_path / "sota_comparison_results.csv").exists()
            else None,
            out_dir=fig_dir,
            formats=("png", "pdf"),
        )
        print(f"Figures saved to {fig_dir}")

    print("\nBenchmark Suite Execution Finished Successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
