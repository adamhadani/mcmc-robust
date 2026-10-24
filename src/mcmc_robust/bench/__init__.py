"""Benchmark harness, stability sweeps, transformed adapters, and publication plotting."""

from typing import Any

__all__ = [
    # Adapters
    "TransformedSamplerAdapter",
    "make_stereographic_mala",
    "make_radial_mala",
    # Runner
    "BenchmarkConfig",
    "RunRecord",
    "BenchmarkResult",
    "BenchmarkRunner",
    # Applied PosteriorDB Runner
    "AppliedBenchmarkConfig",
    "AppliedRunRecord",
    "AppliedBenchmarkResult",
    "AppliedBenchmarkRunner",
    # Sweep
    "SweepConfig",
    "SweepRecord",
    "StepSizeSweep",
    # Plotting
    "generate_all_figures",
    "generate_mock_benchmark_results",
    "load_benchmark_data",
]


def __getattr__(name: str) -> Any:
    if name in ("TransformedSamplerAdapter", "make_stereographic_mala", "make_radial_mala"):
        from mcmc_robust.bench import adapters

        return getattr(adapters, name)
    if name in ("BenchmarkConfig", "RunRecord", "BenchmarkResult", "BenchmarkRunner"):
        from mcmc_robust.bench import runner

        return getattr(runner, name)
    if name in ("AppliedBenchmarkConfig", "AppliedRunRecord", "AppliedBenchmarkResult", "AppliedBenchmarkRunner"):
        from mcmc_robust.bench import applied_runner

        return getattr(applied_runner, name)
    if name in ("SweepConfig", "SweepRecord", "StepSizeSweep"):
        from mcmc_robust.bench import sweep

        return getattr(sweep, name)
    if name in ("generate_all_figures", "generate_mock_benchmark_results", "load_benchmark_data"):
        from mcmc_robust.bench import plot

        return getattr(plot, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
