"""Automated MCMC benchmark harness (Feature F20)."""

import argparse
import csv
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from mcmc_robust.bench.adapters import (
    make_radial_mala,
    make_stereographic_mala,
)
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.diagnostics.convergence import (
    compute_folded_rhat,
    compute_split_rhat,
)
from mcmc_robust.diagnostics.ess import (
    compute_bulk_ess,
    compute_tail_ess,
)
from mcmc_robust.diagnostics.metrics import (
    compute_ks_test,
    compute_quantile_relative_error,
    compute_wasserstein1,
)
from mcmc_robust.diagnostics.tracker import (
    BudgetExceededError,
    GradBudgetTracker,
)
from mcmc_robust.samplers.barker import BarkerLangevin
from mcmc_robust.samplers.hmc import HMC
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.novel.anisotropic_tail import AnisotropicTailMALA
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.samplers.rwm import RandomWalkMetropolis
from mcmc_robust.samplers.tamed_mala import GloballyTamedMALA
from mcmc_robust.targets.suite import get_target


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration specification for benchmark execution."""

    regimes: Sequence[str] = ("benign", "roughness", "heavy_tails", "dual_pathology")
    dimensions: Sequence[int] = (2, 10, 50, 100)
    initializations: Sequence[str] = ("bulk", "tail")
    samplers: Sequence[str] = (
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
    )
    budget: int = 3000  # Total evaluation budget per run
    n_chains: int = 4  # Number of independent chains per run
    burn_in_ratio: float = 0.20  # Fraction of budget discarded as warmup
    seed: int = 42  # Master random seed for reproducibility
    output_dir: str = "reports"  # Output directory for JSON/CSV results
    quick: bool = False  # Quick execution override flag


@dataclass
class RunRecord:
    """Standardized metric record for a single sampler-target evaluation configuration."""

    regime: str
    target_name: str
    dimension: int
    init_type: str
    sampler_name: str
    step_size: float
    budget: int
    n_chains: int
    n_samples: int
    grad_evals: int
    log_prob_evals: int
    bulk_ess: float
    tail_ess: float
    ess_per_grad: float
    tail_ess_per_grad: float
    split_rhat: float
    folded_rhat: float
    ks_pvalue: float
    ks_stat: float
    wasserstein1: float
    q05_error: float
    q95_error: float
    q99_error: float
    acceptance_rate: float
    wall_clock_time: float
    non_divergence: bool
    divergence_reason: str | None = None
    extra_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts record to JSON-serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.bool_, bool)):
                d[k] = bool(v)
            elif isinstance(v, (np.floating, float)):
                d[k] = float(v) if np.isfinite(v) else None
            elif isinstance(v, (np.integer, int)):
                d[k] = int(v)
        return d


@dataclass
class BenchmarkResult:
    """Top-level benchmark execution artifact containing metadata and records."""

    metadata: dict[str, Any]
    summary: dict[str, Any]
    records: list[RunRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.records],
        }


def get_canonical_step_size(sampler_name: str, dim: int) -> float:
    """Returns canonical step size scaled appropriately for dimension d."""
    dim_factor = max(1.0, float(dim) ** 0.25)
    base_steps = {
        "rwm": 0.20 / dim_factor,
        "mala": 0.10 / dim_factor,
        "hmc": 0.05 / dim_factor,
        "tamed_mala": 0.10 / dim_factor,
        "barker": 0.15 / dim_factor,
        "stereographic_mala": 0.10 / dim_factor,
        "radial_mala": 0.10 / dim_factor,
        "gentle_mala": 0.15 / dim_factor,
        "anisotropic_tail_mala": 0.15 / dim_factor,
        "unified_robust": 0.15 / dim_factor,
    }
    return base_steps.get(sampler_name, 0.10 / dim_factor)


SAMPLER_FACTORIES: dict[str, Callable[[int, float], Sampler]] = {
    "rwm": lambda d, h: RandomWalkMetropolis(step_size=h),
    "mala": lambda d, h: MALA(step_size=h),
    "hmc": lambda d, h: HMC(step_size=h, n_leapfrog=5),
    "tamed_mala": lambda d, h: GloballyTamedMALA(step_size=h, gamma=0.2, mode="coordinate"),
    "barker": lambda d, h: BarkerLangevin(step_size=h),
    "stereographic_mala": lambda d, h: make_stereographic_mala(step_size=h, dim=d, scale=1.0),
    "radial_mala": lambda d, h: make_radial_mala(step_size=h, dim=d, radius=3.0, stiffness=0.5),
    "gentle_mala": lambda d, h: GentleMALA(step_size=h, gamma=1.0, mode="coordinate"),
    "anisotropic_tail_mala": lambda d, h: AnisotropicTailMALA(step_size=h, radii=1.0),
    "unified_robust": lambda d, h: UnifiedRobustSampler(step_size=h, gamma=0.1, radius=3.0, mode="coordinate"),
}

REGIME_TARGET_MAP: dict[str, list[str]] = {
    "benign": ["gaussian_isotropic", "gaussian_anisotropic"],
    "roughness": ["quartic", "superlinear_p6", "l0_l1_smooth", "neals_funnel", "rippled_gaussian"],
    "heavy_tails": ["student_t_1.5", "student_t_2.5", "student_t_4.0", "heterogeneous_heavy"],
    "dual_pathology": ["dual_coupled", "dual_confinement"],
}

QUICK_TARGET_MAP: dict[str, list[str]] = {
    "benign": ["gaussian_isotropic"],
    "roughness": ["quartic"],
    "heavy_tails": ["student_t_2.5"],
    "dual_pathology": ["dual_coupled"],
}


def generate_initial_state(
    init_type: str,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generates standardized initialization vectors."""
    if init_type == "bulk":
        return np.zeros(dim, dtype=np.float64)
    elif init_type == "tail":
        direction = rng.normal(size=dim)
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-12:
            direction = np.ones(dim)
            norm_dir = np.sqrt(dim)
        target_norm = 10.0 * np.sqrt(float(dim))
        return (target_norm / norm_dir) * direction
    else:
        raise ValueError(f"Unknown init_type: {init_type}. Must be 'bulk' or 'tail'.")


class BenchmarkRunner:
    """Automated benchmark harness evaluating samplers across regimes under strict budgets."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def run_single_configuration(
        self,
        regime: str,
        target_name: str,
        dim: int,
        init_type: str,
        sampler_name: str,
    ) -> RunRecord:
        """Executes multi-chain evaluation for a single (target, dim, init, sampler) setup."""
        target = get_target(target_name, dim=dim)
        h = get_canonical_step_size(sampler_name, dim)
        n_chains = self.config.n_chains
        remaining_budget = self.config.budget

        chain_samples: list[np.ndarray] = []
        total_grad_evals = 0
        total_lp_evals = 0
        total_accepted = 0
        total_transitions = 0
        post_warmup_accepted = 0
        diverged = False
        divergence_reason: str | None = None

        start_time = time.perf_counter()

        for chain_idx in range(n_chains):
            chains_remaining = n_chains - chain_idx
            budget_per_chain = max(1, remaining_budget // chains_remaining)
            remaining_budget -= budget_per_chain
            burn_in_budget = int(self.config.burn_in_ratio * budget_per_chain)

            chain_seed = self.config.seed + chain_idx * 1000 + dim * 10
            rng = np.random.default_rng(chain_seed)
            x0 = generate_initial_state(init_type, dim, rng)

            sampler: Sampler = SAMPLER_FACTORIES[sampler_name](dim, h)

            # Wrap target in strict budget tracker
            max_grad = budget_per_chain if sampler.requires_grad else None
            max_lp = budget_per_chain if not sampler.requires_grad else None
            tracker = GradBudgetTracker(
                target,
                max_grad_evals=max_grad,
                max_log_prob_evals=max_lp,
            )

            chain_pos: list[np.ndarray] = []
            chain_step_idx = 0
            adaptation_finalized = False

            try:
                state = sampler.init_state(x0, tracker)

                # Defensive check: non-finite initial state
                if not np.isfinite(state.log_prob) or (
                    state.grad_log_prob is not None and not np.all(np.isfinite(state.grad_log_prob))
                ):
                    diverged = True
                    divergence_reason = "Numerical coordinate explosion / NaN detected at initial state"
                    break

                step_limit = max(100, budget_per_chain * 5)
                chain_step_count = 0

                while not tracker.is_exhausted:
                    chain_step_count += 1
                    if chain_step_count > step_limit:
                        diverged = True
                        divergence_reason = "Evaluation budget loop stalled without progress"
                        break

                    # Evaluate single step
                    res = sampler.step(state, tracker, rng)
                    state = res.state

                    total_transitions += 1
                    curr_evals = tracker.grad_evals if sampler.requires_grad else tracker.log_prob_evals
                    is_warmup = curr_evals <= burn_in_budget

                    if res.accepted:
                        total_accepted += 1
                        if not is_warmup:
                            post_warmup_accepted += 1

                    # Warmup adaptation protocol (Finding 2)
                    if is_warmup:
                        sampler.adapt(chain_step_idx, res)
                        chain_step_idx += 1
                    else:
                        if not adaptation_finalized:
                            if hasattr(sampler, "finalize_adaptation"):
                                sampler.finalize_adaptation()
                            adaptation_finalized = True

                    # Extract primal coordinates
                    if "primal_x" in state.auxiliary:
                        primal = state.auxiliary["primal_x"]
                    else:
                        primal = state.x

                    # Check for numerical explosion
                    if (
                        not np.all(np.isfinite(primal))
                        or float(np.max(np.abs(primal))) > 1e12
                        or not np.isfinite(state.log_prob)
                        or (state.grad_log_prob is not None and not np.all(np.isfinite(state.grad_log_prob)))
                    ):
                        diverged = True
                        divergence_reason = "Numerical coordinate explosion / NaN detected"
                        break

                    if not is_warmup:
                        chain_pos.append(np.asarray(primal, dtype=np.float64).copy())

            except BudgetExceededError:
                # Normal termination on budget boundary
                pass
            except Exception as exc:
                diverged = True
                divergence_reason = f"Exception during execution: {type(exc).__name__}: {exc!s}"
            finally:
                if not adaptation_finalized and hasattr(sampler, "finalize_adaptation"):
                    try:
                        sampler.finalize_adaptation()
                    except Exception:
                        pass
                    adaptation_finalized = True

            total_grad_evals += tracker.grad_evals
            total_lp_evals += tracker.log_prob_evals

            if len(chain_pos) >= 2:
                chain_samples.append(np.array(chain_pos))
            else:
                diverged = True
                if not divergence_reason:
                    divergence_reason = "Insufficient samples gathered before budget exhaustion"

            if diverged:
                break

        # Check for frozen chain / zero acceptance rate
        if not diverged:
            if total_accepted == 0 or post_warmup_accepted == 0:
                diverged = True
                divergence_reason = "Frozen chain / zero acceptance rate"

        wall_clock = time.perf_counter() - start_time
        acceptance_rate = total_accepted / max(1, total_transitions)

        # Metric computation
        if not diverged and len(chain_samples) == n_chains:
            # Multi-chain array shape: (M, N_min, d)
            min_len = min(len(c) for c in chain_samples)
            stacked_chains = np.array([c[:min_len] for c in chain_samples])
            flat_samples = stacked_chains.reshape(-1, dim)

            bulk_ess_arr = compute_bulk_ess(stacked_chains)
            tail_ess_arr = compute_tail_ess(stacked_chains)
            bulk_ess = float(np.min(bulk_ess_arr))
            tail_ess = float(np.min(tail_ess_arr))

            split_rhat_arr = compute_split_rhat(stacked_chains)
            folded_rhat_arr = compute_folded_rhat(stacked_chains)
            split_rhat = float(np.max(split_rhat_arr))
            folded_rhat = float(np.max(folded_rhat_arr))

            ks_res = compute_ks_test(flat_samples, target)
            ks_pvalue = float(ks_res["min_pvalue"])
            ks_stat = float(ks_res["max_statistic"])

            w1_arr = compute_wasserstein1(flat_samples, target)
            wasserstein1 = float(np.mean(w1_arr))

            q_errs = compute_quantile_relative_error(flat_samples, target, quantiles=(0.05, 0.95, 0.99))
            q05_err = float(np.mean(q_errs[0.05]))
            q95_err = float(np.mean(q_errs[0.95]))
            q99_err = float(np.mean(q_errs[0.99]))

            effective_evals = max(1, total_grad_evals if total_grad_evals > 0 else total_lp_evals)
            ess_per_grad = bulk_ess / effective_evals
            tail_ess_per_grad = tail_ess / effective_evals
            n_samples = len(flat_samples)
        else:
            bulk_ess = tail_ess = ess_per_grad = tail_ess_per_grad = float("nan")
            split_rhat = folded_rhat = float("nan")
            ks_pvalue = ks_stat = wasserstein1 = float("nan")
            q05_err = q95_err = q99_err = float("nan")
            n_samples = 0

        return RunRecord(
            regime=regime,
            target_name=target_name,
            dimension=dim,
            init_type=init_type,
            sampler_name=sampler_name,
            step_size=h,
            budget=self.config.budget,
            n_chains=n_chains,
            n_samples=n_samples,
            grad_evals=total_grad_evals,
            log_prob_evals=total_lp_evals,
            bulk_ess=bulk_ess,
            tail_ess=tail_ess,
            ess_per_grad=ess_per_grad,
            tail_ess_per_grad=tail_ess_per_grad,
            split_rhat=split_rhat,
            folded_rhat=folded_rhat,
            ks_pvalue=ks_pvalue,
            ks_stat=ks_stat,
            wasserstein1=wasserstein1,
            q05_error=q05_err,
            q95_error=q95_err,
            q99_error=q99_err,
            acceptance_rate=acceptance_rate,
            wall_clock_time=wall_clock,
            non_divergence=not diverged,
            divergence_reason=divergence_reason,
        )

    def run_all(self) -> BenchmarkResult:
        """Executes full benchmark suite across all configured regimes."""
        records: list[RunRecord] = []
        target_map = QUICK_TARGET_MAP if self.config.quick else REGIME_TARGET_MAP

        for regime in self.config.regimes:
            targets = target_map.get(regime, [])
            for target_name in targets:
                for dim in self.config.dimensions:
                    # Funnel requires d >= 2
                    if "funnel" in target_name and dim < 2:
                        continue
                    for init_type in self.config.initializations:
                        for sampler_name in self.config.samplers:
                            record = self.run_single_configuration(
                                regime=regime,
                                target_name=target_name,
                                dim=dim,
                                init_type=init_type,
                                sampler_name=sampler_name,
                            )
                            records.append(record)

        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "quick": self.config.quick,
            "seed": self.config.seed,
            "budget": self.config.budget,
            "n_chains": self.config.n_chains,
            "samplers": list(self.config.samplers),
            "dimensions": list(self.config.dimensions),
        }

        summary = self._compute_summary(records)
        result = BenchmarkResult(metadata=metadata, summary=summary, records=records)

        # Export files
        self.export_results(result)
        return result

    def _compute_summary(self, records: list[RunRecord]) -> dict[str, Any]:
        """Calculates aggregate statistics and verifies guardrail acceptance criteria."""
        total = len(records)
        completed = sum(1 for r in records if r.non_divergence)
        diverged = total - completed

        # AC2 Check: Benign parity on d=50 Gaussian
        ac2_records = [r for r in records if r.dimension == 50 and "gaussian" in r.target_name and r.non_divergence]
        ac2_pass = False
        ac2_ratio = 0.0
        if ac2_records:
            mala_ess = max((r.ess_per_grad for r in ac2_records if r.sampler_name == "mala"), default=0.0)
            unified_ess = max((r.ess_per_grad for r in ac2_records if r.sampler_name == "unified_robust"), default=0.0)
            if mala_ess > 0:
                ac2_ratio = unified_ess / mala_ess
                ac2_pass = bool(ac2_ratio >= 0.70)

        # AC4 Check: Heavy tail efficiency on d>=10 mixed target
        ac4_records = [
            r for r in records if r.dimension >= 10 and "heterogeneous" in r.target_name and r.non_divergence
        ]
        ac4_pass = False
        if ac4_records:
            barker_tail = max((r.tail_ess_per_grad for r in ac4_records if r.sampler_name == "barker"), default=0.0)
            unified_tail = max(
                (r.tail_ess_per_grad for r in ac4_records if r.sampler_name == "unified_robust"), default=0.0
            )
            if barker_tail > 0:
                ac4_pass = bool((unified_tail / barker_tail) >= 1.5)

        return {
            "total_runs": total,
            "completed_runs": completed,
            "diverged_runs": diverged,
            "completion_rate": float(completed / max(1, total)),
            "guardrails": {
                "AC1_programmatic_correctness": bool(completed > 0),
                "AC2_benign_parity": ac2_pass,
                "AC2_ess_ratio": ac2_ratio,
                "AC4_heavy_tail_efficiency": ac4_pass,
            },
        }

    def export_results(self, result: BenchmarkResult) -> tuple[str, str]:
        """Saves benchmark results to JSON and CSV formats."""
        json_path = os.path.join(self.config.output_dir, "benchmark_results.json")
        csv_path = os.path.join(self.config.output_dir, "benchmark_results.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        dicts = [r.to_dict() for r in result.records]
        if dicts:
            fieldnames = [k for k in dicts[0].keys() if k != "extra_stats"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
                writer.writeheader()
                writer.writerows(dicts)
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                f.write("")

        return json_path, csv_path


def main_runner() -> None:
    """CLI entrypoint for automated benchmark harness."""
    parser = argparse.ArgumentParser(description="MCMC Robust Automated Benchmark Harness (Feature F20)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Quick smoke test mode (< 15s)")
    group.add_argument("--full", action="store_true", help="Full comprehensive evaluation across all 5 regimes")

    parser.add_argument("--regimes", nargs="+", default=None, help="Target regimes to evaluate")
    parser.add_argument("--samplers", nargs="+", default=None, help="Specific samplers to evaluate")
    parser.add_argument("--dims", nargs="+", type=int, default=None, help="State space dimensions (e.g. 2 10 50)")
    parser.add_argument("--budget", type=int, default=None, help="Total evaluation budget per run")
    parser.add_argument("--n-chains", type=int, default=None, help="Number of independent chains")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")

    args = parser.parse_args()

    is_quick = args.quick or (not args.full)
    dimensions = args.dims or ((2, 10) if is_quick else (2, 10, 50, 100))
    budget = args.budget or (200 if is_quick else 3000)
    n_chains = args.n_chains or (2 if is_quick else 4)

    config = BenchmarkConfig(
        regimes=tuple(args.regimes) if args.regimes else BenchmarkConfig.regimes,
        dimensions=tuple(dimensions),
        samplers=tuple(args.samplers)
        if args.samplers
        else (
            ("mala", "gentle_mala", "unified_robust") if is_quick and not args.samplers else BenchmarkConfig.samplers
        ),
        budget=budget,
        n_chains=n_chains,
        seed=args.seed,
        output_dir=args.output_dir,
        quick=is_quick,
    )

    runner = BenchmarkRunner(config)
    print(f"Starting MCMC Benchmark Harness [Quick={is_quick}, Budget={budget}, Chains={n_chains}]...")
    result = runner.run_all()
    print(f"Benchmark complete! Results exported to {args.output_dir}/benchmark_results.json and .csv")
    print(f"Completed runs: {result.summary['completed_runs']}/{result.summary['total_runs']}")
    print(f"Guardrail Checks: {result.summary.get('guardrails', {})}")


if __name__ == "__main__":
    main_runner()
