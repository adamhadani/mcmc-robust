"""Step-size stability sweeps and operating window quantification (AC3)."""

import argparse
import csv
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mcmc_robust.bench.runner import SAMPLER_FACTORIES, generate_initial_state
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.targets.suite import get_target


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for step-size stability sweeps."""

    targets: Sequence[str] = ("quartic", "neals_funnel")
    dimensions: Sequence[int] = (2, 10, 50)
    samplers: Sequence[str] = ("mala", "tamed_mala", "gentle_mala", "barker", "unified_robust")
    h_min: float = 1e-4
    h_max: float = 1e1
    n_grid: int = 25
    steps_per_run: int = 500
    bulk_radius: float = 2.0
    seed: int = 777
    output_dir: str = "reports"
    quick: bool = False


@dataclass
class SweepRecord:
    """Results of a step-size sweep evaluation at a specific (target, sampler, h)."""

    target_name: str
    dimension: int
    sampler_name: str
    step_size: float
    acceptance_rate: float
    diverged: bool
    burn_in_steps: int | None
    terminal_norm: float
    divergence_reason: str | None = None

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


class StepSizeSweep:
    """Executes log-spaced step-size stability sweeps across Quartic and Funnel targets."""

    def __init__(self, config: SweepConfig | None = None) -> None:
        self.config = config or SweepConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def run_sweep(self) -> dict[str, Any]:
        """Executes all step-size sweep evaluations."""
        records: list[SweepRecord] = []
        grid = np.logspace(
            np.log10(self.config.h_min),
            np.log10(self.config.h_max),
            num=self.config.n_grid,
        )

        for target_name in self.config.targets:
            for dim in self.config.dimensions:
                if "funnel" in target_name and dim < 2:
                    continue
                target = get_target(target_name, dim=dim)

                for sampler_name in self.config.samplers:
                    for h in grid:
                        record = self._run_single_sweep(
                            target_name=target_name,
                            target=target,
                            dim=dim,
                            sampler_name=sampler_name,
                            step_size=float(h),
                        )
                        records.append(record)

        summary = self._analyze_stability_windows(records)
        output_payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "h_min": self.config.h_min,
                "h_max": self.config.h_max,
                "n_grid": self.config.n_grid,
                "targets": list(self.config.targets),
                "samplers": list(self.config.samplers),
                "quick": self.config.quick,
            },
            "summary": summary,
            "results": [r.to_dict() for r in records],
        }

        self._export_sweep(output_payload, records)
        return output_payload

    def _run_single_sweep(
        self,
        target_name: str,
        target: Any,
        dim: int,
        sampler_name: str,
        step_size: float,
    ) -> SweepRecord:
        """Runs a single trajectory from tail initialization to check stability and burn-in."""
        rng = np.random.default_rng(self.config.seed)
        x0 = generate_initial_state("tail", dim, rng)

        sampler: Sampler = SAMPLER_FACTORIES[sampler_name](dim, step_size)
        n_steps = self.config.steps_per_run
        bulk_r = self.config.bulk_radius

        accepted_count = 0
        burn_in_steps: int | None = None
        diverged = False
        divergence_reason: str | None = None
        terminal_norm = float(np.linalg.norm(x0))

        try:
            state = sampler.init_state(x0, target)
            for step_idx in range(1, n_steps + 1):
                res = sampler.step(state, target, rng)
                state = res.state

                if res.accepted:
                    accepted_count += 1

                primal = state.auxiliary.get("primal_x", state.x)
                if not np.all(np.isfinite(primal)) or float(np.max(np.abs(primal))) > 1e12:
                    diverged = True
                    divergence_reason = "NaN or extreme explosion > 1e12"
                    break

                current_norm = float(np.linalg.norm(primal))
                terminal_norm = current_norm

                if burn_in_steps is None and current_norm <= bulk_r:
                    burn_in_steps = step_idx

        except Exception as exc:
            diverged = True
            divergence_reason = f"Exception: {type(exc).__name__}: {exc!s}"

        acceptance_rate = accepted_count / max(1, n_steps)

        return SweepRecord(
            target_name=target_name,
            dimension=dim,
            sampler_name=sampler_name,
            step_size=step_size,
            acceptance_rate=acceptance_rate,
            diverged=diverged,
            burn_in_steps=burn_in_steps,
            terminal_norm=terminal_norm,
            divergence_reason=divergence_reason,
        )

    def _analyze_stability_windows(self, records: list[SweepRecord]) -> dict[str, Any]:
        """Quantifies AC3 operating stability windows and burn-in acceleration."""
        summary: dict[str, Any] = {}

        for target in self.config.targets:
            target_summary: dict[str, Any] = {}
            for dim in self.config.dimensions:
                dim_key = f"d_{dim}"
                windows: dict[str, dict[str, Any]] = {}

                for s_name in self.config.samplers:
                    matching = [
                        r
                        for r in records
                        if r.target_name == target
                        and r.dimension == dim
                        and r.sampler_name == s_name
                        and not r.diverged
                        and r.acceptance_rate > 0.01
                    ]

                    if matching:
                        h_min = float(min(r.step_size for r in matching))
                        h_max = float(max(r.step_size for r in matching))
                        window_ratio = h_max / max(1e-12, h_min)
                        burnins = [r.burn_in_steps for r in matching if r.burn_in_steps is not None]
                        median_burnin = float(np.median(burnins)) if burnins else None
                    else:
                        h_min = h_max = window_ratio = 0.0
                        median_burnin = None

                    windows[s_name] = {
                        "h_min": h_min,
                        "h_max": h_max,
                        "window_ratio": window_ratio,
                        "median_burnin_steps": median_burnin,
                    }

                # AC3 Verification: GentleMALA vs Classical MALA
                mala_win = windows.get("mala", {}).get("window_ratio", 0.0)
                gentle_win = windows.get("gentle_mala", {}).get("window_ratio", 0.0)
                stability_speedup = (
                    (gentle_win / mala_win) if mala_win > 0 else (float("inf") if gentle_win > 0 else 0.0)
                )

                # Burn-in reduction vs Globally Tamed MALA
                tamed_burn = windows.get("tamed_mala", {}).get("median_burnin_steps")
                gentle_burn = windows.get("gentle_mala", {}).get("median_burnin_steps")
                burnin_reduction = (
                    (tamed_burn / gentle_burn)
                    if (tamed_burn is not None and gentle_burn is not None and gentle_burn > 0)
                    else None
                )

                target_summary[dim_key] = {
                    "sampler_windows": windows,
                    "AC3_stability_factor": stability_speedup,
                    "AC3_stability_pass": bool(stability_speedup >= 10.0),
                    "AC3_burnin_reduction": burnin_reduction,
                    "AC3_burnin_pass": bool(burnin_reduction is not None and burnin_reduction >= 2.0),
                }

            summary[target] = target_summary

        return summary

    def _export_sweep(self, payload: dict[str, Any], records: list[SweepRecord]) -> None:
        """Saves sweep results to JSON and CSV formats."""
        json_path = os.path.join(self.config.output_dir, "sweep_results.json")
        csv_path = os.path.join(self.config.output_dir, "sweep_results.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        dicts = [r.to_dict() for r in records]
        if dicts:
            fieldnames = list(dicts[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(dicts)
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                f.write("")


def main_sweep() -> None:
    """CLI entrypoint for step-size stability sweeps."""
    parser = argparse.ArgumentParser(description="MCMC Robust Step-Size Stability Sweeps (Feature F20 / AC3)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test mode")
    parser.add_argument("--targets", nargs="+", default=None, help="Targets to sweep")
    parser.add_argument("--samplers", nargs="+", default=None, help="Samplers to sweep")
    parser.add_argument("--dims", nargs="+", type=int, default=None, help="Dimensions")
    parser.add_argument("--n-grid", type=int, default=None, help="Grid points count")
    parser.add_argument("--steps", type=int, default=None, help="Steps per run")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")

    args = parser.parse_args()

    if args.quick:
        targets = tuple(args.targets) if args.targets else ("quartic",)
        dimensions = tuple(args.dims) if args.dims else (2,)
        samplers = tuple(args.samplers) if args.samplers else ("mala", "gentle_mala")
        n_grid = args.n_grid or 5
        steps = args.steps or 100
    else:
        targets = tuple(args.targets) if args.targets else tuple(SweepConfig.targets)
        dimensions = tuple(args.dims) if args.dims else tuple(SweepConfig.dimensions)
        samplers = tuple(args.samplers) if args.samplers else tuple(SweepConfig.samplers)
        n_grid = args.n_grid or 25
        steps = args.steps or 500

    config = SweepConfig(
        targets=targets,
        dimensions=dimensions,
        samplers=samplers,
        n_grid=n_grid,
        steps_per_run=steps,
        output_dir=args.output_dir,
        quick=bool(args.quick),
    )

    runner = StepSizeSweep(config)
    print(f"Starting Step-Size Sweeps [Quick={args.quick}, Grid={n_grid}, Steps={steps}]...")
    res = runner.run_sweep()
    print(f"Sweep complete! Exported to {args.output_dir}/sweep_results.json and .csv")
    print(f"Summary: {res['summary']}")


if __name__ == "__main__":
    main_sweep()
