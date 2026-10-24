"""Applied PosteriorDB-Style Bayesian Benchmark Runner (Track 3)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import blackjax
import jax.numpy as jnp
import numpy as np

from mcmc_robust.bench.sota_adapters import (
    SOTATargetAdapter,
    evaluate_chains_with_arviz,
    run_blackjax_sampler,
    run_mcmc_robust_sampler,
)
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.integrations import JaxArcsinhTransform
from mcmc_robust.samplers.automala import AutoMALA
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.novel.nuts import UnifiedRobustNUTS
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.targets.registry import CANONICAL_APPLIED_TARGETS, TargetRegistry

APPLIED_TARGET_NAMES: tuple[str, ...] = CANONICAL_APPLIED_TARGETS

APPLIED_SAMPLER_NAMES: tuple[str, ...] = (
    "ClassicalMALA",
    "AutoMALA",
    "UnifiedRobustSampler",
    "BlackJAXNUTS (Stock)",
    "BlackJAXNUTS + JaxArcsinhTransform",
    "UnifiedRobustNUTS",
)


def _sanitize_float(val: Any, default: float | None = None) -> float | None:
    """Converts value to RFC 8259-compliant finite float or default."""
    try:
        fval = float(val)
        return fval if np.isfinite(fval) else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AppliedBenchmarkConfig:
    """Configuration for the applied PosteriorDB-style benchmark suite."""

    targets: Sequence[str] = APPLIED_TARGET_NAMES
    samplers: Sequence[str] = APPLIED_SAMPLER_NAMES
    n_chains: int = 2
    n_samples: int = 120
    n_warmup: int = 80
    max_depth: int = 7
    seed: int = 42
    output_dir: str = "reports"
    quick: bool = False


@dataclass
class AppliedRunRecord:
    """Metric record for a single (applied target, sampler) benchmark evaluation."""

    target_name: str
    sampler_name: str
    dimension: int
    n_chains: int
    n_samples: int
    n_warmup: int
    bulk_ess: float
    tail_ess: float
    ess_per_grad: float
    tail_ess_per_grad: float
    split_rhat: float
    divergences: int
    grad_evals: int
    wall_time_sec: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes record into an RFC 8259-compliant dictionary."""
        raw = asdict(self)
        out: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, (np.floating, float)):
                out[k] = _sanitize_float(v, default=None)
            elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
                out[k] = int(v)
            else:
                out[k] = v
        return out


@dataclass
class AppliedBenchmarkResult:
    """Aggregate benchmark result container for the applied PosteriorDB suite."""

    metadata: dict[str, Any]
    summary: dict[str, Any]
    records: list[AppliedRunRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.records],
        }


def _run_blackjax_arcsinh_nuts(
    target: TargetDistribution,
    n_chains: int,
    n_samples: int,
    n_warmup: int,
    seed: int,
    init_positions: np.ndarray,
    radius: float = 2.5,
    target_accept: float = 0.80,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Executes BlackJAX NUTS in JaxArcsinhTransform space with exact gradient accounting."""
    adapter = SOTATargetAdapter(target)
    raw_logdensity = adapter.jax_logdensity_fn()
    transform = JaxArcsinhTransform(radius=radius, dim=target.d)
    transformed_logdensity = transform.wrap_logdensity(raw_logdensity)

    t0 = time.perf_counter()
    all_chain_samples: list[np.ndarray] = []
    total_warmup_grads = 0
    total_draw_grads = 0
    total_draw_divergences = 0

    for c in range(n_chains):
        rng_key = jax.random.PRNGKey(seed + c * 10007)
        rng_warmup, rng_sample = jax.random.split(rng_key)
        init_z = transform.forward(jnp.asarray(init_positions[c], dtype=jnp.float64))

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            transformed_logdensity,
            target_acceptance_rate=target_accept,
        )
        (state, params), w_info = warmup.run(rng_warmup, init_z, num_steps=n_warmup)
        total_warmup_grads += int(jnp.sum(w_info.info.num_integration_steps))

        step_fn = blackjax.nuts(transformed_logdensity, **params).step

        def one_step(carry: Any, subkey: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            next_state, info = step_fn(subkey, carry)
            return next_state, (next_state.position, info.num_integration_steps, info.is_divergent)

        keys = jax.random.split(rng_sample, n_samples)
        _, (z_samples, chain_steps, chain_divs) = jax.lax.scan(one_step, state, keys)
        x_samples = jax.vmap(transform.inverse)(z_samples)

        all_chain_samples.append(np.asarray(x_samples, dtype=np.float64))
        total_draw_grads += int(jnp.sum(chain_steps))
        total_draw_divergences += int(jnp.sum(chain_divs))

    wall_time = time.perf_counter() - t0
    samples_arr = np.stack(all_chain_samples, axis=0)
    telemetry = {
        "total_grad_evals": total_warmup_grads + total_draw_grads,
        "divergences": total_draw_divergences,
        "wall_clock_time": wall_time,
    }
    return samples_arr, telemetry


class AppliedBenchmarkRunner:
    """Runner evaluating reference and novel samplers across the 5 applied PosteriorDB targets."""

    def __init__(
        self,
        config: AppliedBenchmarkConfig | None = None,
        *,
        quick: bool | None = None,
        output_dir: str | None = None,
        seed: int | None = None,
    ) -> None:
        if config is None:
            is_quick = bool(quick) if quick is not None else False
            config = AppliedBenchmarkConfig(
                n_chains=2,
                n_samples=30 if is_quick else 120,
                n_warmup=20 if is_quick else 80,
                max_depth=5 if is_quick else 7,
                seed=42 if seed is None else seed,
                output_dir="reports" if output_dir is None else output_dir,
                quick=is_quick,
            )
        elif quick is not None or output_dir is not None or seed is not None:
            is_quick = config.quick if quick is None else bool(quick)
            config = AppliedBenchmarkConfig(
                targets=config.targets,
                samplers=config.samplers,
                n_chains=config.n_chains,
                n_samples=30 if (quick and config.n_samples == 120) else config.n_samples,
                n_warmup=20 if (quick and config.n_warmup == 80) else config.n_warmup,
                max_depth=5 if (quick and config.max_depth == 7) else config.max_depth,
                seed=config.seed if seed is None else seed,
                output_dir=config.output_dir if output_dir is None else output_dir,
                quick=is_quick,
            )

        self.config: AppliedBenchmarkConfig = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def run_single(self, target_name: str, sampler_name: str) -> AppliedRunRecord:
        """Executes a single (target_name, sampler_name) configuration."""
        target = TargetRegistry.create(target_name)
        d = target.d
        n_chains = self.config.n_chains
        n_samples = 30 if (self.config.quick and self.config.n_samples > 40) else self.config.n_samples
        n_warmup = 20 if (self.config.quick and self.config.n_warmup > 30) else self.config.n_warmup
        max_depth = 5 if (self.config.quick and self.config.max_depth > 5) else self.config.max_depth
        seed = self.config.seed

        rng_init = np.random.default_rng(seed + 911)
        init_positions = rng_init.normal(0.0, 0.1, size=(n_chains, d)).astype(np.float64)
        base_step = 0.08 / (float(d) ** 0.25)

        t0 = time.perf_counter()
        try:
            if sampler_name == "ClassicalMALA":
                samples, info = run_mcmc_robust_sampler(
                    sampler_cls=MALA,
                    target=target,
                    n_chains=n_chains,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    seed=seed,
                    x0=init_positions,
                    sampler_kwargs={
                        "step_size": base_step,
                        "adapt_step_size": True,
                    },
                )
            elif sampler_name == "AutoMALA":
                samples, info = run_mcmc_robust_sampler(
                    sampler_cls=AutoMALA,
                    target=target,
                    n_chains=n_chains,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    seed=seed,
                    x0=init_positions,
                    sampler_kwargs={
                        "step_size": base_step,
                        "max_halvings": 6,
                        "max_doublings": 6,
                        "adapt_step_size": True,
                        "adapt_preconditioner": True,
                    },
                )
            elif sampler_name == "UnifiedRobustSampler":
                samples, info = run_mcmc_robust_sampler(
                    sampler_cls=UnifiedRobustSampler,
                    target=target,
                    n_chains=n_chains,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    seed=seed,
                    x0=init_positions,
                    sampler_kwargs={
                        "step_size": base_step,
                        "gamma": 0.15,
                        "radius": 2.5,
                        "adapt_step_size": True,
                        "adapt_preconditioner": True,
                        "adapt_tail_scale": True,
                        "warmup_steps": n_warmup,
                    },
                )
            elif sampler_name == "BlackJAXNUTS (Stock)":
                samples, info = run_blackjax_sampler(
                    target=target,
                    sampler_type="nuts",
                    n_chains=n_chains,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    seed=seed,
                    x0=init_positions,
                )
            elif sampler_name == "BlackJAXNUTS + JaxArcsinhTransform":
                samples, info = _run_blackjax_arcsinh_nuts(
                    target=target,
                    n_chains=n_chains,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    seed=seed,
                    init_positions=init_positions,
                    radius=2.5,
                )
            elif sampler_name == "UnifiedRobustNUTS":
                samples, info = run_mcmc_robust_sampler(
                    sampler_cls=UnifiedRobustNUTS,
                    target=target,
                    n_chains=n_chains,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    seed=seed,
                    x0=init_positions,
                    sampler_kwargs={
                        "step_size": 0.10,
                        "max_depth": max_depth,
                        "gamma": 0.15,
                        "radius": 2.5,
                        "adapt_step_size": True,
                        "adapt_preconditioner": True,
                        "adapt_tail_scale": True,
                        "warmup_steps": n_warmup,
                    },
                )
            else:
                raise ValueError(f"Unknown sampler '{sampler_name}'")

            wall_time = max(1e-6, time.perf_counter() - t0)
            grad_evals = max(1, int(info["total_grad_evals"]))
            divergences = int(info["divergences"])

            metrics = evaluate_chains_with_arviz(
                samples=samples,
                total_grad_evals=grad_evals,
                divergences=divergences,
                wall_clock_time=wall_time,
            )
            bulk_ess = _sanitize_float(metrics["bulk_ess"], default=float(n_chains)) or float(n_chains)
            tail_ess = _sanitize_float(metrics["tail_ess"], default=float(n_chains)) or float(n_chains)
            split_rhat = _sanitize_float(metrics["split_rhat"], default=1.0) or 1.0
            ess_per_grad = float(bulk_ess / grad_evals)
            tail_ess_per_grad = float(tail_ess / grad_evals)

            return AppliedRunRecord(
                target_name=target_name,
                sampler_name=sampler_name,
                dimension=d,
                n_chains=n_chains,
                n_samples=n_samples,
                n_warmup=n_warmup,
                bulk_ess=bulk_ess,
                tail_ess=tail_ess,
                ess_per_grad=ess_per_grad,
                tail_ess_per_grad=tail_ess_per_grad,
                split_rhat=split_rhat,
                divergences=divergences,
                grad_evals=grad_evals,
                wall_time_sec=wall_time,
                error=None,
            )
        except Exception as exc:
            wall_time = max(1e-6, time.perf_counter() - t0)
            return AppliedRunRecord(
                target_name=target_name,
                sampler_name=sampler_name,
                dimension=d,
                n_chains=n_chains,
                n_samples=n_samples,
                n_warmup=n_warmup,
                bulk_ess=0.0,
                tail_ess=0.0,
                ess_per_grad=0.0,
                tail_ess_per_grad=0.0,
                split_rhat=float("nan"),
                divergences=0,
                grad_evals=0,
                wall_time_sec=wall_time,
                error=f"{type(exc).__name__}: {exc!s}",
            )

    def run_all(self) -> AppliedBenchmarkResult:
        """Executes all configured samplers across all 5 applied targets and exports JSON/CSV."""
        records: list[AppliedRunRecord] = []
        for target_name in self.config.targets:
            for sampler_name in self.config.samplers:
                rec = self.run_single(target_name=target_name, sampler_name=sampler_name)
                records.append(rec)

        summary = self._compute_summary(records)
        metadata = {
            "suite": "posteriordb_applied_bayesian_benchmarks",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "quick": self.config.quick,
            "seed": self.config.seed,
            "n_chains": self.config.n_chains,
            "n_samples": self.config.n_samples,
            "n_warmup": self.config.n_warmup,
            "targets": list(self.config.targets),
            "samplers": list(self.config.samplers),
        }
        result = AppliedBenchmarkResult(metadata=metadata, summary=summary, records=records)
        self.export_results(result)
        return result

    def _compute_summary(self, records: list[AppliedRunRecord]) -> dict[str, Any]:
        """Aggregates per-sampler performance across the applied benchmark suite."""
        by_sampler: dict[str, dict[str, Any]] = {}
        for s_name in self.config.samplers:
            s_recs = [r for r in records if r.sampler_name == s_name and r.error is None]
            if s_recs:
                by_sampler[s_name] = {
                    "mean_bulk_ess": _sanitize_float(np.mean([r.bulk_ess for r in s_recs]), 0.0),
                    "mean_tail_ess": _sanitize_float(np.mean([r.tail_ess for r in s_recs]), 0.0),
                    "mean_ess_per_grad": _sanitize_float(np.mean([r.ess_per_grad for r in s_recs]), 0.0),
                    "mean_tail_ess_per_grad": _sanitize_float(np.mean([r.tail_ess_per_grad for r in s_recs]), 0.0),
                    "total_divergences": int(sum(r.divergences for r in s_recs)),
                    "total_grad_evals": int(sum(r.grad_evals for r in s_recs)),
                    "mean_wall_time_sec": _sanitize_float(np.mean([r.wall_time_sec for r in s_recs]), 0.0),
                }
        return {
            "total_runs": len(records),
            "successful_runs": sum(1 for r in records if r.error is None),
            "sampler_aggregates": by_sampler,
        }

    def export_results(self, result: AppliedBenchmarkResult) -> tuple[str, str]:
        """Writes RFC 8259-compliant JSON and CSV files to output_dir."""
        json_path = os.path.join(self.config.output_dir, "applied_benchmark_results.json")
        csv_path = os.path.join(self.config.output_dir, "applied_benchmark_results.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, allow_nan=False)
            f.write("\n")

        rows = [r.to_dict() for r in result.records]
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("")

        return json_path, csv_path


def main() -> None:
    """CLI entrypoint for the PosteriorDB-style applied benchmark suite."""
    parser = argparse.ArgumentParser(description="Applied PosteriorDB-Style Bayesian Benchmark Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark configuration")
    parser.add_argument("--n-chains", type=int, default=2, help="Number of chains per run")
    parser.add_argument("--n-samples", type=int, default=120, help="Post-warmup draws per chain")
    parser.add_argument("--n-warmup", type=int, default=80, help="Warmup steps per chain")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for JSON/CSV artifacts")
    args = parser.parse_args()

    config = AppliedBenchmarkConfig(
        n_chains=args.n_chains,
        n_samples=30 if args.quick else args.n_samples,
        n_warmup=20 if args.quick else args.n_warmup,
        max_depth=5 if args.quick else 7,
        seed=args.seed,
        output_dir=args.output_dir,
        quick=args.quick,
    )
    runner = AppliedBenchmarkRunner(config=config)
    print(
        f"Running Applied PosteriorDB Benchmark Suite "
        f"[quick={config.quick}, chains={config.n_chains}, draws={config.n_samples}, warmup={config.n_warmup}]..."
    )
    result = runner.run_all()
    print(f"Completed {result.summary['successful_runs']}/{result.summary['total_runs']} runs.")
    print(f"Artifacts written to {config.output_dir}/applied_benchmark_results.{{json,csv}}")


if __name__ == "__main__":
    main()
