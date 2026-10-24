"""Applied Bayesian benchmark runner over synthetic, PosteriorDB-inspired models (Track 3).

Reporting policy:
- Non-finite diagnostics (ESS, R-hat) are reported as ``null``, never replaced by defaults.
- Every record carries ``converged`` (max split R-hat <= ``RHAT_CONVERGED``); efficiency
  ratios from unconverged runs should not be quoted.
- ``bulk_ess``/``tail_ess`` are coordinate means; ``min_bulk_ess``/``min_tail_ess`` are the
  worst coordinate, which is what limits inference on hierarchical scale parameters.
- All NUTS variants share one maximum tree depth (``AppliedBenchmarkConfig.max_depth``,
  BlackJAX's default of 10 doublings).
- ``wall_time_sec`` includes JAX JIT compilation for the BlackJAX samplers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
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


#: Split R-hat threshold below which a run counts as converged (Vehtari et al. 2021).
RHAT_CONVERGED: float = 1.01

#: Maximum NUTS tree depth shared by every NUTS variant (BlackJAX's default ``max_num_doublings``).
NUTS_MAX_DEPTH: int = 10


def _sanitize_float(val: Any, default: float | None = None) -> float | None:
    """Converts value to RFC 8259-compliant finite float or default."""
    try:
        fval = float(val)
        return fval if np.isfinite(fval) else default
    except (TypeError, ValueError):
        return default


def _per_grad(value: float | None, grad_evals: int) -> float | None:
    """Normalises a finite metric by the gradient budget; propagates missing values."""
    return None if value is None else value / grad_evals


def _mean_of_finite(values: Sequence[float | None]) -> float | None:
    """Mean over the finite entries, or None when there are none."""
    finite = [v for v in values if v is not None]
    return float(np.mean(finite)) if finite else None


@dataclass(frozen=True)
class AppliedBenchmarkConfig:
    """Configuration for the applied benchmark suite.

    ``quick=True`` selects smoke-test sizes (see :meth:`resolved`); use it for CI, never for
    reported results.
    """

    targets: Sequence[str] = APPLIED_TARGET_NAMES
    samplers: Sequence[str] = APPLIED_SAMPLER_NAMES
    n_chains: int = 4
    n_samples: int = 1000
    n_warmup: int = 1000
    max_depth: int = NUTS_MAX_DEPTH
    seed: int = 42
    output_dir: str = "reports"
    quick: bool = False

    def resolved(self) -> AppliedBenchmarkConfig:
        """Returns the effective configuration, applying smoke-test sizes when ``quick`` is set."""
        if not self.quick:
            return self
        return replace(self, n_chains=2, n_samples=30, n_warmup=20)


@dataclass
class AppliedRunRecord:
    """Metric record for a single (applied target, sampler) benchmark evaluation.

    Diagnostics that ArviZ could not compute (non-finite) are stored as ``None``.
    """

    target_name: str
    sampler_name: str
    dimension: int
    n_chains: int
    n_samples: int
    n_warmup: int
    bulk_ess: float | None
    tail_ess: float | None
    min_bulk_ess: float | None
    min_tail_ess: float | None
    ess_per_grad: float | None
    tail_ess_per_grad: float | None
    split_rhat: float | None
    converged: bool
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
    """Aggregate benchmark result container for the applied suite."""

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
    max_num_doublings: int = NUTS_MAX_DEPTH,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Executes BlackJAX NUTS in JaxArcsinhTransform space with exact gradient accounting.

    The transform is static (fixed ``radius``, centred at the origin); no location or scale
    is learned from the target.
    """
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
            max_num_doublings=max_num_doublings,
        )
        (state, params), w_info = warmup.run(rng_warmup, init_z, num_steps=n_warmup)
        total_warmup_grads += int(jnp.sum(w_info.info.num_integration_steps))

        # Window adaptation echoes `max_num_doublings` in `params`; merging keeps one value.
        step_fn = blackjax.nuts(transformed_logdensity, **{**params, "max_num_doublings": max_num_doublings}).step

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
    """Runner evaluating reference and novel samplers across the 5 applied targets."""

    def __init__(
        self,
        config: AppliedBenchmarkConfig | None = None,
        *,
        quick: bool | None = None,
        output_dir: str | None = None,
        seed: int | None = None,
    ) -> None:
        base = config if config is not None else AppliedBenchmarkConfig()
        if quick is not None:
            base = replace(base, quick=quick)
        if output_dir is not None:
            base = replace(base, output_dir=output_dir)
        if seed is not None:
            base = replace(base, seed=seed)

        self.config: AppliedBenchmarkConfig = base.resolved()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def run_single(self, target_name: str, sampler_name: str) -> AppliedRunRecord:
        """Executes a single (target_name, sampler_name) configuration."""
        target = TargetRegistry.create(target_name)
        d = target.d
        n_chains = self.config.n_chains
        n_samples = self.config.n_samples
        n_warmup = self.config.n_warmup
        max_depth = self.config.max_depth
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
                    max_num_doublings=max_depth,
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
                    max_num_doublings=max_depth,
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
            # Keep non-finite diagnostics as None: substituting defaults (e.g. R-hat = 1.0)
            # would report a failed run as a converged one.
            bulk_ess = _sanitize_float(metrics["bulk_ess"])
            tail_ess = _sanitize_float(metrics["tail_ess"])
            split_rhat = _sanitize_float(metrics["split_rhat"])

            return AppliedRunRecord(
                target_name=target_name,
                sampler_name=sampler_name,
                dimension=d,
                n_chains=n_chains,
                n_samples=n_samples,
                n_warmup=n_warmup,
                bulk_ess=bulk_ess,
                tail_ess=tail_ess,
                min_bulk_ess=_sanitize_float(metrics["min_bulk_ess"]),
                min_tail_ess=_sanitize_float(metrics["min_tail_ess"]),
                ess_per_grad=_per_grad(bulk_ess, grad_evals),
                tail_ess_per_grad=_per_grad(tail_ess, grad_evals),
                split_rhat=split_rhat,
                converged=split_rhat is not None and split_rhat <= RHAT_CONVERGED,
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
                bulk_ess=None,
                tail_ess=None,
                min_bulk_ess=None,
                min_tail_ess=None,
                ess_per_grad=None,
                tail_ess_per_grad=None,
                split_rhat=None,
                converged=False,
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
            "suite": "synthetic_posteriordb_inspired_applied_benchmarks",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "quick": self.config.quick,
            "seed": self.config.seed,
            "n_chains": self.config.n_chains,
            "n_samples": self.config.n_samples,
            "n_warmup": self.config.n_warmup,
            "nuts_max_depth": self.config.max_depth,
            "rhat_converged_threshold": RHAT_CONVERGED,
            "notes": [
                "Datasets are simulated; these are not PosteriorDB posteriors.",
                "Single seed per (target, sampler); ratios carry no error bars.",
                "wall_time_sec includes JAX JIT compilation for BlackJAX samplers.",
            ],
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
                    "n_converged": sum(1 for r in s_recs if r.converged),
                    "mean_bulk_ess": _mean_of_finite([r.bulk_ess for r in s_recs]),
                    "mean_tail_ess": _mean_of_finite([r.tail_ess for r in s_recs]),
                    "mean_ess_per_grad": _mean_of_finite([r.ess_per_grad for r in s_recs]),
                    "mean_tail_ess_per_grad": _mean_of_finite([r.tail_ess_per_grad for r in s_recs]),
                    "total_divergences": int(sum(r.divergences for r in s_recs)),
                    "total_grad_evals": int(sum(r.grad_evals for r in s_recs)),
                    "mean_wall_time_sec": _mean_of_finite([r.wall_time_sec for r in s_recs]),
                }
        return {
            "total_runs": len(records),
            "successful_runs": sum(1 for r in records if r.error is None),
            "converged_runs": sum(1 for r in records if r.converged),
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
    """CLI entrypoint for the applied benchmark suite."""
    defaults = AppliedBenchmarkConfig()
    parser = argparse.ArgumentParser(description="Applied Bayesian benchmark runner (synthetic PosteriorDB-inspired)")
    parser.add_argument("--quick", action="store_true", help="Smoke-test sizes (overrides chain/draw/warm-up counts)")
    parser.add_argument("--n-chains", type=int, default=defaults.n_chains, help="Number of chains per run")
    parser.add_argument("--n-samples", type=int, default=defaults.n_samples, help="Post-warmup draws per chain")
    parser.add_argument("--n-warmup", type=int, default=defaults.n_warmup, help="Warmup steps per chain")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Master random seed")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for JSON/CSV artifacts")
    args = parser.parse_args()

    config = AppliedBenchmarkConfig(
        n_chains=args.n_chains,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
        output_dir=args.output_dir,
        quick=args.quick,
    )
    runner = AppliedBenchmarkRunner(config=config)
    cfg = runner.config
    print(
        f"Running applied benchmark suite "
        f"[quick={cfg.quick}, chains={cfg.n_chains}, draws={cfg.n_samples}, warmup={cfg.n_warmup}, "
        f"nuts_max_depth={cfg.max_depth}]..."
    )
    result = runner.run_all()
    print(
        f"Completed {result.summary['successful_runs']}/{result.summary['total_runs']} runs "
        f"({result.summary['converged_runs']} with split R-hat <= {RHAT_CONVERGED})."
    )
    print(f"Artifacts written to {cfg.output_dir}/applied_benchmark_results.{{json,csv}}")


if __name__ == "__main__":
    main()
