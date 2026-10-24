r"""Multi-seed ablation of the calibrated arcsinh transform for NUTS.

Question: how much of NUTS's efficiency on heavy-tailed targets does the calibrated coordinate-wise
arcsinh transform buy, and where does the gain come from?

Arms (every arm: BlackJAX NUTS with window adaptation, `tune` warm-up steps, `draws` draws,
`chains` chains, at most `max_num_doublings` tree doublings, paired per-seed initial positions):
- `stock`: stock NUTS in $x$-space, target acceptance 0.8.
- `stock_ta95`: the same with target acceptance 0.95 (the practitioner's first fix).
- `arcsinh`: `sample_blackjax_unified_nuts`, NUTS on the calibrated arcsinh pullback.
- `affine`: the same pipeline and Phase 1 calibration of $(\mu, R)$ with $z = (x - \mu) / R$
  and no arcsinh; separates rescaling from tail warping.
- `scale_mixture`: stock NUTS on the practitioner's non-centred Gaussian scale-mixture
  reparameterisation (`targets/scale_mixture.py`), t-family targets and Task 1 only. ESS and
  accuracy are measured on $x$; gradients are counted in the augmented space.

All arms run through `sample_blackjax_unified_nuts` (`transform_kind` "identity", "arcsinh" or
"affine"), so they share the kernel, the vectorised chains and the gradient accounting.

Per-run metrics (one row per target x arm x seed): total gradients including warm-up;
coordinate-mean and worst-coordinate bulk and tail ESS (ArviZ, via `evaluate_chains_with_arviz`);
max rank-normalised R-hat; post-warm-up divergences; ESS per gradient; the legacy q99 relative
error; and the tail-probability accuracy of `diagnostics/tail_accuracy.py` at
$p \in \{0.01, 0.99\}$ against exact marginal quantiles.

Aggregates per target x arm: median and IQR of each metric over seeds, the paired per-seed ratio
to `stock` (median and IQR), and the fraction of seeds with max R-hat <= 1.01.

Usage::

    uv run python -m mcmc_robust.bench.ablation_runner [--full | --quick] [--seeds ...]
        [--targets ...] [--arms ...] [--output-dir reports] [--workers {1,2}]

Outputs `ablation_results.json` (provenance, config, rows, aggregates), `ablation_results.csv`
(per-seed rows) and `ablation_summary.csv` (aggregates, flattened).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mcmc_robust.bench.sota_adapters import SOTATargetAdapter, evaluate_chains_with_arviz
from mcmc_robust.diagnostics.tail_accuracy import tail_probability_accuracy
from mcmc_robust.jax_unified_nuts import JaxUnifiedNUTSResult, TransformKind, sample_blackjax_unified_nuts
from mcmc_robust.targets.ablation import ABLATION_TARGET_NAMES, AblationTarget, make_ablation_target

SCHEMA_VERSION: int = 1

#: Rank-normalised R-hat at or below which a run counts as converged (Vehtari et al. 2021).
RHAT_CONVERGED: float = 1.01

#: Half-width of the uniform initial-position box (Stan's default init is Uniform(-2, 2)).
INIT_HALF_WIDTH: float = 2.0

BASELINE_ARM: str = "stock"


@dataclass(frozen=True)
class ArmSpec:
    """How one arm calls the shared JAX NUTS pipeline."""

    transform_kind: TransformKind
    target_accept: float
    augmented: bool = False
    description: str = ""


ARMS: dict[str, ArmSpec] = {
    "stock": ArmSpec("identity", 0.80, description="BlackJAX NUTS, window adaptation, target acceptance 0.8"),
    "stock_ta95": ArmSpec("identity", 0.95, description="The same with target acceptance 0.95"),
    "arcsinh": ArmSpec("arcsinh", 0.80, description="NUTS on the calibrated coordinate-wise arcsinh pullback"),
    "affine": ArmSpec(
        "affine", 0.80, description="Same pipeline and Phase 1 calibration, z = (x - mu) / R, no arcsinh"
    ),
    "scale_mixture": ArmSpec(
        "identity",
        0.80,
        augmented=True,
        description="Stock NUTS on the non-centred Gaussian scale-mixture reparameterisation",
    ),
}
ARM_NAMES: tuple[str, ...] = tuple(ARMS)

#: Per-run record fields, in output order.
ROW_FIELDS: tuple[str, ...] = (
    "target",
    "arm",
    "seed",
    "d",
    "d_sampled",
    "chains",
    "draws",
    "tune",
    "target_accept",
    "transform_kind",
    "total_grad_evals",
    "warmup_grad_evals",
    "sampling_grad_evals",
    "grads_per_draw",
    "frac_max_tree_depth",
    "bulk_ess",
    "min_bulk_ess",
    "tail_ess",
    "min_tail_ess",
    "ess_per_grad",
    "tail_ess_per_grad",
    "min_tail_ess_per_grad",
    "max_rhat",
    "converged",
    "divergences",
    "q99_rel_error",
    "tail_prob_abs_err",
    "tail_prob_abs_z",
    "tail_prob_max_abs_z",
    "tail_prob_n_degenerate",
    "step_size_median",
    "wall_time_sec",
    "error",
)

#: Metrics summarised by median and IQR over seeds.
AGG_METRICS: tuple[str, ...] = (
    "total_grad_evals",
    "grads_per_draw",
    "frac_max_tree_depth",
    "bulk_ess",
    "min_bulk_ess",
    "tail_ess",
    "min_tail_ess",
    "ess_per_grad",
    "tail_ess_per_grad",
    "min_tail_ess_per_grad",
    "max_rhat",
    "divergences",
    "q99_rel_error",
    "tail_prob_abs_err",
    "tail_prob_abs_z",
    "tail_prob_max_abs_z",
    "wall_time_sec",
)

#: Metrics whose paired per-seed ratio to the `stock` arm is summarised.
RATIO_METRICS: tuple[str, ...] = ("tail_ess_per_grad", "ess_per_grad", "min_tail_ess_per_grad", "total_grad_evals")


@dataclass(frozen=True)
class AblationConfig:
    """Grid and sampler settings of one ablation run (use `full()` for reported results)."""

    targets: tuple[str, ...] = ABLATION_TARGET_NAMES
    arms: tuple[str, ...] = ARM_NAMES
    seeds: tuple[int, ...] = tuple(range(10))
    tune: int = 500
    draws: int = 1000
    chains: int = 4
    max_num_doublings: int = 10
    tail_probs: tuple[float, ...] = (0.01, 0.99)
    mode: str = "full"
    output_dir: str = "reports"
    workers: int = 1
    arm_targets: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def full(cls, **overrides: Any) -> AblationConfig:
        """The reported protocol: 10 seeds, tune 500, draws 1000, 4 chains, tree depth 10."""
        return replace(cls(), **overrides)

    @classmethod
    def quick(cls, **overrides: Any) -> AblationConfig:
        """Smoke-test sizes for tests and CI (2 seeds, short runs); never for reported results."""
        return replace(cls(mode="quick", seeds=(0, 1), tune=100, draws=100), **overrides)

    def applies(self, target: AblationTarget, arm: str) -> bool:
        """Whether `arm` runs on `target` (scale mixtures exist only for the t family; `arm_targets` restricts arms)."""
        if ARMS[arm].augmented and target.scale_mixture is None:
            return False
        allowed = self.arm_targets.get(arm)
        return allowed is None or target.name in allowed


def initial_positions(seed: int, chains: int, d: int) -> np.ndarray:
    """Per-seed starting points shared by every arm: Uniform(-2, 2)^d per chain (Stan's default box)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-INIT_HALF_WIDTH, INIT_HALF_WIDTH, size=(chains, d))


def _finite_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _empty_row(spec: AblationTarget, arm: str, seed: int, config: AblationConfig) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
    sm = spec.scale_mixture
    row.update(
        target=spec.name,
        arm=arm,
        seed=int(seed),
        d=spec.d,
        d_sampled=sm.d_aug if (ARMS[arm].augmented and sm is not None) else spec.d,
        chains=config.chains,
        draws=config.draws,
        tune=config.tune,
        target_accept=ARMS[arm].target_accept,
        transform_kind=ARMS[arm].transform_kind,
    )
    return row


def run_single(spec: AblationTarget, arm: str, seed: int, config: AblationConfig) -> dict[str, Any]:
    """Runs one (target, arm, seed) cell and returns its row (`error` holds the message if it failed)."""
    arm_spec = ARMS[arm]
    row = _empty_row(spec, arm, seed, config)
    x0 = initial_positions(seed, config.chains, spec.d)
    common: dict[str, Any] = {
        "draws": config.draws,
        "tune": config.tune,
        "chains": config.chains,
        "seed": int(seed),
        "target_accept": arm_spec.target_accept,
        "max_num_doublings": config.max_num_doublings,
        "transform_kind": arm_spec.transform_kind,
    }
    t0 = time.perf_counter()
    try:
        res: JaxUnifiedNUTSResult
        if arm_spec.augmented:
            sm = spec.scale_mixture
            if sm is None:
                raise ValueError(f"Target {spec.name!r} has no scale-mixture representation")
            res = sample_blackjax_unified_nuts(sm.logdensity, d=sm.d_aug, x0=sm.init_from_x(x0), **common)
            x = np.asarray(jax.vmap(jax.vmap(sm.to_x))(jnp.asarray(res.samples)), dtype=np.float64)
        else:
            logdensity = SOTATargetAdapter(spec.target).jax_logdensity_fn()
            res = sample_blackjax_unified_nuts(logdensity, d=spec.d, x0=x0, **common)
            x = res.samples
        wall = time.perf_counter() - t0

        m = evaluate_chains_with_arviz(
            x, res.grad_evals, target=spec.target, divergences=res.divergences, wall_clock_time=wall
        )
        tail = tail_probability_accuracy(x, {p: spec.marginal_ppf(p) for p in config.tail_probs}).summary()
        n_steps = np.asarray(res.idata.sample_stats["n_steps"].values)
        grads = int(res.grad_evals)
        min_tail_ess = _finite_or_none(m["min_tail_ess"])
        max_rhat = _finite_or_none(m["split_rhat"])
        row.update(
            total_grad_evals=grads,
            warmup_grad_evals=int(res.phase1_grad_evals + res.phase2_grad_evals),
            sampling_grad_evals=int(res.sampling_grad_evals),
            grads_per_draw=float(np.mean(n_steps)),
            frac_max_tree_depth=float(np.mean(n_steps >= 2**config.max_num_doublings - 1)),
            bulk_ess=_finite_or_none(m["bulk_ess"]),
            min_bulk_ess=_finite_or_none(m["min_bulk_ess"]),
            tail_ess=_finite_or_none(m["tail_ess"]),
            min_tail_ess=min_tail_ess,
            ess_per_grad=_finite_or_none(m["ess_per_grad"]),
            tail_ess_per_grad=_finite_or_none(m["tail_ess_per_grad"]),
            min_tail_ess_per_grad=None if min_tail_ess is None else min_tail_ess / grads,
            max_rhat=max_rhat,
            converged=max_rhat is not None and max_rhat <= RHAT_CONVERGED,
            divergences=int(res.divergences),
            q99_rel_error=_finite_or_none(m.get("q99_rel_error")),
            tail_prob_abs_err=_finite_or_none(tail["tail_prob_abs_err"]),
            tail_prob_abs_z=_finite_or_none(tail["tail_prob_abs_z"]),
            tail_prob_max_abs_z=_finite_or_none(tail["tail_prob_max_abs_z"]),
            tail_prob_n_degenerate=int(tail["tail_prob_n_degenerate"]),
            step_size_median=float(np.median(res.step_size)),
            wall_time_sec=float(wall),
        )
    except Exception as exc:
        row.update(converged=False, wall_time_sec=float(time.perf_counter() - t0), error=f"{type(exc).__name__}: {exc}")
    return row


def _iqr_summary(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    q25, q50, q75 = np.percentile(np.asarray(values, dtype=np.float64), [25.0, 50.0, 75.0])
    return {"median": float(q50), "q25": float(q25), "q75": float(q75), "n": len(values)}


def aggregate_rows(rows: Sequence[dict[str, Any]], baseline: str = BASELINE_ARM) -> list[dict[str, Any]]:
    """Summarises per-seed rows per (target, arm): medians, IQRs, paired ratios to `baseline`, converged fraction.

    Missing (null) values are skipped; a failed run counts as not converged. Ratios pair runs by seed.
    """
    targets = list(dict.fromkeys(r["target"] for r in rows))
    arm_rank = {a: i for i, a in enumerate(ARM_NAMES)}
    order = sorted(
        {(r["target"], r["arm"]) for r in rows},
        key=lambda k: (targets.index(k[0]), arm_rank.get(k[1], len(arm_rank)), k[1]),
    )

    out: list[dict[str, Any]] = []
    for target, arm in order:
        mine = [r for r in rows if r["target"] == target and r["arm"] == arm]
        converged = [r["max_rhat"] is not None and r["max_rhat"] <= RHAT_CONVERGED for r in mine]
        entry: dict[str, Any] = {
            "target": target,
            "arm": arm,
            "n_seeds": len(mine),
            "n_errors": sum(r["error"] is not None for r in mine),
            "frac_converged": sum(converged) / len(mine),
            "metrics": {m: _iqr_summary([r[m] for r in mine if r[m] is not None]) for m in AGG_METRICS},
        }
        if arm != baseline:
            base = {r["seed"]: r for r in rows if r["target"] == target and r["arm"] == baseline}
            ratios: dict[str, Any] = {}
            for m in RATIO_METRICS:
                vals = [
                    r[m] / base[r["seed"]][m]
                    for r in mine
                    if r["seed"] in base and r[m] is not None and base[r["seed"]][m] not in (None, 0.0)
                ]
                ratios[m] = _iqr_summary(vals)
            entry["ratio_vs_stock"] = ratios
        out.append(entry)
    return out


def _git(*args: str) -> str | None:
    repo = Path(__file__).resolve().parent
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip()


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def collect_provenance() -> dict[str, Any]:
    """Git state, platform and library versions of the process that produced the results."""
    status = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "jax": jax.__version__,
            "jaxlib": _version("jaxlib"),
            "blackjax": _version("blackjax"),
            "arviz": _version("arviz"),
            "mcmc_robust": _version("mcmc-robust"),
        },
    }


def _config_record(config: AblationConfig) -> dict[str, Any]:
    rec = asdict(config)
    rec["targets"] = list(config.targets)
    rec["arms"] = list(config.arms)
    rec["seeds"] = list(config.seeds)
    rec["tail_probs"] = list(config.tail_probs)
    rec["arm_targets"] = {k: list(v) for k, v in config.arm_targets.items()}
    rec["arm_settings"] = {a: asdict(ARMS[a]) for a in config.arms}
    rec["rhat_converged"] = RHAT_CONVERGED
    rec["init"] = (
        f"Uniform(-{INIT_HALF_WIDTH}, {INIT_HALF_WIDTH})^d per chain from numpy default_rng(seed), shared by all arms"
    )
    rec["jax_key"] = "jax.random.PRNGKey(seed), shared by all arms"
    rec["target_descriptions"] = {t: make_ablation_target(t).description for t in config.targets}
    return rec


NOTES: tuple[str, ...] = (
    "Every arm runs sample_blackjax_unified_nuts (BlackJAX window adaptation + NUTS) with the listed transform_kind.",
    "total_grad_evals counts every leapfrog gradient including warm-up (and the initial evaluation per chain and phase).",
    "bulk_ess/tail_ess are coordinate means; min_* are the worst coordinate. ESS and accuracy are measured on x.",
    "max_rhat is the maximum rank-normalised split R-hat (ArviZ method='rank'); converged means max_rhat <= 1.01.",
    (
        "tail_prob_*: err = P_hat(x_i < q_p) - p at exact marginal quantiles q_p, z = err / ArviZ MCSE of the "
        "indicator; tail_prob_abs_* average |.| over coordinates and p."
    ),
    "q99_rel_error is the legacy |q99_hat - q99| / |q99| (coordinate mean); it is noise-dominated on Cauchy marginals.",
    "Ratios vs stock are paired by seed (same initial positions and JAX key) and summarised by median and IQR.",
    "wall_time_sec includes JAX compilation and is machine-dependent; do not quote it as an efficiency measure.",
)


def _run_cell(args: tuple[str, str, int, AblationConfig]) -> dict[str, Any]:
    target_name, arm, seed, config = args
    return run_single(make_ablation_target(target_name), arm, seed, config)


def run_ablation(config: AblationConfig, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Runs the grid, writes JSON/CSV to `config.output_dir` and returns the JSON document."""
    if not 1 <= config.workers <= 2:
        raise ValueError(f"workers must be 1 or 2, got {config.workers}")
    for name in config.arms:
        if name not in ARMS:
            raise ValueError(f"Unknown arm {name!r}; choose from {ARM_NAMES}")
    provenance = collect_provenance()
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    specs = {t: make_ablation_target(t) for t in config.targets}
    cells = [
        (t, a, s, config)
        for t in config.targets
        for a in config.arms
        for s in config.seeds
        if config.applies(specs[t], a)
    ]
    rows: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        if log is not None:
            status = row["error"] or " ".join(
                f"{label}={row[key]:{spec}}" if row[key] is not None else f"{label}=null"
                for label, key, spec in (
                    ("tail-ESS/grad", "tail_ess_per_grad", ".3g"),
                    ("grads", "total_grad_evals", "d"),
                    ("rhat", "max_rhat", ".3f"),
                    ("|z|", "tail_prob_abs_z", ".2f"),
                    ("t", "wall_time_sec", ".1f"),
                )
            )
            log(f"[{len(rows)}/{len(cells)}] {row['target']:<12} {row['arm']:<14} seed={row['seed']:<3} {status}")

    if config.workers == 1:
        for cell in cells:
            record(_run_cell(cell))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=ctx) as pool:
            for row in pool.map(_run_cell, cells):
                record(row)

    position = {(t, a, s): i for i, (t, a, s, _) in enumerate(cells)}
    rows.sort(key=lambda r: position[(r["target"], r["arm"], r["seed"])])
    provenance["timestamp_utc"] = started.isoformat()
    provenance["total_wall_time_sec"] = time.perf_counter() - t0

    document = {
        "schema_version": SCHEMA_VERSION,
        "description": "Multi-seed ablation of the calibrated coordinate-wise arcsinh transform for NUTS",
        "notes": list(NOTES),
        "provenance": provenance,
        "config": _config_record(config),
        "rows": rows,
        "aggregates": aggregate_rows(rows),
    }
    write_outputs(document, Path(config.output_dir))
    return json.loads(json.dumps(document, allow_nan=False))


def _flatten_aggregate(agg: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {k: agg[k] for k in ("target", "arm", "n_seeds", "n_errors", "frac_converged")}
    for m in AGG_METRICS:
        stats = agg["metrics"][m] or {}
        for s in ("median", "q25", "q75"):
            flat[f"{m}_{s}"] = stats.get(s)
    for m in RATIO_METRICS:
        stats = (agg.get("ratio_vs_stock") or {}).get(m) or {}
        for s in ("median", "q25", "q75"):
            flat[f"ratio_{m}_{s}"] = stats.get(s)
    return flat


def write_outputs(document: dict[str, Any], output_dir: Path) -> None:
    """Writes `ablation_results.json`, `ablation_results.csv` (rows) and `ablation_summary.csv` (aggregates)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_results.json").open("w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=1, allow_nan=False)
        fh.write("\n")
    with (output_dir / "ablation_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(document["rows"])
    flat = [_flatten_aggregate(a) for a in document["aggregates"]]
    with (output_dir / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        if flat:
            writer = csv.DictWriter(fh, fieldnames=list(flat[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flat)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the ablation runner."""
    parser = argparse.ArgumentParser(
        prog="python -m mcmc_robust.bench.ablation_runner",
        description="Multi-seed ablation of the calibrated arcsinh transform for NUTS.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Reported protocol: 10 seeds, tune 500, draws 1000 (default)")
    mode.add_argument(
        "--quick", action="store_true", help="Smoke test: 2 seeds, short runs (never for reported results)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the seeds of the selected mode")
    parser.add_argument("--targets", nargs="+", choices=ABLATION_TARGET_NAMES, help="Subset of targets")
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, help="Subset of arms")
    parser.add_argument("--output-dir", default="reports", help="Directory for the JSON and CSV outputs")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1, help="Worker processes (at most 2)")
    parser.add_argument(
        "--stock-ta95-targets",
        nargs="+",
        choices=ABLATION_TARGET_NAMES,
        help="Run the stock_ta95 arm on these targets only (to bound compute)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AblationConfig:
    """Turns parsed CLI arguments into an `AblationConfig`."""
    overrides: dict[str, Any] = {"output_dir": args.output_dir, "workers": args.workers}
    if args.seeds:
        overrides["seeds"] = tuple(args.seeds)
    if args.targets:
        overrides["targets"] = tuple(args.targets)
    if args.arms:
        overrides["arms"] = tuple(args.arms)
    if args.stock_ta95_targets:
        overrides["arm_targets"] = {"stock_ta95": tuple(args.stock_ta95_targets)}
    return AblationConfig.quick(**overrides) if args.quick else AblationConfig.full(**overrides)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    config = config_from_args(build_parser().parse_args(argv))
    print(
        f"Ablation [{config.mode}]: {len(config.targets)} targets x {len(config.arms)} arms x {len(config.seeds)} seeds, "
        f"tune={config.tune} draws={config.draws} chains={config.chains} workers={config.workers}",
        flush=True,
    )
    doc = run_ablation(config, log=lambda msg: print(msg, flush=True))
    n_err = sum(r["error"] is not None for r in doc["rows"])
    print(
        f"Done: {len(doc['rows'])} runs ({n_err} errors) in {doc['provenance']['total_wall_time_sec']:.0f}s; "
        f"wrote {config.output_dir}/ablation_results.{{json,csv}} and ablation_summary.csv",
        flush=True,
    )
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
