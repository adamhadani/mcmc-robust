r"""Real posteriors: stock NUTS, arcsinh, Tukey-h and its h = 0 control on every posteriordb posterior with reference draws.

Question: on real models, does the transform pay, and does `tukey_h` match or beat `arcsinh`
without losing accuracy (the evidence needed before making it the default)?

Posteriors: every posterior of posteriordb (`bench/posteriordb.py`, commit `POSTERIORDB_SHA`) that
has reference draws, none selected or excluded by results. Each Stan model is compiled with
BridgeStan and sampled on Stan's unconstrained space through `stan_bridge.bridgestan_logdensity_fn`.

Arms (`ablation_runner.ARMS`; every arm is `sample_blackjax_unified_nuts` with `tune` warm-up steps,
`draws` draws, `chains` chains, at most `max_num_doublings` tree doublings):
- `stock`: transform_kind "identity", target acceptance 0.8.
- `arcsinh`: the calibrated arcsinh pullback (the library default).
- `tukey_h`: the Tukey h-map, warping only coordinates the tail probe finds polynomial on both sides.
- `tukey_h0`: the same pipeline with warping off, so h = 0 everywhere; on posteriors where
  `tukey_h` sets h > 0, the paired ratio `tukey_h` / `tukey_h0` is the effect of those decisions.
Initial positions are Stan's: Uniform(-2, 2) per unconstrained coordinate, redrawn (at most 100 times)
until the log density is finite, from `numpy.random.default_rng(seed)` and shared by all arms.

Per-run metrics (one row per posterior x arm x seed), on the constrained parameters that have reference
draws (posteriordb names, transformed parameters included):
- gradients including warm-up; mean and worst-parameter bulk and tail ESS (ArviZ); ESS per gradient;
  max rank-normalised R-hat; post-warm-up divergences;
- agreement with the reference draws (`diagnostics/tail_accuracy.reference_agreement`): tail
  probabilities at the reference's 1%, 5%, 95% and 99% quantiles, and means, as z-scores that count
  the Monte Carlo error of the run and of the reference;
- the tail-alignment check on the sampled (unconstrained) draws, which the library's warning would see;
- for `tukey_h` (and `tukey_h0`), how many coordinates got $h > 0$ and the largest $h$; for `tukey_h`, how many
  the guard would warp if it started from exact posterior draws instead of Phase 1's: the probe at the median
  and scale of as many reference draws as the Phase 1 pilot has, subsampled with the row's seed and mapped to
  the unconstrained space.

Aggregates per posterior x arm: median and IQR over seeds, and paired per-seed ratios to `stock`,
`arcsinh` and `tukey_h0`.

Usage::

    uv sync --extra posteriordb
    uv run python -m mcmc_robust.bench.posteriordb_runner --fetch   # shallow checkout of POSTERIORDB_SHA
    uv run python -m mcmc_robust.bench.posteriordb_runner [--full | --quick] [--seeds ...]
        [--posteriors ...] [--arms ...] [--workers N] [--posteriordb PATH] [--output-dir reports] [--resume]

Outputs `posteriordb_results.json` (provenance, config, rows, aggregates) and `posteriordb_results.csv`.

Resuming: each finished row is appended to `posteriordb_results.partial.jsonl` in the output directory, which
is deleted once the results are written. If a run stops, the same command with `--resume` runs only the cells
that have no row. It refuses a checkpoint written with a different grid, sampler settings, posteriordb commit,
commit or uncommitted change under `src/`, or library versions, so rows from different code are never mixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import signal
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from mcmc_robust.bench.ablation_runner import (
    ARMS,
    INIT_HALF_WIDTH,
    RHAT_CONVERGED,
    _finite_or_none,
    _git,
    _iqr_summary,
    _version,
    collect_provenance,
)
from mcmc_robust.bench.posteriordb import (
    DEFAULT_POSTERIORDB_PATH,
    POSTERIORDB_SHA,
    PosteriorDatabase,
    fetch_posteriordb,
    stan_name_to_posteriordb,
)
from mcmc_robust.bench.sota_adapters import evaluate_chains_with_arviz
from mcmc_robust.diagnostics.tail_accuracy import REFERENCE_PROBS, reference_agreement
from mcmc_robust.diagnostics.tail_alignment import tail_alignment_diagnostic
from mcmc_robust.jax_unified_nuts import calibrate_jax_tukey_h_transform, phase1_schedule, sample_blackjax_unified_nuts
from mcmc_robust.stan_bridge import StanLogDensity, bridgestan_logdensity_fn

SCHEMA_VERSION: int = 1

#: IQR of a standard normal: IQR / IQR_TO_SD is a normal-consistent scale.
IQR_TO_SD: float = 1.3489795003921634

#: Draws per chain tried before an initial position is given up on (Stan's limit).
MAX_INIT_ATTEMPTS: int = 100

DEFAULT_ARMS: tuple[str, ...] = ("stock", "arcsinh", "tukey_h", "tukey_h0")

#: Arms whose paired per-seed ratios are reported, as the `ratio_vs_<arm>` keys of the aggregates.
RATIO_BASELINES: tuple[str, ...] = ("stock", "arcsinh", "tukey_h0")

ROW_FIELDS: tuple[str, ...] = (
    "posterior",
    "arm",
    "seed",
    "d",
    "k",
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
    "ref_tail_prob_abs_err",
    "ref_tail_prob_abs_z",
    "ref_tail_prob_max_abs_z",
    "ref_tail_prob_n_degenerate",
    "ref_mean_abs_z",
    "ref_mean_max_abs_z",
    "n_nonfinite_draws",
    "tail_alignment_warning",
    "tail_alignment_max_chi",
    "tail_alignment_n_heavy",
    "tukey_h_n_positive",
    "tukey_h_max",
    "tukey_h_ref_positive",
    "calib_location_offset_max",
    "calib_scale_ratio_max",
    "step_size_median",
    "wall_time_sec",
    "error",
)

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
    "ref_tail_prob_abs_z",
    "ref_tail_prob_max_abs_z",
    "ref_mean_abs_z",
    "ref_mean_max_abs_z",
    "tukey_h_n_positive",
    "tukey_h_ref_positive",
    "calib_location_offset_max",
    "calib_scale_ratio_max",
    "wall_time_sec",
)

RATIO_METRICS: tuple[str, ...] = ("min_tail_ess_per_grad", "tail_ess_per_grad", "ess_per_grad", "total_grad_evals")


class StanModelLike(StanLogDensity, Protocol):
    """The part of `bridgestan.StanModel` the runner uses."""

    def param_names(self, *, include_tp: bool = ..., include_gq: bool = ...) -> list[str]: ...

    def param_constrain(self, theta_unc: np.ndarray, *, include_tp: bool = ..., include_gq: bool = ...) -> Any: ...

    def param_unconstrain(self, theta: np.ndarray) -> Any: ...


@dataclass(frozen=True)
class LoadedPosterior:
    """A compiled model with its reference draws and the columns of `param_constrain` they correspond to."""

    name: str
    model: StanModelLike
    reference_names: tuple[str, ...]
    reference_draws: np.ndarray
    columns: tuple[int, ...]
    #: Reference columns of the model's parameters, in `param_unconstrain` order; None if the reference lacks one.
    parameter_columns: tuple[int, ...] | None = None

    @classmethod
    def from_model(
        cls, name: str, model: StanModelLike, reference_names: Sequence[str], reference_draws: np.ndarray
    ) -> LoadedPosterior:
        """Matches each reference name to a parameter or transformed parameter of `model` (posteriordb names)."""
        names = [stan_name_to_posteriordb(n) for n in model.param_names(include_tp=True)]
        missing = [n for n in reference_names if n not in names]
        if missing:
            raise ValueError(f"{name}: reference parameters {missing} are not parameters of the model")
        params = [stan_name_to_posteriordb(n) for n in model.param_names(include_tp=False)]
        return cls(
            name=name,
            model=model,
            reference_names=tuple(reference_names),
            reference_draws=np.asarray(reference_draws, dtype=np.float64),
            columns=tuple(names.index(n) for n in reference_names),
            parameter_columns=(
                tuple(list(reference_names).index(n) for n in params) if set(params) <= set(reference_names) else None
            ),
        )

    @property
    def d(self) -> int:
        """Unconstrained dimension (what NUTS samples)."""
        return int(self.model.param_unc_num())

    @property
    def k(self) -> int:
        """Number of reference parameters (what is scored)."""
        return len(self.reference_names)

    def unconstrained_reference(self, n: int, seed: int) -> np.ndarray | None:
        """`n` reference draws (subsampled with `seed`) on the unconstrained space, or None if the reference lacks
        some parameter of the model."""
        if self.parameter_columns is None:
            return None
        flat = self.reference_draws.reshape(-1, self.k)
        pick = np.random.default_rng(seed).choice(len(flat), size=n, replace=False)
        cols = list(self.parameter_columns)
        return np.stack([np.asarray(self.model.param_unconstrain(np.ascontiguousarray(flat[i, cols]))) for i in pick])

    def constrain(self, unconstrained: np.ndarray) -> np.ndarray:
        """Maps draws of shape `(..., d)` to the reference columns, shape `(..., k)`."""
        flat = np.asarray(unconstrained, dtype=np.float64).reshape(-1, self.d)
        cols = list(self.columns)
        out = np.stack([np.asarray(self.model.param_constrain(u, include_tp=True))[cols] for u in flat])
        return out.reshape(*np.shape(unconstrained)[:-1], self.k)


def load_posterior(db: PosteriorDatabase, name: str) -> LoadedPosterior:
    """Compiles (if needed) and loads posterior `name` with BridgeStan (the `posteriordb` extra)."""
    import bridgestan

    spec = db.posterior(name)
    with warnings.catch_warnings():
        # Posteriors that share a model (posteriordb's nes, earnings, kidiq, ...) load the same shared object;
        # BridgeStan warns that a changed file would not be reloaded, which cannot happen within one run.
        warnings.filterwarnings("ignore", message="Loading a shared object .* that has already been loaded")
        model = bridgestan.StanModel(str(spec.stan_file), data=spec.data_json)
    return LoadedPosterior.from_model(name, model, spec.reference_names, spec.reference_draws)


def initial_positions(model: StanLogDensity, seed: int, chains: int) -> np.ndarray:
    """Stan's initialisation: Uniform(-2, 2) per unconstrained coordinate, redrawn until the density is finite."""
    rng = np.random.default_rng(seed)
    d = int(model.param_unc_num())
    out = np.empty((chains, d))
    for c in range(chains):
        for _ in range(MAX_INIT_ATTEMPTS):
            x = rng.uniform(-INIT_HALF_WIDTH, INIT_HALF_WIDTH, size=d)
            try:
                lp, grad = model.log_density_gradient(x, propto=True, jacobian=True)
            except RuntimeError:
                continue
            if np.isfinite(lp) and np.all(np.isfinite(grad)):
                out[c] = x
                break
        else:
            raise RuntimeError(f"No finite initial position in {MAX_INIT_ATTEMPTS} attempts (chain {c})")
    return out


@dataclass(frozen=True)
class PosteriorDBConfig:
    """Grid and sampler settings (use `full()` for reported results)."""

    posteriors: tuple[str, ...] | None = None
    arms: tuple[str, ...] = DEFAULT_ARMS
    seeds: tuple[int, ...] = tuple(range(10))
    tune: int = 500
    draws: int = 1000
    chains: int = 4
    max_num_doublings: int = 10
    mode: str = "full"
    output_dir: str = "reports"
    workers: int = 1
    posteriordb_path: str = str(DEFAULT_POSTERIORDB_PATH)

    @classmethod
    def full(cls, **overrides: Any) -> PosteriorDBConfig:
        """The reported protocol: every posterior with reference draws, 10 seeds, tune 500, draws 1000, 4 chains."""
        return replace(cls(), **overrides)

    @classmethod
    def quick(cls, **overrides: Any) -> PosteriorDBConfig:
        """Smoke-test sizes (2 seeds, short runs); never for reported results."""
        return replace(cls(mode="quick", seeds=(0, 1), tune=100, draws=100), **overrides)

    def validate(self) -> None:
        for arm in self.arms:
            if arm not in ARMS or ARMS[arm].augmented:
                allowed = [a for a in ARMS if not ARMS[a].augmented]
                raise ValueError(f"Arm {arm!r} cannot run on a Stan model; choose from {allowed}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")


def _empty_row(post: LoadedPosterior, arm: str, seed: int, config: PosteriorDBConfig) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
    row.update(
        posterior=post.name,
        arm=arm,
        seed=int(seed),
        d=post.d,
        k=post.k,
        chains=config.chains,
        draws=config.draws,
        tune=config.tune,
        target_accept=ARMS[arm].target_accept,
        transform_kind=ARMS[arm].transform_kind,
    )
    return row


def _calibration_offsets(transform: Any, samples: np.ndarray) -> tuple[float | None, float | None]:
    """How far Phase 1's (mu, R) sit from the run's post-warm-up draws: the largest |mu - median| / scale and the
    largest max(R / scale, scale / R) over unconstrained coordinates, with scale = IQR / 1.349 of the draws.

    Phase 1 calibrates on the first 15% of warm-up; a large offset means it saw draws that had not yet reached
    the posterior.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1, samples.shape[-1])
    scale = (np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)) / IQR_TO_SD
    mu, radius = np.asarray(transform.location), np.asarray(transform.radius)
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.max(np.abs(mu - np.median(x, axis=0)) / scale)
        ratio = np.max(np.maximum(radius / scale, scale / radius))
    return _finite_or_none(offset), _finite_or_none(ratio)


def _guard_on_reference(post: LoadedPosterior, seed: int, config: PosteriorDBConfig) -> int | None:
    """How many coordinates get h > 0 when the guard starts from exact draws: the tail probe at the median and scale
    of as many reference draws as the Phase 1 pilot has, then the fit on them."""
    w1, start = phase1_schedule(config.tune)
    draws = post.unconstrained_reference(config.chains * (w1 - start), seed)
    if draws is None:
        return None
    cal = calibrate_jax_tukey_h_transform(draws, bridgestan_logdensity_fn(post.model))
    return int(np.sum(cal.h > 0.0))


def run_single(post: LoadedPosterior, arm: str, seed: int, config: PosteriorDBConfig) -> dict[str, Any]:
    """Runs one (posterior, arm, seed) cell and returns its row (`error` holds the message if it failed)."""
    arm_spec = ARMS[arm]
    row = _empty_row(post, arm, seed, config)
    t0 = time.perf_counter()
    try:
        x0 = initial_positions(post.model, seed, config.chains)
        res = sample_blackjax_unified_nuts(
            bridgestan_logdensity_fn(post.model),
            d=post.d,
            x0=x0,
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            seed=int(seed),
            target_accept=arm_spec.target_accept,
            max_num_doublings=config.max_num_doublings,
            transform_kind=arm_spec.transform_kind,
            tukey_h_warp=arm_spec.tukey_h_warp,
            # Scored below by the runner, without a warning that `-W error` would turn into an error row.
            check_tail_alignment=False,
        )
        wall = time.perf_counter() - t0
        constrained = post.constrain(res.samples)
        m = evaluate_chains_with_arviz(constrained, res.grad_evals, divergences=res.divergences)
        agreement = reference_agreement(constrained, post.reference_draws).summary()
        alignment = tail_alignment_diagnostic(res.samples)
        n_steps = np.asarray(res.idata.sample_stats["n_steps"].values)
        grads = int(res.grad_evals)
        min_tail_ess = _finite_or_none(m["min_tail_ess"])
        max_rhat = _finite_or_none(m["split_rhat"])
        h = np.asarray(getattr(res.transform, "h", np.zeros(0)))
        tukey = arm_spec.transform_kind == "tukey_h"
        calibrated = arm_spec.transform_kind != "identity"
        offset, ratio = _calibration_offsets(res.transform, res.samples) if calibrated else (None, None)
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
            **{key: (int(v) if key.endswith("n_degenerate") else _finite_or_none(v)) for key, v in agreement.items()},
            n_nonfinite_draws=int(np.sum(~np.all(np.isfinite(constrained), axis=-1))),
            tail_alignment_warning=alignment.warn,
            tail_alignment_max_chi=_finite_or_none(alignment.max_chi),
            tail_alignment_n_heavy=len(alignment.heavy_coordinates),
            tukey_h_n_positive=int(np.sum(h > 0.0)) if tukey else None,
            tukey_h_max=float(np.max(h)) if tukey and h.size else None,
            tukey_h_ref_positive=(_guard_on_reference(post, seed, config) if tukey and arm_spec.tukey_h_warp else None),
            calib_location_offset_max=offset,
            calib_scale_ratio_max=ratio,
            step_size_median=float(np.median(res.step_size)),
            wall_time_sec=float(wall),
        )
    except Exception as exc:
        row.update(converged=False, wall_time_sec=float(time.perf_counter() - t0), error=f"{type(exc).__name__}: {exc}")
    return row


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (posterior, arm): medians and IQRs over seeds, paired per-seed ratios to each `RATIO_BASELINES` arm."""
    posteriors = list(dict.fromkeys(r["posterior"] for r in rows))
    arm_rank = {a: i for i, a in enumerate(ARMS)}
    order = sorted(
        {(r["posterior"], r["arm"]) for r in rows},
        key=lambda k: (posteriors.index(k[0]), arm_rank.get(k[1], len(arm_rank)), k[1]),
    )
    out: list[dict[str, Any]] = []
    for posterior, arm in order:
        mine = [r for r in rows if r["posterior"] == posterior and r["arm"] == arm]
        converged = [r["max_rhat"] is not None and r["max_rhat"] <= RHAT_CONVERGED for r in mine]
        entry: dict[str, Any] = {
            "posterior": posterior,
            "arm": arm,
            "n_seeds": len(mine),
            "n_errors": sum(r["error"] is not None for r in mine),
            "frac_converged": sum(converged) / len(mine),
            "metrics": {m: _iqr_summary([r[m] for r in mine if r[m] is not None]) for m in AGG_METRICS},
        }
        for baseline in RATIO_BASELINES:
            if arm == baseline:
                continue
            base = {r["seed"]: r for r in rows if r["posterior"] == posterior and r["arm"] == baseline}
            if not base:
                continue
            entry[f"ratio_vs_{baseline}"] = {
                m: _iqr_summary(
                    [
                        r[m] / base[r["seed"]][m]
                        for r in mine
                        if r["seed"] in base and r[m] is not None and base[r["seed"]][m] not in (None, 0.0)
                    ]
                )
                for m in RATIO_METRICS
            }
        out.append(entry)
    return out


NOTES: tuple[str, ...] = (
    "Every arm runs sample_blackjax_unified_nuts on Stan's unconstrained space via BridgeStan (propto, Jacobian).",
    "ESS, R-hat and reference agreement are computed on the constrained parameters that have reference draws.",
    "min_* ESS is the worst reference parameter; converged means max rank-normalised R-hat <= 1.01.",
    (
        "ref_*: agreement with posteriordb's reference draws; tail probabilities at the reference's 1/5/95/99% "
        "quantiles and means, z = (run - reference) / sqrt(MCSE_run^2 + MCSE_ref^2), averaged (abs) or maximal."
    ),
    "tail_alignment_* is the library's tail-alignment check on the sampled (unconstrained) draws.",
    "tukey_h_n_positive counts unconstrained coordinates whose calibrated h is positive (the guard warped them).",
    (
        "tukey_h_ref_positive: coordinates given h > 0 when the guard's probe starts at the median and scale of as many "
        "reference draws as Phase 1 calibrates from (subsampled with the row's seed, unconstrained); tukey_h only, "
        "null when the reference lacks a parameter of the model."
    ),
    (
        "calib_*: Phase 1's location and scale against the run's own post-warm-up draws (unconstrained, scale = "
        "IQR / 1.349): max |mu - median| / scale and max of R / scale or scale / R. Null for stock."
    ),
    "Ratios are paired by seed (same initial positions and JAX key) and summarised by median and IQR.",
    (
        "wall_time_sec includes JAX compilation and the host callbacks; vectorised chains also evaluate the model for "
        "chains whose trajectory has ended. Do not quote it as an efficiency measure."
    ),
)


_WORKER_DB: PosteriorDatabase | None = None
_WORKER_CACHE: dict[str, LoadedPosterior] = {}


def _default_loader(path: str) -> Callable[[str], LoadedPosterior]:
    def load(name: str) -> LoadedPosterior:
        global _WORKER_DB
        if name not in _WORKER_CACHE:
            if _WORKER_DB is None:
                _WORKER_DB = PosteriorDatabase(path)
            _WORKER_CACHE[name] = load_posterior(_WORKER_DB, name)
        return _WORKER_CACHE[name]

    return load


def _ignore_interrupts() -> None:
    """Worker initializer: Ctrl-C reaches the whole process group, and only the parent should act on it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _terminate_workers(pool: ProcessPoolExecutor) -> None:
    terminate = getattr(pool, "terminate_workers", None)  # Python >= 3.14
    if terminate is not None:
        terminate()
        return
    for process in list(pool._processes.values()):
        process.terminate()


def _run_cell(args: tuple[str, str, int, PosteriorDBConfig]) -> dict[str, Any]:
    name, arm, seed, config = args
    try:
        post = _default_loader(config.posteriordb_path)(name)
    except Exception as exc:
        row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
        row.update(posterior=name, arm=arm, seed=seed, converged=False, error=f"load: {type(exc).__name__}: {exc}")
        return row
    return run_single(post, arm, seed, config)


def _config_record(config: PosteriorDBConfig, posteriors: Sequence[str]) -> dict[str, Any]:
    rec = asdict(config)
    rec.update(
        posteriors=list(posteriors),
        arms=list(config.arms),
        seeds=list(config.seeds),
        arm_settings={a: asdict(ARMS[a]) for a in config.arms},
        reference_probs=list(REFERENCE_PROBS),
        rhat_converged=RHAT_CONVERGED,
        init=(
            f"Uniform(-{INIT_HALF_WIDTH}, {INIT_HALF_WIDTH})^d per chain from numpy default_rng(seed), redrawn "
            f"up to {MAX_INIT_ATTEMPTS} times until the log density is finite; shared by all arms"
        ),
    )
    rec.pop("posteriordb_path")
    return rec


#: Rows of an unfinished run, one JSON record per line, in the output directory (see the module docstring).
CHECKPOINT_NAME: str = "posteriordb_results.partial.jsonl"


def _code_identity() -> dict[str, Any]:
    """The commit and a hash of any uncommitted change under `src/`: the code a resumed run must share."""
    diff = _git("diff", "HEAD", "--", ":/src")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "src_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def _run_identity(
    config: PosteriorDBConfig, names: Sequence[str], posteriordb_sha: str | None, versions: dict[str, str]
) -> dict[str, Any]:
    """Everything that determines the rows: the grid and sampler settings, the data, the code and the libraries."""
    rec = _config_record(config, names)
    for key in ("workers", "output_dir"):
        rec.pop(key)
    return {"config": rec, "posteriordb_sha": posteriordb_sha, **_code_identity(), "versions": versions}


def _identity_differences(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    keys = sorted(set(old) | set(new))
    nested = [f"{k}.{sub}" for k in keys if k == "config" for sub in _identity_differences(old[k], new[k])]
    return nested + [k for k in keys if k != "config" and old.get(k) != new.get(k)]


def _write_record(fh: Any, record: dict[str, Any]) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _read_records(path: Path) -> list[dict[str, Any]]:
    """The checkpoint's records; a last line cut off mid-write by a killed process is dropped (that cell reruns)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    return records


def _segments_and_rows(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each segment's start, wall time up to its last row and row count, and all rows in completion order."""
    segments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec["kind"] == "segment":
            segments.append({"started_utc": rec["started_utc"], "wall_time_sec": 0.0, "rows": 0})
        elif rec["kind"] == "row":
            seg = segments[-1]
            elapsed = datetime.fromisoformat(rec["completed_utc"]) - datetime.fromisoformat(seg["started_utc"])
            seg.update(wall_time_sec=elapsed.total_seconds(), rows=seg["rows"] + 1)
            rows.append(rec["row"])
    return segments, rows


def _open_checkpoint(
    path: Path, identity: dict[str, Any], provenance: dict[str, Any], resume: bool
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Starts or continues the checkpoint; returns its open handle, the run's provenance, segments and rows."""
    records: list[dict[str, Any]] = [{"kind": "header", "identity": identity, "provenance": provenance}]
    if path.exists():
        if not resume:
            raise RuntimeError(
                f"{path} holds the rows of an unfinished run; pass --resume to finish it or delete it to start over"
            )
        records = _read_records(path)
        differences = _identity_differences(records[0]["identity"], identity)
        if differences:
            raise ValueError(
                f"{path} was written by a run that differs in {', '.join(differences)}; "
                "finish it with the same code and settings, or delete it to start over"
            )
    # Rewritten without any cut-off last line, so that appending starts on a fresh line.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(rec) + "\n" for rec in records), encoding="utf-8")
    os.replace(tmp, path)
    segments, rows = _segments_and_rows(records)
    return path.open("a", encoding="utf-8"), records[0]["provenance"], segments, rows


def run_benchmark(
    config: PosteriorDBConfig,
    loader: Callable[[str], LoadedPosterior] | None = None,
    posteriordb_sha: str | None = None,
    log: Callable[[str], None] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Runs the grid, writes JSON and CSV to `config.output_dir` and returns the JSON document.

    `loader` maps a posterior name to a `LoadedPosterior` (default: BridgeStan on the posteriordb checkout
    at `config.posteriordb_path`, which also supplies `posteriordb_sha` and the default posterior list).
    A custom loader runs in this process (`workers` is then ignored).

    Rows are checkpointed as they finish. With `resume`, a checkpoint left by an interrupted run with the same
    identity (`_run_identity`) is continued; without it, an existing checkpoint is an error.
    """
    config.validate()
    provenance = collect_provenance()
    provenance["versions"]["bridgestan"] = _version("bridgestan")
    if loader is None:
        db = PosteriorDatabase(config.posteriordb_path)
        posteriordb_sha = db.git_sha()
        names = list(config.posteriors) if config.posteriors is not None else db.reference_posterior_names()
    else:
        if config.posteriors is None:
            raise ValueError("A custom loader needs config.posteriors")
        names = list(config.posteriors)
    provenance["posteriordb_sha"] = posteriordb_sha

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / CHECKPOINT_NAME
    identity = _run_identity(config, names, posteriordb_sha, provenance["versions"])
    fh, provenance, segments, rows = _open_checkpoint(checkpoint, identity, provenance, resume)
    started = datetime.now(UTC)
    segments.append({"started_utc": started.isoformat(), "wall_time_sec": 0.0, "rows": 0})
    _write_record(fh, {"kind": "segment", "started_utc": started.isoformat()})

    all_cells = [(n, a, s) for n in names for a in config.arms for s in config.seeds]
    done = {(r["posterior"], r["arm"], r["seed"]) for r in rows}
    cells = [(n, a, s, config) for n, a, s in all_cells if (n, a, s) not in done]
    if log is not None and done:
        log(f"Resuming: {len(done)} of {len(all_cells)} cells already have rows")

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        segments[-1]["rows"] += 1
        _write_record(fh, {"kind": "row", "completed_utc": datetime.now(UTC).isoformat(), "row": row})
        if log is not None:
            status = row["error"] or " ".join(
                f"{label}={row[key]:{spec}}" if row[key] is not None else f"{label}=null"
                for label, key, spec in (
                    ("min-tail-ESS/grad", "min_tail_ess_per_grad", ".3g"),
                    ("grads", "total_grad_evals", "d"),
                    ("rhat", "max_rhat", ".3f"),
                    ("div", "divergences", "d"),
                    ("ref|z|max", "ref_tail_prob_max_abs_z", ".2f"),
                    ("t", "wall_time_sec", ".1f"),
                )
            )
            log(f"[{len(rows)}/{len(all_cells)}] {row['posterior']:<44} {row['arm']:<8} seed={row['seed']:<3} {status}")

    try:
        if loader is not None:
            for name, arm, seed, _ in cells:
                record(run_single(loader(name), arm, seed, config))
        elif config.workers == 1:
            for cell in cells:
                record(_run_cell(cell))
        else:
            pool = ProcessPoolExecutor(
                max_workers=config.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_ignore_interrupts,
            )
            try:
                futures = [pool.submit(_run_cell, cell) for cell in cells]
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                # Ctrl-C or a failure: stop now and leave the unfinished cells to --resume, rather than waiting
                # for every queued cell at interpreter exit.
                _terminate_workers(pool)
                raise
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    finally:
        fh.close()

    position = {cell: i for i, cell in enumerate(all_cells)}
    rows.sort(key=lambda r: position[(r["posterior"], r["arm"], r["seed"])])
    segments[-1]["wall_time_sec"] = (datetime.now(UTC) - started).total_seconds()
    provenance["timestamp_utc"] = segments[0]["started_utc"]
    provenance["total_wall_time_sec"] = sum(s["wall_time_sec"] for s in segments)
    provenance["segments"] = segments
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "posteriordb",
        "provenance": provenance,
        "config": _config_record(config, names),
        "notes": list(NOTES),
        "rows": rows,
        "aggregates": aggregate_rows(rows),
    }

    (out / "posteriordb_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (out / "posteriordb_results.csv").open("w", encoding="utf-8", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=list(ROW_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    checkpoint.unlink()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="reported protocol (default)")
    mode.add_argument("--quick", action="store_true", help="smoke-test sizes")
    parser.add_argument(
        "--fetch", action="store_true", help=f"check out posteriordb at {POSTERIORDB_SHA[:12]} and exit"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--posteriors", nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--posteriordb", default=str(DEFAULT_POSTERIORDB_PATH), help="posteriordb checkout")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--resume", action="store_true", help=f"finish an interrupted run from {CHECKPOINT_NAME} in the output dir"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PosteriorDBConfig:
    overrides: dict[str, Any] = {
        "output_dir": args.output_dir,
        "workers": args.workers,
        "posteriordb_path": args.posteriordb,
    }
    if args.seeds:
        overrides["seeds"] = tuple(args.seeds)
    if args.posteriors:
        overrides["posteriors"] = tuple(args.posteriors)
    if args.arms:
        overrides["arms"] = tuple(args.arms)
    return PosteriorDBConfig.quick(**overrides) if args.quick else PosteriorDBConfig.full(**overrides)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.fetch:
        print(f"posteriordb at {fetch_posteriordb(args.posteriordb)}")
        return
    config = config_from_args(args)
    result = run_benchmark(config, log=lambda line: print(line, flush=True), resume=args.resume)
    errors = sum(r["error"] is not None for r in result["rows"])
    print(f"{len(result['rows'])} runs, {errors} errors -> {config.output_dir}/posteriordb_results.json")


if __name__ == "__main__":
    main(sys.argv[1:])
