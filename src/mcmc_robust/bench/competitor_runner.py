r"""Direct-competitor benchmark: the repo's arcsinh NUTS against stock NUTS and nutpie.

Writes ``reports/competitor_results.{json,csv}``:
- one row per (target, arm, seed), with the full adapter telemetry in the JSON;
- aggregates per (target, arm): median, IQR, min and max over seeds, and paired per-seed ratios
  against ``stock_nuts`` (``aggregate_rows`` recomputes them from the rows);
- provenance: git SHA, platform, package versions (nutpie's included), seeds and configuration.

Run the full protocol (5 seeds, tune 500, draws 1000, 4 chains, maximum tree depth 10) with

    uv sync --extra competitors
    uv run python -m mcmc_robust.bench.competitor_runner

Large runs can be split into shards that each append to a resumable checkpoint, then merged:

    uv run python -m mcmc_robust.bench.competitor_runner --targets task1 task3 --checkpoint a.jsonl --no-report
    uv run python -m mcmc_robust.bench.competitor_runner --targets corr_t1 corr_gauss --checkpoint b.jsonl --no-report
    uv run python -m mcmc_robust.bench.competitor_runner --merge a.jsonl b.jsonl

Targets (``make_target``):
- ``task1``: `SparseSignalRegressionTarget(d=10, n_obs=40, n_active=3, seed=42)`, the SOTA Task 1
  target: three Gaussian-like active coefficients and seven exactly Cauchy inactive ones.
- ``task3``: `HeterogeneousHeavyTailedTarget(d=20)`, ten Student-t coordinates with
  ``nu = linspace(1.2, 2.5, 10)`` and ten Gaussian ones (SOTA Task 3).
- ``corr_t1``: multivariate t with ``nu = 1`` (Cauchy), d = 10, rotated scale matrix with
  condition number 25.
- ``corr_gauss``: Gaussian, d = 20, rotated covariance with condition number 100.

Every target has exact marginal quantiles (``exact_marginal_quantiles``), so accuracy is scored by
the tail-probability error at exact quantiles (``tail_probability_errors``), with its Monte Carlo
standard error, rather than by the noisy relative error of the 99% quantile.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import numpy as np
from scipy import stats

from mcmc_robust.bench.competitors import (
    COMPETITORS_EXTRA_HINT,
    DEFAULT_MAX_DEPTH,
    DEFAULT_TARGET_ACCEPT,
    AdapterResult,
    CompetitorUnavailableError,
    nutpie_available,
    nutpie_flow_available,
    run_arcsinh_nuts,
    run_nutpie_diag,
    run_nutpie_flow,
    run_stock_nuts,
)
from mcmc_robust.bench.sota_adapters import SparseSignalRegressionTarget, evaluate_chains_with_arviz
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.targets.benign import DiagonalGaussian, FullCovarianceGaussian, IsotropicGaussian
from mcmc_robust.targets.heavy_tailed import HeterogeneousHeavyTailedTarget, MultivariateStudentTTarget

TARGET_NAMES: tuple[str, ...] = ("task1", "task3", "corr_t1", "corr_gauss")

TARGET_DESCRIPTIONS: dict[str, str] = {
    "task1": "SparseSignalRegressionTarget(d=10, n_obs=40, n_active=3, seed=42): 3 active coefficients, 7 exactly Cauchy",
    "task3": "HeterogeneousHeavyTailedTarget(d=20): Student-t coordinates 0-9 with nu=linspace(1.2, 2.5, 10), Gaussian 10-19",
    "corr_t1": "MultivariateStudentTTarget(d=10, nu=1), scale = rotated_covariance(10, 25, seed=0)",
    "corr_gauss": "FullCovarianceGaussian(d=20), cov = rotated_covariance(20, 100, seed=0)",
}

#: Arm every paired ratio is taken against.
REFERENCE_ARM: str = "stock_nuts"

#: nutpie's default warm-up length for flow adaptation (nuts-rs ``FlowNutsSettings::default``).
NUTPIE_FLOW_DEFAULT_TUNE: int = 1500

#: Probabilities at which the tail-probability error is scored.
TAIL_PROBS: tuple[float, ...] = (0.01, 0.99)

#: Split R-hat at or below which a run counts as converged (Vehtari et al. 2021).
RHAT_CONVERGED: float = 1.01

#: Salt for the per-seed initial positions, so they do not reuse the samplers' own random streams.
INIT_SEED_SALT: int = 20261024


@dataclass(frozen=True)
class ArmSpec:
    """One benchmark arm.

    Attributes:
        tune: Warm-up length for this arm, overriding the configuration's ``tune`` (ignored in
            ``quick`` mode).
        requires: ``"nutpie"`` or ``"nutpie_flow"`` when the arm needs the optional extra.
    """

    name: str
    adapter: Callable[..., AdapterResult]
    description: str
    tune: int | None = None
    requires: str | None = None


ARMS: dict[str, ArmSpec] = {
    "stock_nuts": ArmSpec("stock_nuts", run_stock_nuts, "BlackJAX NUTS, Stan-style window adaptation (reference)"),
    "arcsinh_nuts": ArmSpec(
        "arcsinh_nuts",
        run_arcsinh_nuts,
        "sample_blackjax_unified_nuts: BlackJAX NUTS on the calibrated coordinate-wise arcsinh pullback (this repo)",
    ),
    "nutpie_diag": ArmSpec(
        "nutpie_diag",
        run_nutpie_diag,
        "nutpie default: NUTS with gradient-based diagonal mass-matrix adaptation",
        requires="nutpie",
    ),
    "nutpie_flow": ArmSpec(
        "nutpie_flow",
        run_nutpie_flow,
        "nutpie normalizing-flow adaptation (default flow settings), same warm-up length as the other arms",
        requires="nutpie_flow",
    ),
    "nutpie_flow_tune1500": ArmSpec(
        "nutpie_flow_tune1500",
        run_nutpie_flow,
        "nutpie normalizing-flow adaptation with nutpie's default flow warm-up of 1500 draws (all counted)",
        tune=NUTPIE_FLOW_DEFAULT_TUNE,
        requires="nutpie_flow",
    ),
}

ARM_NAMES: tuple[str, ...] = tuple(ARMS)

#: Competitors that were looked for but not run, and why.
EXCLUDED_ARMS: dict[str, dict[str, Any]] = {
    "dcs_bresar_mijatovic_2026": {
        "paper": (
            "Bresar & Mijatovic (2026), 'Diffeomorphic Markov Chain Monte Carlo: fast mixing for heavy-tailed "
            "distributions', arXiv:2608.04284v1 (submitted 4 Aug 2026)"
        ),
        "status": "not run: no official code found",
        "searched_on": "2026-10-24",
        "searched": [
            "arXiv abstract, HTML full text and PDF of 2608.04284v1: no code, repository or supplementary link",
            "arXiv 'Code, Data, Media' tab: only generic links, no ancillary files; Hugging Face papers page: 404",
            (
                "GitHub: no repositories for 'diffeomorphic mcmc', 'diffeomorphic contraction sampler', "
                "'contraction sampler'; github.com/aleks-mijatovic holds only a website repository"
            ),
            (
                "Authors' pages (Warwick staff page, aleks-mijatovic.github.io) and the CUHK and PolyU talk "
                "pages: arXiv and YouTube links only"
            ),
            "Related preprint arXiv:2512.18255 (Bresar, Mijatovic, Roberts): no code",
        ],
        "why_no_reimplementation": (
            "The paper specifies the radial contraction, the Mobius recentring and the ball-walk and hit-and-run "
            "kernels, but not the variational-tuning settings (steps, batch size, learning rate, initial values), "
            "the ball-walk radius, the slice-sampler settings, or the beta, mu and R used for its posteriordb runs. "
            "Its published comparisons are at matched wall time, so a home-made version could not be checked "
            "against them and would not be 'their method'."
        ),
    }
}

METRIC_DEFINITIONS: dict[str, str] = {
    "total_grad_evals": "target gradient evaluations, warm-up and adaptation or flow training included (see telemetry.grad_eval_accounting)",
    "tail_ess": "coordinate mean of ArviZ rank-normalised tail-ESS",
    "min_tail_ess": "worst-coordinate tail-ESS",
    "tail_ess_per_grad": "tail_ess / total_grad_evals",
    "min_tail_ess_per_grad": "min_tail_ess / total_grad_evals",
    "tail_ess_per_sec": "tail_ess / wall_clock_time",
    "split_rhat": "max over coordinates of rank-normalised split R-hat",
    "combined_rhat": "max of split and folded R-hat over coordinates",
    "converged": f"split_rhat <= {RHAT_CONVERGED}",
    "divergences": "post-warm-up divergent transitions, all chains",
    "tail_prob_abs_err_p01": "mean over coordinates of |P_hat(x_j <= q_j(0.01)) - 0.01|, q_j the exact marginal quantile",
    "tail_prob_abs_err_p99": "mean over coordinates of |P_hat(x_j > q_j(0.99)) - 0.01|",
    "tail_prob_mcse_*": "mean over coordinates of the MCSE of that tail-indicator mean (ArviZ, ESS-based); if a tail is never visited, the i.i.d. binomial standard error at the true probability",
    "tail_prob_max_abs_z_*": "max over coordinates of |error| / MCSE",
    "tail_prob_rms_z_*": "root mean square over coordinates of error / MCSE (about 1 when MCSEs are honest)",
    "q99_rel_error": "mean over coordinates of |q_hat(0.99) - q(0.99)| / |q(0.99)| (noisy on Cauchy-like coordinates)",
    "wall_clock_time": "seconds for the whole run, JIT compilation included",
}

NOTES: tuple[str, ...] = (
    (
        "Every arm gets the same per-seed initial positions, Uniform(-2, 2)^d per chain (initial_positions); "
        "nutpie assigns the rows to chains in the order it asks for them."
    ),
    (
        "Every arm uses maximum tree depth 10 and target acceptance 0.8. nutpie_flow keeps nutpie's flow "
        "default divergence threshold (max_energy_error 20, versus 1000 in the other arms), so its divergence "
        "counts are not directly comparable."
    ),
    (
        "nutpie runs through nutpie.compiled_pyfunc.from_pyfunc: each gradient is a Python call into a jitted "
        "JAX function, nutpie's own JAX-backend route. That per-gradient overhead is absent from nutpie's "
        "compiled numba and Stan backends, so nutpie's tail-ESS per second here understates what those "
        "backends reach."
    ),
    (
        "Wall-clock times include JIT compilation and, for the flow arms, flow compilation and training. Chains "
        "run one after another in every arm except arcsinh_nuts, which vmaps its chains."
    ),
    (
        "Paired ratios divide an arm's per-seed metric by stock_nuts' on the same target and seed; their "
        "median and IQR are over seeds."
    ),
)

ROW_FIELDS: tuple[str, ...] = (
    "target",
    "arm",
    "seed",
    "d",
    "tune",
    "draws",
    "chains",
    "max_depth",
    "target_accept",
    "status",
    "error",
    "total_grad_evals",
    "warmup_grad_evals",
    "draw_grad_evals",
    "divergences",
    "warmup_divergences",
    "wall_clock_time",
    "bulk_ess",
    "tail_ess",
    "min_bulk_ess",
    "min_tail_ess",
    "split_rhat",
    "combined_rhat",
    "converged",
    "ess_per_grad",
    "tail_ess_per_grad",
    "min_tail_ess_per_grad",
    "tail_ess_per_sec",
    "q99_rel_error",
    "tail_prob_abs_err_p01",
    "tail_prob_mcse_p01",
    "tail_prob_max_abs_z_p01",
    "tail_prob_rms_z_p01",
    "tail_prob_abs_err_p99",
    "tail_prob_mcse_p99",
    "tail_prob_max_abs_z_p99",
    "tail_prob_rms_z_p99",
    "leapfrog_steps",
    "flow_training_seconds",
)

AGGREGATE_METRICS: tuple[str, ...] = (
    "tail_ess_per_grad",
    "tail_ess_per_sec",
    "tail_ess",
    "min_tail_ess",
    "min_tail_ess_per_grad",
    "total_grad_evals",
    "wall_clock_time",
    "split_rhat",
    "combined_rhat",
    "divergences",
    "tail_prob_abs_err_p01",
    "tail_prob_abs_err_p99",
    "tail_prob_mcse_p01",
    "tail_prob_mcse_p99",
    "tail_prob_max_abs_z_p01",
    "tail_prob_max_abs_z_p99",
    "q99_rel_error",
)

RATIO_METRICS: tuple[str, ...] = ("tail_ess_per_grad", "tail_ess_per_sec", "min_tail_ess_per_grad")

#: Configuration keys that must agree before checkpoint rows can be combined.
_CHECKPOINT_KEYS: tuple[str, ...] = ("tune", "draws", "chains", "max_depth", "target_accept", "quick")


# =============================================================================
# 1. Targets, exact quantiles and initial positions
# =============================================================================


def rotated_covariance(d: int, condition_number: float, seed: int = 0) -> np.ndarray:
    """Covariance ``Q diag(geomspace(1, condition_number, d)) Q^T`` with a random rotation ``Q``."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = np.geomspace(1.0, condition_number, d)
    cov = (q * eig) @ q.T
    return 0.5 * (cov + cov.T)


def make_target(name: str) -> TargetDistribution:
    """Builds one of the benchmark targets in `TARGET_NAMES`."""
    if name == "task1":
        return SparseSignalRegressionTarget(d=10, n_obs=40, n_active=3, seed=42)
    if name == "task3":
        return HeterogeneousHeavyTailedTarget(
            d=20, heavy_coords=list(range(10)), nu_list=list(np.linspace(1.2, 2.5, 10))
        )
    if name == "corr_t1":
        return MultivariateStudentTTarget(d=10, nu=1.0, scale=rotated_covariance(10, 25.0, seed=0))
    if name == "corr_gauss":
        return FullCovarianceGaussian(d=20, cov=rotated_covariance(20, 100.0, seed=0))
    raise KeyError(f"unknown target {name!r}; expected one of {TARGET_NAMES}")


def exact_marginal_quantiles(target: TargetDistribution, probs: Iterable[float]) -> dict[float, np.ndarray]:
    """Exact per-coordinate marginal quantiles ``q_j(p)`` for the benchmark target families."""
    out: dict[float, np.ndarray] = {}
    for p in probs:
        p = float(p)
        if isinstance(target, SparseSignalRegressionTarget):
            # The posterior factorises: active coordinates have tabulated exact CDFs, inactive
            # coordinates (zero design columns) are exactly Cauchy(0, prior_scale).
            q = np.empty(target.d, dtype=np.float64)
            for j in range(target.d):
                if j < len(target._active_grids):
                    q[j] = float(np.interp(p, target._active_cdfs[j], target._active_grids[j]))
                else:
                    q[j] = float(stats.cauchy.ppf(p, loc=0.0, scale=target.prior_scale))
        elif isinstance(target, HeterogeneousHeavyTailedTarget):
            q = target.mu + target.scale_arr * stats.norm.ppf(p)
            for idx, c in enumerate(target.heavy_coords):
                q[c] = target.mu[c] + target.scale_arr[c] * stats.t.ppf(p, df=target.nu_arr[idx])
        elif isinstance(target, MultivariateStudentTTarget):
            q = target.mu + np.sqrt(np.diag(target.cov)) * stats.t.ppf(p, df=target.nu)
        elif isinstance(target, (FullCovarianceGaussian, DiagonalGaussian, IsotropicGaussian)):
            q = np.asarray(target.mean, dtype=np.float64) + np.sqrt(np.diag(target.cov)) * stats.norm.ppf(p)
        else:
            raise NotImplementedError(f"no exact marginal quantiles for {type(target).__name__}")
        out[p] = np.asarray(q, dtype=np.float64)
    return out


def initial_positions(seed: int, chains: int, d: int) -> np.ndarray:
    """Per-seed initial positions shared by every arm: Uniform(-2, 2)^d for each chain."""
    rng = np.random.default_rng([INIT_SEED_SALT, int(seed)])
    return rng.uniform(-2.0, 2.0, size=(chains, d))


# =============================================================================
# 2. Metrics
# =============================================================================


def _prob_tag(p: float) -> str:
    return f"p{round(100 * p):02d}"


def _finite_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _mcse_of_mean(indicator: np.ndarray) -> np.ndarray:
    """ArviZ MCSE of each coordinate's mean for an array of shape ``(chains, draws, d)``."""
    data = az.from_dict({"posterior": {"ind": indicator}})
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(az.mcse(data, var_names=["ind"], method="mean")["ind"].values, dtype=np.float64)


def tail_probability_errors(samples: np.ndarray, quantiles: Mapping[float, np.ndarray]) -> dict[str, float]:
    """Tail-probability error at exact quantiles, with its Monte Carlo standard error.

    For ``p < 0.5`` the tail event is ``x_j <= q_j(p)`` (true probability ``p``); for ``p >= 0.5``
    it is ``x_j > q_j(p)`` (true probability ``1 - p``). For each coordinate the error is the
    estimated minus the true probability and the MCSE comes from ArviZ's ESS of the indicator
    chain. A tail that is never (or always) visited has no ESS; its MCSE falls back to the i.i.d.
    binomial standard error at the true probability, so the z-score is still finite and large.

    Returns, per probability tag (``p01``, ``p99``): ``tail_prob_abs_err_*`` (mean absolute error),
    ``tail_prob_mcse_*`` (mean MCSE), ``tail_prob_max_abs_z_*`` and ``tail_prob_rms_z_*``.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"samples must have shape (chains, draws, d), got {arr.shape}")
    n_total = arr.shape[0] * arr.shape[1]
    out: dict[str, float] = {}
    for p, q in quantiles.items():
        q_arr = np.asarray(q, dtype=np.float64)
        if p < 0.5:
            indicator = (arr <= q_arr).astype(np.float64)
            p_true = float(p)
        else:
            indicator = (arr > q_arr).astype(np.float64)
            p_true = 1.0 - float(p)
        err = indicator.mean(axis=(0, 1)) - p_true
        mcse = _mcse_of_mean(indicator)
        floor = np.sqrt(p_true * (1.0 - p_true) / n_total)
        mcse = np.where(np.isfinite(mcse) & (mcse > 0.0), mcse, floor)
        z = err / mcse
        tag = _prob_tag(float(p))
        out[f"tail_prob_abs_err_{tag}"] = float(np.mean(np.abs(err)))
        out[f"tail_prob_mcse_{tag}"] = float(np.mean(mcse))
        out[f"tail_prob_max_abs_z_{tag}"] = float(np.max(np.abs(z)))
        out[f"tail_prob_rms_z_{tag}"] = float(np.sqrt(np.mean(z**2)))
    return out


def _to_jsonable(value: Any) -> Any:
    """Converts numpy scalars and arrays to JSON types and non-finite floats to None."""
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    return value


def _score(
    samples: np.ndarray, telemetry: Mapping[str, Any], target: TargetDistribution, quantiles: Mapping[float, np.ndarray]
) -> dict[str, Any]:
    grads = int(telemetry["total_grad_evals"])
    wall = float(telemetry["wall_clock_time"])
    with warnings.catch_warnings():
        # ArviZ warns on near-constant coordinates; the resulting NaNs are reported as None.
        warnings.simplefilter("ignore", RuntimeWarning)
        m = evaluate_chains_with_arviz(
            samples, grads, target=target, divergences=int(telemetry["divergences"]), wall_clock_time=wall
        )
    bulk = _finite_or_none(m["bulk_ess"])
    tail = _finite_or_none(m["tail_ess"])
    min_tail = _finite_or_none(m["min_tail_ess"])
    split_rhat = _finite_or_none(m["split_rhat"])
    leapfrog = None
    if "leapfrog_steps_warmup" in telemetry and "leapfrog_steps_draws" in telemetry:
        leapfrog = int(telemetry["leapfrog_steps_warmup"]) + int(telemetry["leapfrog_steps_draws"])
    row: dict[str, Any] = {
        "total_grad_evals": grads,
        "warmup_grad_evals": int(telemetry["warmup_grad_evals"]),
        "draw_grad_evals": int(telemetry["draw_grad_evals"]),
        "divergences": int(telemetry["divergences"]),
        "warmup_divergences": telemetry["warmup_divergences"],
        "wall_clock_time": wall,
        "bulk_ess": bulk,
        "tail_ess": tail,
        "min_bulk_ess": _finite_or_none(m["min_bulk_ess"]),
        "min_tail_ess": min_tail,
        "split_rhat": split_rhat,
        "combined_rhat": _finite_or_none(m["combined_rhat"]),
        "converged": bool(split_rhat is not None and split_rhat <= RHAT_CONVERGED),
        "ess_per_grad": None if bulk is None else bulk / grads,
        "tail_ess_per_grad": None if tail is None else tail / grads,
        "min_tail_ess_per_grad": None if min_tail is None else min_tail / grads,
        "tail_ess_per_sec": None if tail is None or wall <= 0 else tail / wall,
        "q99_rel_error": _finite_or_none(m.get("q99_rel_error")),
        "leapfrog_steps": leapfrog,
        "flow_training_seconds": _finite_or_none(telemetry.get("flow_training_seconds")),
    }
    row.update(tail_probability_errors(samples, quantiles))
    return row


# =============================================================================
# 3. Aggregation
# =============================================================================


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"median": None, "q25": None, "q75": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]], reference_arm: str = REFERENCE_ARM) -> list[dict[str, Any]]:
    """Per-(target, arm) summaries over seeds, with per-seed paired ratios against ``reference_arm``.

    Only rows with ``status == "ok"`` enter a summary. A paired ratio needs both runs of the pair to
    be ok and the reference metric to be positive.
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["target"], row["arm"]), []).append(row)

    out: list[dict[str, Any]] = []
    for (target, arm), group in groups.items():
        ok = sorted((r for r in group if r["status"] == "ok"), key=lambda r: r["seed"])
        metrics = {m: _summary([r[m] for r in ok if r.get(m) is not None]) for m in AGGREGATE_METRICS}
        rhats = [r["split_rhat"] for r in ok if r.get("split_rhat") is not None]
        ratios: dict[str, Any] = {}
        if arm != reference_arm:
            ref_by_seed = {r["seed"]: r for r in groups.get((target, reference_arm), []) if r["status"] == "ok"}
            for metric in RATIO_METRICS:
                seeds: list[int] = []
                values: list[float] = []
                both_converged = 0
                for r in ok:
                    ref = ref_by_seed.get(r["seed"])
                    if ref is None or r.get(metric) is None or not ref.get(metric):
                        continue
                    seeds.append(int(r["seed"]))
                    values.append(float(r[metric]) / float(ref[metric]))
                    both_converged += int(bool(r["converged"]) and bool(ref["converged"]))
                ratios[metric] = {
                    **_summary(values),
                    "n_pairs": len(values),
                    "n_pairs_both_converged": both_converged,
                    "seeds": seeds,
                    "values": values,
                }
        out.append(
            {
                "target": target,
                "arm": arm,
                "n_seeds": len(group),
                "n_ok": len(ok),
                "n_converged": sum(1 for r in ok if r["converged"]),
                "max_split_rhat": max(rhats) if rhats else None,
                "total_divergences": int(sum(int(r["divergences"]) for r in ok)),
                "metrics": metrics,
                "paired_ratios_vs_reference": ratios,
            }
        )
    return out


# =============================================================================
# 4. Provenance, configuration and checkpoints
# =============================================================================

_PROVENANCE_PACKAGES: tuple[str, ...] = (
    "mcmc-robust",
    "numpy",
    "scipy",
    "jax",
    "jaxlib",
    "blackjax",
    "arviz",
    "pymc",
    "nutpie",
    "flowjax",
    "equinox",
    "optax",
)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_state() -> tuple[str, bool | None]:
    repo = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    return sha, bool(status.strip())


def collect_provenance() -> dict[str, Any]:
    """Git state, platform, package versions and command line for the current process."""
    sha, dirty = _git_state()
    try:
        loadavg: list[float] | None = [float(x) for x in os.getloadavg()]
    except OSError:
        loadavg = None
    return {
        "generated_by": "mcmc_robust.bench.competitor_runner",
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "loadavg_at_start": loadavg,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "env": {key: os.environ.get(key) for key in ("XLA_FLAGS", "OMP_NUM_THREADS")},
        "packages": {name: _package_version(name) for name in _PROVENANCE_PACKAGES},
        "command": list(sys.argv),
    }


@dataclass(frozen=True)
class CompetitorBenchmarkConfig:
    """Benchmark configuration. ``quick=True`` shrinks the run for smoke tests; never report it."""

    targets: Sequence[str] = TARGET_NAMES
    arms: Sequence[str] = ARM_NAMES
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    tune: int = 500
    draws: int = 1000
    chains: int = 4
    max_depth: int = DEFAULT_MAX_DEPTH
    target_accept: float = DEFAULT_TARGET_ACCEPT
    output_dir: str = "reports"
    quick: bool = False

    def resolved(self) -> CompetitorBenchmarkConfig:
        """Returns the effective configuration, applying smoke-test sizes when ``quick`` is set."""
        return replace(self, tune=60, draws=60, chains=2) if self.quick else self

    def arm_tune(self, arm: ArmSpec) -> int:
        """Warm-up length an arm runs with."""
        return int(arm.tune) if arm.tune is not None and not self.quick else int(self.tune)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["targets"] = list(self.targets)
        data["arms"] = list(self.arms)
        data["seeds"] = [int(s) for s in self.seeds]
        return data


def _check_config(config: CompetitorBenchmarkConfig) -> None:
    unknown_targets = [t for t in config.targets if t not in TARGET_DESCRIPTIONS]
    if unknown_targets:
        raise KeyError(f"unknown targets {unknown_targets}; expected a subset of {TARGET_NAMES}")
    unknown_arms = [a for a in config.arms if a not in ARMS]
    if unknown_arms:
        raise KeyError(f"unknown arms {unknown_arms}; expected a subset of {tuple(ARMS)}")
    for name in config.arms:
        requires = ARMS[name].requires
        if requires == "nutpie" and not nutpie_available():
            raise CompetitorUnavailableError(f"arm {name!r} needs nutpie; {COMPETITORS_EXTRA_HINT}")
        if requires == "nutpie_flow" and not nutpie_flow_available():
            raise CompetitorUnavailableError(
                f"arm {name!r} needs nutpie with flowjax and equinox; {COMPETITORS_EXTRA_HINT}"
            )


def _read_checkpoint(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not path.exists():
        return None, []
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["type"] == "header":
                header = record
            elif record["type"] == "row":
                rows.append(record["row"])
    return header, rows


def _check_compatible(reference: Mapping[str, Any], other: Mapping[str, Any], what: str) -> None:
    for key in _CHECKPOINT_KEYS:
        if reference.get(key) != other.get(key):
            raise ValueError(f"{what}: {key} differs ({reference.get(key)!r} vs {other.get(key)!r})")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, allow_nan=False) + "\n")


# =============================================================================
# 5. Running and reporting
# =============================================================================


def _error_row(base: Mapping[str, Any], message: str) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
    row.update(base)
    row.update({"status": "error", "error": message, "converged": False, "telemetry": None})
    return row


def run_cell(
    target_name: str,
    target: TargetDistribution,
    arm: ArmSpec,
    seed: int,
    init: np.ndarray,
    quantiles: Mapping[float, np.ndarray],
    config: CompetitorBenchmarkConfig,
) -> dict[str, Any]:
    """Runs one (target, arm, seed) cell and scores it; a failing sampler gives an error row."""
    tune = config.arm_tune(arm)
    base = {
        "target": target_name,
        "arm": arm.name,
        "seed": int(seed),
        "d": int(target.d),
        "tune": tune,
        "draws": int(config.draws),
        "chains": int(config.chains),
        "max_depth": int(config.max_depth),
        "target_accept": float(config.target_accept),
    }
    try:
        samples, telemetry = arm.adapter(
            target,
            d=target.d,
            seed=int(seed),
            tune=tune,
            draws=int(config.draws),
            chains=int(config.chains),
            init=init,
            max_depth=int(config.max_depth),
            target_accept=float(config.target_accept),
        )
        samples = np.asarray(samples, dtype=np.float64)
        expected = (config.chains, config.draws, target.d)
        if samples.shape != expected:
            raise ValueError(f"adapter returned samples of shape {samples.shape}, expected {expected}")
        scored = _score(samples, telemetry, target, quantiles)
    except CompetitorUnavailableError:
        raise
    except Exception as exc:
        return _to_jsonable(_error_row(base, f"{type(exc).__name__}: {exc}"))
    row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
    row.update(base)
    row.update(scored)
    row.update({"status": "ok", "error": None, "telemetry": telemetry})
    return _to_jsonable(row)


def _sorted_rows(
    rows: Iterable[Mapping[str, Any]], targets: Sequence[str], arms: Sequence[str]
) -> list[dict[str, Any]]:
    t_index = {t: i for i, t in enumerate(targets)}
    a_index = {a: i for i, a in enumerate(arms)}
    return sorted(
        (dict(r) for r in rows), key=lambda r: (t_index.get(r["target"], 99), a_index.get(r["arm"], 99), r["seed"])
    )


def build_result(
    config: CompetitorBenchmarkConfig, provenance: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Assembles the JSON report from configuration, provenance and per-seed rows."""
    ordered = _sorted_rows(rows, config.targets, config.arms)
    result = {
        "schema_version": 1,
        "provenance": dict(provenance),
        "config": config.to_dict(),
        "reference_arm": REFERENCE_ARM,
        "targets": {name: TARGET_DESCRIPTIONS[name] for name in config.targets},
        "arms": {
            name: {"description": ARMS[name].description, "tune": config.arm_tune(ARMS[name])} for name in config.arms
        },
        "excluded_arms": EXCLUDED_ARMS,
        "metric_definitions": METRIC_DEFINITIONS,
        "notes": list(NOTES),
        "rows": ordered,
        "aggregates": aggregate_rows(ordered, reference_arm=REFERENCE_ARM),
    }
    return _to_jsonable(result)


def write_results(result: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Writes ``competitor_results.json`` (everything) and ``competitor_results.csv`` (per-seed rows)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "competitor_results.json"
    csv_path = out / "competitor_results.csv"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
        fh.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
    return json_path, csv_path


def run_competitor_benchmark(
    config: CompetitorBenchmarkConfig,
    checkpoint: str | Path | None = None,
    write_report: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Runs every (target, seed, arm) cell of ``config`` and returns (and by default writes) the report.

    With ``checkpoint``, each finished row is appended to that JSON-lines file and cells already in
    it are skipped, so an interrupted run resumes where it stopped.
    """
    cfg = config.resolved()
    _check_config(cfg)
    provenance = collect_provenance()
    rows: list[dict[str, Any]] = []
    ckpt = Path(checkpoint) if checkpoint is not None else None
    if ckpt is not None:
        header, rows = _read_checkpoint(ckpt)
        if header is None:
            _append_jsonl(ckpt, {"type": "header", "provenance": provenance, "config": cfg.to_dict()})
        else:
            _check_compatible(header["config"], cfg.to_dict(), f"checkpoint {ckpt}")
            if header["provenance"]["git_sha"] != provenance["git_sha"]:
                raise ValueError(f"checkpoint {ckpt} was written at git {header['provenance']['git_sha']}")
            provenance = header["provenance"]
    done = {(r["target"], r["arm"], r["seed"]) for r in rows}

    for target_name in cfg.targets:
        target = make_target(target_name)
        quantiles = exact_marginal_quantiles(target, TAIL_PROBS)
        for seed in cfg.seeds:
            init = initial_positions(int(seed), cfg.chains, target.d)
            for arm_name in cfg.arms:
                if (target_name, arm_name, int(seed)) in done:
                    continue
                t0 = time.perf_counter()
                row = run_cell(target_name, target, ARMS[arm_name], int(seed), init, quantiles, cfg)
                rows.append(row)
                if ckpt is not None:
                    _append_jsonl(ckpt, {"type": "row", "row": row})
                if log is not None:
                    log(
                        f"{target_name:>10} {arm_name:>20} seed={seed} {row['status']:>5} "
                        f"grads={row['total_grad_evals']} tail_ess/grad={row['tail_ess_per_grad']} "
                        f"rhat={row['split_rhat']} div={row['divergences']} ({time.perf_counter() - t0:.1f}s)"
                    )

    keep = {(t, a, int(s)) for t in cfg.targets for a in cfg.arms for s in cfg.seeds}
    rows = [r for r in rows if (r["target"], r["arm"], r["seed"]) in keep]
    result = build_result(cfg, provenance, rows)
    if write_report:
        write_results(result, cfg.output_dir)
    return result


def merge_checkpoints(paths: Sequence[str | Path], output_dir: str | Path = "reports") -> dict[str, Any]:
    """Combines shard checkpoints (same code, packages and run sizes) into one report."""
    if not paths:
        raise ValueError("no checkpoints to merge")
    headers: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        header, rows = _read_checkpoint(Path(path))
        if header is None:
            raise ValueError(f"checkpoint {path} has no header")
        if headers:
            first = headers[0]
            _check_compatible(first["config"], header["config"], f"checkpoint {path}")
            for prov_key in ("git_sha", "packages", "platform"):
                if first["provenance"][prov_key] != header["provenance"][prov_key]:
                    raise ValueError(f"checkpoint {path}: provenance {prov_key} differs from {paths[0]}")
        headers.append(header)
        for row in rows:
            key = (row["target"], row["arm"], row["seed"])
            if key in seen:
                raise ValueError(f"cell {key} appears in more than one checkpoint")
            seen.add(key)
            all_rows.append(row)

    def ordered_union(values: Iterable[Any]) -> list[Any]:
        out: list[Any] = []
        for v in values:
            if v not in out:
                out.append(v)
        return out

    base_cfg = headers[0]["config"]
    config = CompetitorBenchmarkConfig(
        targets=tuple(ordered_union(r["target"] for r in all_rows)),
        arms=tuple(ordered_union(a for a in ARM_NAMES if any(r["arm"] == a for r in all_rows))),
        seeds=tuple(sorted({int(r["seed"]) for r in all_rows})),
        tune=base_cfg["tune"],
        draws=base_cfg["draws"],
        chains=base_cfg["chains"],
        max_depth=base_cfg["max_depth"],
        target_accept=base_cfg["target_accept"],
        output_dir=str(output_dir),
        quick=base_cfg["quick"],
    )
    provenance = dict(headers[0]["provenance"])
    provenance["shards"] = [
        {
            "checkpoint": Path(path).name,
            "timestamp_utc": h["provenance"]["timestamp_utc"],
            "loadavg_at_start": h["provenance"].get("loadavg_at_start"),
            "command": h["provenance"]["command"],
            "targets": h["config"]["targets"],
            "arms": h["config"]["arms"],
        }
        for path, h in zip(paths, headers)
    ]
    result = build_result(config, provenance, all_rows)
    write_results(result, output_dir)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    defaults = CompetitorBenchmarkConfig()
    parser = argparse.ArgumentParser(description="Direct-competitor benchmark (stock NUTS, arcsinh NUTS, nutpie)")
    parser.add_argument("--targets", nargs="+", default=list(defaults.targets), choices=TARGET_NAMES)
    parser.add_argument("--arms", nargs="+", default=list(defaults.arms), choices=ARM_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(defaults.seeds))
    parser.add_argument("--tune", type=int, default=defaults.tune)
    parser.add_argument("--draws", type=int, default=defaults.draws)
    parser.add_argument("--chains", type=int, default=defaults.chains)
    parser.add_argument("--max-depth", type=int, default=defaults.max_depth)
    parser.add_argument("--target-accept", type=float, default=defaults.target_accept)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--quick", action="store_true", help="smoke-test sizes; never report these numbers")
    parser.add_argument("--checkpoint", help="JSON-lines file to append rows to and resume from")
    parser.add_argument("--no-report", action="store_true", help="only fill the checkpoint (for shards)")
    parser.add_argument("--merge", nargs="+", metavar="CHECKPOINT", help="merge shard checkpoints into the report")
    args = parser.parse_args(argv)

    if args.merge:
        result = merge_checkpoints(args.merge, output_dir=args.output_dir)
        print(f"Merged {len(result['rows'])} rows into {args.output_dir}/competitor_results.{{json,csv}}")
        return

    config = CompetitorBenchmarkConfig(
        targets=tuple(args.targets),
        arms=tuple(args.arms),
        seeds=tuple(args.seeds),
        tune=args.tune,
        draws=args.draws,
        chains=args.chains,
        max_depth=args.max_depth,
        target_accept=args.target_accept,
        output_dir=args.output_dir,
        quick=args.quick,
    )
    result = run_competitor_benchmark(
        config, checkpoint=args.checkpoint, write_report=not args.no_report, log=lambda s: print(s, flush=True)
    )
    n_ok = sum(1 for r in result["rows"] if r["status"] == "ok")
    print(f"{n_ok}/{len(result['rows'])} runs succeeded.")
    if not args.no_report:
        print(f"Wrote {args.output_dir}/competitor_results.{{json,csv}}")


if __name__ == "__main__":
    main()
