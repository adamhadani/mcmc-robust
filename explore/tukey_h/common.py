"""Targets, exact marginal quantiles and metrics for the Tukey-h exploration (explore/tukey-h-transform)."""

from __future__ import annotations

from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import numpy as np
from scipy import stats

from mcmc_robust.bench.sota_adapters import SparseSignalRegressionTarget, evaluate_chains_with_arviz
from mcmc_robust.targets.benign import FullCovarianceGaussian, IsotropicGaussian
from mcmc_robust.targets.heavy_tailed import HeterogeneousHeavyTailedTarget, MultivariateStudentTTarget

TARGETS = [
    "prod_t0.5",
    "prod_t1",
    "prod_t2",
    "prod_t5",
    "task1",
    "task3",
    "corr_t1",
    "corr_t2",
    "iso_gauss",
    "corr_gauss",
]


def rotated_cov(d: int, cond: float, seed: int = 0) -> np.ndarray:
    """Same construction as the session's ablation spike (§3.1 of the state assessment)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = np.geomspace(1.0, cond, d)
    return (q * eig) @ q.T


def make_target(key: str) -> Any:
    if key.startswith("prod_t"):
        nu = float(key.removeprefix("prod_t"))
        return HeterogeneousHeavyTailedTarget(d=10, heavy_coords=list(range(10)), nu_list=[nu] * 10)
    if key == "task1":
        return SparseSignalRegressionTarget(d=10, n_obs=40, n_active=3, seed=42)
    if key == "task3":
        return HeterogeneousHeavyTailedTarget(
            d=20, heavy_coords=list(range(10)), nu_list=list(np.linspace(1.2, 2.5, 10))
        )
    if key == "corr_t1":
        return MultivariateStudentTTarget(d=10, nu=1.0, scale=rotated_cov(10, 25.0))
    if key == "corr_t2":
        return MultivariateStudentTTarget(d=10, nu=2.0, scale=rotated_cov(10, 25.0))
    if key == "iso_gauss":
        return IsotropicGaussian(d=20)
    if key == "corr_gauss":
        return FullCovarianceGaussian(d=20, cov=rotated_cov(20, 100.0))
    if key == "cauchy1d":
        return HeterogeneousHeavyTailedTarget(d=1, heavy_coords=[0], nu_list=[1.0])
    raise KeyError(key)


def true_heavy_mask(key: str, target: Any) -> np.ndarray:
    """Which coordinates have polynomial (regularly varying) marginal tails."""
    d = target.d
    if key.startswith(("prod_t", "corr_t")) or key == "cauchy1d":
        return np.ones(d, dtype=bool)
    if key == "task1":
        return np.arange(d) >= target.n_active
    if key == "task3":
        return np.arange(d) < 10
    return np.zeros(d, dtype=bool)


def true_alpha(key: str, target: Any) -> np.ndarray:
    """Marginal tail index alpha per coordinate (inf for light tails)."""
    d = target.d
    out = np.full(d, np.inf)
    if key.startswith("prod_t") or key == "cauchy1d":
        out[:] = target.nu_arr
    elif key.startswith("corr_t"):
        out[:] = target.nu
    elif key == "task1":
        out[target.n_active :] = 1.0
    elif key == "task3":
        out[:10] = target.nu_arr
    return out


def exact_quantile(key: str, target: Any, p: float) -> np.ndarray:
    """Exact marginal p-quantile per coordinate."""
    d = target.d
    if key.startswith("prod_t") or key in ("task3", "cauchy1d"):
        q = np.empty(d)
        for idx, c in enumerate(target.heavy_coords):
            q[c] = target.mu[c] + target.scale_arr[c] * stats.t.ppf(p, df=target.nu_arr[idx])
        for c in target.light_coords:
            q[c] = target.mu[c] + target.scale_arr[c] * stats.norm.ppf(p)
        return q
    if key == "task1":
        q = np.empty(d)
        for j in range(d):
            if j < len(target._active_grids):
                q[j] = float(np.interp(p, target._active_cdfs[j], target._active_grids[j]))
            else:
                q[j] = float(stats.cauchy.ppf(p, scale=target.prior_scale))
        return q
    if key.startswith("corr_t"):
        return target.mu + np.sqrt(np.diag(target.cov)) * stats.t.ppf(p, df=target.nu)
    if key in ("iso_gauss", "corr_gauss"):
        return target.mean + np.sqrt(np.diag(target.cov)) * stats.norm.ppf(p)
    raise KeyError(key)


def tail_prob_metrics(samples: np.ndarray, key: str, target: Any) -> dict[str, float]:
    """|P_hat(x_i < q_p) - p| at exact quantiles, p in {0.01, 0.99}, with MCSE from the indicator's ESS.

    Errors and MCSEs are reported relative to the tail mass min(p, 1 - p) = 0.01, averaged over coordinates
    and both tails; `tp_z_max` is the largest |error| / MCSE.
    """
    arr = np.asarray(samples, dtype=np.float64)
    errs, mcses, zs = [], [], []
    for p in (0.01, 0.99):
        q = exact_quantile(key, target, p)
        ind = (arr < q).astype(np.float64)
        phat = ind.mean(axis=(0, 1))
        dt = az.from_dict({"posterior": {"ind": ind}})
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = np.asarray(az.ess(dt, var_names=["ind"], method="mean")["ind"].values, dtype=np.float64)
        ess = np.where(np.isfinite(ess) & (ess > 0), ess, np.nan)
        mcse = np.sqrt(p * (1 - p) / ess)  # MCSE of P_hat under the true p
        tail_mass = min(p, 1 - p)
        err = np.abs(phat - p)
        errs.append(err / tail_mass)
        mcses.append(mcse / tail_mass)
        zs.append(err / mcse)
    e, m, z = np.concatenate(errs), np.concatenate(mcses), np.concatenate(zs)
    return {
        "tp_err_rel": float(np.mean(e)),
        "tp_err_rel_max": float(np.max(e)),
        "tp_mcse_rel": float(np.nanmean(m)) if np.any(np.isfinite(m)) else float("nan"),
        "tp_z_max": float(np.nanmax(z)) if np.any(np.isfinite(z)) else float("nan"),
        "tp_n_undefined_ess": int(np.sum(~np.isfinite(m))),
    }


KEEP = [
    "total_grad_evals",
    "bulk_ess",
    "tail_ess",
    "min_bulk_ess",
    "min_tail_ess",
    "split_rhat",
    "combined_rhat",
    "ess_per_grad",
    "tail_ess_per_grad",
    "divergences",
    "q99_rel_error",
]


def summarise(samples: np.ndarray, grads: int, divs: int, key: str, target: Any, wall: float) -> dict[str, float]:
    m = evaluate_chains_with_arviz(samples, grads, target=target, divergences=divs, wall_clock_time=wall)
    out = {k: float(m[k]) for k in KEEP if k in m}
    out["min_tail_ess_per_grad"] = out["min_tail_ess"] / max(1.0, out["total_grad_evals"])
    out["wall_clock_time"] = float(wall)
    out.update(tail_prob_metrics(samples, key, target))
    return out
