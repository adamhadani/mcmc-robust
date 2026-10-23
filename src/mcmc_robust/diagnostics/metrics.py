"""Statistical distance and distribution metrics: KS test, Wasserstein-1, and quantile errors."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats

from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.targets.benign import (
    DiagonalGaussian,
    FullCovarianceGaussian,
    IsotropicGaussian,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)


def _get_analytical_marginal_cdf(
    target: TargetDistribution,
    coord_idx: int,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Attempts to construct exact 1D analytical marginal CDF for coordinate coord_idx.

    Args:
        target: TargetDistribution instance.
        coord_idx: Coordinate index (0 <= coord_idx < target.d).

    Returns:
        Callable CDF function or None if no closed-form marginal CDF is available.
    """
    if isinstance(target, IsotropicGaussian):
        mu = float(target.mean[coord_idx])
        sigma = float(target.std)

        def cdf_iso(x: Any) -> Any:
            return stats.norm.cdf(x, loc=mu, scale=sigma)

        return cdf_iso

    if isinstance(target, DiagonalGaussian):
        mu = float(target.mean[coord_idx])
        sigma = float(target.stds[coord_idx])

        def cdf_diag(x: Any) -> Any:
            return stats.norm.cdf(x, loc=mu, scale=sigma)

        return cdf_diag

    if isinstance(target, FullCovarianceGaussian):
        mu = float(target.mean[coord_idx])
        sigma = float(np.sqrt(target.cov[coord_idx, coord_idx]))

        def cdf_full(x: Any) -> Any:
            return stats.norm.cdf(x, loc=mu, scale=sigma)

        return cdf_full

    if isinstance(target, MultivariateStudentTTarget):
        mu = float(target.mu[coord_idx])
        scale = float(np.sqrt(target._cov[coord_idx, coord_idx]))
        nu = float(target.nu)

        def cdf_t(x: Any) -> Any:
            return stats.t.cdf((x - mu) / scale, df=nu)

        return cdf_t

    if isinstance(target, HeterogeneousHeavyTailedTarget):
        mu = float(target.mu[coord_idx])
        s = float(target.scale_arr[coord_idx])
        if coord_idx in target.heavy_coords:
            heavy_idx = list(target.heavy_coords).index(coord_idx)
            nu = float(target.nu_arr[heavy_idx])

            def cdf_het_t(x: Any) -> Any:
                return stats.t.cdf((x - mu) / s, df=nu)

            return cdf_het_t
        else:

            def cdf_het_norm(x: Any) -> Any:
                return stats.norm.cdf((x - mu) / s)

            return cdf_het_norm

    return None


def compute_ks_test(
    samples: np.ndarray,
    target: TargetDistribution,
    n_ref: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    r"""Computes 1D marginal Kolmogorov-Smirnov test per coordinate against target truth.

    If the target distribution provides an analytical 1D continuous CDF, an exact
    one-sample KS test (scipy.stats.kstest) is performed.
    Otherwise, an empirical two-sample KS test (scipy.stats.ks_2samp) is performed
    against n_ref exact independent samples drawn from target.sample_ground_truth.

    Args:
        samples: MCMC chain samples of shape (N, d) or (N,).
        target: TargetDistribution instance to validate against.
        n_ref: Number of reference samples if two-sample test is needed (default 10000).
        rng: Optional NumPy random generator for ground truth sampling.

    Returns:
        Dict with keys:
            - 'statistic': np.ndarray of shape (d,) with KS test statistics
            - 'pvalue': np.ndarray of shape (d,) with KS p-values
            - 'passed': bool, True if all coordinate p-values > 0.01
            - 'min_pvalue': float, minimum p-value across all dimensions
            - 'max_statistic': float, maximum KS statistic across all dimensions
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    N, d = s.shape

    if d != target.d:
        raise ValueError(f"Samples dimension {d} does not match target dimension {target.d}")

    statistic = np.empty(d, dtype=np.float64)
    pvalue = np.empty(d, dtype=np.float64)

    ref_samples: np.ndarray | None = None

    for j in range(d):
        cdf_fn = _get_analytical_marginal_cdf(target, j)
        if cdf_fn is not None:
            # 1-sample KS test against exact analytical CDF
            res = stats.kstest(s[:, j], cdf_fn)
            statistic[j] = res.statistic
            pvalue[j] = res.pvalue
        else:
            # 2-sample KS test against exact ground truth samples
            if ref_samples is None:
                if rng is None:
                    rng = np.random.default_rng(42)
                n_draw = max(n_ref, N)
                ref_samples = target.sample_ground_truth(n=n_draw, rng=rng)
            res = stats.ks_2samp(s[:, j], ref_samples[:, j])
            statistic[j] = res.statistic
            pvalue[j] = res.pvalue

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "passed": bool(np.all(pvalue > 0.01)),
        "min_pvalue": float(np.min(pvalue)),
        "max_statistic": float(np.max(statistic)),
    }


def compute_wasserstein1(
    samples: np.ndarray,
    target: TargetDistribution,
    n_ref: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Computes 1D marginal Wasserstein-1 distance per coordinate against target truth.

    Evaluates:
        W_1(P, Q) = \int_{-\infty}^\infty |F_P(x) - F_Q(x)| dx
    using scipy.stats.wasserstein_distance against reference samples drawn from
    target.sample_ground_truth.

    Args:
        samples: MCMC chain samples of shape (N, d) or (N,).
        target: TargetDistribution instance.
        n_ref: Number of reference samples (default 10000).
        rng: Optional NumPy random generator.

    Returns:
        np.ndarray of shape (d,) containing Wasserstein-1 distance per coordinate.
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    N, d = s.shape

    if d != target.d:
        raise ValueError(f"Samples dimension {d} does not match target dimension {target.d}")

    if rng is None:
        rng = np.random.default_rng(42)
    ref_samples = target.sample_ground_truth(n=n_ref, rng=rng)

    w1 = np.empty(d, dtype=np.float64)
    for j in range(d):
        w1[j] = stats.wasserstein_distance(s[:, j], ref_samples[:, j])

    return w1


def compute_quantile_relative_error(
    samples: np.ndarray,
    target: TargetDistribution,
    quantiles: Sequence[float] = (0.05, 0.95, 0.99),
    n_ref: int = 100000,
    rng: np.random.Generator | None = None,
) -> dict[float, np.ndarray]:
    r"""Computes relative quantile estimation error |\hat{q}_p - q_p^\star| / |q_p^\star|.

    Checks target.exact_moments_or_quantiles() for analytical quantiles.
    If unavailable, estimates high-precision reference quantiles q_p^\star from
    n_ref independent ground-truth draws.

    Guards against division-by-zero when q_p^\star \approx 0:
    When |q_p^\star| < 10^{-5} (e.g. median of symmetric zero-mean targets),
    returns absolute quantile error |\hat{q}_p - q_p^\star| to prevent
    misleading relative error explosion.

    Args:
        samples: MCMC chain samples of shape (N, d) or (N,).
        target: TargetDistribution instance.
        quantiles: Sequence of percentile probabilities in (0, 1) (default (0.05, 0.95, 0.99)).
        n_ref: Reference sample size if analytical quantiles are not cached (default 100000).
        rng: Optional NumPy random generator.

    Returns:
        Dict mapping each quantile float p -> np.ndarray of shape (d,) with relative errors.
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    N, d = s.shape

    if d != target.d:
        raise ValueError(f"Samples dimension {d} does not match target dimension {target.d}")

    # Standard key naming map in target.exact_moments_or_quantiles()
    key_map = {
        0.01: "q01",
        0.05: "q05",
        0.10: "q10",
        0.50: "q50",
        0.90: "q90",
        0.95: "q95",
        0.99: "q99",
    }

    target_moments = target.exact_moments_or_quantiles()
    ref_samples: np.ndarray | None = None

    rel_errors: dict[float, np.ndarray] = {}

    for p in quantiles:
        emp_q = np.quantile(s, p, axis=0)

        # Check if analytical quantile is directly in moments dict
        key = key_map.get(round(p, 4))
        if key is not None and key in target_moments:
            true_q = np.asarray(target_moments[key], dtype=np.float64)
            if true_q.ndim == 0:
                true_q = np.full(d, true_q, dtype=np.float64)
            elif true_q.shape[0] != d:
                true_q = np.full(d, true_q[0], dtype=np.float64)
        else:
            # Fall back to high-precision empirical ground truth
            if ref_samples is None:
                if rng is None:
                    rng = np.random.default_rng(42)
                ref_samples = target.sample_ground_truth(n=n_ref, rng=rng)
            true_q = np.quantile(ref_samples, p, axis=0)

        # Relative error with zero/near-zero guard (returns absolute error if |true_q| < 1e-5)
        abs_err = np.abs(emp_q - true_q)
        is_near_zero = np.abs(true_q) < 1e-5
        denom = np.where(is_near_zero, 1.0, np.abs(true_q))
        rel_errors[p] = abs_err / denom

    return rel_errors
