r"""Noise-aware tail accuracy: tail-probability errors and their Monte Carlo z-scores.

For an exact marginal quantile $q_p$ of coordinate $i$, the estimate
$\hat P_i = \frac{1}{n} \sum_t \mathbf 1\{x_{t,i} < q_{p,i}\}$ should equal $p$. The metric reports

$$\mathrm{err}_{p,i} = \hat P_i - p, \qquad z_{p,i} = \mathrm{err}_{p,i} / \mathrm{MCSE}(\hat P_i),$$

with the MCSE of the indicator mean from ArviZ (`az.mcse(..., method="mean")`, which uses the
multi-chain ESS of the indicator). A correct, well-mixing sampler gives $|z| = O(1)$
($E|z| \approx 0.80$); a sampler that misses or over-weights a tail gives $|z| \gg 1$.

Why not the relative error of $\hat q_{0.99}$: for a Cauchy marginal at ESS $\approx 1000$ its
standard deviation is about 30% even under perfect sampling, so single-run values cannot separate
samplers. $|z|$ is normalised by the run's own Monte Carlo error.

Degenerate case: if no draw (or every draw) lies below $q_p$, the indicator is constant and
ArviZ's MCSE is 0. The z-score then uses $\sqrt{p(1-p)/n}$, the standard error of $n$ independent
draws at the true $p$. That is no larger than the MCSE of a positively autocorrelated chain, so the
reported $|z|$ is finite and conservative. Such cells are counted in `degenerate`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import arviz as az
import numpy as np


@dataclass(frozen=True)
class TailProbabilityAccuracy:
    """Per-(probability, coordinate) tail-probability estimates, errors, MCSEs and z-scores.

    Arrays have shape `(len(probs), d)`; `error` is signed ($\\hat P - p$).
    """

    probs: tuple[float, ...]
    p_hat: np.ndarray
    error: np.ndarray
    mcse: np.ndarray
    z: np.ndarray
    degenerate: np.ndarray

    def summary(self) -> dict[str, float]:
        """Coordinate- and probability-averaged summary used in benchmark rows."""
        abs_z = np.abs(self.z)
        return {
            "tail_prob_abs_err": float(np.mean(np.abs(self.error))),
            "tail_prob_abs_z": float(np.mean(abs_z)),
            "tail_prob_max_abs_z": float(np.max(abs_z)),
            "tail_prob_n_degenerate": int(np.sum(self.degenerate)),
        }


def tail_probability_accuracy(
    samples: np.ndarray,
    quantiles: Mapping[float, np.ndarray],
) -> TailProbabilityAccuracy:
    r"""Computes $\hat P(x_i < q_{p,i})$, its error, ArviZ MCSE and z-score for each $p$ in `quantiles`.

    Parameters:
        samples: Draws of shape `(chains, draws, d)`, or `(draws, d)` for a single chain.
        quantiles: Map from probability $p \in (0, 1)$ to the exact marginal quantiles $q_p$, shape `(d,)`.

    Returns:
        `TailProbabilityAccuracy` with arrays of shape `(len(quantiles), d)`.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Expected samples of shape (chains, draws, d) or (draws, d), got {arr.shape}")
    n_total = arr.shape[0] * arr.shape[1]
    d = arr.shape[2]

    probs = tuple(float(p) for p in quantiles)
    p_hat = np.empty((len(probs), d))
    mcse = np.empty((len(probs), d))
    for k, p in enumerate(probs):
        if not 0.0 < p < 1.0:
            raise ValueError(f"Tail probabilities must lie in (0, 1), got {p}")
        q = np.asarray(quantiles[p], dtype=np.float64)
        if q.shape != (d,):
            raise ValueError(f"quantile array for p={p} has shape {q.shape}, expected ({d},)")
        indicator = (arr < q).astype(np.float64)
        p_hat[k] = indicator.mean(axis=(0, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            mcse[k] = np.asarray(az.mcse(indicator, method="mean"), dtype=np.float64)

    p_col = np.asarray(probs)[:, np.newaxis]
    error = p_hat - p_col
    degenerate = ~(np.isfinite(mcse) & (mcse > 0.0))
    binomial_floor = np.sqrt(p_col * (1.0 - p_col) / n_total)
    z = error / np.where(degenerate, binomial_floor, mcse)
    return TailProbabilityAccuracy(probs=probs, p_hat=p_hat, error=error, mcse=mcse, z=z, degenerate=degenerate)


#: Reference tail probabilities of `reference_agreement`.
REFERENCE_PROBS: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99)


@dataclass(frozen=True)
class ReferenceAgreement:
    """Agreement of a run with reference draws: tail probabilities at reference quantiles, and means.

    Tail arrays have shape `(len(probs), k)` and mean arrays `(k,)`; errors are run minus reference.
    """

    probs: tuple[float, ...]
    thresholds: np.ndarray
    p_hat: np.ndarray
    p_ref: np.ndarray
    error: np.ndarray
    z: np.ndarray
    degenerate: np.ndarray
    mean_error: np.ndarray
    mean_z: np.ndarray

    def summary(self) -> dict[str, float]:
        """Parameter- and probability-averaged summary used in benchmark rows."""
        return {
            "ref_tail_prob_abs_err": float(np.mean(np.abs(self.error))),
            "ref_tail_prob_abs_z": float(np.mean(np.abs(self.z))),
            "ref_tail_prob_max_abs_z": float(np.max(np.abs(self.z))),
            "ref_tail_prob_n_degenerate": int(np.sum(self.degenerate)),
            "ref_mean_abs_z": float(np.mean(np.abs(self.mean_z))),
            "ref_mean_max_abs_z": float(np.max(np.abs(self.mean_z))),
        }


def _as_chains(samples: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Expected {name} of shape (chains, draws, k) or (draws, k), got {arr.shape}")
    return arr


def _mcse_mean(arr: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(az.mcse(arr, method="mean"), dtype=np.float64)


def reference_agreement(
    samples: np.ndarray,
    reference: np.ndarray,
    probs: tuple[float, ...] = REFERENCE_PROBS,
) -> ReferenceAgreement:
    r"""Compares a run with reference draws of the same $k$ parameters, counting the reference's own error.

    For each $p$, the threshold is the reference quantile $\hat q_p$ (pooled over its chains). With
    $\hat P$ and $\hat P_{\mathrm{ref}}$ the fractions of run and reference draws below it,
    $$\mathrm{err} = \hat P - \hat P_{\mathrm{ref}}, \qquad
    z = \mathrm{err} / \sqrt{\mathrm{MCSE}(\hat P)^2 + \mathrm{MCSE}(\hat P_{\mathrm{ref}})^2},$$
    with ArviZ's mean-MCSE of each indicator. Means are compared the same way. When an indicator
    MCSE is 0 (no draw, or every draw, below the threshold) the denominator is the binomial
    $\sqrt{p(1-p)(1/n + 1/n_{\mathrm{ref}})}$ and the cell counts as degenerate.

    Parameters:
        samples: Run draws of shape `(chains, draws, k)` or `(draws, k)`.
        reference: Reference draws of shape `(chains, draws, k)` or `(draws, k)`.
        probs: Tail probabilities in $(0, 1)$ (default `REFERENCE_PROBS`).
    """
    run = _as_chains(samples, "samples")
    ref = _as_chains(reference, "reference")
    if run.shape[2] != ref.shape[2]:
        raise ValueError(f"samples have {run.shape[2]} parameters and reference {ref.shape[2]}")
    probs = tuple(float(p) for p in probs)
    if not all(0.0 < p < 1.0 for p in probs):
        raise ValueError(f"Tail probabilities must lie in (0, 1), got {probs}")
    k = run.shape[2]
    n_run = run.shape[0] * run.shape[1]
    n_ref = ref.shape[0] * ref.shape[1]

    thresholds = np.quantile(ref.reshape(-1, k), probs, axis=0)
    p_hat = np.empty((len(probs), k))
    p_ref = np.empty((len(probs), k))
    se = np.empty((len(probs), k))
    degenerate = np.zeros((len(probs), k), dtype=bool)
    for i, (p, q) in enumerate(zip(probs, thresholds)):
        ind_run = (run < q).astype(np.float64)
        ind_ref = (ref < q).astype(np.float64)
        p_hat[i] = ind_run.mean(axis=(0, 1))
        p_ref[i] = ind_ref.mean(axis=(0, 1))
        mcse_run, mcse_ref = _mcse_mean(ind_run), _mcse_mean(ind_ref)
        degenerate[i] = ~(np.isfinite(mcse_run) & (mcse_run > 0.0) & np.isfinite(mcse_ref) & (mcse_ref > 0.0))
        floor = np.sqrt(p * (1.0 - p) * (1.0 / n_run + 1.0 / n_ref))
        se[i] = np.where(degenerate[i], floor, np.sqrt(mcse_run**2 + mcse_ref**2))
    error = p_hat - p_ref

    mean_error = run.mean(axis=(0, 1)) - ref.mean(axis=(0, 1))
    mean_z = mean_error / np.sqrt(_mcse_mean(run) ** 2 + _mcse_mean(ref) ** 2)
    return ReferenceAgreement(
        probs=probs,
        thresholds=thresholds,
        p_hat=p_hat,
        p_ref=p_ref,
        error=error,
        z=error / se,
        degenerate=degenerate,
        mean_error=mean_error,
        mean_z=mean_z,
    )


__all__ = [
    "REFERENCE_PROBS",
    "ReferenceAgreement",
    "TailProbabilityAccuracy",
    "reference_agreement",
    "tail_probability_accuracy",
]
