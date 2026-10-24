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


__all__ = ["TailProbabilityAccuracy", "tail_probability_accuracy"]
