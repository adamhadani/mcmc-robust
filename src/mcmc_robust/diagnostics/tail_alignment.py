r"""Tail-alignment diagnostic: are heavy tails shared across coordinates?

Why it exists. A coordinate-wise transform such as the calibrated arcsinh map handles heavy tails
that run along the coordinate axes. When a heavy tail runs along a direction that is not an axis
(a rotated product of Cauchys, an elliptical multivariate *t*, a heavy common factor), several
coordinates become extreme together. Under the coordinate-wise map that direction becomes a ridge
in $z$ that narrows exponentially, NUTS cannot follow it, and every chain misses the same tail
region, so R-hat stays low (REPORT.md §5.1). A never-visited region cannot be detected from the
draws alone. This module checks the *risk condition* instead: heavy tails that are shared across
coordinates. A warning flags that risk; it does not prove that a tail was missed.

Method (sampler-agnostic; post-warm-up draws pooled over chains; $u$ = `level`):

1. **Heaviness.** For each coordinate, $a_i = \vert x_i - \mathop{\mathrm{median}}(x_i) \vert$ and
   $k = \lfloor (1 - u) n \rfloor$. The Hill estimate of the tail index from the $k$ largest values,
   $\hat\alpha_i = 1 / \big(\frac{1}{k} \sum_{m \le k} \log a_{i,(m)} - \log a_{i,(k+1)}\big)$, classes
   coordinate $i$ as heavy when $\hat\alpha_i <$ `max_tail_index` (default 4, the tail index below
   which the fourth moment is infinite). At this level a Gaussian coordinate's estimate lies well
   above 4 and a Student-*t* with $\nu \le 2.5$ well below; exponential tails can fall below 4,
   so "heavy" here means heavier than Gaussian, and step 2 decides.
2. **Pairwise tail co-exceedance** among heavy coordinates, with $E_i$ the event
   $a_i > a_{i,(k+1)}$ (coordinate $i$ in its top $1 - u$ fraction):
   $\chi_{ij}(u) = \hat P(E_i \cap E_j) / (1 - u) = \#(E_i \cap E_j) / k$. Under independence
   $\chi_{ij} \approx 1 - u$; when one heavy direction drives both coordinates, $\chi_{ij}$ stays
   large as $u \to 1$ (two identical coordinates give exactly 1).
3. **Warn** when $\max_{ij} \chi_{ij}$ exceeds `chi_threshold`.

No whitening. Removing linear correlation first (for example with a rank-based correlation
matrix) was tried and rejected: on MCMC draws of independent heavy coordinates, the estimated
correlation is noisy, the linear whitening map mixes each heavy tail into other coordinates, and
co-exceedance then appears where there was none. Linear correlation between light-tailed
coordinates needs no correction, because the heaviness gate of step 1 already excludes them.

Defaults. `level = 0.95` keeps $k = 200$ exceedances per coordinate at 4000 draws; higher levels
lose the signal exactly when a sampler never reaches the far tails. `chi_threshold = 0.3` was
chosen on pilot runs of the ablation (`bench/ablation_runner.py`, seeds 100-104, which the reported
seeds 0-9 do not include): it lies between the largest value seen on product-form and Gaussian
targets and the smallest value seen on targets with shared heavy tails. `min_exceedances = 50`:
with fewer exceedances per coordinate the check returns no verdict.

Known limits.
- It flags elliptical targets (multivariate *t*): their heavy tails are not axis-aligned either,
  although the arcsinh pipeline sometimes samples them well.
- Heavy marginals whose dependence is strong in the bulk but vanishes in the far tails (for
  example a Gaussian copula with Student-*t* marginals) can also fire at a finite level.
- A sampler that never leaves the bulk produces draws without heavy tails, and the check then
  stays silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Default level $u$: a coordinate exceeds when it is in its top $1 - u$ fraction of $\vert x_i - \mathrm{median}\vert$.
DEFAULT_LEVEL: float = 0.95

#: Default warning threshold on the maximum pairwise co-exceedance $\chi_{ij}(u)$ (see the module docstring).
DEFAULT_CHI_THRESHOLD: float = 0.3

#: Coordinates whose Hill tail-index estimate is below this value count as heavy-tailed.
DEFAULT_MAX_TAIL_INDEX: float = 4.0

#: Minimum number of exceedances per coordinate, $\lfloor (1 - u) n \rfloor$, needed for a verdict.
DEFAULT_MIN_EXCEEDANCES: int = 50


class TailAlignmentWarning(UserWarning):
    """Heavy tails appear to be shared across coordinates (a risk of unvisited tails; REPORT.md §5.1)."""


@dataclass(frozen=True)
class TailAlignmentReport:
    r"""Result of `tail_alignment_diagnostic`.

    Attributes:
        warn: Whether some pair of heavy coordinates has $\chi_{ij}$ above `chi_threshold`.
        max_chi: Largest $\chi_{ij}$ over pairs of heavy coordinates; NaN when there are fewer than
            two heavy coordinates or too few draws.
        chi: Pairwise co-exceedance matrix, shape `(d, d)`, symmetric; NaN on the diagonal and for
            any pair that includes a coordinate that is not heavy.
        offending_pairs: Pairs `(i, j)`, `i < j`, of heavy coordinates with $\chi_{ij}$ above the
            threshold, largest first.
        heavy_coordinates: Coordinates whose Hill tail index is below `max_tail_index`.
        tail_index: Hill tail-index estimate per coordinate, shape `(d,)` (NaN when undefined,
            infinite when the top values are all equal).
        level: The level $u$.
        chi_threshold: The warning threshold.
        max_tail_index: The heaviness cut-off on the Hill estimate.
        n_draws: Finite draws used (rows with a non-finite value are dropped).
        n_exceedances: $k = \lfloor (1 - u) n \rfloor$, the exceedances per coordinate.
        message: Plain-language summary; the text of the `TailAlignmentWarning` when `warn` is true.
    """

    warn: bool
    max_chi: float
    chi: np.ndarray
    offending_pairs: tuple[tuple[int, int], ...]
    heavy_coordinates: tuple[int, ...]
    tail_index: np.ndarray
    level: float
    chi_threshold: float
    max_tail_index: float
    n_draws: int
    n_exceedances: int
    message: str

    @property
    def independence_chi(self) -> float:
        r"""Value of $\chi_{ij}(u)$ for independent coordinates, $1 - u$."""
        return 1.0 - self.level


def _pooled_finite(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim != 2:
        raise ValueError(f"Expected samples of shape (chains, draws, d) or (draws, d), got shape {arr.shape}")
    return arr[np.all(np.isfinite(arr), axis=1)]


def _hill_and_exceedances(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Hill tail index from the k largest |x - median| per column, and the exceedance indicators."""
    n = x.shape[0]
    a = np.abs(x - np.median(x, axis=0))
    s = np.sort(a, axis=0)
    threshold = s[n - k - 1]
    top = s[n - k :]
    valid = threshold > 0.0
    safe_threshold = np.where(valid, threshold, 1.0)
    log_excess = np.mean(np.log(np.where(valid, top, 1.0)), axis=0) - np.log(safe_threshold)
    tail_index = np.full(x.shape[1], np.inf)
    np.divide(1.0, log_excess, out=tail_index, where=log_excess > 0.0)
    tail_index[~valid] = np.nan
    return tail_index, a > threshold


def tail_alignment_diagnostic(
    samples: np.ndarray,
    *,
    level: float = DEFAULT_LEVEL,
    chi_threshold: float = DEFAULT_CHI_THRESHOLD,
    max_tail_index: float = DEFAULT_MAX_TAIL_INDEX,
    min_exceedances: int = DEFAULT_MIN_EXCEEDANCES,
) -> TailAlignmentReport:
    r"""Checks whether heavy tails are shared across coordinates (see the module docstring).

    Parameters:
        samples: Draws of shape `(chains, draws, d)` (pooled over chains) or `(draws, d)`.
        level: Exceedance level $u \in (0, 1)$.
        chi_threshold: Warn when the largest pairwise $\chi_{ij}(u)$ exceeds this value ($> 0$).
        max_tail_index: Coordinates with Hill estimate below this value ($> 0$) count as heavy.
        min_exceedances: Minimum $\lfloor (1 - u) n \rfloor$ ($\ge 2$) for a verdict; below it the
            report has `warn=False` and says that there were too few draws.

    Returns:
        `TailAlignmentReport`. A warning flags a risk (tails shared across coordinates), not a
        proven miss.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if not chi_threshold > 0.0:
        raise ValueError(f"chi_threshold must be positive, got {chi_threshold}")
    if not max_tail_index > 0.0:
        raise ValueError(f"max_tail_index must be positive, got {max_tail_index}")
    if min_exceedances < 2:
        raise ValueError(f"min_exceedances must be at least 2, got {min_exceedances}")

    x = _pooled_finite(samples)
    n, d = x.shape
    k = int(np.floor((1.0 - level) * n + 1e-9))
    chi = np.full((d, d), np.nan)

    def report(
        warn: bool,
        max_chi: float,
        pairs: tuple[tuple[int, int], ...],
        heavy: tuple[int, ...],
        tail_index: np.ndarray,
        message: str,
    ) -> TailAlignmentReport:
        return TailAlignmentReport(
            warn=warn,
            max_chi=max_chi,
            chi=chi,
            offending_pairs=pairs,
            heavy_coordinates=heavy,
            tail_index=tail_index,
            level=float(level),
            chi_threshold=float(chi_threshold),
            max_tail_index=float(max_tail_index),
            n_draws=int(n),
            n_exceedances=k,
            message=message,
        )

    if k < min_exceedances:
        return report(
            False,
            float("nan"),
            (),
            (),
            np.full(d, np.nan),
            f"Too few draws for the tail-alignment check: {n} finite draws give {k} exceedances per coordinate "
            f"at level {level}, fewer than {min_exceedances}.",
        )

    tail_index, exceeds = _hill_and_exceedances(x, k)
    heavy = tuple(int(i) for i in np.flatnonzero(tail_index < max_tail_index))
    if len(heavy) < 2:
        return report(
            False,
            float("nan"),
            (),
            heavy,
            tail_index,
            f"Fewer than two heavy-tailed coordinates ({len(heavy)} with Hill tail index below {max_tail_index:g}); "
            "no shared heavy tails to flag.",
        )

    h = np.asarray(heavy)
    e = exceeds[:, h].astype(np.float64)
    sub = (e.T @ e) / k
    np.fill_diagonal(sub, np.nan)
    chi[np.ix_(h, h)] = sub
    max_chi = float(np.nanmax(sub))

    upper = [(int(i), int(j)) for a, i in enumerate(h) for j in h[a + 1 :] if chi[i, j] > chi_threshold]
    pairs = tuple(sorted(upper, key=lambda p: (-chi[p], p)))
    if not pairs:
        return report(
            False,
            max_chi,
            (),
            heavy,
            tail_index,
            f"No shared heavy tails detected: the largest tail co-exceedance over {len(heavy)} heavy coordinates is "
            f"chi = {max_chi:.2f}, below the threshold {chi_threshold:.2f} (about {1.0 - level:.2f} if independent).",
        )

    i, j = pairs[0]
    return report(
        True,
        max_chi,
        pairs,
        heavy,
        tail_index,
        f"Heavy tails appear to be shared across coordinates: tail co-exceedance chi = {max_chi:.2f} for "
        f"coordinates {i} and {j} (threshold {chi_threshold:.2f}; about {1.0 - level:.2f} if their tails were "
        f"independent), and {len(pairs)} pair(s) of heavy coordinates exceed the threshold. When heavy tails lie "
        "along directions that are not coordinate axes, coordinate-wise transforms such as the arcsinh map, and "
        "NUTS itself, can leave tail regions unvisited without R-hat noticing (REPORT.md §5.1). This flags a risk, "
        "not a proven miss: check the tails with a reparameterisation that exposes the shared scale (for example "
        "a Gaussian scale mixture) or with another method.",
    )


__all__ = [
    "DEFAULT_CHI_THRESHOLD",
    "DEFAULT_LEVEL",
    "DEFAULT_MAX_TAIL_INDEX",
    "DEFAULT_MIN_EXCEEDANCES",
    "TailAlignmentReport",
    "TailAlignmentWarning",
    "tail_alignment_diagnostic",
]
