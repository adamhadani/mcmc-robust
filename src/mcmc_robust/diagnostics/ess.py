"""Effective Sample Size (ESS) diagnostics: Rank-normalized Bulk ESS and Tail ESS.

References:
- Charles C. Margossian, Matthew D. Hoffman, Pavel Sountsov, et al. (2024),
  "For how many iterations should we run Markov chain Monte Carlo?",
  Bayesian Analysis / arXiv:2311.02726.
- Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, & Paul-Christian Bürkner (2021),
  "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC",
  Bayesian Analysis 16(2), 667-718.
- Charles J. Geyer (1992),
  "Practical Markov Chain Monte Carlo",
  Statistical Science 7(4), 473-483.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats


def _normalize_chains_input(
    chains: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    """Normalizes arbitrary MCMC chain inputs into a 3D ndarray of shape (M, N, d).

    Supported input formats:
    - 1D array of shape (N,): single chain, 1D target -> (1, N, 1)
    - 2D array of shape (N, d): single chain of N iterations, d dimensions -> (1, N, d)
    - 3D array of shape (M, N, d): M chains, N iterations, d dimensions -> (M, N, d)
    - Sequence (list/tuple) of arrays:
        - If each is 1D (N,): stacked to (M, N, 1)
        - If each is 2D (N, d): stacked to (M, N, d)

    Returns:
        np.ndarray of shape (M, N, d) with dtype float64.

    Raises:
        ValueError: If input dimensions are invalid or arrays cannot be stacked.
    """
    if isinstance(chains, (list, tuple)):
        if len(chains) == 0:
            raise ValueError("Input chains sequence cannot be empty.")
        stacked = np.stack([np.asarray(c, dtype=np.float64) for c in chains], axis=0)
        if stacked.ndim == 2:
            return stacked[:, :, np.newaxis]
        elif stacked.ndim == 3:
            return stacked
        else:
            raise ValueError(f"Stacked sequence has invalid shape {stacked.shape}, expected 2D or 3D.")

    arr = np.asarray(chains, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :, np.newaxis]
    elif arr.ndim == 2:
        return arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        return arr
    else:
        raise ValueError(f"Expected array of ndim 1, 2, or 3; got {arr.ndim}")


def _split_chains_in_half(chains: np.ndarray) -> np.ndarray:
    """Splits each chain in half along the time axis to detect non-stationarity.

    Given chains of shape (M, N, d):
    Truncates N to 2 * (N // 2) if odd, and splits each chain into two halves:
    - First half: chains[:, :N // 2, :]
    - Second half: chains[:, N // 2 : 2 * (N // 2), :]
    Concatenates halves along axis 0 to produce shape (2 * M, N // 2, d).

    Args:
        chains: 3D array of shape (M, N, d).

    Returns:
        3D array of shape (2 * M, N // 2, d).
    """
    M, N, d = chains.shape
    half = N // 2
    if half < 2:
        return chains
    h1 = chains[:, :half, :]
    h2 = chains[:, half : 2 * half, :]
    return np.concatenate([h1, h2], axis=0)


def _compute_multi_chain_variogram(z: np.ndarray) -> np.ndarray:
    """Computes the cross-lag variogram V_t across M chains using FFT in O(N log N).

    For lag t in 0 ... N - 1:
        V_t = 1 / (M * (N - t)) * sum_{m=1}^M sum_{n=t+1}^N (z_{m, n} - z_{m, n-t})^2

    Expanding the square:
        (z_n - z_{n-t})^2 = z_n^2 + z_{n-t}^2 - 2 * z_n * z_{n-t}
    The cross-product sum is the sample autocovariance computed via FFT, and
    the squared terms are computed via cumulative sums in O(1) per lag.

    Args:
        z: 2D array of shape (M, N) containing normalized samples.

    Returns:
        1D array V_t of shape (N,) for lags t in 0 ... N - 1.
    """
    M, N = z.shape
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))

    # FFT-based cross-product sum: sum_{n=t}^{N-1} z_n * z_{n-t}
    fz = np.fft.rfft(z, n=n_fft, axis=1)
    autocov = np.fft.irfft(fz * np.conj(fz), n=n_fft, axis=1)[:, :N]

    # Cumulative sums of z^2 for O(1) windowed sums
    z2 = z**2
    z2_fwd = np.pad(np.cumsum(z2, axis=1), ((0, 0), (1, 0)))
    z2_bwd = np.pad(np.cumsum(z2[:, ::-1], axis=1), ((0, 0), (1, 0)))

    lags = np.arange(N)
    sum_end = z2_fwd[:, N : N + 1] - z2_fwd[:, lags]
    sum_start = z2_bwd[:, N : N + 1] - z2_bwd[:, lags]

    diff2_sum = sum_end + sum_start - 2.0 * autocov
    counts = np.maximum(1, N - lags)[np.newaxis, :]
    v_per_chain = diff2_sum / counts
    return np.mean(v_per_chain, axis=0)


def _geyer_monotone_autocorr_time(rho: np.ndarray) -> float:
    """Computes integrated autocorrelation time using Geyer's initial monotone sequence.

    Forms pair sums P_k = rho[2k] + rho[2k+1].
    Finds the initial positive sequence (first k where P_k <= 0), then applies
    monotone non-increasing smoothing:
        P_tilde_k = min(P_tilde_{k-1}, P_k)
    The integrated autocorrelation time is:
        tau = -1 + 2 * sum_{k=0}^{K_pos - 1} P_tilde_k

    Args:
        rho: 1D autocorrelation array of shape (N,) with rho[0] = 1.0.

    Returns:
        Estimated integrated autocorrelation time tau (clamped to >= 1.0).
    """
    N = len(rho)
    K_pairs = N // 2
    if K_pairs < 1:
        return 1.0

    pair_sums = rho[0 : 2 * K_pairs : 2] + rho[1 : 2 * K_pairs : 2]

    # 1. Initial positive sequence condition: stop at first k where P_k <= 0
    pos_idx = 0
    while pos_idx < len(pair_sums) and pair_sums[pos_idx] > 0.0:
        pos_idx += 1

    if pos_idx == 0:
        return 1.0

    # 2. Initial monotone sequence condition: enforce non-increasing order
    p_trunc = pair_sums[:pos_idx].copy()
    for i in range(1, len(p_trunc)):
        p_trunc[i] = min(p_trunc[i - 1], p_trunc[i])
        if p_trunc[i] <= 0.0:
            p_trunc = p_trunc[:i]
            break

    tau = -1.0 + 2.0 * float(np.sum(p_trunc))
    return max(1.0, tau)


def _compute_single_coord_ess(split_chains: np.ndarray) -> float:
    """Computes multi-chain ESS for a single 1D coordinate across split chains.

    Args:
        split_chains: 2D array of shape (S, N_prime) with S split chains of length N_prime.

    Returns:
        Scalar effective sample size as a float.
    """
    S, N_prime = split_chains.shape
    total_samples = float(S * N_prime)

    if N_prime < 4:
        return total_samples

    # Check for NaN or Inf
    if not np.all(np.isfinite(split_chains)):
        return np.nan

    # Within-chain variance W and between-chain variance B
    chain_means = np.mean(split_chains, axis=1)
    grand_mean = np.mean(chain_means)
    chain_vars = np.var(split_chains, axis=1, ddof=1)
    W = float(np.mean(chain_vars))

    if S > 1:
        B = float((N_prime / (S - 1)) * np.sum((chain_means - grand_mean) ** 2))
    else:
        B = 0.0

    var_plus = ((N_prime - 1.0) / N_prime) * W + (1.0 / N_prime) * B

    # Guard: zero variance / constant chain
    if var_plus < 1e-14 or W < 1e-14:
        return 0.0

    # Variogram and autocorrelation estimation
    V_t = _compute_multi_chain_variogram(split_chains)
    rho = 1.0 - V_t / (2.0 * var_plus)
    rho[0] = 1.0

    # Geyer's initial monotone sequence truncation
    tau = _geyer_monotone_autocorr_time(rho)
    return total_samples / tau


def compute_bulk_ess(
    chains: np.ndarray | Sequence[np.ndarray],
    split_chains: bool = True,
    reduction: str | None = None,
) -> np.ndarray | float:
    r"""Computes rank-normalized Bulk Effective Sample Size (ESS) per coordinate.

    Follows Margossian et al. (2024) and Vehtari et al. (2021):
    1. Transforms samples into rank-normalized normal scores via Blom's formula:
       z_i = \Phi^{-1}((r_i - 3/8) / (T + 1/4))
    2. Splits each chain in half along the time axis (if split_chains=True).
    3. Estimates cross-lag variograms via FFT.
    4. Computes integrated autocorrelation time \hat{\tau} via Geyer's initial
       monotone sequence.
    5. Returns ESS = M * N / \hat{\tau}.

    Args:
        chains: Input MCMC chains. Accepts:
            - 1D array of shape (N,)
            - 2D array of shape (N, d)
            - 3D array of shape (M, N, d)
            - Sequence of 1D or 2D arrays
        split_chains: Whether to split chains in half (default True).
        reduction: Optional aggregation: None (returns ndarray of shape (d,)),
            'min' (returns float min across dimensions), or 'mean' (float mean).

    Returns:
        np.ndarray of shape (d,) with Bulk ESS per coordinate, or float if reduction is set.
    """
    arr = _normalize_chains_input(chains)
    M, N, d = arr.shape
    ess = np.empty(d, dtype=np.float64)

    for j in range(d):
        col = arr[:, :, j]
        v = col.ravel()

        # Guard: constant values or NaNs
        if not np.all(np.isfinite(v)):
            ess[j] = np.nan
            continue
        if np.var(v) < 1e-14:
            ess[j] = 0.0
            continue

        # Rank-normalization using Blom (1958) scores
        ranks = stats.rankdata(v, method="average")
        total_len = len(v)
        normal_scores = stats.norm.ppf((ranks - 0.375) / (total_len + 0.25))
        z = normal_scores.reshape(M, N)

        if split_chains and N >= 4:
            z_split = _split_chains_in_half(z[:, :, np.newaxis])[:, :, 0]
        else:
            z_split = z

        ess[j] = _compute_single_coord_ess(z_split)

    if reduction == "min":
        return float(np.min(ess))
    elif reduction == "mean":
        return float(np.mean(ess))
    return ess


def compute_mean_ess(
    chains: np.ndarray | Sequence[np.ndarray],
    split_chains: bool = True,
    reduction: str | None = None,
) -> np.ndarray | float:
    r"""Computes the Effective Sample Size for the mean of each coordinate, without rank normalization.

    The estimator of `compute_bulk_ess` (split chains, multi-chain variogram, Geyer's initial monotone
    sequence) applied to the values themselves, so $\mathrm{Var}(\bar x) \approx \sigma^2 / \mathrm{ESS}$
    when the variance is finite. Chains that disagree, or drift between their halves, give a small ESS.
    It is clamped to at most the number of draws.

    Args:
        chains: Input MCMC chains, in any shape `compute_bulk_ess` accepts.
        split_chains: Whether to split chains in half (default True).
        reduction: Optional aggregation: None (ndarray of shape (d,)), 'min' or 'mean' (float).

    Returns:
        np.ndarray of shape (d,), NaN for a coordinate with non-finite draws and 0 for a constant one, or a
        float if `reduction` is set.
    """
    arr = _normalize_chains_input(chains)
    M, N, d = arr.shape
    ess = np.empty(d, dtype=np.float64)

    for j in range(d):
        col = arr[:, :, j]
        if not np.all(np.isfinite(col)):
            ess[j] = np.nan
            continue
        if np.var(col) < 1e-14:
            ess[j] = 0.0
            continue
        split = _split_chains_in_half(col[:, :, np.newaxis])[:, :, 0] if split_chains and N >= 4 else col
        ess[j] = _compute_single_coord_ess(split)

    if reduction == "min":
        return float(np.min(ess))
    elif reduction == "mean":
        return float(np.mean(ess))
    return ess


def compute_tail_ess(
    chains: np.ndarray | Sequence[np.ndarray],
    prob_low: float = 0.05,
    prob_high: float = 0.95,
    split_chains: bool = True,
    reduction: str | None = None,
) -> np.ndarray | float:
    r"""Computes indicator-based Tail Effective Sample Size per coordinate.

    Follows Margossian et al. (2024) and Vehtari et al. (2021):
    1. Computes pooled empirical quantiles q_{low} and q_{high}.
    2. Constructs binary indicator series:
       I_{low}(x) = \mathbb{I}(x \le q_{low})
       I_{high}(x) = \mathbb{I}(x \ge q_{high})
    3. Computes ESS on both indicator series using multi-chain variograms
       and Geyer's monotone truncation.
    4. Returns Tail-ESS = \min(ESS(I_{low}), ESS(I_{high})).

    Args:
        chains: Input MCMC chains. Accepts:
            - 1D array of shape (N,)
            - 2D array of shape (N, d)
            - 3D array of shape (M, N, d)
            - Sequence of 1D or 2D arrays
        prob_low: Lower tail quantile probability (default 0.05).
        prob_high: Upper tail quantile probability (default 0.95).
        split_chains: Whether to split chains in half (default True).
        reduction: Optional aggregation: None (returns ndarray of shape (d,)),
            'min' (returns float min across dimensions), or 'mean' (float mean).

    Returns:
        np.ndarray of shape (d,) with Tail ESS per coordinate, or float if reduction is set.
    """
    arr = _normalize_chains_input(chains)
    M, N, d = arr.shape
    tail_ess = np.empty(d, dtype=np.float64)

    for j in range(d):
        col = arr[:, :, j]
        v = col.ravel()

        if not np.all(np.isfinite(v)):
            tail_ess[j] = np.nan
            continue
        if np.var(v) < 1e-14:
            tail_ess[j] = 0.0
            continue

        q_low, q_high = np.quantile(v, [prob_low, prob_high])
        ind_low = (col <= q_low).astype(np.float64)
        ind_high = (col >= q_high).astype(np.float64)

        if split_chains and N >= 4:
            low_split = _split_chains_in_half(ind_low[:, :, np.newaxis])[:, :, 0]
            high_split = _split_chains_in_half(ind_high[:, :, np.newaxis])[:, :, 0]
        else:
            low_split = ind_low
            high_split = ind_high

        ess_low = _compute_single_coord_ess(low_split)
        ess_high = _compute_single_coord_ess(high_split)
        tail_ess[j] = min(ess_low, ess_high)

    if reduction == "min":
        return float(np.min(tail_ess))
    elif reduction == "mean":
        return float(np.mean(tail_ess))
    return tail_ess
