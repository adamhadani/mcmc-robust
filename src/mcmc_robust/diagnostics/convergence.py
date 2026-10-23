"""Convergence diagnostics: Multi-chain Split R-hat and Folded R-hat.

References:
- Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, & Paul-Christian Bürkner (2021),
  "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC",
  Bayesian Analysis 16(2), 667-718.
- Andrew Gelman & Donald B. Rubin (1992),
  "Inference from Iterative Simulation Using Multiple Sequences",
  Statistical Science 7(4), 457-472.
"""

from collections.abc import Sequence

import numpy as np

from mcmc_robust.diagnostics.ess import _normalize_chains_input, _split_chains_in_half


def compute_split_rhat(
    chains: np.ndarray | Sequence[np.ndarray],
    split_chains: bool = True,
    reduction: str | None = None,
) -> np.ndarray | float:
    r"""Computes multi-chain split Gelman-Rubin \hat{R} convergence diagnostic per coordinate.

    Splits each chain in half along the iteration axis to detect non-stationarity,
    then computes between-chain variance B and within-chain variance W:
        W = 1 / S * sum_{s=1}^S s_s^2
        B = N' / (S - 1) * sum_{s=1}^S (\bar{x}_s - \bar{x}_{\cdot})^2
        \widehat{\text{var}}^+ = (N' - 1) / N' * W + 1 / N' * B
        \hat{R} = \sqrt{\widehat{\text{var}}^+ / W}

    Args:
        chains: Input MCMC chains. Accepts:
            - 1D array of shape (N,)
            - 2D array of shape (N, d)
            - 3D array of shape (M, N, d)
            - Sequence of 1D or 2D arrays
        split_chains: Whether to split chains in half (default True).
        reduction: Optional aggregation: None (returns ndarray of shape (d,)),
            'max' (returns float max across dimensions), or 'mean' (float mean).

    Returns:
        np.ndarray of shape (d,) with split \hat{R} per coordinate, or float if reduction is set.
    """
    arr = _normalize_chains_input(chains)
    M, N, d = arr.shape

    if split_chains and N >= 4:
        split_arr = _split_chains_in_half(arr)
    else:
        split_arr = arr

    S, N_prime, d = split_arr.shape

    # If only 1 chain and not split, or length < 2, cannot compute between-chain variance
    if S < 2 or N_prime < 2:
        res = np.ones(d, dtype=np.float64)
        return float(res[0]) if reduction is not None else res

    rhat = np.empty(d, dtype=np.float64)

    # Check for NaN / Inf per coordinate cleanly
    is_nan_coord = np.any(np.isnan(split_arr), axis=(0, 1))
    is_inf_coord = np.any(np.isinf(split_arr), axis=(0, 1))

    for j in range(d):
        if is_nan_coord[j]:
            rhat[j] = np.nan
        elif is_inf_coord[j]:
            rhat[j] = np.inf

    valid_coords = ~(is_nan_coord | is_inf_coord)
    if np.any(valid_coords):
        valid_split = split_arr[:, :, valid_coords]
        with np.errstate(all="ignore"):
            chain_means = np.mean(valid_split, axis=1)  # (S, d_valid)
            grand_mean = np.mean(chain_means, axis=0)  # (d_valid,)
            chain_vars = np.var(valid_split, axis=1, ddof=1)  # (S, d_valid)

            W = np.mean(chain_vars, axis=0)  # (d_valid,)
            B = (N_prime / (S - 1.0)) * np.sum((chain_means - grand_mean) ** 2, axis=0)  # (d_valid,)
            var_plus = ((N_prime - 1.0) / N_prime) * W + (1.0 / N_prime) * B  # (d_valid,)

            valid_indices = np.where(valid_coords)[0]
            for idx_k, j in enumerate(valid_indices):
                w_j = W[idx_k]
                b_j = B[idx_k]
                vp_j = var_plus[idx_k]
                if w_j < 1e-14:
                    if b_j > 1e-14:
                        # Chains stuck at separated constant values
                        rhat[j] = np.inf
                    else:
                        # Chains identical constants
                        rhat[j] = 1.0
                else:
                    ratio = vp_j / w_j
                    rhat[j] = np.sqrt(max(1.0, ratio))

    if reduction == "max":
        return float(np.max(rhat))
    elif reduction == "mean":
        return float(np.mean(rhat))
    return rhat


def compute_folded_rhat(
    chains: np.ndarray | Sequence[np.ndarray],
    split_chains: bool = True,
    reduction: str | None = None,
) -> np.ndarray | float:
    r"""Computes folded split \hat{R} to detect differences in scale and tail variance.

    Transforms samples by folding around the overall median:
        y_{m, n, j} = |x_{m, n, j} - \text{median}(x_{\cdot, \cdot, j})|
    then computes split \hat{R} on the folded series y. This converts differences
    in scale or tail thickness into differences in mean, making \hat{R} sensitive
    to localized variance or tail discrepancies (Vehtari et al. 2021).

    Args:
        chains: Input MCMC chains. Accepts:
            - 1D array of shape (N,)
            - 2D array of shape (N, d)
            - 3D array of shape (M, N, d)
            - Sequence of 1D or 2D arrays
        split_chains: Whether to split chains in half (default True).
        reduction: Optional aggregation: None (returns ndarray of shape (d,)),
            'max' (returns float max across dimensions), or 'mean' (float mean).

    Returns:
        np.ndarray of shape (d,) with folded \hat{R} per coordinate, or float if reduction is set.
    """
    arr = _normalize_chains_input(chains)
    M, N, d = arr.shape

    folded_arr = np.empty_like(arr)

    is_nan_coord = np.any(np.isnan(arr), axis=(0, 1))
    is_inf_coord = np.any(np.isinf(arr), axis=(0, 1))

    for j in range(d):
        if is_nan_coord[j]:
            folded_arr[:, :, j] = np.nan
        elif is_inf_coord[j]:
            folded_arr[:, :, j] = np.inf

    valid_coords = ~(is_nan_coord | is_inf_coord)
    if np.any(valid_coords):
        valid_arr = arr[:, :, valid_coords]
        pooled = valid_arr.reshape(-1, int(np.sum(valid_coords)))
        with np.errstate(all="ignore"):
            medians = np.median(pooled, axis=0)  # (d_valid,)
            folded_arr[:, :, valid_coords] = np.abs(valid_arr - medians)

    return compute_split_rhat(folded_arr, split_chains=split_chains, reduction=reduction)
