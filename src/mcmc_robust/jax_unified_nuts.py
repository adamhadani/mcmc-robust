r"""Pure-JAX 3-Phase Adaptive Arcsinh-Warped NUTS (`sample_blackjax_unified_nuts`).

Runs BlackJAX NUTS on the exact pullback of the target under a calibrated coordinate-wise
arcsinh diffeomorphism (`jax.vmap` over chains, `jax.jit` + `jax.lax.scan` for sampling).
The NUTS kernel always targets the exact transformed density $\log \pi_Z$; nothing is
clipped, floored or smoothed, so post-warm-up draws (and divergence counts) refer to the
posterior itself.

It follows the same three-phase plan as the NumPy `UnifiedRobustNUTS`
(`samplers/novel/nuts.py`) with the same transform and Phase 1 calibration rule, but it is
not a line-by-line port:
- no gentle stiffness damping (it corresponds to `UnifiedRobustNUTS(gamma=0.0)`);
- Phases 2 and 3 are BlackJAX's Stan-style `window_adaptation` rather than the NumPy
  15% / 65% / 20% schedule;
- Phase 1 calibrates from the second half of its draws, pooled across chains;
- each chain samples with its own adapted step size and diagonal inverse mass matrix.

1. **Coordinate Diffeomorphism (`JaxArcsinhTransform` $\leftrightarrow$ `CoordinateWiseArcsinhTransform`)**:
   - Forward map: $z_i = \operatorname{arcsinh}((x_i - \mu_i) / R_i)$ on masked coordinates (`mask[i] == True`).
   - Inverse map: $x_i = \mu_i + R_i \sinh(z_i)$ with overflow-free log-Jacobian determinant
     $\log |\det J_\psi(z)| = \sum_{i: m_i=1} (\log R_i + |z_i| - \log 2 + \operatorname{log1p}(e^{-2|z_i|}))$.
   - Pullback gradient: $\nabla_{z_i} \log \pi_Z(z) = \sqrt{R_i^2 + (x_i - \mu_i)^2} \nabla_{x_i} \log \pi_X(x)
     + (x_i - \mu_i) / \sqrt{R_i^2 + (x_i - \mu_i)^2}$.

2. **Phase 1 — Coordinate Location & Tail-Scale Calibration ($W_1 = \max(10, \lfloor 0.15 \cdot \text{tune} \rfloor)$)**:
   - Same rule as `UnifiedRobustNUTS.adapt` and
     `CoordinateWiseArcsinhTransform.compute_phase1_calibration`:
     $$\mu_i = \operatorname{median}(x_{\cdot, i}), \qquad
       R_i = \min\!\left(R_{\max},\, \max\!\left(R_{\min},\, c \cdot \hat{\sigma}_i\right)\right)$$
     where $\hat{\sigma}_i = \operatorname{IQR}(x_{\cdot, i} - \mu_i) / 1.3489795$ when
     $\hat{\sigma}_{\text{IQR}, i} > R_{\min}$, falling back to
     $\max(\operatorname{MAD}(x_{\cdot, i} - \mu_i) / 0.67448975,\, q_{0.90}(|x_{\cdot, i} - \mu_i|))$.
   - Formally verified in `formal/McmcRobustFormal.lean` §8 (`calibratedRadius_ge_min`,
     `calibratedRadius_pos`, `calibratedRadius_le_max`, `calibratedRadius_mono`,
     `calibrated_shifted_roundtrip_forward`, `calibrated_shifted_roundtrip_inverse`).

3. **Phase 2 & 3 — Joint Metric & Step-Size Window Adaptation ($W_2 = \max(10, \text{tune} - W_1)$)**:
   - Maps Phase 1 terminal primal positions into the newly calibrated $z$-space
     (`z_start = transform.forward(x_p1_final)`), as `UnifiedRobustNUTS.adapt` does at the
     Phase 1 $\to$ Phase 2 transition.
   - Runs BlackJAX `window_adaptation` per chain in calibrated $z$-space to adapt the diagonal
     inverse mass matrix $M^{-1}$ (Welford variance in $z$-space) and the step size $h$
     (dual averaging targeting `target_accept`).

4. **Post-Warmup Sampling (`draws` steps per chain)**:
   - Executes `jax.jit` + `jax.vmap` + `jax.lax.scan` in calibrated $z$-space with each chain's own
     adapted parameters, transforms draws back to primal $x$-space via `transform.inverse`, and
     records exact leapfrog gradient evaluations
     (`grad_evals == phase1_grad_evals + phase2_grad_evals + sampling_grad_evals`).

5. **Ablation controls (`transform_kind`)**:
   - `"arcsinh"` (default): the pipeline above.
   - `"affine"`: the same three phases and the same Phase 1 calibration rule, but with the affine
     map $z_i = (x_i - \mu_i) / R_i$ (`JaxAffineTransform`), so polynomial tails are not warped.
   - `"identity"`: no transform and no Phase 1; one BlackJAX `window_adaptation` of `tune` steps
     per chain in $x$-space, i.e. stock BlackJAX NUTS sharing this module's sampling code and
     gradient accounting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import blackjax
import jax.numpy as jnp
import numpy as np
from scipy import optimize, stats
from xarray import DataTree

from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.integrations import (
    JaxAffineTransform,
    JaxArcsinhTransform,
    JaxCoordinateTransform,
    JaxTukeyHTransform,
    lambertw_from_log,
)

#: Coordinate transforms supported by `sample_blackjax_unified_nuts` (see the module docstring, item 5).
TransformKind = Literal["arcsinh", "affine", "identity"]
TRANSFORM_KINDS: tuple[str, ...] = ("arcsinh", "affine", "identity")


def arviz_from_samples(
    samples: np.ndarray,
    sample_stats: dict[str, np.ndarray] | None = None,
) -> DataTree:
    r"""Constructs an ArviZ `DataTree` from a 3D MCMC samples array of shape `(chains, draws, d)`.

    Parameters:
        samples: Primal MCMC draws array of shape `(chains, draws, d)` (or 1D/2D array).
        sample_stats: Optional dictionary of per-step sample statistics arrays of shape `(chains, draws)`.

    Returns:
        ArviZ `DataTree` containing `/posterior` (`"x"`) and optional `/sample_stats`.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :, np.newaxis]
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 1D, 2D, or 3D samples array, got shape {arr.shape}")

    data_dict: dict[str, Any] = {"posterior": {"x": arr}}
    if sample_stats is not None:
        data_dict["sample_stats"] = sample_stats
    return az.from_dict(data_dict)


@dataclass
class JaxUnifiedNUTSResult:
    r"""Container for pure-JAX 3-phase adaptive `UnifiedRobustNUTS` sampling results.

    Attributes:
        samples: Post-warmup draws in primal $x$-space, shape `(chains, draws, d)`.
        z_samples: Post-warmup draws in warped $z$-space, shape `(chains, draws, d)`.
        transform: Calibrated transform (`radii`, `location`, `mask`): a `JaxArcsinhTransform` by
            default, a `JaxAffineTransform` for `transform_kind="affine"`, and a unit-scale,
            zero-location `JaxAffineTransform` (the identity) for `transform_kind="identity"`.
        step_size: Step size each chain sampled with, shape `(chains,)`.
        inverse_mass_matrix: Diagonal inverse mass matrix each chain sampled with in $z$-space,
            shape `(chains, d)`.
        grad_evals: Exact total leapfrog gradient evaluations (`phase1 + phase2 + sampling`).
        divergences: Total post-warmup divergent transitions across all chains.
        acceptance_rate: Mean post-warmup Metropolis/multinomial acceptance probability.
        phase1_grad_evals: Exact gradient evaluations during Phase 1 scale/location calibration.
        phase2_grad_evals: Exact gradient evaluations during Phase 2/3 metric & step-size warmup.
        sampling_grad_evals: Exact gradient evaluations during post-warmup production sampling.
        idata: ArviZ `DataTree` object constructed via `arviz_from_samples(samples)`.
    """

    samples: np.ndarray
    z_samples: np.ndarray
    transform: JaxCoordinateTransform
    step_size: np.ndarray
    inverse_mass_matrix: np.ndarray
    grad_evals: int
    divergences: int
    acceptance_rate: float
    phase1_grad_evals: int
    phase2_grad_evals: int
    sampling_grad_evals: int
    idata: DataTree


def _phase1_location_and_radii(
    pilot_samples: Any,
    d: int | None,
    min_radius: float,
    max_radius: float,
    scale_multiplier: float,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    r"""Phase 1 calibration rule shared by every transform kind; returns `(location, radii, d)`.

    See `calibrate_jax_arcsinh_transform` for the formula.
    """
    if min_radius <= 0.0:
        raise ValueError(f"min_radius must be strictly positive, got {min_radius}")
    if min_radius > max_radius:
        raise ValueError(f"min_radius ({min_radius}) must be <= max_radius ({max_radius})")
    if scale_multiplier < 0.0:
        raise ValueError(f"scale_multiplier must be non-negative, got {scale_multiplier}")

    arr = jnp.asarray(pilot_samples, dtype=jnp.float64)
    if arr.ndim == 1:
        if d is not None and d > 1:
            arr = arr.reshape(-1, d)
        else:
            arr = arr[:, jnp.newaxis]
    elif arr.ndim >= 3:
        arr = arr.reshape(-1, arr.shape[-1])

    d_actual = int(arr.shape[-1])
    if d is not None and int(d) != d_actual:
        raise ValueError(f"pilot_samples last dimension {d_actual} does not match d={d}")

    # Exact parity with CoordinateWiseArcsinhTransform.compute_phase1_calibration
    med = jnp.median(arr, axis=0)
    centered = arr - med
    abs_centered = jnp.abs(centered)

    q75, q25 = jnp.quantile(centered, jnp.array([0.75, 0.25], dtype=jnp.float64), axis=0)
    iqr_scale = (q75 - q25) / 1.3489795
    mad_scale = jnp.median(abs_centered, axis=0) / 0.67448975
    q90_abs = jnp.quantile(abs_centered, 0.90, axis=0)

    base_scale = jnp.where(iqr_scale > min_radius, iqr_scale, jnp.maximum(mad_scale, q90_abs))
    base_scale = jnp.where(jnp.isfinite(base_scale), base_scale, 1.0)
    med_clean = jnp.where(jnp.isfinite(med), med, 0.0)

    # Matches Lean 4 `calibratedRadius σ Rmin Rmax c := min Rmax (max Rmin (c * σ))`
    radii = jnp.minimum(
        jnp.float64(max_radius),
        jnp.maximum(jnp.float64(min_radius), jnp.float64(scale_multiplier) * base_scale),
    )
    return med_clean, radii, d_actual


def calibrate_jax_affine_transform(
    pilot_samples: Any,
    d: int | None = None,
    mask: Any | None = None,
    min_radius: float = 0.1,
    max_radius: float = 100.0,
    scale_multiplier: float = 1.5,
) -> JaxAffineTransform:
    r"""Calibrates a `JaxAffineTransform` with exactly the Phase 1 rule of `calibrate_jax_arcsinh_transform`.

    Parameters and the formula for $(\mu, R)$ are those of `calibrate_jax_arcsinh_transform`.
    """
    location, radii, d_actual = _phase1_location_and_radii(pilot_samples, d, min_radius, max_radius, scale_multiplier)
    return JaxAffineTransform(radii=radii, location=location, mask=mask, d=d_actual)


def calibrate_jax_arcsinh_transform(
    pilot_samples: Any,
    d: int | None = None,
    mask: Any | None = None,
    min_radius: float = 0.1,
    max_radius: float = 100.0,
    scale_multiplier: float = 1.5,
) -> JaxArcsinhTransform:
    r"""Calibrates shift-equivariant coordinate medians $\mu_i$ and clipped scales $R_i$ from pilot draws.

    Implements the exact pure-JAX counterpart of
    `CoordinateWiseArcsinhTransform.compute_phase1_calibration` (`transforms/coordinate_arcsinh.py`)
    and `UnifiedRobustNUTS.adapt` (`samplers/novel/nuts.py`), matching the formally verified
    Phase 1 calibration map (`formal/McmcRobustFormal.lean` §8):
        $$\mu_i = \operatorname{median}(x_{\cdot, i}), \qquad
          R_i = \min\!\left(R_{\max},\, \max\!\left(R_{\min},\, c \cdot \hat{\sigma}_i\right)\right)$$
    where $\hat{\sigma}_i = \operatorname{IQR}(x_{\cdot, i} - \mu_i) / 1.3489795$ when
    $\hat{\sigma}_{\text{IQR}, i} > R_{\min}$, falling back to
    $\max(\operatorname{MAD}(x_{\cdot, i} - \mu_i) / 0.67448975,\, q_{0.90}(|x_{\cdot, i} - \mu_i|))$.

    Parameters:
        pilot_samples: Exploratory primal draws of shape `(n_samples, d)` or `(chains, draws, d)`.
        d: Target dimension $d \ge 1$. If `None`, inferred from `pilot_samples.shape[-1]`.
        mask: Optional boolean mask of shape `(d,)` selecting coordinates to warp (default: all `True`).
        min_radius: Strictly positive lower bound $R_{\min} > 0$ on calibrated radii.
        max_radius: Upper bound $R_{\max} \ge R_{\min}$ on calibrated radii.
        scale_multiplier: Scale multiplier $c \ge 0$ applied to base dispersion $\hat{\sigma}_i$.

    Returns:
        Calibrated `JaxArcsinhTransform` with `.location`, `.radii`, and `.mask`.
    """
    location, radii, d_actual = _phase1_location_and_radii(pilot_samples, d, min_radius, max_radius, scale_multiplier)
    return JaxArcsinhTransform(
        radii=radii,
        location=location,
        mask=mask,
        d=d_actual,
    )


#: Default level of the per-coordinate likelihood-ratio test of h = 0 in `calibrate_jax_tukey_h_transform`.
TUKEY_H_TEST_LEVEL: float = 0.01

#: Default upper bound on the fitted h, i.e. a fitted Tukey-h tail index 1 / h of at least 0.1.
TUKEY_H_MAX_H: float = 10.0

#: Coordinates with fewer finite pilot draws than this are not tested and keep h = 0.
TUKEY_H_MIN_DRAWS: int = 10

#: The (log R, h) search spans R within this factor of the coordinate's normal-consistent IQR scale.
_TUKEY_H_RADIUS_SEARCH_FACTOR: float = 1e3

_IQR_TO_SIGMA: float = 1.3489795003921634  # interquartile range of N(0, 1)
_MEAN_ABS_TO_SIGMA: float = 0.7978845608028654  # E|Z| = sqrt(2 / pi) for Z ~ N(0, 1)
_LOG_SQRT_2PI: float = 0.9189385332046727  # log sqrt(2 pi)


@dataclass(frozen=True, eq=False)
class TukeyHCalibration:
    r"""Result of `calibrate_jax_tukey_h_transform`: the calibrated transform and the per-coordinate fit.

    Arrays have shape `(d,)`. A coordinate that was not fitted (too few finite draws, or no spread)
    has `radius_mle = h_mle = NaN`, `lr_statistic = 0` and `p_value = 1`.

    Attributes:
        transform: `JaxTukeyHTransform` with `location`, `radii`, `h` and the mask.
        location: $\mu_i$, the median of the finite draws.
        radii: $R_i$ of the transform, after the clip to `[min_radius, max_radius]`.
        h: $h_i$ of the transform: 0 unless `heavy[i]`, rescaled with the clip of $R_i$.
        heavy: Whether coordinate $i$ is masked and the likelihood-ratio test rejected $h_i = 0$.
        lr_statistic: $\Lambda_i = 2(\hat\ell_{1,i} - \hat\ell_{0,i}) \ge 0$, the maximised Tukey-h log-likelihood
            minus the maximised Gaussian ($h = 0$) log-likelihood, both with location $\mu_i$.
        p_value: $\tfrac12 P(\chi^2_1 > \Lambda_i)$ for $\Lambda_i > 0$ and 1 otherwise.
        radius_mle: $\hat R_i$ of the Tukey-h fit (before clipping).
        h_mle: $\hat h_i \in [0, h_{\max}]$ of the Tukey-h fit.
        radius_gaussian: $\hat R_{0,i} = (\overline{(x_i - \mu_i)^2})^{1/2}$, the $h = 0$ fit.
        level: Level of the test.
    """

    transform: JaxTukeyHTransform
    location: np.ndarray
    radii: np.ndarray
    h: np.ndarray
    heavy: np.ndarray
    lr_statistic: np.ndarray
    p_value: np.ndarray
    radius_mle: np.ndarray
    h_mle: np.ndarray
    radius_gaussian: np.ndarray
    level: float


def _tukey_h_mean_nll_and_grad(theta: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Mean negative log-likelihood of centred draws $u = x - \mu$ under $x = \mu + R z e^{h z^2 / 2}$, $z \sim N(0, 1)$.

    With $\theta = (\log R, h)$, $w = W(h u^2 / R^2)$ and $z^2 = (u / R)^2 e^{-w} = w / h$:
    $$-\log f(x) = \tfrac12 z^2 + \tfrac12 w + \log R + \mathrm{log1p}(w) + \log\sqrt{2\pi},$$
    using $h z^2 = w$. With $W'(y) = e^{-W} / (1 + W)$ the gradient is
    $\partial_{\log R} = 1 - (z^2 + w) / (1 + w) - 2 w / (1 + w)^2$ and
    $\partial_h = z^2 (1 - z^2) / (2 (1 + w)) + z^2 / (1 + w)^2$ per draw.
    """
    log_r, h = float(theta[0]), max(float(theta[1]), 0.0)
    abs_u = np.abs(u)
    nonzero = abs_u > 0.0
    log_abs_a = np.where(nonzero, np.log(np.where(nonzero, abs_u, 1.0)), -np.inf) - log_r
    with np.errstate(over="ignore"):
        if h > 0.0:
            w = lambertw_from_log(np.log(h) + 2.0 * log_abs_a, xp=np)
            z2 = np.where(w > 1.0, w / h, np.exp(2.0 * log_abs_a - w))
        else:
            w = np.zeros_like(abs_u)
            z2 = np.exp(2.0 * log_abs_a)
    w1 = 1.0 + w
    nll = float(np.mean(0.5 * z2 + 0.5 * w + np.log1p(w))) + log_r + _LOG_SQRT_2PI
    g_log_r = float(np.mean(1.0 - (z2 + w) / w1 - 2.0 * w / w1**2))
    g_h = float(np.mean(0.5 * z2 * (1.0 - z2) / w1 + z2 / w1**2))
    return nll, np.array([g_log_r, g_h])


def _fit_tukey_h_mle(u: np.ndarray, max_h: float) -> tuple[float, float, float]:
    r"""Maximum-likelihood $(\hat R, \hat h)$, $\hat h \in [0, h_{\max}]$, of centred draws; returns `(R, h, mean NLL)`.

    L-BFGS-B on $(\log R, h)$ with the analytic gradient of `_tukey_h_mean_nll_and_grad`, from three
    starts ($h$ = 0, 0.5 and 2, capped at `max_h`); the best optimum is kept.
    """
    q25, q75 = np.quantile(u, [0.25, 0.75])
    scale = (q75 - q25) / _IQR_TO_SIGMA
    if not scale > 0.0:
        scale = float(np.mean(np.abs(u))) / _MEAN_ABS_TO_SIGMA
    log_s = float(np.log(scale))
    span = float(np.log(_TUKEY_H_RADIUS_SEARCH_FACTOR))
    bounds = [(log_s - span, log_s + span), (0.0, max_h)]
    best: Any = None
    for shift, h0 in ((0.0, 0.0), (-0.25, 0.5), (-0.5, 2.0)):
        res = optimize.minimize(
            _tukey_h_mean_nll_and_grad,
            x0=np.array([log_s + shift, min(h0, max_h)]),
            args=(u,),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
        )
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Tukey-h likelihood is not finite at any start")
    return float(np.exp(best.x[0])), float(best.x[1]), float(best.fun)


def calibrate_jax_tukey_h_transform(
    pilot_samples: Any,
    d: int | None = None,
    mask: Any | None = None,
    min_radius: float = 0.1,
    max_radius: float = 100.0,
    level: float = TUKEY_H_TEST_LEVEL,
    max_h: float = TUKEY_H_MAX_H,
) -> TukeyHCalibration:
    r"""Calibrates a `JaxTukeyHTransform` by Tukey-h maximum likelihood with an $h = 0$ guard.

    Per coordinate $i$, from the finite pilot draws (pooled over chains):

    1. $\mu_i = \mathop{\mathrm{median}}(x_{\cdot, i})$.
    2. $(\hat R_i, \hat h_i)$ maximise the likelihood of the Tukey-h (Lambert W $\times$ Gaussian,
       Goerg 2015) model $x = \mu_i + R z e^{h z^2 / 2}$, $z \sim N(0, 1)$, over $R > 0$ and
       $0 \le h \le h_{\max}$; $\hat R_{0,i}$ is the closed-form maximiser at $h = 0$.
    3. **Guard.** $\Lambda_i = 2(\hat\ell_1 - \hat\ell_0)$ tests $h = 0$ against $h > 0$. Under the null
       $h = 0$ lies on the boundary of the parameter space, so $\Lambda_i$ is asymptotically
       $\tfrac12\chi^2_0 + \tfrac12\chi^2_1$ (Self & Liang 1987) and $p_i = \tfrac12 P(\chi^2_1 > \Lambda_i)$.
       A coordinate gets $(\hat R_i, \hat h_i)$ only if it is masked and $p_i <$ `level`; otherwise
       it gets $(\hat R_{0,i}, 0)$, the affine map. Light coordinates thus keep $h = 0$, which the
       pullback needs for its Gaussian-tail (geometric ergodicity) window, unless the data reject it.
    4. **Clips.** $h_{\max}$ bounds $\hat h_i$ in the fitted model's own units ($z \sim N(0, 1)$).
       $R_i$ is then clipped to $[R_{\min}, R_{\max}]$ and $h_i$ multiplied by $(R_i^{\mathrm{clip}} / R_i)^2$.
       That rescales $z$ only, since the map's shape depends on $h / R^2$ alone (`hMap_rescale`), so
       the clip never changes which density is sampled or the tail shape.

    The test assumes independent draws; Phase 1 warm-up draws are autocorrelated, which can make it
    reject more often than `level`. Coordinates with fewer than `TUKEY_H_MIN_DRAWS` finite draws or
    no spread are not tested ($h_i = 0$, $R_i = R_{\min}$ if there is no spread).

    Parameters:
        pilot_samples: Primal draws of shape `(n_samples, d)` or `(chains, draws, d)`.
        d: Target dimension; inferred from `pilot_samples` if `None`.
        mask: Optional boolean mask of shape `(d,)`; unmasked coordinates are never warped.
        min_radius: Lower clip $R_{\min} > 0$ on $R_i$.
        max_radius: Upper clip $R_{\max} \ge R_{\min}$ on $R_i$.
        level: Per-coordinate level of the test of $h = 0$, in $(0, 1)$ (default `TUKEY_H_TEST_LEVEL`).
        max_h: Upper bound $h_{\max} > 0$ on the fitted $\hat h_i$ (default `TUKEY_H_MAX_H`).

    Returns:
        `TukeyHCalibration` with the transform, the fitted values and the per-coordinate test decisions.
    """
    if min_radius <= 0.0:
        raise ValueError(f"min_radius must be strictly positive, got {min_radius}")
    if min_radius > max_radius:
        raise ValueError(f"min_radius ({min_radius}) must be <= max_radius ({max_radius})")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if not (np.isfinite(max_h) and max_h > 0.0):
        raise ValueError(f"max_h must be finite and positive, got {max_h}")

    arr = np.asarray(pilot_samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, d) if d is not None and d > 1 else arr[:, np.newaxis]
    elif arr.ndim >= 3:
        arr = arr.reshape(-1, arr.shape[-1])
    d_actual = int(arr.shape[-1])
    if d is not None and int(d) != d_actual:
        raise ValueError(f"pilot_samples last dimension {d_actual} does not match d={d}")
    mask_arr = np.ones(d_actual, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_arr.shape != (d_actual,):
        raise ValueError(f"mask must have shape ({d_actual},), got {mask_arr.shape}")

    location = np.zeros(d_actual)
    radius_gaussian = np.zeros(d_actual)
    radius_mle = np.full(d_actual, np.nan)
    h_mle = np.full(d_actual, np.nan)
    lr_statistic = np.zeros(d_actual)
    for j in range(d_actual):
        x = arr[:, j][np.isfinite(arr[:, j])]
        if x.size == 0:
            continue
        location[j] = float(np.median(x))
        u = x - location[j]
        peak = float(np.max(np.abs(u)))
        if peak > 0.0:
            radius_gaussian[j] = peak * float(np.sqrt(np.mean((u / peak) ** 2)))
        if x.size < TUKEY_H_MIN_DRAWS or not radius_gaussian[j] > 0.0:
            continue
        nll0 = float(np.log(radius_gaussian[j])) + 0.5 + _LOG_SQRT_2PI
        radius_mle[j], h_mle[j], nll1 = _fit_tukey_h_mle(u, max_h)
        lr_statistic[j] = max(2.0 * x.size * (nll0 - nll1), 0.0)

    p_value = np.where(lr_statistic > 0.0, 0.5 * stats.chi2.sf(lr_statistic, df=1), 1.0)
    heavy = mask_arr & (p_value < level)
    radius = np.where(heavy, radius_mle, radius_gaussian)
    h = np.where(heavy, h_mle, 0.0)
    radii = np.clip(radius, min_radius, max_radius)
    h = np.where(heavy, h * (radii / np.where(radius > 0.0, radius, 1.0)) ** 2, 0.0)

    transform = JaxTukeyHTransform(radii=radii, location=location, h=h, mask=mask_arr, d=d_actual)
    return TukeyHCalibration(
        transform=transform,
        location=location,
        radii=radii,
        h=h,
        heavy=heavy,
        lr_statistic=lr_statistic,
        p_value=p_value,
        radius_mle=radius_mle,
        h_mle=h_mle,
        radius_gaussian=radius_gaussian,
        level=float(level),
    )


def sample_blackjax_unified_nuts(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray] | TargetDistribution,
    d: int | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    seed: int = 42,
    x0: np.ndarray | jnp.ndarray | Sequence[float] | None = None,
    *,
    random_seed: int | None = None,
    rng_key: Any | None = None,
    init_position: np.ndarray | jnp.ndarray | Sequence[float] | None = None,
    target_accept: float = 0.80,
    adapt_tail_scale: bool = True,
    initial_radius: float | Sequence[float] | np.ndarray | jnp.ndarray = 3.0,
    initial_location: float | Sequence[float] | np.ndarray | jnp.ndarray = 0.0,
    min_radius: float = 0.1,
    max_radius: float = 100.0,
    scale_multiplier: float = 1.5,
    mask: Sequence[bool] | np.ndarray | jnp.ndarray | None = None,
    max_num_doublings: int = 10,
    divergence_threshold: float = 1000.0,
    dim: int | None = None,
    step_size: float = 0.1,
    transform_kind: TransformKind = "arcsinh",
) -> JaxUnifiedNUTSResult:
    r"""Runs 3-phase adaptive NUTS on the arcsinh-warped target with `JaxArcsinhTransform` and BlackJAX.

    Every phase samples the exact transformed density `transform.wrap_logdensity(logdensity_fn)`.

    Three-Phase Schedule (same plan as `UnifiedRobustNUTS.adapt`; differences in the module docstring):
        - **Phase 1** ($W_1 = \max(10, \lfloor 0.15 \cdot \text{tune} \rfloor)$ when `adapt_tail_scale=True`
          and `tune >= 20`): Runs BlackJAX `window_adaptation` on initial `JaxArcsinhTransform`, pulls
          Phase 1 draws back to primal $x$-space, and calibrates `transform = calibrate_jax_arcsinh_transform(...)`.
        - **Phase 2 & 3** ($W_2 = \max(10, \text{tune} - W_1)$): Maps Phase 1 terminal primal positions
          into the calibrated $z$-space (`z_start = transform.forward(x_p1_final)`), then runs BlackJAX
          `window_adaptation` per chain for $W_2$ steps to adapt that chain's diagonal inverse mass
          matrix $M^{-1}$ in calibrated $z$-space and its step size $h$.
        - **Post-Warmup Sampling** (`draws` steps per chain via `jax.jit` + `jax.vmap` + `jax.lax.scan`):
          Each chain samples with its own adapted parameters in calibrated $z$-space; draws are mapped
          back to primal $x$-space via `transform.inverse`, and leapfrog gradient evaluations are
          counted across all phases.

    Parameters:
        logdensity_fn: JAX-traceable callable `logdensity_fn(x) -> scalar` in primal $x$-space,
            or a `TargetDistribution` instance.
        d: Target dimension $d \ge 1$. Optional if inferred from `logdensity_fn`, `x0`, `init_position`,
            `dim`, or `initial_radius`.
        draws: Number of post-warmup MCMC samples per chain.
        tune: Total number of warmup adaptation steps per chain.
        chains: Number of independent Markov chains.
        seed: Integer PRNG seed (default: `42`).
        x0: Optional initial primal positions of shape `(d,)` or `(chains, d)`.
        random_seed: Optional alias for `seed`.
        rng_key: Optional explicit JAX PRNGKey (overrides `seed` if provided).
        init_position: Optional alias for `x0`.
        target_accept: Target acceptance rate $\delta \in (0, 1)$ for dual averaging (default: `0.80`).
        adapt_tail_scale: Whether to run Phase 1 coordinate location & scale calibration (default: `True`).
        initial_radius: Initial coordinate scale $R_0 > 0$ before Phase 1 calibration (default: `3.0`).
        initial_location: Initial coordinate location $\mu_0$ before Phase 1 calibration (default: `0.0`).
        min_radius: Minimum allowable calibrated coordinate scale $R_{\min} > 0$ (default: `0.1`).
        max_radius: Maximum allowable calibrated coordinate scale $R_{\max} \ge R_{\min}$ (default: `100.0`).
        scale_multiplier: Multiplier $c > 0$ applied to normal-consistent IQR scale in Phase 1 (default: `1.5`).
        mask: Optional boolean mask of shape `(d,)` indicating warped coordinates.
        max_num_doublings: Maximum NUTS binary tree depth (default: `10`).
        divergence_threshold: Maximum Hamiltonian energy error $\Delta H$ before divergence (default: `1000.0`).
        dim: Optional alias for `d`.
        step_size: Fallback step size when `tune == 0` (default: `0.1`).
        transform_kind: `"arcsinh"` (default), `"affine"` (same calibration, no tail warping) or
            `"identity"` (stock window-adapted NUTS in $x$-space, no Phase 1); see the module
            docstring, item 5.

    Returns:
        `JaxUnifiedNUTSResult` containing primal `samples`, `z_samples`, calibrated `transform`,
        per-chain `step_size` and `inverse_mass_matrix`, exact per-phase and total `grad_evals`,
        `divergences`, `acceptance_rate`, and ArviZ `idata`.
    """
    if draws < 1:
        raise ValueError(f"draws must be >= 1, got {draws}")
    if tune < 0:
        raise ValueError(f"tune must be >= 0, got {tune}")
    if chains < 1:
        raise ValueError(f"chains must be >= 1, got {chains}")
    if transform_kind not in TRANSFORM_KINDS:
        raise ValueError(f"transform_kind must be one of {TRANSFORM_KINDS}, got {transform_kind!r}")

    # Resolve TargetDistribution vs callable logdensity_fn
    if isinstance(logdensity_fn, TargetDistribution):
        from mcmc_robust.bench.sota_adapters import SOTATargetAdapter

        if d is None:
            d = logdensity_fn.d
        raw_logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray] = SOTATargetAdapter(logdensity_fn).jax_logdensity_fn()
    else:
        raw_logdensity_fn = logdensity_fn

    # Resolve dimensionality d
    eff_init = x0 if x0 is not None else init_position
    if d is None and dim is not None:
        d = int(dim)
    if d is None and eff_init is not None:
        d = int(np.asarray(eff_init).shape[-1])
    if d is None and hasattr(initial_radius, "__len__"):
        d = len(initial_radius)  # type: ignore[arg-type]
    if d is None:
        raise ValueError("Target dimension 'd' must be specified when it cannot be inferred from inputs.")
    d_int = int(d)

    # Resolve PRNG key
    eff_seed = int(random_seed) if random_seed is not None else int(seed)
    master_key = rng_key if rng_key is not None else jax.random.PRNGKey(eff_seed)
    key_init, key_p1, key_p2, key_sample = jax.random.split(master_key, 4)

    # Resolve initial positions in primal x-space: shape (chains, d)
    if eff_init is not None:
        x0_arr = jnp.asarray(eff_init, dtype=jnp.float64)
        if x0_arr.ndim == 1:
            if x0_arr.shape[0] != d_int:
                raise ValueError(f"Initial position shape {x0_arr.shape} does not match d={d_int}")
            jitter = 0.05 * jax.random.normal(key_init, shape=(chains, d_int), dtype=jnp.float64)
            init_x = jnp.tile(x0_arr[jnp.newaxis, :], (chains, 1)) + jitter
        elif x0_arr.ndim == 2 and x0_arr.shape == (chains, d_int):
            init_x = x0_arr
        else:
            raise ValueError(f"Expected x0 shape ({d_int},) or ({chains}, {d_int}), got {x0_arr.shape}")
    else:
        init_x = 0.5 * jax.random.normal(key_init, shape=(chains, d_int), dtype=jnp.float64)

    # Initial transform (matches UnifiedRobustNUTS._ensure_transform)
    transform_init: JaxCoordinateTransform
    if transform_kind == "arcsinh":
        transform_init = JaxArcsinhTransform(radii=initial_radius, location=initial_location, mask=mask, d=d_int)
    elif transform_kind == "affine":
        transform_init = JaxAffineTransform(radii=initial_radius, location=initial_location, mask=mask, d=d_int)
    else:
        transform_init = JaxAffineTransform(radii=1.0, location=0.0, d=d_int)
    calibrate = calibrate_jax_affine_transform if transform_kind == "affine" else calibrate_jax_arcsinh_transform

    # Phase 1 horizon W1 = max(10, int(0.15 * tune)) matches UnifiedRobustNUTS.adapt (nuts.py line 666)
    run_phase1 = bool(adapt_tail_scale and tune >= 20 and transform_kind != "identity")
    phase1_grad_evals = 0
    phase2_grad_evals = 0

    if run_phase1:
        w1_steps = max(10, int(0.15 * tune))
        w2_steps = max(10, tune - w1_steps)

        z0_batch = jax.vmap(transform_init.forward)(init_x)
        logdensity_z1 = transform_init.wrap_logdensity(raw_logdensity_fn)

        warmup_p1 = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_z1,
            target_acceptance_rate=target_accept,
            max_num_doublings=max_num_doublings,
        )
        p1_keys = jax.random.split(key_p1, chains)
        (state_p1, _), info_p1 = jax.vmap(lambda k, z: warmup_p1.run(k, z, num_steps=w1_steps))(p1_keys, z0_batch)

        # Exact Phase 1 gradient evaluations: 1 init eval per chain + sum of leapfrog steps
        phase1_grad_evals = int(chains + int(jnp.sum(info_p1.info.num_integration_steps)))

        # Pull second half of Phase 1 draws back to primal x-space and calibrate location & radii
        half_idx = max(1, w1_steps // 2)
        z_p1_tail = info_p1.state.position[:, half_idx:, :]
        x_p1_tail = jax.vmap(jax.vmap(transform_init.inverse))(z_p1_tail)
        transform: JaxCoordinateTransform = calibrate(
            pilot_samples=x_p1_tail,
            d=d_int,
            mask=mask,
            min_radius=min_radius,
            max_radius=max_radius,
            scale_multiplier=scale_multiplier,
        )
        x_p1_final = jax.vmap(transform_init.inverse)(state_p1.position)
    else:
        w2_steps = tune
        transform = transform_init
        x_p1_final = init_x

    # Map Phase 1 terminal positions into calibrated z-space (Phase 2 & 3 entry)
    z_start_batch = jax.vmap(transform.forward)(x_p1_final)
    logdensity_z2 = transform.wrap_logdensity(raw_logdensity_fn)

    if w2_steps > 0:
        warmup_p2 = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_z2,
            target_acceptance_rate=target_accept,
            max_num_doublings=max_num_doublings,
        )
        p2_keys = jax.random.split(key_p2, chains)
        (state_p2, params_p2), info_p2 = jax.vmap(lambda k, z: warmup_p2.run(k, z, num_steps=w2_steps))(
            p2_keys, z_start_batch
        )

        # Exact Phase 2/3 gradient evaluations: 1 init eval per chain + sum of leapfrog steps
        phase2_grad_evals = int(chains + int(jnp.sum(info_p2.info.num_integration_steps)))

        # Each chain keeps its own adapted step size and metric (Stan's convention), so one
        # poorly adapted chain cannot slow down the others.
        chain_step_sizes = jnp.asarray(params_p2["step_size"], dtype=jnp.float64)
        chain_inv_masses = jnp.asarray(params_p2["inverse_mass_matrix"], dtype=jnp.float64)
        sampling_init_states = state_p2
        init_sampling_evals = 0
    else:
        chain_step_sizes = jnp.full((chains,), float(step_size), dtype=jnp.float64)
        chain_inv_masses = jnp.ones((chains, d_int), dtype=jnp.float64)
        sampling_init_states = jax.vmap(lambda z: blackjax.nuts.init(z, logdensity_z2))(z_start_batch)
        init_sampling_evals = int(chains)

    # Post-Warmup Production Sampling via jax.jit + jax.vmap + jax.lax.scan
    def _scan_single_chain(
        init_state: Any,
        chain_ss: jnp.ndarray,
        chain_inv_m: jnp.ndarray,
        chain_rng_key: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        nuts_kernel = blackjax.nuts(
            logdensity_z2,
            step_size=chain_ss,
            inverse_mass_matrix=chain_inv_m,
            max_num_doublings=max_num_doublings,
            divergence_threshold=divergence_threshold,
        ).step
        step_keys = jax.random.split(chain_rng_key, draws)

        def _one_step(
            state: Any,
            subkey: Any,
        ) -> tuple[Any, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            next_state, info = nuts_kernel(subkey, state)
            return next_state, (
                next_state.position,
                info.is_divergent,
                info.num_integration_steps,
                info.acceptance_rate,
                info.energy,
            )

        _, trajectory = jax.lax.scan(_one_step, init_state, step_keys)
        return trajectory

    sample_keys = jax.random.split(key_sample, chains)
    z_draws, div_draws, n_steps_draws, acc_draws, energy_draws = jax.jit(jax.vmap(_scan_single_chain))(
        sampling_init_states, chain_step_sizes, chain_inv_masses, sample_keys
    )

    x_draws = jax.vmap(jax.vmap(transform.inverse))(z_draws)

    samples_np = np.asarray(x_draws, dtype=np.float64)
    z_samples_np = np.asarray(z_draws, dtype=np.float64)
    step_size_np = np.asarray(chain_step_sizes, dtype=np.float64)
    inv_mass_np = np.asarray(chain_inv_masses, dtype=np.float64)

    sampling_grad_evals = int(init_sampling_evals + int(jnp.sum(n_steps_draws)))
    total_grad_evals = int(phase1_grad_evals + phase2_grad_evals + sampling_grad_evals)
    total_divergences = int(jnp.sum(div_draws))
    mean_acceptance = float(jnp.mean(acc_draws))

    sample_stats_dict: dict[str, np.ndarray] = {
        "diverging": np.asarray(div_draws, dtype=bool),
        "n_steps": np.asarray(n_steps_draws, dtype=np.int64),
        "acceptance_rate": np.asarray(acc_draws, dtype=np.float64),
        "energy": np.asarray(energy_draws, dtype=np.float64),
        "step_size": np.repeat(step_size_np[:, np.newaxis], draws, axis=1),
    }
    idata = arviz_from_samples(samples_np, sample_stats=sample_stats_dict)

    return JaxUnifiedNUTSResult(
        samples=samples_np,
        z_samples=z_samples_np,
        transform=transform,
        step_size=step_size_np,
        inverse_mass_matrix=inv_mass_np,
        grad_evals=total_grad_evals,
        divergences=total_divergences,
        acceptance_rate=mean_acceptance,
        phase1_grad_evals=phase1_grad_evals,
        phase2_grad_evals=phase2_grad_evals,
        sampling_grad_evals=sampling_grad_evals,
        idata=idata,
    )
