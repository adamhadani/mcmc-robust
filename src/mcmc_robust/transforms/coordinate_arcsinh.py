"""Coordinate-wise arcsinh diffeomorphism for anisotropic heavy tails (Feature F16)."""

from collections.abc import Sequence

import numpy as np

from mcmc_robust.transforms.base import SpaceTransformation


class CoordinateWiseArcsinhTransform(SpaceTransformation):
    r"""Coordinate-wise C^\infty diffeomorphism for heterogeneous heavy-tailed targets.

    For heavy-tailed coordinates (mask[i] == True):
        Forward:
            u_i = (x_i - \mu_i) / R_i
            z_i = arcsinh(u_i)
            log |det J| = -sum_i ln(R_i * sqrt(1 + u_i^2))
        Inverse:
            x_i = \mu_i + R_i * sinh(z_i)
            log |det J^{-1}| = sum_i (ln R_i + ln cosh(z_i))
        Pullback gradient:
            nabla_{z_i} log pi_z(z) = R_i * cosh(z_i) * nabla_{x_i} log pi_x(x) + tanh(z_i)
                                    = sqrt(R_i^2 + (x_i - \mu_i)^2) * nabla_{x_i} log pi_x(x)
                                      + (x_i - \mu_i) / sqrt(R_i^2 + (x_i - \mu_i)^2)

    For light-tailed coordinates (mask[i] == False):
        z_i = x_i,  x_i = z_i
        log |det J| = 0,  log |det J^{-1}| = 0
        nabla_{z_i} log pi_z(z) = nabla_{x_i} log pi_x(x)

    Satisfies:
    - Exact translation equivariance when location \mu_i is calibrated from coordinate medians
    - Round-trip invertibility || inverse(forward(x)[0])[0] - x ||_inf < 1e-12
    - Jacobian log-determinant cancellation | log |det J| + log |det J^{-1}| | < 1e-12
    - Pullback gradient consistency vs finite differences < 1e-7
    - Strictly O(d) time and memory complexity.
    """

    def __init__(
        self,
        d: int | None = None,
        radii: float | Sequence[float] | np.ndarray = 1.0,
        mask: Sequence[bool] | np.ndarray | None = None,
        location: float | Sequence[float] | np.ndarray = 0.0,
    ) -> None:
        r"""Initializes CoordinateWiseArcsinhTransform.

        Args:
            d: Dimension of the space R^d. If None, inferred from radii or location length.
            radii: Confinement radius R_i > 0 per coordinate (scalar or vector of shape (d,)).
            mask: Optional boolean mask of shape (d,) indicating which coordinates
                are transformed (True) vs left untransformed (False). Defaults to all True.
            location: Centering location \mu_i per coordinate (scalar or vector of shape (d,)).
        """
        if d is None:
            if hasattr(radii, "__len__"):
                d = len(radii)  # type: ignore[arg-type]
            elif hasattr(location, "__len__"):
                d = len(location)  # type: ignore[arg-type]
            else:
                raise ValueError("Must provide dimension d when radii is a scalar float")

        d_int = int(d)
        if d_int < 1:
            raise ValueError(f"Dimension d must be positive, got {d_int}")

        super().__init__(input_dim=d_int, output_dim=d_int)

        if hasattr(radii, "__len__"):
            r_arr = np.asarray(radii, dtype=np.float64)
            if r_arr.shape != (d_int,):
                raise ValueError(f"radii shape {r_arr.shape} must match (d,) with d={d_int}")
        else:
            r_val = float(radii)
            r_arr = np.full(d_int, r_val, dtype=np.float64)

        if np.any(r_arr <= 0.0) or not np.all(np.isfinite(r_arr)):
            raise ValueError("All coordinate radii must be strictly positive and finite")
        self._radii: np.ndarray = r_arr

        if hasattr(location, "__len__"):
            loc_arr = np.asarray(location, dtype=np.float64)
            if loc_arr.shape != (d_int,):
                raise ValueError(f"location shape {loc_arr.shape} must match (d,) with d={d_int}")
        else:
            loc_val = float(location)
            loc_arr = np.full(d_int, loc_val, dtype=np.float64)

        if not np.all(np.isfinite(loc_arr)):
            raise ValueError("All coordinate locations must be finite")
        self._location: np.ndarray = loc_arr

        if mask is None:
            self._mask: np.ndarray = np.ones(d_int, dtype=bool)
        else:
            m_arr = np.asarray(mask, dtype=bool)
            if m_arr.shape != (d_int,):
                raise ValueError(f"mask shape {m_arr.shape} must match (d,) with d={d_int}")
            self._mask = m_arr

    @property
    def radii(self) -> np.ndarray:
        """Coordinate scale vector R of shape (d,)."""
        return self._radii

    @property
    def location(self) -> np.ndarray:
        """Coordinate location vector mu of shape (d,)."""
        return self._location

    @property
    def mask(self) -> np.ndarray:
        """Boolean coordinate transformation mask of shape (d,)."""
        return self._mask

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Applies forward diffeomorphism z = phi(x).

        Args:
            x: Coordinate vector in primal space, shape (d,).

        Returns:
            Tuple of (z, log_det_jacobian_dz_dx).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != self.d:
            raise ValueError(f"Expected input shape ({self.d},), got {x_arr.shape}")

        z = x_arr.copy()
        m = self._mask
        if np.any(m):
            xm = x_arr[m] - self._location[m]
            rm = self._radii[m]
            scaled = xm / rm
            z[m] = np.arcsinh(scaled)
            # Numerically stabilized log-Jacobian determinant:
            # For |scaled| > 1e8, 0.5 * ln(1 + scaled^2) = ln|scaled| + O(scaled^{-2}),
            # where scaled^{-2} < 1e-16 <= eps_mach, matching float64 precision bit-for-bit
            # while avoiding intermediate overflow to inf for |scaled| > 1.34e154.
            # np.maximum and np.minimum guard both branches against eager-evaluation warnings.
            abs_scaled = np.abs(scaled)
            log_term = np.where(
                abs_scaled > 1e8,
                np.log(np.maximum(abs_scaled, 1.0)),
                0.5 * np.log1p(np.minimum(abs_scaled, 1e8) ** 2),
            )
            log_det = -float(np.sum(np.log(rm) + log_term))
        else:
            log_det = 0.0

        return z, log_det

    def inverse(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Applies inverse diffeomorphism x = phi^{-1}(z).

        Args:
            y: Transformed coordinate vector z, shape (d,).

        Returns:
            Tuple of (x, log_det_jacobian_dx_dz).
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1 or y_arr.shape[0] != self.d:
            raise ValueError(f"Expected input shape ({self.d},), got {y_arr.shape}")

        x = y_arr.copy()
        m = self._mask
        if np.any(m):
            ym = y_arr[m]
            rm = self._radii[m]
            locm = self._location[m]
            with np.errstate(over="ignore"):
                x[m] = locm + rm * np.sinh(ym)
            # Robust log(cosh(z)) = |z| - ln 2 + log1p(exp(-2|z|)) avoiding overflow
            abs_ym = np.abs(ym)
            log_cosh = abs_ym - np.log(2.0) + np.log1p(np.exp(-2.0 * abs_ym))
            log_det = float(np.sum(np.log(rm) + log_cosh))
        else:
            log_det = 0.0

        return x, log_det

    def forward_grad(self, x: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        """Computes pullback gradient nabla_z log pi_z(z) at z = phi(x).

        Args:
            x: Primal coordinate vector, shape (d,).
            grad_x: Primal target gradient nabla_x log pi_x(x), shape (d,).

        Returns:
            Transformed gradient vector nabla_z log pi_z(z), shape (d,).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        gx = np.asarray(grad_x, dtype=np.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != self.d:
            raise ValueError(f"Expected input shape ({self.d},), got {x_arr.shape}")
        if gx.ndim != 1 or gx.shape[0] != self.d:
            raise ValueError(f"Expected grad_x shape ({self.d},), got {gx.shape}")

        gz = gx.copy()
        m = self._mask
        if np.any(m):
            with np.errstate(over="ignore", invalid="ignore"):
                xm = x_arr[m] - self._location[m]
                rm = self._radii[m]
                # Numerically stabilized scale = sqrt(rm^2 + xm^2) avoiding intermediate squaring overflow:
                # np.hypot(rm, xm) prevents overflow to inf when xm^2 > 1.8e308 (for |xm| > 1.34e154).
                scale = np.hypot(rm, xm)
                gz[m] = scale * gx[m] + (xm / scale)

        return gz

    @staticmethod
    def compute_phase1_calibration(
        samples: np.ndarray,
        min_radius: float = 0.1,
        max_radius: float = 1e4,
        scale_multiplier: float = 1.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Computes Phase 1 coordinate location medians $\mu_i$ and clipped scales $R_i$.

        Shared calibration kernel used by `UnifiedRobustNUTS.adapt`, `UnifiedRobustHMC.adapt`,
        `UnifiedRobustSampler.adapt`, and `calibrate_jax_arcsinh_transform` (`jax_unified_nuts.py`),
        implementing the formally verified calibration map (`formal/McmcRobustFormal.lean` §8):
            $$\mu_i = \operatorname{median}(x_{\cdot, i}), \qquad
              R_i = \min\!\left(R_{\max},\, \max\!\left(R_{\min},\, c \cdot \hat{\sigma}_i\right)\right)$$
        where $\hat{\sigma}_i = \operatorname{IQR}(x_{\cdot, i} - \mu_i) / 1.3489795$ when
        $\hat{\sigma}_{\text{IQR}, i} > R_{\min}$, falling back to
        $\max(\operatorname{MAD}(x_{\cdot, i} - \mu_i) / 0.67448975,\, q_{0.90}(|x_{\cdot, i} - \mu_i|))$.

        Args:
            samples: Exploratory primal draws of shape `(n_samples, d)` or `(chains, draws, d)`.
            min_radius: Minimum allowable coordinate scale $R_{\min} > 0$.
            max_radius: Maximum allowable coordinate scale $R_{\max} \ge R_{\min}$.
            scale_multiplier: Multiplier $c \ge 0$ applied to base scale estimate (default: `1.5`).

        Returns:
            Tuple `(location, radii)` of 1D float64 arrays of shape `(d,)`.
        """
        if min_radius <= 0.0:
            raise ValueError(f"min_radius must be strictly positive, got {min_radius}")
        if min_radius > max_radius:
            raise ValueError(f"min_radius ({min_radius}) must be <= max_radius ({max_radius})")
        if scale_multiplier < 0.0:
            raise ValueError(f"scale_multiplier must be non-negative, got {scale_multiplier}")

        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim >= 3:
            arr = arr.reshape(-1, arr.shape[-1])

        med = np.median(arr, axis=0)
        centered = arr - med
        q75, q25 = np.quantile(centered, [0.75, 0.25], axis=0)
        iqr_scale = (q75 - q25) / 1.3489795
        mad_scale = np.median(np.abs(centered), axis=0) / 0.67448975
        q90_abs = np.quantile(np.abs(centered), 0.90, axis=0)
        base_scale = np.where(iqr_scale > min_radius, iqr_scale, np.maximum(mad_scale, q90_abs))
        base_scale = np.where(np.isfinite(base_scale), base_scale, 1.0)
        med = np.where(np.isfinite(med), med, 0.0)
        radii = np.clip(base_scale * scale_multiplier, min_radius, max_radius)
        return np.asarray(med, dtype=np.float64), np.asarray(radii, dtype=np.float64)

    @classmethod
    def calibrate_from_samples(
        cls,
        samples: np.ndarray,
        alpha_threshold: float = 3.5,
        quantile_ratio_threshold: float = 3.0,
        min_radius: float = 1e-4,
    ) -> "CoordinateWiseArcsinhTransform":
        """Factory calibrating location, radii, and heavy/light mask from exploratory samples.

        Uses translation-equivariant coordinate medians for location mu_i and normal-consistent
        signed interquartile range IQR(x_i - mu_i) / 1.3489795 (with MAD fallback) for scale R_i.

        Args:
            samples: 2D array of exploratory samples of shape (n_samples, d).
            alpha_threshold: Tail index cutoff for heavy tail classification (default: 3.5).
            quantile_ratio_threshold: q99/q75 ratio threshold for heavy tail classification (default: 3.0).
            min_radius: Minimum allowable value for R_i (default: 1e-4).

        Returns:
            Configured CoordinateWiseArcsinhTransform instance.
        """
        s = np.asarray(samples, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError(f"Expected 2D array of samples, got shape {s.shape}")
        n, d = s.shape
        if n < 20:
            raise ValueError(f"Need at least 20 samples to calibrate, got {n}")

        med = np.median(s, axis=0)
        centered = s - med
        centered_abs = np.abs(centered)

        # Normal-consistent scale estimate via signed IQR: for N(0, sigma^2), IQR = 1.3489795 * sigma
        q75_signed = np.quantile(centered, 0.75, axis=0)
        q25_signed = np.quantile(centered, 0.25, axis=0)
        iqr_scale = (q75_signed - q25_signed) / 1.3489795
        mad_scale = np.median(centered_abs, axis=0) / 0.67448975
        radii = np.where(iqr_scale > min_radius, iqr_scale, np.maximum(min_radius, mad_scale))

        # Quantile ratio q99(|x - med|) / q75(|x - med|)
        q75_abs = np.quantile(centered_abs, 0.75, axis=0)
        q99_abs = np.quantile(centered_abs, 0.99, axis=0)
        q_ratio = q99_abs / np.maximum(1e-6, q75_abs)

        # Hill estimator on top 10% of |x - med|
        k = max(5, int(0.10 * n))
        alpha_hat = np.empty(d, dtype=np.float64)
        for j in range(d):
            sorted_col = np.sort(centered_abs[:, j])
            tail = sorted_col[-k:]
            thresh = sorted_col[-k - 1] if k < n else sorted_col[0]
            if thresh <= 0.0:
                thresh = 1e-6
            gamma = float(np.mean(np.log(np.maximum(1e-12, tail / thresh))))
            alpha_hat[j] = 1.0 / max(1e-4, gamma)

        mask = (q_ratio > quantile_ratio_threshold) | (alpha_hat <= alpha_threshold)
        return cls(d=d, radii=radii, mask=mask, location=med)
