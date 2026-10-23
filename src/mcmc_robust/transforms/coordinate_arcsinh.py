"""Coordinate-wise arcsinh diffeomorphism for anisotropic heavy tails (Feature F16)."""

from collections.abc import Sequence

import numpy as np

from mcmc_robust.transforms.base import SpaceTransformation


class CoordinateWiseArcsinhTransform(SpaceTransformation):
    r"""Coordinate-wise C^\infty diffeomorphism for heterogeneous heavy-tailed targets.

    For heavy-tailed coordinates (mask[i] == True):
        Forward:
            z_i = arcsinh(x_i / R_i)
            log |det J| = -sum_i ln(R_i * sqrt(1 + (x_i / R_i)^2))
        Inverse:
            x_i = R_i * sinh(z_i)
            log |det J^{-1}| = sum_i (ln R_i + ln cosh(z_i))
        Pullback gradient:
            nabla_{z_i} log pi_z(z) = R_i * cosh(z_i) * nabla_{x_i} log pi_x(x) + tanh(z_i)
                                    = sqrt(R_i^2 + x_i^2) * nabla_{x_i} log pi_x(x) + x_i / sqrt(R_i^2 + x_i^2)

    For light-tailed coordinates (mask[i] == False):
        z_i = x_i,  x_i = z_i
        log |det J| = 0,  log |det J^{-1}| = 0
        nabla_{z_i} log pi_z(z) = nabla_{x_i} log pi_x(x)

    Satisfies:
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
    ) -> None:
        """Initializes CoordinateWiseArcsinhTransform.

        Args:
            d: Dimension of the space R^d. If None, inferred from radii length.
            radii: Confinement radius R_i > 0 per coordinate (scalar or vector of shape (d,)).
            mask: Optional boolean mask of shape (d,) indicating which coordinates
                are transformed (True) vs left untransformed (False). Defaults to all True.
        """
        if d is None:
            if hasattr(radii, "__len__"):
                d = len(radii)  # type: ignore[arg-type]
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
            xm = x_arr[m]
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
            with np.errstate(over="ignore"):
                x[m] = rm * np.sinh(ym)
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
            xm = x_arr[m]
            rm = self._radii[m]
            # Numerically stabilized scale = sqrt(rm^2 + xm^2) avoiding intermediate squaring overflow:
            # np.hypot(rm, xm) prevents overflow to inf when xm^2 > 1.8e308 (for |xm| > 1.34e154).
            scale = np.hypot(rm, xm)
            gz[m] = scale * gx[m] + (xm / scale)

        return gz

    @classmethod
    def calibrate_from_samples(
        cls,
        samples: np.ndarray,
        alpha_threshold: float = 3.5,
        quantile_ratio_threshold: float = 3.0,
        min_radius: float = 1e-4,
    ) -> "CoordinateWiseArcsinhTransform":
        """Factory calibrating radii and heavy/light mask from exploratory samples.

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
        centered_abs = np.abs(s - med)

        # Scale estimate via 75th percentile / semi-IQR
        q75 = np.quantile(centered_abs, 0.75, axis=0)
        q25 = np.quantile(centered_abs, 0.25, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        radii = np.where(iqr_scale > min_radius, iqr_scale, np.maximum(min_radius, q75))

        # Quantile ratio q99 / q75
        q99 = np.quantile(centered_abs, 0.99, axis=0)
        q_ratio = q99 / np.maximum(1e-6, q75)

        # Hill estimator on top 10%
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
        return cls(d=d, radii=radii, mask=mask)
