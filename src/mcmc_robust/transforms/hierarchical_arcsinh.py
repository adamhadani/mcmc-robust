"""Triangular hierarchical scale-coupled arcsinh diffeomorphism for deep funnels."""

from collections.abc import Sequence

import numpy as np

from mcmc_robust.transforms.base import SpaceTransformation


class HierarchicalArcsinhTransform(SpaceTransformation):
    r"""Lower-triangular hierarchical scale-coupled arcsinh diffeomorphism.

    Unwinds cross-coordinate scale coupling in hierarchical funnel geometries where
    child coordinates x_i (i != p) have conditional scale modulated by a parent
    log-scale coordinate x_p:
        u_p(x_p) = (x_p - \mu_p) / S_p
        g_i(x_p) = \gamma_{\max} \tanh(\alpha_i u_p(x_p))
        R_i(x_p) = R_{0,i} \exp(g_i(x_p))   (with \alpha_p = 0, g_p(x_p) = 0)

    Forward Map z = \Phi(x):
        z_p = \operatorname{arsinh}((x_p - \mu_p) / R_{0,p})
        z_i = \operatorname{arsinh}((x_i - \mu_i) / R_i(x_p))   (i != p)

    Inverse Map x = \Phi^{-1}(z) (exact closed-form triangular evaluation):
        x_p = \mu_p + R_{0,p} \sinh(z_p)
        x_i = \mu_i + R_i(x_p) \sinh(z_i)   (i != p)

    Triangular Jacobian Log-Determinant:
        \log |\det D\Phi^{-1}(z)| = \sum_i (\log R_{0,i} + g_i(x_p) + \log \cosh(z_i))
        \log |\det D\Phi(x)| = -\log |\det D\Phi^{-1}(\Phi(x))|

    Exact Pullback Gradient (\nabla_z \log \tilde{\pi}(z)):
        For child coordinates i != p:
            \partial_{z_i} \log \tilde{\pi}(z) =
                \sqrt{R_i(x_p)^2 + (x_i - \mu_i)^2} (\nabla_x \log \pi(x))_i + \tanh(z_i)
        For parent coordinate p:
            \partial_{z_p} \log \tilde{\pi}(z) =
                (\partial x_p / \partial z_p) [
                    (\nabla_x \log \pi(x))_p
                    + \sum_{i != p} g'_i(x_p) ((x_i - \mu_i) (\nabla_x \log \pi(x))_i + 1)
                ] + \tanh(z_p)
    """

    def __init__(
        self,
        d: int | None = None,
        radii: float | Sequence[float] | np.ndarray = 1.0,
        location: float | Sequence[float] | np.ndarray = 0.0,
        coupling: float | Sequence[float] | np.ndarray = 0.0,
        parent_idx: int = 0,
        parent_scale: float = 3.0,
        max_log_scale: float = 4.0,
        mask: Sequence[bool] | np.ndarray | None = None,
    ) -> None:
        """Initializes HierarchicalArcsinhTransform.

        Args:
            d: Dimension of the space R^d. Inferred from array arguments if None.
            radii: Base scale R_{0,i} > 0 per coordinate (scalar or shape (d,)).
            location: Centering location mu_i per coordinate (scalar or shape (d,)).
            coupling: Dimensionless funnel coupling alpha_i per coordinate (scalar or shape (d,)).
                Forced to 0.0 at parent_idx.
            parent_idx: Index p in {0, ..., d-1} of the hierarchical parent coordinate.
            parent_scale: Characteristic scale S_p > 0 of the parent coordinate.
            max_log_scale: Smooth tanh cap gamma_max > 0 on the log-scale shift |g_i(x_p)|.
            mask: Optional boolean mask of shape (d,) indicating transformed coordinates.
        """
        if d is None:
            for candidate in (radii, location, coupling, mask):
                if candidate is not None and hasattr(candidate, "__len__"):
                    d = len(candidate)  # type: ignore[arg-type]
                    break
            if d is None:
                raise ValueError("Must provide dimension d when all parameters are scalars")

        d_int = int(d)
        if d_int < 1:
            raise ValueError(f"Dimension d must be positive, got {d_int}")

        super().__init__(input_dim=d_int, output_dim=d_int)

        p_idx = int(parent_idx)
        if p_idx < 0 or p_idx >= d_int:
            raise ValueError(f"parent_idx must be in [0, {d_int}), got {p_idx}")
        self._parent_idx: int = p_idx

        s_p = float(parent_scale)
        if s_p <= 0.0 or not np.isfinite(s_p):
            raise ValueError(f"parent_scale must be strictly positive and finite, got {parent_scale}")
        self._parent_scale: float = s_p

        g_max = float(max_log_scale)
        if g_max <= 0.0 or not np.isfinite(g_max):
            raise ValueError(f"max_log_scale must be strictly positive and finite, got {max_log_scale}")
        self._max_log_scale: float = g_max

        if hasattr(radii, "__len__"):
            r_arr = np.asarray(radii, dtype=np.float64).copy()
            if r_arr.shape != (d_int,):
                raise ValueError(f"radii shape {r_arr.shape} must match (d,) with d={d_int}")
        else:
            r_arr = np.full(d_int, float(radii), dtype=np.float64)

        if np.any(r_arr <= 0.0) or not np.all(np.isfinite(r_arr)):
            raise ValueError("All coordinate radii must be strictly positive and finite")
        self._radii: np.ndarray = r_arr

        if hasattr(location, "__len__"):
            loc_arr = np.asarray(location, dtype=np.float64).copy()
            if loc_arr.shape != (d_int,):
                raise ValueError(f"location shape {loc_arr.shape} must match (d,) with d={d_int}")
        else:
            loc_arr = np.full(d_int, float(location), dtype=np.float64)

        if not np.all(np.isfinite(loc_arr)):
            raise ValueError("All coordinate locations must be finite")
        self._location: np.ndarray = loc_arr

        if mask is None:
            self._mask: np.ndarray = np.ones(d_int, dtype=bool)
        else:
            m_arr = np.asarray(mask, dtype=bool).copy()
            if m_arr.shape != (d_int,):
                raise ValueError(f"mask shape {m_arr.shape} must match (d,) with d={d_int}")
            self._mask = m_arr

        if hasattr(coupling, "__len__"):
            c_arr = np.asarray(coupling, dtype=np.float64).copy()
            if c_arr.shape != (d_int,):
                raise ValueError(f"coupling shape {c_arr.shape} must match (d,) with d={d_int}")
        else:
            c_arr = np.full(d_int, float(coupling), dtype=np.float64)

        if not np.all(np.isfinite(c_arr)):
            raise ValueError("All coupling coefficients must be finite")
        c_arr[self._parent_idx] = 0.0
        c_arr[~self._mask] = 0.0
        self._coupling: np.ndarray = c_arr

    @property
    def radii(self) -> np.ndarray:
        """Base coordinate scale vector R_0 of shape (d,)."""
        return self._radii

    @property
    def location(self) -> np.ndarray:
        """Coordinate location vector mu of shape (d,)."""
        return self._location

    @property
    def coupling(self) -> np.ndarray:
        """Coupling coefficient vector alpha of shape (d,) with alpha[parent_idx] == 0."""
        return self._coupling

    @property
    def parent_idx(self) -> int:
        """Index p of the parent coordinate."""
        return self._parent_idx

    @property
    def parent_scale(self) -> float:
        """Characteristic scale S_p > 0 of the parent coordinate."""
        return self._parent_scale

    @property
    def max_log_scale(self) -> float:
        """Maximum log-scale shift gamma_max > 0."""
        return self._max_log_scale

    @property
    def mask(self) -> np.ndarray:
        """Boolean coordinate transformation mask of shape (d,)."""
        return self._mask

    def _log_scale_and_deriv(self, x_p: float) -> tuple[np.ndarray, np.ndarray]:
        """Computes log-scale shift g_i(x_p) and derivative g'_i(x_p) wrt x_p."""
        if not np.isfinite(x_p):
            sgn = 1.0 if x_p > 0 else -1.0
            g = self._max_log_scale * np.sign(self._coupling) * sgn
            return g, np.zeros(self.d, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            u_p = (x_p - float(self._location[self._parent_idx])) / self._parent_scale
            arg = np.clip(self._coupling * u_p, -500.0, 500.0)
            tanh_arg = np.tanh(arg)
            g = self._max_log_scale * tanh_arg
            g_prime = (self._max_log_scale * self._coupling / self._parent_scale) * (1.0 - tanh_arg**2)
        return g, g_prime

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Applies forward triangular diffeomorphism z = Phi(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != self.d:
            raise ValueError(f"Expected input shape ({self.d},), got {x_arr.shape}")

        z = x_arr.copy()
        m = self._mask
        if np.any(m):
            x_p = float(x_arr[self._parent_idx])
            g, _ = self._log_scale_and_deriv(x_p)
            r_eff = self._radii * np.exp(g)

            xm = x_arr[m] - self._location[m]
            rm = r_eff[m]
            scaled = xm / rm
            zm = np.arcsinh(scaled)
            z[m] = zm

            abs_zm = np.abs(zm)
            log_cosh = abs_zm - np.log(2.0) + np.log1p(np.exp(-2.0 * abs_zm))
            log_det = -float(np.sum(np.log(self._radii[m]) + g[m] + log_cosh))
        else:
            log_det = 0.0

        return z, log_det

    def inverse(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Applies exact closed-form triangular inverse diffeomorphism x = Phi^{-1}(z)."""
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1 or y_arr.shape[0] != self.d:
            raise ValueError(f"Expected input shape ({self.d},), got {y_arr.shape}")

        x = y_arr.copy()
        m = self._mask
        if np.any(m):
            p = self._parent_idx
            if m[p]:
                with np.errstate(over="ignore"):
                    x_p = float(self._location[p] + self._radii[p] * np.sinh(y_arr[p]))
            else:
                x_p = float(y_arr[p])

            g, _ = self._log_scale_and_deriv(x_p)
            r_eff = self._radii * np.exp(g)

            ym = y_arr[m]
            rm = r_eff[m]
            locm = self._location[m]
            with np.errstate(over="ignore"):
                x[m] = locm + rm * np.sinh(ym)
            x[p] = x_p

            abs_ym = np.abs(ym)
            log_cosh = abs_ym - np.log(2.0) + np.log1p(np.exp(-2.0 * abs_ym))
            log_det = float(np.sum(np.log(self._radii[m]) + g[m] + log_cosh))
        else:
            log_det = 0.0

        return x, log_det

    def forward_grad(self, x: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        """Computes exact pullback gradient nabla_z log tilde{pi}(z) at z = Phi(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        gx = np.asarray(grad_x, dtype=np.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != self.d:
            raise ValueError(f"Expected input shape ({self.d},), got {x_arr.shape}")
        if gx.ndim != 1 or gx.shape[0] != self.d:
            raise ValueError(f"Expected grad_x shape ({self.d},), got {gx.shape}")

        gz = gx.copy()
        m = self._mask
        if np.any(m):
            p = self._parent_idx
            x_p = float(x_arr[p])
            g, g_prime = self._log_scale_and_deriv(x_p)
            r_eff = self._radii * np.exp(g)

            with np.errstate(over="ignore", invalid="ignore"):
                xm = x_arr[m] - self._location[m]
                rm = r_eff[m]
                scale = np.hypot(rm, xm)
                gz[m] = scale * gx[m] + (xm / scale)

                # Cross-coupling chain-rule contribution into parent coordinate p
                if np.any(g_prime != 0.0):
                    centered_all = x_arr - self._location
                    cross_term = float(np.sum(g_prime * (centered_all * gx + 1.0)))
                    dx_p_dz_p = float(np.hypot(self._radii[p], x_p - float(self._location[p]))) if m[p] else 1.0
                    gz[p] += dx_p_dz_p * cross_term

        return gz

    @classmethod
    def calibrate_from_samples(
        cls,
        samples: np.ndarray,
        parent_idx: int = 0,
        max_log_scale: float = 6.0,
        corr_threshold: float = 0.15,
        min_radius: float = 1e-4,
        max_radius: float = 1e4,
        mask: Sequence[bool] | np.ndarray | None = None,
    ) -> "HierarchicalArcsinhTransform":
        """Calibrates location, base radii, parent scale, and funnel coupling from samples.

        Estimates coupling alpha_i via log-magnitude regression of |x_i - mu_i| against
        the normalized parent coordinate u_p = (x_p - mu_p) / S_p, applying a statistical
        correlation threshold so benign independent targets retain alpha_i = 0.
        For coupled child coordinates, pools slopes via empirical Bayes shrinkage toward
        the group median and estimates the funnel apex location mu_i via inverse-variance
        weighting in the funnel neck.
        """
        s = np.asarray(samples, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError(f"Expected 2D array of samples, got shape {s.shape}")
        n, d = s.shape
        if n < 20:
            raise ValueError(f"Need at least 20 samples to calibrate, got {n}")
        if parent_idx < 0 or parent_idx >= d:
            raise ValueError(f"parent_idx must be in [0, {d}), got {parent_idx}")

        med = np.median(s, axis=0)
        centered = s - med

        # 1. Parent scale S_p and location mu_p
        cp_raw = centered[:, parent_idx]
        q75_p, q25_p = np.quantile(cp_raw, [0.75, 0.25])
        iqr_p = float((q75_p - q25_p) / 1.3489795)
        mad_p = float(np.median(np.abs(cp_raw)) / 0.67448975)
        std_p = float(np.std(cp_raw))
        raw_s_p = iqr_p if iqr_p > min_radius else max(mad_p, std_p, min_radius)
        parent_scale = float(np.clip(raw_s_p, min_radius, max_radius))

        if abs(float(med[parent_idx])) <= 2.5 * parent_scale / np.sqrt(float(n)):
            med[parent_idx] = 0.0
        cp = s[:, parent_idx] - med[parent_idx]

        u_p = cp / parent_scale
        u_c = u_p - float(np.mean(u_p))
        var_u = float(np.mean(u_c**2))

        coupling = np.zeros(d, dtype=np.float64)
        raw_slopes = np.zeros(d, dtype=np.float64)
        slope_ses = np.ones(d, dtype=np.float64)
        coupled_indices: list[int] = []
        eff_corr_thresh = max(corr_threshold, 3.2 / np.sqrt(float(n)))

        if var_u > 1e-8:
            for i in range(d):
                if i == parent_idx:
                    continue
                ci = centered[:, i]
                mad_i = max(float(np.median(np.abs(ci))), 1e-8)
                log_abs = np.log(np.maximum(np.abs(ci), 1e-4 * mad_i))
                y_c = log_abs - float(np.mean(log_abs))
                var_y = float(np.mean(y_c**2))
                if var_y <= 1e-10:
                    continue
                cov_uy = float(np.mean(u_c * y_c))
                corr = cov_uy / np.sqrt(var_u * var_y)
                if abs(corr) >= eff_corr_thresh:
                    b_i = cov_uy / var_u
                    resid_var = max(1e-6, var_y - (cov_uy**2) / var_u)
                    se_b = float(np.sqrt(resid_var / (max(1, n - 2) * var_u)))
                    raw_slopes[i] = b_i
                    slope_ses[i] = se_b
                    coupled_indices.append(i)

        if len(coupled_indices) > 0:
            med_slope = float(np.median(raw_slopes[coupled_indices]))
            for i in coupled_indices:
                b_i = raw_slopes[i]
                if len(coupled_indices) >= 2 and abs(b_i - med_slope) <= 3.0 * slope_ses[i]:
                    weight_pool = 1.0 - 1.0 / float(len(coupled_indices) + 1)
                    b_eff = weight_pool * med_slope + (1.0 - weight_pool) * b_i
                else:
                    b_eff = b_i
                ratio = float(np.clip(b_eff / max_log_scale, -0.92, 0.92))
                coupling[i] = float(np.arctanh(ratio))

            # Refine funnel apex location mu_i using inverse-variance weights in the neck
            g_matrix_init = max_log_scale * np.tanh(np.outer(u_p, coupling))
            for i in coupled_indices:
                log_w = np.clip(-2.0 * g_matrix_init[:, i], -40.0, 40.0)
                log_w -= float(np.max(log_w))
                w = np.exp(log_w)
                w /= float(np.sum(w))
                xi = s[:, i]
                mu_blue = float(np.sum(w * xi))
                n_eff = 1.0 / float(np.sum(w**2))
                w_var = float(np.sum(w * ((xi - mu_blue) ** 2)))
                se_mu = float(np.sqrt(max(1e-16, w_var) / max(1.0, n_eff)))
                med[i] = 0.0 if abs(mu_blue) <= 3.5 * se_mu else mu_blue

        centered = s - med

        # 2. Compute de-warped residuals tilde_x_i = (x_i - mu_i) * exp(-g_i(x_p))
        g_matrix = max_log_scale * np.tanh(np.outer(u_p, coupling))
        dewarped = centered * np.exp(-g_matrix)
        q75, q25 = np.quantile(dewarped, [0.75, 0.25], axis=0)
        iqr_scale = (q75 - q25) / 1.3489795
        mad_scale = np.median(np.abs(dewarped), axis=0) / 0.67448975
        q90_abs = np.quantile(np.abs(dewarped), 0.90, axis=0)
        base_scale = np.where(iqr_scale > min_radius, iqr_scale, np.maximum(mad_scale, q90_abs))

        if len(coupled_indices) > 0:
            if len(coupled_indices) >= 2:
                med_child_scale = float(np.median(base_scale[coupled_indices]))
                for i in coupled_indices:
                    if 0.5 * med_child_scale <= base_scale[i] <= 2.0 * med_child_scale:
                        base_scale[i] = 0.75 * med_child_scale + 0.25 * base_scale[i]
            radii = np.clip(base_scale * 6.0, min_radius, max_radius)
            radii[parent_idx] = float(np.clip(parent_scale * 6.0, min_radius, max_radius))
        else:
            radii = np.clip(base_scale, min_radius, max_radius)
            radii[parent_idx] = parent_scale

        return cls(
            d=d,
            radii=radii,
            location=med,
            coupling=coupling,
            parent_idx=parent_idx,
            parent_scale=parent_scale,
            max_log_scale=max_log_scale,
            mask=mask,
        )
