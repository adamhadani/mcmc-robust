"""C^2 smooth radial diffeomorphic compression outside ball of radius R (Power & Vasdekis §4.2.1)."""

import numpy as np

from mcmc_robust.transforms.base import SpaceTransformation


class RadialDiffeomorphicTransform(SpaceTransformation):
    r"""Invertible radial compression on R^d with C^2 smooth transition at r = R.

    Leaves the bulk ||x|| <= R strictly unmodified (identity map), and applies
    logarithmic radial compression outside R:
        y = \phi(r) \frac{x}{r} \quad (r = ||x|| > R)
    with:
        \phi(r) = R + \frac{1}{k} \ln\left(1 + k(r - R) + \frac{1}{2} k^2 (r - R)^2\right)
    satisfying:
        \phi(R) = R, \quad \phi'(R) = 1, \quad \phi''(R) = 0
    guaranteeing C^2 smoothness across the boundary r = R.

    Exact closed-form inverse:
        r = \phi^{-1}(s) = R + \frac{2 \operatorname{expm1}(k(s - R))}{k (\sqrt{1 + 2 \operatorname{expm1}(k(s - R))} + 1)}

    Exact closed-form Jacobian log-determinant:
        \log |\det J| = (d - 1) \ln(\phi(r) / r) + \ln \phi'(r)
    """

    def __init__(self, d: int, radius: float = 3.0, stiffness: float = 0.5) -> None:
        """Initializes RadialDiffeomorphicTransform.

        Args:
            d: Dimension of R^d (must be >= 1).
            radius: Confinement boundary radius R > 0.
            stiffness: Compression stiffness k > 0.
        """
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        if stiffness <= 0.0:
            raise ValueError(f"stiffness must be positive, got {stiffness}")
        super().__init__(input_dim=d, output_dim=d)
        self._radius: float = float(radius)
        self._stiffness: float = float(stiffness)

    @property
    def radius(self) -> float:
        """Confinement radius R."""
        return self._radius

    @property
    def stiffness(self) -> float:
        """Compression stiffness k."""
        return self._stiffness

    def _phi(self, r: float) -> tuple[float, float, float]:
        """Computes phi(r), phi'(r), phi''(r) for r >= R."""
        R = self._radius
        k = self._stiffness
        if r <= R:
            return r, 1.0, 0.0
        u = r - R
        denom = 1.0 + k * u + 0.5 * k**2 * u**2
        val = R + (1.0 / k) * np.log(denom)
        d1 = (1.0 + k * u) / denom
        d2 = (-(k**2) * u - 0.5 * k**3 * u**2) / (denom**2)
        return val, d1, d2

    def _phi_inv(self, s: float) -> tuple[float, float]:
        """Computes psi(s) = phi^{-1}(s) and psi'(s) for s >= R."""
        R = self._radius
        k = self._stiffness
        if s <= R:
            return s, 1.0
        v = s - R
        if k * v > 709.0:
            return float("nan"), float("nan")
        with np.errstate(all="ignore"):
            ex = np.expm1(k * v)
            rad = 1.0 + 2.0 * ex
            sq = np.sqrt(rad)
            u = (2.0 * ex) / (sq + 1.0) / k
            r = R + u
            d1 = np.exp(k * v) / sq
        return r, d1

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != self.input_dim:
            raise ValueError(f"Expected shape ({self.input_dim},), got {x_arr.shape}")

        r = float(np.linalg.norm(x_arr))
        if r <= self._radius:
            return x_arr.copy(), 0.0

        val, d1, _ = self._phi(r)
        y = (val / r) * x_arr
        log_det = (self.input_dim - 1) * np.log(val / r) + np.log(d1)
        return y, float(log_det)

    def inverse(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1 or y_arr.shape[0] != self.output_dim:
            raise ValueError(f"Expected shape ({self.output_dim},), got {y_arr.shape}")

        s = float(np.linalg.norm(y_arr))
        if s <= self._radius:
            return y_arr.copy(), 0.0

        r, inv_d1 = self._phi_inv(s)
        if np.isnan(r) or np.isnan(inv_d1):
            return np.full_like(y_arr, np.nan), float("nan")

        x = (r / s) * y_arr
        with np.errstate(all="ignore"):
            log_det = (self.output_dim - 1) * np.log(r / s) + np.log(inv_d1)
        return x, float(log_det)

    def forward_grad(self, x: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        gx = np.asarray(grad_x, dtype=np.float64)

        r = float(np.linalg.norm(x_arr))
        if r <= self._radius:
            return gx.copy()

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            val, d1, d2 = self._phi(r)
            u = x_arr / r
            ug = float(np.dot(u, gx))

            # Scalar curvature term of log-det Jacobian: d/dr log |det J|
            kappa = (self.input_dim - 1) * (d1 / val - 1.0 / r) + (d2 / d1)

            # Matrix-free Sherman-Morrison inversion:
            # gy = J^{-T} (gx - kappa * u)
            rad_comp = (ug - kappa) / d1
            tan_scale = r / val
            gy = tan_scale * gx + (rad_comp - tan_scale * ug) * u
        return gy
