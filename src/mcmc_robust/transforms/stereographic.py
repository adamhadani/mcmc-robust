"""Exact stereographic projection between Euclidean R^d and sphere S^d (Yang et al. 2024)."""

import numpy as np

from mcmc_robust.transforms.base import SpaceTransformation


class StereographicTransform(SpaceTransformation):
    r"""Stereographic projection between R^d and hypersphere S^d \subset R^{d+1}.

    Maps Euclidean coordinate x \in R^d to hypersphere z \in S^d:
        z_{1:d} = \frac{2 R x}{R^2 + ||x||^2}
        z_{d+1} = \frac{||x||^2 - R^2}{R^2 + ||x||^2}

    Inverse mapping S^d \ {N} -> R^d:
        x = \frac{R}{1 - z_{d+1}} z_{1:d}

    Conformal volume element:
        d\sigma(z) = (2R / (R^2 + ||x||^2))^d dx
        \log |\det J_fwd(x)| = d \ln(2R) - d \ln(R^2 + ||x||^2)
        \log |\det J_inv(z)| = d \ln(R) - d \ln(1 - z_{d+1})
    """

    def __init__(self, d: int, scale: float = 1.0) -> None:
        """Initializes StereographicTransform.

        Args:
            d: Dimension of Euclidean space R^d (must be >= 1).
            scale: Conformal sphere radius R > 0.
        """
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        super().__init__(input_dim=d, output_dim=d + 1)
        self._scale: float = float(scale)

    @property
    def scale(self) -> float:
        """Conformal scale radius R."""
        return self._scale

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != self.input_dim:
            raise ValueError(f"Expected shape ({self.input_dim},), got {x_arr.shape}")

        R = self._scale
        r2 = float(np.sum(x_arr**2))
        denom = R**2 + r2

        z = np.empty(self.output_dim, dtype=np.float64)
        z[: self.input_dim] = (2.0 * R / denom) * x_arr
        z[self.input_dim] = (r2 - R**2) / denom

        log_det = self.input_dim * (np.log(2.0 * R) - np.log(denom))
        return z, float(log_det)

    def inverse(self, z: np.ndarray) -> tuple[np.ndarray, float]:
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim != 1 or z_arr.shape[0] != self.output_dim:
            raise ValueError(f"Expected shape ({self.output_dim},), got {z_arr.shape}")

        R = self._scale
        z_last = float(z_arr[-1])
        if z_last >= 1.0 - 1e-15:
            raise ValueError(f"North pole z_{{d+1}} = {z_last:.15f} is singular under stereographic inverse")

        # Numerically stable 1 - z_last without cancellation near North pole
        if z_last > 0.0:
            omz = float(np.sum(z_arr[:-1] ** 2) / (1.0 + z_last))
        else:
            omz = 1.0 - z_last

        x = (R / omz) * z_arr[:-1]
        log_det = self.input_dim * (np.log(R) - np.log(omz))
        return x, float(log_det)

    def forward_grad(self, x: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        r"""Computes the tangent Riemannian gradient on T_z S^d."""
        x_arr = np.asarray(x, dtype=np.float64)
        gx = np.asarray(grad_x, dtype=np.float64)
        d = self.input_dim
        R = self._scale

        r2 = float(np.sum(x_arr**2))
        denom = R**2 + r2

        z = np.empty(d + 1, dtype=np.float64)
        z[:d] = (2.0 * R / denom) * x_arr
        z[d] = (r2 - R**2) / denom
        omz = (2.0 * R**2) / denom

        # Ambient Euclidean gradient in R^{d+1}
        g_amb = np.empty(d + 1, dtype=np.float64)
        g_amb[:d] = (R / omz) * gx
        xg = float(np.dot(x_arr, gx))
        g_amb[d] = (xg + d) / omz

        # Tangent space projection: grad_S = g_amb - <z, g_amb> z
        inner = float(np.dot(z, g_amb))
        g_tan = g_amb - inner * z
        return g_tan
