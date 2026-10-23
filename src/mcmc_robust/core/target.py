"""Target distribution abstract base class and evaluation protocol."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TargetDistribution(ABC):
    """Abstract base class for probability distribution targets on R^d.

    Subclasses implement log_prob and grad_log_prob for continuous targets.
    Optionally, targets provide exact ground-truth samples and analytical
    moments/quantiles for statistical verification.
    """

    def __init__(self, d: int) -> None:
        """Initialize target distribution with state space dimension d.

        Args:
            d: Dimensionality of the continuous state space (must be >= 1).

        Raises:
            ValueError: If d < 1.
        """
        if d < 1:
            raise ValueError(f"Dimension d must be a positive integer, got {d}")
        self._d: int = int(d)

    @property
    def d(self) -> int:
        """Dimensionality of the state space."""
        return self._d

    @property
    def dim(self) -> int:
        """Alias for dimensionality d."""
        return self._d

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """Validates that x is a 1D float64 array of shape (d,)."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != self._d:
            raise ValueError(f"Expected 1D array of shape ({self._d},), got {x_arr.shape}")
        return x_arr

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> float:
        r"""Computes log-density log \pi(x) = -U(x).

        Args:
            x: Coordinate vector of shape (d,).

        Returns:
            Scalar log-probability value as a float.
        """

    @abstractmethod
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        r"""Computes gradient of log-density \nabla log \pi(x) = -\nabla U(x).

        Args:
            x: Coordinate vector of shape (d,).

        Returns:
            Gradient vector of shape (d,) with dtype float64.
        """

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Jointly computes log_prob and grad_log_prob.

        Subclasses should override this method to share intermediate computations
        (e.g., shared matrix-vector multiplications, norms, or activations).

        Args:
            x: Coordinate vector of shape (d,).

        Returns:
            Tuple of (log_prob, grad_log_prob).
        """
        return self.log_prob(x), self.grad_log_prob(x)

    @abstractmethod
    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draws exact i.i.d. samples from the target distribution.

        Args:
            n: Number of samples to draw (must be >= 1).
            rng: Optional numpy Generator instance for reproducibility.

        Returns:
            Array of shape (n, d) containing exact independent samples.
        """

    @abstractmethod
    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        """Returns exact analytical moments, covariance, or marginal quantiles.

        Returns:
            Dictionary containing available analytical ground truths, e.g.:
            - 'mean': np.ndarray of shape (d,)
            - 'variance': np.ndarray of shape (d,)
            - 'cov': np.ndarray of shape (d, d)
            - 'quantiles': dict mapping percentile floats/strings to ndarrays
        """
