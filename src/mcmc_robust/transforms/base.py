"""Base classes and contracts for space transformations and transformed targets."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from mcmc_robust.core.target import TargetDistribution


class SpaceTransformation(ABC):
    """Abstract base class for invertible space transformations T: X -> Y.

    Subclasses implement forward and inverse mappings with analytical log-determinants
    of the Jacobian, satisfying the machine-precision inversion and determinant contracts:
        || inverse(forward(x)[0])[0] - x ||_inf < 1e-10
        | forward(x)[1] + inverse(forward(x)[0])[1] | < 1e-10
    """

    def __init__(self, input_dim: int, output_dim: int | None = None) -> None:
        """Initializes transformation dimensions.

        Args:
            input_dim: Dimension d of the primal space X.
            output_dim: Dimension m of the transformed space Y (defaults to input_dim).
        """
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self._input_dim: int = int(input_dim)
        self._output_dim: int = int(output_dim) if output_dim is not None else int(input_dim)

    @property
    def input_dim(self) -> int:
        """Dimension of primal space X."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Dimension of transformed space Y."""
        return self._output_dim

    @property
    def d(self) -> int:
        """Primary dimension alias."""
        return self._input_dim

    @abstractmethod
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        r"""Applies forward transformation y = T(x).

        Args:
            x: Coordinate vector in primal space X, shape (input_dim,).

        Returns:
            Tuple of:
            - y: Transformed coordinate vector in Y, shape (output_dim,).
            - log_det_jacobian: Analytical log |\det J_T(x)|.
        """

    @abstractmethod
    def inverse(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        r"""Applies inverse transformation x = T^{-1}(y).

        Args:
            y: Coordinate vector in transformed space Y, shape (output_dim,).

        Returns:
            Tuple of:
            - x: Primal coordinate vector in X, shape (input_dim,).
            - log_det_jacobian: Analytical log |\det J_{T^{-1}}(y)|.
        """

    @abstractmethod
    def forward_grad(self, x: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        r"""Computes transformed gradient \nabla_y \log \pi_Y(y) at y = T(x).

        Args:
            x: Primal coordinate vector, shape (input_dim,).
            grad_x: Primal gradient \nabla_x \log \pi_X(x), shape (input_dim,).

        Returns:
            Transformed gradient vector \nabla_y \log \pi_Y(y), shape (output_dim,).
        """


class TransformedTarget(TargetDistribution):
    r"""Target distribution induced under an invertible space transformation Y = T(X).

    Given underlying target \pi_X(x) on R^d and bijection T: R^d -> R^m, the push-forward
    density on Y is:
        \log \pi_Y(y) = \log \pi_X(T^{-1}(y)) + \log |\det J_{T^{-1}}(y)|
    with exact analytical gradient:
        \nabla_y \log \pi_Y(y) = [J_{T^{-1}}(y)]^\top \nabla_x \log \pi_X(x) + \nabla_y \log |\det J_{T^{-1}}(y)|

    TransformedTarget implements the full TargetDistribution ABC and integrates
    seamlessly with any MCMC sampler and GradBudgetTracker.
    """

    def __init__(self, target: TargetDistribution, transform: SpaceTransformation) -> None:
        r"""Initializes TransformedTarget.

        Args:
            target: Underlying primal target distribution instance \pi_X.
            transform: Invertible space transformation instance T.
        """
        if target.d != transform.input_dim:
            raise ValueError(f"Target dimension {target.d} does not match transform input_dim {transform.input_dim}")
        super().__init__(d=transform.output_dim)
        self._target: TargetDistribution = target
        self._transform: SpaceTransformation = transform

    @property
    def target(self) -> TargetDistribution:
        """Underlying primal target distribution."""
        return self._target

    @property
    def transform(self) -> SpaceTransformation:
        """Space transformation instance."""
        return self._transform

    def log_prob(self, y: np.ndarray) -> float:
        y_arr = self._validate_input(y)
        x, log_det_inv = self._transform.inverse(y_arr)
        lp_x = self._target.log_prob(x)
        return float(lp_x + log_det_inv)

    def grad_log_prob(self, y: np.ndarray) -> np.ndarray:
        y_arr = self._validate_input(y)
        x, _ = self._transform.inverse(y_arr)
        grad_x = self._target.grad_log_prob(x)
        return self._transform.forward_grad(x, grad_x)

    def log_prob_and_grad(self, y: np.ndarray) -> tuple[float, np.ndarray]:
        y_arr = self._validate_input(y)
        x, log_det_inv = self._transform.inverse(y_arr)
        lp_x, grad_x = self._target.log_prob_and_grad(x)
        lp_y = float(lp_x + log_det_inv)
        grad_y = self._transform.forward_grad(x, grad_x)
        return lp_y, grad_y

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draws exact i.i.d. samples by transforming exact primal samples."""
        x_samples = self._target.sample_ground_truth(n, rng=rng)
        y_samples = np.empty((n, self.d), dtype=np.float64)
        for i in range(n):
            y_i, _ = self._transform.forward(x_samples[i])
            y_samples[i] = y_i
        return y_samples

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        """Returns metadata referencing underlying primal analytical moments."""
        return {
            "type": "TransformedTarget",
            "base_target": type(self._target).__name__,
            "transform": type(self._transform).__name__,
            "base_moments": self._target.exact_moments_or_quantiles(),
        }
