"""Warmup adaptation algorithms: Dual Averaging and Welford Preconditioning."""

import numpy as np


class DualAveragingAdaptation:
    """Hoffman-Gelman dual averaging step size adaptation.

    Dynamically tunes step size h during warmup to achieve a specified
    target acceptance rate alpha_star.
    """

    def __init__(
        self,
        initial_step_size: float = 1.0,
        target_accept: float = 0.574,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
        min_step_size: float = 1e-8,
        max_step_size: float = 1e3,
    ) -> None:
        r"""Initializes dual averaging parameters.

        Args:
            initial_step_size: Initial step size h_0 > 0.
            target_accept: Target acceptance probability \alpha^\star \in (0, 1).
            gamma: Shrinkage weight parameter \gamma > 0.
            t0: Stabilization parameter t_0 >= 0.
            kappa: Polyak-Ruppert smoothing exponent \kappa \in (0.5, 1.0].
            min_step_size: Lower clamping bound on step size.
            max_step_size: Upper clamping bound on step size.
        """
        if initial_step_size <= 0.0:
            raise ValueError(f"initial_step_size must be positive, got {initial_step_size}")
        if not (0.0 < target_accept < 1.0):
            raise ValueError(f"target_accept must be in (0, 1), got {target_accept}")

        self.initial_step_size: float = float(initial_step_size)
        self.target_accept: float = float(target_accept)
        self.gamma: float = float(gamma)
        self.t0: float = float(t0)
        self.kappa: float = float(kappa)
        self.min_step_size: float = float(min_step_size)
        self.max_step_size: float = float(max_step_size)

        self.reset()

    def reset(self, initial_step_size: float | None = None) -> None:
        """Resets adaptation counters and state variables."""
        if initial_step_size is not None:
            self.initial_step_size = float(initial_step_size)
        self.mu: float = float(np.log(10.0 * self.initial_step_size))
        self.m: int = 0
        self.h: float = self.initial_step_size
        self.log_h_bar: float = float(np.log(self.initial_step_size))
        self.h_bar: float = self.initial_step_size
        self.H_bar: float = 0.0

    @property
    def step_size(self) -> float:
        """Current exploration step size h_m."""
        return self.h

    @property
    def smoothed_step_size(self) -> float:
        r"""Smoothed step size \bar{h}_m for post-warmup sampling."""
        return self.h_bar

    @property
    def final_step_size(self) -> float:
        """Alias for smoothed_step_size."""
        return self.h_bar

    def update(self, accept_prob: float) -> float:
        r"""Updates adaptation state with the observed acceptance probability.

        Args:
            accept_prob: Observed acceptance probability \alpha \in [0, 1].

        Returns:
            Updated exploration step size h_{m+1}.
        """
        # Clamp accept_prob to valid [0, 1] range, guarding against non-finite/NaN values
        if not np.isfinite(accept_prob):
            a = 0.0
        else:
            a = float(np.clip(accept_prob, 0.0, 1.0))
        self.m += 1

        # 1. Update running statistic H_bar
        eta = 1.0 / (self.m + self.t0)
        self.H_bar = (1.0 - eta) * self.H_bar + eta * (self.target_accept - a)

        # 2. Update log exploration step size
        log_h = self.mu - (np.sqrt(self.m) / self.gamma) * self.H_bar
        self.h = float(np.clip(np.exp(log_h), self.min_step_size, self.max_step_size))

        # 3. Update smoothed log step size
        weight = self.m ** (-self.kappa)
        self.log_h_bar = weight * np.log(self.h) + (1.0 - weight) * self.log_h_bar
        self.h_bar = float(np.clip(np.exp(self.log_h_bar), self.min_step_size, self.max_step_size))

        return self.h

    def finalize(self) -> float:
        """Freezes and returns the final smoothed step size."""
        return self.h_bar


class WelfordPreconditioner:
    """Online diagonal covariance estimation using Welford's algorithm.

    Accumulates streaming sample statistics coordinate-wise in O(d) time and
    O(d) space without storing trajectory history in memory.
    """

    def __init__(
        self,
        d: int,
        shrinkage_samples: float = 10.0,
        prior_variance: float = 1.0,
        min_variance: float = 1e-4,
        max_variance: float = 1e4,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> None:
        """Initializes Welford accumulator for dimension d.

        Args:
            d: Dimension of state space (must be >= 1).
            shrinkage_samples: Pseudo-sample weight for prior shrinkage.
            prior_variance: Prior variance value for shrinkage.
            min_variance: Lower clipping bound for estimated variance.
            max_variance: Upper clipping bound for estimated variance.
            clip_min: Optional alias for min_variance.
            clip_max: Optional alias for max_variance.
        """
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d: int = int(d)
        self.shrinkage_samples: float = float(shrinkage_samples)
        self.prior_variance: float = float(prior_variance)

        if clip_min is not None:
            min_variance = clip_min
        if clip_max is not None:
            max_variance = clip_max

        self.min_variance: float = float(min_variance)
        self.max_variance: float = float(max_variance)

        if self.shrinkage_samples < 0.0:
            raise ValueError(f"shrinkage_samples must be non-negative, got {self.shrinkage_samples}")
        if self.prior_variance <= 0.0:
            raise ValueError(f"prior_variance must be positive, got {self.prior_variance}")
        if self.min_variance <= 0.0:
            raise ValueError(f"min_variance must be positive, got {self.min_variance}")
        if self.min_variance >= self.max_variance:
            raise ValueError(
                f"min_variance ({self.min_variance}) must be strictly less than max_variance ({self.max_variance})"
            )

        self.reset()

    def reset(self) -> None:
        """Resets sample counters and accumulators."""
        self.n: int = 0
        self.mean: np.ndarray = np.zeros(self.d, dtype=np.float64)
        self.M2: np.ndarray = np.zeros(self.d, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Updates streaming statistics with a single sample vector x.

        Args:
            x: Sample coordinate vector of shape (d,).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (self.d,):
            raise ValueError(f"Expected shape ({self.d},), got {x_arr.shape}")

        self.n += 1
        delta = x_arr - self.mean
        self.mean += delta / self.n
        delta2 = x_arr - self.mean
        self.M2 += delta * delta2

    def update_batch(self, X: np.ndarray) -> None:
        """Updates streaming statistics with a batch of samples.

        Args:
            X: Array of shape (n_samples, d).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.d:
            raise ValueError(f"Expected 2D array of shape (N, {self.d}), got {X_arr.shape}")
        for i in range(X_arr.shape[0]):
            self.update(X_arr[i])

    @property
    def sample_variance(self) -> np.ndarray:
        """Unbiased coordinate-wise sample variance vector."""
        if self.n <= 1:
            return np.ones(self.d, dtype=np.float64)
        return self.M2 / (self.n - 1)

    @property
    def sample_std(self) -> np.ndarray:
        """Sample standard deviation vector."""
        return np.sqrt(np.maximum(self.sample_variance, 0.0))

    @property
    def variance(self) -> np.ndarray:
        """Alias for regularized diagonal preconditioner variance."""
        return self.diagonal_preconditioner()

    @property
    def scale(self) -> np.ndarray:
        """Alias for diagonal standard deviation scale."""
        return self.diagonal_scale()

    def diagonal_preconditioner(self) -> np.ndarray:
        r"""Returns regularized diagonal variance scale vector \Sigma \in R^d.

        Applies Bayesian shrinkage towards prior_variance:
        \Sigma_i = \frac{n}{n + n_0} s_i^2 + \frac{n_0}{n + n_0} \sigma_0^2
        """
        s2 = self.sample_variance
        if self.n == 0:
            return np.full(self.d, self.prior_variance, dtype=np.float64)

        weight = self.n / (self.n + self.shrinkage_samples)
        reg_s2 = weight * s2 + (1.0 - weight) * self.prior_variance
        return np.clip(reg_s2, self.min_variance, self.max_variance)

    def diagonal_scale(self) -> np.ndarray:
        r"""Returns standard deviation scale vector \Sigma^{1/2} for noise injection."""
        return np.sqrt(self.diagonal_preconditioner())

    def inverse_preconditioner(self) -> np.ndarray:
        r"""Returns inverse diagonal metric M = \Sigma^{-1} (mass matrix)."""
        return 1.0 / self.diagonal_preconditioner()
