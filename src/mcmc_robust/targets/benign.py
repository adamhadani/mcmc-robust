"""Benign Gaussian target distributions with analytical solutions."""

from typing import Any

import numpy as np
from scipy import linalg, stats

from mcmc_robust.core.target import TargetDistribution


class IsotropicGaussian(TargetDistribution):
    """Standard isotropic log-concave Gaussian N(mu, sigma^2 I_d)."""

    def __init__(
        self,
        d: int,
        mean: np.ndarray | None = None,
        std: float = 1.0,
        cov: np.ndarray | None = None,
    ) -> None:
        super().__init__(d=d)
        if cov is not None:
            # If cov is passed (e.g. sigma^2 * I_d)
            cov_arr = np.asarray(cov, dtype=np.float64)
            if cov_arr.ndim == 2:
                std = float(np.sqrt(np.mean(np.diag(cov_arr))))
            elif cov_arr.ndim == 1:
                std = float(np.sqrt(np.mean(cov_arr)))

        if std <= 0.0:
            raise ValueError(f"std must be positive, got {std}")
        self.std: float = float(std)

        if mean is None:
            self.mean: np.ndarray = np.zeros(self.d, dtype=np.float64)
        else:
            m = np.asarray(mean, dtype=np.float64)
            if m.shape != (self.d,):
                raise ValueError(f"mean shape {m.shape} must match ({self.d},)")
            self.mean = m

    @property
    def cov(self) -> np.ndarray:
        return (self.std**2) * np.eye(self.d, dtype=np.float64)

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        diff = x_arr - self.mean
        sq_norm = float(np.dot(diff, diff))
        return float(-0.5 * self.d * np.log(2.0 * np.pi) - self.d * np.log(self.std) - 0.5 * sq_norm / (self.std**2))

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mean
        return -diff / (self.std**2)

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mean
        sq_norm = float(np.dot(diff, diff))
        grad = -diff / (self.std**2)
        lp = float(-0.5 * self.d * np.log(2.0 * np.pi) - self.d * np.log(self.std) - 0.5 * sq_norm / (self.std**2))
        return lp, grad

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(loc=self.mean, scale=self.std, size=(n, self.d))

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "variance": np.full(self.d, self.std**2, dtype=np.float64),
            "cov": self.cov,
            "q05": self.mean + self.std * stats.norm.ppf(0.05),
            "q50": self.mean.copy(),
            "q95": self.mean + self.std * stats.norm.ppf(0.95),
            "q99": self.mean + self.std * stats.norm.ppf(0.99),
        }


class DiagonalGaussian(TargetDistribution):
    """Anisotropic Gaussian with diagonal covariance and condition number kappa."""

    def __init__(
        self,
        d: int,
        mean: np.ndarray | None = None,
        stds: np.ndarray | None = None,
        condition_number: float = 100.0,
        kappa: float | None = None,
        sigma_min: float | None = None,
    ) -> None:
        super().__init__(d=d)
        if kappa is not None:
            condition_number = kappa
        if condition_number < 1.0:
            raise ValueError(f"condition_number must be >= 1.0, got {condition_number}")
        self.condition_number: float = float(condition_number)

        if mean is None:
            self.mean: np.ndarray = np.zeros(self.d, dtype=np.float64)
        else:
            m = np.asarray(mean, dtype=np.float64)
            if m.shape != (self.d,):
                raise ValueError(f"mean shape {m.shape} must match ({self.d},)")
            self.mean = m

        if stds is not None:
            s = np.asarray(stds, dtype=np.float64)
            if s.shape != (self.d,):
                raise ValueError(f"stds shape {s.shape} must match ({self.d},)")
            if np.any(s <= 0.0):
                raise ValueError("All stds must be strictly positive")
            self.stds: np.ndarray = s
        else:
            # Construct geometric progression with condition number kappa = (sigma_max / sigma_min)^2
            if self.d == 1:
                self.stds = np.array([1.0], dtype=np.float64)
            else:
                if sigma_min is not None and sigma_min > 0:
                    s_min = float(sigma_min)
                    s_max = s_min * np.sqrt(self.condition_number)
                    log_stds = np.linspace(np.log(s_max), np.log(s_min), self.d)
                else:
                    # sigma_1 = 1.0, sigma_d = 1 / sqrt(kappa)
                    log_stds = np.linspace(0.0, -0.5 * np.log(self.condition_number), self.d)
                self.stds = np.exp(log_stds)

        self._inv_var: np.ndarray = 1.0 / (self.stds**2)
        self._log_det: float = float(np.sum(np.log(self.stds)))

    @property
    def cov(self) -> np.ndarray:
        return np.diag(self.stds**2)

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        diff = x_arr - self.mean
        weighted_sq_norm = float(np.sum((diff**2) * self._inv_var))
        return float(-0.5 * self.d * np.log(2.0 * np.pi) - self._log_det - 0.5 * weighted_sq_norm)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mean
        return -diff * self._inv_var

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mean
        grad = -diff * self._inv_var
        weighted_sq_norm = float(np.sum((diff**2) * self._inv_var))
        lp = float(-0.5 * self.d * np.log(2.0 * np.pi) - self._log_det - 0.5 * weighted_sq_norm)
        return lp, grad

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return self.mean + rng.normal(size=(n, self.d)) * self.stds

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "variance": (self.stds**2).copy(),
            "cov": self.cov,
            "q05": self.mean + self.stds * stats.norm.ppf(0.05),
            "q50": self.mean.copy(),
            "q95": self.mean + self.stds * stats.norm.ppf(0.95),
            "q99": self.mean + self.stds * stats.norm.ppf(0.99),
        }


class FullCovarianceGaussian(TargetDistribution):
    """Dense correlated Gaussian with positive definite covariance matrix."""

    def __init__(
        self,
        d: int,
        mean: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        condition_number: float = 50.0,
    ) -> None:
        super().__init__(d=d)
        if mean is None:
            self.mean: np.ndarray = np.zeros(self.d, dtype=np.float64)
        else:
            m = np.asarray(mean, dtype=np.float64)
            if m.shape != (self.d,):
                raise ValueError(f"mean shape {m.shape} must match ({self.d},)")
            self.mean = m

        if cov is not None:
            cov_mat = np.asarray(cov, dtype=np.float64)
            if cov_mat.shape != (self.d, self.d):
                raise ValueError(f"cov shape {cov_mat.shape} must match ({self.d}, {self.d})")
            self._cov: np.ndarray = cov_mat
        else:
            # Generate reproducible positive definite matrix with specified condition number
            rng = np.random.default_rng(12345 + self.d)
            G = rng.normal(size=(self.d, self.d))
            Q, _ = np.linalg.qr(G)
            if self.d == 1:
                eigs = np.array([1.0], dtype=np.float64)
            else:
                eigs = np.exp(np.linspace(0.0, -np.log(condition_number), self.d))
            self._cov = Q @ np.diag(eigs) @ Q.T
            # Ensure exact symmetry
            self._cov = 0.5 * (self._cov + self._cov.T)

        try:
            self.L: np.ndarray = linalg.cholesky(self._cov, lower=True)
        except linalg.LinAlgError:
            # Add small jitter if necessary
            self._cov += 1e-8 * np.eye(self.d)
            self.L = linalg.cholesky(self._cov, lower=True)

        self._log_det: float = float(2.0 * np.sum(np.log(np.diag(self.L))))

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        diff = x_arr - self.mean
        z = linalg.solve_triangular(self.L, diff, lower=True)
        sq_norm = float(np.dot(z, z))
        return float(-0.5 * self.d * np.log(2.0 * np.pi) - 0.5 * self._log_det - 0.5 * sq_norm)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mean
        z = linalg.solve_triangular(self.L, diff, lower=True)
        w = linalg.solve_triangular(self.L.T, z, lower=False)
        return -w

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mean
        z = linalg.solve_triangular(self.L, diff, lower=True)
        w = linalg.solve_triangular(self.L.T, z, lower=False)
        sq_norm = float(np.dot(z, z))
        lp = float(-0.5 * self.d * np.log(2.0 * np.pi) - 0.5 * self._log_det - 0.5 * sq_norm)
        return lp, -w

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return self.mean + rng.normal(size=(n, self.d)) @ self.L.T

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        vars_ = np.diag(self._cov)
        stds = np.sqrt(vars_)
        return {
            "mean": self.mean.copy(),
            "variance": vars_.copy(),
            "cov": self._cov.copy(),
            "q05": self.mean + stds * stats.norm.ppf(0.05),
            "q50": self.mean.copy(),
            "q95": self.mean + stds * stats.norm.ppf(0.95),
            "q99": self.mean + stds * stats.norm.ppf(0.99),
        }


# Aliases for flexible import
GaussianTarget = IsotropicGaussian
AnisotropicGaussianTarget = DiagonalGaussian
