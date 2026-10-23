"""Roughness pathology target distributions."""

from typing import Any

import numpy as np
from scipy import special, stats

from mcmc_robust.core.target import TargetDistribution


class QuarticTarget(TargetDistribution):
    """Superlinear polynomial potential U(x) = (1 / (p * scale)) * ||x||^p, default p=4."""

    def __init__(
        self,
        d: int,
        p: float = 4.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(d=d)
        if p <= 0.0:
            raise ValueError(f"p must be positive, got {p}")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.p: float = float(p)
        self.scale: float = float(scale)

        # Analytical partition function:
        # Z = (2 * pi^(d/2) / Gamma(d/2)) * (1/p) * (p * scale)^(d/p) * Gamma(d/p)
        log_surface_area = np.log(2.0) + (self.d / 2.0) * np.log(np.pi) - special.gammaln(self.d / 2.0)
        log_radial_integral = (
            -np.log(self.p) + (self.d / self.p) * np.log(self.p * self.scale) + special.gammaln(self.d / self.p)
        )
        self._log_Z: float = float(log_surface_area + log_radial_integral)

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf
        with np.errstate(over="ignore", invalid="ignore"):
            sq_norm = float(np.dot(x_arr, x_arr))
            if not np.isfinite(sq_norm) or sq_norm > 1e100:
                return -np.inf
            norm = np.sqrt(sq_norm)
            u = (norm**self.p) / (self.p * self.scale)
            if not np.isfinite(u):
                return -np.inf
            return float(-u - self._log_Z)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return np.full(self.d, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            sq_norm = float(np.dot(x_arr, x_arr))
            if not np.isfinite(sq_norm) or sq_norm > 1e100:
                return -np.sign(x_arr) * 1e100
            if sq_norm == 0.0:
                return np.zeros(self.d, dtype=np.float64)
            norm = np.sqrt(sq_norm)
            # Safeguard against extreme floating point overflow
            if norm > 1e50:
                return -np.sign(x_arr) * 1e100
            factor = (norm ** (self.p - 2.0)) / self.scale
            return -factor * x_arr

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            sq_norm = float(np.dot(x_arr, x_arr))
            if not np.isfinite(sq_norm) or sq_norm > 1e100:
                return -np.inf, -np.sign(x_arr) * 1e100
            if sq_norm == 0.0:
                return float(-self._log_Z), np.zeros(self.d, dtype=np.float64)
            norm = np.sqrt(sq_norm)
            u = (norm**self.p) / (self.p * self.scale)
            factor = (norm ** (self.p - 2.0)) / self.scale
            return float(-u - self._log_Z), -factor * x_arr

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Spherical decomposition: u ~ Uniform(S^{d-1}), Y ~ Gamma(shape=d/p, scale=1)
        z = rng.normal(size=(n, self.d))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        # Handle zero-probability exact 0 norm
        norms = np.where(norms == 0.0, 1.0, norms)
        u = z / norms

        y = rng.gamma(shape=self.d / self.p, scale=1.0, size=n)
        r = (self.p * self.scale * y) ** (1.0 / self.p)
        return u * r[:, None]

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        # E[||X||^2] = (p * scale)^(2/p) * Gamma((d + 2)/p) / Gamma(d/p)
        expected_sq_norm = (self.p * self.scale) ** (2.0 / self.p) * np.exp(
            special.gammaln((self.d + 2.0) / self.p) - special.gammaln(self.d / self.p)
        )
        coord_var = expected_sq_norm / self.d
        res: dict[str, Any] = {
            "mean": np.zeros(self.d, dtype=np.float64),
            "variance": np.full(self.d, coord_var, dtype=np.float64),
            "cov": coord_var * np.eye(self.d, dtype=np.float64),
            "expected_sq_norm": expected_sq_norm,
        }
        if self.d == 1 and self.p == 4.0 and self.scale == 1.0:
            # 1D marginal CDF: F(x) = 0.5 + 0.5 * sign(x) * gammainc(0.25, x^4 / 4)
            # Find analytical quantiles via root finding or numerical inverse
            # gammainc(0.25, z) = 2 * (p - 0.5)
            # For 95th percentile (p = 0.95): gammainc(0.25, z) = 0.90 => z = special.gammaincinv(0.25, 0.90)
            z95 = special.gammaincinv(0.25, 0.90)
            q95 = (4.0 * z95) ** 0.25
            z99 = special.gammaincinv(0.25, 0.98)
            q99 = (4.0 * z99) ** 0.25
            res["q05"] = np.array([-q95])
            res["q50"] = np.array([0.0])
            res["q95"] = np.array([q95])
            res["q99"] = np.array([q99])
        return res


class SuperlinearPolynomialTarget(QuarticTarget):
    """Superlinear polynomial potential with arbitrary degree p (default p=6)."""

    def __init__(
        self,
        d: int,
        p: float = 6.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(d=d, p=p, scale=scale)


# Alias
SuperlinearTarget = SuperlinearPolynomialTarget


class L0L1SmoothTarget(TargetDistribution):
    """(L0, L1)-smooth potential U(x) = (L0 / 2) * ||x||^2 + (lambda / 4) * ||x||^4."""

    def __init__(
        self,
        d: int,
        L0: float = 1.0,
        lambda_param: float = 1.0,
    ) -> None:
        super().__init__(d=d)
        if L0 <= 0.0:
            raise ValueError(f"L0 must be positive, got {L0}")
        if lambda_param <= 0.0:
            raise ValueError(f"lambda_param must be positive, got {lambda_param}")
        self.L0: float = float(L0)
        self.lambda_param: float = float(lambda_param)
        self.L1: float = float(3.0 * np.sqrt(self.lambda_param))

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf
        with np.errstate(over="ignore", invalid="ignore"):
            sq_norm = float(np.dot(x_arr, x_arr))
            if not np.isfinite(sq_norm) or sq_norm > 1e100:
                return -np.inf
            u = 0.5 * self.L0 * sq_norm + 0.25 * self.lambda_param * (sq_norm**2)
            return float(-u)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return np.full(self.d, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            sq_norm = float(np.dot(x_arr, x_arr))
            if not np.isfinite(sq_norm) or sq_norm > 1e50:
                return -np.sign(x_arr) * 1e100
            factor = self.L0 + self.lambda_param * sq_norm
            return -factor * x_arr

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            sq_norm = float(np.dot(x_arr, x_arr))
            if not np.isfinite(sq_norm) or sq_norm > 1e100:
                return -np.inf, -np.sign(x_arr) * 1e100
            u = 0.5 * self.L0 * sq_norm + 0.25 * self.lambda_param * (sq_norm**2)
            factor = self.L0 + self.lambda_param * sq_norm
            return float(-u), -factor * x_arr

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        r"""Draw n exact, independent ground truth samples in O(1) expected steps.

        Algorithm:
        1. Spherically symmetric decomposition: x = r * u, u ~ Unif(S^{d-1}).
        2. Squared radius R = r^2 has density p(R) \propto R^alpha exp(-L0/2 R - lambda/4 R^2).
        3. For d=1: Gaussian envelope rejection on R or direct 1D.
        4. For d=2: R ~ TruncatedNormal(-L0/lambda, 2/lambda, [0, inf)), exact & rejection-free.
        5. For d > 2: Mode-centered TruncatedNormal envelope rejection in R, with
           exact algebraic acceptance probability alpha_acc = (R/R*)^alpha exp(-alpha (R/R* - 1)) <= 1.
           Acceptance rate is ~65-71% uniformly for all d >= 2.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.d == 1:
            samples = np.empty((n, 1), dtype=np.float64)
            count = 0
            while count < n:
                batch = max(n - count, 100)
                cand = rng.normal(scale=1.0 / np.sqrt(self.L0), size=batch)
                accept_prob = np.exp(-0.25 * self.lambda_param * (cand**4))
                acc = cand[rng.uniform(size=batch) <= accept_prob]
                take = min(n - count, len(acc))
                samples[count : count + take, 0] = acc[:take]
                count += take
            return samples

        alpha = 0.5 * (self.d - 2)
        sig = np.sqrt(2.0 / self.lambda_param)

        if alpha == 0.0:
            # d = 2: Exactly Truncated Normal on [0, inf) with mu = -L0/lambda, sig^2 = 2/lambda
            mu = -self.L0 / self.lambda_param
            a_std = -mu / sig
            phi_a = 0.5 * (1.0 + special.erf(a_std / np.sqrt(2.0)))
            u = rng.uniform(phi_a, 1.0, size=n)
            z = special.ndtri(u)
            R = mu + sig * z
        else:
            # d > 2: Mode-centered Truncated Normal envelope rejection
            # Mode R* from quadratic equation: lambda (R*)^2 + L0 R* - 2 alpha = 0
            R_star = (-self.L0 + np.sqrt(self.L0**2 + 8.0 * self.lambda_param * alpha)) / (2.0 * self.lambda_param)
            a_std = -R_star / sig
            phi_a = 0.5 * (1.0 + special.erf(a_std / np.sqrt(2.0)))

            R = np.empty(n, dtype=np.float64)
            count = 0

            while count < n:
                cur_batch = max(int((n - count) * 1.6), 100)
                u = rng.uniform(phi_a, 1.0, size=cur_batch)
                z = special.ndtri(u)
                cand_R = np.maximum(R_star + sig * z, 1e-300)

                # Exact algebraic acceptance ratio: (cand_R / R_star)^alpha * exp(-alpha * (cand_R / R_star - 1))
                ratio = cand_R / R_star
                log_acc = alpha * (np.log(ratio) - (ratio - 1.0))
                log_acc = np.clip(log_acc, -500.0, 0.0)
                acc_mask = rng.uniform(size=cur_batch) <= np.exp(log_acc)

                accepted = cand_R[acc_mask]
                take = min(n - count, len(accepted))
                R[count : count + take] = accepted[:take]
                count += take

        r = np.sqrt(R)

        # Uniform draws on S^{d-1} via normalized standard Gaussian
        z_sph = rng.normal(size=(n, self.d))
        norm_z = np.linalg.norm(z_sph, axis=1, keepdims=True)
        norm_z = np.where(norm_z == 0.0, 1.0, norm_z)
        u_sph = z_sph / norm_z

        return u_sph * r[:, None]

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "mean": np.zeros(self.d, dtype=np.float64),
            "L0": self.L0,
            "L1": self.L1,
        }


class NealsFunnelTarget(TargetDistribution):
    """Neal's funnel: x_0 ~ N(0, sigma_v^2), x_{1:d} | x_0 ~ N(0, exp(x_0) I_{d-1})."""

    def __init__(
        self,
        d: int,
        sigma_v: float = 3.0,
    ) -> None:
        if d < 2:
            raise ValueError(f"Neal's funnel requires dimension d >= 2, got {d}")
        super().__init__(d=d)
        if sigma_v <= 0.0:
            raise ValueError(f"sigma_v must be positive, got {sigma_v}")
        self.sigma_v: float = float(sigma_v)

    LOG_QUAD_MAX: float = 700.0
    LARGE_GRAD: float = 1e300

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf
        with np.errstate(over="ignore", invalid="ignore"):
            v = float(x_arr[0])
            rest = x_arr[1:]
            sq_rest = float(np.dot(rest, rest))
            if not np.isfinite(sq_rest) or sq_rest > 1e100:
                return -np.inf

            lp_v = -0.5 * np.log(2.0 * np.pi) - np.log(self.sigma_v) - 0.5 * (v**2) / (self.sigma_v**2)
            lp_det = -0.5 * (self.d - 1) * np.log(2.0 * np.pi) - 0.5 * (self.d - 1) * v

            if sq_rest == 0.0:
                return float(lp_v + lp_det)

            # Check for overflow in 0.5 * exp(-v) * sq_rest
            log_quad = -np.log(2.0) - v + np.log(sq_rest)
            if log_quad > self.LOG_QUAD_MAX:
                return -np.inf

            exp_neg_v = 0.0 if (v > 740.0 or v < -709.0) else np.exp(-v)
            lp_quad = -0.5 * exp_neg_v * sq_rest
            return float(lp_v + lp_det + lp_quad)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return np.full(self.d, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            v = float(x_arr[0])
            rest = x_arr[1:]
            sq_rest = float(np.dot(rest, rest))
            if not np.isfinite(sq_rest) or sq_rest > 1e100:
                grad = np.empty(self.d, dtype=np.float64)
                grad[0] = self.LARGE_GRAD
                grad[1:] = -np.sign(rest) * self.LARGE_GRAD
                return grad

            grad = np.empty(self.d, dtype=np.float64)
            base_grad_v = -v / (self.sigma_v**2) - 0.5 * (self.d - 1)

            if sq_rest == 0.0:
                grad[0] = base_grad_v
                grad[1:] = 0.0
                return grad

            log_quad = -np.log(2.0) - v + np.log(sq_rest)
            if log_quad > self.LOG_QUAD_MAX:
                grad[0] = self.LARGE_GRAD
                grad[1:] = -np.sign(rest) * self.LARGE_GRAD
                return grad

            exp_neg_v = 0.0 if (v > 740.0 or v < -709.0) else np.exp(-v)
            grad[0] = base_grad_v + 0.5 * exp_neg_v * sq_rest
            grad[1:] = -exp_neg_v * rest
            return grad

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            v = float(x_arr[0])
            rest = x_arr[1:]
            sq_rest = float(np.dot(rest, rest))
            if not np.isfinite(sq_rest) or sq_rest > 1e100:
                grad = np.empty(self.d, dtype=np.float64)
                grad[0] = self.LARGE_GRAD
                grad[1:] = -np.sign(rest) * self.LARGE_GRAD
                return -np.inf, grad

            lp_v = -0.5 * np.log(2.0 * np.pi) - np.log(self.sigma_v) - 0.5 * (v**2) / (self.sigma_v**2)
            lp_det = -0.5 * (self.d - 1) * np.log(2.0 * np.pi) - 0.5 * (self.d - 1) * v
            base_grad_v = -v / (self.sigma_v**2) - 0.5 * (self.d - 1)

            grad = np.empty(self.d, dtype=np.float64)

            if sq_rest == 0.0:
                grad[0] = base_grad_v
                grad[1:] = 0.0
                return float(lp_v + lp_det), grad

            log_quad = -np.log(2.0) - v + np.log(sq_rest)
            if log_quad > self.LOG_QUAD_MAX:
                grad[0] = self.LARGE_GRAD
                grad[1:] = -np.sign(rest) * self.LARGE_GRAD
                return -np.inf, grad

            exp_neg_v = 0.0 if (v > 740.0 or v < -709.0) else np.exp(-v)
            lp_quad = -0.5 * exp_neg_v * sq_rest
            grad[0] = base_grad_v + 0.5 * exp_neg_v * sq_rest
            grad[1:] = -exp_neg_v * rest
            return float(lp_v + lp_det + lp_quad), grad

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        v = rng.normal(loc=0.0, scale=self.sigma_v, size=n)
        rest = rng.normal(size=(n, self.d - 1)) * np.exp(0.5 * v[:, None])
        return np.column_stack([v, rest])

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        # E[x_0] = 0, Var(x_0) = sigma_v^2
        # E[x_j] = 0, Var(x_j) = E[exp(v)] = exp(0.5 * sigma_v^2)
        var_rest = np.exp(0.5 * (self.sigma_v**2))
        variances = np.full(self.d, var_rest, dtype=np.float64)
        variances[0] = self.sigma_v**2
        return {
            "mean": np.zeros(self.d, dtype=np.float64),
            "variance": variances,
            "v_std": self.sigma_v,
            "q05_v": stats.norm.ppf(0.05, loc=0.0, scale=self.sigma_v),
            "q95_v": stats.norm.ppf(0.95, loc=0.0, scale=self.sigma_v),
        }


class RippledGaussianTarget(TargetDistribution):
    """Spatially oscillating Gaussian U(x) = (1 / (2 sigma^2)) * ||x||^2 + A * sum(cos(omega * x_i))."""

    def __init__(
        self,
        d: int,
        sigma: float = 1.0,
        amplitude: float = 0.5,
        omega: float = 10.0,
        A: float | None = None,
        frequency: float | None = None,
    ) -> None:
        super().__init__(d=d)
        if A is not None:
            amplitude = A
        if frequency is not None:
            omega = frequency
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma: float = float(sigma)
        self.amplitude: float = float(amplitude)
        self.omega: float = float(omega)

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        sq_norm = float(np.dot(x_arr, x_arr))
        cos_sum = float(np.sum(np.cos(self.omega * x_arr)))
        u = 0.5 * sq_norm / (self.sigma**2) + self.amplitude * cos_sum
        return float(-u)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        grad = -x_arr / (self.sigma**2) + self.amplitude * self.omega * np.sin(self.omega * x_arr)
        return grad

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        sq_norm = float(np.dot(x_arr, x_arr))
        cos_sum = float(np.sum(np.cos(self.omega * x_arr)))
        u = 0.5 * sq_norm / (self.sigma**2) + self.amplitude * cos_sum
        grad = -x_arr / (self.sigma**2) + self.amplitude * self.omega * np.sin(self.omega * x_arr)
        return float(-u), grad

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Coordinates are independent; rejection sampling with Gaussian proposal
        samples = np.empty((n, self.d), dtype=np.float64)
        for i in range(self.d):
            coord_samples: list[float] = []
            while len(coord_samples) < n:
                batch = rng.normal(loc=0.0, scale=self.sigma, size=max(n - len(coord_samples), 1000))
                # Accept prob = exp(-A * (1 + cos(omega * z)))
                prob = np.exp(-self.amplitude * (1.0 + np.cos(self.omega * batch)))
                unif = rng.uniform(size=len(batch))
                accepted = batch[unif <= prob]
                coord_samples.extend(accepted)
            samples[:, i] = np.array(coord_samples[:n])
        return samples

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "mean": np.zeros(self.d, dtype=np.float64),
            "amplitude": self.amplitude,
            "omega": self.omega,
            "sigma": self.sigma,
        }


# Alias
OscillatingGaussianTarget = RippledGaussianTarget
