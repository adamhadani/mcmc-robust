"""Heavy-tailed and heterogeneous flatness target distributions."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import linalg, special, stats

from mcmc_robust.core.target import TargetDistribution


class MultivariateStudentTTarget(TargetDistribution):
    """Multivariate Student-t distribution with degrees of freedom nu, location mu, scale Sigma."""

    def __init__(
        self,
        d: int,
        nu: float = 2.5,
        loc: np.ndarray | None = None,
        mu: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        Sigma: np.ndarray | None = None,
    ) -> None:
        super().__init__(d=d)
        if nu <= 0.0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.nu: float = float(nu)

        # Handle location aliases
        if mu is not None:
            loc = mu
        if loc is None:
            self.mu: np.ndarray = np.zeros(self.d, dtype=np.float64)
        else:
            m = np.asarray(loc, dtype=np.float64)
            if m.shape != (self.d,):
                raise ValueError(f"loc shape {m.shape} must match ({self.d},)")
            self.mu = m

        # Handle scale / covariance aliases
        if Sigma is not None:
            scale = Sigma
        elif cov is not None:
            scale = cov

        if scale is None:
            self._cov: np.ndarray = np.eye(self.d, dtype=np.float64)
        else:
            s_arr = np.asarray(scale, dtype=np.float64)
            if s_arr.ndim == 2:
                if s_arr.shape != (self.d, self.d):
                    raise ValueError(f"scale shape {s_arr.shape} must match ({self.d}, {self.d})")
                self._cov = s_arr
            elif s_arr.ndim == 1:
                if s_arr.shape != (self.d,):
                    raise ValueError(f"scale shape {s_arr.shape} must match ({self.d},)")
                self._cov = np.diag(s_arr**2)
            else:
                raise ValueError(f"Unsupported scale shape {s_arr.shape}")

        try:
            self.L: np.ndarray = linalg.cholesky(self._cov, lower=True)
        except linalg.LinAlgError:
            self._cov += 1e-8 * np.eye(self.d)
            self.L = linalg.cholesky(self._cov, lower=True)

        self._log_det: float = float(np.sum(np.log(np.diag(self.L))))
        # Precompute constant:
        # log_C = gammaln((nu + d)/2) - gammaln(nu/2) - (d/2)*log(nu * pi) - sum(log(L_ii))
        self._log_const: float = float(
            special.gammaln((self.nu + self.d) / 2.0)
            - special.gammaln(self.nu / 2.0)
            - (self.d / 2.0) * np.log(self.nu * np.pi)
            - self._log_det
        )

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        diff = x_arr - self.mu
        z = linalg.solve_triangular(self.L, diff, lower=True)
        delta_sq = float(np.dot(z, z))
        # log(1 + delta_sq / nu) using log1p
        log_term = np.log1p(delta_sq / self.nu)
        return float(self._log_const - 0.5 * (self.nu + self.d) * log_term)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mu
        z = linalg.solve_triangular(self.L, diff, lower=True)
        delta_sq = float(np.dot(z, z))
        w = linalg.solve_triangular(self.L.T, z, lower=False)
        factor = (self.nu + self.d) / (self.nu + delta_sq)
        return -factor * w

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        diff = x_arr - self.mu
        z = linalg.solve_triangular(self.L, diff, lower=True)
        delta_sq = float(np.dot(z, z))
        w = linalg.solve_triangular(self.L.T, z, lower=False)
        log_term = np.log1p(delta_sq / self.nu)
        lp = float(self._log_const - 0.5 * (self.nu + self.d) * log_term)
        factor = (self.nu + self.d) / (self.nu + delta_sq)
        return lp, -factor * w

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Scale mixture: xi ~ N(0, I_d), w ~ Gamma(nu/2, 2/nu), x = mu + (L * xi) / sqrt(w)
        xi = rng.normal(size=(n, self.d))
        w = rng.gamma(shape=self.nu / 2.0, scale=2.0 / self.nu, size=n)
        z = xi @ self.L.T
        return self.mu + z / np.sqrt(w[:, None])

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        marginal_scales = np.sqrt(np.diag(self._cov))
        res: dict[str, Any] = {
            "mean": self.mu.copy(),
            "nu": self.nu,
            "q05": self.mu + marginal_scales * stats.t.ppf(0.05, df=self.nu),
            "q50": self.mu.copy(),
            "q95": self.mu + marginal_scales * stats.t.ppf(0.95, df=self.nu),
            "q99": self.mu + marginal_scales * stats.t.ppf(0.99, df=self.nu),
        }
        if self.nu > 2.0:
            cov_mat = (self.nu / (self.nu - 2.0)) * self._cov
            res["cov"] = cov_mat
            res["variance"] = np.diag(cov_mat)
        return res


# Alias
StudentTTarget = MultivariateStudentTTarget


class HeterogeneousHeavyTailedTarget(TargetDistribution):
    """Product-form target mixing Student-t (heavy-tailed) and Gaussian (light-tailed) coordinates."""

    def __init__(
        self,
        d: int,
        heavy_coords: Sequence[int] | None = None,
        nu_list: Sequence[float] | None = None,
        scale_list: Sequence[float] | None = None,
        mu: np.ndarray | None = None,
    ) -> None:
        super().__init__(d=d)
        if mu is None:
            self.mu: np.ndarray = np.zeros(self.d, dtype=np.float64)
        else:
            m = np.asarray(mu, dtype=np.float64)
            if m.shape != (self.d,):
                raise ValueError(f"mu shape {m.shape} must match ({self.d},)")
            self.mu = m

        if heavy_coords is None:
            # Default: first floor(d/2) coordinates are heavy
            d1 = max(1, self.d // 2)
            self.heavy_coords: list[int] = list(range(d1))
        else:
            self.heavy_coords = [int(i) for i in heavy_coords]

        self.heavy_set = set(self.heavy_coords)
        self.light_coords: list[int] = [i for i in range(self.d) if i not in self.heavy_set]

        # Heavy degrees of freedom
        n_heavy = len(self.heavy_coords)
        if nu_list is None:
            if n_heavy == 1:
                self.nu_arr = np.array([2.0], dtype=np.float64)
            else:
                self.nu_arr = np.linspace(1.5, 2.5, n_heavy)
        else:
            self.nu_arr = np.asarray(nu_list, dtype=np.float64)

        # Scale list
        if scale_list is None:
            self.scale_arr = np.ones(self.d, dtype=np.float64)
        else:
            self.scale_arr = np.asarray(scale_list, dtype=np.float64)

        # Precompute constants for heavy coordinates
        self._heavy_log_consts = np.zeros(n_heavy, dtype=np.float64)
        for idx, (c, nu) in enumerate(zip(self.heavy_coords, self.nu_arr)):
            s = self.scale_arr[c]
            const = special.gammaln((nu + 1.0) / 2.0) - special.gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi) - np.log(s)
            self._heavy_log_consts[idx] = const

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        lp = 0.0

        # Heavy coordinates
        for idx, c in enumerate(self.heavy_coords):
            nu = self.nu_arr[idx]
            s = self.scale_arr[c]
            diff = x_arr[c] - self.mu[c]
            z_sq = (diff / s) ** 2
            log_term = np.log1p(z_sq / nu)
            lp += self._heavy_log_consts[idx] - 0.5 * (nu + 1.0) * log_term

        # Light coordinates (standard normal scaled by scale_arr)
        for c in self.light_coords:
            s = self.scale_arr[c]
            diff = x_arr[c] - self.mu[c]
            z_sq = (diff / s) ** 2
            lp += -0.5 * np.log(2.0 * np.pi) - np.log(s) - 0.5 * z_sq

        return float(lp)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        grad = np.empty(self.d, dtype=np.float64)

        # Heavy coordinates: grad_i = - (nu_i + 1) / (nu_i * s_i^2 + (x_i - mu_i)^2) * (x_i - mu_i)
        for idx, c in enumerate(self.heavy_coords):
            nu = self.nu_arr[idx]
            s = self.scale_arr[c]
            diff = x_arr[c] - self.mu[c]
            denom = nu * (s**2) + (diff**2)
            grad[c] = -((nu + 1.0) / denom) * diff

        # Light coordinates: grad_j = - (x_j - mu_j) / s_j^2
        for c in self.light_coords:
            s = self.scale_arr[c]
            diff = x_arr[c] - self.mu[c]
            grad[c] = -diff / (s**2)

        return grad

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return self.log_prob(x), self.grad_log_prob(x)

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        samples = np.empty((n, self.d), dtype=np.float64)

        for idx, c in enumerate(self.heavy_coords):
            nu = self.nu_arr[idx]
            s = self.scale_arr[c]
            samples[:, c] = self.mu[c] + s * rng.standard_t(df=nu, size=n)

        for c in self.light_coords:
            s = self.scale_arr[c]
            samples[:, c] = self.mu[c] + s * rng.normal(size=n)

        return samples

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        q05 = np.empty(self.d, dtype=np.float64)
        q95 = np.empty(self.d, dtype=np.float64)
        q99 = np.empty(self.d, dtype=np.float64)

        for idx, c in enumerate(self.heavy_coords):
            nu = self.nu_arr[idx]
            s = self.scale_arr[c]
            q05[c] = self.mu[c] + s * stats.t.ppf(0.05, df=nu)
            q95[c] = self.mu[c] + s * stats.t.ppf(0.95, df=nu)
            q99[c] = self.mu[c] + s * stats.t.ppf(0.99, df=nu)

        for c in self.light_coords:
            s = self.scale_arr[c]
            q05[c] = self.mu[c] + s * stats.norm.ppf(0.05)
            q95[c] = self.mu[c] + s * stats.norm.ppf(0.95)
            q99[c] = self.mu[c] + s * stats.norm.ppf(0.99)

        return {
            "mean": self.mu.copy(),
            "heavy_coords": list(self.heavy_coords),
            "light_coords": list(self.light_coords),
            "q05": q05,
            "q50": self.mu.copy(),
            "q95": q95,
            "q99": q99,
        }


# Alias
HeterogeneousStudentTTarget = HeterogeneousHeavyTailedTarget


class RotatedProductCauchyTarget(TargetDistribution):
    r"""Rotated product of Cauchy variables: $x = A u$ with $A = Q \mathrm{diag}(s)$ and $u_j$ iid standard Cauchy.

    The heavy tails lie along the columns of $A$, not the coordinate axes, and they are not elliptical:
    one $u_j$ at a time is extreme. By stability of the Cauchy law, marginal $i$ is exactly
    Cauchy$(0, \sum_j |A_{ij}|)$. $Q$ is the fixed-seed random orthogonal matrix of the QR factorisation
    of a standard Gaussian matrix.
    """

    def __init__(self, d: int = 10, scales: Sequence[float] | None = None, rotation_seed: int = 0) -> None:
        super().__init__(d=d)
        s = np.geomspace(1.0, 5.0, self.d) if scales is None else np.asarray(scales, dtype=np.float64)
        if s.shape != (self.d,) or np.any(s <= 0.0):
            raise ValueError(f"scales must be {self.d} positive numbers")
        q, _ = np.linalg.qr(np.random.default_rng(rotation_seed).normal(size=(self.d, self.d)))
        self.A: np.ndarray = q * s
        self._A_inv = np.linalg.inv(self.A)
        self._log_norm = -float(np.linalg.slogdet(self.A)[1]) - self.d * float(np.log(np.pi))
        self.marginal_scales: np.ndarray = np.sum(np.abs(self.A), axis=1)

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = self._validate_input(x)
        u = self._A_inv @ x_arr
        lp = self._log_norm - float(np.sum(np.log1p(u**2)))
        return lp, self._A_inv.T @ (-2.0 * u / (1.0 + u**2))

    def log_prob(self, x: np.ndarray) -> float:
        return self.log_prob_and_grad(x)[0]

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return self.log_prob_and_grad(x)[1]

    def jax_log_prob(self, x: Any) -> Any:
        """The same log density for JAX (used by `SOTATargetAdapter.jax_logdensity_fn`)."""
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        u = jnp.asarray(self._A_inv) @ x
        return self._log_norm - jnp.sum(jnp.log1p(u**2))

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        gen = rng if rng is not None else np.random.default_rng()
        return gen.standard_cauchy(size=(n, self.d)) @ self.A.T

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        def q(p: float) -> np.ndarray:
            return self.marginal_scales * np.tan(np.pi * (p - 0.5))

        return {
            "q05": q(0.05),
            "q50": np.zeros(self.d),
            "q95": q(0.95),
            "q99": q(0.99),
            "marginal_scales": self.marginal_scales,
        }
