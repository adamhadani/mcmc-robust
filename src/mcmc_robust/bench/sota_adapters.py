"""External SOTA MCMC Adapters and ArviZ 1.3 Diagnostic Pipeline (Feature F27).

Provides:
- Target adapters converting TargetDistribution into JAX log-density functions and PyMC models.
- Native applied Bayesian targets (EightSchoolsTarget, SparseSignalRegressionTarget, RobustOutlierRegressionTarget).
- Execution runners for BlackJAX (NUTS, HMC, MCLMC) with exact gradient evaluation accounting via info.num_integration_steps.
- Execution runner for PyMC NUTS with exact leapfrog accounting via sample_stats['n_steps'].
- Execution runner for novel mcmc_robust samplers (UnifiedRobustNUTS, UnifiedRobustHMC, UnifiedRobustSampler, etc.)
  with zero-redundancy oracle accounting via GradBudgetTracker.
- ArviZ 1.3 evaluation pipeline computing Bulk-ESS, Tail-ESS, split R-hat, ESS/grad, Tail-ESS/grad, and quantile metrics.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

# Enable JAX 64-bit float precision before importing jax.numpy
import jax
import numpy as np
from scipy import special

jax.config.update("jax_enable_x64", True)
import arviz as az
import blackjax
import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt

from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.diagnostics.metrics import compute_quantile_relative_error
from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.targets.benign import (
    AnisotropicGaussianTarget,
    DiagonalGaussian,
    FullCovarianceGaussian,
    GaussianTarget,
    IsotropicGaussian,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    HeterogeneousStudentTTarget,
    MultivariateStudentTTarget,
    StudentTTarget,
)
from mcmc_robust.targets.roughness import (
    NealsFunnelTarget,
    QuarticTarget,
)

# =============================================================================
# 1. Real-World Applied Bayesian Target Distributions
# =============================================================================

# Canonical Rubin (1981) Eight-Schools ETS meta-analysis dataset
Y_EIGHT_SCHOOLS = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], dtype=np.float64)
SIGMA_EIGHT_SCHOOLS = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], dtype=np.float64)


class EightSchoolsTarget(TargetDistribution):
    r"""Rubin's (1981) Eight-Schools ETS Meta-Analysis (Centered Parameterization).

    State space: R^10 with coordinates x = [\mu, \eta, \theta_0, \dots, \theta_7]
    where \eta = \log \tau \in R is the unconstrained between-school log standard deviation.

    Priors:
        \mu \sim N(0, 10^2)
        \tau \sim \text{Half-Cauchy}(0, 5) \implies p(\eta) = \frac{2 e^\eta}{5\pi (1 + (e^\eta/5)^2)}
        \theta_j \mid \mu, \eta \sim N(\mu, e^{2\eta}) \quad (j=0,\dots,7)

    Likelihood:
        y_j \mid \theta_j \sim N(\theta_j, \sigma_j^2) \quad (j=0,\dots,7)

    Pathology:
        As \eta \to -\infty (\tau \to 0), the conditional variance collapses as e^{2\eta},
        inducing an acute funnel neck with curvature \sim e^{-2\eta} \to \infty.
        Standard NUTS suffers divergent transitions (\Delta H > 1000) or fails to explore
        the deep neck (\eta < -2.0).
    """

    def __init__(
        self,
        y: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        mu_prior_sigma: float = 10.0,
        tau_prior_scale: float = 5.0,
    ) -> None:
        super().__init__(d=10)
        self.y = np.asarray(Y_EIGHT_SCHOOLS if y is None else y, dtype=np.float64)
        self.sigma = np.asarray(SIGMA_EIGHT_SCHOOLS if sigma is None else sigma, dtype=np.float64)
        self.n_schools = len(self.y)
        if self.sigma.shape[0] != self.n_schools or self.n_schools != 8:
            raise ValueError(f"Eight schools requires exactly 8 observations, got {self.n_schools}")
        self.mu_prior_sigma = float(mu_prior_sigma)
        self.tau_prior_scale = float(tau_prior_scale)

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        mu = float(x_arr[0])
        eta = float(x_arr[1])
        theta = x_arr[2:]

        # Numerical bounds on log_tau = eta
        if eta > 100.0 or eta < -100.0:
            return -np.inf

        tau_sq = np.exp(2.0 * eta)

        # 1. Prior on mu
        lp_mu = -0.5 * np.log(2.0 * np.pi * (self.mu_prior_sigma**2)) - 0.5 * (mu**2) / (self.mu_prior_sigma**2)

        # 2. Prior on eta = log(tau)
        scale = self.tau_prior_scale
        lp_eta = np.log(2.0 / (scale * np.pi)) + eta - np.log1p(np.exp(2.0 * eta) / (scale**2))

        # 3. Conditional prior on theta_j | mu, eta
        diff_theta = theta - mu
        lp_theta = (
            -0.5 * self.n_schools * np.log(2.0 * np.pi) - self.n_schools * eta - 0.5 * np.sum(diff_theta**2) / tau_sq
        )

        # 4. Likelihood on y_j | theta_j
        diff_y = self.y - theta
        lp_y = -0.5 * np.sum(np.log(2.0 * np.pi * (self.sigma**2))) - 0.5 * np.sum((diff_y**2) / (self.sigma**2))

        return float(lp_mu + lp_eta + lp_theta + lp_y)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        mu = float(x_arr[0])
        eta = float(x_arr[1])
        theta = x_arr[2:]

        if eta > 100.0 or eta < -100.0:
            return np.full(self.d, np.nan, dtype=np.float64)

        tau_sq = np.exp(2.0 * eta)
        scale_sq = self.tau_prior_scale**2
        diff_theta = theta - mu
        diff_y = self.y - theta

        grad = np.empty(self.d, dtype=np.float64)
        # d/dmu
        grad[0] = -mu / (self.mu_prior_sigma**2) + np.sum(diff_theta) / tau_sq
        # d/deta
        grad_eta_prior = 1.0 - (2.0 * tau_sq) / (scale_sq + tau_sq)
        grad_eta_theta = -float(self.n_schools) + np.sum(diff_theta**2) / tau_sq
        grad[1] = grad_eta_prior + grad_eta_theta
        # d/dtheta_j
        grad[2:] = -diff_theta / tau_sq + diff_y / (self.sigma**2)

        return grad

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draws reference prior predictive samples."""
        if rng is None:
            rng = np.random.default_rng(42)
        samples = np.empty((n, self.d), dtype=np.float64)
        mu = rng.normal(0.0, self.mu_prior_sigma, size=n)
        tau = np.abs(rng.standard_cauchy(size=n)) * self.tau_prior_scale
        samples[:, 0] = mu
        samples[:, 1] = np.log(np.maximum(tau, 1e-10))
        for i in range(n):
            samples[i, 2:] = rng.normal(mu[i], tau[i], size=self.n_schools)
        return samples

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        """Provides analytical reference summary."""
        return {
            "n_schools": self.n_schools,
            "y": self.y,
            "sigma": self.sigma,
            "mu_prior_sigma": self.mu_prior_sigma,
            "tau_prior_scale": self.tau_prior_scale,
        }


class SparseSignalRegressionTarget(TargetDistribution):
    r"""Sparse Signal Recovery / Biomarker Regression with Heavy-Tailed Shrinkage Priors.

    Evaluates regression in dimension d=20 with N=50 observations:
        y = X \beta^* + \epsilon, \quad \epsilon \sim N(0, \sigma^2 I)
    True signal \beta^* is sparse (3 non-zero active coordinates, 17 inactive zeros).

    Prior:
        Each \beta_j \sim \text{Cauchy}(0, \lambda).

    Pathology:
        For inactive coordinates, the restoring likelihood force is weak, while the Cauchy
        prior has polynomial tails \propto 1 / (1 + \beta_j^2). Leapfrog trajectories drifting
        into tails of inactive coordinates experience vanishing restoring forces, triggering
        NUTS binary tree-depth blowup to max_depth.
    """

    def __init__(
        self,
        d: int = 20,
        n_obs: int = 50,
        n_active: int = 3,
        noise_std: float = 0.5,
        prior_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(d=d)
        self.n_obs = int(n_obs)
        self.n_active = int(n_active)
        self.noise_std = float(noise_std)
        self.prior_scale = float(prior_scale)

        rng = np.random.default_rng(seed)
        self.X = rng.normal(0.0, 1.0, size=(self.n_obs, self.d)).astype(np.float64)

        # True sparse signal
        self.beta_true = np.zeros(self.d, dtype=np.float64)
        active_vals = [3.0, -2.5, 2.0]
        for idx in range(min(self.n_active, len(active_vals))):
            self.beta_true[idx] = active_vals[idx]

        # Generate observations
        self.y = (self.X @ self.beta_true + rng.normal(0.0, self.noise_std, size=self.n_obs)).astype(np.float64)

        # Precompute X^T X and X^T y
        self.XtX = self.X.T @ self.X
        self.Xty = self.X.T @ self.y
        self.inv_var = 1.0 / (self.noise_std**2)

    def log_prob(self, beta: np.ndarray) -> float:
        b = np.asarray(beta, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            return -np.inf

        res = self.y - self.X @ b
        lp_lik = -0.5 * self.n_obs * np.log(2.0 * np.pi * (self.noise_std**2)) - 0.5 * self.inv_var * np.dot(res, res)
        # Cauchy prior log-prob: -log(pi * s) - log(1 + (b / s)^2)
        scaled_b = b / self.prior_scale
        abs_scaled = np.abs(scaled_b)
        scaled_b_safe = np.clip(scaled_b, -1e100, 1e100)
        log_terms = np.where(
            abs_scaled > 1e100,
            2.0 * np.log(np.maximum(abs_scaled, 1.0)),
            np.log1p(scaled_b_safe**2),
        )
        lp_prior = -self.d * np.log(np.pi * self.prior_scale) - np.sum(log_terms)
        return float(lp_lik + lp_prior)

    def grad_log_prob(self, beta: np.ndarray) -> np.ndarray:
        b = np.asarray(beta, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            return np.full(self.d, np.nan, dtype=np.float64)

        res = self.y - self.X @ b
        grad_lik = self.inv_var * (self.X.T @ res)
        abs_b = np.abs(b)
        scale_sq = self.prior_scale**2
        b_safe = np.clip(b, -1e100, 1e100)
        grad_prior = np.where(
            abs_b > 1e100,
            -2.0 / np.where(abs_b == 0.0, 1.0, b),
            -2.0 * b_safe / (scale_sq + b_safe**2),
        )
        return grad_lik + grad_prior

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.tile(self.beta_true, (n, 1))

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "beta_true": self.beta_true,
            "mean": self.beta_true,
            "n_active": self.n_active,
            "noise_std": self.noise_std,
        }


class RobustOutlierRegressionTarget(TargetDistribution):
    r"""Robust Bayesian Regression / GLM with Outlier Contamination.

    Evaluates regression in dimension d=10 with N=60 observations where 10% of
    observations are severe outliers. Uses heavy-tailed Student-t likelihood (\nu=2.0)
    and Cauchy shrinkage priors.

    Likelihood:
        y_i \mid \beta \sim \text{Student-t}(\nu=2.0, \text{loc}=x_i^T \beta, \text{scale}=\sigma)

    Prior:
        \beta_j \sim \text{Cauchy}(0, \lambda)
    """

    def __init__(
        self,
        d: int = 10,
        n_obs: int = 60,
        nu: float = 2.0,
        scale: float = 1.0,
        prior_scale: float = 2.5,
        outlier_fraction: float = 0.10,
        seed: int = 42,
    ) -> None:
        super().__init__(d=d)
        self.n_obs = int(n_obs)
        self.nu = float(nu)
        self.scale = float(scale)
        self.prior_scale = float(prior_scale)

        rng = np.random.default_rng(seed)
        self.X = rng.normal(0.0, 1.0, size=(self.n_obs, self.d)).astype(np.float64)
        self.beta_true = np.array([1.5, -1.0, 0.8, -0.5, 0.0, 0.0, 1.2, -0.7, 0.3, -1.1][: self.d], dtype=np.float64)

        clean_y = self.X @ self.beta_true + rng.normal(0.0, self.scale, size=self.n_obs)
        n_outliers = int(outlier_fraction * self.n_obs)
        outlier_indices = rng.choice(self.n_obs, size=n_outliers, replace=False)
        clean_y[outlier_indices] += rng.choice([-30.0, 30.0], size=n_outliers)
        self.y = clean_y.astype(np.float64)

        # Precompute Student-t log-density constant
        self._log_const_lik = float(
            special.gammaln((self.nu + 1.0) / 2.0)
            - special.gammaln(self.nu / 2.0)
            - 0.5 * np.log(self.nu * np.pi * (self.scale**2))
        )

    def log_prob(self, beta: np.ndarray) -> float:
        b = np.asarray(beta, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            return -np.inf

        res = self.y - self.X @ b
        lp_lik = self.n_obs * self._log_const_lik - 0.5 * (self.nu + 1.0) * np.sum(
            np.log1p((res**2) / (self.nu * (self.scale**2)))
        )
        scaled_b = b / self.prior_scale
        abs_scaled = np.abs(scaled_b)
        scaled_b_safe = np.clip(scaled_b, -1e100, 1e100)
        log_terms = np.where(
            abs_scaled > 1e100,
            2.0 * np.log(np.maximum(abs_scaled, 1.0)),
            np.log1p(scaled_b_safe**2),
        )
        lp_prior = -self.d * np.log(np.pi * self.prior_scale) - np.sum(log_terms)
        return float(lp_lik + lp_prior)

    def grad_log_prob(self, beta: np.ndarray) -> np.ndarray:
        b = np.asarray(beta, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            return np.full(self.d, np.nan, dtype=np.float64)

        res = self.y - self.X @ b
        weights = (self.nu + 1.0) / (self.nu * (self.scale**2) + (res**2))
        grad_lik = self.X.T @ (weights * res)
        abs_b = np.abs(b)
        scale_sq = self.prior_scale**2
        b_safe = np.clip(b, -1e100, 1e100)
        grad_prior = np.where(
            abs_b > 1e100,
            -2.0 / np.where(abs_b == 0.0, 1.0, b),
            -2.0 * b_safe / (scale_sq + b_safe**2),
        )
        return grad_lik + grad_prior

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.tile(self.beta_true, (n, 1))

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "beta_true": self.beta_true,
            "mean": self.beta_true,
            "nu": self.nu,
            "scale": self.scale,
        }


# =============================================================================
# 2. SOTA Target Adapter (JAX & PyMC Translation)
# =============================================================================


class SOTATargetAdapter:
    """Adapts any TargetDistribution for JAX (BlackJAX) and PyMC execution."""

    def __init__(self, target: TargetDistribution) -> None:
        self.target = target
        self.d = target.d

    def jax_logdensity_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Returns a compiled JAX callable logdensity_fn(x) -> scalar."""
        t = self.target

        # 1. Rubin's Eight-Schools ETS Meta-Analysis
        if isinstance(t, EightSchoolsTarget):
            y_jax = jnp.asarray(t.y, dtype=jnp.float64)
            sig_jax = jnp.asarray(t.sigma, dtype=jnp.float64)
            mu_sig = float(t.mu_prior_sigma)
            tau_scale = float(t.tau_prior_scale)
            n_sch = float(t.n_schools)

            def logdensity_eight_schools(x: jnp.ndarray) -> jnp.ndarray:
                mu = x[0]
                eta = x[1]
                theta = x[2:]
                tau_sq = jnp.exp(2.0 * eta)

                lp_mu = -0.5 * jnp.log(2.0 * jnp.pi * (mu_sig**2)) - 0.5 * (mu**2) / (mu_sig**2)
                lp_eta = jnp.log(2.0 / (tau_scale * jnp.pi)) + eta - jnp.log1p(jnp.exp(2.0 * eta) / (tau_scale**2))
                lp_theta = (
                    -0.5 * n_sch * jnp.log(2.0 * jnp.pi) - n_sch * eta - 0.5 * jnp.sum((theta - mu) ** 2) / tau_sq
                )
                lp_y = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * (sig_jax**2))) - 0.5 * jnp.sum(
                    ((y_jax - theta) ** 2) / (sig_jax**2)
                )
                return lp_mu + lp_eta + lp_theta + lp_y

            return logdensity_eight_schools

        # 2. Sparse Signal Recovery / Cauchy Regression
        if isinstance(t, SparseSignalRegressionTarget):
            X_jax = jnp.asarray(t.X, dtype=jnp.float64)
            y_jax = jnp.asarray(t.y, dtype=jnp.float64)
            inv_var = float(t.inv_var)
            scale = float(t.prior_scale)
            n_obs = float(t.n_obs)
            d_dim = float(t.d)

            def logdensity_sparse(beta: jnp.ndarray) -> jnp.ndarray:
                res = y_jax - jnp.dot(X_jax, beta)
                lp_lik = -0.5 * n_obs * jnp.log(2.0 * jnp.pi / inv_var) - 0.5 * inv_var * jnp.sum(res**2)
                lp_prior = -d_dim * jnp.log(jnp.pi * scale) - jnp.sum(jnp.log1p((beta / scale) ** 2))
                return lp_lik + lp_prior

            return logdensity_sparse

        # 3. Robust Outlier Regression
        if isinstance(t, RobustOutlierRegressionTarget):
            X_jax = jnp.asarray(t.X, dtype=jnp.float64)
            y_jax = jnp.asarray(t.y, dtype=jnp.float64)
            nu = float(t.nu)
            sc = float(t.scale)
            psc = float(t.prior_scale)
            log_c = float(t._log_const_lik)
            n_obs = float(t.n_obs)
            d_dim = float(t.d)

            def logdensity_robust(beta: jnp.ndarray) -> jnp.ndarray:
                res = y_jax - jnp.dot(X_jax, beta)
                lp_lik = n_obs * log_c - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p((res**2) / (nu * (sc**2))))
                lp_prior = -d_dim * jnp.log(jnp.pi * psc) - jnp.sum(jnp.log1p((beta / psc) ** 2))
                return lp_lik + lp_prior

            return logdensity_robust

        # 4. Neal's Funnel
        if isinstance(t, NealsFunnelTarget):
            sigma_v = float(t.sigma_v)
            d_minus_1 = float(t.d - 1)

            def logdensity_funnel(x: jnp.ndarray) -> jnp.ndarray:
                v = x[0]
                rest = x[1:]
                lp_v = -0.5 * jnp.log(2.0 * jnp.pi * (sigma_v**2)) - 0.5 * (v**2) / (sigma_v**2)
                lp_det = -0.5 * d_minus_1 * jnp.log(2.0 * jnp.pi) - 0.5 * d_minus_1 * v
                lp_quad = -0.5 * jnp.exp(-v) * jnp.sum(rest**2)
                return lp_v + lp_det + lp_quad

            return logdensity_funnel

        # 5. Quartic Superlinear Potential
        if isinstance(t, QuarticTarget):
            p = float(t.p)
            sc = float(t.scale)
            log_z = float(t._log_Z)

            def logdensity_quartic(x: jnp.ndarray) -> jnp.ndarray:
                norm_sq = jnp.sum(x**2)
                norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
                u = (norm**p) / (p * sc)
                return -u - log_z

            return logdensity_quartic

        # 6. Multivariate Student-t Target
        if isinstance(t, (MultivariateStudentTTarget, StudentTTarget)):
            nu = float(t.nu)
            mu_jax = jnp.asarray(t.mu, dtype=jnp.float64)
            L_jax = jnp.asarray(t.L, dtype=jnp.float64)
            log_c = float(t._log_const)
            dim_t = float(t.d)

            def logdensity_mvt(x: jnp.ndarray) -> jnp.ndarray:
                diff = x - mu_jax
                z = jax.scipy.linalg.solve_triangular(L_jax, diff, lower=True)
                delta_sq = jnp.sum(z**2)
                return log_c - 0.5 * (nu + dim_t) * jnp.log1p(delta_sq / nu)

            return logdensity_mvt

        # 7. Heterogeneous Heavy-Tailed Target
        if isinstance(t, (HeterogeneousHeavyTailedTarget, HeterogeneousStudentTTarget)):
            heavy_idx = jnp.asarray(list(t.heavy_coords), dtype=jnp.int32)
            light_idx = jnp.asarray(list(t.light_coords), dtype=jnp.int32)
            nu_arr = jnp.asarray(t.nu_arr, dtype=jnp.float64)
            scale_arr = jnp.asarray(t.scale_arr, dtype=jnp.float64)
            mu_arr = jnp.asarray(t.mu, dtype=jnp.float64)

            def logdensity_hetero(x: jnp.ndarray) -> jnp.ndarray:
                diff = x - mu_arr
                heavy_diff = diff[heavy_idx] / scale_arr[heavy_idx]
                heavy_lp = -0.5 * (nu_arr + 1.0) * jnp.log1p((heavy_diff**2) / nu_arr)
                light_diff = diff[light_idx] / scale_arr[light_idx]
                light_lp = -0.5 * (light_diff**2)
                return jnp.sum(heavy_lp) + jnp.sum(light_lp)

            return logdensity_hetero

        # 8. Gaussian Baselines
        if isinstance(
            t, (IsotropicGaussian, DiagonalGaussian, FullCovarianceGaussian, GaussianTarget, AnisotropicGaussianTarget)
        ):
            if hasattr(t, "mean") and hasattr(t, "cov"):
                mean_jax = jnp.asarray(t.mean, dtype=jnp.float64)
                cov_jax = jnp.asarray(t.cov, dtype=jnp.float64)
                L_jax = jnp.asarray(t.L if hasattr(t, "L") else np.linalg.cholesky(t.cov), dtype=jnp.float64)

                def logdensity_gauss(x: jnp.ndarray) -> jnp.ndarray:
                    diff = x - mean_jax
                    z = jax.scipy.linalg.solve_triangular(L_jax, diff, lower=True)
                    return -0.5 * jnp.sum(z**2)

                return logdensity_gauss

        # Fallback: custom VJP using pure_callback
        target_ref = self.target

        @jax.custom_vjp
        def generic_logdensity(x_vec: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                lambda x: float(target_ref.log_prob(np.asarray(x, dtype=np.float64))),
                jnp.float64(0.0),
                x_vec,
            )

        def generic_logdensity_fwd(x_vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            val = generic_logdensity(x_vec)
            return val, x_vec

        def generic_logdensity_bwd(res: jnp.ndarray, g: jnp.ndarray) -> tuple[jnp.ndarray]:
            x_vec = res
            grad = jax.pure_callback(
                lambda x: np.asarray(target_ref.grad_log_prob(np.asarray(x, dtype=np.float64)), dtype=np.float64),
                jnp.zeros_like(x_vec),
                x_vec,
            )
            return (g * grad,)

        generic_logdensity.defvjp(generic_logdensity_fwd, generic_logdensity_bwd)
        return generic_logdensity

    def transformed_jax_logdensity_fn(
        self,
        radii: float | np.ndarray = 3.0,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Returns transformed JAX log density under z_i = arcsinh(x_i / R_i)."""
        base_logdensity = self.jax_logdensity_fn()
        r_jax = jnp.asarray(radii, dtype=jnp.float64)

        def logdensity_transformed(z: jnp.ndarray) -> jnp.ndarray:
            x = r_jax * jnp.sinh(z)
            log_det = jnp.sum(jnp.log(r_jax * jnp.cosh(z)))
            return base_logdensity(x) + log_det

        return logdensity_transformed

    def pymc_model(self) -> pm.Model:
        """Constructs an equivalent PyMC model where position vector is named 'x'."""
        t = self.target

        # 1. Rubin's Eight-Schools Centered Parameterization
        if isinstance(t, EightSchoolsTarget):
            with pm.Model() as model:
                mu = pm.Normal("mu", mu=0.0, sigma=t.mu_prior_sigma)
                tau = pm.HalfCauchy("tau", beta=t.tau_prior_scale)
                theta = pm.Normal("theta", mu=mu, sigma=tau, shape=t.n_schools)
                pm.Normal("y_obs", mu=theta, sigma=t.sigma, observed=t.y)
                # Deterministic coordinate vector x = [mu, log(tau), theta_0, ..., theta_7]
                pm.Deterministic("x", pt.concatenate([[mu], [pt.log(tau)], theta]))
            return model

        # 2. Sparse Signal Recovery / Cauchy Regression
        if isinstance(t, SparseSignalRegressionTarget):
            with pm.Model() as model:
                x = pm.Cauchy("x", alpha=0.0, beta=t.prior_scale, shape=t.d)
                mu = pm.math.dot(t.X, x)
                pm.Normal("y_obs", mu=mu, sigma=t.noise_std, observed=t.y)
            return model

        # 3. Robust Outlier Regression
        if isinstance(t, RobustOutlierRegressionTarget):
            with pm.Model() as model:
                x = pm.Cauchy("x", alpha=0.0, beta=t.prior_scale, shape=t.d)
                mu = pm.math.dot(t.X, x)
                pm.StudentT("y_obs", nu=t.nu, mu=mu, sigma=t.scale, observed=t.y)
            return model

        # 4. Neal's Funnel
        if isinstance(t, NealsFunnelTarget):
            with pm.Model() as model:
                v = pm.Normal("v", mu=0.0, sigma=t.sigma_v)
                rest = pm.Normal("rest", mu=0.0, sigma=pt.exp(0.5 * v), shape=t.d - 1)
                pm.Deterministic("x", pt.concatenate([[v], rest]))
            return model

        # 5. Quartic Target
        if isinstance(t, QuarticTarget):
            with pm.Model() as model:
                x = pm.Flat("x", shape=(t.d,))
                sq_norm = pt.sum(x**2)
                norm = pt.sqrt(pt.maximum(sq_norm, 1e-30))
                pot = -(norm**t.p) / (t.p * t.scale)
                pm.Potential("pot", pot)
            return model

        # 6. Multivariate Student-t Target
        if isinstance(t, (MultivariateStudentTTarget, StudentTTarget)):
            with pm.Model() as model:
                x = pm.Flat("x", shape=(t.d,))
                diff = x - t.mu
                # Quad form via solve or precision
                inv_cov = np.linalg.inv(t._cov)
                delta_sq = pt.dot(diff, pt.dot(inv_cov, diff))
                pot = -0.5 * (t.nu + t.d) * pt.log1p(delta_sq / t.nu)
                pm.Potential("pot", pot)
            return model

        # 7. Heterogeneous Heavy-Tailed Target
        if isinstance(t, (HeterogeneousHeavyTailedTarget, HeterogeneousStudentTTarget)):
            heavy_list = list(t.heavy_coords)
            light_list = list(t.light_coords)
            nu_arr = t.nu_arr
            scale_arr = t.scale_arr
            with pm.Model() as model:
                x = pm.Flat("x", shape=(t.d,))
                diff = x - t.mu
                pot_terms = []
                for i, h_idx in enumerate(heavy_list):
                    d_h = diff[h_idx] / scale_arr[h_idx]
                    pot_terms.append(-0.5 * (nu_arr[i] + 1.0) * pt.log1p((d_h**2) / nu_arr[i]))
                for l_idx in light_list:
                    d_l = diff[l_idx] / scale_arr[l_idx]
                    pot_terms.append(-0.5 * (d_l**2))
                pm.Potential("pot", pt.sum(pot_terms))
            return model

        # Default Gaussian / General Fallback
        with pm.Model() as model:
            x = pm.Flat("x", shape=(t.d,))
            # Standard quadratic form if covariance is available
            if hasattr(t, "mean") and hasattr(t, "cov"):
                diff = x - t.mean
                inv_cov = np.linalg.inv(t.cov)
                pot = -0.5 * pt.dot(diff, pt.dot(inv_cov, diff))
                pm.Potential("pot", pot)
            else:
                pot = -0.5 * pt.sum(x**2)
                pm.Potential("pot", pot)
        return model


# =============================================================================
# 3. Execution Runners: BlackJAX, PyMC, mcmc_robust
# =============================================================================


def run_blackjax_sampler(
    target: TargetDistribution,
    sampler_type: str = "nuts",
    n_samples: int = 500,
    n_warmup: int = 300,
    n_chains: int = 4,
    step_size: float | None = None,
    seed: int = 42,
    x0: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Executes BlackJAX sampler tracking exact gradient evaluations via info.num_integration_steps.

    Args:
        target: TargetDistribution instance.
        sampler_type: "nuts" (default), "hmc", or "mclmc".
        n_samples: Retained samples per chain.
        n_warmup: Warmup/tuning steps per chain.
        n_chains: Number of independent Markov chains.
        step_size: Optional fixed step size (if None, adapted during warmup).
        seed: Master random seed.
        x0: Optional initial state array of shape (n_chains, d) or (d,).
        **kwargs: Additional sampler kwargs.

    Returns:
        Tuple of:
        - samples array of shape (n_chains, n_samples, d).
        - telemetry dictionary with exact gradient evaluations and divergences.
    """
    adapter = SOTATargetAdapter(target)
    logdensity_fn = adapter.jax_logdensity_fn()
    d = target.d

    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=np.float64)
        if x0_arr.ndim == 1:
            init_positions = np.tile(x0_arr, (n_chains, 1))
        else:
            init_positions = x0_arr
    else:
        init_positions = np.zeros((n_chains, d), dtype=np.float64)

    t0 = time.perf_counter()
    all_chain_samples: list[np.ndarray] = []
    total_warmup_grads = 0
    total_draw_grads = 0
    total_divergences = 0

    sampler_type_lower = sampler_type.lower()

    for c in range(n_chains):
        rng_key = jax.random.PRNGKey(seed + c * 10007)
        rng_warmup, rng_sample = jax.random.split(rng_key)
        init_pos = jnp.asarray(init_positions[c], dtype=jnp.float64)

        if sampler_type_lower == "nuts":
            # NUTS with window adaptation
            if n_warmup > 0:
                warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
                (state, params), w_info = warmup.run(rng_warmup, init_pos, num_steps=n_warmup)
                total_warmup_grads += int(jnp.sum(w_info.info.num_integration_steps))
                total_divergences += int(jnp.sum(w_info.info.is_divergent))
            else:
                eps = step_size or 0.1
                state = blackjax.nuts.init(init_pos, logdensity_fn)
                params = {"step_size": eps, "inverse_mass_matrix": jnp.ones(d, dtype=jnp.float64)}

            step_fn = blackjax.nuts(logdensity_fn, **params).step

            def one_step_nuts(carry: Any, subkey: Any) -> tuple[Any, tuple[Any, Any, Any]]:
                cur_state = carry
                next_state, info = step_fn(subkey, cur_state)
                return next_state, (next_state.position, info.num_integration_steps, info.is_divergent)

            keys = jax.random.split(rng_sample, n_samples)
            _, (chain_pos, chain_steps, chain_divs) = jax.lax.scan(one_step_nuts, state, keys)
            all_chain_samples.append(np.array(chain_pos))
            total_draw_grads += int(jnp.sum(chain_steps))
            total_divergences += int(jnp.sum(chain_divs))

        elif sampler_type_lower == "hmc":
            n_leapfrog = kwargs.get("num_integration_steps", kwargs.get("n_leapfrog", 10))
            if n_warmup > 0:
                warmup = blackjax.window_adaptation(blackjax.hmc, logdensity_fn, num_integration_steps=n_leapfrog)
                (state, params), w_info = warmup.run(rng_warmup, init_pos, num_steps=n_warmup)
                total_warmup_grads += int(jnp.sum(w_info.info.num_integration_steps))
                total_divergences += int(jnp.sum(w_info.info.is_divergent))
            else:
                eps = step_size or 0.1
                state = blackjax.hmc.init(init_pos, logdensity_fn)
                params = {
                    "step_size": eps,
                    "inverse_mass_matrix": jnp.ones(d, dtype=jnp.float64),
                    "num_integration_steps": n_leapfrog,
                }

            step_fn = blackjax.hmc(logdensity_fn, **params).step

            def one_step_hmc(carry: Any, subkey: Any) -> tuple[Any, tuple[Any, Any, Any]]:
                cur_state = carry
                next_state, info = step_fn(subkey, cur_state)
                return next_state, (next_state.position, info.num_integration_steps, info.is_divergent)

            keys = jax.random.split(rng_sample, n_samples)
            _, (chain_pos, chain_steps, chain_divs) = jax.lax.scan(one_step_hmc, state, keys)
            all_chain_samples.append(np.array(chain_pos))
            total_draw_grads += int(jnp.sum(chain_steps))
            total_divergences += int(jnp.sum(chain_divs))

        else:
            raise ValueError(f"Unsupported BlackJAX sampler_type: {sampler_type}")

    wall_clock = time.perf_counter() - t0
    samples_array = np.stack(all_chain_samples, axis=0)  # shape (n_chains, n_samples, d)
    total_grads = total_warmup_grads + total_draw_grads

    telemetry: dict[str, Any] = {
        "framework": "blackjax",
        "sampler_type": sampler_type,
        "n_chains": n_chains,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "warmup_grad_evals": total_warmup_grads,
        "draw_grad_evals": total_draw_grads,
        "total_grad_evals": total_grads,
        "divergences": total_divergences,
        "divergence_rate": float(total_divergences / max(1, n_chains * (n_samples + n_warmup))),
        "wall_clock_time": float(wall_clock),
    }

    return samples_array, telemetry


def run_pymc_sampler(
    target: TargetDistribution,
    n_samples: int = 500,
    n_warmup: int = 300,
    n_chains: int = 4,
    seed: int = 42,
    x0: np.ndarray | None = None,
    cores: int = 1,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Executes PyMC NUTS extracting exact leapfrog steps via sample_stats['n_steps'].

    Args:
        target: TargetDistribution instance.
        n_samples: Retained samples per chain.
        n_warmup: Tuning steps per chain.
        n_chains: Number of independent chains.
        seed: Random seed.
        x0: Optional initial states.
        cores: Number of parallel jobs (default 1 for deterministic sequential execution).
        **kwargs: Additional pm.sample kwargs.

    Returns:
        Tuple of:
        - samples array of shape (n_chains, n_samples, d).
        - telemetry dictionary with exact gradient evaluations and divergences.
    """
    adapter = SOTATargetAdapter(target)
    model = adapter.pymc_model()

    initvals: Any = None
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=np.float64)
        if x0_arr.ndim == 1:
            initvals = {"x": x0_arr}
        elif x0_arr.ndim == 2 and x0_arr.shape[0] == n_chains:
            initvals = [{"x": x0_arr[c]} for c in range(n_chains)]

    t0 = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=n_samples,
            tune=n_warmup,
            chains=n_chains,
            cores=cores,
            random_seed=seed,
            initvals=initvals,
            progressbar=False,
            compute_convergence_checks=False,
            discard_tuned_samples=False,
            return_inferencedata=True,
            **kwargs,
        )
    wall_clock = time.perf_counter() - t0

    # Extract coordinates array of shape (n_chains, n_samples, d)
    if "x" in idata.posterior:
        samples_raw = idata.posterior["x"].values
    else:
        # Concatenate all posterior variables
        vars_list = [v for v in idata.posterior.data_vars if not v.endswith("_")]
        arrays = [idata.posterior[v].values for v in vars_list]
        reshaped = []
        for arr in arrays:
            if arr.ndim == 2:
                reshaped.append(arr[:, :, None])
            else:
                reshaped.append(arr)
        samples_raw = np.concatenate(reshaped, axis=-1)

    # Extract leapfrog steps
    draw_steps = int(np.sum(idata["sample_stats"]["n_steps"].values))
    warmup_steps = 0
    if "warmup_sample_stats" in idata:
        warmup_steps = int(np.sum(idata["warmup_sample_stats"]["n_steps"].values))

    # Extract divergences
    draw_divs = int(np.sum(idata["sample_stats"]["diverging"].values))
    warmup_divs = 0
    if "warmup_sample_stats" in idata:
        warmup_divs = int(np.sum(idata["warmup_sample_stats"]["diverging"].values))

    total_grads = draw_steps + warmup_steps
    total_divs = draw_divs + warmup_divs

    telemetry: dict[str, Any] = {
        "framework": "pymc",
        "sampler_type": "nuts",
        "n_chains": n_chains,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "warmup_grad_evals": warmup_steps,
        "draw_grad_evals": draw_steps,
        "total_grad_evals": total_grads,
        "divergences": total_divs,
        "divergence_rate": float(total_divs / max(1, n_chains * (n_samples + n_warmup))),
        "wall_clock_time": float(wall_clock),
    }

    return np.asarray(samples_raw, dtype=np.float64), telemetry


def run_mcmc_robust_sampler(
    sampler_cls: type[Sampler] | Callable[..., Sampler] | Sampler,
    target: TargetDistribution,
    n_samples: int = 500,
    n_warmup: int = 300,
    n_chains: int = 4,
    seed: int = 42,
    x0: np.ndarray | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Executes novel mcmc_robust or reference sampler tracking exact oracle evaluations via GradBudgetTracker.

    Supports UnifiedRobustNUTS, UnifiedRobustHMC, UnifiedRobustSampler, GentleMALA,
    AnisotropicTailMALA, BarkerLangevin, GloballyTamedMALA, Stereographic MALA, and Radial MALA.

    Args:
        sampler_cls: Sampler class, factory callable, or Sampler instance.
        target: TargetDistribution instance.
        n_samples: Retained samples per chain.
        n_warmup: Warmup steps per chain.
        n_chains: Number of independent chains.
        seed: Random seed.
        x0: Optional initial states.
        sampler_kwargs: Optional dictionary of constructor arguments for the sampler.

    Returns:
        Tuple of:
        - samples array of shape (n_chains, n_samples, d).
        - telemetry dictionary with exact gradient evaluations and divergences.
    """
    import inspect

    d = target.d
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=np.float64)
        if x0_arr.ndim == 1:
            init_positions = np.tile(x0_arr, (n_chains, 1))
        else:
            init_positions = x0_arr
    else:
        init_positions = np.zeros((n_chains, d), dtype=np.float64)

    t0 = time.perf_counter()
    all_chain_samples: list[np.ndarray] = []
    total_grad_evals = 0
    total_divergences = 0

    base_kwargs = dict(sampler_kwargs or {})

    # Determine sampler name
    if isinstance(sampler_cls, Sampler):
        sampler_name = sampler_cls.__class__.__name__
    elif hasattr(sampler_cls, "__name__"):
        sampler_name = sampler_cls.__name__
    else:
        sampler_name = str(sampler_cls)

    for c in range(n_chains):
        rng_chain = np.random.default_rng(seed + c * 10007)
        tracker = GradBudgetTracker(target)

        # Build chain-specific sampler instance
        if isinstance(sampler_cls, Sampler):
            sampler = sampler_cls
        else:
            # Inspect signature to filter compatible arguments
            supported_params: Mapping[str, inspect.Parameter]
            try:
                sig = inspect.signature(sampler_cls)
                supported_params = sig.parameters
            except Exception:
                supported_params = {}

            chain_kw = dict(base_kwargs)
            if "warmup_steps" in supported_params and "warmup_steps" not in chain_kw:
                chain_kw["warmup_steps"] = n_warmup
            if "dim" in supported_params and "dim" not in chain_kw:
                chain_kw["dim"] = d

            # Filter kwargs to only supported ones if no **kwargs in signature
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in supported_params.values())
            if not has_var_kwargs and supported_params:
                filtered_kw = {k: v for k, v in chain_kw.items() if k in supported_params}
            else:
                filtered_kw = chain_kw

            sampler = sampler_cls(**filtered_kw)

        chain_samples_raw, step_results = sampler.sample(
            x0=init_positions[c],
            target=tracker,
            n_steps=n_samples + n_warmup,
            burn_in=n_warmup,
            rng=rng_chain,
            adapt=True,
        )

        # Pull back primal coordinates for space-transformed samplers if present
        chain_samples_list = []
        for res in step_results[n_warmup:]:
            primal = res.state.auxiliary.get("primal_x", None)
            if primal is not None:
                chain_samples_list.append(np.asarray(primal, dtype=np.float64))
            else:
                chain_samples_list.append(np.asarray(res.state.x, dtype=np.float64))
        chain_samples = np.asarray(chain_samples_list, dtype=np.float64)

        all_chain_samples.append(chain_samples)
        total_grad_evals += tracker.grad_evals

        # Tally post-warmup divergences
        for res in step_results[n_warmup:]:
            is_div = (
                float(res.stats.get("divergent", 0.0)) > 0.0 or float(res.stats.get("hamiltonian_error", 0.0)) > 1000.0
            )
            if is_div:
                total_divergences += 1

    wall_clock = time.perf_counter() - t0
    samples_array = np.stack(all_chain_samples, axis=0)  # shape (n_chains, n_samples, d)

    telemetry: dict[str, Any] = {
        "framework": "mcmc_robust",
        "sampler_type": sampler_name,
        "n_chains": n_chains,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "total_grad_evals": total_grad_evals,
        "divergences": total_divergences,
        "divergence_rate": float(total_divergences / max(1, n_chains * n_samples)),
        "wall_clock_time": float(wall_clock),
    }

    return samples_array, telemetry


# =============================================================================
# 4. ArviZ 1.3 Diagnostic Pipeline
# =============================================================================


def evaluate_chains_with_arviz(
    samples: np.ndarray,
    total_grad_evals: int,
    target: TargetDistribution | None = None,
    divergences: int = 0,
    wall_clock_time: float = 0.0,
) -> dict[str, float]:
    r"""Constructs ArviZ 1.3 DataTree and computes comprehensive convergence metrics.

    Args:
        samples: MCMC draws array of shape (n_chains, n_samples, d) or (n_samples, d).
        total_grad_evals: Exact total gradient evaluations consumed across all chains.
        target: Optional TargetDistribution instance for analytical quantile verification.
        divergences: Divergent transition count.
        wall_clock_time: Wall-clock execution time in seconds.

    Returns:
        Dictionary containing Bulk-ESS, Tail-ESS, split R-hat, ESS/grad, Tail-ESS/grad,
        divergence rate, and quantile errors.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :, np.newaxis]
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 1D, 2D, or 3D samples array, got shape {arr.shape}")

    n_chains, n_samples, d = arr.shape

    # Construct ArviZ 1.3 DataTree via az.from_dict
    dt = az.from_dict({"posterior": {"x": arr}})

    # Bulk and Tail ESS
    bulk_ess_res = az.ess(dt, var_names=["x"], method="bulk")["x"].values
    tail_ess_res = az.ess(dt, var_names=["x"], method="tail")["x"].values

    bulk_ess = float(np.mean(bulk_ess_res))
    tail_ess = float(np.mean(tail_ess_res))

    # Split R-hat and Folded R-hat (Vehtari, Gelman, Simpson, Carpenter, & Bürkner 2021)
    if n_chains >= 2 and n_samples >= 4:
        rhat_res = az.rhat(dt, var_names=["x"])["x"].values
        split_rhat = float(np.max(rhat_res))
        try:
            folded_rhat_res = az.rhat(dt, var_names=["x"], method="folded")["x"].values
            folded_rhat = float(np.max(folded_rhat_res))
        except Exception:
            folded_rhat = split_rhat
        combined_rhat = float(max(split_rhat, folded_rhat))
    else:
        split_rhat = 1.0
        folded_rhat = 1.0
        combined_rhat = 1.0

    evals = max(1, int(total_grad_evals))
    ess_per_grad = float(bulk_ess / evals)
    tail_ess_per_grad = float(tail_ess / evals)
    div_rate = float(divergences / max(1, n_chains * n_samples))

    # Quantile MCSE at prob=0.99
    try:
        mcse_q99_res = az.mcse(dt, var_names=["x"], method="quantile", prob=0.99)["x"].values
        mcse_q99 = float(np.nanmean(mcse_q99_res))
    except Exception:
        mcse_q99 = float("nan")

    metrics: dict[str, float] = {
        "bulk_ess": bulk_ess,
        "tail_ess": tail_ess,
        "split_rhat": split_rhat,
        "folded_rhat": folded_rhat,
        "combined_rhat": combined_rhat,
        "mcse_q99": mcse_q99,
        "total_grad_evals": float(evals),
        "ess_per_grad": ess_per_grad,
        "tail_ess_per_grad": tail_ess_per_grad,
        "divergences": float(divergences),
        "divergence_rate": div_rate,
        "wall_clock_time": float(wall_clock_time),
    }

    # Quantile estimation errors if target is provided
    if target is not None:
        try:
            pooled = arr.reshape(-1, d)
            q_errors = compute_quantile_relative_error(pooled, target, quantiles=(0.05, 0.50, 0.95, 0.99))
            metrics["q05_rel_error"] = float(np.mean(q_errors[0.05]))
            metrics["q50_rel_error"] = float(np.mean(q_errors[0.50]))
            metrics["q95_rel_error"] = float(np.mean(q_errors[0.95]))
            metrics["q99_rel_error"] = float(np.mean(q_errors[0.99]))
        except Exception:
            pass

    return metrics
