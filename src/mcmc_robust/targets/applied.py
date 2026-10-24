"""Synthetic, PosteriorDB-inspired applied Bayesian target distributions.

The five models mirror the structure of well-known applied posteriors (hierarchical
regression, horseshoe logistic regression, robust regression, 2PL IRT, stochastic
volatility), but their datasets are small and *simulated* from fixed data-generating
parameters. They are not the PosteriorDB posteriors themselves.

None of these targets has a closed-form posterior. ``sample_ground_truth`` therefore
raises ``NotImplementedError`` (the same convention as ``PyMCModelTarget``), and
``exact_moments_or_quantiles`` exposes the data-generating parameters under an explicit
key rather than presenting them as posterior moments.
"""

from __future__ import annotations

from typing import Any, NoReturn

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import special

from mcmc_robust.core.target import TargetDistribution


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid 1 / (1 + exp(-z))."""
    z_clip = np.clip(z, -500.0, 500.0)
    return np.where(
        z_clip >= 0.0,
        1.0 / (1.0 + np.exp(-z_clip)),
        np.exp(z_clip) / (1.0 + np.exp(z_clip)),
    )


def _stable_softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable softplus log(1 + exp(z)) = logaddexp(0, z)."""
    return np.logaddexp(0.0, z)


def _raise_no_reference_posterior(target: TargetDistribution) -> NoReturn:
    """Raises for simulated-data targets whose posterior has no closed form.

    Returning draws around the data-generating parameters would silently score samplers
    against a fabricated reference, so callers must obtain one from a long MCMC run instead.
    """
    raise NotImplementedError(
        f"{type(target).__name__} has no closed-form posterior (simulated data); "
        "estimate reference quantities with a long, converged MCMC run instead."
    )


class RadonHierarchicalTarget(TargetDistribution):
    r"""Hierarchical county-intercept regression modelled on the Gelman & Hill radon example.

    Synthetic analogue of the PosteriorDB radon models: the data are simulated
    (default 15 counties x 8 observations), not the Minnesota radon dataset.

    Default dimension `d = 18`:
    - $J = 15$ county-level intercepts $\alpha_{0:J-1} \in \mathbb{R}^J$
    - Population intercept mean $\mu_\alpha \in \mathbb{R}$
    - Log between-county standard deviation $v = \log \sigma_\alpha \in \mathbb{R}$
    - Floor measurement slope $\beta_{\text{floor}} \in \mathbb{R}$ ($0 = \text{basement}, 1 = \text{first floor}$)

    Centered hierarchical parameterization:
        $$\mu_\alpha \sim \mathcal{N}(0, \sigma_\mu^2), \quad v \sim \mathcal{N}(0, \sigma_v^2), \quad \beta_{\text{floor}} \sim \mathcal{N}(0, \sigma_\beta^2)$$
        $$\alpha_j \mid \mu_\alpha, v \sim \mathcal{N}(\mu_\alpha, e^{2v}), \quad j = 0, \dots, J-1$$
        $$y_i \mid \alpha, \beta_{\text{floor}} \sim \mathcal{N}(\alpha_{c_i} + \beta_{\text{floor}} x_i, \sigma_y^2), \quad i = 0, \dots, N-1$$
    Induces a genuine data-driven hierarchical funnel neck as $v \to -\infty$.
    """

    def __init__(
        self,
        n_counties: int = 15,
        n_obs_per_county: int = 8,
        sigma_y: float = 0.75,
        sigma_mu: float = 10.0,
        sigma_v: float = 2.0,
        sigma_beta: float = 5.0,
        seed: int = 101,
    ) -> None:
        if n_counties < 2:
            raise ValueError(f"n_counties must be >= 2, got {n_counties}")
        if n_obs_per_county < 1:
            raise ValueError(f"n_obs_per_county must be >= 1, got {n_obs_per_county}")
        if sigma_y <= 0.0 or sigma_mu <= 0.0 or sigma_v <= 0.0 or sigma_beta <= 0.0:
            raise ValueError("All scale hyperparameters must be strictly positive")

        self.n_counties: int = int(n_counties)
        self.n_obs_per_county: int = int(n_obs_per_county)
        self.n_obs: int = self.n_counties * self.n_obs_per_county
        self.sigma_y: float = float(sigma_y)
        self.sigma_mu: float = float(sigma_mu)
        self.sigma_v: float = float(sigma_v)
        self.sigma_beta: float = float(sigma_beta)

        super().__init__(d=self.n_counties + 3)

        rng = np.random.default_rng(seed)
        self.county_idx: np.ndarray = np.repeat(np.arange(self.n_counties, dtype=np.int64), self.n_obs_per_county)
        self.floor_x: np.ndarray = rng.binomial(1, 0.25, size=self.n_obs).astype(np.float64)

        # True hierarchical parameters modeled on Gelman & Hill (2007) Minnesota radon data
        self.mu_alpha_true: float = 1.45
        self.v_true: float = -1.15  # sigma_alpha ~ 0.316
        self.beta_floor_true: float = -0.68
        self.alpha_true: np.ndarray = self.mu_alpha_true + np.exp(self.v_true) * rng.normal(size=self.n_counties)
        mean_y = self.alpha_true[self.county_idx] + self.beta_floor_true * self.floor_x
        self.y: np.ndarray = (mean_y + rng.normal(0.0, self.sigma_y, size=self.n_obs)).astype(np.float64)

        # Precompute county incidence matrix for fast vectorized reductions
        self._county_indicator: np.ndarray = np.eye(self.n_counties, dtype=np.float64)[self.county_idx]
        self._county_counts: np.ndarray = np.sum(self._county_indicator, axis=0)
        self._log_norm_const: float = float(
            -0.5 * np.log(2.0 * np.pi * (self.sigma_mu**2))
            - 0.5 * np.log(2.0 * np.pi * (self.sigma_v**2))
            - 0.5 * np.log(2.0 * np.pi * (self.sigma_beta**2))
            - 0.5 * self.n_counties * np.log(2.0 * np.pi)
            - 0.5 * self.n_obs * np.log(2.0 * np.pi * (self.sigma_y**2))
        )

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = self._validate_input(x)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        J = self.n_counties
        alpha = x_arr[:J]
        mu_alpha = float(x_arr[J])
        v = float(x_arr[J + 1])
        beta_floor = float(x_arr[J + 2])

        if v < -120.0 or v > 120.0:
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        inv_tau_sq = float(np.exp(-2.0 * v))
        diff_alpha = alpha - mu_alpha
        sq_diff_alpha = float(np.dot(diff_alpha, diff_alpha))

        pred = alpha[self.county_idx] + beta_floor * self.floor_x
        res = self.y - pred
        inv_sig_y_sq = 1.0 / (self.sigma_y**2)

        lp_priors = (
            -0.5 * (mu_alpha**2) / (self.sigma_mu**2)
            - 0.5 * (v**2) / (self.sigma_v**2)
            - 0.5 * (beta_floor**2) / (self.sigma_beta**2)
        )
        lp_alpha = -J * v - 0.5 * inv_tau_sq * sq_diff_alpha
        lp_lik = -0.5 * inv_sig_y_sq * float(np.dot(res, res))
        total_lp = float(self._log_norm_const + lp_priors + lp_alpha + lp_lik)

        if not np.isfinite(total_lp):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        grad = np.empty(self.d, dtype=np.float64)
        # d/d(alpha_j)
        grad[:J] = -inv_tau_sq * diff_alpha + inv_sig_y_sq * (self._county_indicator.T @ res)
        # d/d(mu_alpha)
        grad[J] = -mu_alpha / (self.sigma_mu**2) + inv_tau_sq * float(np.sum(diff_alpha))
        # d/dv
        grad[J + 1] = -v / (self.sigma_v**2) - float(J) + inv_tau_sq * sq_diff_alpha
        # d/d(beta_floor)
        grad[J + 2] = -beta_floor / (self.sigma_beta**2) + inv_sig_y_sq * float(np.dot(res, self.floor_x))

        return total_lp, grad

    def log_prob(self, x: np.ndarray) -> float:
        lp, _ = self.log_prob_and_grad(x)
        return lp

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        _, grad = self.log_prob_and_grad(x)
        return grad

    def jax_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX log-posterior evaluation for XLA / BlackJAX execution."""
        J = self.n_counties
        alpha = x[:J]
        mu_alpha = x[J]
        v = x[J + 1]
        beta_floor = x[J + 2]

        inv_tau_sq = jnp.exp(-2.0 * v)
        diff_alpha = alpha - mu_alpha
        sq_diff_alpha = jnp.sum(diff_alpha**2)

        county_idx_jax = jnp.asarray(self.county_idx, dtype=jnp.int32)
        floor_x_jax = jnp.asarray(self.floor_x, dtype=jnp.float64)
        y_jax = jnp.asarray(self.y, dtype=jnp.float64)

        pred = alpha[county_idx_jax] + beta_floor * floor_x_jax
        res = y_jax - pred
        inv_sig_y_sq = 1.0 / (self.sigma_y**2)

        lp_priors = (
            -0.5 * (mu_alpha**2) / (self.sigma_mu**2)
            - 0.5 * (v**2) / (self.sigma_v**2)
            - 0.5 * (beta_floor**2) / (self.sigma_beta**2)
        )
        lp_alpha = -float(J) * v - 0.5 * inv_tau_sq * sq_diff_alpha
        lp_lik = -0.5 * inv_sig_y_sq * jnp.sum(res**2)
        return jnp.float64(self._log_norm_const) + lp_priors + lp_alpha + lp_lik

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        _raise_no_reference_posterior(self)

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "has_reference_posterior": False,
            "data_generating_params": np.concatenate(
                [self.alpha_true, [self.mu_alpha_true, self.v_true, self.beta_floor_true]]
            ),
            "n_counties": self.n_counties,
            "n_obs": self.n_obs,
            "v_index": self.n_counties + 1,
        }


class SparseHorseshoeLogisticTarget(TargetDistribution):
    r"""Sparse Bayesian Logistic Regression with Piironen-Vehtari (2017) Horseshoe Prior.

    Default dimension `d = 21`:
    - $p = 10$ regression weights $\beta_{0:p-1} \in \mathbb{R}^p$
    - $p = 10$ unconstrained local log-shrinkage scales $u_{0:p-1} = \log \lambda_{0:p-1} \in \mathbb{R}^p$
    - Global log-shrinkage parameter $w = \log \tau \in \mathbb{R}$

    Model specification:
        $$\tau = e^w \sim \text{Half-Cauchy}(0, \tau_0), \quad \lambda_j = e^{u_j} \sim \text{Half-Cauchy}(0, 1)$$
        $$\beta_j \mid u_j, w \sim \mathcal{N}(0, e^{2(u_j + w)}), \quad j = 0, \dots, p-1$$
        $$y_i \mid \beta \sim \text{Bernoulli}(\sigma(x_i^T \beta)), \quad i = 0, \dots, N-1$$
    Only 2 of the $p = 10$ covariates carry true nonzero signal, combining heavy-tailed
    Half-Cauchy shrinkage slabs with acute funnel necks on inactive coordinates.
    The design matrix and labels are simulated (default 80 observations).
    """

    def __init__(
        self,
        n_features: int = 10,
        n_obs: int = 80,
        n_active: int = 2,
        tau0: float = 0.5,
        seed: int = 202,
    ) -> None:
        if n_features < 2:
            raise ValueError(f"n_features must be >= 2, got {n_features}")
        if n_obs < 10:
            raise ValueError(f"n_obs must be >= 10, got {n_obs}")
        if n_active < 1 or n_active > n_features:
            raise ValueError(f"n_active must be in [1, {n_features}], got {n_active}")
        if tau0 <= 0.0:
            raise ValueError(f"tau0 must be positive, got {tau0}")

        self.n_features: int = int(n_features)
        self.n_obs: int = int(n_obs)
        self.n_active: int = int(n_active)
        self.tau0: float = float(tau0)
        self._log_tau0: float = float(np.log(self.tau0))

        super().__init__(d=2 * self.n_features + 1)

        rng = np.random.default_rng(seed)
        raw_x = rng.normal(0.0, 1.0, size=(self.n_obs, self.n_features))
        self.X: np.ndarray = ((raw_x - np.mean(raw_x, axis=0)) / np.std(raw_x, axis=0)).astype(np.float64)

        self.beta_true: np.ndarray = np.zeros(self.n_features, dtype=np.float64)
        signal_template = [2.4, -2.0, 1.8, -1.5]
        for j in range(self.n_active):
            self.beta_true[j] = signal_template[j % len(signal_template)]

        logits = self.X @ self.beta_true
        probs = _stable_sigmoid(logits)
        self.y: np.ndarray = rng.binomial(1, probs).astype(np.float64)

        self._log_prior_const: float = float(
            -0.5 * self.n_features * np.log(2.0 * np.pi)
            + self.n_features * np.log(2.0 / np.pi)
            + np.log(2.0 / (np.pi * self.tau0))
        )

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = self._validate_input(x)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        p = self.n_features
        beta = x_arr[:p]
        u = x_arr[p : 2 * p]
        w = float(x_arr[2 * p])

        if np.any(np.abs(u) > 100.0) or abs(w) > 100.0:
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        s = u + w
        inv_var = np.exp(-2.0 * s)
        scaled_beta_sq = (beta**2) * inv_var

        # 1. Conditional Gaussian prior on beta | u, w
        lp_beta_cond = -float(np.sum(s)) - 0.5 * float(np.sum(scaled_beta_sq))

        # 2. Half-Cauchy(0, 1) prior on lambda_j = exp(u_j): u_j - log(1 + exp(2 u_j))
        lp_u = float(np.sum(u - _stable_softplus(2.0 * u)))

        # 3. Half-Cauchy(0, tau0) prior on tau = exp(w): (w - log(tau0)) - log(1 + exp(2(w - log(tau0)))) + log(tau0)
        w_rel = w - self._log_tau0
        lp_w = float(w - float(np.logaddexp(0.0, 2.0 * w_rel)))

        # 4. Bernoulli logistic likelihood
        eta = self.X @ beta
        lp_lik = float(np.sum(self.y * eta - _stable_softplus(eta)))

        total_lp = float(self._log_prior_const + lp_beta_cond + lp_u + lp_w + lp_lik)
        if not np.isfinite(total_lp):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        probs = _stable_sigmoid(eta)
        grad = np.empty(self.d, dtype=np.float64)
        # d/d(beta)
        grad[:p] = -beta * inv_var + self.X.T @ (self.y - probs)
        # d/du_j: from conditional prior (-1 + beta_j^2 * exp(-2(u_j + w))) + from Half-Cauchy (-tanh(u_j))
        grad[p : 2 * p] = -1.0 + scaled_beta_sq - np.tanh(u)
        # d/dw: from conditional prior (-p + sum(beta_j^2 * exp(-2(u_j + w)))) + from Half-Cauchy (-tanh(w - log(tau0)))
        grad[2 * p] = -float(p) + float(np.sum(scaled_beta_sq)) - float(np.tanh(w_rel))

        return total_lp, grad

    def log_prob(self, x: np.ndarray) -> float:
        lp, _ = self.log_prob_and_grad(x)
        return lp

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        _, grad = self.log_prob_and_grad(x)
        return grad

    def jax_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX log-posterior evaluation for XLA / BlackJAX execution."""
        p = self.n_features
        beta = x[:p]
        u = x[p : 2 * p]
        w = x[2 * p]

        s = u + w
        inv_var = jnp.exp(-2.0 * s)
        scaled_beta_sq = (beta**2) * inv_var

        lp_beta_cond = -jnp.sum(s) - 0.5 * jnp.sum(scaled_beta_sq)
        lp_u = jnp.sum(u - jnp.logaddexp(0.0, 2.0 * u))
        w_rel = w - self._log_tau0
        lp_w = w - jnp.logaddexp(0.0, 2.0 * w_rel)

        X_jax = jnp.asarray(self.X, dtype=jnp.float64)
        y_jax = jnp.asarray(self.y, dtype=jnp.float64)
        eta = jnp.dot(X_jax, beta)
        lp_lik = jnp.sum(y_jax * eta - jnp.logaddexp(0.0, eta))

        return jnp.float64(self._log_prior_const) + lp_beta_cond + lp_u + lp_w + lp_lik

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        _raise_no_reference_posterior(self)

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        # Only the regression weights have data-generating values; the local and global
        # log-scales are latent prior quantities with no "true" value to report.
        return {
            "has_reference_posterior": False,
            "beta_true": self.beta_true,
            "n_features": self.n_features,
            "n_active": self.n_active,
        }


class RobustStudentTRegressionTarget(TargetDistribution):
    r"""Bayesian Robust Linear Regression with Student-$t$ Likelihood & Outlier Contamination.

    Default dimension `d = 12` (simulated data with gross outliers; not a PosteriorDB dataset):
    - Intercept $\beta_0 \in \mathbb{R}$
    - $p = 10$ regression slopes $\beta_{1:p} \in \mathbb{R}^p$ with heavy-tailed Cauchy priors
    - Unconstrained log observation scale $s = \log \sigma \in \mathbb{R}$

    Model specification:
        $$\beta_0 \sim \mathcal{N}(0, \sigma_0^2), \quad \beta_j \sim \text{Cauchy}(0, \gamma_\beta), \quad s \sim \mathcal{N}(0, \sigma_s^2)$$
        $$y_i \mid \beta_0, \beta, s \sim \text{Student-}t(\nu = 3, \mu_i = \beta_0 + x_i^T \beta, \sigma = e^s)$$
    Dataset includes 15% gross outlier contamination ($\pm 25\sigma$), creating local
    non-concavity in the likelihood and polynomial heavy tails in the Cauchy prior.
    """

    def __init__(
        self,
        n_features: int = 10,
        n_obs: int = 60,
        nu: float = 3.0,
        prior_scale: float = 2.5,
        sigma_0: float = 10.0,
        sigma_s: float = 2.0,
        outlier_fraction: float = 0.15,
        seed: int = 303,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if n_obs < 5:
            raise ValueError(f"n_obs must be >= 5, got {n_obs}")
        if nu <= 0.0 or prior_scale <= 0.0 or sigma_0 <= 0.0 or sigma_s <= 0.0:
            raise ValueError("nu and scale parameters must be strictly positive")
        if not (0.0 <= outlier_fraction < 1.0):
            raise ValueError(f"outlier_fraction must be in [0, 1), got {outlier_fraction}")

        self.n_features: int = int(n_features)
        self.n_obs: int = int(n_obs)
        self.nu: float = float(nu)
        self.prior_scale: float = float(prior_scale)
        self.sigma_0: float = float(sigma_0)
        self.sigma_s: float = float(sigma_s)
        self.outlier_fraction: float = float(outlier_fraction)

        super().__init__(d=self.n_features + 2)

        rng = np.random.default_rng(seed)
        self.X: np.ndarray = rng.normal(0.0, 1.0, size=(self.n_obs, self.n_features)).astype(np.float64)
        self.beta0_true: float = 1.25
        template = np.array([1.8, -1.4, 0.9, -0.7, 0.0, 0.0, 1.1, -0.6, 0.4, -0.9], dtype=np.float64)
        self.beta_true: np.ndarray = np.resize(template, self.n_features)
        self.log_sigma_true: float = -0.20

        y_clean = (
            self.beta0_true
            + self.X @ self.beta_true
            + np.exp(self.log_sigma_true) * rng.standard_t(df=self.nu, size=self.n_obs)
        )
        n_outliers = round(self.outlier_fraction * self.n_obs)
        if n_outliers > 0:
            outlier_idx = rng.choice(self.n_obs, size=n_outliers, replace=False)
            signs = rng.choice([-1.0, 1.0], size=n_outliers)
            y_clean[outlier_idx] += signs * 25.0
        self.y: np.ndarray = y_clean.astype(np.float64)

        self._log_lik_const: float = float(
            self.n_obs
            * (special.gammaln(0.5 * (self.nu + 1.0)) - special.gammaln(0.5 * self.nu) - 0.5 * np.log(self.nu * np.pi))
        )
        self._log_prior_const: float = float(
            -0.5 * np.log(2.0 * np.pi * (self.sigma_0**2))
            - self.n_features * np.log(np.pi * self.prior_scale)
            - 0.5 * np.log(2.0 * np.pi * (self.sigma_s**2))
        )

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = self._validate_input(x)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        p = self.n_features
        beta0 = float(x_arr[0])
        beta = x_arr[1 : p + 1]
        s = float(x_arr[p + 1])

        if s < -120.0 or s > 120.0:
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        inv_sig_sq = float(np.exp(-2.0 * s))
        res = self.y - (beta0 + self.X @ beta)
        res_sq = res**2
        scaled_res_sq = (res_sq * inv_sig_sq) / self.nu

        # Student-t log-likelihood
        lp_lik = self._log_lik_const - self.n_obs * s - 0.5 * (self.nu + 1.0) * float(np.sum(np.log1p(scaled_res_sq)))

        # Priors on beta0, beta_{1:p}, s
        gamma_sq = self.prior_scale**2
        lp_prior = (
            self._log_prior_const
            - 0.5 * (beta0**2) / (self.sigma_0**2)
            - float(np.sum(np.log1p((beta**2) / gamma_sq)))
            - 0.5 * (s**2) / (self.sigma_s**2)
        )

        total_lp = float(lp_lik + lp_prior)
        if not np.isfinite(total_lp):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        # Analytical weights w_i = (nu + 1) * exp(-2s) / (nu + e_i^2 * exp(-2s))
        weights = ((self.nu + 1.0) * inv_sig_sq) / (self.nu + res_sq * inv_sig_sq)
        weighted_res = weights * res

        grad = np.empty(self.d, dtype=np.float64)
        grad[0] = -beta0 / (self.sigma_0**2) + float(np.sum(weighted_res))
        grad[1 : p + 1] = -(2.0 * beta) / (gamma_sq + beta**2) + (self.X.T @ weighted_res)
        grad[p + 1] = -s / (self.sigma_s**2) - float(self.n_obs) + float(np.sum(weights * res_sq))

        return total_lp, grad

    def log_prob(self, x: np.ndarray) -> float:
        lp, _ = self.log_prob_and_grad(x)
        return lp

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        _, grad = self.log_prob_and_grad(x)
        return grad

    def jax_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX log-posterior evaluation for XLA / BlackJAX execution."""
        p = self.n_features
        beta0 = x[0]
        beta = x[1 : p + 1]
        s = x[p + 1]

        inv_sig_sq = jnp.exp(-2.0 * s)
        X_jax = jnp.asarray(self.X, dtype=jnp.float64)
        y_jax = jnp.asarray(self.y, dtype=jnp.float64)
        res = y_jax - (beta0 + jnp.dot(X_jax, beta))
        scaled_res_sq = ((res**2) * inv_sig_sq) / self.nu

        lp_lik = (
            jnp.float64(self._log_lik_const)
            - float(self.n_obs) * s
            - 0.5 * (self.nu + 1.0) * jnp.sum(jnp.log1p(scaled_res_sq))
        )
        gamma_sq = self.prior_scale**2
        lp_prior = (
            jnp.float64(self._log_prior_const)
            - 0.5 * (beta0**2) / (self.sigma_0**2)
            - jnp.sum(jnp.log1p((beta**2) / gamma_sq))
            - 0.5 * (s**2) / (self.sigma_s**2)
        )
        return lp_lik + lp_prior

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        _raise_no_reference_posterior(self)

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "has_reference_posterior": False,
            "data_generating_params": np.concatenate([[self.beta0_true], self.beta_true, [self.log_sigma_true]]),
            "beta0_true": self.beta0_true,
            "beta_true": self.beta_true,
            "log_sigma_true": self.log_sigma_true,
            "nu": self.nu,
        }


class HierarchicalItemResponse2PLTarget(TargetDistribution):
    r"""2-Parameter Logistic (2PL) Hierarchical Item Response Theory Model.

    Synthetic analogue of the PosteriorDB `irt_2pl` model: responses are simulated
    (default 10 students x 5 items), not the PosteriorDB dataset.

    Default dimension `d = 22`:
    - $I = 10$ student abilities $\theta_{0:I-1} \in \mathbb{R}^I$
    - $K = 5$ item difficulties $b_{0:K-1} \in \mathbb{R}^K$
    - $K = 5$ unconstrained log-discriminations $u_{0:K-1} = \log a_{0:K-1} \in \mathbb{R}^K$
    - Difficulty hyper-mean $\mu_b \in \mathbb{R}$
    - Log discrimination scale $v = \log \sigma_a \in \mathbb{R}$

    Model specification:
        $$\mu_b \sim \mathcal{N}(0, \sigma_{\mu_b}^2), \quad v \sim \mathcal{N}(0, \sigma_v^2)$$
        $$\theta_i \sim \mathcal{N}(0, 1), \quad b_k \mid \mu_b \sim \mathcal{N}(\mu_b, \sigma_b^2), \quad u_k \mid v \sim \mathcal{N}(0, e^{2v})$$
        $$Y_{ik} \mid \theta_i, b_k, u_k \sim \text{Bernoulli}(\sigma(e^{u_k}(\theta_i - b_k)))$$
    Combines multiplicative bilinear non-identifiability $e^{u_k}(\theta_i - b_k)$ with a
    hierarchical variance funnel in $(u_{0:K-1}, v)$.
    """

    def __init__(
        self,
        n_students: int = 10,
        n_items: int = 5,
        sigma_b: float = 2.0,
        sigma_mu_b: float = 2.0,
        sigma_v: float = 1.5,
        seed: int = 404,
    ) -> None:
        if n_students < 2:
            raise ValueError(f"n_students must be >= 2, got {n_students}")
        if n_items < 2:
            raise ValueError(f"n_items must be >= 2, got {n_items}")
        if sigma_b <= 0.0 or sigma_mu_b <= 0.0 or sigma_v <= 0.0:
            raise ValueError("Scale hyperparameters must be strictly positive")

        self.n_students: int = int(n_students)
        self.n_items: int = int(n_items)
        self.sigma_b: float = float(sigma_b)
        self.sigma_mu_b: float = float(sigma_mu_b)
        self.sigma_v: float = float(sigma_v)

        super().__init__(d=self.n_students + 2 * self.n_items + 2)

        rng = np.random.default_rng(seed)
        self.theta_true: np.ndarray = rng.normal(0.0, 1.0, size=self.n_students).astype(np.float64)
        self.mu_b_true: float = -0.25
        self.v_true: float = -0.60
        self.b_true: np.ndarray = (self.mu_b_true + rng.normal(0.0, 1.0, size=self.n_items)).astype(np.float64)
        self.u_true: np.ndarray = (np.exp(self.v_true) * rng.normal(0.0, 1.0, size=self.n_items)).astype(np.float64)

        a_true = np.exp(self.u_true)
        logits = (self.theta_true[:, None] - self.b_true[None, :]) * a_true[None, :]
        probs = _stable_sigmoid(logits)
        self.Y: np.ndarray = rng.binomial(1, probs).astype(np.float64)

        self._log_norm_const: float = float(
            -0.5 * self.n_students * np.log(2.0 * np.pi)
            - 0.5 * self.n_items * np.log(2.0 * np.pi * (self.sigma_b**2))
            - 0.5 * self.n_items * np.log(2.0 * np.pi)
            - 0.5 * np.log(2.0 * np.pi * (self.sigma_mu_b**2))
            - 0.5 * np.log(2.0 * np.pi * (self.sigma_v**2))
        )

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = self._validate_input(x)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        n_stud = self.n_students
        n_item = self.n_items
        theta = x_arr[:n_stud]
        b = x_arr[n_stud : n_stud + n_item]
        u = x_arr[n_stud + n_item : n_stud + 2 * n_item]
        mu_b = float(x_arr[n_stud + 2 * n_item])
        v = float(x_arr[n_stud + 2 * n_item + 1])

        if np.any(np.abs(u) > 50.0) or v < -120.0 or v > 120.0:
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        inv_sig_a_sq = float(np.exp(-2.0 * v))
        inv_sig_b_sq = 1.0 / (self.sigma_b**2)
        diff_b = b - mu_b
        u_sq_sum = float(np.dot(u, u))

        a = np.exp(u)
        diff_mat = theta[:, None] - b[None, :]  # shape (I, K)
        eta = diff_mat * a[None, :]  # shape (I, K)

        lp_priors = (
            -0.5 * float(np.dot(theta, theta))
            - 0.5 * inv_sig_b_sq * float(np.dot(diff_b, diff_b))
            - n_item * v
            - 0.5 * inv_sig_a_sq * u_sq_sum
            - 0.5 * (mu_b**2) / (self.sigma_mu_b**2)
            - 0.5 * (v**2) / (self.sigma_v**2)
        )
        lp_lik = float(np.sum(self.Y * eta - _stable_softplus(eta)))
        total_lp = float(self._log_norm_const + lp_priors + lp_lik)

        if not np.isfinite(total_lp):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        res_mat = self.Y - _stable_sigmoid(eta)  # shape (I, K)

        grad = np.empty(self.d, dtype=np.float64)
        # d/d(theta_i)
        grad[:n_stud] = -theta + (res_mat @ a)
        # d/d(b_k)
        grad[n_stud : n_stud + n_item] = -inv_sig_b_sq * diff_b - a * np.sum(res_mat, axis=0)
        # d/d(u_k)
        grad[n_stud + n_item : n_stud + 2 * n_item] = -inv_sig_a_sq * u + a * np.sum(res_mat * diff_mat, axis=0)
        # d/d(mu_b)
        grad[n_stud + 2 * n_item] = -mu_b / (self.sigma_mu_b**2) + inv_sig_b_sq * float(np.sum(diff_b))
        # d/dv
        grad[n_stud + 2 * n_item + 1] = -v / (self.sigma_v**2) - float(n_item) + inv_sig_a_sq * u_sq_sum

        return total_lp, grad

    def log_prob(self, x: np.ndarray) -> float:
        lp, _ = self.log_prob_and_grad(x)
        return lp

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        _, grad = self.log_prob_and_grad(x)
        return grad

    def jax_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX log-posterior evaluation for XLA / BlackJAX execution."""
        n_stud = self.n_students
        n_item = self.n_items
        theta = x[:n_stud]
        b = x[n_stud : n_stud + n_item]
        u = x[n_stud + n_item : n_stud + 2 * n_item]
        mu_b = x[n_stud + 2 * n_item]
        v = x[n_stud + 2 * n_item + 1]

        inv_sig_a_sq = jnp.exp(-2.0 * v)
        inv_sig_b_sq = 1.0 / (self.sigma_b**2)
        diff_b = b - mu_b
        u_sq_sum = jnp.sum(u**2)

        a = jnp.exp(u)
        diff_mat = theta[:, None] - b[None, :]
        eta = diff_mat * a[None, :]
        Y_jax = jnp.asarray(self.Y, dtype=jnp.float64)

        lp_priors = (
            -0.5 * jnp.sum(theta**2)
            - 0.5 * inv_sig_b_sq * jnp.sum(diff_b**2)
            - float(n_item) * v
            - 0.5 * inv_sig_a_sq * u_sq_sum
            - 0.5 * (mu_b**2) / (self.sigma_mu_b**2)
            - 0.5 * (v**2) / (self.sigma_v**2)
        )
        lp_lik = jnp.sum(Y_jax * eta - jnp.logaddexp(0.0, eta))
        return jnp.float64(self._log_norm_const) + lp_priors + lp_lik

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        _raise_no_reference_posterior(self)

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "has_reference_posterior": False,
            "data_generating_params": np.concatenate(
                [self.theta_true, self.b_true, self.u_true, [self.mu_b_true, self.v_true]]
            ),
            "n_students": self.n_students,
            "n_items": self.n_items,
        }


class StochasticVolatilityLogReturnTarget(TargetDistribution):
    r"""Kim, Shephard & Chib (1998) Canonical AR(1) Stochastic Volatility Model.

    Default dimension `d = 22`:
    - $T = 20$ latent log-volatilities $h_{0:T-1} \in \mathbb{R}^T$
    - Unconstrained AR(1) persistence $u_\phi = \operatorname{atanh}(\phi) \in \mathbb{R}$ where $\phi = \tanh(u_\phi) \in (-1, 1)$
    - Unconstrained log volatility innovation scale $v = \log \sigma_\eta \in \mathbb{R}$

    Model specification:
        $$u_\phi \sim \mathcal{N}(\mu_\phi, \sigma_{u_\phi}^2), \quad v \sim \mathcal{N}(\mu_v, \sigma_v^2)$$
        $$h_0 \mid u_\phi, v \sim \mathcal{N}\!\left(0, \frac{e^{2v}}{1 - \phi^2}\right), \quad h_t \mid h_{t-1}, u_\phi, v \sim \mathcal{N}(\phi h_{t-1}, e^{2v})$$
        $$y_t \mid h_t \sim \mathcal{N}(0, e^{h_t}), \quad t = 0, \dots, T-1$$
    Exhibits strong temporal latent correlation coupled with a state-space scale funnel as $v \to -\infty$.
    The return series is simulated (default T = 20), not a market dataset.
    """

    def __init__(
        self,
        n_steps: int = 20,
        mu_phi: float = 1.0,
        sigma_u_phi: float = 1.0,
        mu_v: float = -1.0,
        sigma_v: float = 1.0,
        seed: int = 505,
    ) -> None:
        if n_steps < 4:
            raise ValueError(f"n_steps must be >= 4, got {n_steps}")
        if sigma_u_phi <= 0.0 or sigma_v <= 0.0:
            raise ValueError("Prior standard deviations must be strictly positive")

        self.n_steps: int = int(n_steps)
        self.mu_phi: float = float(mu_phi)
        self.sigma_u_phi: float = float(sigma_u_phi)
        self.mu_v: float = float(mu_v)
        self.sigma_v: float = float(sigma_v)

        super().__init__(d=self.n_steps + 2)

        rng = np.random.default_rng(seed)
        self.u_phi_true: float = 1.25  # phi ~ 0.848
        self.phi_true: float = float(np.tanh(self.u_phi_true))
        self.v_true: float = -1.05  # sigma_eta ~ 0.35
        sigma_eta_true = float(np.exp(self.v_true))

        h = np.empty(self.n_steps, dtype=np.float64)
        h[0] = rng.normal(0.0, sigma_eta_true / np.sqrt(1.0 - self.phi_true**2))
        for t in range(1, self.n_steps):
            h[t] = self.phi_true * h[t - 1] + rng.normal(0.0, sigma_eta_true)
        self.h_true: np.ndarray = h
        # Add small floor to y^2 to avoid exact zero log-return degeneracy
        raw_y = np.exp(0.5 * h) * rng.normal(0.0, 1.0, size=self.n_steps)
        self.y: np.ndarray = np.where(np.abs(raw_y) < 1e-3, 1e-3, raw_y).astype(np.float64)
        self.y_sq: np.ndarray = self.y**2

        self._log_norm_const: float = float(
            -0.5 * np.log(2.0 * np.pi * (self.sigma_u_phi**2))
            - 0.5 * np.log(2.0 * np.pi * (self.sigma_v**2))
            - 0.5 * self.n_steps * np.log(2.0 * np.pi)
            - 0.5 * self.n_steps * np.log(2.0 * np.pi)
        )

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x_arr = self._validate_input(x)
        if not np.all(np.isfinite(x_arr)) or np.any(np.abs(x_arr) > 1e70):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        T = self.n_steps
        h = x_arr[:T]
        u_phi = float(x_arr[T])
        v = float(x_arr[T + 1])

        if np.any(np.abs(h) > 100.0) or abs(u_phi) > 50.0 or v < -120.0 or v > 120.0:
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        phi = float(np.tanh(u_phi))
        cosh_u = float(np.cosh(u_phi))
        sech_sq = 1.0 / (cosh_u * cosh_u)  # 1 - phi^2
        log_one_minus_phi_sq = -2.0 * float(np.log(cosh_u))
        inv_sig_eta_sq = float(np.exp(-2.0 * v))

        # 1. Hyperpriors on (u_phi, v)
        lp_hyper = -0.5 * ((u_phi - self.mu_phi) ** 2) / (self.sigma_u_phi**2) - 0.5 * ((v - self.mu_v) ** 2) / (
            self.sigma_v**2
        )

        # 2. Latent AR(1) prior on h_{0:T-1}
        h0 = float(h[0])
        delta = h[1:] - phi * h[:-1]
        delta_sq_sum = float(np.dot(delta, delta))
        h0_term = sech_sq * (h0**2)
        quad_h = h0_term + delta_sq_sum

        lp_latent = -T * v + 0.5 * log_one_minus_phi_sq - 0.5 * inv_sig_eta_sq * quad_h

        # 3. Observation log-likelihood y_t ~ N(0, exp(h_t))
        exp_neg_h = np.exp(-h)
        y_sq_exp_neg_h = self.y_sq * exp_neg_h
        lp_lik = -0.5 * float(np.sum(h)) - 0.5 * float(np.sum(y_sq_exp_neg_h))

        total_lp = float(self._log_norm_const + lp_hyper + lp_latent + lp_lik)
        if not np.isfinite(total_lp):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)

        grad = np.empty(self.d, dtype=np.float64)
        # Gradient wrt h_0..h_{T-1}
        grad_h = -0.5 + 0.5 * y_sq_exp_neg_h
        grad_h[0] -= inv_sig_eta_sq * sech_sq * h0
        grad_h[1:] -= inv_sig_eta_sq * delta
        grad_h[:-1] += inv_sig_eta_sq * phi * delta
        grad[:T] = grad_h

        # Gradient wrt u_phi
        grad[T] = (
            -(u_phi - self.mu_phi) / (self.sigma_u_phi**2)
            - phi
            + inv_sig_eta_sq * sech_sq * (phi * (h0**2) + float(np.dot(delta, h[:-1])))
        )

        # Gradient wrt v
        grad[T + 1] = -(v - self.mu_v) / (self.sigma_v**2) - float(T) + inv_sig_eta_sq * quad_h

        return total_lp, grad

    def log_prob(self, x: np.ndarray) -> float:
        lp, _ = self.log_prob_and_grad(x)
        return lp

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        _, grad = self.log_prob_and_grad(x)
        return grad

    def jax_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX log-posterior evaluation for XLA / BlackJAX execution."""
        T = self.n_steps
        h = x[:T]
        u_phi = x[T]
        v = x[T + 1]

        phi = jnp.tanh(u_phi)
        cosh_u = jnp.cosh(u_phi)
        sech_sq = 1.0 / (cosh_u * cosh_u)
        log_one_minus_phi_sq = -2.0 * jnp.log(cosh_u)
        inv_sig_eta_sq = jnp.exp(-2.0 * v)

        lp_hyper = -0.5 * ((u_phi - self.mu_phi) ** 2) / (self.sigma_u_phi**2) - 0.5 * ((v - self.mu_v) ** 2) / (
            self.sigma_v**2
        )

        h0 = h[0]
        delta = h[1:] - phi * h[:-1]
        quad_h = sech_sq * (h0**2) + jnp.sum(delta**2)
        lp_latent = -float(T) * v + 0.5 * log_one_minus_phi_sq - 0.5 * inv_sig_eta_sq * quad_h

        y_sq_jax = jnp.asarray(self.y_sq, dtype=jnp.float64)
        lp_lik = -0.5 * jnp.sum(h) - 0.5 * jnp.sum(y_sq_jax * jnp.exp(-h))

        return jnp.float64(self._log_norm_const) + lp_hyper + lp_latent + lp_lik

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        _raise_no_reference_posterior(self)

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "has_reference_posterior": False,
            "data_generating_params": np.concatenate([self.h_true, [self.u_phi_true, self.v_true]]),
            "h_true": self.h_true,
            "phi_true": self.phi_true,
            "v_true": self.v_true,
        }
