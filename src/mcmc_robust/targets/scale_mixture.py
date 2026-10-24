r"""Non-centred Gaussian scale-mixture reparameterisation of Student-t and Cauchy models.

This is what a practitioner writes instead of a t density when they know the model is a
scale mixture of normals: an augmented, non-centred model

$$x = \mu + L \left(\sqrt{\lambda_{g(j)}}\, \eta_j\right)_{j=1}^{d}, \qquad
  \eta \sim N(0, I_d), \qquad \lambda_k \sim \mathrm{InvGamma}(\nu_k / 2, \nu_k / 2),$$

sampled on the unconstrained $\theta = (\eta, u)$ with $u_k = \log \lambda_k$. Marginally over
$\lambda_k$, the coordinates sharing group $k$ are multivariate $t_{\nu_k}$ with scale $L L^\top$:
- one shared $\lambda$ (all $g(j) = 0$) gives the multivariate t;
- one $\lambda_j$ per coordinate gives a product of univariate t's;
- $g(j) = -1$ marks a Gaussian coordinate with no mixing variable.

An optional likelihood $\ell(x)$ multiplies in (Task 1: a Cauchy(0, s) prior is $s \sqrt{\lambda_j} \eta_j$
with $\lambda_j \sim \mathrm{InvGamma}(1/2, 1/2)$, and the regression likelihood is unchanged).

Log density on $\theta$ (the $+u_k$ is the Jacobian of $\lambda_k = e^{u_k}$):

$$\log p(\theta) = \sum_j \log N(\eta_j; 0, 1)
  + \sum_k \left(a_k \log b_k - \log \Gamma(a_k) - a_k u_k - b_k e^{-u_k}\right) + \ell(x(\theta)),
  \qquad a_k = b_k = \nu_k / 2.$$
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import linalg, special


class ScaleMixtureModel:
    r"""Augmented non-centred scale-mixture model $\theta = (\eta, u) \mapsto x$ (see the module docstring).

    Parameters:
        loc: Location $\mu$, shape `(d,)`.
        chol: Lower-triangular scale factor $L$, shape `(d, d)`.
        nu: Degrees of freedom $\nu_k > 0$ of each mixing variable, shape `(K,)`.
        group: Mixing variable index $g(j) \in \{0, \dots, K-1\}$ of each coordinate, or $-1$ for a
            Gaussian coordinate, shape `(d,)`.
        loglik_fn: Optional JAX log-likelihood $\ell(x)$ added to the augmented log density.
    """

    def __init__(
        self,
        loc: np.ndarray | Sequence[float],
        chol: np.ndarray,
        nu: np.ndarray | Sequence[float],
        group: np.ndarray | Sequence[int],
        loglik_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> None:
        self.loc = np.asarray(loc, dtype=np.float64)
        self.chol = np.asarray(chol, dtype=np.float64)
        self.nu = np.asarray(nu, dtype=np.float64)
        self.group = np.asarray(group, dtype=np.int64)
        self.d = int(self.loc.shape[0])
        self.n_mix = int(self.nu.shape[0])
        self.d_aug = self.d + self.n_mix
        if self.chol.shape != (self.d, self.d) or not np.allclose(self.chol, np.tril(self.chol)):
            raise ValueError(f"chol must be lower-triangular with shape ({self.d}, {self.d})")
        if self.group.shape != (self.d,) or np.any(self.group < -1) or np.any(self.group >= self.n_mix):
            raise ValueError(f"group must have shape ({self.d},) with entries in [-1, {self.n_mix - 1}]")
        if np.any(self.nu <= 0.0):
            raise ValueError("nu must be strictly positive")
        self.loglik_fn = loglik_fn

        half_nu = 0.5 * self.nu
        # Constant of the log-InvGamma density on u: a log b - log Gamma(a), with a = b = nu / 2.
        self._mix_const = jnp.asarray(half_nu * np.log(half_nu) - special.gammaln(half_nu))
        self._a = jnp.asarray(half_nu)
        self._loc_j = jnp.asarray(self.loc)
        self._chol_j = jnp.asarray(self.chol)
        self._mixed = jnp.asarray(self.group >= 0)
        self._group_idx = jnp.asarray(np.clip(self.group, 0, max(self.n_mix - 1, 0)))

    def to_x(self, theta: jnp.ndarray) -> jnp.ndarray:
        r"""Maps one augmented point $\theta = (\eta, u)$ of shape `(d_aug,)` to $x$ of shape `(d,)`."""
        eta = theta[: self.d]
        u = theta[self.d :]
        log_scale = jnp.where(self._mixed, 0.5 * u[self._group_idx], 0.0)
        return self._loc_j + self._chol_j @ (jnp.exp(log_scale) * eta)

    def logdensity(self, theta: jnp.ndarray) -> jnp.ndarray:
        r"""Augmented log density $\log p(\theta)$ (module docstring), JAX-traceable."""
        eta = theta[: self.d]
        u = theta[self.d :]
        lp = -0.5 * jnp.sum(eta**2) - 0.5 * self.d * jnp.log(2.0 * jnp.pi)
        lp = lp + jnp.sum(self._mix_const - self._a * u - self._a * jnp.exp(-u))
        if self.loglik_fn is not None:
            lp = lp + self.loglik_fn(self.to_x(theta))
        return lp

    def init_from_x(self, x0: np.ndarray) -> np.ndarray:
        r"""Augmented starting points mapping exactly to `x0` (shape `(chains, d)`): $u = 0$, $\eta = L^{-1}(x_0 - \mu)$."""
        x_arr = np.atleast_2d(np.asarray(x0, dtype=np.float64))
        eta = linalg.solve_triangular(self.chol, (x_arr - self.loc).T, lower=True).T
        return np.concatenate([eta, np.zeros((x_arr.shape[0], self.n_mix))], axis=1)

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        r"""Exact draws of the augmented prior ($\eta \sim N(0, I)$, $\lambda_k \sim \mathrm{InvGamma}(\nu_k/2, \nu_k/2)$), shape `(n, d_aug)`.

        Without a likelihood this is the full augmented model, so `to_x` of these draws is
        distributed exactly as the target.
        """
        eta = rng.normal(size=(n, self.d))
        half_nu = 0.5 * self.nu
        # lambda = b / G with G ~ Gamma(a, 1) is InvGamma(a, b); here a = b = nu / 2.
        lam = half_nu / rng.gamma(shape=half_nu, size=(n, self.n_mix))
        return np.concatenate([eta, np.log(lam)], axis=1)
