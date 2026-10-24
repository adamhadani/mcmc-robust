r"""Targets of the arcsinh-transform ablation (`bench/ablation_runner.py`), with exact marginals.

Each `AblationTarget` bundles a `TargetDistribution`, its exact one-dimensional marginal
distributions (for the tail-probability accuracy metric) and, for the t family, the Gaussian
scale-mixture reparameterisation a practitioner would sample instead (`ScaleMixtureModel`).

| Name | Target | d | Scale-mixture form |
|---|---|---|---|
| `task1` | `SparseSignalRegressionTarget(d=10, n_obs=40, n_active=3, seed=42)` (SOTA Task 1) | 10 | one $\lambda_j$ per Cauchy(0, s) prior |
| `task3` | `HeterogeneousHeavyTailedTarget(d=20)`, $t_{\nu_i}$ on 10 coordinates, $\nu$ = linspace(1.2, 2.5) | 20 | one $\lambda_j$ per t coordinate |
| `mvt_rot_nu1` | multivariate t, $\nu = 1$, scale $Q \mathrm{diag}(\mathrm{geomspace}(1, 25)) Q^\top$ | 10 | one shared $\lambda$ |
| `mvt_rot_nu2` | the same with $\nu = 2$ | 10 | one shared $\lambda$ |
| `mvt_iso_nu2` | multivariate t, $\nu = 2$, identity scale | 10 | one shared $\lambda$ |
| `rot_prod_cauchy` | `RotatedProductCauchyTarget(d=10)`: $x = Q \mathrm{diag}(s) u$, $u_j$ iid Cauchy, $s$ = geomspace(1, 5) | 10 | one $\lambda_j$ per $u_j$, scale matrix $Q \mathrm{diag}(s)$ |
| `gauss_rot` | Gaussian, covariance $Q \mathrm{diag}(\mathrm{geomspace}(1, 100)) Q^\top$ (condition number 100) | 20 | none |
| `gauss_aniso` | diagonal Gaussian, standard deviations geomspace(0.1, 10) | 20 | none |

$Q$ is a fixed-seed random orthogonal matrix (`ROTATION_SEED`). Marginal $i$ of a multivariate
$t_\nu(\mu, \Sigma)$ is the univariate $t_\nu$ with location $\mu_i$ and scale $\sqrt{\Sigma_{ii}}$.
Task 1's active coefficients have a Gaussian-likelihood-times-Cauchy marginal, tabulated on the
target's own quadrature grid; its inactive coefficients are exactly Cauchy(0, s).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import stats

from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.targets.benign import DiagonalGaussian, FullCovarianceGaussian
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
    RotatedProductCauchyTarget,
)
from mcmc_robust.targets.scale_mixture import ScaleMixtureModel

#: Seed of the fixed random rotation shared by the rotated targets.
ROTATION_SEED: int = 0

ABLATION_TARGET_NAMES: tuple[str, ...] = (
    "task1",
    "task3",
    "mvt_rot_nu1",
    "mvt_rot_nu2",
    "mvt_iso_nu2",
    "rot_prod_cauchy",
    "gauss_rot",
    "gauss_aniso",
)


class Marginal(Protocol):
    """A univariate distribution with a CDF and a quantile function (scipy frozen distributions qualify)."""

    def cdf(self, x: Any) -> Any: ...

    def ppf(self, q: Any) -> Any: ...


@dataclass(frozen=True)
class GridMarginal:
    """Univariate distribution given by a tabulated CDF on an increasing grid (linear interpolation)."""

    grid: np.ndarray
    cdf_values: np.ndarray

    def cdf(self, x: Any) -> Any:
        return np.interp(x, self.grid, self.cdf_values)

    def ppf(self, q: Any) -> Any:
        return np.interp(q, self.cdf_values, self.grid)


@dataclass(frozen=True)
class AblationTarget:
    """A target with exact marginals and, for the t family, its scale-mixture reparameterisation."""

    name: str
    description: str
    target: TargetDistribution
    heavy_tailed: bool
    marginals: tuple[Marginal, ...]
    scale_mixture: ScaleMixtureModel | None = None

    @property
    def d(self) -> int:
        return self.target.d

    def marginal_ppf(self, p: float) -> np.ndarray:
        """Exact marginal quantile $q_p$ of every coordinate, shape `(d,)`."""
        return np.array([float(m.ppf(p)) for m in self.marginals], dtype=np.float64)

    def marginal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Exact marginal CDF applied coordinate-wise to `x` of shape `(..., d)`."""
        arr = np.asarray(x, dtype=np.float64)
        return np.stack([np.asarray(m.cdf(arr[..., i])) for i, m in enumerate(self.marginals)], axis=-1)


def rotated_spd_matrix(eigenvalues: np.ndarray, seed: int = ROTATION_SEED) -> np.ndarray:
    r"""Returns $Q \mathrm{diag}(\text{eigenvalues}) Q^\top$ for a fixed-seed random orthogonal $Q$ (QR of a Gaussian matrix)."""
    eig = np.asarray(eigenvalues, dtype=np.float64)
    d = eig.shape[0]
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    m = (q * eig) @ q.T
    return 0.5 * (m + m.T)


def _gaussian_loglik_fn(x_mat: np.ndarray, y: np.ndarray, noise_std: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """JAX log-likelihood of y ~ N(X beta, noise_std^2 I)."""
    x_j = jnp.asarray(x_mat, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = float(y.shape[0])
    const = -0.5 * n * float(np.log(2.0 * np.pi * noise_std**2))
    inv_var = 1.0 / noise_std**2

    def loglik(beta: jnp.ndarray) -> jnp.ndarray:
        res = y_j - x_j @ beta
        return const - 0.5 * inv_var * jnp.sum(res**2)

    return loglik


def _task1() -> AblationTarget:
    # Imported here: bench.sota_adapters imports the targets package.
    from mcmc_robust.bench.sota_adapters import SparseSignalRegressionTarget

    t = SparseSignalRegressionTarget(d=10, n_obs=40, n_active=3, seed=42)
    marginals: list[Marginal] = []
    for j in range(t.d):
        if j < len(t._active_grids):
            marginals.append(GridMarginal(t._active_grids[j], t._active_cdfs[j]))
        else:
            marginals.append(stats.cauchy(loc=0.0, scale=t.prior_scale))
    sm = ScaleMixtureModel(
        loc=np.zeros(t.d),
        chol=t.prior_scale * np.eye(t.d),
        nu=np.ones(t.d),
        group=np.arange(t.d),
        loglik_fn=_gaussian_loglik_fn(t.X, t.y, t.noise_std),
    )
    return AblationTarget(
        name="task1",
        description="Sparse regression, Cauchy(0, 1) priors, 3 active of 10 coefficients (SOTA Task 1)",
        target=t,
        heavy_tailed=True,
        marginals=tuple(marginals),
        scale_mixture=sm,
    )


def _task3() -> AblationTarget:
    t = HeterogeneousHeavyTailedTarget(d=20, heavy_coords=list(range(10)), nu_list=list(np.linspace(1.2, 2.5, 10)))
    heavy = dict(zip(t.heavy_coords, t.nu_arr))
    marginals: list[Marginal] = [
        stats.t(df=heavy[i], loc=t.mu[i], scale=t.scale_arr[i]) if i in heavy else stats.norm(t.mu[i], t.scale_arr[i])
        for i in range(t.d)
    ]
    group = np.full(t.d, -1, dtype=np.int64)
    for k, i in enumerate(t.heavy_coords):
        group[i] = k
    sm = ScaleMixtureModel(loc=t.mu, chol=np.diag(t.scale_arr), nu=t.nu_arr, group=group)
    return AblationTarget(
        name="task3",
        description="Product form: t(nu) on 10 coordinates (nu = 1.2 to 2.5), N(0, 1) on 10 (SOTA Task 3)",
        target=t,
        heavy_tailed=True,
        marginals=tuple(marginals),
        scale_mixture=sm,
    )


def _mvt(name: str, nu: float, scale: np.ndarray | None, description: str) -> AblationTarget:
    t = MultivariateStudentTTarget(d=10, nu=nu, scale=scale)
    sd = np.sqrt(np.diag(t.cov))
    marginals = tuple(stats.t(df=nu, loc=t.mu[i], scale=sd[i]) for i in range(t.d))
    sm = ScaleMixtureModel(loc=t.mu, chol=t.L, nu=[nu], group=np.zeros(t.d, dtype=np.int64))
    return AblationTarget(name, description, t, True, marginals, sm)


def _rot_prod_cauchy() -> AblationTarget:
    t = RotatedProductCauchyTarget(d=10, rotation_seed=ROTATION_SEED)
    marginals = tuple(stats.cauchy(loc=0.0, scale=s) for s in t.marginal_scales)
    sm = ScaleMixtureModel(loc=np.zeros(t.d), chol=t.A, nu=np.ones(t.d), group=np.arange(t.d))
    return AblationTarget(
        "rot_prod_cauchy",
        "Rotated product of Cauchys, x = Q diag(s) u with u iid Cauchy and s = geomspace(1, 5), d = 10",
        t,
        True,
        marginals,
        sm,
    )


def _gauss_rot() -> AblationTarget:
    t = FullCovarianceGaussian(d=20, cov=rotated_spd_matrix(np.geomspace(1.0, 100.0, 20)))
    sd = np.sqrt(np.diag(t.cov))
    marginals = tuple(stats.norm(t.mean[i], sd[i]) for i in range(t.d))
    return AblationTarget("gauss_rot", "Rotated Gaussian, d = 20, covariance condition number 100", t, False, marginals)


def _gauss_aniso() -> AblationTarget:
    t = DiagonalGaussian(d=20, stds=np.geomspace(0.1, 10.0, 20))
    marginals = tuple(stats.norm(t.mean[i], t.stds[i]) for i in range(t.d))
    return AblationTarget(
        "gauss_aniso", "Axis-aligned Gaussian, d = 20, standard deviations 0.1 to 10", t, False, marginals
    )


#: Factories keyed by ablation target name.
ABLATION_TARGET_FACTORIES: dict[str, Callable[[], AblationTarget]] = {
    "task1": _task1,
    "task3": _task3,
    "mvt_rot_nu1": lambda: _mvt(
        "mvt_rot_nu1",
        1.0,
        rotated_spd_matrix(np.geomspace(1.0, 25.0, 10)),
        "Multivariate Cauchy (t, nu = 1), d = 10, rotated scale with condition number 25",
    ),
    "mvt_rot_nu2": lambda: _mvt(
        "mvt_rot_nu2",
        2.0,
        rotated_spd_matrix(np.geomspace(1.0, 25.0, 10)),
        "Multivariate t, nu = 2, d = 10, rotated scale with condition number 25",
    ),
    "mvt_iso_nu2": lambda: _mvt("mvt_iso_nu2", 2.0, None, "Isotropic multivariate t, nu = 2, d = 10"),
    "rot_prod_cauchy": _rot_prod_cauchy,
    "gauss_rot": _gauss_rot,
    "gauss_aniso": _gauss_aniso,
}


def make_ablation_target(name: str) -> AblationTarget:
    """Builds the ablation target registered under `name` (see `ABLATION_TARGET_NAMES`)."""
    try:
        factory = ABLATION_TARGET_FACTORIES[name]
    except KeyError:
        raise KeyError(f"Unknown ablation target {name!r}; choose from {ABLATION_TARGET_NAMES}") from None
    return factory()


__all__ = [
    "ABLATION_TARGET_FACTORIES",
    "ABLATION_TARGET_NAMES",
    "ROTATION_SEED",
    "AblationTarget",
    "GridMarginal",
    "Marginal",
    "make_ablation_target",
    "rotated_spd_matrix",
]
