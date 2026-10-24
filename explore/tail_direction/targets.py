"""Targets for the tail-direction spike, each with a JAX log-density, exact marginals and the oracle partition.

``rotated_cov`` is copied from the earlier ablation spike so that ``rot_t2_d10``, ``rot_t1_d10`` and
``rot_gauss_d20`` are exactly that spike's ``corr_t2``, ``corr_t1`` and ``corr_gauss``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import special, stats


def rotated_cov(d: int, cond: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = np.geomspace(1.0, cond, d)
    return (q * eig) @ q.T


@dataclass
class Target:
    name: str
    d: int
    logdensity: Callable[[jnp.ndarray], jnp.ndarray]
    # marginal i: ("t", nu, scale) or ("normal", sd); location 0 everywhere
    marginals: list[tuple]
    # oracle plan: radial blocks (elliptical heavy), coord (independent heavy), affine (light)
    oracle_plan: dict

    def quantiles(self, p: float) -> np.ndarray:
        out = []
        for m in self.marginals:
            if m[0] == "t":
                out.append(m[2] * stats.t.ppf(p, m[1]))
            else:
                out.append(m[1] * stats.norm.ppf(p))
        return np.asarray(out)


def _mvt_logdensity(nu: float, cov: np.ndarray) -> Callable:
    d = cov.shape[0]
    chol = jnp.asarray(np.linalg.cholesky(cov))
    const = float(special.gammaln((nu + d) / 2) - special.gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)) - float(
        np.sum(np.log(np.diag(np.linalg.cholesky(cov))))
    )

    def lp(x: jnp.ndarray) -> jnp.ndarray:
        w = jax.scipy.linalg.solve_triangular(chol, x, lower=True)
        return const - 0.5 * (nu + d) * jnp.log1p(jnp.sum(w * w) / nu)

    return lp


def _t1d_logdensity(nu: np.ndarray, scale: np.ndarray) -> Callable:
    nu_j, sc_j = jnp.asarray(nu), jnp.asarray(scale)
    const = float(
        np.sum(special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale))
    )

    def lp(x: jnp.ndarray) -> jnp.ndarray:
        return const - jnp.sum(0.5 * (nu_j + 1.0) * jnp.log1p((x / sc_j) ** 2 / nu_j))

    return lp


def _gauss_logdensity(cov: np.ndarray) -> Callable:
    chol = jnp.asarray(np.linalg.cholesky(cov))

    def lp(x: jnp.ndarray) -> jnp.ndarray:
        w = jax.scipy.linalg.solve_triangular(chol, x, lower=True)
        return -0.5 * jnp.sum(w * w)

    return lp


def rot_t(nu: float, d: int, cond: float = 25.0) -> Target:
    cov = rotated_cov(d, cond)
    return Target(
        name=f"rot_t{nu:g}_d{d}",
        d=d,
        logdensity=_mvt_logdensity(nu, cov),
        marginals=[("t", nu, float(np.sqrt(cov[i, i]))) for i in range(d)],
        oracle_plan={"radial": [list(range(d))]},
    )


def product_t(d: int = 10) -> Target:
    nu = np.linspace(1.0, 3.0, d)
    sc = np.geomspace(1.0, 5.0, d)
    return Target(
        name=f"prod_t_d{d}",
        d=d,
        logdensity=_t1d_logdensity(nu, sc),
        marginals=[("t", float(nu[i]), float(sc[i])) for i in range(d)],
        oracle_plan={"coord": list(range(d))},
    )


def mixed() -> Target:
    """5-d rotated MVT block (nu=1, cond 25), 5 independent Cauchy, 5 independent Gaussian; d = 15."""
    cov_b = rotated_cov(5, 25.0)
    lp_b = _mvt_logdensity(1.0, cov_b)
    c_sc = np.geomspace(1.0, 3.0, 5)
    lp_c = _t1d_logdensity(np.ones(5), c_sc)
    g_sd = np.geomspace(1.0, 3.0, 5)
    g_sd_j = jnp.asarray(g_sd)

    def lp(x: jnp.ndarray) -> jnp.ndarray:
        return lp_b(x[:5]) + lp_c(x[5:10]) - 0.5 * jnp.sum((x[10:] / g_sd_j) ** 2)

    marg = [("t", 1.0, float(np.sqrt(cov_b[i, i]))) for i in range(5)]
    marg += [("t", 1.0, float(c_sc[i])) for i in range(5)]
    marg += [("normal", float(g_sd[i])) for i in range(5)]
    return Target(
        name="mixed_d15",
        d=15,
        logdensity=lp,
        marginals=marg,
        oracle_plan={"radial": [list(range(5))], "coord": list(range(5, 10)), "affine": list(range(10, 15))},
    )


def rot_gauss(d: int = 20, cond: float = 100.0) -> Target:
    cov = rotated_cov(d, cond)
    return Target(
        name=f"rot_gauss_d{d}",
        d=d,
        logdensity=_gauss_logdensity(cov),
        marginals=[("normal", float(np.sqrt(cov[i, i]))) for i in range(d)],
        oracle_plan={"affine": list(range(d))},
    )


def make_target(name: str) -> Target:
    table = {
        "rot_t1_d10": lambda: rot_t(1.0, 10),
        "rot_t2_d10": lambda: rot_t(2.0, 10),
        "rot_t1_d20": lambda: rot_t(1.0, 20),
        "rot_t2_d20": lambda: rot_t(2.0, 20),
        "prod_t_d10": lambda: product_t(10),
        "mixed_d15": mixed,
        "rot_gauss_d20": lambda: rot_gauss(20, 100.0),
    }
    return table[name]()


ALL_TARGETS = ["rot_t1_d10", "rot_t2_d10", "rot_t1_d20", "rot_t2_d20", "prod_t_d10", "mixed_d15", "rot_gauss_d20"]
