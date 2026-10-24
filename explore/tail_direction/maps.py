r"""Tail-direction maps for NUTS pullbacks (exploration spike, not library code).

Every map is block-diagonal: the coordinates are split into disjoint blocks, and each block has one of
four kinds. Each map is written as the *inverse* x = psi(z) (sampling space z -> target space x), plus
its exact log |det J_psi(z)|.

- ``identity``: x_B = z_B.
- ``affine``:   x_B = mu + L z_B,                                log|det| = sum log L_ii.
- ``radial``:   x_B = mu + L (R sinhc(s) z_B),  s = ||z_B||,      log|det| = sum log L_ii + k log R
                + (k - 1) log sinhc(s) + log cosh(s),   sinhc(s) = sinh(s)/s,  k = |B|.
- ``coord``:    x_i = mu_i + R_i sinh(z_i)                         log|det| = sum (log R_i + log cosh z_i).
- ``lincoord``: x_B = mu + A (R * sinh(z_B)), A any invertible matrix (an oracle tail frame in the held-out
                rotated-product target),                          log|det| = log|det A| + sum (log R_i + log cosh z_i).

The radial map is the radial arcsinh: the whitened radius r = ||L^{-1}(x - mu)|| is sent to
s = arcsinh(r / R). A multivariate t with density ~ r^-(nu + k) becomes a density whose radius s has an
exp(-nu s) (Laplace-like) tail.

Stability: sinhc, log sinhc and log cosh are evaluated as functions of u = s^2 with a Taylor branch for
u < U_SMALL (the "double where" trick), so values *and gradients* are finite at z_B = 0 exactly.
The large-s branch uses expm1/log1p so there is no cancellation for moderate s and no overflow in the
log-terms; sinh(s) itself overflows only for s > ~710, where every target density here is -inf anyway.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

U_SMALL = 1e-3  # switch to Taylor series in u = s^2 below this (s < ~0.032)
LOG2 = float(np.log(2.0))


# ---------------------------------------------------------------------------------------------
# Stable scalar helpers in u = s^2
# ---------------------------------------------------------------------------------------------


def _safe_s(u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    small = u < U_SMALL
    u_safe = jnp.where(small, 1.0, u)
    return small, jnp.sqrt(u_safe)


def sinhc_from_u(u: jnp.ndarray) -> jnp.ndarray:
    """sinh(s)/s with s = sqrt(u)."""
    small, s = _safe_s(u)
    big = jnp.sinh(s) / s
    ser = 1.0 + u / 6.0 + u**2 / 120.0 + u**3 / 5040.0
    return jnp.where(small, ser, big)


def log_sinhc_from_u(u: jnp.ndarray) -> jnp.ndarray:
    """log(sinh(s)/s) = s + log((1 - e^{-2s}) / (2s))."""
    small, s = _safe_s(u)
    big = s + jnp.log(-jnp.expm1(-2.0 * s) / (2.0 * s))
    ser = u / 6.0 - u**2 / 180.0 + u**3 / 2835.0 - u**4 / 37800.0
    return jnp.where(small, ser, big)


def log_cosh_from_u(u: jnp.ndarray) -> jnp.ndarray:
    small, s = _safe_s(u)
    big = s - LOG2 + jnp.log1p(jnp.exp(-2.0 * s))
    ser = u / 2.0 - u**2 / 12.0 + u**3 / 45.0 - 17.0 * u**4 / 2520.0
    return jnp.where(small, ser, big)


def asinhc_from_u(u: jnp.ndarray) -> jnp.ndarray:
    """arcsinh(r)/r with r = sqrt(u) (used by the forward radial map)."""
    small, r = _safe_s(u)
    big = jnp.arcsinh(r) / r
    ser = 1.0 - u / 6.0 + 3.0 * u**2 / 40.0 - 5.0 * u**3 / 112.0 + 35.0 * u**4 / 1152.0
    return jnp.where(small, ser, big)


# Helpers used only by the analytic-gradient check (not by the sampler, which uses jax.grad).


def radial_jac_c2_from_u(u: jnp.ndarray) -> jnp.ndarray:
    """(cosh s - sinhc s) / s^2, so that d/dz [sinhc(s) z] = sinhc(s) I + c2 z z^T."""
    small, s = _safe_s(u)
    big = (jnp.cosh(s) - jnp.sinh(s) / s) / (s * s)
    ser = 1.0 / 3.0 + u / 30.0 + u**2 / 840.0
    return jnp.where(small, ser, big)


def radial_logdet_grad_coef_from_u(u: jnp.ndarray, k: int) -> jnp.ndarray:
    """g(s) with grad_z [(k-1) log sinhc(s) + log cosh(s)] = g(s) z."""
    small, s = _safe_s(u)
    big = ((k - 1) * (1.0 / jnp.tanh(s) - 1.0 / s) + jnp.tanh(s)) / s
    ser = (k - 1) * (1.0 / 3.0 - u / 45.0 + 2.0 * u**2 / 945.0) + (1.0 - u / 3.0 + 2.0 * u**2 / 15.0)
    return jnp.where(small, ser, big)


# ---------------------------------------------------------------------------------------------
# Block map
# ---------------------------------------------------------------------------------------------

KINDS = ("identity", "affine", "radial", "coord", "lincoord")


@dataclass
class Block:
    kind: str
    idx: tuple[int, ...]
    params: dict[str, np.ndarray] = field(default_factory=dict)

    def key(self) -> tuple:
        return (
            self.kind,
            self.idx,
            tuple((k, np.asarray(v).tobytes()) for k, v in sorted(self.params.items())),
        )


@dataclass
class BlockMap:
    d: int
    blocks: list[Block]
    label: str = ""

    def __post_init__(self) -> None:
        seen = sorted(i for b in self.blocks for i in b.idx)
        if seen != list(range(self.d)):
            raise ValueError(f"blocks must partition range({self.d}); got {seen}")
        for b in self.blocks:
            if b.kind not in KINDS:
                raise ValueError(b.kind)

    # Static structure (hashable) and dynamic parameters (pytree) for jit caching.
    def structure(self) -> tuple:
        return (self.d, tuple((b.kind, b.idx) for b in self.blocks))

    def param_tree(self) -> tuple:
        return tuple({k: jnp.asarray(v, dtype=jnp.float64) for k, v in b.params.items()} for b in self.blocks)

    def key(self) -> tuple:
        return (self.d, tuple(b.key() for b in self.blocks))

    def describe(self) -> list[dict]:
        out = []
        for b in self.blocks:
            item: dict = {"kind": b.kind, "idx": list(b.idx)}
            if "R" in b.params:
                r = np.asarray(b.params["R"])
                item["R"] = float(r) if r.ndim == 0 else [float(v) for v in r]
            out.append(item)
        return out


def make_fns(structure: tuple, params: tuple) -> tuple[Callable, Callable, Callable]:
    """Returns (inverse, forward, log_det) closures for a block map; traceable in ``params``."""
    d, blocks = structure
    idx_arrays = [np.asarray(idx, dtype=np.int32) for _, idx in blocks]

    def inverse(z: jnp.ndarray) -> jnp.ndarray:
        x = jnp.zeros((d,), dtype=jnp.float64)
        for (kind, _), ia, p in zip(blocks, idx_arrays, params):
            zb = z[ia]
            if kind == "identity":
                xb = zb
            elif kind == "affine":
                xb = p["mu"] + p["L"] @ zb
            elif kind == "radial":
                u = jnp.sum(zb * zb)
                xb = p["mu"] + p["L"] @ (p["R"] * sinhc_from_u(u) * zb)
            elif kind == "lincoord":
                xb = p["mu"] + p["A"] @ (p["R"] * jnp.sinh(zb))
            else:  # coord
                xb = p["mu"] + p["R"] * jnp.sinh(zb)
            x = x.at[ia].set(xb)
        return x

    def forward(x: jnp.ndarray) -> jnp.ndarray:
        z = jnp.zeros((d,), dtype=jnp.float64)
        for (kind, _), ia, p in zip(blocks, idx_arrays, params):
            xb = x[ia]
            if kind == "identity":
                zb = xb
            elif kind == "affine":
                zb = jax.scipy.linalg.solve_triangular(p["L"], xb - p["mu"], lower=True)
            elif kind == "radial":
                w = jax.scipy.linalg.solve_triangular(p["L"], xb - p["mu"], lower=True) / p["R"]
                zb = asinhc_from_u(jnp.sum(w * w)) * w
            elif kind == "lincoord":
                zb = jnp.arcsinh(jnp.linalg.solve(p["A"], xb - p["mu"]) / p["R"])
            else:
                zb = jnp.arcsinh((xb - p["mu"]) / p["R"])
            z = z.at[ia].set(zb)
        return z

    def log_det(z: jnp.ndarray) -> jnp.ndarray:
        tot = jnp.asarray(0.0, dtype=jnp.float64)
        for (kind, idx), ia, p in zip(blocks, idx_arrays, params):
            zb = z[ia]
            k = len(idx)
            if kind == "identity":
                continue
            if kind == "affine":
                tot = tot + jnp.sum(jnp.log(jnp.diag(p["L"])))
            elif kind == "radial":
                u = jnp.sum(zb * zb)
                tot = tot + (
                    jnp.sum(jnp.log(jnp.diag(p["L"])))
                    + k * jnp.log(p["R"])
                    + (k - 1) * log_sinhc_from_u(u)
                    + log_cosh_from_u(u)
                )
            elif kind == "lincoord":
                tot = tot + p["logabsdet_A"] + jnp.sum(jnp.log(p["R"]) + log_cosh_from_u(zb * zb))
            else:
                tot = tot + jnp.sum(jnp.log(p["R"]) + log_cosh_from_u(zb * zb))
        return tot

    return inverse, forward, log_det


def pullback(logdensity_x: Callable, structure: tuple, params: tuple) -> Callable:
    inverse, _, log_det = make_fns(structure, params)

    def logdensity_z(z: jnp.ndarray) -> jnp.ndarray:
        return logdensity_x(inverse(z)) + log_det(z)

    return logdensity_z


def analytic_pullback_grad(grad_x: Callable, structure: tuple, params: tuple) -> Callable:
    """Hand-derived J^T grad_x log pi(x) + grad_z log|det J| (for verification against jax.grad)."""
    d, blocks = structure
    inverse, _, _ = make_fns(structure, params)
    idx_arrays = [np.asarray(idx, dtype=np.int32) for _, idx in blocks]

    def g(z: jnp.ndarray) -> jnp.ndarray:
        x = inverse(z)
        gx = grad_x(x)
        out = jnp.zeros((d,), dtype=jnp.float64)
        for (kind, idx), ia, p in zip(blocks, idx_arrays, params):
            zb, gb = z[ia], gx[ia]
            k = len(idx)
            if kind == "identity":
                ob = gb
            elif kind == "affine":
                ob = p["L"].T @ gb
            elif kind == "radial":
                u = jnp.sum(zb * zb)
                v = p["L"].T @ gb
                jr_t_v = p["R"] * (sinhc_from_u(u) * v + radial_jac_c2_from_u(u) * zb * jnp.dot(zb, v))
                ob = jr_t_v + radial_logdet_grad_coef_from_u(u, k) * zb
            elif kind == "lincoord":
                ob = p["R"] * jnp.cosh(zb) * (p["A"].T @ gb) + jnp.tanh(zb)
            else:
                ob = p["R"] * jnp.cosh(zb) * gb + jnp.tanh(zb)
            out = out.at[ia].set(ob)
        return out

    return g


# ---------------------------------------------------------------------------------------------
# Phase 1 calibration: robust location / scale, regularised Tyler shape, tail diagnostics
# ---------------------------------------------------------------------------------------------


def robust_scale(xc: np.ndarray, min_scale: float = 0.1) -> np.ndarray:
    """Production rule (calibrate_jax_arcsinh_transform) without the 1.5 multiplier and clipping."""
    q75, q25 = np.quantile(xc, [0.75, 0.25], axis=0)
    iqr = (q75 - q25) / 1.3489795
    mad = np.median(np.abs(xc), axis=0) / 0.67448975
    q90 = np.quantile(np.abs(xc), 0.90, axis=0)
    s = np.where(iqr > min_scale, iqr, np.maximum(mad, q90))
    return np.where(np.isfinite(s) & (s > 0), s, 1.0)


def tyler_shape(y: np.ndarray, rho: float, n_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    r"""Regularised Tyler shape (Chen, Wiesel & Hero 2011; Pascal, Chitour & Quek 2014).

    S <- (1 - rho) (k/n) sum_i y_i y_i^T / (y_i^T S^{-1} y_i) + rho I, then S <- k S / tr S.
    ``y`` must already be centred (and here: robustly standardised, so that I is a sensible target).
    """
    n, k = y.shape
    keep = np.sum(y * y, axis=1) > 1e-24
    y = y[keep]
    n = y.shape[0]
    s_mat = np.eye(k)
    for _ in range(n_iter):
        q = np.einsum("ij,jk,ik->i", y, np.linalg.inv(s_mat), y)
        new = (1.0 - rho) * (k / n) * (y.T / q) @ y + rho * np.eye(k)
        new = k * new / np.trace(new)
        if np.max(np.abs(new - s_mat)) < tol:
            s_mat = new
            break
        s_mat = new
    return s_mat


def robust_scatter(x: np.ndarray, rho: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Coordinate-median location and Gaussian-consistent regularised-Tyler scatter Sigma = L L^T.

    Normalisation: median of the squared Mahalanobis radius equals the chi^2_k median, so for a Gaussian
    Sigma estimates the covariance and for a t it estimates a scatter of the same order.
    """
    from scipy.stats import chi2

    n, k = x.shape
    mu = np.median(x, axis=0)
    xc = x - mu
    sc = robust_scale(xc)
    y = xc / sc
    if rho is None:
        rho = float(np.clip(k / n, 0.02, 0.5))
    shape = tyler_shape(y, rho)
    sigma = shape * np.outer(sc, sc)
    m2 = np.einsum("ij,jk,ik->i", xc, np.linalg.inv(sigma), xc)
    kappa = float(np.median(m2) / chi2.median(k))
    sigma = sigma * kappa
    chol = np.linalg.cholesky(sigma)
    return mu, chol, {"rho": rho, "kappa": kappa, "cond": float(np.linalg.cond(sigma))}


def hill_alpha(a: np.ndarray, k_frac: float = 0.1, k_min: int = 10) -> float:
    """Hill estimate of the tail index of |a| from its top-k order statistics."""
    a = np.sort(np.abs(a))[::-1]
    n = a.shape[0]
    k = max(k_min, int(k_frac * n))
    k = min(k, n - 1)
    top, ref = a[:k], a[k]
    if ref <= 0:
        return float("nan")
    h = float(np.mean(np.log(top / ref)))
    return 1.0 / h if h > 0 else float("inf")


def chi_matrix(xc: np.ndarray, u: float = 0.9) -> np.ndarray:
    """Pairwise extremal co-occurrence chi_ij = P(|x_i| > q_i(u), |x_j| > q_j(u)) / (1 - u)."""
    a = np.abs(xc)
    q = np.quantile(a, u, axis=0)
    e = (a > q).astype(np.float64)
    cnt = e.sum(axis=0)
    co = e.T @ e
    return co / np.maximum(np.sqrt(np.outer(cnt, cnt)), 1.0)


def _exceed(a: np.ndarray, u: float) -> np.ndarray:
    return a > np.quantile(a, u)


def cluster_radius(xs: np.ndarray, members: Sequence[int]) -> np.ndarray:
    """Radius of a cluster in robustly standardised coordinates (|x_i| for a singleton)."""
    return np.linalg.norm(xs[:, list(members)], axis=1)


def cluster_chi(xs: np.ndarray, a: Sequence[int], b: Sequence[int], u: float) -> float:
    """Co-exceedance of two clusters' radii: P(r_A > q_A(u), r_B > q_B(u)) / (1 - u) (empirical)."""
    ea, eb = _exceed(cluster_radius(xs, a), u), _exceed(cluster_radius(xs, b), u)
    return float(np.sum(ea & eb) / max(1.0, np.sqrt(np.sum(ea) * np.sum(eb))))


def radius_linkage(xs: np.ndarray, u: float, tau: float) -> list[list[int]]:
    """Agglomerative clustering: repeatedly merge the two clusters whose *radii* co-exceed most, while
    that co-exceedance is above tau. Under independence between two clusters the statistic has the same
    null (mean 1 - u) whatever the cluster sizes, while inside an elliptical block the radii of growing
    sub-blocks become ever more tail-dependent (they share the mixing scale), so merging accelerates.
    """
    clusters = [[i] for i in range(xs.shape[1])]
    ex = np.stack([_exceed(cluster_radius(xs, c), u) for c in clusters], axis=1).astype(np.float64)
    while len(clusters) > 1:
        cnt = ex.sum(axis=0)
        chi = (ex.T @ ex) / np.maximum(np.sqrt(np.outer(cnt, cnt)), 1.0)
        np.fill_diagonal(chi, -np.inf)
        i, j = np.unravel_index(int(np.argmax(chi)), chi.shape)
        if chi[i, j] <= tau:
            break
        i, j = min(i, j), max(i, j)
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]
        ex = np.delete(ex, j, axis=1)
        ex[:, i] = _exceed(cluster_radius(xs, clusters[i]), u)
    return sorted(clusters, key=lambda c: c[0])


@dataclass
class DiagnosticConfig:
    # Defaults chosen on pilot seeds 100-109 by pilot_tune.py (results/pilot_tune.json).
    chi_u: float = 0.8  # exceedance level for co-occurrence
    tau: float = 0.4  # merge clusters while radius co-exceedance > tau (null mean is 1 - chi_u)
    alpha_single: float = 3.5  # singleton is heavy if Hill index of |x_i| < this
    alpha_block: float = 6.0  # multi-coordinate cluster is heavy if Hill index of its radius < this
    alpha_global: float | None = 6.0  # if set: Hill index of the full radius above this -> all light
    hill_k_frac: float = 0.1


def tail_diagnostic(x: np.ndarray, cfg: DiagnosticConfig) -> dict:
    """Phase 1 diagnostic -> partition {tail-dependent heavy blocks, independent heavy coords, light coords}.

    1. Robustly standardise (coordinate median, IQR scale).
    2. Cluster coordinates by co-exceedance of cluster radii (``radius_linkage``).
    3. Each cluster is heavy if the Hill index of its radius is below a size-dependent threshold.
    Heavy clusters of size >= 2 get a whitened radial map, heavy singletons a coordinate-wise arcsinh,
    and all light coordinates one whitened affine block.
    """
    n, d = x.shape
    xc = x - np.median(x, axis=0)
    xs = xc / robust_scale(xc)
    alpha_radius = float(hill_alpha(np.linalg.norm(xs, axis=1), cfg.hill_k_frac))
    if cfg.alpha_global is not None and alpha_radius > cfg.alpha_global:
        clusters = [list(range(d))]  # gate: no heavy tail anywhere -> one light cluster
    else:
        clusters = radius_linkage(xs, cfg.chi_u, cfg.tau)
    dep_blocks, singles, light, cl_alpha = [], [], [], []
    for c in clusters:
        a = hill_alpha(cluster_radius(xs, c), cfg.hill_k_frac)
        cl_alpha.append(float(a))
        thr = cfg.alpha_single if len(c) == 1 else cfg.alpha_block
        if a < thr:
            (dep_blocks.append(c) if len(c) >= 2 else singles.append(c[0]))
        else:
            light.extend(c)
    light = sorted(light)
    heavy = sorted([i for b in dep_blocks for i in b] + singles)
    if not light and len(dep_blocks) == 1 and not singles:
        choice = "radial"
    elif not light and not dep_blocks:
        choice = "coord"
    elif not heavy:
        choice = "affine"
    else:
        choice = "composite"
    chi = chi_matrix(xc, cfg.chi_u)
    off = chi[np.triu_indices(d, 1)]
    return {
        "clusters": clusters,
        "cluster_alpha": cl_alpha,
        "heavy": heavy,
        "light": light,
        "dep_blocks": dep_blocks,
        "singles": singles,
        "choice": choice,
        "alpha_hill_marginal": [float(hill_alpha(xs[:, i], cfg.hill_k_frac)) for i in range(d)],
        "alpha_hill_radius": float(hill_alpha(np.linalg.norm(xs, axis=1), cfg.hill_k_frac)),
        "chi_pair_mean_offdiag": float(np.mean(off)) if off.size else float("nan"),
        "n_draws": int(n),
    }


# ---------------------------------------------------------------------------------------------
# Map builders from Phase 1 draws
# ---------------------------------------------------------------------------------------------


def coord_params(x: np.ndarray, min_radius: float = 0.1, max_radius: float = 100.0, mult: float = 1.5) -> dict:
    """Same rule as calibrate_jax_arcsinh_transform (median, clipped 1.5 x IQR-scale)."""
    mu = np.median(x, axis=0)
    sc = robust_scale(x - mu, min_scale=min_radius)
    return {"mu": mu, "R": np.minimum(max_radius, np.maximum(min_radius, mult * sc))}


def radial_R(k: int, rule: str) -> float:
    if rule == "sqrtk":
        return float(np.sqrt(k))
    return float(rule)  # a constant radius in whitened (Gaussian-consistent) units, e.g. "1.5"


def build_map(x: np.ndarray, plan: dict, radial_rule: str, label: str) -> BlockMap:
    """plan: {'radial': [[idx...], ...], 'coord': [idx...], 'affine': [idx...], 'identity': [...]}"""
    d = x.shape[1]
    blocks: list[Block] = []
    for grp in plan.get("radial", []):
        grp = tuple(sorted(grp))
        if len(grp) == 1:  # a one-coordinate "radial" block is just coordinate-wise
            blocks.append(Block("coord", grp, coord_params(x[:, list(grp)])))
            continue
        mu, chol, _ = robust_scatter(x[:, list(grp)])
        blocks.append(Block("radial", grp, {"mu": mu, "L": chol, "R": np.asarray(radial_R(len(grp), radial_rule))}))
    if plan.get("coord"):
        grp = tuple(sorted(plan["coord"]))
        blocks.append(Block("coord", grp, coord_params(x[:, list(grp)])))
    if plan.get("affine"):
        grp = tuple(sorted(plan["affine"]))
        mu, chol, _ = robust_scatter(x[:, list(grp)])
        blocks.append(Block("affine", grp, {"mu": mu, "L": chol}))
    if plan.get("lincoord"):  # oracle tail frame: plan["lincoord"] = (idx, A)
        idx, a_mat = plan["lincoord"]
        grp = tuple(sorted(idx))
        xb = x[:, list(grp)]
        mu = np.median(xb, axis=0)
        src = np.linalg.solve(a_mat, (xb - mu).T).T
        r = np.minimum(100.0, np.maximum(0.1, 1.5 * robust_scale(src)))
        blocks.append(
            Block(
                "lincoord", grp, {"mu": mu, "A": a_mat, "R": r, "logabsdet_A": np.asarray(np.linalg.slogdet(a_mat)[1])}
            )
        )
    if plan.get("identity"):
        grp = tuple(sorted(plan["identity"]))
        blocks.append(Block("identity", grp, {}))
    return BlockMap(d=d, blocks=blocks, label=label)


def plan_from_diagnostic(diag: dict) -> dict:
    return {"radial": diag["dep_blocks"], "coord": diag["singles"], "affine": diag["light"]}
