r"""Exploration spike: Tukey-h ("Gaussianising") coordinate map as an exact NUTS pullback.

Exploration snapshot on branch explore/tukey-h-transform; not library code.

Per coordinate i, with h_i = 1 / c_i >= 0 (h_i = 0 is c_i = infinity, the affine map):

    x_i = mu_i + R_i * z_i * exp(h_i z_i^2 / 2)                         (inverse: z -> x)
    log |dx_i/dz_i| = log R_i + h_i z_i^2 / 2 + log1p(h_i z_i^2)        (log-det term)
    z_i = u_i * exp(-W(h_i u_i^2) / 2),  u_i = (x_i - mu_i) / R_i        (forward: x -> z)

The forward formula is sign(u) sqrt(W(h u^2) / h) rewritten with W(y)/y = exp(-W(y)), which
is valid at h = 0 (W(0) = 0 gives z = u) and never squares u (W is evaluated from log y).

A polynomial tail pi_x ~ |x|^-(alpha+1) pulls back to log pi_z = -alpha h z^2 / 2 + O(log|z|),
i.e. a Gaussian tail with variance c / alpha, whatever alpha is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import blackjax
import jax.numpy as jnp
import numpy as np

from mcmc_robust.integrations import JaxArcsinhTransform
from mcmc_robust.jax_unified_nuts import calibrate_jax_arcsinh_transform

# ---------------------------------------------------------------------------------------------
# Lambert W, principal branch, real argument y >= 0, evaluated from log y (no overflow)
# ---------------------------------------------------------------------------------------------

_W_NEWTON_ITERS = 12


def lambertw_from_log(log_y: jnp.ndarray) -> jnp.ndarray:
    r"""W_0(y) for y = exp(log_y) >= 0, accepting log_y = -inf (y = 0).

    Solves w + log w = log y by Newton's method (f concave increasing in w > 0).
    Initial guesses: log1p(y) for y <= e (an upper bound on W), log y - log log y for y > e
    (a lower bound on W), so the iteration is monotone. For log y < -30 the series
    W(y) = y - y^2 + 1.5 y^3 is exact to double precision and is used directly.
    """
    ly = jnp.asarray(log_y, dtype=jnp.float64)
    tiny = ly < -30.0
    y_small = jnp.exp(jnp.minimum(ly, -30.0))
    w_series = y_small * (1.0 - y_small * (1.0 - 1.5 * y_small))

    ly_c = jnp.clip(ly, -30.0, 1e300)
    y_mid = jnp.exp(jnp.minimum(ly_c, 1.0))
    w0_small = jnp.log1p(y_mid)
    ly_big = jnp.maximum(ly_c, 1.0)
    w0_big = ly_big - jnp.log(ly_big)
    w = jnp.where(ly_c <= 1.0, w0_small, w0_big)

    def body(_: int, w_cur: jnp.ndarray) -> jnp.ndarray:
        w_pos = jnp.maximum(w_cur, 1e-300)
        f = w_pos + jnp.log(w_pos) - ly_c
        fp = 1.0 + 1.0 / w_pos
        return jnp.maximum(w_pos - f / fp, 1e-300)

    w = jax.lax.fori_loop(0, _W_NEWTON_ITERS, body, w)
    return jnp.where(tiny, w_series, w)


# ---------------------------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class JaxTukeyHTransform:
    """Coordinate-wise Tukey-h map. `h = 1/c` per coordinate; h = 0 is the affine map."""

    location: jnp.ndarray
    radius: jnp.ndarray
    h: jnp.ndarray

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        return (self.location, self.radius, self.h), None

    @classmethod
    def tree_unflatten(cls, aux: Any, children: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> JaxTukeyHTransform:
        return cls(*children)

    @classmethod
    def from_c(cls, location: Any, radius: Any, c: Any) -> JaxTukeyHTransform:
        c_arr = jnp.asarray(c, dtype=jnp.float64)
        h = jnp.where(jnp.isinf(c_arr), 0.0, 1.0 / c_arr)
        return cls(
            location=jnp.asarray(location, dtype=jnp.float64),
            radius=jnp.asarray(radius, dtype=jnp.float64),
            h=h,
        )

    @property
    def c(self) -> jnp.ndarray:
        return jnp.where(self.h == 0.0, jnp.inf, 1.0 / jnp.where(self.h == 0.0, 1.0, self.h))

    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        """z -> x (the map NUTS composes with in the hot loop)."""
        z = jnp.asarray(z, dtype=jnp.float64)
        return self.location + self.radius * z * jnp.exp(0.5 * self.h * z * z)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """x -> z via Lambert W (used only at initialisation and calibration)."""
        x = jnp.asarray(x, dtype=jnp.float64)
        u = (x - self.location) / self.radius
        abs_u = jnp.abs(u)
        # log(h u^2) with h = 0 or u = 0 giving -inf, hence W = 0 and z = u.
        safe_h = jnp.where(self.h > 0.0, self.h, 1.0)
        safe_u = jnp.where(abs_u > 0.0, abs_u, 1.0)
        log_y = jnp.where((self.h > 0.0) & (abs_u > 0.0), jnp.log(safe_h) + 2.0 * jnp.log(safe_u), -jnp.inf)
        w = lambertw_from_log(log_y)
        return u * jnp.exp(-0.5 * w)

    def log_jacobian_terms(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float64)
        hz2 = self.h * z * z
        return jnp.log(self.radius) + 0.5 * hz2 + jnp.log1p(hz2)

    def log_jacobian_det(self, z: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.log_jacobian_terms(z), axis=-1)

    def forward_grad(self, x: jnp.ndarray, grad_x: jnp.ndarray) -> jnp.ndarray:
        r"""Analytical pullback gradient at z = forward(x):

        d/dz log pi_z = R exp(h z^2/2)(1 + h z^2) * d/dx log pi_x + h z + 2 h z / (1 + h z^2).
        """
        return self.pullback_grad_z(self.forward(x), grad_x)

    def pullback_grad_z(self, z: jnp.ndarray, grad_x: jnp.ndarray) -> jnp.ndarray:
        """Same closed form, evaluated at a given z (avoids the x -> z round trip)."""
        z = jnp.asarray(z, dtype=jnp.float64)
        hz2 = self.h * z * z
        dxdz = self.radius * jnp.exp(0.5 * hz2) * (1.0 + hz2)
        return dxdz * jnp.asarray(grad_x, dtype=jnp.float64) + self.h * z + 2.0 * self.h * z / (1.0 + hz2)

    def wrap_logdensity(
        self, logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def logdensity_z(z: jnp.ndarray) -> jnp.ndarray:
            return logdensity_fn(self.inverse(z)) + self.log_jacobian_det(z)

        return logdensity_z


# ---------------------------------------------------------------------------------------------
# Calibration rules for c (per coordinate) from Phase-1 draws
# ---------------------------------------------------------------------------------------------


def hill_alpha(pilot: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Hill tail-index estimate on |x - median| using the top `frac` order statistics."""
    a = np.sort(np.abs(pilot - np.median(pilot, axis=0)), axis=0)
    n = a.shape[0]
    k = max(5, int(frac * n))
    thresh = np.maximum(a[-k - 1], 1e-12)
    gamma = np.mean(np.log(np.maximum(a[-k:], 1e-300) / thresh), axis=0)
    return 1.0 / np.maximum(gamma, 1e-4)


def quantile_ratio(pilot: np.ndarray) -> np.ndarray:
    """q99 / q75 of |x - median| (the ratio test of CoordinateWiseArcsinhTransform.calibrate_from_samples)."""
    a = np.abs(pilot - np.median(pilot, axis=0))
    return np.quantile(a, 0.99, axis=0) / np.maximum(np.quantile(a, 0.75, axis=0), 1e-6)


_NU_GRID = np.geomspace(0.2, 200.0, 61)


def t_mle_nu(pilot: np.ndarray, n_em: int = 30) -> np.ndarray:
    """Profile-likelihood MLE of the Student-t degrees of freedom per coordinate.

    For each nu on a log grid, location and scale are fitted by the standard t EM iteration
    (weights (nu + 1) / (nu + r^2)), and the nu with the largest log-likelihood is returned.
    """
    from scipy import special

    n, d = pilot.shape
    out = np.empty(d)
    for j in range(d):
        x = pilot[:, j]
        best_ll, best_nu = -np.inf, _NU_GRID[-1]
        for nu in _NU_GRID:
            mu = float(np.median(x))
            s = float(max(np.quantile(np.abs(x - mu), 0.5) / 0.6745, 1e-8))
            for _ in range(n_em):
                r2 = ((x - mu) / s) ** 2
                w = (nu + 1.0) / (nu + r2)
                mu = float(np.sum(w * x) / np.sum(w))
                s = float(max(np.sqrt(np.sum(w * (x - mu) ** 2) / n), 1e-8))
            r2 = ((x - mu) / s) ** 2
            ll = n * (
                special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s)
            ) - 0.5 * (nu + 1) * float(np.sum(np.log1p(r2 / nu)))
            if ll > best_ll:
                best_ll, best_nu = ll, nu
        out[j] = best_nu
    return out


def tukey_h_mle(pilot: np.ndarray, mu: np.ndarray, r0: np.ndarray, h0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate MLE of (R, h) for the heavy-tail Lambert W x Gaussian (Tukey-h) model
    x = mu + R z exp(h z^2 / 2), z ~ N(0, 1), with mu fixed (Goerg 2015, Definition 2, delta = h).

    log f(x) = -z^2/2 - log(2 pi)/2 - log R - h z^2/2 - log1p(h z^2), z = u exp(-W(h u^2)/2), u = (x - mu)/R.
    Optimised over (log R, log h) by Nelder-Mead from (r0, h0); h is kept in [1e-3, 20].
    """
    from scipy import optimize, special

    n, d = pilot.shape
    r_out, h_out = np.empty(d), np.empty(d)
    for j in range(d):
        x = pilot[:, j] - mu[j]

        def nll(theta: np.ndarray, x: np.ndarray = x) -> float:
            log_r, log_h = theta
            r, h = np.exp(log_r), np.exp(np.clip(log_h, np.log(1e-3), np.log(20.0)))
            u = x / r
            w = np.real(special.lambertw(h * u * u))
            z2 = u * u * np.exp(-w)
            return float(np.sum(0.5 * z2 + log_r + 0.5 * h * z2 + np.log1p(h * z2)))

        res = optimize.minimize(
            nll,
            np.array([np.log(max(r0[j], 1e-8)), np.log(np.clip(h0[j], 2e-3, 10.0))]),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
        )
        r_out[j] = float(np.exp(res.x[0]))
        h_out[j] = float(np.exp(np.clip(res.x[1], np.log(1e-3), np.log(20.0))))
    return r_out, h_out


@dataclass
class CalibrationRule:
    """How to turn Phase-1 draws into c_i. `kind` selects the heavy/light test and the alpha estimate.

    - "hill":     heavy iff Hill alpha <= alpha_thr; c = kappa * alpha_hill.
    - "hill_qr":  heavy iff (q99/q75 > qr_thr) or (Hill alpha <= alpha_thr) (the repo's existing mask
                  rule in CoordinateWiseArcsinhTransform.calibrate_from_samples); c = kappa * alpha_hill.
    - "qr":       heavy iff q99/q75 > qr_thr; c = kappa * alpha_hill.
    - "tmle":     heavy iff t-MLE nu <= nu_thr; c = kappa * nu_mle.
    - "lw_mle":   heavy iff t-MLE nu <= nu_thr; on heavy coordinates (R, c) are fitted jointly by the
                  Tukey-h (Lambert W x Gaussian) MLE with mu = median, so that the pullback of the Phase-1
                  draws is as close to N(0, 1) as the family allows; R is returned and overrides the IQR rule.
    `force_all` marks every coordinate heavy (the light-tail cost arm); `c_mult` rescales c afterwards
    (the misspecification arms).
    """

    kind: str = "qr"
    kappa: float = 1.0
    alpha_thr: float = 3.5
    qr_thr: float = 3.0
    nu_thr: float = 5.0
    alpha_clip: tuple[float, float] = (0.2, 20.0)
    force_all: bool = False
    c_mult: float = 1.0
    name: str = field(default="")

    def compute(self, pilot: np.ndarray) -> dict[str, np.ndarray]:
        pilot = np.asarray(pilot, dtype=np.float64).reshape(-1, np.asarray(pilot).shape[-1])
        alpha_h = hill_alpha(pilot)
        qr = quantile_ratio(pilot)
        if self.kind in ("tmle", "lw_mle"):
            alpha_est = t_mle_nu(pilot)
            heavy = alpha_est <= self.nu_thr
        else:
            alpha_est = alpha_h
            if self.kind == "hill":
                heavy = alpha_h <= self.alpha_thr
            elif self.kind == "hill_qr":
                heavy = (qr > self.qr_thr) | (alpha_h <= self.alpha_thr)
            elif self.kind == "qr":
                heavy = qr > self.qr_thr
            else:
                raise ValueError(self.kind)
        if self.force_all:
            heavy = np.ones_like(heavy, dtype=bool)
        alpha_c = np.clip(alpha_est, *self.alpha_clip)
        c = np.where(heavy, self.kappa * self.c_mult * alpha_c, np.inf)
        out = {"alpha_hill": alpha_h, "qratio": qr, "alpha_est": alpha_est, "heavy": heavy, "c": c}
        if self.kind == "lw_mle":
            med = np.median(pilot, axis=0)
            a = np.abs(pilot - med)
            r0 = (np.quantile(pilot - med, 0.75, axis=0) - np.quantile(pilot - med, 0.25, axis=0)) / 1.3489795
            r0 = np.where(r0 > 1e-8, r0, np.median(a, axis=0) / 0.6745)
            r_fit, h_fit = tukey_h_mle(pilot, med, r0, 1.0 / alpha_c)
            out["R_override"] = np.where(heavy, r_fit, np.nan)  # NaN: keep the IQR-rule R
            out["c"] = np.where(heavy, self.c_mult / h_fit, np.inf)
            out["h_mle"] = h_fit
        return out


# ---------------------------------------------------------------------------------------------
# Pipeline: Phase 1 (identical to sample_blackjax_unified_nuts), then Phase 2/3 + sampling
# ---------------------------------------------------------------------------------------------


@dataclass
class Phase1:
    x_tail: np.ndarray  # (chains, W1 - W1//2, d) Phase-1 draws in x used for calibration
    x_final: jnp.ndarray  # (chains, d)
    grad_evals: int
    w2_steps: int
    key_p2: Any
    key_sample: Any
    arcsinh_transform: JaxArcsinhTransform  # the repo's calibrated arcsinh map (mu, R)


def run_phase1(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d: int,
    tune: int = 500,
    chains: int = 4,
    seed: int = 0,
    target_accept: float = 0.8,
    max_num_doublings: int = 10,
) -> Phase1:
    """Phase 1 exactly as in `sample_blackjax_unified_nuts` (same key splits, initial R=3 arcsinh map,
    W1 = max(10, 0.15 tune), calibration from the second half pooled across chains)."""
    master_key = jax.random.PRNGKey(int(seed))
    key_init, key_p1, key_p2, key_sample = jax.random.split(master_key, 4)
    init_x = 0.5 * jax.random.normal(key_init, shape=(chains, d), dtype=jnp.float64)
    transform_init = JaxArcsinhTransform(radii=3.0, location=0.0, mask=None, d=d)

    w1 = max(10, int(0.15 * tune))
    w2 = max(10, tune - w1)
    z0 = jax.vmap(transform_init.forward)(init_x)
    warm = blackjax.window_adaptation(
        blackjax.nuts,
        transform_init.wrap_logdensity(logdensity_fn),
        target_acceptance_rate=target_accept,
        max_num_doublings=max_num_doublings,
    )
    keys = jax.random.split(key_p1, chains)
    (state_p1, _), info_p1 = jax.vmap(lambda k, z: warm.run(k, z, num_steps=w1))(keys, z0)
    grads = int(chains + int(jnp.sum(info_p1.info.num_integration_steps)))
    half = max(1, w1 // 2)
    x_tail = jax.vmap(jax.vmap(transform_init.inverse))(info_p1.state.position[:, half:, :])
    arcsinh_tr = calibrate_jax_arcsinh_transform(pilot_samples=x_tail, d=d)
    x_final = jax.vmap(transform_init.inverse)(state_p1.position)
    return Phase1(
        x_tail=np.asarray(x_tail),
        x_final=x_final,
        grad_evals=grads,
        w2_steps=w2,
        key_p2=key_p2,
        key_sample=key_sample,
        arcsinh_transform=arcsinh_tr,
    )


@dataclass
class RunResult:
    samples: np.ndarray  # (chains, draws, d) in x
    z_samples: np.ndarray
    n_steps: np.ndarray  # (chains, draws) leapfrog steps per transition
    depth: np.ndarray  # (chains, draws) tree depth (num_trajectory_expansions)
    divergent: np.ndarray  # (chains, draws)
    grad_evals: int
    phase1_grad_evals: int
    phase2_grad_evals: int
    sampling_grad_evals: int
    divergences: int
    step_size: np.ndarray
    inverse_mass_matrix: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


def run_phase23(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    transform: Any,
    x_start: jnp.ndarray,
    w2_steps: int,
    key_p2: Any,
    key_sample: Any,
    draws: int = 1000,
    target_accept: float = 0.8,
    max_num_doublings: int = 10,
    divergence_threshold: float = 1000.0,
    phase1_grad_evals: int = 0,
) -> RunResult:
    """BlackJAX window adaptation per chain in z-space, then jit+vmap+scan sampling (as the repo does)."""
    chains = int(x_start.shape[0])
    logdensity_z = transform.wrap_logdensity(logdensity_fn)
    z_start = jax.vmap(transform.forward)(x_start)
    warm = blackjax.window_adaptation(
        blackjax.nuts, logdensity_z, target_acceptance_rate=target_accept, max_num_doublings=max_num_doublings
    )
    keys = jax.random.split(key_p2, chains)
    (state, params), info = jax.vmap(lambda k, z: warm.run(k, z, num_steps=w2_steps))(keys, z_start)
    p2_grads = int(chains + int(jnp.sum(info.info.num_integration_steps)))
    ss = jnp.asarray(params["step_size"], dtype=jnp.float64)
    imm = jnp.asarray(params["inverse_mass_matrix"], dtype=jnp.float64)

    def one_chain(init_state: Any, chain_ss: jnp.ndarray, chain_imm: jnp.ndarray, key: Any) -> Any:
        kernel = blackjax.nuts(
            logdensity_z,
            step_size=chain_ss,
            inverse_mass_matrix=chain_imm,
            max_num_doublings=max_num_doublings,
            divergence_threshold=divergence_threshold,
        ).step

        def step(st: Any, k: Any) -> Any:
            nst, inf = kernel(k, st)
            return nst, (nst.position, inf.is_divergent, inf.num_integration_steps, inf.num_trajectory_expansions)

        _, traj = jax.lax.scan(step, init_state, jax.random.split(key, draws))
        return traj

    z_draws, div, nsteps, depth = jax.jit(jax.vmap(one_chain))(state, ss, imm, jax.random.split(key_sample, chains))
    x_draws = jax.vmap(jax.vmap(transform.inverse))(z_draws)
    s_grads = int(jnp.sum(nsteps))
    return RunResult(
        samples=np.asarray(x_draws),
        z_samples=np.asarray(z_draws),
        n_steps=np.asarray(nsteps),
        depth=np.asarray(depth),
        divergent=np.asarray(div),
        grad_evals=int(phase1_grad_evals + p2_grads + s_grads),
        phase1_grad_evals=int(phase1_grad_evals),
        phase2_grad_evals=p2_grads,
        sampling_grad_evals=s_grads,
        divergences=int(jnp.sum(div)),
        step_size=np.asarray(ss),
        inverse_mass_matrix=np.asarray(imm),
    )


def sample_tukey_h_nuts(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d: int,
    rule: CalibrationRule,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    seed: int = 0,
    phase1: Phase1 | None = None,
    max_num_doublings: int = 10,
) -> RunResult:
    """Phase 1 (arcsinh, shared with the repo pipeline), calibrate (mu, R) as the repo does and c by `rule`,
    then BlackJAX window adaptation and sampling on the exact Tukey-h pullback."""
    p1 = phase1 or run_phase1(
        logdensity_fn, d, tune=tune, chains=chains, seed=seed, max_num_doublings=max_num_doublings
    )
    cal = rule.compute(p1.x_tail.reshape(-1, d))
    radius = np.asarray(p1.arcsinh_transform.radius)
    if "R_override" in cal:
        radius = np.where(np.isnan(cal["R_override"]), radius, cal["R_override"])
    tr = JaxTukeyHTransform.from_c(p1.arcsinh_transform.location, radius, cal["c"])
    res = run_phase23(
        logdensity_fn,
        tr,
        p1.x_final,
        p1.w2_steps,
        p1.key_p2,
        p1.key_sample,
        draws=draws,
        max_num_doublings=max_num_doublings,
        phase1_grad_evals=p1.grad_evals,
    )
    res.extra = {k: np.asarray(v).tolist() for k, v in cal.items()}
    res.extra["R"] = np.asarray(tr.radius).tolist()
    res.extra["mu"] = np.asarray(tr.location).tolist()
    return res


def sample_stock_nuts_with_stats(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    seed: int = 0,
    max_num_doublings: int = 10,
) -> RunResult:
    """Stock BlackJAX NUTS (identity map, full `tune` window adaptation) with per-draw statistics.

    Used by the cost-scaling study, which needs per-transition step counts; the main grid uses the repo's
    `run_blackjax_sampler` for the stock arm.
    """
    master_key = jax.random.PRNGKey(int(seed))
    key_init, _, key_p2, key_sample = jax.random.split(master_key, 4)
    init_x = 0.5 * jax.random.normal(key_init, shape=(chains, d), dtype=jnp.float64)
    ident = JaxTukeyHTransform.from_c(jnp.zeros(d), jnp.ones(d), jnp.full(d, jnp.inf))
    return run_phase23(
        logdensity_fn, ident, init_x, tune, key_p2, key_sample, draws=draws, max_num_doublings=max_num_doublings
    )
