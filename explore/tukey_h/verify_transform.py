"""Mandatory correctness check of JaxTukeyHTransform against JAX autodiff (and SciPy) references.

Checks, for c in {0.5, 1, 4, 16, inf} and a z grid that reaches the float64 overflow edge of the map:
  1. Lambert W against scipy.special.lambertw, and the defining identity w + log w = log y beyond 1e300.
  2. Inverse: forward(inverse(z)) == z and inverse(forward(x)) == x (the latter divided by the map's
     condition number 1 + h z^2, which reaches 1400 at the overflow edge).
  3. Log-det: closed form vs slogdet(jax.jacfwd(inverse)) on a d=5 vector map (one c per coordinate), and,
     where x overflows, vs an autodiff reference built from the log-form L(z) = log|x - mu| (no overflow):
     log|dx/dz| = L(z) + log|L'(z)|.
  4. Pullback gradient: closed form at z (pullback_grad_z) vs jax.grad(transform.wrap_logdensity(f)); the
     x-input version forward_grad, which re-derives z from x, is reported separately.
  5. Normalisation: integral of pi_z over R equals 1 (scipy quad) for 1-D Student-t targets.
  6. Tail claim: d^2/dz^2 log pi_z -> -alpha / c for a Student-t (alpha = nu), via jax.hessian.

Usage: uv run python explore/tukey_h/verify_transform.py   (writes explore/tukey_h/results/verify_results.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import integrate, special

sys.path.insert(0, str(Path(__file__).parent))
from tukey_h import JaxTukeyHTransform, lambertw_from_log

C_VALUES = [0.5, 1.0, 4.0, 16.0, np.inf]
MU = np.array([0.3, -1.2, 0.0, 2.5, -0.7])
R = np.array([1.7, 0.4, 3.0, 1.0, 2.2])
NU = np.array([0.5, 1.0, 2.0, 5.0, 3.0])  # Student-t index per coordinate for the gradient check


def z_grid(c: float) -> np.ndarray:
    """Symmetric grid out to the overflow edge h z^2 / 2 = 700 (or |z| = 1e6 for c = inf), plus z = 0."""
    zmax = 1e6 if np.isinf(c) else float(np.sqrt(1400.0 * c))
    pos = np.unique(np.concatenate([np.linspace(0.0, min(zmax, 50.0), 201), np.geomspace(1e-8, zmax, 200)]))
    return np.concatenate([-pos[::-1], pos[1:]])


def logt(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(-0.5 * (NU + 1.0) * jnp.log1p(x * x / NU))


def main() -> None:
    out: dict = {"checks": {}}

    # 1. Lambert W
    ly = np.linspace(-700.0, 690.0, 4001)
    w_ours = np.asarray(lambertw_from_log(jnp.asarray(ly)))
    w_ref = np.real(special.lambertw(np.exp(ly)))
    rel = np.abs(w_ours - w_ref) / np.maximum(np.abs(w_ref), 1e-300)
    ly_big = np.geomspace(700.0, 1e300, 200)
    w_big = np.asarray(lambertw_from_log(jnp.asarray(ly_big)))
    resid = np.abs(w_big + np.log(w_big) - ly_big) / ly_big
    out["checks"]["lambertw"] = {
        "max_rel_err_vs_scipy": float(rel.max()),
        "max_rel_residual_logy_700_to_1e300": float(resid.max()),
        "pass": bool(rel.max() < 1e-13 and resid.max() < 1e-14),
    }

    # 2-4 per c (scalar maps), plus the vector map
    per_c = {}
    for c in C_VALUES:
        tr = JaxTukeyHTransform.from_c(0.3, 1.7, c)
        z = jnp.asarray(z_grid(c))
        x = jax.vmap(tr.inverse)(z)
        finite = np.isfinite(np.asarray(x))
        z_rt = np.asarray(jax.vmap(tr.forward)(x))
        rt_err = np.abs(z_rt - np.asarray(z))[finite] / np.maximum(1.0, np.abs(np.asarray(z)[finite]))
        x_rt = np.asarray(jax.vmap(tr.inverse)(jax.vmap(tr.forward)(x)))
        xrt_err = np.abs(x_rt - np.asarray(x))[finite] / np.maximum(1.0, np.abs(np.asarray(x)[finite]))
        # relative x error = (1 + h z^2) * relative z error: divide by that condition number
        cond = 1.0 + float(tr.h) * np.asarray(z)[finite] ** 2
        xrt_err_cond = xrt_err / cond

        # log-det vs jacfwd (scalar => derivative) where finite
        dxdz = jax.vmap(jax.grad(lambda t: tr.inverse(t)))(z)
        ld_ad = np.log(np.abs(np.asarray(dxdz)))
        ld_cf = np.asarray(jax.vmap(tr.log_jacobian_det)(z[:, None]))
        ok = finite & np.isfinite(ld_ad)
        ld_err = np.abs(ld_ad - ld_cf)[ok] / np.maximum(1.0, np.abs(ld_cf[ok]))

        # log-det vs log-form autodiff reference everywhere (z != 0)
        def logabs_xm(t: jnp.ndarray, tr: JaxTukeyHTransform = tr) -> jnp.ndarray:
            return jnp.log(tr.radius) + jnp.log(jnp.abs(t)) + 0.5 * tr.h * t * t

        nz = np.abs(np.asarray(z)) > 0
        zn = z[nz]
        lf = np.asarray(jax.vmap(logabs_xm)(zn)) + np.log(np.abs(np.asarray(jax.vmap(jax.grad(logabs_xm))(zn))))
        ld_cf_n = ld_cf[nz]
        lf_err = np.abs(lf - ld_cf_n) / np.maximum(1.0, np.abs(ld_cf_n))

        # pullback gradient: closed form vs jax.grad(wrap) on a 1-D Cauchy, where finite
        def f1(xx: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(-jnp.log1p(xx * xx))

        wrapped = tr.wrap_logdensity(f1)
        g_ad = np.asarray(jax.vmap(jax.grad(lambda t: wrapped(t[None])))(z))
        g_x = jax.vmap(jax.grad(lambda xx: f1(xx[None])))(x)
        g_cf = np.asarray(jax.vmap(tr.forward_grad)(x, g_x))
        g_cf_z = np.asarray(jax.vmap(tr.pullback_grad_z)(z, g_x))
        okg = finite & np.isfinite(g_ad) & np.isfinite(g_cf)
        g_err = np.abs(g_ad - g_cf)[okg] / np.maximum(1.0, np.abs(g_ad[okg]))
        g_err_z = np.abs(g_ad - g_cf_z)[okg] / np.maximum(1.0, np.abs(g_ad[okg]))

        per_c[str(c)] = {
            "n_grid": int(z.shape[0]),
            "z_max": float(np.max(np.asarray(z))),
            "n_x_finite": int(finite.sum()),
            "roundtrip_z_max_err": float(rt_err.max()),
            "roundtrip_x_max_err": float(xrt_err.max()),
            "roundtrip_x_max_err_over_condition_number": float(xrt_err_cond.max()),
            "logdet_vs_jacfwd_max_err": float(ld_err.max()),
            "logdet_vs_logform_autodiff_max_err_all_z": float(lf_err.max()),
            "pullback_grad_max_err_via_forward_x": float(g_err.max()),
            "pullback_grad_max_err": float(g_err_z.max()),
            "n_grad_checked": int(okg.sum()),
        }
    out["checks"]["scalar_maps"] = per_c

    # Vector map: d = 5 coordinates, one c each; slogdet(jacfwd) and full gradient vs closed form
    trv = JaxTukeyHTransform.from_c(MU, R, np.array(C_VALUES))
    rng = np.random.default_rng(0)
    zz = rng.normal(size=(2000, 5)) * np.array([3.0, 4.0, 8.0, 15.0, 1e3])
    zz = jnp.asarray(zz)
    jac = jax.vmap(jax.jacfwd(trv.inverse))(zz)
    sign, slog = jnp.linalg.slogdet(jac)
    ld = jax.vmap(trv.log_jacobian_det)(zz)
    okv = np.isfinite(np.asarray(slog))
    vec_ld_err = np.abs(np.asarray(slog) - np.asarray(ld))[okv] / np.maximum(1.0, np.abs(np.asarray(ld)[okv]))
    wrapped_v = trv.wrap_logdensity(logt)
    g_ad = np.asarray(jax.vmap(jax.grad(wrapped_v))(zz))
    xx = jax.vmap(trv.inverse)(zz)
    g_cf = np.asarray(jax.vmap(trv.forward_grad)(xx, jax.vmap(jax.grad(logt))(xx)))
    okg = np.all(np.isfinite(g_ad), axis=1) & np.all(np.isfinite(g_cf), axis=1)
    vec_g_err = np.abs(g_ad - g_cf)[okg] / np.maximum(1.0, np.abs(g_ad[okg]))
    out["checks"]["vector_map_d5"] = {
        "c": [str(c) for c in C_VALUES],
        "n_points": int(zz.shape[0]),
        "logdet_vs_slogdet_jacfwd_max_err": float(vec_ld_err.max()),
        "jacobian_offdiag_max_abs": float(np.max(np.abs(np.asarray(jac) * (1 - np.eye(5))))),
        "pullback_grad_max_err": float(vec_g_err.max()),
        "n_finite": int(okv.sum()),
    }

    # 5. Normalisation of the pullback density (scalar maps, Student-t nu in {0.5, 1, 2, 5})
    norm = {}
    for nu in [0.5, 1.0, 2.0, 5.0]:
        logc = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
        for c in C_VALUES:
            tr = JaxTukeyHTransform.from_c(0.3, 1.7, c)
            fz = jax.jit(
                tr.wrap_logdensity(
                    lambda xx, nu=nu, logc=logc: logc - 0.5 * (nu + 1) * jnp.log1p((xx[0] - 0.3) ** 2 / nu)
                )
            )

            def dens(t: float, fz=fz) -> float:
                return float(np.exp(fz(jnp.array([t]))))

            if np.isinf(c):
                val = sum(
                    integrate.quad(dens, a, b, limit=500)[0]
                    for a, b in [(-np.inf, -1e3), (-1e3, 0), (0, 1e3), (1e3, np.inf)]
                )
            else:
                zmax = float(np.sqrt(1400.0 * c))
                val = integrate.quad(dens, -zmax, zmax, limit=500, points=[0.0])[0]
            norm[f"nu={nu},c={c}"] = val
    out["checks"]["normalisation"] = {
        "integrals": norm,
        "max_abs_dev_from_1": float(max(abs(v - 1) for v in norm.values())),
    }

    # 6. Tail curvature of the pullback: d2/dz2 log pi_z -> -nu / c (Student-t, alpha = nu)
    curv = {}
    for nu in [0.5, 1.0, 3.0]:
        for c in [1.0, 4.0]:
            tr = JaxTukeyHTransform.from_c(0.0, 1.0, c)
            fz = tr.wrap_logdensity(lambda xx, nu=nu: -0.5 * (nu + 1) * jnp.log1p(xx[0] ** 2 / nu))
            hess = jax.hessian(lambda t, fz=fz: fz(jnp.array([t])))
            zs = [3.0 * np.sqrt(c), 6.0 * np.sqrt(c), 10.0 * np.sqrt(c)]
            curv[f"nu={nu},c={c}"] = {
                "target": -nu / c,
                "z": zs,
                "d2": [float(hess(jnp.float64(zv))) for zv in zs],
            }
    out["checks"]["tail_curvature"] = curv

    sc = out["checks"]["scalar_maps"]
    worst = {
        "roundtrip": max(
            max(v["roundtrip_z_max_err"], v["roundtrip_x_max_err_over_condition_number"]) for v in sc.values()
        ),
        "logdet": max(
            max(v["logdet_vs_jacfwd_max_err"], v["logdet_vs_logform_autodiff_max_err_all_z"]) for v in sc.values()
        ),
        "grad": max(v["pullback_grad_max_err"] for v in sc.values()),
    }
    vm = out["checks"]["vector_map_d5"]
    out["summary"] = {
        **{f"max_rel_err_{k}": v for k, v in worst.items()},
        "max_rel_err_vector_logdet": vm["logdet_vs_slogdet_jacfwd_max_err"],
        "max_rel_err_vector_grad": vm["pullback_grad_max_err"],
        "max_abs_normalisation_dev": out["checks"]["normalisation"]["max_abs_dev_from_1"],
        "all_pass": bool(
            out["checks"]["lambertw"]["pass"]
            and worst["roundtrip"] < 1e-12
            and worst["logdet"] < 1e-13
            and worst["grad"] < 1e-12
            and vm["logdet_vs_slogdet_jacfwd_max_err"] < 1e-12
            and vm["pullback_grad_max_err"] < 1e-10
            and out["checks"]["normalisation"]["max_abs_dev_from_1"] < 1e-6
        ),
    }
    path = Path(__file__).parent / "results" / "verify_results.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1))
    for k, v in curv.items():
        print(k, "target", v["target"], "d2", np.round(v["d2"], 4))


if __name__ == "__main__":
    main()
