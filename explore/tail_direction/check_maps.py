"""Mandatory map checks: inverse, log-det and pullback gradient against autodiff, near s = 0 and at large s.

Writes explore/tail_direction/results/check_maps.json and exits non-zero on any failure.
Run: uv run python -m explore.tail_direction.check_maps
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mcmc_robust.integrations import JaxArcsinhTransform

from .maps import (
    U_SMALL,
    Block,
    BlockMap,
    analytic_pullback_grad,
    asinhc_from_u,
    log_cosh_from_u,
    log_sinhc_from_u,
    make_fns,
    pullback,
    sinhc_from_u,
)
from .targets import _mvt_logdensity, rotated_cov

OUT = Path(__file__).parent / "results" / "check_maps.json"
S_GRID = [0.0, 1e-12, 1e-8, 1e-5, 1e-3, 0.0315, 0.0317, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
TOL = {"scalar_rel": 1e-12, "roundtrip_rel": 1e-10, "logdet_abs": 1e-9, "grad_rel": 1e-8}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| relative to max(max |b|, 1): relative for large values, absolute for small ones."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


def scalar_checks() -> dict:
    """Taylor/closed-form branches vs 50-digit Decimal reference, on both sides of U_SMALL."""
    getcontext().prec = 50
    out = {}
    worst = 0.0
    for s in [1e-9, 1e-6, 1e-3, 0.02, float(np.sqrt(U_SMALL)) * 0.999, float(np.sqrt(U_SMALL)) * 1.001, 0.3, 2.0, 40.0]:
        sd = Decimal(s)
        e, ei = sd.exp(), (-sd).exp()
        sinh, cosh = (e - ei) / 2, (e + ei) / 2
        ref = {
            "sinhc": float(sinh / sd),
            "log_sinhc": float((sinh / sd).ln()),
            "log_cosh": float(cosh.ln()),
            "asinhc": float((sd + (sd * sd + 1).sqrt()).ln() / sd),
        }
        u = jnp.asarray(s * s)
        got = {
            "sinhc": float(sinhc_from_u(u)),
            "log_sinhc": float(log_sinhc_from_u(u)),
            "log_cosh": float(log_cosh_from_u(u)),
            "asinhc": float(asinhc_from_u(u)),
        }
        errs = {k: abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-300) for k in ref}
        worst = max(worst, *errs.values())
        out[f"s={s:.6g}"] = errs
    return {"per_s": out, "max_rel_err": worst, "pass": worst < TOL["scalar_rel"]}


def build_test_map(d_r: int = 5, d_c: int = 5, d_a: int = 5, seed: int = 0) -> tuple[BlockMap, np.ndarray]:
    rng = np.random.default_rng(seed)

    def chol(k: int) -> np.ndarray:
        a = rng.normal(size=(k, k))
        return np.linalg.cholesky(a @ a.T / k + 0.5 * np.eye(k))

    d_l = 4
    d = d_r + d_c + d_a + d_l
    a_mat = rng.normal(size=(d_l, d_l)) + 2.0 * np.eye(d_l)  # general (non-triangular) frame
    blocks = [
        Block("radial", tuple(range(d_r)), {"mu": rng.normal(size=d_r), "L": chol(d_r), "R": np.asarray(1.3)}),
        Block("coord", tuple(range(d_r, d_r + d_c)), {"mu": rng.normal(size=d_c), "R": rng.uniform(0.2, 3.0, d_c)}),
        Block("affine", tuple(range(d_r + d_c, d_r + d_c + d_a)), {"mu": rng.normal(size=d_a), "L": chol(d_a)}),
        Block(
            "lincoord",
            tuple(range(d_r + d_c + d_a, d)),
            {
                "mu": rng.normal(size=d_l),
                "A": a_mat,
                "R": rng.uniform(0.2, 3.0, d_l),
                "logabsdet_A": np.asarray(np.linalg.slogdet(a_mat)[1]),
            },
        ),
    ]
    bm = BlockMap(d=d, blocks=blocks, label="test")
    cov = rotated_cov(d, 10.0, seed=3)
    return bm, cov


def map_checks() -> dict:
    results: dict = {"cases": []}
    worst = {"roundtrip_z": 0.0, "roundtrip_x": 0.0, "logdet": 0.0, "grad": 0.0, "grad_logdet_at_0_finite": True}
    rng = np.random.default_rng(1)
    for kind in ["composite", "radial_only_d20"]:
        if kind == "composite":
            bm, cov = build_test_map()
            rad_idx = np.arange(5)
        else:
            k = 20
            a = rng.normal(size=(k, k))
            chol = np.linalg.cholesky(a @ a.T / k + 0.5 * np.eye(k))
            bm = BlockMap(
                d=k,
                blocks=[Block("radial", tuple(range(k)), {"mu": rng.normal(size=k), "L": chol, "R": np.asarray(0.7)})],
            )
            cov = rotated_cov(k, 25.0, seed=4)
            rad_idx = np.arange(k)
        st, pt = bm.structure(), bm.param_tree()
        inverse, forward, log_det = make_fns(st, pt)
        lp_x = _mvt_logdensity(1.5, cov)
        lp_z = pullback(lp_x, st, pt)
        g_auto = jax.jit(jax.grad(lp_z))
        g_hand = jax.jit(analytic_pullback_grad(jax.grad(lp_x), st, pt))
        jac = jax.jit(jax.jacfwd(inverse))
        inv_j, fwd_j, ld_j = jax.jit(inverse), jax.jit(forward), jax.jit(log_det)
        d = bm.d
        for s in S_GRID:
            direction = rng.normal(size=rad_idx.size)
            direction /= np.linalg.norm(direction)
            z = rng.normal(size=d) * 0.7
            z[rad_idx] = s * direction
            z = jnp.asarray(z)
            x = inv_j(z)
            z_back = fwd_j(x)
            x_back = inv_j(z_back)
            sign, ld_auto = jnp.linalg.slogdet(jac(z))
            ld = ld_j(z)
            ga, gh = g_auto(z), g_hand(z)
            case = {
                "map": kind,
                "s": s,
                "roundtrip_z_abs": float(jnp.max(jnp.abs(z_back - z))),
                "roundtrip_x_rel": rel(x_back, x),
                "logdet": float(ld),
                "logdet_slogdet_jacfwd": float(ld_auto),
                "logdet_abs_err": float(abs(ld - ld_auto)),
                "jac_sign": float(sign),
                "grad_rel_err": rel(ga, gh),
                "grad_finite": bool(jnp.all(jnp.isfinite(ga))),
                "logdensity_z_finite": bool(jnp.isfinite(lp_z(z))),
            }
            results["cases"].append(case)
            # round-trip in z is only well-posed while the radius in x is representable (s < ~355)
            worst["roundtrip_z"] = max(worst["roundtrip_z"], case["roundtrip_z_abs"] / max(1.0, s))
            worst["roundtrip_x"] = max(worst["roundtrip_x"], case["roundtrip_x_rel"])
            worst["logdet"] = max(worst["logdet"], case["logdet_abs_err"] / max(1.0, abs(case["logdet"])))
            worst["grad"] = max(worst["grad"], case["grad_rel_err"] if case["grad_finite"] else np.inf)
        g0 = jax.grad(log_det)(jnp.zeros(d))
        worst["grad_logdet_at_0_finite"] = worst["grad_logdet_at_0_finite"] and bool(jnp.all(jnp.isfinite(g0)))

    # The coord block must agree with the production JaxArcsinhTransform (pullback and forward_grad).
    rngc = np.random.default_rng(7)
    mu, rr = rngc.normal(size=6), rngc.uniform(0.2, 3.0, 6)
    bm = BlockMap(d=6, blocks=[Block("coord", tuple(range(6)), {"mu": mu, "R": rr})])
    prod = JaxArcsinhTransform(radii=rr, location=mu)
    lp_x = _mvt_logdensity(2.0, rotated_cov(6, 5.0, seed=2))
    lp_mine = pullback(lp_x, bm.structure(), bm.param_tree())
    lp_prod = prod.wrap_logdensity(lp_x)
    coord_err = 0.0
    for scale in [0.0, 0.1, 1.0, 5.0, 30.0]:
        z = jnp.asarray(rngc.normal(size=6) * scale)
        x = prod.inverse(z)
        coord_err = max(
            coord_err,
            abs(float(lp_mine(z) - lp_prod(z))) / max(1.0, abs(float(lp_prod(z)))),
            rel(jax.grad(lp_mine)(z), prod.forward_grad(x, jax.grad(lp_x)(x))),
        )
    results["coord_vs_production_max_rel_err"] = coord_err
    results["worst"] = worst
    results["pass"] = bool(
        worst["roundtrip_z"] < TOL["roundtrip_rel"]
        and worst["roundtrip_x"] < TOL["roundtrip_rel"]
        and worst["logdet"] < TOL["logdet_abs"]
        and worst["grad"] < TOL["grad_rel"]
        and worst["grad_logdet_at_0_finite"]
        and coord_err < TOL["grad_rel"]
    )
    return results


def main() -> None:
    sc = scalar_checks()
    mc = map_checks()
    report = {"tolerances": TOL, "scalar": sc, "maps": mc, "pass": bool(sc["pass"] and mc["pass"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "scalar": {"max_rel_err": sc["max_rel_err"], "pass": sc["pass"]},
                "maps_worst": mc["worst"],
                "coord_vs_production": mc["coord_vs_production_max_rel_err"],
                "pass": report["pass"],
            },
            indent=1,
        )
    )
    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
