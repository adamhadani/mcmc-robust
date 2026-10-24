"""Main experiment: stock NUTS vs coordinate arcsinh vs whitened / radial / composite / diagnostic-selected maps.

Run: uv run python -m explore.tail_direction.run_experiment OUT.jsonl TARGETS SEEDS [--radial-rule 0.5]
     [--arms stock,stock_dense,coord,whiten,radial,composite,selected]
Appends one JSON line per (target, seed, arm) to OUT.jsonl.

Arms (all: tune 500, draws 1000, 4 chains, max 10 doublings, BlackJAX window adaptation):
- stock:     identity map, 500 window-adaptation steps (diagonal metric, BlackJAX default).
- stock_dense: same with a dense metric (BlackJAX ``is_mass_matrix_diagonal=False``); baseline for whitening.
- coord:     ``sample_blackjax_unified_nuts`` (production coordinate-wise arcsinh).
- whiten:    Phase 1, then a global regularised-Tyler affine whitening (ablation: dense-metric effect only).
- radial:    Phase 1, then global whitening + radial arcsinh (R = --radial-rule in whitened units).
- composite: Phase 1, then the block map for the *oracle* partition (tail-dependent block -> whitened radial,
             independent heavy coordinates -> coordinate arcsinh, light coordinates -> whitened affine).
- selected:  Phase 1, then the block map for the partition found by the Phase 1 diagnostic.
Phase 1 is the production Phase 1 (identical draws, checked per run), shared by the four Phase-1 arms;
its gradient cost is charged to each of them. Maps that are identical (same blocks and parameters) are run
once and the record says ``same_map_as``.
"""

from __future__ import annotations

import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import numpy as np

from mcmc_robust.bench.sota_adapters import evaluate_chains_with_arviz
from mcmc_robust.jax_unified_nuts import calibrate_jax_arcsinh_transform, sample_blackjax_unified_nuts

from .maps import DiagnosticConfig, build_map, plan_from_diagnostic, tail_diagnostic
from .pipeline import run_map, run_phase1, run_stock
from .targets import Target, make_target

TUNE, DRAWS, CHAINS, MAX_DOUBLINGS = 500, 1000, 4, 10
TAIL_PS = (0.01, 0.99)


def tail_prob_metrics(samples: np.ndarray, target: Target) -> dict:
    """|P_hat(x_i < q_p) - p| at exact marginal quantiles, with MCSE from the indicator's ESS."""
    out: dict = {}
    errs, mcses, zs = [], [], []
    for p in TAIL_PS:
        q = target.quantiles(p)
        ind = (samples < q[None, None, :]).astype(np.float64)
        phat = ind.mean(axis=(0, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = az.ess(az.from_dict({"posterior": {"x": ind}}), var_names=["x"], method="mean")["x"].values
            mcse = np.sqrt(phat * (1 - phat) / ess)
        err = np.abs(phat - p)
        out[f"tp_err_{p}"] = [float(v) for v in err]
        out[f"tp_mcse_{p}"] = [float(v) for v in mcse]
        errs.append(err)
        mcses.append(mcse)
        with np.errstate(divide="ignore", invalid="ignore"):
            zs.append(err / mcse)
    e, m, z = np.concatenate(errs), np.concatenate(mcses), np.concatenate(zs)
    out["tp_err_mean"] = float(np.mean(e))
    out["tp_err_max"] = float(np.max(e))
    out["tp_mcse_mean"] = float(np.nanmean(m)) if np.any(np.isfinite(m)) else float("nan")
    out["tp_z_median"] = float(np.nanmedian(z[np.isfinite(z)])) if np.any(np.isfinite(z)) else float("nan")
    out["tp_mcse_undefined"] = int(np.sum(~np.isfinite(m)))
    return out


def metrics(samples: np.ndarray, grads: int, divs: int, target: Target, wall: float) -> dict:
    m = evaluate_chains_with_arviz(samples, grads, target=None, divergences=divs, wall_clock_time=wall)
    keep = [
        "total_grad_evals",
        "bulk_ess",
        "tail_ess",
        "min_bulk_ess",
        "min_tail_ess",
        "split_rhat",
        "combined_rhat",
        "ess_per_grad",
        "tail_ess_per_grad",
        "divergences",
        "wall_clock_time",
    ]
    out = {k: float(m[k]) for k in keep}
    out.update(tail_prob_metrics(samples, target))
    return out


def run_one(target: Target, seed: int, arms: list[str], radial_rule: str, cfg: DiagnosticConfig) -> list[dict]:
    rows: list[dict] = []
    base = {"target": target.name, "seed": seed, "radial_rule": radial_rule}

    for arm in [a for a in ("stock", "stock_dense") if a in arms]:
        t0 = time.perf_counter()
        r = run_stock(
            target.name,
            target.logdensity,
            target.d,
            seed,
            TUNE,
            DRAWS,
            CHAINS,
            MAX_DOUBLINGS,
            dense=arm == "stock_dense",
        )
        wall = time.perf_counter() - t0
        rows.append(
            {
                **base,
                "arm": arm,
                **metrics(r["samples"], r["grad_evals"], r["divergences"], target, wall),
                "max_depth_hits": r["max_depth_hits"],
                "mean_leapfrog": r["mean_leapfrog"],
            }
        )
        print(
            json.dumps({k: rows[-1][k] for k in ("target", "seed", "arm", "tail_ess_per_grad", "split_rhat")}),
            flush=True,
        )

    coord_transform = None
    if "coord" in arms:
        t0 = time.perf_counter()
        res = sample_blackjax_unified_nuts(
            target.logdensity,
            d=target.d,
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            seed=seed,
            max_num_doublings=MAX_DOUBLINGS,
        )
        wall = time.perf_counter() - t0
        coord_transform = res.transform
        n_steps = np.asarray(res.idata["sample_stats"]["n_steps"].values)
        rows.append(
            {
                **base,
                "arm": "coord",
                **metrics(res.samples, res.grad_evals, res.divergences, target, wall),
                "max_depth_hits": int(np.sum(n_steps >= 2**MAX_DOUBLINGS - 1)),
                "mean_leapfrog": float(np.mean(n_steps)),
                "phase1_grad_evals": res.phase1_grad_evals,
            }
        )
        print(
            json.dumps({k: rows[-1][k] for k in ("target", "seed", "arm", "tail_ess_per_grad", "split_rhat")}),
            flush=True,
        )

    p1_arms = [a for a in arms if a in ("whiten", "radial", "composite", "selected")]
    if not p1_arms:
        return rows
    t0 = time.perf_counter()
    p1 = run_phase1(target.logdensity, target.d, seed, TUNE, CHAINS, MAX_DOUBLINGS)
    p1_wall = time.perf_counter() - t0
    x = p1["x_tail"]
    phase1_matches = None
    if coord_transform is not None:
        mine = calibrate_jax_arcsinh_transform(x, d=target.d)
        phase1_matches = bool(
            np.array_equal(np.asarray(mine.radii), np.asarray(coord_transform.radii))
            and np.array_equal(np.asarray(mine.location), np.asarray(coord_transform.location))
        )
    diag = tail_diagnostic(x, cfg)
    all_idx = list(range(target.d))
    plans = {
        "whiten": {"affine": all_idx},
        "radial": {"radial": [all_idx]},
        "composite": target.oracle_plan,
        "selected": plan_from_diagnostic(diag),
    }
    done: dict[tuple, tuple[str, dict]] = {}
    for arm in p1_arms:
        bmap = build_map(x, plans[arm], radial_rule, arm)
        key = bmap.key()
        if key in done:
            src, rec = done[key]
            rows.append({**rec, "arm": arm, "same_map_as": src, "map": bmap.describe()})
        else:
            t0 = time.perf_counter()
            r = run_map(
                target.name,
                target.logdensity,
                bmap,
                p1["x_final"],
                seed,
                p1["w2"],
                p1["grads"],
                DRAWS,
                CHAINS,
                MAX_DOUBLINGS,
            )
            wall = time.perf_counter() - t0 + p1_wall
            rec = {
                **base,
                "arm": arm,
                **metrics(r["samples"], r["grad_evals"], r["divergences"], target, wall),
                "max_depth_hits": r["max_depth_hits"],
                "mean_leapfrog": r["mean_leapfrog"],
                "phase1_grad_evals": r["phase1_grad_evals"],
                "warmup_divergences": r["warmup_divergences"],
                "map": bmap.describe(),
                "same_map_as": None,
            }
            done[key] = (arm, rec)
            rows.append(rec)
        if arm == "selected":
            rows[-1]["diagnostic"] = diag
        rows[-1]["phase1_matches_production"] = phase1_matches
        print(
            json.dumps(
                {k: rows[-1][k] for k in ("target", "seed", "arm", "tail_ess_per_grad", "split_rhat", "same_map_as")}
            ),
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("targets")
    ap.add_argument("seeds")
    ap.add_argument("--radial-rule", default="0.5")
    ap.add_argument("--arms", default="stock,stock_dense,coord,whiten,radial,composite,selected")
    a = ap.parse_args()
    arms = a.arms.split(",")
    cfg = DiagnosticConfig()
    for name in a.targets.split(","):
        target = make_target(name)
        for seed in [int(s) for s in a.seeds.split(",")]:
            rows = run_one(target, seed, arms, a.radial_rule, cfg)
            with open(a.out, "a") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
