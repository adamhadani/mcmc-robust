"""Compare calibration rules for c on the real Phase-1 draws (no Phase 2 needed).

For every target and seed 0-4, runs the repo pipeline's Phase 1 (tune=500 -> W1 = 75 steps, 4 chains,
second half pooled = 152 draws), then applies each rule and scores it against the known tail class
and tail index of every coordinate.

Usage: uv run python explore/tukey_h/calib_compare.py  (writes results/calib_results.json)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import TARGETS, make_target, true_alpha, true_heavy_mask
from tukey_h import CalibrationRule, run_phase1

from mcmc_robust.bench.sota_adapters import SOTATargetAdapter

RULES = {
    "hill": CalibrationRule(kind="hill", alpha_thr=3.5),
    "hill_qr": CalibrationRule(kind="hill_qr", alpha_thr=3.5, qr_thr=3.0),
    "qr": CalibrationRule(kind="qr", qr_thr=3.0),
    "tmle": CalibrationRule(kind="tmle", nu_thr=5.0),
}


def main() -> None:
    seeds = range(5)
    rows = []
    for key in TARGETS:
        target = make_target(key)
        logp = SOTATargetAdapter(target).jax_logdensity_fn()
        heavy_true = true_heavy_mask(key, target)
        alpha_true = true_alpha(key, target)
        for seed in seeds:
            t0 = time.perf_counter()
            p1 = run_phase1(logp, target.d, tune=500, chains=4, seed=seed)
            pilot = p1.x_tail.reshape(-1, target.d)
            row = {
                "target": key,
                "seed": seed,
                "n_pilot": int(pilot.shape[0]),
                "phase1_grads": p1.grad_evals,
                "heavy_true": heavy_true.tolist(),
                "alpha_true": [None if np.isinf(a) else float(a) for a in alpha_true],
                "rules": {},
            }
            for name, rule in RULES.items():
                cal = rule.compute(pilot)
                heavy = np.asarray(cal["heavy"], dtype=bool)
                est = np.asarray(cal["alpha_est"], dtype=np.float64)
                both = heavy & heavy_true
                log_ratio = np.abs(np.log(est[both] / alpha_true[both])) if both.any() else np.array([])
                row["rules"][name] = {
                    "heavy": heavy.tolist(),
                    "alpha_est": est.tolist(),
                    "false_heavy": int(np.sum(heavy & ~heavy_true)),
                    "missed_heavy": int(np.sum(~heavy & heavy_true)),
                    "n_light": int(np.sum(~heavy_true)),
                    "n_heavy": int(np.sum(heavy_true)),
                    "median_abs_log_alpha_ratio": float(np.median(log_ratio)) if log_ratio.size else None,
                }
            row["wall"] = time.perf_counter() - t0
            rows.append(row)
            print(key, seed, {n: (r["false_heavy"], r["missed_heavy"]) for n, r in row["rules"].items()}, flush=True)

    # aggregate
    agg = {}
    for name in RULES:
        per_t = {}
        for key in TARGETS:
            rs = [r["rules"][name] for r in rows if r["target"] == key]
            n_light = sum(r["n_light"] for r in rs)
            n_heavy = sum(r["n_heavy"] for r in rs)
            ratios = [r["median_abs_log_alpha_ratio"] for r in rs if r["median_abs_log_alpha_ratio"] is not None]
            per_t[key] = {
                "false_heavy_rate": (sum(r["false_heavy"] for r in rs) / n_light) if n_light else None,
                "missed_heavy_rate": (sum(r["missed_heavy"] for r in rs) / n_heavy) if n_heavy else None,
                "median_abs_log_alpha_ratio": float(np.median(ratios)) if ratios else None,
            }
        tot_light = sum(r["rules"][name]["n_light"] for r in rows)
        tot_heavy = sum(r["rules"][name]["n_heavy"] for r in rows)
        agg[name] = {
            "per_target": per_t,
            "false_heavy_rate_all": sum(r["rules"][name]["false_heavy"] for r in rows) / tot_light,
            "missed_heavy_rate_all": sum(r["rules"][name]["missed_heavy"] for r in rows) / tot_heavy,
        }
    out = {
        "rules": {n: {k: v for k, v in vars(r).items()} for n, r in RULES.items()},
        "rows": rows,
        "aggregate": agg,
    }
    path = Path(__file__).parent / "results" / "calib_results.json"
    path.write_text(json.dumps(out, indent=1, default=lambda o: o if not isinstance(o, tuple) else list(o)))
    for name, a in agg.items():
        print(name, "false_heavy", round(a["false_heavy_rate_all"], 3), "missed", round(a["missed_heavy_rate_all"], 3))
        for key, v in a["per_target"].items():
            print("   ", key, v)


if __name__ == "__main__":
    main()
