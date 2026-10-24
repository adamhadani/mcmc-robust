"""Pilot (seeds disjoint from the main run): Phase 1 statistics used to set the diagnostic thresholds.

Dumps, per target and pilot seed, the Hill tail index of every coordinate, the extremal co-occurrence
matrix chi (u = 0.9) split into oracle within-block / between-block / light pairs, and the Hill index of
the standardised radius. Run: uv run python -m explore.tail_direction.pilot_phase1 OUT.json SEEDS
"""

from __future__ import annotations

import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from .maps import DiagnosticConfig, chi_matrix, hill_alpha, robust_scale, tail_diagnostic
from .pipeline import run_phase1
from .targets import ALL_TARGETS, make_target


def oracle_groups(t) -> tuple[list[list[int]], list[int], list[int]]:
    plan = t.oracle_plan
    return plan.get("radial", []), plan.get("coord", []), plan.get("affine", [])


def main() -> None:
    out_path, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    npz_path = sys.argv[3] if len(sys.argv) > 3 else None
    rows = []
    draws: dict[str, np.ndarray] = {}
    for name in ALL_TARGETS:
        t = make_target(name)
        blocks, singles, light = oracle_groups(t)
        heavy = sorted([i for b in blocks for i in b] + singles)
        for seed in seeds:
            t0 = time.perf_counter()
            p1 = run_phase1(t.logdensity, t.d, seed)
            x = p1["x_tail"]
            draws[f"{name}|{seed}"] = x
            diag = tail_diagnostic(x, DiagnosticConfig())
            xc = x - np.median(x, axis=0)
            alphas = [hill_alpha(xc[:, i]) for i in range(t.d)]
            chi = chi_matrix(xc, 0.9)
            within, between = [], []
            for a in heavy:
                for b in heavy:
                    if a < b:
                        same = any(a in blk and b in blk for blk in blocks)
                        (within if same else between).append(float(chi[a, b]))
            rad = np.linalg.norm(xc / robust_scale(xc), axis=1)
            rows.append(
                {
                    "target": name,
                    "seed": seed,
                    "n": int(x.shape[0]),
                    "phase1_grads": p1["grads"],
                    "wall": time.perf_counter() - t0,
                    "alpha_heavy": [alphas[i] for i in heavy],
                    "alpha_light": [alphas[i] for i in light],
                    "chi_within": within,
                    "chi_between_heavy": between,
                    "alpha_radius": hill_alpha(rad),
                    "diagnostic_default_config": diag,
                }
            )
            r = rows[-1]
            print(
                name,
                seed,
                f"aH={np.round(r['alpha_heavy'], 1).tolist()} aL={np.round(r['alpha_light'], 1).tolist()}",
                f"chiW={np.mean(within) if within else float('nan'):.2f} chiB={np.mean(between) if between else float('nan'):.2f}",
                f"maxB={max(between) if between else float('nan'):.2f} minW={min(within) if within else float('nan'):.2f}",
                f"aRad={r['alpha_radius']:.2f}",
                f"-> {diag['choice']} blocks={diag['dep_blocks']} singles={diag['singles']} light={diag['light']}",
                flush=True,
            )
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent=1)
    if npz_path:
        np.savez(npz_path, **draws)


if __name__ == "__main__":
    main()
