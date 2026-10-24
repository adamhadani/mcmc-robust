"""Choose the diagnostic thresholds on pilot seeds 100-109 (disjoint from the main seeds 0-4).

Scores every configuration in a small grid by how well the Phase 1 partition matches the oracle partition:
- exact: the partition (tail-dependent blocks, independent heavy coordinates, light coordinates) is exact;
- role: fraction of coordinates whose role (block / single / light) matches the oracle.

Run: uv run python -m explore.tail_direction.pilot_tune PILOT_DRAWS.npz OUT.json
(PILOT_DRAWS.npz is written by ``pilot_phase1 ... 100,...,109 PILOT_DRAWS.npz``.)
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from .maps import DiagnosticConfig, tail_diagnostic
from .targets import make_target


def roles(d: int, blocks: list[list[int]], singles: list[int]) -> list[str]:
    out = ["light"] * d
    for b in blocks:
        for i in b:
            out[i] = "block"
    for i in singles:
        out[i] = "single"
    return out


def score(diag: dict, plan: dict, d: int) -> tuple[bool, float]:
    o_blocks = [sorted(b) for b in plan.get("radial", [])]
    o_singles = sorted(plan.get("coord", []))
    exact = sorted(map(sorted, diag["dep_blocks"])) == sorted(o_blocks) and sorted(diag["singles"]) == o_singles
    ro = roles(d, o_blocks, o_singles)
    re = roles(d, diag["dep_blocks"], diag["singles"])
    return exact, float(np.mean([a == b for a, b in zip(ro, re)]))


def main() -> None:
    draws = np.load(sys.argv[1])
    out_path = sys.argv[2]
    keys = sorted(draws.files)
    targets = {k.split("|")[0]: make_target(k.split("|")[0]) for k in keys}
    grid = list(
        itertools.product(
            [0.8, 0.85, 0.9], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [3.0, 3.5, 4.0], [5.0, 6.0, 8.0], [None, 6.0]
        )
    )
    rows = []
    for u, delta, a1, ab, gate in grid:
        cfg = DiagnosticConfig(chi_u=u, tau=(1 - u) + delta, alpha_single=a1, alpha_block=ab, alpha_global=gate)
        per_target: dict[str, list[tuple[bool, float]]] = {}
        for k in keys:
            name = k.split("|")[0]
            t = targets[name]
            per_target.setdefault(name, []).append(score(tail_diagnostic(draws[k], cfg), t.oracle_plan, t.d))
        summary = {
            n: {"exact": float(np.mean([e for e, _ in v])), "role": float(np.mean([r for _, r in v]))}
            for n, v in per_target.items()
        }
        rows.append(
            {
                "chi_u": u,
                "tau": (1 - u) + delta,
                "alpha_single": a1,
                "alpha_block": ab,
                "alpha_global": gate,
                "mean_exact": float(np.mean([s["exact"] for s in summary.values()])),
                "mean_role": float(np.mean([s["role"] for s in summary.values()])),
                "worst_target_role": float(min(s["role"] for s in summary.values())),
                "per_target": summary,
            }
        )
    rows.sort(key=lambda r: (-r["mean_role"], -r["mean_exact"]))
    with open(out_path, "w") as fh:
        json.dump({"n_pilot_runs": len(keys), "grid": rows}, fh, indent=1)
    for r in rows[:12]:
        print(
            f"u={r['chi_u']} gate={r['alpha_global']} tau={r['tau']:.2f} a1={r['alpha_single']} ab={r['alpha_block']} "
            f"role={r['mean_role']:.3f} exact={r['mean_exact']:.2f} worst={r['worst_target_role']:.2f} "
            + " ".join(f"{n[:9]}:{s['role']:.2f}/{s['exact']:.1f}" for n, s in r["per_target"].items())
        )


if __name__ == "__main__":
    main()
