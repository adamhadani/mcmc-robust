"""Summarise results/main.jsonl into results/summary.json and results/tables.md (the FINDINGS tables).

Run: uv run python -m explore.tail_direction.summarise [IN.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .targets import ALL_TARGETS, make_target

RES = Path(__file__).parent / "results"
ARMS = ["stock", "coord", "whiten", "radial", "composite", "selected"]
ORACLE_LABEL = {
    "rot_t1_d10": "radial",
    "rot_t2_d10": "radial",
    "rot_t1_d20": "radial",
    "rot_t2_d20": "radial",
    "prod_t_d10": "coord",
    "mixed_d15": "composite",
    "rot_gauss_d20": "affine",
}
# arm whose result represents each diagnostic label (for "affine", the composite map is the whiten map)
LABEL_ARM = {"coord": "coord", "radial": "radial", "composite": "composite", "affine": "whiten"}


def med_rng(v: list[float], fmt: str = "{:.4f}") -> str:
    a = np.asarray(v, dtype=float)
    return f"{fmt.format(np.median(a))} [{fmt.format(np.min(a))}, {fmt.format(np.max(a))}]"


def ratio_stats(num: dict[int, float], den: dict[int, float]) -> dict:
    seeds = sorted(set(num) & set(den))
    r = [num[s] / den[s] for s in seeds]
    return {
        "per_seed": dict(zip(seeds, r)),
        "median": float(np.median(r)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "n_gt_1": int(np.sum(np.asarray(r) > 1.0)),
        "n": len(r),
    }


def roles(d: int, blocks: list[list[int]], singles: list[int]) -> list[str]:
    out = ["light"] * d
    for b in blocks:
        for i in b:
            out[i] = "block"
    for i in singles:
        out[i] = "single"
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "main.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    by = defaultdict(dict)  # (target, arm) -> seed -> row
    for r in rows:
        by[(r["target"], r["arm"])][r["seed"]] = r
    summary: dict = {"source": src.name, "targets": {}}
    md: list[str] = []

    md.append("### Table 1: per-arm results (5 seeds; median [min, max] over seeds unless stated)\n")
    md.append(
        "| Target | Arm | Median grads | Tail-ESS/grad | Worst-coord tail-ESS | Max R-hat | Divergences (sum) "
        "| Tail-prob error, mean over coords and p | Mean MCSE |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    for t in ALL_TARGETS:
        summary["targets"][t] = {"arms": {}}
        for arm in ARMS:
            seeds = by.get((t, arm), {})
            if not seeds:
                continue
            v = list(seeds.values())
            s = {
                "n_seeds": len(v),
                "grads_median": float(np.median([r["total_grad_evals"] for r in v])),
                "tail_ess_per_grad": [r["tail_ess_per_grad"] for r in v],
                "min_tail_ess": [r["min_tail_ess"] for r in v],
                "max_split_rhat": float(max(r["split_rhat"] for r in v)),
                "max_combined_rhat": float(max(r["combined_rhat"] for r in v)),
                "divergences_sum": int(sum(r["divergences"] for r in v)),
                "tp_err_mean": [r["tp_err_mean"] for r in v],
                "tp_mcse_mean": [r["tp_mcse_mean"] for r in v],
                "same_map_as": sorted({r.get("same_map_as") for r in v if r.get("same_map_as")}),
            }
            summary["targets"][t]["arms"][arm] = s
            same = f" (= {', '.join(s['same_map_as'])})" if s["same_map_as"] else ""
            md.append(
                f"| {t} | {arm}{same} | {s['grads_median']:,.0f} | {med_rng(s['tail_ess_per_grad'])} "
                f"| {med_rng(s['min_tail_ess'], '{:.0f}')} | {s['max_split_rhat']:.3f} | {s['divergences_sum']} "
                f"| {med_rng(s['tp_err_mean'], '{:.4f}')} | {np.nanmedian(s['tp_mcse_mean']):.4f} |"
            )

    md.append("\n### Table 2: paired per-seed tail-ESS/grad ratios, median [min, max] (seeds with ratio > 1)\n")
    md.append("| Target | Arm | vs stock | vs coord | Worst-coord tail-ESS/grad vs coord |")
    md.append("|---|---|---|---|---|")
    for t in ALL_TARGETS:
        teg = {arm: {s: r["tail_ess_per_grad"] for s, r in by.get((t, arm), {}).items()} for arm in ARMS}
        wst = {
            arm: {s: r["min_tail_ess"] / r["total_grad_evals"] for s, r in by.get((t, arm), {}).items()} for arm in ARMS
        }
        summary["targets"][t]["ratios"] = {}
        for arm in ARMS[1:]:
            if not teg[arm]:
                continue
            vs_stock = ratio_stats(teg[arm], teg["stock"])
            vs_coord = ratio_stats(teg[arm], teg["coord"]) if arm != "coord" else None
            w_coord = ratio_stats(wst[arm], wst["coord"]) if arm != "coord" else None
            summary["targets"][t]["ratios"][arm] = {
                "vs_stock": vs_stock,
                "vs_coord": vs_coord,
                "worst_vs_coord": w_coord,
            }

            def f(x: dict | None) -> str:
                if x is None:
                    return "—"
                return f"{x['median']:.2f}× [{x['min']:.2f}, {x['max']:.2f}] ({x['n_gt_1']}/{x['n']})"

            md.append(f"| {t} | {arm} | {f(vs_stock)} | {f(vs_coord)} | {f(w_coord)} |")

    # Diagnostic assessment
    md.append("\n### Table 3: Phase 1 diagnostic per seed\n")
    md.append(
        "| Target | Seed | Diagnostic label | Partition exact | Role accuracy | Best label (tail-ESS/grad) "
        "| Picked best? | Selected / best of {coord, radial, composite} |"
    )
    md.append("|---|---|---|---|---|---|---|---|")
    diag_rows = []
    for t in ALL_TARGETS:
        tgt = make_target(t)
        plan = tgt.oracle_plan
        o_blocks = [sorted(b) for b in plan.get("radial", [])]
        o_singles = sorted(plan.get("coord", []))
        for seed, r in sorted(by.get((t, "selected"), {}).items()):
            dg = r["diagnostic"]
            exact = sorted(map(sorted, dg["dep_blocks"])) == sorted(o_blocks) and sorted(dg["singles"]) == o_singles
            ro, re = roles(tgt.d, o_blocks, o_singles), roles(tgt.d, dg["dep_blocks"], dg["singles"])
            role_acc = float(np.mean([a == b for a, b in zip(ro, re)]))
            # candidate labels for this target: the three maps, with "affine" standing in for the composite
            # map when the oracle partition is all-light, and composite only where it differs from both
            cands = {"coord": "coord", "radial": "radial"}
            if ORACLE_LABEL[t] == "composite":
                cands["composite"] = "composite"
            if ORACLE_LABEL[t] == "affine":
                cands["affine"] = "whiten"
            vals = {lab: by[(t, arm)][seed]["tail_ess_per_grad"] for lab, arm in cands.items()}
            best = max(vals, key=vals.get)
            sel_val = r["tail_ess_per_grad"]
            regret = sel_val / vals[best]
            diag_rows.append(
                {
                    "target": t,
                    "seed": seed,
                    "label": dg["choice"],
                    "oracle_label": ORACLE_LABEL[t],
                    "best_label": best,
                    "picked_best": dg["choice"] == best,
                    "label_is_oracle": dg["choice"] == ORACLE_LABEL[t],
                    "partition_exact": bool(exact),
                    "role_accuracy": role_acc,
                    "selected_over_best": float(regret),
                    "candidates": vals,
                    "dep_blocks": dg["dep_blocks"],
                    "singles": dg["singles"],
                    "light": dg["light"],
                }
            )
            md.append(
                f"| {t} | {seed} | {dg['choice']} | {'yes' if exact else 'no'} | {role_acc:.2f} | {best} "
                f"| {'yes' if dg['choice'] == best else 'no'} | {regret:.2f} |"
            )
    n = len(diag_rows)
    agg = {
        "n": n,
        "picked_best": int(sum(d["picked_best"] for d in diag_rows)),
        "label_is_oracle": int(sum(d["label_is_oracle"] for d in diag_rows)),
        "partition_exact": int(sum(d["partition_exact"] for d in diag_rows)),
        "role_accuracy_mean": float(np.mean([d["role_accuracy"] for d in diag_rows])) if n else float("nan"),
        "selected_over_best_median": float(np.median([d["selected_over_best"] for d in diag_rows]))
        if n
        else float("nan"),
        "selected_over_best_min": float(np.min([d["selected_over_best"] for d in diag_rows])) if n else float("nan"),
        "selected_within_10pct_of_best": int(sum(d["selected_over_best"] >= 0.9 for d in diag_rows)),
        "oracle_label_is_best": int(sum(d["oracle_label"] == d["best_label"] for d in diag_rows)),
    }
    summary["diagnostic"] = {"per_run": diag_rows, "aggregate": agg}
    md.append(
        f"\nAggregate over {n} (target, seed) runs: label = empirical best in {agg['picked_best']}/{n}; "
        f"label = oracle label in {agg['label_is_oracle']}/{n}; exact partition in {agg['partition_exact']}/{n}; "
        f"mean role accuracy {agg['role_accuracy_mean']:.2f}; selected tail-ESS/grad ÷ best of the candidate maps: "
        f"median {agg['selected_over_best_median']:.2f}, min {agg['selected_over_best_min']:.2f}, "
        f"≥ 0.9 in {agg['selected_within_10pct_of_best']}/{n}; the oracle label was the empirical best in "
        f"{agg['oracle_label_is_best']}/{n}."
    )
    p1 = [r.get("phase1_matches_production") for r in rows if r.get("phase1_matches_production") is not None]
    summary["phase1_matches_production"] = {"true": int(sum(p1)), "n": len(p1)}
    md.append(
        f"\nPhase 1 draws reproduce the production calibration exactly in {sum(p1)}/{len(p1)} Phase-1 arm records."
    )
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
