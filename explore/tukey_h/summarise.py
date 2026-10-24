"""Turn the committed grid results into the tables quoted in FINDINGS.md.

Reads results/grid_A.json and grid_B.json (the pre-registered main grid, seeds 0-4), results/grid_lwmle.json
(the Tukey-h MLE arms, added after the main grid, same seeds) and results/kappa_sweep.json (held-out seeds
10-14), and writes results/summary.json and results/tables.md.
Every number in FINDINGS.md's tables comes from tables.md (or summary.json, calib_results.json,
cost_scaling.json, verify_results.json).

Usage: uv run python explore/tukey_h/summarise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import TARGETS

HERE = Path(__file__).parent / "results"
ARMS = [
    "stock",
    "arcsinh",
    "hmap_tmle",
    "hmap_qr",
    "hmap_tmle_c0.25",
    "hmap_tmle_c4",
    "hmap_forced",
    "hmap_lwmle",
    "hmap_lwmle_all",
]


def load_rows() -> list[dict]:
    rows = []
    for name in ("grid_A.json", "grid_B.json", "grid_lwmle.json"):
        if (HERE / name).exists():
            rows += json.loads((HERE / name).read_text())["rows"]
    return rows


def med_rng(v: list[float]) -> dict:
    a = np.asarray(v, dtype=float)
    return {"median": float(np.median(a)), "min": float(np.min(a)), "max": float(np.max(a)), "n": int(a.size)}


def fmt_ratio(s: dict) -> str:
    return f"{s['median']:.2f}× [{s['min']:.2f}, {s['max']:.2f}]"


def main() -> None:
    rows = load_rows()
    by = {(r["target"], r["arm"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    summary: dict = {"seeds": seeds, "targets": {}}
    for t in TARGETS:
        st: dict = {}
        for arm in ARMS:
            rs = [by[(t, arm, s)] for s in seeds if (t, arm, s) in by]
            if not rs:
                continue
            e = {
                "n_seeds": len(rs),
                "grads": med_rng([r["total_grad_evals"] for r in rs]),
                "tail_ess_per_grad": med_rng([r["tail_ess_per_grad"] for r in rs]),
                "min_tail_ess": med_rng([r["min_tail_ess"] for r in rs]),
                "min_tail_ess_per_grad": med_rng([r["min_tail_ess_per_grad"] for r in rs]),
                "max_rhat": float(max(r["split_rhat"] for r in rs)),
                "max_combined_rhat": float(max(r["combined_rhat"] for r in rs)),
                "divergences": med_rng([r["divergences"] for r in rs]),
                "tp_err_rel": med_rng([r["tp_err_rel"] for r in rs]),
                "tp_mcse_rel": med_rng([r["tp_mcse_rel"] for r in rs]),
                "tp_err_over_mcse": med_rng([r["tp_err_rel"] / r["tp_mcse_rel"] for r in rs]),
                "mean_steps_per_draw": med_rng([r["mean_steps_per_draw"] for r in rs]),
            }
            if rs[0].get("frac_max_depth") is not None:
                e["frac_max_depth"] = med_rng([r["frac_max_depth"] for r in rs])
            if "n_heavy" in rs[0]:
                e["n_heavy"] = med_rng([r["n_heavy"] for r in rs])
            for ref in ("stock", "arcsinh"):
                if arm == ref:
                    continue
                pair = [(by[(t, arm, s)], by[(t, ref, s)]) for s in seeds if (t, arm, s) in by and (t, ref, s) in by]
                if pair:
                    e[f"ratio_vs_{ref}"] = med_rng([a["tail_ess_per_grad"] / b["tail_ess_per_grad"] for a, b in pair])
                    e[f"worst_ratio_vs_{ref}"] = med_rng(
                        [a["min_tail_ess_per_grad"] / b["min_tail_ess_per_grad"] for a, b in pair]
                    )
            st[arm] = e
        summary["targets"][t] = st
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = [
        "# Generated tables (explore/tukey_h/summarise.py; do not edit by hand)",
        "",
        (
            f"Seeds: {seeds}. Each cell is a median over seeds; brackets are [min, max] over seeds. "
            "Ratios are paired per seed on tail-ESS/grad (coordinate-mean tail-ESS divided by all gradients, "
            "warm-up included). 'Worst' ratios use worst-coordinate tail-ESS per gradient. "
            "Tail-prob error is |P_hat(x_i < q_p) - p| / 0.01 averaged over coordinates and p in {0.01, 0.99}; "
            "MCSE is the same quantity's Monte Carlo standard error (indicator ESS)."
        ),
        "",
    ]
    lines += [
        "## Table 1: per-arm metrics",
        "",
        (
            "| Target | Arm | Grads | Tail-ESS/grad [min, max] | Worst-coord tail-ESS | Max R-hat | Divergences | "
            "Tail-prob err (MCSE) | Max-depth frac | Heavy coords |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in TARGETS:
        for arm in ARMS:
            e = summary["targets"][t].get(arm)
            if e is None:
                continue
            tes = e["tail_ess_per_grad"]
            md = f"{e['frac_max_depth']['median']:.3f}" if "frac_max_depth" in e else "n/a"
            nh = f"{e['n_heavy']['median']:.0f}" if "n_heavy" in e else ""
            lines.append(
                f"| {t} | {arm} | {e['grads']['median']:,.0f} | {tes['median']:.4f} [{tes['min']:.4f}, {tes['max']:.4f}] | "
                f"{e['min_tail_ess']['median']:,.0f} | {e['max_rhat']:.3f} | {e['divergences']['median']:.0f} "
                f"[{e['divergences']['min']:.0f}, {e['divergences']['max']:.0f}] | "
                f"{100 * e['tp_err_rel']['median']:.1f}% ({100 * e['tp_mcse_rel']['median']:.1f}%) | {md} | {nh} |"
            )
    lines += [
        "",
        "## Table 2: paired tail-ESS/grad ratios, median [min, max] over seeds",
        "",
        (
            "| Target | arcsinh / stock | hmap_tmle / stock | hmap_lwmle / stock | hmap_tmle / arcsinh | "
            "hmap_qr / arcsinh | hmap_lwmle / arcsinh | hmap_lwmle_all / arcsinh |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in TARGETS:
        s = summary["targets"][t]

        def g(arm: str, ref: str, s: dict = s) -> str:
            return fmt_ratio(s[arm][f"ratio_vs_{ref}"]) if arm in s and f"ratio_vs_{ref}" in s[arm] else "n/a"

        lines.append(
            f"| {t} | {g('arcsinh', 'stock')} | {g('hmap_tmle', 'stock')} | {g('hmap_lwmle', 'stock')} | "
            f"{g('hmap_tmle', 'arcsinh')} | {g('hmap_qr', 'arcsinh')} | {g('hmap_lwmle', 'arcsinh')} | "
            f"{g('hmap_lwmle_all', 'arcsinh')} |"
        )
    lines += [
        "",
        "## Table 3: paired worst-coordinate tail-ESS/grad ratios, median [min, max]",
        "",
        "| Target | arcsinh / stock | hmap_tmle / stock | hmap_tmle / arcsinh | forced / hmap_tmle |",
        "|---|---|---|---|---|",
    ]
    by_seed = by
    for t in TARGETS:
        s = summary["targets"][t]
        pair = [
            by_seed[(t, "hmap_forced", sd)]["min_tail_ess_per_grad"]
            / by_seed[(t, "hmap_tmle", sd)]["min_tail_ess_per_grad"]
            for sd in seeds
            if (t, "hmap_forced", sd) in by_seed and (t, "hmap_tmle", sd) in by_seed
        ]
        forced_vs_tmle = fmt_ratio(med_rng(pair)) if pair else "n/a"
        s["hmap_forced"]["worst_ratio_vs_hmap_tmle"] = med_rng(pair) if pair else None
        lines.append(
            f"| {t} | {fmt_ratio(s['arcsinh']['worst_ratio_vs_stock'])} | {fmt_ratio(s['hmap_tmle']['worst_ratio_vs_stock'])} | "
            f"{fmt_ratio(s['hmap_tmle']['worst_ratio_vs_arcsinh'])} | {forced_vs_tmle} |"
        )

    def paired(t: str, arm: str, ref: str, key: str = "tail_ess_per_grad") -> dict | None:
        v = [by[(t, arm, sd)][key] / by[(t, ref, sd)][key] for sd in seeds if (t, arm, sd) in by and (t, ref, sd) in by]
        return med_rng(v) if v else None

    # Table 4: misspecification of c relative to the calibrated rule
    lines += [
        "",
        "## Table 4: misspecified c, paired against hmap_tmle (same Phase 1, same c up to the factor)",
        "",
        (
            "| Target | c×1/4 / tmle | c×4 / tmle | c×1/4 / arcsinh | c×4 / arcsinh | Max R-hat (c×1/4, tmle, c×4) | "
            "Max divergences (c×1/4, tmle, c×4) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    mis = {}
    for t in TARGETS:
        s = summary["targets"][t]
        lo, hi = paired(t, "hmap_tmle_c0.25", "hmap_tmle"), paired(t, "hmap_tmle_c4", "hmap_tmle")
        mis[t] = {"c0.25_vs_tmle": lo, "c4_vs_tmle": hi}
        rh = ", ".join(f"{s[a]['max_rhat']:.3f}" for a in ("hmap_tmle_c0.25", "hmap_tmle", "hmap_tmle_c4"))
        dv = ", ".join(f"{s[a]['divergences']['max']:.0f}" for a in ("hmap_tmle_c0.25", "hmap_tmle", "hmap_tmle_c4"))
        lo_a = s["hmap_tmle_c0.25"]["ratio_vs_arcsinh"]
        hi_a = s["hmap_tmle_c4"]["ratio_vs_arcsinh"]
        lines.append(
            f"| {t} | {fmt_ratio(lo)} | {fmt_ratio(hi)} | {fmt_ratio(lo_a)} | {fmt_ratio(hi_a)} | {rh} | {dv} |"
        )
    summary["misspecification"] = mis

    # Table 5: light-tail cost of forcing the map onto every coordinate
    lines += [
        "",
        "## Table 5: light-tail cost, hmap_forced (Hill c on every coordinate) vs hmap_tmle and vs stock",
        "",
        (
            "| Target | Light coords | Median forced c on light coords [min, max] | forced / tmle (mean tail-ESS/grad) | "
            "forced / tmle (worst coord) | forced / stock | Max divergences (forced, tmle) | lwmle_all / stock |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    light_cost = {}
    for t in ("iso_gauss", "corr_gauss", "task3", "task1", "prod_t5"):
        rows_f = [by[(t, "hmap_forced", sd)] for sd in seeds if (t, "hmap_forced", sd) in by]
        rows_t = [by[(t, "hmap_tmle", sd)] for sd in seeds if (t, "hmap_tmle", sd) in by]
        # light = coordinates the t-MLE rule left at c = inf
        light_c = [
            cf for rf, rt in zip(rows_f, rows_t) for cf, ct in zip(rf["c"], rt["c"]) if ct is None and cf is not None
        ]
        n_light = int(np.median([sum(c is None for c in rt["c"]) for rt in rows_t]))
        a = paired(t, "hmap_forced", "hmap_tmle")
        w = paired(t, "hmap_forced", "hmap_tmle", "min_tail_ess_per_grad")
        st = paired(t, "hmap_forced", "stock")
        cstat = med_rng(light_c) if light_c else None
        light_cost[t] = {
            "n_light": n_light,
            "forced_c_light": cstat,
            "forced_vs_tmle": a,
            "forced_vs_tmle_worst": w,
            "forced_vs_stock": st,
        }
        cs = f"{cstat['median']:.2f} [{cstat['min']:.2f}, {cstat['max']:.2f}]" if cstat else "n/a"
        dv = f"{summary['targets'][t]['hmap_forced']['divergences']['max']:.0f}, {summary['targets'][t]['hmap_tmle']['divergences']['max']:.0f}"
        la = paired(t, "hmap_lwmle_all", "stock")
        light_cost[t]["lwmle_all_vs_stock"] = la
        lines.append(
            f"| {t} | {n_light} | {cs} | {fmt_ratio(a)} | {fmt_ratio(w)} | {fmt_ratio(st)} | {dv} | "
            f"{fmt_ratio(la) if la else 'n/a'} |"
        )
    summary["light_tail_cost"] = light_cost

    # Table 6: tail-probability calibration (error / MCSE ~ |N(0,1)| has median 0.67 if unbiased)
    lines += [
        "",
        "## Table 6: tail-probability error divided by its MCSE, median [min, max] over seeds",
        "",
        (
            "(For an unbiased estimator with a correct MCSE this ratio is around 0.8 on average; values well above 1-2 "
            "indicate bias or an underestimated MCSE.)"
        ),
        "",
        "| Target | stock | arcsinh | hmap_tmle | c×1/4 | c×4 | forced |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in TARGETS:
        s = summary["targets"][t]
        cells = [
            f"{s[a]['tp_err_over_mcse']['median']:.2f} [{s[a]['tp_err_over_mcse']['min']:.2f}, {s[a]['tp_err_over_mcse']['max']:.2f}]"
            for a in ("stock", "arcsinh", "hmap_tmle", "hmap_tmle_c0.25", "hmap_tmle_c4", "hmap_forced")
        ]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    # Table 7: held-out kappa sweep
    ks_path = HERE / "kappa_sweep.json"
    if ks_path.exists():
        krows = json.loads(ks_path.read_text())["rows"]
        kby = {(r["target"], r["arm"], r["seed"]): r for r in krows}
        kseeds = sorted({r["seed"] for r in krows})
        karms = [("hmap_tmle_c0.125", "1/8"), ("hmap_tmle_c0.25", "1/4"), ("hmap_tmle_c0.5", "1/2"), ("hmap_tmle", "1")]
        ktargets = [t for t in TARGETS if any(r["target"] == t for r in krows)]
        lines += [
            "",
            f"## Table 7: held-out kappa sweep (seeds {kseeds}), c = kappa × nu_hat on heavy coordinates",
            "",
            (
                "Paired tail-ESS/grad ratio against the arcsinh arm, median [min, max]; last column is the max R-hat "
                "over all kappa arms and seeds."
            ),
            "",
            "| Target | κ=1/8 | κ=1/4 | κ=1/2 | κ=1 | Max R-hat |",
            "|---|---|---|---|---|---|",
        ]
        ksum = {}
        for t in ktargets:
            cells, kt = [], {}
            for arm, lab in karms:
                v = [
                    kby[(t, arm, sd)]["tail_ess_per_grad"] / kby[(t, "arcsinh", sd)]["tail_ess_per_grad"]
                    for sd in kseeds
                    if (t, arm, sd) in kby and (t, "arcsinh", sd) in kby
                ]
                kt[lab] = med_rng(v) if v else None
                cells.append(fmt_ratio(kt[lab]) if v else "n/a")
            mr = max(kby[(t, a, sd)]["split_rhat"] for a, _ in karms for sd in kseeds if (t, a, sd) in kby)
            kt["max_rhat"] = float(mr)
            ksum[t] = kt
            lines.append(f"| {t} | " + " | ".join(cells) + f" | {mr:.3f} |")
        summary["kappa_sweep"] = {"seeds": kseeds, "targets": ksum}

        lines += [
            "",
            f"## Table 8: held-out seeds {kseeds}, calibration rules against the arcsinh arm",
            "",
            (
                "Paired tail-ESS/grad ratio and worst-coordinate tail-ESS/grad ratio against arcsinh, median "
                "[min, max]; max R-hat and max divergences per run over seeds."
            ),
            "",
            (
                "| Target | hmap_tmle / arcsinh | hmap_lwmle / arcsinh | hmap_lwmle_all / arcsinh | "
                "lwmle worst-coord / arcsinh | lwmle_all worst-coord / arcsinh | "
                "Max R-hat (arcsinh, tmle, lwmle, lwmle_all) | Max divergences (same order) |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]
        held = {}
        for t in TARGETS:

            def kr(arm: str, key: str = "tail_ess_per_grad", t: str = t) -> dict | None:
                v = [
                    kby[(t, arm, sd)][key] / kby[(t, "arcsinh", sd)][key]
                    for sd in kseeds
                    if (t, arm, sd) in kby and (t, "arcsinh", sd) in kby
                ]
                return med_rng(v) if v else None

            arms4 = ("arcsinh", "hmap_tmle", "hmap_lwmle", "hmap_lwmle_all")
            if not all((t, a, sd) in kby for a in arms4 for sd in kseeds):
                continue
            ht = {
                "tmle": kr("hmap_tmle"),
                "lwmle": kr("hmap_lwmle"),
                "lwmle_all": kr("hmap_lwmle_all"),
                "lwmle_worst": kr("hmap_lwmle", "min_tail_ess_per_grad"),
                "lwmle_all_worst": kr("hmap_lwmle_all", "min_tail_ess_per_grad"),
            }
            rh = ", ".join(f"{max(kby[(t, a, sd)]['split_rhat'] for sd in kseeds):.3f}" for a in arms4)
            dv = ", ".join(f"{max(kby[(t, a, sd)]['divergences'] for sd in kseeds):.0f}" for a in arms4)
            ht["max_rhat"] = {a: float(max(kby[(t, a, sd)]["split_rhat"] for sd in kseeds)) for a in arms4}
            ht["max_div"] = {a: float(max(kby[(t, a, sd)]["divergences"] for sd in kseeds)) for a in arms4}
            held[t] = ht
            lines.append(
                f"| {t} | {fmt_ratio(ht['tmle'])} | {fmt_ratio(ht['lwmle'])} | {fmt_ratio(ht['lwmle_all'])} | "
                f"{fmt_ratio(ht['lwmle_worst'])} | {fmt_ratio(ht['lwmle_all_worst'])} | {rh} | {dv} |"
            )
        summary["heldout_rules"] = {"seeds": kseeds, "targets": held}

    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    (HERE / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
