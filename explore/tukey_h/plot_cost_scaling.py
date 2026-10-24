"""Figure for the cost-scaling study, drawn from results/cost_scaling.json -> results/cost_scaling.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent / "results"
# First three slots of the dataviz reference categorical palette (validated all-pairs, light mode).
COLORS = {"stock": "#2a78d6", "arcsinh": "#eb6834", "hmap_tmle": "#1baf7a"}
LABELS = {"stock": "stock (x-space)", "arcsinh": "arcsinh", "hmap_tmle": "Tukey-h (t-MLE c)"}
MARKERS = {"stock": "o", "arcsinh": "s", "hmap_tmle": "D"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    data = json.loads((HERE / "cost_scaling.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    # (a) fixed start, 1-D Cauchy
    pb = data["part_b_cauchy1d"]
    ax = axes[0]
    x0 = np.asarray(pb["x0"])
    xplot = np.where(x0 == 0, 0.3, x0)
    for arm, v in pb["arms"].items():
        s = [r["mean_steps"] for r in v["rows"]]
        ax.plot(xplot, s, color=COLORS[arm], lw=2, marker=MARKERS[arm], ms=8, mec="#fcfcfb", mew=1.5, label=LABELS[arm])
        ax.annotate(
            LABELS[arm],
            (xplot[-1], s[-1]),
            xytext=(6, -12 if arm == "stock" else 0),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            va="center",
        )
    ax.axhline(1023, color=MUTED, lw=1, ls="--")
    ax.text(0.35, 1023 * 1.15, "max tree depth 10 (1023 steps)", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("start position |x0| (x0 = 0 plotted at 0.3)", color=TEXT)
    ax.set_ylabel("mean leapfrog steps per transition", color=TEXT)
    ax.set_title("1-D Cauchy, one NUTS transition from x0 (400 momenta)", color=TEXT, fontsize=10, loc="left")
    ax.set_xlim(0.2, 2e8)
    ax.legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.42))

    # (b) chain-binned, 10-d product Cauchy: per-bin mean steps, median over seeds
    pa = data["part_a_prod_cauchy_d10"]
    ax = axes[1]
    q = pa["bin_quantiles"]
    names = [f"{100 * q[i]:g}–{100 * q[i + 1]:g}%" for i in range(len(q) - 1)]
    seeds = list(pa["seeds"])
    for arm in pa["seeds"][seeds[0]]:
        per_seed = [
            [b["mean_steps"] if b["mean_steps"] is not None else np.nan for b in pa["seeds"][sd][arm]["bins"]]
            for sd in seeds
        ]
        s = np.nanmedian(np.asarray(per_seed), axis=0)
        ax.plot(
            range(len(s)),
            s,
            color=COLORS[arm],
            lw=2,
            marker=MARKERS[arm],
            ms=8,
            mec="#fcfcfb",
            mew=1.5,
            label=LABELS[arm],
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("quantile bin of the start state's energy E(x) under the exact target", color=TEXT)
    ax.set_ylabel("mean leapfrog steps per transition", color=TEXT)
    ax.set_title("10-d product Cauchy, post-warm-up chains (median of 3 seeds)", color=TEXT, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "cost_scaling.png", dpi=130)
    print("wrote", HERE / "cost_scaling.png")


if __name__ == "__main__":
    main()
