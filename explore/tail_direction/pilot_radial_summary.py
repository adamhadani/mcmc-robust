"""Summarise the radial-radius pilot (seeds 100-102): results/pilot_radial_R_*.jsonl -> results/pilot_radial_R_summary.json.

Run: uv run python -m explore.tail_direction.pilot_radial_summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).parent / "results"
RULES = ["0.35", "0.5", "0.75", "1.5", "3.0", "sqrtk"]


def main() -> None:
    d: dict = defaultdict(dict)
    for rule in RULES:
        f = RES / f"pilot_radial_R_{rule}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("same_map_as"):
                continue
            d[f"{r['target']}|{r['arm']}"].setdefault(rule, []).append(
                {"seed": r["seed"], "tail_ess_per_grad": r["tail_ess_per_grad"], "split_rhat": r["split_rhat"]}
            )
    out = {}
    for k, v in sorted(d.items()):
        out[k] = {
            rule: {
                "median_tail_ess_per_grad": float(np.median([x["tail_ess_per_grad"] for x in xs])),
                "max_split_rhat": float(max(x["split_rhat"] for x in xs)),
                "runs": xs,
            }
            for rule, xs in v.items()
        }
        print(
            f"{k:28s}",
            "  ".join(
                f"R={rule}: {out[k][rule]['median_tail_ess_per_grad']:.4f} (R-hat {out[k][rule]['max_split_rhat']:.3f})"
                for rule in RULES
                if rule in out[k]
            ),
        )
    (RES / "pilot_radial_R_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
