"""Main 5-seed grid: stock NUTS vs arcsinh (repo) vs Tukey-h arms, all BlackJAX, tune 500 / draws 1000 / 4 chains,
max tree depth 10 (BlackJAX's max_num_doublings=10).

Arms:
  stock            run_blackjax_sampler (repo), NUTS in x-space with window adaptation.
  arcsinh          sample_blackjax_unified_nuts (repo).
  hmap_tmle        Tukey-h, c = alpha_hat (t-MLE nu), heavy iff nu_hat <= 5, c = inf otherwise  [best rule].
  hmap_qr          Tukey-h, heavy iff q99/q75 > 3, c = Hill alpha.
  hmap_tmle_c0.25  hmap_tmle with c x 1/4 on heavy coordinates.
  hmap_tmle_c4     hmap_tmle with c x 4 on heavy coordinates.
  hmap_forced      Tukey-h on every coordinate, c = Hill alpha (what a pipeline without a light/heavy test does).
All Tukey-h arms reuse the arcsinh arm's Phase 1 (identical draws: same seed and key splits), and pay for it in
their gradient count.

Usage: uv run python explore/tukey_h/run_grid.py OUT.json TARGET[,TARGET...] SEEDS [ARMS]
Resumable: rows already in OUT.json are skipped.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import make_target, summarise
from tukey_h import CalibrationRule, run_phase1, sample_tukey_h_nuts

from mcmc_robust.bench.sota_adapters import SOTATargetAdapter, run_blackjax_sampler
from mcmc_robust.jax_unified_nuts import sample_blackjax_unified_nuts

TUNE, DRAWS, CHAINS, DEPTH = 500, 1000, 4, 10

HMAP_ARMS = {
    "hmap_tmle": CalibrationRule(kind="tmle", nu_thr=5.0),
    "hmap_qr": CalibrationRule(kind="qr", qr_thr=3.0),
    "hmap_tmle_c0.25": CalibrationRule(kind="tmle", nu_thr=5.0, c_mult=0.25),
    "hmap_tmle_c4": CalibrationRule(kind="tmle", nu_thr=5.0, c_mult=4.0),
    "hmap_forced": CalibrationRule(kind="hill", force_all=True),
}
ALL_ARMS = ["stock", "arcsinh", *HMAP_ARMS]


def provenance() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    import arviz
    import blackjax

    return {
        "git_sha": sha,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "blackjax": blackjax.__version__,
        "arviz": arviz.__version__,
        "tune": TUNE,
        "draws": DRAWS,
        "chains": CHAINS,
        "max_num_doublings": DEPTH,
    }


def nsteps_stats(n_steps: np.ndarray) -> dict:
    n = np.asarray(n_steps)
    return {
        "mean_steps_per_draw": float(n.mean()),
        "frac_max_depth": float(np.mean(n >= 2**DEPTH - 1)),
    }


def main() -> None:
    out_path = Path(sys.argv[1])
    targets = sys.argv[2].split(",")
    seeds = [int(s) for s in sys.argv[3].split(",")]
    arms = sys.argv[4].split(",") if len(sys.argv) > 4 else ALL_ARMS
    data = json.loads(out_path.read_text()) if out_path.exists() else {"provenance": provenance(), "rows": []}
    done = {(r["target"], r["seed"], r["arm"]) for r in data["rows"]}

    def save() -> None:
        out_path.write_text(json.dumps(data, indent=1))

    for key in targets:
        target = make_target(key)
        logp = SOTATargetAdapter(target).jax_logdensity_fn()
        d = target.d
        for seed in seeds:
            if "stock" in arms and (key, seed, "stock") not in done:
                t0 = time.perf_counter()
                s, info = run_blackjax_sampler(
                    target,
                    sampler_type="nuts",
                    n_samples=DRAWS,
                    n_warmup=TUNE,
                    n_chains=CHAINS,
                    seed=seed,
                    max_num_doublings=DEPTH,
                )
                wall = time.perf_counter() - t0
                row = {"target": key, "seed": seed, "arm": "stock"}
                row.update(summarise(s, info["total_grad_evals"], info["divergences"], key, target, wall))
                row["frac_max_depth"] = None  # run_blackjax_sampler does not expose per-draw steps
                row["mean_steps_per_draw"] = info["draw_grad_evals"] / (CHAINS * DRAWS)
                data["rows"].append(row)
                save()
                print(
                    json.dumps({k: row[k] for k in ("target", "seed", "arm", "tail_ess_per_grad", "split_rhat")}),
                    flush=True,
                )

            need_h = [a for a in arms if a in HMAP_ARMS and (key, seed, a) not in done]
            p1 = None
            if need_h or ("arcsinh" in arms and (key, seed, "arcsinh") not in done):
                p1 = run_phase1(logp, d, tune=TUNE, chains=CHAINS, seed=seed, max_num_doublings=DEPTH)
            if "arcsinh" in arms and (key, seed, "arcsinh") not in done:
                t0 = time.perf_counter()
                res = sample_blackjax_unified_nuts(
                    target, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=seed, max_num_doublings=DEPTH
                )
                wall = time.perf_counter() - t0
                row = {"target": key, "seed": seed, "arm": "arcsinh"}
                row.update(summarise(res.samples, res.grad_evals, res.divergences, key, target, wall))
                n_steps = np.asarray(res.idata["sample_stats"]["n_steps"].values)
                row.update(nsteps_stats(n_steps))
                # Paired-design check: our Phase 1 reproduces the repo's calibrated (mu, R) exactly.
                row["phase1_mu_R_maxdiff_vs_repo"] = float(
                    max(
                        np.max(np.abs(np.asarray(res.transform.location) - np.asarray(p1.arcsinh_transform.location))),
                        np.max(np.abs(np.asarray(res.transform.radius) - np.asarray(p1.arcsinh_transform.radius))),
                    )
                )
                row["phase1_grads_repo"] = res.phase1_grad_evals
                row["phase1_grads_ours"] = p1.grad_evals
                data["rows"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "target",
                                "seed",
                                "arm",
                                "tail_ess_per_grad",
                                "split_rhat",
                                "phase1_mu_R_maxdiff_vs_repo",
                            )
                        }
                    ),
                    flush=True,
                )

            for arm in need_h:
                t0 = time.perf_counter()
                res = sample_tukey_h_nuts(
                    logp,
                    d,
                    HMAP_ARMS[arm],
                    draws=DRAWS,
                    tune=TUNE,
                    chains=CHAINS,
                    seed=seed,
                    phase1=p1,
                    max_num_doublings=DEPTH,
                )
                wall = time.perf_counter() - t0
                row = {"target": key, "seed": seed, "arm": arm}
                row.update(summarise(res.samples, res.grad_evals, res.divergences, key, target, wall))
                row.update(nsteps_stats(res.n_steps))
                row["c"] = [None if np.isinf(c) else float(c) for c in res.extra["c"]]
                row["n_heavy"] = int(np.sum(res.extra["heavy"]))
                row["phase1_grads"] = res.phase1_grad_evals
                data["rows"].append(row)
                save()
                print(
                    json.dumps(
                        {k: row[k] for k in ("target", "seed", "arm", "tail_ess_per_grad", "split_rhat", "n_heavy")}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
