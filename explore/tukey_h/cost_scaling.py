"""NUTS cost scaling against position in the tail: stock (x-space) vs arcsinh vs Tukey-h.

Prediction (Betancourt 2017, §4.3: for pi ~ exp(-|q|^beta), T_opt ~ H^((2 - beta) / (2 beta))):
  - x-space, polynomial tail (log potential): time to U-turn grows like exp(E / 2) for a Cauchy (~ |x|);
  - arcsinh pullback, beta = 1: ~ sqrt(E);
  - Tukey-h pullback, beta = 2: O(1).
E is the x-space potential energy above the mode, E(x) = -log pi_x(x) + log pi_x(mode).

Part A (chain-binned): 1-D Cauchy and 10-d product Cauchy, 4 chains x 4000 draws after tune 500, seeds 0-2;
  each transition's leapfrog count is binned by the exact-distribution quantile of E at its starting state.
Part B (fixed start, 1-D Cauchy): with each arm's adapted step size, metric and transform (chain 0, seed 0),
  run 400 independent NUTS transitions from x0 in {0, 1, 3, ..., 1e6} with fresh momenta and record the
  leapfrog count and integration time (steps x step size).

Usage: uv run python explore/tukey_h/cost_scaling.py   (writes results/cost_scaling.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import blackjax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import make_target
from tukey_h import CalibrationRule, JaxTukeyHTransform, run_phase1, run_phase23, sample_stock_nuts_with_stats

from mcmc_robust.bench.sota_adapters import SOTATargetAdapter
from mcmc_robust.integrations import JaxArcsinhTransform

TUNE, DRAWS, CHAINS, DEPTH = 500, 4000, 4, 10
RULE = CalibrationRule(kind="tmle", nu_thr=5.0)
QBINS = [0.0, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0]
X0_GRID = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 1e3, 1e4, 1e5, 1e6]
N_FIXED = 400


def energy(x: np.ndarray) -> np.ndarray:
    """Product-Cauchy potential above the mode: sum_i log(1 + x_i^2)."""
    return np.sum(np.log1p(np.asarray(x) ** 2), axis=-1)


def mixing_stats(e_chain: np.ndarray, e_med: float, e_q99: float, sampling_grads: int) -> dict:
    """Energy mixing: ESS of E(x) (arviz bulk ESS over chains) and tail-excursion return times.

    A return time is the number of transitions from a draw that first exceeds the exact q99 level of E until
    the chain is back below the exact median level (excursions still open at the end are dropped)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = float(az.ess(az.from_dict({"posterior": {"E": e_chain}}), var_names=["E"], method="bulk")["E"].values)
    returns = []
    for ch in e_chain:
        t = 0
        n = ch.shape[0]
        while t < n:
            if ch[t] > e_q99:
                u = t
                while u < n and ch[u] > e_med:
                    u += 1
                if u < n:
                    returns.append(u - t)
                t = u
            t += 1
    return {
        "energy_ess": ess,
        "energy_ess_per_1k_draws": 1000.0 * ess / e_chain.size,
        "energy_ess_per_grad": ess / max(1, sampling_grads),
        "n_excursions": len(returns),
        "median_return_transitions": float(np.median(returns)) if returns else None,
        "mean_return_transitions": float(np.mean(returns)) if returns else None,
    }


def run_arms(key: str, seed: int) -> dict:
    target = make_target(key)
    logp = SOTATargetAdapter(target).jax_logdensity_fn()
    d = target.d
    out = {}
    stock = sample_stock_nuts_with_stats(
        logp, d, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=seed, max_num_doublings=DEPTH
    )
    out["stock"] = (stock, JaxTukeyHTransform.from_c(jnp.zeros(d), jnp.ones(d), jnp.full(d, jnp.inf)))
    p1 = run_phase1(logp, d, tune=TUNE, chains=CHAINS, seed=seed, max_num_doublings=DEPTH)
    asin_tr = JaxArcsinhTransform(radii=p1.arcsinh_transform.radius, location=p1.arcsinh_transform.location, d=d)
    asin = run_phase23(
        logp,
        asin_tr,
        p1.x_final,
        p1.w2_steps,
        p1.key_p2,
        p1.key_sample,
        draws=DRAWS,
        max_num_doublings=DEPTH,
        phase1_grad_evals=p1.grad_evals,
    )
    out["arcsinh"] = (asin, asin_tr)
    cal = RULE.compute(p1.x_tail.reshape(-1, d))
    h_tr = JaxTukeyHTransform.from_c(p1.arcsinh_transform.location, p1.arcsinh_transform.radius, cal["c"])
    hm = run_phase23(
        logp,
        h_tr,
        p1.x_final,
        p1.w2_steps,
        p1.key_p2,
        p1.key_sample,
        draws=DRAWS,
        max_num_doublings=DEPTH,
        phase1_grad_evals=p1.grad_evals,
    )
    hm.extra = {"c": np.asarray(cal["c"]).tolist()}
    out["hmap_tmle"] = (hm, h_tr)
    return {"logp": logp, "d": d, "arms": out}


def part_a(key: str, seeds: list[int], rng: np.random.Generator) -> tuple[dict, dict]:
    d = make_target(key).d
    ref = energy(rng.standard_cauchy(size=(2_000_000, d)))
    e_med, e_q99 = (float(v) for v in np.quantile(ref, [0.5, 0.99]))
    edges = np.quantile(ref, QBINS)
    edges[0], edges[-1] = -np.inf, np.inf
    res: dict = {"bin_quantiles": QBINS, "bin_E_edges": [float(e) for e in edges[1:-1]], "seeds": {}}
    fitted = {}
    for seed in seeds:
        runs = run_arms(key, seed)
        per_arm = {}
        for arm, (r, _tr) in runs["arms"].items():
            x = r.samples  # (chains, draws, d)
            # transition t starts at draw t-1; drop the first transition of each chain
            e_start = energy(x[:, :-1, :]).ravel()
            steps = r.n_steps[:, 1:].ravel()
            depth = r.depth[:, 1:].ravel()
            ss = np.repeat(r.step_size[:, None], x.shape[1] - 1, axis=1).ravel()
            idx = np.digitize(e_start, edges[1:-1])
            bins = []
            for b in range(len(QBINS) - 1):
                m = idx == b
                bins.append(
                    {
                        "n": int(m.sum()),
                        "median_E": float(np.median(e_start[m])) if m.any() else None,
                        "median_steps": float(np.median(steps[m])) if m.any() else None,
                        "mean_steps": float(np.mean(steps[m])) if m.any() else None,
                        "mean_depth": float(np.mean(depth[m])) if m.any() else None,
                        "frac_max_depth": float(np.mean(steps[m] >= 2**DEPTH - 1)) if m.any() else None,
                        "median_integration_time": float(np.median(steps[m] * ss[m])) if m.any() else None,
                    }
                )
            per_arm[arm] = {
                "bins": bins,
                "step_size": r.step_size.tolist(),
                "mean_steps_all": float(r.n_steps.mean()),
                "divergences": int(r.divergences),
                "frac_draws_beyond_q999": float(np.mean(idx == len(QBINS) - 2)),
                "mixing": mixing_stats(energy(x), e_med, e_q99, r.sampling_grad_evals),
            }
            if seed == seeds[0]:
                fitted[arm] = (r, _tr, runs["logp"])
        res["seeds"][str(seed)] = per_arm
        print(key, seed, {a: [b["mean_steps"] for b in v["bins"]] for a, v in per_arm.items()}, flush=True)
        print(key, seed, {a: v["mixing"] for a, v in per_arm.items()}, flush=True)
    return res, fitted


def part_b(fitted: dict, rng_seed: int = 123) -> dict:
    """Fixed-start transitions on the 1-D Cauchy with each arm's adapted kernel (chain 0)."""
    out = {"x0": X0_GRID, "E": [float(np.log1p(x0**2)) for x0 in X0_GRID], "arms": {}}
    for arm, (r, tr, logp) in fitted.items():
        ss = float(r.step_size[0])
        imm = jnp.asarray(r.inverse_mass_matrix[0])
        logp_z = tr.wrap_logdensity(logp)
        kernel = blackjax.nuts(logp_z, step_size=ss, inverse_mass_matrix=imm, max_num_doublings=DEPTH)

        @jax.jit
        def many(z0: jnp.ndarray, keys: jnp.ndarray, kernel: blackjax.base.SamplingAlgorithm = kernel) -> tuple:
            st = kernel.init(z0)

            def one(k: jnp.ndarray) -> tuple:
                _, info = kernel.step(k, st)
                return info.num_integration_steps, info.num_trajectory_expansions, info.is_divergent

            return jax.vmap(one)(keys)

        rows = []
        for i, x0 in enumerate(X0_GRID):
            z0 = tr.forward(jnp.array([x0]))
            keys = jax.random.split(jax.random.PRNGKey(rng_seed + i), N_FIXED)
            steps, depth, div = (np.asarray(a) for a in many(z0, keys))
            rows.append(
                {
                    "x0": x0,
                    "z0": float(z0[0]),
                    "median_steps": float(np.median(steps)),
                    "mean_steps": float(np.mean(steps)),
                    "q90_steps": float(np.quantile(steps, 0.9)),
                    "mean_depth": float(np.mean(depth)),
                    "frac_max_depth": float(np.mean(steps >= 2**DEPTH - 1)),
                    "median_integration_time": float(np.median(steps) * ss),
                    "frac_divergent": float(np.mean(div)),
                }
            )
        out["arms"][arm] = {"step_size": ss, "inverse_mass_matrix": float(imm[0]), "rows": rows}
        print(arm, [(row["x0"], row["median_steps"]) for row in rows], flush=True)
    return out


def scaling_fits(part_b_res: dict) -> dict:
    """Log-log slope of median steps on E over x0 in [10, 1e6] (unsaturated points only); 0 means O(1) and 0.5
    means sqrt(E). Also the slope of log(steps) on E, which is 0.5 for steps ~ |x0| = exp(E / 2), and the same
    two slopes for the mean step count from x0 = 3 on."""
    fits = {}
    e = np.asarray(part_b_res["E"])
    for arm, v in part_b_res["arms"].items():
        s = np.asarray([row["median_steps"] for row in v["rows"]])
        sat = np.asarray([row["frac_max_depth"] for row in v["rows"]]) > 0.5
        m = (np.asarray(part_b_res["x0"]) >= 10.0) & ~sat
        fit = {"n_points": int(m.sum()), "n_saturated": int(sat.sum())}
        if m.sum() >= 2:
            fit["loglog_slope_steps_vs_E"] = float(np.polyfit(np.log(e[m]), np.log(s[m]), 1)[0])
            fit["semilog_slope_logsteps_vs_E"] = float(np.polyfit(e[m], np.log(s[m]), 1)[0])
        # mean steps (less quantised than the median) from x0 = 3 on, unsaturated
        mean_s = np.asarray([row["mean_steps"] for row in v["rows"]])
        m3 = (np.asarray(part_b_res["x0"]) >= 3.0) & ~sat
        if m3.sum() >= 2:
            fit["loglog_slope_mean_steps_vs_E_from_x0_3"] = float(np.polyfit(np.log(e[m3]), np.log(mean_s[m3]), 1)[0])
            fit["semilog_slope_log_mean_steps_vs_E_from_x0_3"] = float(np.polyfit(e[m3], np.log(mean_s[m3]), 1)[0])
        fits[arm] = fit
    return fits


def main() -> None:
    rng = np.random.default_rng(2026)
    out: dict = {"settings": {"tune": TUNE, "draws": DRAWS, "chains": CHAINS, "max_num_doublings": DEPTH}}
    a1, fitted = part_a("cauchy1d", [0, 1, 2], rng)
    out["part_a_cauchy1d"] = a1
    out["part_b_cauchy1d"] = part_b(fitted)
    out["part_b_fits"] = scaling_fits(out["part_b_cauchy1d"])
    a10, _ = part_a("prod_t1", [0, 1, 2], rng)
    out["part_a_prod_cauchy_d10"] = a10
    path = Path(__file__).parent / "results" / "cost_scaling.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["part_b_fits"], indent=1))


if __name__ == "__main__":
    main()
