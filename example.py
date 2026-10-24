#!/usr/bin/env python3
"""Stock BlackJAX NUTS vs. the same NUTS on the calibrated arcsinh pullback (about 10 seconds).

The target is the d=20 mixed-tail benchmark (REPORT.md, Table 3): ten independent Student-t
coordinates with nu from 1.2 to 2.5 and ten standard Gaussian coordinates. Both samplers use
BlackJAX window adaptation with the same number of warm-up steps, so the only difference is the
coordinate-wise arcsinh transform. Every sentence printed at the end is computed from the numbers
measured in this run; one run is weak evidence (REPORT.md, section 6).

Execution:
    uv run python example.py
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from mcmc_robust import sample_blackjax_unified_nuts
from mcmc_robust.bench.sota_adapters import evaluate_chains_with_arviz, run_blackjax_sampler
from mcmc_robust.targets import HeterogeneousHeavyTailedTarget

RHAT_CONVERGED = 1.01


def run_comparison(draws: int, tune: int, chains: int, seed: int) -> dict[str, dict[str, float]]:
    """Run both samplers on the mixed-tail target and return their diagnostics."""
    target = HeterogeneousHeavyTailedTarget(d=20, heavy_coords=list(range(10)), nu_list=list(np.linspace(1.2, 2.5, 10)))

    t0 = time.perf_counter()
    samples, info = run_blackjax_sampler(
        target, sampler_type="nuts", n_samples=draws, n_warmup=tune, n_chains=chains, seed=seed
    )
    stock = evaluate_chains_with_arviz(
        samples,
        info["total_grad_evals"],
        target=target,
        divergences=info["divergences"],
        wall_clock_time=time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    result = sample_blackjax_unified_nuts(target, draws=draws, tune=tune, chains=chains, seed=seed)
    robust = evaluate_chains_with_arviz(
        result.samples,
        result.grad_evals,
        target=target,
        divergences=result.divergences,
        wall_clock_time=time.perf_counter() - t0,
    )
    return {"stock": stock, "arcsinh": robust}


def describe_comparison(stock: Mapping[str, float], robust: Mapping[str, float]) -> list[str]:
    """Sentences comparing the transformed run with stock NUTS, derived only from the metrics."""
    ratio = robust["tail_ess_per_grad"] / stock["tail_ess_per_grad"]
    direction = f"{ratio:.1f}× higher" if ratio > 1.0 else f"{ratio:.1f}× (lower)"
    lines = [
        (
            f"Tail-ESS per gradient: {robust['tail_ess_per_grad']:.3g} with the transform vs "
            f"{stock['tail_ess_per_grad']:.3g} for stock NUTS, {direction}."
        ),
        (
            f"Gradient evaluations (warm-up included): {int(robust['total_grad_evals']):,} vs "
            f"{int(stock['total_grad_evals']):,}."
        ),
        f"Max R-hat: {robust['split_rhat']:.3f} vs {stock['split_rhat']:.3f}.",
        f"Divergences after warm-up: {int(robust['divergences'])} vs {int(stock['divergences'])}.",
        (
            f"0.99-quantile relative error, mean over coordinates: {robust['q99_rel_error']:.1%} vs "
            f"{stock['q99_rel_error']:.1%}."
        ),
    ]
    for name, metrics in (("transformed", robust), ("stock", stock)):
        if not metrics["split_rhat"] <= RHAT_CONVERGED:
            lines.append(
                f"The {name} run has not converged (R-hat above {RHAT_CONVERGED}); treat its numbers with care."
            )
    return lines


def main(draws: int = 1000, tune: int = 500, chains: int = 4, seed: int = 0) -> None:
    print("Stock BlackJAX NUTS vs. NUTS on the calibrated arcsinh pullback")
    print(
        f"Target: 10 Student-t (nu 1.2-2.5) + 10 Gaussian coordinates; {chains} chains x {draws} draws, {tune} warm-up"
    )
    results = run_comparison(draws=draws, tune=tune, chains=chains, seed=seed)
    print(f"\n{'Sampler':<24} {'Gradients':>10} {'Tail-ESS':>9} {'Tail-ESS/grad':>14} {'Max R-hat':>9} {'Divs':>5}")
    for label, key in (("BlackJAX NUTS (stock)", "stock"), ("BlackJAX NUTS + arcsinh", "arcsinh")):
        m = results[key]
        print(
            f"{label:<24} {int(m['total_grad_evals']):>10,} {m['tail_ess']:>9.0f} {m['tail_ess_per_grad']:>14.4g} "
            f"{m['split_rhat']:>9.3f} {int(m['divergences']):>5}"
        )
    print()
    for line in describe_comparison(results["stock"], results["arcsinh"]):
        print("- " + line)
    print(f"- Single run with seed {seed}; see REPORT.md section 6 before generalising.")


if __name__ == "__main__":
    main()
