"""Phase 1 + window adaptation + sampling for block maps, mirroring ``sample_blackjax_unified_nuts``.

- Same key split (key_init, key_p1, key_p2, key_sample), same 0.5 N(0, I) initial positions, same Phase 1
  (BlackJAX window adaptation for W1 = max(10, 0.15 tune) steps on the initial arcsinh map R = 3, mu = 0),
  same calibration draws (second half of Phase 1, pooled), same W2 = tune - W1 window adaptation per chain
  in the new z-space, same per-chain step size / diagonal metric for sampling.
- Gradient accounting as in production: per phase, one initial evaluation per chain plus the sum of
  ``num_integration_steps``.
- ``run_stock``: identity map, no Phase 1, ``tune`` window-adaptation steps from the same initial positions.

Phase 2 + sampling is jit-compiled once per (target, map structure) with the map parameters as traced
arguments, so seeds and calibrations reuse the compiled kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import blackjax
import jax.numpy as jnp
import numpy as np

from mcmc_robust.integrations import JaxArcsinhTransform

from .maps import BlockMap, make_fns, pullback


def keys_for(seed: int) -> tuple[Any, Any, Any, Any]:
    return tuple(jax.random.split(jax.random.PRNGKey(seed), 4))  # type: ignore[return-value]


def init_positions(seed: int, d: int, chains: int) -> jnp.ndarray:
    key_init = keys_for(seed)[0]
    return 0.5 * jax.random.normal(key_init, shape=(chains, d), dtype=jnp.float64)


def run_phase1(
    logdensity_x: Callable, d: int, seed: int, tune: int = 500, chains: int = 4, max_doublings: int = 10
) -> dict:
    """Production Phase 1, verbatim in structure (eager vmap of window_adaptation.run)."""
    _, key_p1, _, _ = keys_for(seed)
    init_x = init_positions(seed, d, chains)
    t0 = JaxArcsinhTransform(radii=3.0, location=0.0, d=d)
    w1 = max(10, int(0.15 * tune))
    z0 = jax.vmap(t0.forward)(init_x)
    warm = blackjax.window_adaptation(
        blackjax.nuts, t0.wrap_logdensity(logdensity_x), target_acceptance_rate=0.8, max_num_doublings=max_doublings
    )
    (state, _), info = jax.vmap(lambda k, z: warm.run(k, z, num_steps=w1))(jax.random.split(key_p1, chains), z0)
    grads = int(chains + int(jnp.sum(info.info.num_integration_steps)))
    half = max(1, w1 // 2)
    x_tail = jax.vmap(jax.vmap(t0.inverse))(info.state.position[:, half:, :])
    x_final = jax.vmap(t0.inverse)(state.position)
    return {
        "x_tail": np.asarray(x_tail).reshape(-1, d),
        "x_final": np.asarray(x_final),
        "grads": grads,
        "w1": w1,
        "w2": max(10, tune - w1),
    }


_COMPILED: dict[tuple, Callable] = {}


def _compiled_runner(
    target_name: str,
    logdensity_x: Callable,
    structure: tuple,
    w2: int,
    draws: int,
    chains: int,
    max_doublings: int,
    dense: bool = False,
) -> Callable:
    cache_key = (target_name, structure, w2, draws, chains, max_doublings, dense)
    if cache_key in _COMPILED:
        return _COMPILED[cache_key]

    def run(params: tuple, x_start: jnp.ndarray, key_p2: Any, key_sample: Any) -> dict:
        inverse, forward, _ = make_fns(structure, params)
        logdens_z = pullback(logdensity_x, structure, params)
        z_start = jax.vmap(forward)(x_start)
        warm = blackjax.window_adaptation(
            blackjax.nuts,
            logdens_z,
            is_mass_matrix_diagonal=not dense,
            target_acceptance_rate=0.8,
            max_num_doublings=max_doublings,
        )
        (state, ad), info = jax.vmap(lambda k, z: warm.run(k, z, num_steps=w2))(
            jax.random.split(key_p2, chains), z_start
        )
        warm_steps = jnp.sum(info.info.num_integration_steps)
        warm_div = jnp.sum(info.info.is_divergent)

        def one_chain(st: Any, ss: jnp.ndarray, imm: jnp.ndarray, key: Any) -> tuple:
            kern = blackjax.nuts(logdens_z, step_size=ss, inverse_mass_matrix=imm, max_num_doublings=max_doublings)

            def step(s: Any, k: Any) -> tuple:
                ns, inf = kern.step(k, s)
                return ns, (ns.position, inf.is_divergent, inf.num_integration_steps, inf.acceptance_rate)

            _, tr = jax.lax.scan(step, st, jax.random.split(key, draws))
            return tr

        z_draws, div, nsteps, acc = jax.vmap(one_chain)(
            state, ad["step_size"], ad["inverse_mass_matrix"], jax.random.split(key_sample, chains)
        )
        x_draws = jax.vmap(jax.vmap(inverse))(z_draws)
        return {
            "x": x_draws,
            "z": z_draws,
            "div": div,
            "nsteps": nsteps,
            "acc": acc,
            "warm_steps": warm_steps,
            "warm_div": warm_div,
            "step_size": ad["step_size"],
            "inv_mass": ad["inverse_mass_matrix"],
        }

    fn = jax.jit(run)
    _COMPILED[cache_key] = fn
    return fn


def run_map(
    target_name: str,
    logdensity_x: Callable,
    bmap: BlockMap,
    x_start: np.ndarray,
    seed: int,
    w2: int,
    pre_grads: int,
    draws: int = 1000,
    chains: int = 4,
    max_doublings: int = 10,
    dense: bool = False,
) -> dict:
    """Phase 2/3 window adaptation + sampling in the z-space of ``bmap``; ``pre_grads`` = Phase 1 cost."""
    _, _, key_p2, key_sample = keys_for(seed)
    fn = _compiled_runner(target_name, logdensity_x, bmap.structure(), w2, draws, chains, max_doublings, dense)
    out = fn(bmap.param_tree(), jnp.asarray(x_start), key_p2, key_sample)
    out = jax.tree_util.tree_map(np.asarray, out)
    phase2 = int(chains + int(out["warm_steps"]))
    sampling = int(np.sum(out["nsteps"]))
    nsteps = out["nsteps"]
    return {
        "samples": out["x"],
        "z_samples": out["z"],
        "grad_evals": int(pre_grads + phase2 + sampling),
        "phase1_grad_evals": int(pre_grads),
        "phase2_grad_evals": phase2,
        "sampling_grad_evals": sampling,
        "divergences": int(np.sum(out["div"])),
        "warmup_divergences": int(out["warm_div"]),
        "acceptance_rate": float(np.mean(out["acc"])),
        "max_depth_hits": int(np.sum(nsteps >= 2**max_doublings - 1)),
        "mean_leapfrog": float(np.mean(nsteps)),
        "step_size": [float(v) for v in out["step_size"]],
    }


def run_stock(
    target_name: str,
    logdensity_x: Callable,
    d: int,
    seed: int,
    tune: int = 500,
    draws: int = 1000,
    chains: int = 4,
    max_doublings: int = 10,
    dense: bool = False,
) -> dict:
    """Stock BlackJAX NUTS: identity map, ``tune`` window-adaptation steps, no Phase 1 (dense: full metric)."""
    from .maps import Block

    ident = BlockMap(d=d, blocks=[Block("identity", tuple(range(d)), {})], label="stock")
    x0 = np.asarray(init_positions(seed, d, chains))
    return run_map(
        target_name,
        logdensity_x,
        ident,
        x0,
        seed,
        w2=tune,
        pre_grads=0,
        draws=draws,
        chains=chains,
        max_doublings=max_doublings,
        dense=dense,
    )
