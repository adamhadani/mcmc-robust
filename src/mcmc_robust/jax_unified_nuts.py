r"""Pure-JAX 3-Phase Adaptive UnifiedRobustNUTS Kernel (`sample_blackjax_unified_nuts`).

Implements a fully XLA-compiled (`jax.jit` + `jax.vmap` + `jax.lax.scan`) 3-phase adaptive
MCMC pipeline combining the shift-equivariant coordinate-wise $\operatorname{arcsinh}$
diffeomorphism (`JaxArcsinhTransform`) with BlackJAX's No-U-Turn Sampler (`blackjax.nuts`):

1. **Phase 1 — Coordinate Location & Tail-Scale Calibration ($W_1 = \max(20, \lfloor 0.20 \cdot \text{tune} \rfloor)$)**:
   Runs initial exploratory warmup in warped coordinates, pulls Phase 1 draws back to
   primal $x$-space, and calibrates coordinate medians $\mu_i$ and clipped normal-consistent
   scales $R_i = \min(R_{\max}, \max(R_{\min}, c \cdot \hat{\sigma}_i))$ (formally verified in
   `formal/McmcRobustFormal.lean` §7).
2. **Phase 2 & 3 — Joint Metric & Step-Size Window Adaptation ($W_2 = \max(20, \text{tune} - W_1)$)**:
   Maps Phase 1 terminal states into the newly calibrated $z$-space (`z_start = transform.forward(x_p1_final)`),
   then runs BlackJAX `window_adaptation` on the calibrated transformed log-density to jointly adapt
   the diagonal inverse mass matrix $M^{-1}$ in $z$-space and the symplectic step size $h$.
3. **Post-Warmup Production Sampling (`draws` steps across `chains`)**:
   Executes `jax.jit` + `jax.vmap` + `jax.lax.scan` in calibrated $z$-space, pulls samples back to
   primal $x$-space via `transform.inverse`, and records exact leapfrog gradient evaluations across
   all phases (`grad_evals == phase1_grad_evals + phase2_grad_evals + sampling_grad_evals`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import blackjax
import jax.numpy as jnp
import numpy as np
from xarray import DataTree

from mcmc_robust.core.target import TargetDistribution

if TYPE_CHECKING:
    from mcmc_robust.integrations import JaxArcsinhTransform


def arviz_from_samples(
    samples: np.ndarray,
    sample_stats: dict[str, np.ndarray] | None = None,
) -> DataTree:
    r"""Constructs an ArviZ `DataTree` from a 3D MCMC samples array of shape `(chains, draws, d)`.

    Parameters:
        samples: Primal MCMC draws array of shape `(chains, draws, d)` (or 1D/2D array).
        sample_stats: Optional dictionary of per-step sample statistics arrays of shape `(chains, draws)`.

    Returns:
        ArviZ `DataTree` containing `/posterior` (`"x"`) and optional `/sample_stats`.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :, np.newaxis]
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 1D, 2D, or 3D samples array, got shape {arr.shape}")

    data_dict: dict[str, Any] = {"posterior": {"x": arr}}
    if sample_stats is not None:
        data_dict["sample_stats"] = sample_stats
    return az.from_dict(data_dict)


@dataclass
class JaxUnifiedNUTSResult:
    r"""Container for pure-JAX 3-phase adaptive `UnifiedRobustNUTS` sampling results.

    Attributes:
        samples: Post-warmup draws in primal $x$-space, shape `(chains, draws, d)`.
        z_samples: Post-warmup draws in warped $z$-space, shape `(chains, draws, d)`.
        transform: Calibrated `JaxArcsinhTransform` instance (`radii`, `location`, `mask`).
        step_size: Final adapted symplectic leapfrog step size $h > 0$.
        inverse_mass_matrix: Adapted diagonal inverse mass matrix $M^{-1}$ in $z$-space, shape `(d,)`.
        grad_evals: Exact total leapfrog gradient evaluations (`phase1 + phase2 + sampling`).
        divergences: Total post-warmup divergent transitions across all chains.
        acceptance_rate: Mean post-warmup Metropolis/multinomial acceptance probability.
        phase1_grad_evals: Exact gradient evaluations during Phase 1 scale/location calibration.
        phase2_grad_evals: Exact gradient evaluations during Phase 2/3 metric & step-size warmup.
        sampling_grad_evals: Exact gradient evaluations during post-warmup production sampling.
        idata: ArviZ `DataTree` object constructed via `arviz_from_samples(samples)`.
    """

    samples: np.ndarray
    z_samples: np.ndarray
    transform: JaxArcsinhTransform
    step_size: float
    inverse_mass_matrix: np.ndarray
    grad_evals: int
    divergences: int
    acceptance_rate: float
    phase1_grad_evals: int
    phase2_grad_evals: int
    sampling_grad_evals: int
    idata: DataTree


def calibrate_jax_arcsinh_transform(
    pilot_samples: Any,
    d: int | None = None,
    mask: Any | None = None,
    min_radius: float = 0.1,
    max_radius: float = 100.0,
    scale_multiplier: float = 1.5,
) -> JaxArcsinhTransform:
    r"""Calibrates shift-equivariant coordinate medians $\mu_i$ and clipped scales $R_i$ from pilot draws.

    Implements the formally verified Phase 1 calibration map (`formal/McmcRobustFormal.lean` §7):
        $$\mu_i = \operatorname{median}(x_{\cdot, i}), \qquad
          R_i = \min\!\left(R_{\max},\, \max\!\left(R_{\min},\, c \cdot \hat{\sigma}_i\right)\right)$$
    where $\hat{\sigma}_i = \operatorname{IQR}(x_{\cdot, i} - \mu_i) / 1.3489795$ with normal-consistent
    MAD and 90th-percentile fallbacks for near-degenerate coordinates.

    Parameters:
        pilot_samples: Exploratory primal draws of shape `(n_samples, d)` or `(chains, draws, d)`.
        d: Target dimension $d \ge 1$. If `None`, inferred from `pilot_samples.shape[-1]`.
        mask: Optional boolean mask of shape `(d,)` selecting coordinates to warp (default: all `True`).
        min_radius: Strictly positive lower bound $R_{\min} > 0$ on calibrated radii.
        max_radius: Upper bound $R_{\max} \ge R_{\min}$ on calibrated radii.
        scale_multiplier: Scale multiplier $c \ge 0$ applied to normal-consistent dispersion $\hat{\sigma}_i$.

    Returns:
        Calibrated `JaxArcsinhTransform` with `.location`, `.radii`, and `.mask`.
    """
    from mcmc_robust.integrations import JaxArcsinhTransform

    if min_radius <= 0.0:
        raise ValueError(f"min_radius must be strictly positive, got {min_radius}")
    if min_radius > max_radius:
        raise ValueError(f"min_radius ({min_radius}) must be <= max_radius ({max_radius})")
    if scale_multiplier < 0.0:
        raise ValueError(f"scale_multiplier must be non-negative, got {scale_multiplier}")

    arr = jnp.asarray(pilot_samples, dtype=jnp.float64)
    if arr.ndim == 1:
        if d is not None and d > 1:
            arr = arr.reshape(-1, d)
        else:
            arr = arr[:, jnp.newaxis]
    elif arr.ndim >= 3:
        arr = arr.reshape(-1, arr.shape[-1])

    d_actual = int(arr.shape[-1])
    if d is not None and int(d) != d_actual:
        raise ValueError(f"pilot_samples last dimension {d_actual} does not match d={d}")

    med = jnp.median(arr, axis=0)
    centered = arr - med
    abs_centered = jnp.abs(centered)

    q75 = jnp.percentile(centered, 75.0, axis=0)
    q25 = jnp.percentile(centered, 25.0, axis=0)
    iqr_scale = (q75 - q25) / 1.3489795
    mad_scale = jnp.median(abs_centered, axis=0) / 0.67448975
    q90_scale = jnp.percentile(abs_centered, 90.0, axis=0) / 1.6448536

    raw_sigma = jnp.where(iqr_scale > 1e-6, iqr_scale, jnp.maximum(mad_scale, q90_scale))
    raw_sigma = jnp.where(jnp.isfinite(raw_sigma), raw_sigma, 1.0)
    med_clean = jnp.where(jnp.isfinite(med), med, 0.0)

    radii = jnp.minimum(
        jnp.float64(max_radius),
        jnp.maximum(jnp.float64(min_radius), jnp.float64(scale_multiplier) * raw_sigma),
    )

    return JaxArcsinhTransform(
        radii=radii,
        location=med_clean,
        mask=mask,
        d=d_actual,
    )


def _make_smooth_wall_logdensity(
    base_logdensity_z: Callable[[jnp.ndarray], jnp.ndarray],
    l_floor: float,
    wall_scale: float = 10.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    r"""Constructs an exact $C^1$-smooth scalar log-density that is identical to `base_logdensity_z`
    throughout the posterior support ($\ell(z) \ge L_{\text{floor}}$) and smoothly attenuates
    exponential wall stiffness in the zero-measure exterior ($\ell(z) < L_{\text{floor}}$).

    Because `soft_logp` is a pure scalar function differentiated directly by JAX autodiff,
    the leapfrog force is the exact conservative gradient of the potential energy everywhere,
    preserving symplectic energy conservation across funnel neck reflections.
    """
    l_floor_f = jnp.float64(l_floor)
    scale_f = jnp.float64(wall_scale)

    def soft_logp(z: jnp.ndarray) -> jnp.ndarray:
        val = base_logdensity_z(z)
        val_clean = jnp.where(jnp.isfinite(val), val, l_floor_f - 1e8)
        diff = jnp.clip((l_floor_f - val_clean) / scale_f, 0.0, 1e100)
        return jnp.where(
            val_clean < l_floor_f,
            l_floor_f - scale_f * jnp.arcsinh(diff),
            val_clean,
        )

    return soft_logp


def sample_blackjax_unified_nuts(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray] | TargetDistribution,
    d: int | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    seed: int = 42,
    x0: np.ndarray | jnp.ndarray | Sequence[float] | None = None,
    *,
    random_seed: int | None = None,
    rng_key: Any | None = None,
    init_position: np.ndarray | jnp.ndarray | Sequence[float] | None = None,
    target_accept: float = 0.80,
    adapt_tail_scale: bool = True,
    initial_radius: float | Sequence[float] | np.ndarray | jnp.ndarray = 3.0,
    initial_location: float | Sequence[float] | np.ndarray | jnp.ndarray = 0.0,
    min_radius: float = 0.1,
    max_radius: float = 100.0,
    scale_multiplier: float = 1.5,
    mask: Sequence[bool] | np.ndarray | jnp.ndarray | None = None,
    max_num_doublings: int = 10,
    divergence_threshold: float = 1000.0,
    dim: int | None = None,
    step_size: float = 0.1,
) -> JaxUnifiedNUTSResult:
    r"""Executes pure-JAX 3-phase adaptive `UnifiedRobustNUTS` using `JaxArcsinhTransform` and BlackJAX.

    Three-Phase Schedule:
        - **Phase 1** ($W_1 = \max(20, \lfloor 0.20 \cdot \text{tune} \rfloor)$ when `adapt_tail_scale=True`
          and `tune >= 40`): Runs BlackJAX `window_adaptation` on initial `JaxArcsinhTransform`, pulls
          Phase 1 draws back to primal $x$-space, and calibrates `transform = calibrate_jax_arcsinh_transform(...)`.
        - **Phase 2 & 3** ($W_2 = \max(20, \text{tune} - W_1)$): Maps Phase 1 terminal primal positions
          into the calibrated $z$-space (`z_start = transform.forward(x_p1_final)`), then runs BlackJAX
          `window_adaptation` on `transform.wrap_logdensity(logdensity_fn)` for $W_2$ steps to adapt
          the diagonal inverse mass matrix $M^{-1}$ in calibrated $z$-space and symplectic step size $h$.
        - **Post-Warmup Sampling** (`draws` steps across `chains` via `jax.jit` + `jax.vmap` + `jax.lax.scan`):
          Samples in calibrated $z$-space, transforms back to primal $x$-space via `transform.inverse`,
          and records exact leapfrog gradient evaluations across all phases.

    Parameters:
        logdensity_fn: JAX-traceable callable `logdensity_fn(x) -> scalar` in primal $x$-space,
            or a `TargetDistribution` instance.
        d: Target dimension $d \ge 1$. Optional if inferred from `logdensity_fn`, `x0`, `init_position`,
            `dim`, or `initial_radius`.
        draws: Number of post-warmup MCMC samples per chain.
        tune: Total number of warmup adaptation steps per chain.
        chains: Number of independent Markov chains.
        seed: Integer PRNG seed (default: `42`).
        x0: Optional initial primal positions of shape `(d,)` or `(chains, d)`.
        random_seed: Optional alias for `seed`.
        rng_key: Optional explicit JAX PRNGKey (overrides `seed` if provided).
        init_position: Optional alias for `x0`.
        target_accept: Target acceptance rate $\delta \in (0, 1)$ for dual averaging (default: `0.80`).
        adapt_tail_scale: Whether to run Phase 1 coordinate location & scale calibration (default: `True`).
        initial_radius: Initial coordinate scale $R_0 > 0$ before Phase 1 calibration (default: `3.0`).
        initial_location: Initial coordinate location $\mu_0$ before Phase 1 calibration (default: `0.0`).
        min_radius: Minimum allowable calibrated coordinate scale $R_{\min} > 0$ (default: `0.1`).
        max_radius: Maximum allowable calibrated coordinate scale $R_{\max} \ge R_{\min}$ (default: `100.0`).
        scale_multiplier: Multiplier $c > 0$ applied to normal-consistent IQR scale in Phase 1 (default: `1.5`).
        mask: Optional boolean mask of shape `(d,)` indicating warped coordinates.
        max_num_doublings: Maximum NUTS binary tree depth (default: `10`).
        divergence_threshold: Maximum Hamiltonian energy error $\Delta H$ before divergence (default: `1000.0`).
        dim: Optional alias for `d`.
        step_size: Fallback step size when `tune == 0` (default: `0.1`).

    Returns:
        `JaxUnifiedNUTSResult` containing primal `samples`, `z_samples`, calibrated `transform`,
        adapted `step_size` and `inverse_mass_matrix`, exact per-phase and total `grad_evals`,
        `divergences`, `acceptance_rate`, and ArviZ `idata`.
    """
    from mcmc_robust.integrations import JaxArcsinhTransform

    if draws < 1:
        raise ValueError(f"draws must be >= 1, got {draws}")
    if tune < 0:
        raise ValueError(f"tune must be >= 0, got {tune}")
    if chains < 1:
        raise ValueError(f"chains must be >= 1, got {chains}")

    # Resolve TargetDistribution vs callable logdensity_fn
    if isinstance(logdensity_fn, TargetDistribution):
        from mcmc_robust.bench.sota_adapters import SOTATargetAdapter

        if d is None:
            d = logdensity_fn.d
        raw_logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray] = SOTATargetAdapter(logdensity_fn).jax_logdensity_fn()
    else:
        raw_logdensity_fn = logdensity_fn

    # Resolve dimensionality d
    eff_init = x0 if x0 is not None else init_position
    if d is None and dim is not None:
        d = int(dim)
    if d is None and eff_init is not None:
        d = int(np.asarray(eff_init).shape[-1])
    if d is None and hasattr(initial_radius, "__len__"):
        d = len(initial_radius)  # type: ignore[arg-type]
    if d is None:
        raise ValueError("Target dimension 'd' must be specified when it cannot be inferred from inputs.")
    d_int = int(d)

    # Resolve PRNG key
    eff_seed = int(random_seed) if random_seed is not None else int(seed)
    master_key = rng_key if rng_key is not None else jax.random.PRNGKey(eff_seed)
    key_init, key_p1, key_p2, key_sample = jax.random.split(master_key, 4)

    # Resolve initial positions in primal x-space: shape (chains, d)
    if eff_init is not None:
        x0_arr = jnp.asarray(eff_init, dtype=jnp.float64)
        if x0_arr.ndim == 1:
            if x0_arr.shape[0] != d_int:
                raise ValueError(f"Initial position shape {x0_arr.shape} does not match d={d_int}")
            jitter = 0.05 * jax.random.normal(key_init, shape=(chains, d_int), dtype=jnp.float64)
            init_x = jnp.tile(x0_arr[jnp.newaxis, :], (chains, 1)) + jitter
        elif x0_arr.ndim == 2 and x0_arr.shape == (chains, d_int):
            init_x = x0_arr
        else:
            raise ValueError(f"Expected x0 shape ({d_int},) or ({chains}, {d_int}), got {x0_arr.shape}")
    else:
        init_x = 0.5 * jax.random.normal(key_init, shape=(chains, d_int), dtype=jnp.float64)

    # Initial transform
    transform_init = JaxArcsinhTransform(
        radii=initial_radius,
        location=initial_location,
        mask=mask,
        d=d_int,
    )

    run_phase1 = bool(adapt_tail_scale and tune >= 40)
    phase1_grad_evals = 0
    phase2_grad_evals = 0

    if run_phase1:
        w1_steps = max(20, int(0.20 * tune))
        w2_steps = max(20, tune - w1_steps)

        z0_batch = jax.vmap(transform_init.forward)(init_x)
        base_logp_z1 = transform_init.wrap_logdensity(raw_logdensity_fn)
        l_init_vals = jax.vmap(base_logp_z1)(z0_batch)
        l_ref1 = float(jnp.max(jnp.where(jnp.isfinite(l_init_vals), l_init_vals, -100.0)))
        logdensity_z1 = _make_smooth_wall_logdensity(base_logp_z1, l_floor=l_ref1 - 45.0, wall_scale=10.0)

        warmup_p1 = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_z1,
            target_acceptance_rate=target_accept,
            max_num_doublings=max_num_doublings,
        )
        p1_keys = jax.random.split(key_p1, chains)
        (state_p1, _), info_p1 = jax.vmap(lambda k, z: warmup_p1.run(k, z, num_steps=w1_steps))(p1_keys, z0_batch)

        # Exact Phase 1 gradient evaluations: 1 init eval per chain + sum of leapfrog steps
        phase1_grad_evals = int(chains + int(jnp.sum(info_p1.info.num_integration_steps)))

        # Pull second half of Phase 1 draws back to primal x-space to calibrate transform
        half_idx = max(1, w1_steps // 2)
        z_p1_tail = info_p1.state.position[:, half_idx:, :]
        x_p1_tail = jax.vmap(jax.vmap(transform_init.inverse))(z_p1_tail)
        transform = calibrate_jax_arcsinh_transform(
            pilot_samples=x_p1_tail,
            d=d_int,
            mask=mask,
            min_radius=min_radius,
            max_radius=max_radius,
            scale_multiplier=scale_multiplier,
        )
        x_p1_final = jax.vmap(transform_init.inverse)(state_p1.position)
        pilot_x_for_ref = x_p1_tail.reshape(-1, d_int)
    else:
        w2_steps = tune
        transform = transform_init
        x_p1_final = init_x
        pilot_x_for_ref = init_x

    # Map starting positions into calibrated z-space
    z_start_batch = jax.vmap(transform.forward)(x_p1_final)
    base_logp_z2 = transform.wrap_logdensity(raw_logdensity_fn)
    z_ref_pts = jax.vmap(transform.forward)(pilot_x_for_ref)
    l_ref2_vals = jax.vmap(base_logp_z2)(z_ref_pts)
    l_ref2 = float(jnp.max(jnp.where(jnp.isfinite(l_ref2_vals), l_ref2_vals, -100.0)))
    logdensity_z2 = _make_smooth_wall_logdensity(base_logp_z2, l_floor=l_ref2 - 45.0, wall_scale=10.0)

    if w2_steps > 0:
        warmup_p2 = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_z2,
            target_acceptance_rate=target_accept,
            max_num_doublings=max_num_doublings,
        )
        p2_keys = jax.random.split(key_p2, chains)
        (state_p2, params_p2), info_p2 = jax.vmap(lambda k, z: warmup_p2.run(k, z, num_steps=w2_steps))(
            p2_keys, z_start_batch
        )

        # Exact Phase 2/3 gradient evaluations: 1 init eval per chain + sum of leapfrog steps
        phase2_grad_evals = int(chains + int(jnp.sum(info_p2.info.num_integration_steps)))

        # Pool adapted step size and diagonal inverse mass matrix across chains via geometric mean
        adapted_step_size = float(jnp.exp(jnp.mean(jnp.log(jnp.maximum(params_p2["step_size"], 1e-8)))))
        adapted_inv_mass = jnp.exp(
            jnp.mean(jnp.log(jnp.maximum(params_p2["inverse_mass_matrix"], 1e-12)), axis=0)
        )
        sampling_init_states = state_p2
        init_sampling_evals = 0
    else:
        adapted_step_size = float(step_size)
        adapted_inv_mass = jnp.ones((d_int,), dtype=jnp.float64)
        sampling_init_states = jax.vmap(lambda z: blackjax.nuts.init(z, logdensity_z2))(z_start_batch)
        init_sampling_evals = int(chains)

    # Post-Warmup Production Sampling via jax.jit + jax.vmap + jax.lax.scan
    nuts_kernel = blackjax.nuts(
        logdensity_z2,
        step_size=adapted_step_size,
        inverse_mass_matrix=adapted_inv_mass,
        max_num_doublings=max_num_doublings,
        divergence_threshold=divergence_threshold,
    ).step

    def _scan_single_chain(
        init_state: Any,
        chain_rng_key: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        step_keys = jax.random.split(chain_rng_key, draws)

        def _one_step(
            state: Any,
            subkey: Any,
        ) -> tuple[Any, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            next_state, info = nuts_kernel(subkey, state)
            return next_state, (
                next_state.position,
                info.is_divergent,
                info.num_integration_steps,
                info.acceptance_rate,
                info.energy,
            )

        _, trajectory = jax.lax.scan(_one_step, init_state, step_keys)
        return trajectory

    sample_keys = jax.random.split(key_sample, chains)
    z_draws, div_draws, n_steps_draws, acc_draws, energy_draws = jax.jit(jax.vmap(_scan_single_chain))(
        sampling_init_states, sample_keys
    )

    x_draws = jax.vmap(jax.vmap(transform.inverse))(z_draws)

    samples_np = np.asarray(x_draws, dtype=np.float64)
    z_samples_np = np.asarray(z_draws, dtype=np.float64)
    inv_mass_np = np.asarray(adapted_inv_mass, dtype=np.float64)

    sampling_grad_evals = int(init_sampling_evals + int(jnp.sum(n_steps_draws)))
    total_grad_evals = int(phase1_grad_evals + phase2_grad_evals + sampling_grad_evals)
    total_divergences = int(jnp.sum(div_draws))
    mean_acceptance = float(jnp.mean(acc_draws))

    sample_stats_dict: dict[str, np.ndarray] = {
        "diverging": np.asarray(div_draws, dtype=bool),
        "n_steps": np.asarray(n_steps_draws, dtype=np.int64),
        "acceptance_rate": np.asarray(acc_draws, dtype=np.float64),
        "energy": np.asarray(energy_draws, dtype=np.float64),
        "step_size": np.full((chains, draws), adapted_step_size, dtype=np.float64),
    }
    idata = arviz_from_samples(samples_np, sample_stats=sample_stats_dict)

    return JaxUnifiedNUTSResult(
        samples=samples_np,
        z_samples=z_samples_np,
        transform=transform,
        step_size=adapted_step_size,
        inverse_mass_matrix=inv_mass_np,
        grad_evals=total_grad_evals,
        divergences=total_divergences,
        acceptance_rate=mean_acceptance,
        phase1_grad_evals=phase1_grad_evals,
        phase2_grad_evals=phase2_grad_evals,
        sampling_grad_evals=sampling_grad_evals,
        idata=idata,
    )
