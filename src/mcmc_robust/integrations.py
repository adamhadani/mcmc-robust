r"""Idiomatic Framework Integrations for PyMC and BlackJAX / JAX (Feature F25/F26/F27/F29).

This module provides plug-and-play adapters enabling novel robust MCMC samplers
(UnifiedRobustNUTS, UnifiedRobustHMC, UnifiedRobustSampler, GentleMALA) to be used
directly with:
1. PyMC models (`PyMCModelTarget` and drop-in `sample_pymc_model(...) -> arviz.DataTree`).
2. BlackJAX and pure JAX ecosystems (`JaxArcsinhTransform` and `as_blackjax_algorithm`).

Key Capabilities:
- `PyMCModelTarget`: Wraps any `pymc.Model` extracting continuous logp and gradients
  via `model.logp_dlogp_function(ravel_inputs=True)`. Handles bijective mapping between
  unconstrained 1D flat vectors and unflattened multidimensional variable dictionaries.
- `sample_pymc_model`: Drop-in replacement for `pm.sample()` running robust samplers
  across multiple chains with 3-phase warmup adaptation, returning a standard ArviZ
  `DataTree` populated with `/posterior` (unflattened parameter arrays) and
  `/sample_stats` (diverging, step_size, n_steps, energy, acceptance_rate).
- `JaxArcsinhTransform`: Pure-JAX coordinate transform $z = \operatorname{arcsinh}(x / R)$
  with exact log-Jacobian determinant $\sum \log(R \cosh(z))$, usable with `blackjax.window_adaptation`
  and compiled `jax.jit` / `jax.lax.scan` kernels for zero Python-loop overhead.
- `as_blackjax_algorithm`: Exposes a standard BlackJAX functional `(init_fn, step_fn)` API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import arviz as az
import jax.numpy as jnp
import pymc as pm
from pymc import blocking
from xarray import DataTree

from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.nuts import UnifiedRobustNUTS
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.samplers.novel.unified_hmc import UnifiedRobustHMC

# =============================================================================
# 1. PyMC Model Target & Sampling Adapter
# =============================================================================


class PyMCModelTarget(TargetDistribution):
    r"""TargetDistribution wrapper around any continuous `pymc.Model`.

    Extracts continuous log-posterior probability and analytical gradients via PyMC's
    compiled `logp_dlogp_function(ravel_inputs=True)`. Maps bijectively between flat 1D
    vectors $x \in \mathbb{R}^d$ and named multi-dimensional parameter dictionaries
    in the unconstrained coordinate space.

    Parameters:
        model: A compiled or uncompiled `pymc.Model` instance containing continuous variables.
    """

    def __init__(self, model: pm.Model) -> None:
        self.model = model
        self._ip = model.initial_point()
        self._raveled = blocking.DictToArrayBijection.map(self._ip)
        self._point_map_info = self._raveled.point_map_info
        self._dim = int(self._raveled.data.shape[0])
        self._logp_dlogp_fn = model.logp_dlogp_function(ravel_inputs=True)
        super().__init__(d=self._dim)

    @property
    def initial_point_array(self) -> np.ndarray:
        """Returns the initial point as a flat 1D numpy array."""
        return np.array(self._raveled.data, dtype=np.float64, copy=True)

    @property
    def variable_names(self) -> list[str]:
        """Returns the list of unconstrained variable names in the flat vector."""
        return [info[0] for info in self._point_map_info]

    def unflatten(self, flat_vec: np.ndarray) -> dict[str, np.ndarray]:
        """Maps a 1D flat parameter array to a dict of unconstrained parameter arrays.

        Parameters:
            flat_vec: 1D array of shape `(d,)`.

        Returns:
            Dictionary mapping variable names to numpy arrays of their model shapes.
        """
        arr = np.asarray(flat_vec, dtype=np.float64)
        raveled = blocking.RaveledVars(arr, self._point_map_info)
        return blocking.DictToArrayBijection.rmap(raveled)

    def log_prob(self, x: np.ndarray) -> float:
        r"""Computes continuous log-posterior density $\log \pi(x)$."""
        arr = np.asarray(x, dtype=np.float64)
        try:
            val, _ = self._logp_dlogp_fn(arr)
            float_val = float(val)
            if not np.isfinite(float_val):
                return -1e300
            return float_val
        except Exception:
            return -1e300

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        r"""Computes gradient of continuous log-posterior density $\nabla \log \pi(x)$."""

        arr = np.asarray(x, dtype=np.float64)
        try:
            _, grad = self._logp_dlogp_fn(arr)
            grad_arr = np.asarray(grad, dtype=np.float64)
            if not np.all(np.isfinite(grad_arr)):
                return np.zeros(self.d, dtype=np.float64)
            return grad_arr
        except Exception:
            return np.zeros(self.d, dtype=np.float64)

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Computes both log-density and its gradient in a single evaluation."""
        arr = np.asarray(x, dtype=np.float64)
        try:
            val, grad = self._logp_dlogp_fn(arr)
            float_val = float(val)
            grad_arr = np.asarray(grad, dtype=np.float64)
            if not np.isfinite(float_val) or not np.all(np.isfinite(grad_arr)):
                return -1e300, np.zeros(self.d, dtype=np.float64)
            return float_val, grad_arr
        except Exception:
            return -1e300, np.zeros(self.d, dtype=np.float64)

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Exact ground-truth sampling is not analytically available for general PyMC models."""
        raise NotImplementedError("Exact ground truth sampling not available for arbitrary PyMC models.")

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        """Returns empty dictionary as analytical moments are not available for arbitrary PyMC models."""
        return {}


def sample_pymc_model(
    model: pm.Model,
    sampler: str | type[Sampler] | Sampler = "unified_nuts",
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    random_seed: int | None = None,
    step_size: float = 0.1,
    target_accept: float = 0.80,
    adapt_tail_scale: bool = True,
    adapt_step_size: bool = True,
    adapt_preconditioner: bool = True,
    x0: np.ndarray | Sequence[np.ndarray] | None = None,
    **sampler_kwargs: Any,
) -> DataTree:
    r"""Drop-in replacement for `pm.sample(model=...)` using robust adaptive samplers.

    Runs robust MCMC samplers (`UnifiedRobustNUTS`, `UnifiedRobustHMC`, `UnifiedRobustSampler`,
    `GentleMALA`) across multiple chains, automatically compiling the PyMC model, extracting
    parameter structures, performing 3-phase warmup adaptation, and returning a standard
    ArviZ `DataTree` with unflattened parameter arrays and diagnostic statistics.

    Parameters:
        model: Continuous PyMC model.
        sampler: Sampler identifier string ('unified_nuts', 'unified_hmc', 'unified', 'gentle_mala')
            or a `Sampler` subclass or instantiated `Sampler` object.
        draws: Number of post-warmup MCMC samples per chain.
        tune: Number of warmup/adaptation steps per chain.
        chains: Number of independent MCMC chains to sample.
        random_seed: Random seed for reproducibility.
        step_size: Initial leapfrog or Langevin step size $\epsilon > 0$.
        target_accept: Dual averaging target acceptance probability $\delta \in (0, 1)$.
        adapt_tail_scale: Whether to automatically adapt coordinate tail scale $R_i$.
        adapt_step_size: Whether to adapt step size via dual averaging.
        adapt_preconditioner: Whether to estimate diagonal inverse mass matrix.
        x0: Optional initial position(s). If None, PyMC's `initial_point` is used with jitter.
        **sampler_kwargs: Additional keyword arguments passed to the sampler constructor.

    Returns:
        ArviZ `DataTree` object with `/posterior` and `/sample_stats` groups.
    """
    target = PyMCModelTarget(model)
    d = target.d
    base_seed = random_seed if random_seed is not None else 42

    # Parse initial positions per chain
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=np.float64)
        if x0_arr.ndim == 1:
            init_positions = np.tile(x0_arr, (chains, 1))
        else:
            init_positions = x0_arr
    else:
        rng_init = np.random.default_rng(base_seed)
        init_base = target.initial_point_array
        init_positions = np.array(
            [init_base + rng_init.normal(0.0, 0.05, size=d) for _ in range(chains)], dtype=np.float64
        )

    all_chain_draws: list[np.ndarray] = []
    all_chain_diverging: list[list[bool]] = []
    all_chain_step_size: list[list[float]] = []
    all_chain_n_steps: list[list[int]] = []
    all_chain_energy: list[list[float]] = []
    all_chain_acceptance: list[list[float]] = []

    for c in range(chains):
        rng_chain = np.random.default_rng(base_seed + c * 10007)

        # Construct chain sampler
        if isinstance(sampler, Sampler):
            s = sampler
        elif isinstance(sampler, type) and issubclass(sampler, Sampler):
            s = sampler(**sampler_kwargs)
        elif isinstance(sampler, str):
            name_lower = sampler.lower().strip()
            if name_lower in ("unified_nuts", "nuts", "arcsinhtransformednuts"):
                s = UnifiedRobustNUTS(
                    step_size=step_size,
                    target_accept=target_accept,
                    adapt_step_size=adapt_step_size,
                    adapt_preconditioner=adapt_preconditioner,
                    adapt_tail_scale=adapt_tail_scale,
                    warmup_steps=tune,
                    **sampler_kwargs,
                )
            elif name_lower in ("unified_hmc", "hmc"):
                s = UnifiedRobustHMC(
                    step_size=step_size,
                    target_accept=target_accept,
                    adapt_step_size=adapt_step_size,
                    adapt_preconditioner=adapt_preconditioner,
                    adapt_tail_scale=adapt_tail_scale,
                    warmup_steps=tune,
                    **sampler_kwargs,
                )
            elif name_lower in ("unified", "unified_sampler", "langevin"):
                s = UnifiedRobustSampler(
                    step_size=step_size,
                    warmup_steps=tune,
                    **sampler_kwargs,
                )
            elif name_lower == "gentle_mala":
                s = GentleMALA(
                    step_size=step_size,
                    **sampler_kwargs,
                )
            else:
                raise ValueError(
                    f"Unknown sampler string '{sampler}'. Choose from 'unified_nuts', "
                    "'unified_hmc', 'unified', or 'gentle_mala'."
                )
        elif callable(sampler):
            s = sampler(**sampler_kwargs)
        else:
            raise TypeError(f"Invalid sampler argument type: {type(sampler)}")

        chain_samples, step_results = s.sample(
            x0=init_positions[c],
            target=target,
            n_steps=draws + tune,
            burn_in=tune,
            rng=rng_chain,
            adapt=True,
        )

        all_chain_draws.append(chain_samples)

        # Extract per-step diagnostics for post-warmup iterations
        div_list = []
        ss_list = []
        ns_list = []
        en_list = []
        acc_list = []

        post_warmup_results = step_results[tune:]
        for res in post_warmup_results:
            is_div = bool(
                float(res.stats.get("divergent", 0.0)) > 0.0 or float(res.stats.get("hamiltonian_error", 0.0)) > 1000.0
            )
            div_list.append(is_div)
            ss_list.append(float(res.stats.get("step_size", step_size)))
            ns_list.append(int(res.stats.get("n_steps", res.stats.get("leapfrog_steps", 1))))
            en_list.append(float(res.stats.get("energy", -res.state.log_prob)))
            acc_list.append(
                float(res.stats.get("acceptance_rate", res.stats.get("acceptance_prob", res.stats.get("alpha", 1.0))))
            )

        all_chain_diverging.append(div_list)
        all_chain_step_size.append(ss_list)
        all_chain_n_steps.append(ns_list)
        all_chain_energy.append(en_list)
        all_chain_acceptance.append(acc_list)

    # Unflatten draws into named variable arrays of shape (chains, draws, *var_shape)
    # First unflatten draw 0 to determine variable names and shapes
    sample_pt = target.unflatten(all_chain_draws[0][0])
    var_names = list(sample_pt.keys())

    posterior_dict: dict[str, np.ndarray] = {}
    for var in var_names:
        var_chains = []
        for c in range(chains):
            chain_var_draws = [target.unflatten(all_chain_draws[c][i])[var] for i in range(draws)]
            var_chains.append(np.stack(chain_var_draws, axis=0))
        posterior_dict[var] = np.stack(var_chains, axis=0)

    sample_stats_dict: dict[str, np.ndarray] = {
        "diverging": np.array(all_chain_diverging, dtype=bool),
        "step_size": np.array(all_chain_step_size, dtype=np.float64),
        "n_steps": np.array(all_chain_n_steps, dtype=np.int64),
        "energy": np.array(all_chain_energy, dtype=np.float64),
        "acceptance_rate": np.array(all_chain_acceptance, dtype=np.float64),
    }

    # Construct standard ArviZ DataTree
    dt = az.from_dict(
        {
            "posterior": posterior_dict,
            "sample_stats": sample_stats_dict,
        }
    )
    return dt


# =============================================================================
# 2. Pure JAX Coordinate-Wise Arcsinh Transform & BlackJAX Integration
# =============================================================================


class JaxArcsinhTransform:
    r"""Pure JAX Coordinate-Wise Arcsinh Diffeomorphism for BlackJAX and XLA.

    Defines the translation-equivariant coordinate diffeomorphism:
        $$z_i = \operatorname{arcsinh}\!\left(\frac{x_i - \mu_i}{R_i}\right), \quad x_i = \mu_i + R_i \sinh(z_i)$$
    with Jacobian factor $dx_i / dz_i = R_i \cosh(z_i)$ and log-determinant:
        $$\log |\det J(z)| = \sum_{i=1}^d \log(R_i \cosh(z_i))$$

    This eliminates gradient explosions and infinite-variance tail instabilities,
    converting polynomial tails into exponentially decaying tails in $z$-space.
    Fully compatible with `jax.jit`, `jax.grad`, `jax.lax.scan`, and `blackjax.window_adaptation`.

    Parameters:
        radius: Coordinate scale parameter $R > 0$. Either a scalar or 1D array of shape `(d,)`.
        dim: Dimensionality of the target space. Optional if `radius` is an array.
        scales: Optional alias for `radius`.
        location: Coordinate centering parameter $\mu \in \mathbb{R}^d$ (default: `0.0`).
        radii: Optional alias for `radius`.
        d: Optional alias for `dim`.
        mask: Optional boolean mask of shape `(d,)` indicating transformed coordinates (default: all `True`).
    """

    def __init__(
        self,
        radius: float | Sequence[float] | np.ndarray | jnp.ndarray = 3.0,
        dim: int | None = None,
        scales: float | Sequence[float] | np.ndarray | jnp.ndarray | None = None,
        location: float | Sequence[float] | np.ndarray | jnp.ndarray = 0.0,
        radii: float | Sequence[float] | np.ndarray | jnp.ndarray | None = None,
        d: int | None = None,
        mask: Sequence[bool] | np.ndarray | jnp.ndarray | None = None,
    ) -> None:
        effective_dim = d if d is not None else dim
        effective_radius = radii if radii is not None else (scales if scales is not None else radius)
        r_arr = jnp.asarray(effective_radius, dtype=jnp.float64)
        if r_arr.ndim == 0:
            if effective_dim is not None:
                self.radius = jnp.full((effective_dim,), r_arr, dtype=jnp.float64)
                self.d = int(effective_dim)
            else:
                self.radius = r_arr
                self.d = 1
        elif r_arr.ndim == 1:
            self.radius = r_arr
            self.d = int(r_arr.shape[0])
        else:
            raise ValueError(f"Radius must be scalar or 1D array, got shape {r_arr.shape}")

        loc_arr = jnp.asarray(location, dtype=jnp.float64)
        if loc_arr.ndim == 0:
            if self.radius.ndim == 1:
                self.location = jnp.full((self.d,), loc_arr, dtype=jnp.float64)
            else:
                self.location = loc_arr
        elif loc_arr.ndim == 1:
            self.location = loc_arr
            if self.radius.ndim == 0:
                self.d = int(loc_arr.shape[0])
                self.radius = jnp.full((self.d,), self.radius, dtype=jnp.float64)
        else:
            raise ValueError(f"Location must be scalar or 1D array, got shape {loc_arr.shape}")

        if mask is None:
            if self.radius.ndim == 1:
                self.mask = jnp.ones((self.d,), dtype=jnp.bool_)
            else:
                self.mask = jnp.asarray(True, dtype=jnp.bool_)
        else:
            m_arr = jnp.asarray(mask, dtype=jnp.bool_)
            if m_arr.ndim == 1 and self.radius.ndim == 0:
                self.d = int(m_arr.shape[0])
                self.radius = jnp.full((self.d,), self.radius, dtype=jnp.float64)
                if self.location.ndim == 0:
                    self.location = jnp.full((self.d,), self.location, dtype=jnp.float64)
            self.mask = m_arr

    @property
    def radii(self) -> jnp.ndarray:
        """Coordinate scale vector R of shape (d,)."""
        return self.radius

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""Maps primal coordinates $x \in \mathbb{R}^d$ to transformed coordinates $z \in \mathbb{R}^d$."""
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        z_warped = jnp.arcsinh((x_arr - self.location) / self.radius)
        return jnp.where(self.mask, z_warped, x_arr)

    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        r"""Maps transformed coordinates $z \in \mathbb{R}^d$ back to primal coordinates $x \in \mathbb{R}^d$."""
        z_arr = jnp.asarray(z, dtype=jnp.float64)
        z_masked = jnp.where(self.mask, z_arr, 0.0)
        x_unwarped = self.location + self.radius * jnp.sinh(z_masked)
        return jnp.where(self.mask, x_unwarped, z_arr)

    def log_jacobian_det(self, z: jnp.ndarray) -> jnp.ndarray:
        r"""Computes the log-determinant of the Jacobian $\log |\det J(z)| = \sum_i \log(R_i \cosh(z_i))$.

        Matches `CoordinateWiseArcsinhTransform.inverse()[1]` (`transforms/coordinate_arcsinh.py`)
        using the overflow-free identity $\log \cosh(z_i) = |z_i| - \log 2 + \operatorname{log1p}(e^{-2|z_i|})$.
        """
        z_arr = jnp.asarray(z, dtype=jnp.float64)
        z_masked = jnp.where(self.mask, z_arr, 0.0)
        abs_z = jnp.abs(z_masked)
        log_cosh = abs_z - jnp.log(2.0) + jnp.log1p(jnp.exp(-2.0 * abs_z))
        terms = jnp.where(self.mask, jnp.log(self.radius) + log_cosh, 0.0)
        return jnp.sum(terms, axis=-1)

    def forward_grad(self, x: jnp.ndarray | np.ndarray, grad_x: jnp.ndarray | np.ndarray) -> jnp.ndarray:
        r"""Computes the analytical pullback gradient $\nabla_z \log \pi_Z(z)$ at $z = \phi(x)$.

        Matches `CoordinateWiseArcsinhTransform.forward_grad` (`transforms/coordinate_arcsinh.py`)
        and equals `jax.grad(self.wrap_logdensity(logdensity_fn))(z)` to machine precision:
            $$\nabla_{z_i} \log \pi_Z(z) = \sqrt{R_i^2 + (x_i - \mu_i)^2} \, \nabla_{x_i} \log \pi_X(x)
              + \frac{x_i - \mu_i}{\sqrt{R_i^2 + (x_i - \mu_i)^2}}$$
        where $\sqrt{R_i^2 + (x_i - \mu_i)^2} = R_i \cosh(z_i)$ is evaluated via `jnp.hypot`
        to avoid intermediate squaring overflow in the far tails.
        """
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        gx_arr = jnp.asarray(grad_x, dtype=jnp.float64)
        xm = x_arr - self.location
        scale = jnp.hypot(self.radius, xm)
        gz_warped = scale * gx_arr + (xm / scale)
        return jnp.where(self.mask, gz_warped, gx_arr)

    def wrap_logdensity(
        self,
        logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        r"""Wraps a primal log-density function $\log \pi_X(x)$ into the transformed log-density $\log \pi_Z(z)$.

        $$\log \pi_Z(z) = \log \pi_X(\mu + R \sinh(z)) + \sum_{i=1}^d \log(R_i \cosh(z_i))$$
        """

        def transformed_logdensity(z: jnp.ndarray) -> jnp.ndarray:
            x = self.inverse(z)
            return logdensity_fn(x) + self.log_jacobian_det(z)

        return transformed_logdensity

    @classmethod
    def calibrate_from_samples(
        cls,
        samples: np.ndarray | jnp.ndarray,
        min_scale: float = 0.1,
        max_scale: float = 100.0,
    ) -> JaxArcsinhTransform:
        r"""Estimates coordinate locations $\mu_i$ and normal-consistent scales $R_i$ from sample IQR.

        $$\mu_i = \operatorname{median}(x_i), \quad R_i = \operatorname{clip}\!\left(\frac{\operatorname{IQR}(x_i - \mu_i)}{1.3489795}, R_{\min}, R_{\max}\right)$$
        """
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])

        loc = np.median(arr, axis=0)
        centered = arr - loc
        q25 = np.percentile(centered, 25.0, axis=0)
        q75 = np.percentile(centered, 75.0, axis=0)
        iqr_scale = np.maximum((q75 - q25) / 1.3489795, 1e-6)
        scales = np.clip(iqr_scale, min_scale, max_scale)
        return cls(radius=scales, location=loc)

    def scan_nuts(
        self,
        rng_key: Any,
        logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
        init_position: jnp.ndarray,
        num_samples: int = 1000,
        num_warmup: int = 500,
        target_accept: float = 0.80,
    ) -> tuple[jnp.ndarray, Any]:
        r"""Runs fast XLA-compiled NUTS using BlackJAX and `jax.lax.scan` in transformed space.

        Returns primal coordinate draws with zero Python-loop overhead.
        """
        import blackjax

        transformed_logdensity = self.wrap_logdensity(logdensity_fn)
        init_z = self.forward(jnp.asarray(init_position, dtype=jnp.float64))

        # 1. Warmup adaptation in transformed space
        warmup_key, sample_key = jax.random.split(rng_key)
        adapt = blackjax.window_adaptation(
            blackjax.nuts,
            transformed_logdensity,
            target_acceptance_rate=target_accept,
        )
        (last_state, parameters), _ = adapt.run(warmup_key, init_z, num_steps=num_warmup)

        # 2. Production kernel
        kernel = blackjax.nuts(transformed_logdensity, **parameters).step

        def one_step(state, key):
            state, info = kernel(key, state)
            return state, (state.position, info)

        keys = jax.random.split(sample_key, num_samples)
        _, (z_samples, info) = jax.lax.scan(one_step, last_state, keys)

        # 3. Pull back to primal space via vectorization
        x_samples = jax.vmap(self.inverse)(z_samples)
        return x_samples, info


# =============================================================================
# 3. BlackJAX Functional Algorithm Adapter
# =============================================================================


class BlackJAXAlgorithmState(NamedTuple):
    """Immutable state container for BlackJAX functional compatibility."""

    position: np.ndarray
    log_p: float
    grad_log_p: np.ndarray
    step_num: int
    auxiliary: dict[str, Any]


class BlackJAXAlgorithm:
    r"""Wraps a `mcmc_robust.core.Sampler` and `TargetDistribution` as a BlackJAX-style algorithm.

    Exposes the standard BlackJAX functional tuple interface:
        `state = algo.init(position)`
        `state, info = algo.step(rng_key, state)`
    """

    def __init__(self, sampler: Sampler, target: TargetDistribution) -> None:
        self.sampler = sampler
        self.target = target

    def init(self, position: np.ndarray) -> BlackJAXAlgorithmState:
        """Initializes algorithm state at given position."""
        pos = np.asarray(position, dtype=np.float64, copy=True)
        lp, grad = self.target.log_prob_and_grad(pos)
        return BlackJAXAlgorithmState(
            position=pos,
            log_p=float(lp),
            grad_log_p=np.asarray(grad, dtype=np.float64),
            step_num=0,
            auxiliary={},
        )

    def step(
        self,
        rng_key: Any,
        state: BlackJAXAlgorithmState,
    ) -> tuple[BlackJAXAlgorithmState, dict[str, Any]]:
        """Executes a single MCMC transition step."""
        # Convert JAX PRNG key to numpy seed/generator if necessary
        if hasattr(rng_key, "__len__") or hasattr(rng_key, "shape"):
            seed_val = int(jax.random.randint(rng_key, (), 0, 2**31 - 1))
        else:
            seed_val = int(rng_key)
        rng = np.random.default_rng(seed_val)

        core_state = ChainState(
            x=state.position,
            log_prob=state.log_p,
            grad_log_prob=state.grad_log_p,
            auxiliary=dict(state.auxiliary),
        )

        step_res = self.sampler.step(
            state=core_state,
            target=self.target,
            rng=rng,
        )

        new_pos = np.asarray(step_res.state.x, dtype=np.float64)
        new_grad = (
            np.asarray(step_res.state.grad_log_prob, dtype=np.float64)
            if step_res.state.grad_log_prob is not None
            else np.zeros_like(new_pos)
        )
        new_state = BlackJAXAlgorithmState(
            position=new_pos,
            log_p=float(step_res.state.log_prob),
            grad_log_p=new_grad,
            step_num=state.step_num + 1,
            auxiliary=dict(step_res.state.auxiliary),
        )

        return new_state, dict(step_res.stats)


def as_blackjax_algorithm(
    sampler: Sampler,
    target: TargetDistribution,
) -> BlackJAXAlgorithm:
    r"""Constructs a BlackJAX-compatible algorithm object with `(init, step)` methods."""
    return BlackJAXAlgorithm(sampler=sampler, target=target)


from mcmc_robust.jax_unified_nuts import (
    JaxUnifiedNUTSResult,
    arviz_from_samples,
    calibrate_jax_arcsinh_transform,
    sample_blackjax_unified_nuts,
)

__all__ = [
    "BlackJAXAlgorithm",
    "BlackJAXAlgorithmState",
    "JaxArcsinhTransform",
    "JaxUnifiedNUTSResult",
    "PyMCModelTarget",
    "arviz_from_samples",
    "as_blackjax_algorithm",
    "calibrate_jax_arcsinh_transform",
    "sample_blackjax_unified_nuts",
    "sample_pymc_model",
]
