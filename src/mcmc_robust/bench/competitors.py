r"""Direct-competitor sampler adapters with one calling convention and honest gradient accounting.

Every adapter has the signature

    adapter(logdensity, d, seed, tune, draws, chains, init, **options) -> (samples, telemetry)

- ``logdensity`` is a `TargetDistribution` or a JAX-traceable callable ``x -> log pi(x)``.
- ``init`` is ``None`` (the sampler's own initialisation) or initial positions of shape
  ``(chains, d)`` (a ``(d,)`` vector is shared by all chains). Passing the same array to every
  arm gives the arms the same starting points.
- ``samples`` has shape ``(chains, draws, d)``.
- ``telemetry`` contains at least `STANDARD_TELEMETRY_KEYS`. ``total_grad_evals`` counts every
  evaluation of the target's gradient, including warm-up and any adaptation or training cost, and
  ``grad_eval_accounting`` says how it was counted.

Arms:
- ``run_stock_nuts``: BlackJAX NUTS with Stan-style window adaptation
  (`bench.sota_adapters.run_blackjax_sampler`).
- ``run_arcsinh_nuts``: the repo's method, BlackJAX NUTS on the calibrated coordinate-wise arcsinh
  pullback (`jax_unified_nuts.sample_blackjax_unified_nuts`).
- ``run_nutpie_diag``: nutpie's default NUTS (Seyboldt et al.), diagonal mass matrix adapted from
  draw and gradient variances ("Fisher" adaptation).
- ``run_nutpie_flow``: nutpie's experimental normalizing-flow adaptation, which learns a flow
  (affine and per-coordinate sinh-arcsinh layers inside coupling layers) by minimising a Fisher
  divergence on warm-up draws and their gradients, then runs NUTS in the flow's coordinates.

How nutpie's gradients are counted (exact counts, not estimates):
- nutpie is driven through ``nutpie.compiled_pyfunc.from_pyfunc`` with the target's JAX log density,
  the route nutpie's own PyMC JAX backend uses. Without a flow, nutpie's Rust core calls our Python
  ``logp`` callback once per gradient, and the callback counts the calls.
- With the flow, nutpie evaluates the target inside its Python ``TransformAdapter``
  (``init_from_transformed_position`` and ``init_from_untransformed_position``, each one
  ``jax.value_and_grad`` of the log density). The adapter is wrapped in a proxy that counts these
  calls and times ``update``, the flow-training step.
- Flow training itself does not evaluate the target: nutpie fits the flow to the positions,
  gradients and log densities its warm-up draws already computed. Its cost is compute time, which
  is reported separately as ``flow_training_seconds`` and is included in ``wall_clock_time``.
- ``total_grad_evals`` is the sum of both counters. It includes initial points and step-size
  initialisation, so it is slightly larger than nutpie's leapfrog-step total (``n_steps`` summed
  over warm-up and draws, the figure nutpie's own documentation uses), which is reported alongside.
  ``draw_grad_evals`` is the post-warm-up leapfrog total; ``warmup_grad_evals`` is the remainder.

nutpie is an optional dependency (``uv sync --extra competitors``); the nutpie adapters raise
`CompetitorUnavailableError` when it is missing. They rely on nutpie internals
(``PyFuncModel._make_model`` and the ``TransformAdapter`` method names) and are pinned to
``nutpie>=0.16.11,<0.17`` in ``pyproject.toml``.

Brešar & Mijatović (2026), "Diffeomorphic Markov Chain Monte Carlo" (arXiv:2608.04284), has no
adapter: no official code was found (see ``competitor_runner.EXCLUDED_ARMS``), and the paper leaves
the settings of its variational tuning, ball-walk and slice-sampling steps unspecified, so a
re-implementation could not be checked against their published numbers.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mcmc_robust.bench.sota_adapters import SOTATargetAdapter, run_blackjax_sampler
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.jax_unified_nuts import sample_blackjax_unified_nuts

LogDensity = TargetDistribution | Callable[[jnp.ndarray], jnp.ndarray]
AdapterResult = tuple[np.ndarray, dict[str, Any]]

#: Maximum NUTS tree depth shared by every arm (BlackJAX's and nutpie's default of 10 doublings).
DEFAULT_MAX_DEPTH: int = 10

#: Dual-averaging target acceptance shared by every arm (the default of all three libraries).
DEFAULT_TARGET_ACCEPT: float = 0.8

#: Keys every adapter's telemetry dictionary contains.
STANDARD_TELEMETRY_KEYS: tuple[str, ...] = (
    "arm",
    "total_grad_evals",
    "warmup_grad_evals",
    "draw_grad_evals",
    "divergences",
    "warmup_divergences",
    "wall_clock_time",
    "grad_eval_accounting",
    "seed",
    "tune",
    "draws",
    "chains",
    "max_depth",
    "target_accept",
)

COMPETITORS_EXTRA_HINT = "install the optional extra with `uv sync --extra competitors`"


class CompetitorUnavailableError(ImportError):
    """Raised when a competitor's optional dependency is not installed."""


# =============================================================================
# 1. Shared helpers
# =============================================================================


class JaxLogdensityTarget(TargetDistribution):
    """Wraps a JAX log density as a `TargetDistribution` for runners that expect one.

    `SOTATargetAdapter` returns ``jax_log_prob`` unchanged, so BlackJAX samples the callable itself.
    """

    def __init__(self, logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray], d: int) -> None:
        super().__init__(d=d)
        self.jax_log_prob = logdensity_fn
        self._value_and_grad = jax.jit(jax.value_and_grad(logdensity_fn))

    def log_prob(self, x: np.ndarray) -> float:
        value, _ = self._value_and_grad(jnp.asarray(x, dtype=jnp.float64))
        return float(value)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        _, grad = self._value_and_grad(jnp.asarray(x, dtype=jnp.float64))
        return np.asarray(grad, dtype=np.float64)

    def sample_ground_truth(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        raise NotImplementedError("A bare log density has no exact sampler.")

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        raise NotImplementedError("A bare log density has no known moments.")


def resolve_logdensity(logdensity: LogDensity, d: int | None) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], int]:
    """Returns the JAX log density and the dimension for a target or a callable."""
    if isinstance(logdensity, TargetDistribution):
        if d is not None and int(d) != logdensity.d:
            raise ValueError(f"d={d} does not match the target dimension {logdensity.d}")
        return SOTATargetAdapter(logdensity).jax_logdensity_fn(), logdensity.d
    if not callable(logdensity):
        raise TypeError(f"logdensity must be a TargetDistribution or a callable, got {type(logdensity).__name__}")
    if d is None:
        raise ValueError("d is required when logdensity is a bare callable")
    return logdensity, int(d)


def as_target(logdensity: LogDensity, d: int | None) -> TargetDistribution:
    """Returns ``logdensity`` itself if it is a target, otherwise wraps the callable."""
    if isinstance(logdensity, TargetDistribution):
        resolve_logdensity(logdensity, d)
        return logdensity
    fn, d_int = resolve_logdensity(logdensity, d)
    return JaxLogdensityTarget(fn, d_int)


def validate_init(init: np.ndarray | None, chains: int, d: int) -> np.ndarray | None:
    """Returns initial positions of shape ``(chains, d)``, or None to keep the sampler's default."""
    if init is None:
        return None
    arr = np.asarray(init, dtype=np.float64)
    if arr.shape == (d,):
        arr = np.tile(arr, (chains, 1))
    if arr.shape != (chains, d):
        raise ValueError(f"init must have shape ({d},) or ({chains}, {d}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("init contains non-finite values")
    return np.ascontiguousarray(arr)


def _base_telemetry(
    arm: str, seed: int, tune: int, draws: int, chains: int, max_depth: int, target_accept: float
) -> dict[str, Any]:
    return {
        "arm": arm,
        "seed": int(seed),
        "tune": int(tune),
        "draws": int(draws),
        "chains": int(chains),
        "max_depth": int(max_depth),
        "target_accept": float(target_accept),
    }


# =============================================================================
# 2. BlackJAX reference arms
# =============================================================================


def run_stock_nuts(
    logdensity: LogDensity,
    d: int | None = None,
    seed: int = 0,
    tune: int = 500,
    draws: int = 1000,
    chains: int = 4,
    init: np.ndarray | None = None,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
) -> AdapterResult:
    """Stock BlackJAX NUTS with window adaptation (`run_blackjax_sampler`), chains run one by one.

    ``init=None`` starts every chain at the origin (`run_blackjax_sampler`'s default).
    """
    target = as_target(logdensity, d)
    x0 = validate_init(init, chains, target.d)
    t0 = time.perf_counter()
    samples, info = run_blackjax_sampler(
        target,
        sampler_type="nuts",
        n_samples=draws,
        n_warmup=tune,
        n_chains=chains,
        seed=seed,
        x0=x0,
        max_num_doublings=max_depth,
        target_accept=target_accept,
    )
    wall = time.perf_counter() - t0
    # run_blackjax_sampler sums info.num_integration_steps; each chain's initial state also
    # evaluates the gradient once (window_adaptation's init), so add one per chain.
    warmup = int(info["warmup_grad_evals"]) + int(chains)
    draw = int(info["draw_grad_evals"])
    telemetry = _base_telemetry("stock_nuts", seed, tune, draws, chains, max_depth, target_accept)
    telemetry.update(
        {
            "total_grad_evals": warmup + draw,
            "warmup_grad_evals": warmup,
            "draw_grad_evals": draw,
            "divergences": int(info["divergences"]),
            "warmup_divergences": int(info["warmup_divergences"]),
            "wall_clock_time": float(wall),
            "grad_eval_accounting": (
                "exact: BlackJAX info.num_integration_steps summed over warm-up and sampling, "
                "plus one initial gradient per chain"
            ),
            "max_tree_depth_seen": int(info["max_tree_depth"]),
        }
    )
    return np.asarray(samples, dtype=np.float64), telemetry


def run_arcsinh_nuts(
    logdensity: LogDensity,
    d: int | None = None,
    seed: int = 0,
    tune: int = 500,
    draws: int = 1000,
    chains: int = 4,
    init: np.ndarray | None = None,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
) -> AdapterResult:
    """The repo's method: `sample_blackjax_unified_nuts` (3-phase arcsinh-warped BlackJAX NUTS).

    ``tune`` includes the Phase 1 calibration. ``warmup_divergences`` is None because the pipeline
    does not return warm-up divergences.
    """
    fn, d_int = resolve_logdensity(logdensity, d)
    source: LogDensity = logdensity if isinstance(logdensity, TargetDistribution) else fn
    x0 = validate_init(init, chains, d_int)
    t0 = time.perf_counter()
    result = sample_blackjax_unified_nuts(
        source,
        d=d_int,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
        x0=x0,
        target_accept=target_accept,
        max_num_doublings=max_depth,
    )
    wall = time.perf_counter() - t0
    warmup = int(result.phase1_grad_evals + result.phase2_grad_evals)
    draw = int(result.sampling_grad_evals)
    telemetry = _base_telemetry("arcsinh_nuts", seed, tune, draws, chains, max_depth, target_accept)
    telemetry.update(
        {
            "total_grad_evals": warmup + draw,
            "warmup_grad_evals": warmup,
            "draw_grad_evals": draw,
            "divergences": int(result.divergences),
            "warmup_divergences": None,
            "wall_clock_time": float(wall),
            "grad_eval_accounting": (
                "exact: BlackJAX info.num_integration_steps summed over Phase 1, Phase 2/3 warm-up and "
                "sampling, plus one initial gradient per chain per warm-up phase"
            ),
            "phase1_grad_evals": int(result.phase1_grad_evals),
            "phase2_grad_evals": int(result.phase2_grad_evals),
            "acceptance_rate": float(result.acceptance_rate),
        }
    )
    return np.asarray(result.samples, dtype=np.float64), telemetry


# =============================================================================
# 3. nutpie arms
# =============================================================================


def nutpie_available() -> bool:
    """True when nutpie can be imported."""
    try:
        import nutpie  # noqa: F401
    except ImportError:
        return False
    return True


def nutpie_flow_available() -> bool:
    """True when nutpie's normalizing-flow adaptation can be imported (needs flowjax and equinox)."""
    if not nutpie_available():
        return False
    try:
        import nutpie.transform_adapter  # noqa: F401
    except ImportError:
        return False
    return True


def _import_nutpie(adaptation: str) -> Any:
    try:
        import nutpie
    except ImportError as exc:
        raise CompetitorUnavailableError(f"nutpie is not installed; {COMPETITORS_EXTRA_HINT}") from exc
    if adaptation == "flow" and not nutpie_flow_available():
        raise CompetitorUnavailableError(
            f"nutpie's flow adaptation needs flowjax and equinox; {COMPETITORS_EXTRA_HINT}"
        )
    return nutpie


class InitPositionQueue:
    """nutpie ``make_initial_point_fn`` that hands out fixed initial positions, one per call.

    nutpie calls it once per chain with a random seed, so the given rows are assigned to chains in
    call order. It raises once the rows are used up (e.g. if nutpie retries a failed start) rather
    than inventing a new position.
    """

    def __init__(self, positions: np.ndarray) -> None:
        self._positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
        if self._positions.ndim != 2:
            raise ValueError(f"positions must have shape (chains, d), got {self._positions.shape}")
        self._lock = threading.Lock()
        self.n_served = 0

    def __call__(self, seed: int) -> np.ndarray:
        with self._lock:
            if self.n_served >= self._positions.shape[0]:
                raise RuntimeError(
                    f"nutpie asked for more initial positions than the {self._positions.shape[0]} provided"
                )
            row = self._positions[self.n_served].copy()
            self.n_served += 1
        return row


@dataclass
class _EvalCounter:
    """Thread-safe counters for target evaluations and flow training made during one nutpie run."""

    logp_callback: int = 0
    flow_adapter: int = 0
    flow_updates: int = 0
    flow_training_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_logp(self) -> None:
        with self._lock:
            self.logp_callback += 1

    def add_adapter_eval(self) -> None:
        with self._lock:
            self.flow_adapter += 1

    def add_update(self, seconds: float) -> None:
        with self._lock:
            self.flow_updates += 1
            self.flow_training_seconds += seconds


class _CountingTransformAdapter:
    """Proxy around nutpie's ``TransformAdapter``.

    It counts the two methods that evaluate the target (each one ``value_and_grad`` of the log
    density) and times ``update`` (flow training). Every other attribute is forwarded, since
    nutpie's Rust core reaches the adapter only through attribute lookup.
    """

    def __init__(self, inner: Any, counter: _EvalCounter) -> None:
        self._inner = inner
        self._counter = counter

    def init_from_transformed_position(self, *args: Any) -> Any:
        self._counter.add_adapter_eval()
        return self._inner.init_from_transformed_position(*args)

    def init_from_untransformed_position(self, *args: Any) -> Any:
        self._counter.add_adapter_eval()
        return self._inner.init_from_untransformed_position(*args)

    def update(self, *args: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return self._inner.update(*args)
        finally:
            self._counter.add_update(time.perf_counter() - t0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@functools.cache
def _counting_model_class() -> type:
    """Builds a ``PyFuncModel`` subclass whose flow adapter is wrapped in `_CountingTransformAdapter`.

    Mirrors ``nutpie.compiled_pyfunc.PyFuncModel._make_model`` (nutpie 0.16.11) apart from the wrap.
    """
    import dataclasses
    from functools import partial

    from nutpie import _lib
    from nutpie.compiled_pyfunc import PyFuncModel

    @dataclasses.dataclass(frozen=True)
    class CountingPyFuncModel(PyFuncModel):  # type: ignore[misc]
        _eval_counter: Any = None

        def _make_model(self, init_mean: Any) -> Any:
            counter = self._eval_counter

            def make_logp_func() -> Any:
                return partial(self._make_logp_func(), **self._shared_data)

            def make_expand_func(seed1: int, seed2: int, chain: int) -> Any:
                return partial(self._make_expand_func(seed1, seed2, chain), **self._shared_data)

            make_adapter = None
            if self._raw_logp_fn is not None:
                outer_kwargs = dict(self._transform_adapt_args or {})
                raw_logp_fn = self._raw_logp_fn

                def make_adapter(*args: Any, **kwargs: Any) -> Any:
                    from nutpie.transform_adapter import make_transform_adapter

                    inner = make_transform_adapter(**outer_kwargs)(*args, **kwargs, logp_fn=raw_logp_fn)
                    return _CountingTransformAdapter(inner, counter)

            return _lib.PyModel(
                make_logp_func,
                make_expand_func,
                self._variables,
                self.n_dim,
                dim_sizes=self._dim_sizes,
                coords=self._coords,
                init_point_func=self._make_initial_points,
                transform_adapter=make_adapter,
            )

    return CountingPyFuncModel


def _build_nutpie_model(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d: int,
    counter: _EvalCounter,
    init_queue: InitPositionQueue | None,
    with_flow: bool,
) -> Any:
    import dataclasses

    from nutpie.compiled_pyfunc import from_pyfunc

    value_and_grad = jax.jit(jax.value_and_grad(logdensity_fn))

    def make_logp_fn() -> Callable[..., tuple[float, np.ndarray]]:
        def logp(x: np.ndarray, **_shared: Any) -> tuple[float, np.ndarray]:
            counter.add_logp()
            value, grad = value_and_grad(x)
            return float(value), np.asarray(grad, dtype=np.float64)

        return logp

    def make_expand_fn(seed1: int, seed2: int, chain: int) -> Callable[..., dict[str, np.ndarray]]:
        def expand(x: np.ndarray, **_shared: Any) -> dict[str, np.ndarray]:
            return {"x": np.array(x, dtype=np.float64)}

        return expand

    def raw_logp_fn(x: jnp.ndarray) -> tuple[jnp.ndarray]:
        # nutpie's TransformAdapter differentiates `logp_fn(x)[0]`.
        return (logdensity_fn(x),)

    base = from_pyfunc(
        ndim=d,
        make_logp_fn=make_logp_fn,
        make_expand_fn=make_expand_fn,
        expanded_dtypes=[np.dtype("float64")],
        expanded_shapes=[(d,)],
        expanded_names=["x"],
        make_initial_point_fn=init_queue,
        raw_logp_fn=raw_logp_fn if with_flow else None,
    )
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return _counting_model_class()(**fields, _eval_counter=counter)


def _nutpie_settings(trace: Any) -> dict[str, Any]:
    """A compact subset of the sampler settings nutpie stores with the trace."""
    raw = trace.sample_stats.attrs.get("inference_library_settings")
    if raw is None:
        return {}
    settings = json.loads(raw).get("settings", {})
    keys = ("num_tune", "num_draws", "maxdepth", "max_energy_error", "num_chains")
    out = {k: settings[k] for k in keys if k in settings}
    adapt = settings.get("adapt_options", {})
    step = adapt.get("step_size_settings", {})
    if "target_accept" in step:
        out["target_accept"] = step["target_accept"]
    for key in ("transform_update_freq", "use_grad_based_mass_matrix"):
        if key in adapt:
            out[key] = adapt[key]
    return out


def run_nutpie(
    logdensity: LogDensity,
    d: int | None = None,
    seed: int = 0,
    tune: int = 500,
    draws: int = 1000,
    chains: int = 4,
    init: np.ndarray | None = None,
    *,
    adaptation: str = "diag",
    max_depth: int = DEFAULT_MAX_DEPTH,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
    cores: int = 1,
    flow_options: Mapping[str, Any] | None = None,
) -> AdapterResult:
    """Runs nutpie NUTS on a JAX log density and counts every target-gradient evaluation.

    Args:
        adaptation: ``"diag"`` (nutpie's default gradient-based diagonal mass matrix) or ``"flow"``
            (normalizing-flow adaptation with nutpie's default flow settings).
        cores: Chains nutpie runs in parallel. The default of 1 runs them one after another, as
            nutpie's documentation recommends for the flow and as the BlackJAX stock arm does.
        flow_options: Optional keyword arguments for ``with_transform_adapt`` (flow only).
        init: ``None`` keeps nutpie's own Uniform(-2, 2) initialisation.
    """
    if adaptation not in ("diag", "flow"):
        raise ValueError(f"adaptation must be 'diag' or 'flow', got {adaptation!r}")
    nutpie = _import_nutpie(adaptation)
    fn, d_int = resolve_logdensity(logdensity, d)
    x0 = validate_init(init, chains, d_int)

    counter = _EvalCounter()
    init_queue = InitPositionQueue(x0) if x0 is not None else None
    model = _build_nutpie_model(fn, d_int, counter, init_queue, with_flow=adaptation == "flow")
    if adaptation == "flow" and flow_options:
        model = model.with_transform_adapt(**dict(flow_options))

    t0 = time.perf_counter()
    trace = nutpie.sample(
        model,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        seed=seed,
        save_warmup=True,
        progress_bar=False,
        adaptation=adaptation,
        maxdepth=max_depth,
        target_accept=target_accept,
    )
    wall = time.perf_counter() - t0

    samples = np.asarray(trace.posterior["x"].values, dtype=np.float64)
    leapfrog_draws = int(trace.sample_stats["n_steps"].values.sum())
    leapfrog_warmup = int(trace.warmup_sample_stats["n_steps"].values.sum()) if tune > 0 else 0
    divergences = int(trace.sample_stats["diverging"].values.sum())
    warmup_divergences = int(trace.warmup_sample_stats["diverging"].values.sum()) if tune > 0 else 0
    total = int(counter.logp_callback + counter.flow_adapter)

    telemetry = _base_telemetry(f"nutpie_{adaptation}", seed, tune, draws, chains, max_depth, target_accept)
    telemetry.update(
        {
            "total_grad_evals": total,
            "warmup_grad_evals": total - leapfrog_draws,
            "draw_grad_evals": leapfrog_draws,
            "divergences": divergences,
            "warmup_divergences": warmup_divergences,
            "wall_clock_time": float(wall),
            "grad_eval_accounting": (
                "exact: calls of nutpie's logp callback plus target evaluations inside the flow's "
                "TransformAdapter; flow training reuses stored warm-up gradients and adds none"
            ),
            "adaptation": adaptation,
            "nutpie_version": str(nutpie.__version__),
            "cores": int(cores),
            "leapfrog_steps_warmup": leapfrog_warmup,
            "leapfrog_steps_draws": leapfrog_draws,
            "logp_callback_evals": int(counter.logp_callback),
            "flow_adapter_evals": int(counter.flow_adapter),
            "flow_updates": int(counter.flow_updates),
            "flow_training_seconds": float(counter.flow_training_seconds),
            "maxdepth_reached_draws": int(trace.sample_stats["maxdepth_reached"].values.sum()),
            "init_positions_served": int(init_queue.n_served) if init_queue is not None else 0,
            "nutpie_settings": _nutpie_settings(trace),
        }
    )
    return samples, telemetry


def run_nutpie_diag(
    logdensity: LogDensity,
    d: int | None = None,
    seed: int = 0,
    tune: int = 500,
    draws: int = 1000,
    chains: int = 4,
    init: np.ndarray | None = None,
    **options: Any,
) -> AdapterResult:
    """nutpie default: NUTS with the gradient-based diagonal mass-matrix adaptation."""
    return run_nutpie(logdensity, d, seed, tune, draws, chains, init, adaptation="diag", **options)


def run_nutpie_flow(
    logdensity: LogDensity,
    d: int | None = None,
    seed: int = 0,
    tune: int = 500,
    draws: int = 1000,
    chains: int = 4,
    init: np.ndarray | None = None,
    **options: Any,
) -> AdapterResult:
    """nutpie with its experimental normalizing-flow adaptation (default flow settings)."""
    return run_nutpie(logdensity, d, seed, tune, draws, chains, init, adaptation="flow", **options)


#: Adapter for each arm, keyed by the arm name used in reports.
COMPETITOR_ADAPTERS: dict[str, Callable[..., AdapterResult]] = {
    "stock_nuts": run_stock_nuts,
    "arcsinh_nuts": run_arcsinh_nuts,
    "nutpie_diag": run_nutpie_diag,
    "nutpie_flow": run_nutpie_flow,
}
