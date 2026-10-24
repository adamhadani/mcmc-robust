r"""Stan models as JAX log densities, through BridgeStan.

`bridgestan_logdensity_fn(model)` turns a compiled Stan model into a JAX-traceable
`logdensity_fn(theta) -> scalar` on Stan's unconstrained parameter space, so that
`sample_blackjax_unified_nuts` (and every transform kind) can sample it unchanged.

- `model` is a `bridgestan.StanModel`, or any object with its `param_unc_num()`,
  `log_density(theta, propto=..., jacobian=...)` and
  `log_density_gradient(theta, propto=..., jacobian=...)` methods. This module does not import
  BridgeStan (the optional `posteriordb` extra).
- The density is Stan's: unnormalised (`propto=True`), with the log-Jacobian of the
  unconstraining transforms (`jacobian=True`), as Stan's own NUTS samples it.
- The model is evaluated on the host with `jax.pure_callback`. The gradient comes from BridgeStan
  through a `jax.custom_vjp`, so a value-and-gradient call costs one `log_density_gradient` and a
  plain value call one `log_density`. Under `jax.vmap` the callback runs once per batch element.
- A Stan exception (a parameter outside its support, a failed ODE solve) is returned as
  $-\infty$ with a zero gradient. NUTS treats that point as divergent and rejects it, as Stan does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


class StanLogDensity(Protocol):
    """The part of `bridgestan.StanModel` the adapter uses."""

    def param_unc_num(self) -> int: ...

    def log_density(self, theta_unc: np.ndarray, *, propto: bool = ..., jacobian: bool = ...) -> float: ...

    def log_density_gradient(
        self, theta_unc: np.ndarray, *, propto: bool = ..., jacobian: bool = ...
    ) -> tuple[float, Any]: ...


def bridgestan_logdensity_fn(model: StanLogDensity) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Returns a JAX-traceable log density of `model` on its unconstrained space (see the module docstring)."""
    d = int(model.param_unc_num())
    value_shape = jax.ShapeDtypeStruct((), jnp.float64)
    grad_shape = jax.ShapeDtypeStruct((d,), jnp.float64)

    def _value(theta: Any) -> np.ndarray:
        try:
            lp = model.log_density(np.asarray(theta, dtype=np.float64), propto=True, jacobian=True)
        except RuntimeError:
            lp = -np.inf
        return np.asarray(lp, dtype=np.float64)

    def _value_and_grad(theta: Any) -> tuple[np.ndarray, np.ndarray]:
        try:
            lp, grad = model.log_density_gradient(np.asarray(theta, dtype=np.float64), propto=True, jacobian=True)
        except RuntimeError:
            return np.asarray(-np.inf, dtype=np.float64), np.zeros(d, dtype=np.float64)
        return np.asarray(lp, dtype=np.float64), np.asarray(grad, dtype=np.float64)

    @jax.custom_vjp
    def logdensity(theta: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(_value, value_shape, theta, vmap_method="sequential")

    def _fwd(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.pure_callback(_value_and_grad, (value_shape, grad_shape), theta, vmap_method="sequential")

    def _bwd(grad: jnp.ndarray, cotangent: jnp.ndarray) -> tuple[jnp.ndarray]:
        return (cotangent * grad,)

    logdensity.defvjp(_fwd, _bwd)
    return logdensity
