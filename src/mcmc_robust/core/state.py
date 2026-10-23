"""Immutable Markov chain state and step result data structures."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChainState:
    r"""Immutable representation of the Markov chain state at a single step.

    Attributes:
        x: Position vector in primary state space, shape (d,).
        log_prob: Cached log \pi(x).
        grad_log_prob: Cached \nabla log \pi(x), shape (d,) or None for zeroth-order.
        auxiliary: Sampler-specific metadata (e.g. transformed coordinates y = \psi(x)).
    """

    x: np.ndarray
    log_prob: float
    grad_log_prob: np.ndarray | None = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforces immutability and array dtype/writeable constraints."""
        # Convert x to read-only float64 array with defensive copy
        x_arr = np.array(self.x, dtype=np.float64, copy=True)
        x_arr.flags.writeable = False
        object.__setattr__(self, "x", x_arr)
        object.__setattr__(self, "log_prob", float(self.log_prob))

        if self.grad_log_prob is not None:
            g_arr = np.array(self.grad_log_prob, dtype=np.float64, copy=True)
            if g_arr.shape != x_arr.shape:
                raise ValueError(f"grad_log_prob shape {g_arr.shape} does not match x shape {x_arr.shape}")
            g_arr.flags.writeable = False
            object.__setattr__(self, "grad_log_prob", g_arr)

    @property
    def aux(self) -> Mapping[str, Any]:
        """Alias for auxiliary."""
        return self.auxiliary

    def copy_with(self, **changes: Any) -> "ChainState":
        """Returns a new ChainState with updated fields."""
        new_x = changes.get("x", self.x)
        new_lp = changes.get("log_prob", self.log_prob)
        new_grad = changes.get("grad_log_prob", self.grad_log_prob)
        new_aux = changes.get("auxiliary", changes.get("aux", self.auxiliary))
        return ChainState(
            x=new_x,
            log_prob=new_lp,
            grad_log_prob=new_grad,
            auxiliary=new_aux,
        )


@dataclass(frozen=True)
class StepResult:
    r"""Encapsulates the complete outcome of an MCMC transition step.

    Attributes:
        state: The resulting ChainState (new state if accepted, prior state if rejected).
        accepted: Boolean flag indicating proposal acceptance.
        accept_prob: Metropolis-Hastings acceptance probability \alpha \in [0, 1].
        stats: Diagnostic telemetry dictionary for the step.
        proposal: Optional candidate position vector proposed in this step.
        step_size: Step size h used during this transition.
    """

    state: ChainState
    accepted: bool
    accept_prob: float
    stats: Mapping[str, Any] = field(default_factory=dict)
    proposal: np.ndarray | None = None
    step_size: float = 0.0

    @property
    def acceptance_prob(self) -> float:
        """Alias for accept_prob."""
        return self.accept_prob

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        """Alias for stats."""
        return self.stats
