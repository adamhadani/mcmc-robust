"""Abstract base class for MCMC samplers with zero-redundancy oracle caching."""

from abc import ABC, abstractmethod

import numpy as np

from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class Sampler(ABC):
    """Abstract base class for all Markov Chain Monte Carlo samplers.

    Enforces the zero-redundancy oracle caching contract:
    - Target distribution evaluations are performed strictly through target methods.
    - Proposal evaluations are stored in ChainState upon acceptance.
    - Rejections return the existing ChainState with zero target evaluations.
    """

    def __init__(
        self,
        step_size: float = 1.0,
        preconditioner: np.ndarray | None = None,
    ) -> None:
        """Initializes sampler parameters.

        Args:
            step_size: Base step size h > 0.
            preconditioner: Optional 1D array of positive diagonal scale factors.
        """
        if step_size <= 0.0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size: float = float(step_size)
        self._preconditioner: np.ndarray | None = None
        if preconditioner is not None:
            self.set_preconditioner(preconditioner)

    @property
    def requires_grad(self) -> bool:
        """Boolean flag: True if sampler evaluates gradients, False for zeroth-order."""
        return True

    @property
    def preconditioner(self) -> np.ndarray | None:
        """Current diagonal preconditioner scale vector."""
        return self._preconditioner

    def set_preconditioner(self, preconditioner: np.ndarray) -> None:
        """Sets diagonal preconditioner scale vector.

        Args:
            preconditioner: 1D array of positive values.
        """
        p = np.asarray(preconditioner, dtype=np.float64)
        if p.ndim != 1 or np.any(p <= 0.0):
            raise ValueError("Preconditioner must be a 1D array of strictly positive values")
        self._preconditioner = p

    @abstractmethod
    def init_state(
        self,
        x0: np.ndarray,
        target: TargetDistribution,
    ) -> ChainState:
        """Initializes ChainState by evaluating the target at x0 exactly once.

        Args:
            x0: Initial coordinate vector, shape (d,).
            target: Target distribution instance.

        Returns:
            Initial immutable ChainState with cached log_prob and grad_log_prob.
        """
        x0_arr = target._validate_input(x0)
        if self.requires_grad:
            lp, grad = target.log_prob_and_grad(x0_arr)
            return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=grad)
        else:
            lp = target.log_prob(x0_arr)
            return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=None)

    @abstractmethod
    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes a single MCMC transition step with zero-redundancy caching.

        Implementations must:
        1. Use state.log_prob and state.grad_log_prob (do not re-evaluate at state.x).
        2. Propose candidate y and evaluate target at y exactly once.
        3. If accepted: return StepResult with new ChainState containing y and its evals.
        4. If rejected: return StepResult with existing state unchanged (0 evals).

        Args:
            state: Current immutable ChainState.
            target: Target distribution oracle.
            rng: NumPy random generator.

        Returns:
            StepResult containing resulting ChainState and step diagnostics.
        """

    @abstractmethod
    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Warmup adaptation hook called after each step.

        Args:
            step_idx: 0-indexed iteration counter.
            result: Result of the transition step just completed.
        """

    def finalize_adaptation(self) -> float:
        """Finalizes warmup adaptation and freezes the post-warmup step size."""
        return self.step_size

    def sample(
        self,
        x0: np.ndarray,
        target: TargetDistribution,
        n_steps: int,
        burn_in: int = 0,
        rng: np.random.Generator | None = None,
        adapt: bool = False,
    ) -> tuple[np.ndarray, list[StepResult]]:
        """Executes an MCMC chain for n_steps iterations.

        Args:
            x0: Initial position vector.
            target: Target distribution.
            n_steps: Total transition steps to execute.
            burn_in: Number of initial steps to discard from the returned samples.
            rng: Optional NumPy random generator (default: default_rng()).
            adapt: If True, calls self.adapt(...) during warmup (steps < burn_in).

        Returns:
            Tuple of:
            - Array of retained samples, shape (n_steps - burn_in, d).
            - List of StepResult objects for diagnostic telemetry.
        """
        if rng is None:
            rng = np.random.default_rng()
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if burn_in < 0 or burn_in >= n_steps:
            raise ValueError(f"burn_in must satisfy 0 <= burn_in < n_steps, got {burn_in}")

        state = self.init_state(x0, target)
        samples = []
        results = []

        for t in range(n_steps):
            res = self.step(state, target, rng)
            if adapt and t < burn_in:
                self.adapt(t, res)
                if t == burn_in - 1:
                    self.finalize_adaptation()
            state = res.state
            results.append(res)
            if t >= burn_in:
                samples.append(state.x.copy())

        return np.asarray(samples, dtype=np.float64), results
