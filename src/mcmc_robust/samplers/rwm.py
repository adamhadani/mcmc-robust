"""Random Walk Metropolis (RWM) zeroth-order MCMC sampler."""

from typing import Any

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class RandomWalkMetropolis(Sampler):
    r"""Random Walk Metropolis (RWM) zeroth-order MCMC sampler.

    Generates Gaussian proposals:
        y = x + \sqrt{2h} (s \odot \xi),  \xi \sim \mathcal{N}(0, I_d)
    where h is the step size and s is an optional diagonal scale vector.

    Metropolis acceptance probability:
        \alpha = \min(1, \exp(\log \pi(y) - \log \pi(x)))
    Consumes strictly 0 gradient evaluations and 1 log_prob evaluation per step.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.234,
        adapt_step_size: bool = False,
        adaptation: DualAveragingAdaptation | None = None,
    ) -> None:
        r"""Initializes RWM sampler.

        Args:
            step_size: Base step size h > 0 (proposal variance is 2h).
            preconditioner: Optional 1D array of positive diagonal scale factors.
            target_accept: Target acceptance rate \alpha^\star for adaptation (default: 0.234).
            adapt_step_size: Whether to adapt step size during warmup.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        self.target_accept: float = float(target_accept)
        self.adapt_step_size: bool = bool(adapt_step_size)

        if adaptation is not None:
            self.adaptation: DualAveragingAdaptation | None = adaptation
        elif self.adapt_step_size:
            self.adaptation = DualAveragingAdaptation(
                initial_step_size=self.step_size,
                target_accept=self.target_accept,
            )
        else:
            self.adaptation = None

    @property
    def requires_grad(self) -> bool:
        """RWM is zeroth-order and evaluates no gradients."""
        return False

    def init_state(
        self,
        x0: np.ndarray,
        target: TargetDistribution,
    ) -> ChainState:
        """Initializes ChainState with strictly 0 gradient evaluations."""
        x0_arr = target._validate_input(x0)
        lp = target.log_prob(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=None)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes a single RWM transition step."""
        d = target.d
        xi = rng.normal(size=d)

        # Scale noise if preconditioner scale vector is provided
        if self._preconditioner is not None:
            noise = np.sqrt(2.0 * self.step_size) * (self._preconditioner * xi)
        else:
            noise = np.sqrt(2.0 * self.step_size) * xi

        y = state.x + noise

        # Evaluate candidate log-density (strictly 0 gradient evaluations)
        lp_y = target.log_prob(y)

        # Metropolis filter
        if not np.isfinite(lp_y):
            accept_prob = 0.0
        else:
            log_alpha = lp_y - state.log_prob
            if log_alpha >= 0.0:
                accept_prob = 1.0
            else:
                accept_prob = float(np.exp(log_alpha))

        u = rng.uniform()
        if u < accept_prob:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=None)
            accepted = True
        else:
            new_state = state
            accepted = False

        stats: dict[str, Any] = {
            "accept_prob": float(accept_prob),
            "grad_evals": 0,
            "log_prob_evals": 1,
            "accepted": float(accepted),
        }

        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=float(accept_prob),
            stats=stats,
            proposal=y,
            step_size=self.step_size,
        )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Warmup adaptation hook tuning step size via Dual Averaging."""
        if self.adaptation is None:
            self.adaptation = DualAveragingAdaptation(
                initial_step_size=self.step_size,
                target_accept=self.target_accept,
            )
        self.step_size = self.adaptation.update(result.accept_prob)

    def finalize_adaptation(self) -> float:
        """Freezes and sets the smoothed step size."""
        if self.adaptation is not None:
            self.step_size = self.adaptation.finalize()
        return self.step_size
