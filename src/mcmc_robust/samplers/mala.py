"""Metropolis-Adjusted Langevin Algorithm (MALA) sampler."""

from typing import Any

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class MALA(Sampler):
    r"""Metropolis-Adjusted Langevin Algorithm (MALA) sampler.

    Generates Langevin drift proposals:
        y = x + h (D \odot \nabla \log \pi(x)) + \sqrt{2h} (s \odot \xi),  \xi \sim \mathcal{N}(0, I_d)
    where s is the diagonal standard deviation scale vector and D = s^2.

    Transition density:
        \log q(y|x) = -\frac{1}{4h} \|y - x - h (D \odot \nabla \log \pi(x))\|_{D^{-1}}^2
                      - \frac{d}{2}\ln(4\pi h) - \frac{1}{2}\sum_{i=1}^d \ln D_i

    Exact skew-reversible Metropolis-Hastings acceptance ratio:
        \log \alpha = \log \pi(y) - \log \pi(x) + \log q(x|y) - \log q(y|x)

    ZERO-REDUNDANCY ORACLE PROTOCOL:
    - Strictly 1 gradient evaluation per step (at candidate y).
    - Caches \nabla \log \pi(y) in ChainState upon acceptance.
    - Returns prior ChainState with 0 additional evaluations upon rejection.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.574,
        adapt_step_size: bool = False,
        adaptation: DualAveragingAdaptation | None = None,
    ) -> None:
        r"""Initializes MALA sampler.

        Args:
            step_size: Langevin step size h > 0.
            preconditioner: Optional 1D array of positive diagonal scale factors s.
            target_accept: Target acceptance rate \alpha^\star (default: 0.574).
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
        """MALA requires first-order gradients."""
        return True

    def init_state(
        self,
        x0: np.ndarray,
        target: TargetDistribution,
    ) -> ChainState:
        """Initializes ChainState with cached log_prob and grad_log_prob."""
        x0_arr = target._validate_input(x0)
        lp, grad = target.log_prob_and_grad(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=grad)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes a single MALA transition step with zero-redundancy caching."""
        d = target.d
        x = state.x
        grad_x = state.grad_log_prob
        if grad_x is None:
            raise ValueError("MALA step requires cached grad_log_prob in state")

        h = self.step_size
        xi = rng.normal(size=d)

        if self._preconditioner is not None:
            s = self._preconditioner
            D = s**2
            drift_x = h * (D * grad_x)
            diff_x = np.sqrt(2.0 * h) * (s * xi)
            y = x + drift_x + diff_x
        else:
            drift_x = h * grad_x
            diff_x = np.sqrt(2.0 * h) * xi
            y = x + drift_x + diff_x

        # Defensive guard against numerical overflow in proposal
        if not np.all(np.isfinite(y)):
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=0.0,
                stats={"grad_evals": 0, "log_prob_evals": 0, "accept_prob": 0.0, "overflow": 1.0},
                proposal=y,
                step_size=h,
            )

        # Evaluate candidate exactly once: log_prob and grad_log_prob jointly
        lp_y, grad_y = target.log_prob_and_grad(y)

        # Check numerical validity
        if not np.isfinite(lp_y) or not np.all(np.isfinite(grad_y)):
            accept_prob = 0.0
            log_alpha = -np.inf
        else:
            # Compute log q(x|y) and log q(y|x)
            # Forward residual: r_fwd = y - x - drift_x
            # Reverse residual: r_rev = x - y - drift_y
            if self._preconditioner is not None:
                s = self._preconditioner
                D = s**2
                drift_y = h * (D * grad_y)
                r_fwd = y - x - drift_x
                r_rev = x - y - drift_y
                sq_fwd = float(np.sum((r_fwd**2) / D))
                sq_rev = float(np.sum((r_rev**2) / D))
            else:
                drift_y = h * grad_y
                r_fwd = y - x - drift_x
                r_rev = x - y - drift_y
                sq_fwd = float(np.dot(r_fwd, r_fwd))
                sq_rev = float(np.dot(r_rev, r_rev))

            log_q_rev_minus_fwd = -0.25 * (sq_rev - sq_fwd) / h
            log_alpha = (lp_y - state.log_prob) + log_q_rev_minus_fwd

            if not np.isfinite(log_alpha):
                accept_prob = 0.0
            elif log_alpha >= 0.0:
                accept_prob = 1.0
            else:
                accept_prob = float(np.exp(log_alpha))

        u = rng.uniform()
        if u < accept_prob:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=grad_y)
            accepted = True
        else:
            new_state = state
            accepted = False

        stats: dict[str, Any] = {
            "accept_prob": float(accept_prob),
            "grad_evals": 1,
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
