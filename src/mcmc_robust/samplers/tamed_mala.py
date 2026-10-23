"""Globally Tamed Metropolis-Adjusted Langevin Algorithm (Feature F12)."""

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class GloballyTamedMALA(Sampler):
    r"""Globally Tamed Metropolis-Adjusted Langevin Algorithm.

    Replaces standard Langevin drift with a globally or coordinate-wise bounded drift:
        Global:      b(x) = \nabla \log \pi(x) / (1 + \gamma \|\nabla \log \pi(x)\|)
        Coordinate:  b_i(x) = \partial_i \log \pi(x) / (1 + \gamma |\partial_i \log \pi(x)|)

    Preserves exact stationary distribution \pi(x) by evaluating reverse tamed drift
    b(y) at the proposed point and adjusting the Metropolis-Hastings acceptance ratio.
    Adheres strictly to the 1-gradient evaluation contract per step.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        gamma: float = 0.1,
        mode: str = "global",
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.574,
        adapt_step_size: bool = False,
        adaptation: DualAveragingAdaptation | None = None,
    ) -> None:
        r"""Initializes GloballyTamedMALA sampler.

        Args:
            step_size: Base step size h > 0.
            gamma: Taming parameter \gamma > 0.
            mode: Taming mode, either 'global' or 'coordinate'.
            preconditioner: Optional 1D array of positive diagonal scale factors.
            target_accept: Optimal target acceptance rate for dual averaging (default: 0.574).
            adapt_step_size: Whether to adapt step size during warmup.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if mode not in ("global", "coordinate"):
            raise ValueError(f"mode must be 'global' or 'coordinate', got {mode}")

        self.gamma: float = float(gamma)
        self.mode: str = mode
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
        """GloballyTamedMALA requires first-order gradients."""
        return True

    def _tamed_drift(self, grad: np.ndarray) -> np.ndarray:
        """Computes tamed drift vector b(x) from raw gradient."""
        if self.mode == "global":
            norm = float(np.linalg.norm(grad))
            return grad / (1.0 + self.gamma * norm)
        else:
            return grad / (1.0 + self.gamma * np.abs(grad))

    def init_state(
        self,
        x0: np.ndarray,
        target: TargetDistribution,
    ) -> ChainState:
        """Initializes ChainState with strictly 1 target evaluation."""
        x0_arr = target._validate_input(x0)
        lp, grad = target.log_prob_and_grad(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=grad)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes one transition step adhering to the 1-gradient caching contract."""
        d = target.d
        h = self.step_size
        x = state.x
        g_x = state.grad_log_prob
        if g_x is None:
            raise RuntimeError("GloballyTamedMALA requires cached grad_log_prob in ChainState")

        # Diagonal preconditioner: scale vector s and variance Sigma = s^2
        s = self._preconditioner if self._preconditioner is not None else np.ones(d, dtype=np.float64)
        var = s**2

        # 1. Compute forward tamed drift using cached gradient (0 target evals)
        b_x = self._tamed_drift(g_x)

        # 2. Propose candidate y = x + h Sigma b(x) + sqrt(2h) s * xi
        xi = rng.normal(size=d)
        y = x + h * (var * b_x) + np.sqrt(2.0 * h) * (s * xi)

        # Defensive guard against numerical overflow in proposal coordinates
        if not np.all(np.isfinite(y)):
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=0.0,
                stats={
                    "grad_evals": 0,
                    "log_prob_evals": 0,
                    "accept_prob": 0.0,
                    "overflow": 1.0,
                    "accepted": 0.0,
                },
                proposal=y,
                step_size=h,
            )

        # 3. Evaluate target at candidate y (strictly 1 gradient evaluation)
        lp_y, g_y = target.log_prob_and_grad(y)

        # Defensive guard against non-finite evaluations in candidate target
        if not np.isfinite(lp_y) or not np.all(np.isfinite(g_y)):
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=0.0,
                stats={
                    "grad_evals": 1,
                    "log_prob_evals": 1,
                    "accept_prob": 0.0,
                    "non_finite": 1.0,
                    "accepted": 0.0,
                },
                proposal=y,
                step_size=h,
            )

        # 4. Evaluate reverse tamed drift b(y)
        b_y = self._tamed_drift(g_y)

        # 5. Compute exact log proposal ratio:
        # log q(x|y) - log q(y|x) = -0.5 * delta^T (b_x + b_y) - (h/4) * (b_y^T Sigma b_y - b_x^T Sigma b_x)
        delta = y - x
        quad_y = float(np.sum(var * (b_y**2)))
        quad_x = float(np.sum(var * (b_x**2)))
        log_q_diff = -0.5 * float(np.dot(delta, b_x + b_y)) - (h / 4.0) * (quad_y - quad_x)

        # 6. Metropolis-Hastings acceptance ratio
        log_alpha = (lp_y - state.log_prob) + log_q_diff
        if not np.isfinite(log_alpha):
            accept_prob = 0.0
        elif log_alpha >= 0.0:
            accept_prob = 1.0
        else:
            accept_prob = float(np.exp(log_alpha))

        # 7. Accept / Reject decision
        accepted = bool(rng.uniform() < accept_prob)
        if accepted:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=g_y)
        else:
            new_state = state

        stats = {
            "accept_prob": float(accept_prob),
            "grad_evals": 1,
            "log_prob_evals": 1,
            "tamed_drift_norm": float(np.linalg.norm(b_x)),
            "raw_grad_norm": float(np.linalg.norm(g_x)),
            "accepted": float(accepted),
        }
        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=float(accept_prob),
            stats=stats,
            proposal=y,
            step_size=h,
        )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Updates step size via dual averaging during warmup."""
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
