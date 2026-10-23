"""Barker-Langevin Proposal MCMC Sampler (Feature F13)."""

import numpy as np
from scipy import special

from mcmc_robust.core.adaptation import DualAveragingAdaptation
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class BarkerLangevin(Sampler):
    r"""Barker-Langevin Proposal MCMC Sampler (Livingstone & Zanella 2022).

    Constructs coordinate-wise locally balanced proposals:
        y_i = x_i + \sqrt{h} s_i b_i \sigma_i
    where \sigma_i \sim \mathcal{N}(0, 1) and b_i \in \{-1, +1\} with probability:
        p(b_i = +1) = \operatorname{sigmoid}(\sqrt{h} s_i \sigma_i \partial_i \log \pi(x))

    Guarantees that proposal displacement magnitude |y_i - x_i| is bounded by Gaussian
    noise \sqrt{h} s_i |\sigma_i|, eliminating explosive overshoots on stiff targets.
    Transition density ratio is computed with machine precision via softplus.
    Adheres strictly to the 1-gradient evaluation contract per step.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.574,
        adapt_step_size: bool = False,
        adaptation: DualAveragingAdaptation | None = None,
    ) -> None:
        """Initializes BarkerLangevin sampler.

        Args:
            step_size: Base step size h > 0.
            preconditioner: Optional 1D array of positive diagonal scale factors.
            target_accept: Target acceptance rate for dual averaging (default: 0.574).
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
        """BarkerLangevin requires first-order gradients."""
        return True

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
            raise RuntimeError("BarkerLangevin requires cached grad_log_prob in ChainState")

        # Diagonal scale preconditioner
        s = self._preconditioner if self._preconditioner is not None else np.ones(d, dtype=np.float64)

        # 1. Coordinate-wise noise and Bernoulli skewing (0 target evals)
        sigma = rng.normal(size=d)
        # Log-odds argument: a_i = sqrt(h) * s_i * sigma_i * g_i(x)
        arg = np.sqrt(h) * s * sigma * g_x
        prob_plus = special.expit(arg)

        # Sample b_i in {-1, +1}
        u = rng.uniform(size=d)
        b = np.where(u < prob_plus, 1.0, -1.0)
        z = b * sigma

        # 2. Proposed coordinate candidate: y = x + sqrt(h) * s * z
        y = x + np.sqrt(h) * s * z

        # Defensive guard against numerical overflow
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

        # 3. Evaluate target at proposed point (strictly 1 gradient evaluation)
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

        # 4. Numerically stable softplus log transition density difference:
        # log q(x|y) - log q(y|x) = sum_i [ softplus(-(y_i - x_i) g_i(x)) - softplus((y_i - x_i) g_i(y)) ]
        delta = y - x
        term_fwd = np.logaddexp(0.0, -delta * g_x)
        term_rev = np.logaddexp(0.0, delta * g_y)
        log_q_diff = float(np.sum(term_fwd - term_rev))

        # 5. Metropolis-Hastings acceptance ratio
        log_alpha = (lp_y - state.log_prob) + log_q_diff
        if not np.isfinite(log_alpha):
            accept_prob = 0.0
        elif log_alpha >= 0.0:
            accept_prob = 1.0
        else:
            accept_prob = float(np.exp(log_alpha))

        # 6. Accept / Reject decision
        accepted = bool(rng.uniform() < accept_prob)
        if accepted:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=g_y)
        else:
            new_state = state

        stats = {
            "accept_prob": float(accept_prob),
            "grad_evals": 1,
            "log_prob_evals": 1,
            "noise_norm": float(np.linalg.norm(sigma)),
            "delta_norm": float(np.linalg.norm(delta)),
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
