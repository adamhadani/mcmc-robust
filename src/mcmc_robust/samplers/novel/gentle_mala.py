"""Gentle Metropolis-Adjusted Langevin Algorithm (Feature F15)."""

from typing import Any

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class GentleMALA(Sampler):
    r"""Gentle Metropolis-Adjusted Langevin Algorithm (Feature F15).

    Experimental. Damps the Langevin drift on stiff (super-linear) targets, motivated by §3.3 of
    Power & Vasdekis 2025 (arXiv:2511.21563); REPORT.md §3.1 states what is and is not established
    (the drift stays bounded, and the coordinate mode has no non-overshoot guarantee). Local
    stiffness adaptation:
        \kappa_h(x) = \gamma h \|s \odot \nabla \log \pi(x)\|_2   (global mode)
        \kappa_{h, i}(x) = \gamma h s_i |\nabla_i \log \pi(x)|     (coordinate mode)

    Gentle attenuation:
        \rho(\kappa) = (1 + \kappa)^{-1/2}

    Gentle drift field:
        b_h(x) = \rho(\kappa_h(x)) \nabla \log \pi(x)

    Proposal:
        y = x + h (s^2 \odot b_h(x)) + \sqrt{2h} (s \odot \xi),  \xi \sim \mathcal{N}(0, I_d)

    Exact transition log-density difference (O(d) complexity):
        \log q(x|y) - \log q(y|x) = -0.5 * \delta^\top (b_h(x) + b_h(y))
                                    - (h / 4) * (b_h(y)^\top D b_h(y) - b_h(x)^\top D b_h(x))
    where \delta = y - x and D = \operatorname{diag}(s^2).

    ZERO-REDUNDANCY ORACLE PROTOCOL:
    - Strictly 1 gradient evaluation per step (at candidate y).
    - Caches \nabla \log \pi(y) in ChainState upon acceptance.
    - Returns prior ChainState with 0 additional evaluations upon rejection.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        gamma: float = 1.0,
        mode: str = "global",
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.574,
        adapt_step_size: bool = False,
        adaptation: DualAveragingAdaptation | None = None,
        kappa_max: float = 25.0,
    ) -> None:
        r"""Initializes GentleMALA sampler.

        Args:
            step_size: Base step size h > 0.
            gamma: Dimensionless stiffness sensitivity parameter \gamma > 0.
            mode: Taming mode, either 'global' or 'coordinate'.
            preconditioner: Optional 1D array of positive diagonal scale factors s.
            target_accept: Optimal target acceptance rate for dual averaging (default: 0.574).
            adapt_step_size: Whether to adapt step size during warmup.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
            kappa_max: Soft asymptotic drift cap \kappa_{\max} > 0 preventing origin overshoot
                outside the square-root basin (pass float('inf') for pure square-root damping).
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if mode not in ("global", "coordinate"):
            raise ValueError(f"mode must be 'global' or 'coordinate', got {mode}")
        if kappa_max <= 0.0:
            raise ValueError(f"kappa_max must be positive, got {kappa_max}")

        self.gamma: float = float(gamma)
        self.mode: str = mode
        self.target_accept: float = float(target_accept)
        self.adapt_step_size: bool = bool(adapt_step_size)
        self.kappa_max: float = float(kappa_max)

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
        """GentleMALA requires first-order gradients."""
        return True

    def _stiffness(self, grad: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Computes dimensionless local stiffness indicator kappa_h(x)."""
        h = self.step_size
        if self.mode == "global":
            norm = float(np.linalg.norm(s * grad))
            return np.full_like(grad, self.gamma * h * norm)
        else:
            return self.gamma * h * (s * np.abs(grad))

    def _gentle_drift(self, grad: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Computes gently attenuated drift b_h(x) with soft asymptotic cap."""
        stiff = self._stiffness(grad, s)
        if np.isinf(self.kappa_max):
            return grad / np.sqrt(1.0 + stiff)
        return grad / np.sqrt(1.0 + stiff + (stiff / self.kappa_max) ** 2)

    def _tamed_drift(self, grad: np.ndarray) -> np.ndarray:
        """Helper computing gentle drift with current preconditioner."""
        s = self._preconditioner if self._preconditioner is not None else np.ones(len(grad), dtype=np.float64)
        return self._gentle_drift(grad, s)

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
            raise RuntimeError("GentleMALA requires cached grad_log_prob in ChainState")

        s = self._preconditioner if self._preconditioner is not None else np.ones(d, dtype=np.float64)
        var = s**2

        with np.errstate(over="ignore", invalid="ignore"):
            # 1. Forward gentle drift using cached gradient (0 target evaluations)
            b_x = self._gentle_drift(g_x, s)

            # 2. Propose candidate y = x + h var b(x) + sqrt(2h) s * xi
            xi = rng.normal(size=d)
            drift_x = h * (var * b_x)
            diff_x = np.sqrt(2.0 * h) * (s * xi)
            y = x + drift_x + diff_x

            # Defensive guard against proposal coordinate overflow
            if not np.all(np.isfinite(y)) or np.any(np.abs(y) > 1e70):
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

            # 3. Target evaluation at candidate y (strictly 1 gradient evaluation)
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

            # 4. Reverse gentle drift b_h(y)
            b_y = self._gentle_drift(g_y, s)

            # 5. Exact log proposal ratio in O(d)
            delta = y - x
            quad_y = float(np.sum(var * (b_y**2)))
            quad_x = float(np.sum(var * (b_x**2)))
            log_q_diff = -0.5 * float(np.dot(delta, b_x + b_y)) - (h / 4.0) * (quad_y - quad_x)

            # 6. Metropolis-Hastings acceptance ratio
            log_alpha = (lp_y - state.log_prob) + log_q_diff
            if not np.isfinite(log_alpha) or log_alpha < -709.0:
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

        stats: dict[str, Any] = {
            "accept_prob": float(accept_prob),
            "grad_evals": 1,
            "log_prob_evals": 1,
            "gentle_drift_norm": float(np.linalg.norm(b_x)),
            "raw_grad_norm": float(np.linalg.norm(g_x)),
            "stiffness_max": float(np.max(self._stiffness(g_x, s))),
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
