"""Hamiltonian Monte Carlo (HMC) with symplectic leapfrog integrator."""

from typing import Any

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class HMC(Sampler):
    r"""Hamiltonian Monte Carlo (HMC) sampler with symplectic leapfrog integration.

    Integrates Hamilton's equations using L leapfrog steps of size \epsilon = h / L:
        p_{1/2} = p_0 + \frac{\epsilon}{2} \nabla \log \pi(x_0)
        x_1     = x_0 + \epsilon M^{-1} p_{1/2}
        p_1     = p_{1/2} + \frac{\epsilon}{2} \nabla \log \pi(x_1)

    Kinetic energy: K(p) = \frac{1}{2} p^T M^{-1} p
    Hamiltonian: H(x, p) = -\log \pi(x) + K(p)
    Acceptance probability with skew-reversible momentum flip R(x, p) = (x, -p):
        \Delta H = H(y, -p_L) - H(x, p_0)
        \alpha = \min(1, \exp(-\Delta H))

    ZERO-REDUNDANCY ORACLE PROTOCOL:
    - Exactly L gradient evaluations per step.
    - Initial leapfrog gradient g_0 uses cached state.grad_log_prob (0 extra evals).
    - Step L evaluates target.log_prob_and_grad(x_L) jointly.
    - Caches \nabla \log \pi(y) upon acceptance; returns prior state upon rejection.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.65,
        adapt_step_size: bool = False,
        adaptation: DualAveragingAdaptation | None = None,
    ) -> None:
        r"""Initializes HMC sampler.

        Args:
            step_size: Trajectory length h > 0 (leapfrog step-size is \epsilon = h / L).
            n_leapfrog: Number of leapfrog integration steps L >= 1.
            preconditioner: Optional 1D array of diagonal scale factors s (M^{-1} = diag(s^2)).
            target_accept: Target acceptance rate \alpha^\star (default: 0.65).
            adapt_step_size: Whether to adapt step size during warmup.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if n_leapfrog < 1:
            raise ValueError(f"n_leapfrog must be >= 1, got {n_leapfrog}")
        self.n_leapfrog: int = int(n_leapfrog)
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
        """HMC requires first-order gradients."""
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

    def _kinetic_energy(self, p: np.ndarray) -> float:
        """Computes kinetic energy K(p) = 0.5 * p^T M^{-1} p."""
        if self._preconditioner is not None:
            s = self._preconditioner
            return float(0.5 * np.sum((s * p) ** 2))
        return float(0.5 * np.dot(p, p))

    def _velocity(self, p: np.ndarray) -> np.ndarray:
        """Computes velocity v = M^{-1} p."""
        if self._preconditioner is not None:
            s = self._preconditioner
            return (s**2) * p
        return p

    def _sample_momentum(self, d: int, rng: np.random.Generator) -> np.ndarray:
        """Samples momentum p ~ N(0, M)."""
        xi = rng.normal(size=d)
        if self._preconditioner is not None:
            # M = diag(1 / s^2), so std is 1 / s
            s = self._preconditioner
            return xi / s
        return xi

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes a single HMC step with L leapfrog steps and exact MH filter."""
        d = target.d
        x = np.array(state.x, dtype=np.float64, copy=True)
        g = np.array(state.grad_log_prob, dtype=np.float64, copy=True)
        if state.grad_log_prob is None:
            raise ValueError("HMC step requires cached grad_log_prob in state")

        p0 = self._sample_momentum(d, rng)
        p = p0.copy()
        k0 = self._kinetic_energy(p0)
        h0 = -state.log_prob + k0

        L = self.n_leapfrog
        epsilon = self.step_size / L

        grad_evals = 0
        lp_evals = 0
        lp_y = -np.inf
        diverged = False

        # Symplectic leapfrog integration loop
        for ell in range(L):
            # 1. Half-step momentum kick
            p += 0.5 * epsilon * g

            # 2. Full-step position drift
            x += epsilon * self._velocity(p)

            # Check numerical sanity
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
                diverged = True
                break

            # 3. Evaluate target gradient (and log_prob on final step)
            if ell == L - 1:
                lp_y, g = target.log_prob_and_grad(x)
                grad_evals += 1
                lp_evals += 1
            else:
                g = target.grad_log_prob(x)
                grad_evals += 1

            if not np.all(np.isfinite(g)):
                diverged = True
                break

            # 4. Half-step momentum kick
            p += 0.5 * epsilon * g

        if diverged or not np.isfinite(lp_y) or not np.all(np.isfinite(g)):
            delta_H = float("inf")
            accept_prob = 0.0
        else:
            # Skew-reversible momentum flip R(y, p_L) = (y, -p_L)
            # Kinetic energy K(-p_L) = K(p_L)
            kL = self._kinetic_energy(p)
            hL = -lp_y + kL
            delta_H = hL - h0

            if not np.isfinite(delta_H):
                accept_prob = 0.0
            elif delta_H <= 0.0:
                accept_prob = 1.0
            else:
                accept_prob = float(np.exp(-delta_H))

        u = rng.uniform()
        if u < accept_prob:
            new_state = ChainState(x=x, log_prob=lp_y, grad_log_prob=g)
            accepted = True
        else:
            new_state = state
            accepted = False

        stats: dict[str, Any] = {
            "accept_prob": float(accept_prob),
            "grad_evals": grad_evals,
            "hamiltonian_error": float(delta_H),
            "energy_error": float(delta_H),
            "log_prob_evals": lp_evals,
            "n_leapfrog": L,
            "epsilon": epsilon,
            "accepted": float(accepted),
        }

        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=float(accept_prob),
            stats=stats,
            proposal=x,
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
