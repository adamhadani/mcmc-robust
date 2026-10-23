"""Locally Adaptive Metropolis-Adjusted Langevin Algorithm (autoMALA, Biron-Lattes et al. 2024)."""

from typing import Any

import numpy as np

from mcmc_robust.core.adaptation import WelfordPreconditioner
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution


class AutoMALA(Sampler):
    r"""Locally Adaptive Metropolis-Adjusted Langevin Algorithm (autoMALA).

    Reference implementation of Biron-Lattes, Surjanovic, Syed, Campbell & Bouchard-Côté
    (2024, *AISTATS*, PMLR 238:460–468, arXiv:2310.14975).

    At each MCMC transition from state $x \in \mathbb{R}^d$:
    1. Draws auxiliary momentum $p \sim \mathcal{N}(0, M)$ with diagonal mass matrix
       $M = \text{diag}(1 / s^2)$, and randomized symmetric energy-ratio thresholds
       $a \sim \text{Uniform}(0, 1/2)$, $b = a + 1/2 \in (1/2, 1)$.
    2. **Forward Doubling/Halving Step-Size Selector (Algorithm 2)**:
       Evaluates single-step leapfrog proposals $(x', p') = \Phi_\epsilon(x, p)$ with
       symmetric energy ratio $r_{\text{sym}}(x, p, \epsilon) = \exp(-|\Delta H_\epsilon(x, p)|)$:
       - If $a < r_{\text{sym}}(x, p, \epsilon_0) < b$, selects $\epsilon_{\text{fwd}} = \epsilon_0$.
       - If $r_{\text{sym}}(x, p, \epsilon_0) \le a$ (step size too aggressive for local stiffness),
         halves $\epsilon_j = 2^{-j}\epsilon_0$ until $r_{\text{sym}} > a$.
       - If $r_{\text{sym}}(x, p, \epsilon_0) \ge b$ (step size too conservative for local flatness),
         doubles $\epsilon_j = 2^j\epsilon_0$ until $r_{\text{sym}} < b$, then halves once
         ($\epsilon_{\text{fwd}} = \epsilon_{j-1}$).
    3. **Backward Involutive Reversibility Check (Algorithm 1)**:
       To preserve exact detailed balance under state-dependent step sizes, runs the
       identical selector from the proposed state with flipped momentum $(x^*, -p^*)$
       starting from $\epsilon_0$ to obtain $\epsilon_{\text{rev}}$.
       - When $\epsilon_{\text{fwd}} = \epsilon_0$, symplectic reversibility
         $\Phi_{\epsilon_0}(x^*, -p^*) = (x, -p)$ guarantees $|\Delta H_{\epsilon_0}(x^*, -p^*)|
         = |\Delta H_{\epsilon_0}(x, p)|$, so $\epsilon_{\text{rev}} = \epsilon_0$ is verified
         analytically with 0 extra gradient evaluations.
       - When $\epsilon_{\text{fwd}} \neq \epsilon_0$, evaluates the backward selector ladder from
         $(x^*, -p^*)$ (reusing the known involution point at $\epsilon = \epsilon_{\text{fwd}}$)
         and rejects immediately if $\epsilon_{\text{rev}} \neq \epsilon_{\text{fwd}}$.
    4. **Metropolis-Hastings Acceptance Filter**:
       If $\epsilon_{\text{rev}} == \epsilon_{\text{fwd}}$, accepts $(x^*, g^*)$ with probability
       $\alpha = \min(1, \exp(-\Delta H_{\epsilon_{\text{fwd}}}(x, p)))$.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        max_halvings: int = 8,
        max_doublings: int = 8,
        preconditioner: np.ndarray | None = None,
        adapt_step_size: bool = True,
        adapt_preconditioner: bool = False,
    ) -> None:
        """Initializes AutoMALA sampler.

        Args:
            step_size: Initial base leapfrog step size epsilon_0 > 0.
            max_halvings: Maximum halving iterations in Algorithm 2 (default: 8).
            max_doublings: Maximum doubling iterations in Algorithm 2 (default: 8).
            preconditioner: Optional 1D array of positive diagonal scale factors s.
            adapt_step_size: Whether to update base step_size during warmup toward selected epsilon.
            adapt_preconditioner: Whether to adapt diagonal preconditioner during warmup.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if max_halvings < 1 or max_doublings < 1:
            raise ValueError("max_halvings and max_doublings must be >= 1")
        self.max_halvings: int = int(max_halvings)
        self.max_doublings: int = int(max_doublings)
        self.adapt_step_size: bool = bool(adapt_step_size)
        self.adapt_preconditioner: bool = bool(adapt_preconditioner)
        self._welford: WelfordPreconditioner | None = None
        self._log_eps_ema: float = float(np.log(self.step_size))

    @property
    def requires_grad(self) -> bool:
        """AutoMALA requires first-order gradients."""
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

    def _kinetic_energy(self, p: np.ndarray, s: np.ndarray) -> float:
        """Computes kinetic energy K(p) = 0.5 * ||s * p||^2."""
        with np.errstate(over="ignore", invalid="ignore"):
            sp = s * p
            return float(0.5 * np.dot(sp, sp))

    def _leapfrog_one_step(
        self,
        x: np.ndarray,
        lp_x: float,
        g_x: np.ndarray,
        p: np.ndarray,
        k_x: float,
        eps: float,
        s: np.ndarray,
        target: TargetDistribution,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float]:
        """Executes 1 leapfrog step of size eps and returns (x', lp', g', p', delta_H, r_sym)."""
        var = s**2
        with np.errstate(over="ignore", invalid="ignore"):
            p_half = p + 0.5 * eps * g_x
            x_new = x + eps * (var * p_half)

        if not np.all(np.isfinite(x_new)):
            nan_vec = np.full_like(x, np.nan)
            return nan_vec, -np.inf, nan_vec, nan_vec, float("inf"), 0.0

        lp_new, g_new = target.log_prob_and_grad(x_new)
        if not np.isfinite(lp_new) or not np.all(np.isfinite(g_new)):
            nan_vec = np.full_like(x, np.nan)
            return x_new, -np.inf, nan_vec, nan_vec, float("inf"), 0.0

        with np.errstate(over="ignore", invalid="ignore"):
            p_new = p_half + 0.5 * eps * g_new
            k_new = self._kinetic_energy(p_new, s)
            h_old = -lp_x + k_x
            h_new = -lp_new + k_new
            delta_h = float(h_new - h_old)

        if not np.isfinite(delta_h):
            return x_new, lp_new, g_new, p_new, float("inf"), 0.0

        abs_dh = abs(delta_h)
        r_sym = 0.0 if abs_dh > 700.0 else float(np.exp(-abs_dh))
        return x_new, float(lp_new), g_new, p_new, delta_h, r_sym

    def _select_step_size_forward(
        self,
        x: np.ndarray,
        lp_x: float,
        g_x: np.ndarray,
        p: np.ndarray,
        k_x: float,
        eps0: float,
        a: float,
        b: float,
        s: np.ndarray,
        target: TargetDistribution,
    ) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray, float, int]:
        """Runs Algorithm 2 forward from (x, p) returning (k_level, x*, lp*, g*, p*, delta_H, grad_evals)."""
        grad_evals = 0
        x_cur, lp_cur, g_cur, p_cur, dh_cur, r_cur = self._leapfrog_one_step(
            x, lp_x, g_x, p, k_x, eps0, s, target
        )
        grad_evals += 1

        if a < r_cur < b:
            return 0, x_cur, lp_cur, g_cur, p_cur, dh_cur, grad_evals

        if r_cur <= a:
            # Halving branch: k = -1, -2, ..., -max_halvings
            for j in range(1, self.max_halvings + 1):
                eps_j = eps0 * (0.5**j)
                x_cur, lp_cur, g_cur, p_cur, dh_cur, r_cur = self._leapfrog_one_step(
                    x, lp_x, g_x, p, k_x, eps_j, s, target
                )
                grad_evals += 1
                if r_cur > a:
                    return -j, x_cur, lp_cur, g_cur, p_cur, dh_cur, grad_evals
            return -self.max_halvings, x_cur, lp_cur, g_cur, p_cur, dh_cur, grad_evals

        # Doubling branch (r_cur >= b): k = +1, +2, ..., +max_doublings, then halve once
        prev_k = 0
        prev_tuple = (x_cur, lp_cur, g_cur, p_cur, dh_cur)
        for j in range(1, self.max_doublings + 1):
            eps_j = eps0 * (2.0**j)
            x_next, lp_next, g_next, p_next, dh_next, r_next = self._leapfrog_one_step(
                x, lp_x, g_x, p, k_x, eps_j, s, target
            )
            grad_evals += 1
            if r_next < b:
                return prev_k, *prev_tuple, grad_evals
            prev_k = j
            prev_tuple = (x_next, lp_next, g_next, p_next, dh_next)

        return prev_k, *prev_tuple, grad_evals

    def _select_step_size_reverse(
        self,
        x_star: np.ndarray,
        lp_star: float,
        g_star: np.ndarray,
        neg_p_star: np.ndarray,
        eps0: float,
        a: float,
        b: float,
        s: np.ndarray,
        target: TargetDistribution,
        k_fwd: int,
        r_sym_at_k_fwd: float,
    ) -> tuple[int, int]:
        """Runs Algorithm 2 backward from (x*, -p*) returning (k_rev, grad_evals).

        Uses exact symplectic reversibility at k == k_fwd to avoid redundant evaluation
        of the known involution point Phi_{eps_fwd}(x*, -p*) = (x, -p).
        """
        if k_fwd == 0:
            # At k=0, Phi_{eps0}(x*, -p*) = (x, -p) has r_sym = r_sym_at_k_fwd in (a, b)
            return 0, 0

        k_star = self._kinetic_energy(neg_p_star, s)
        grad_evals = 0

        def eval_level_rev(k_level: int) -> float:
            nonlocal grad_evals
            if k_level == k_fwd:
                return r_sym_at_k_fwd
            eps_lvl = eps0 * (2.0**k_level)
            _, _, _, _, _, r_lvl = self._leapfrog_one_step(
                x_star, lp_star, g_star, neg_p_star, k_star, eps_lvl, s, target
            )
            grad_evals += 1
            return r_lvl

        r0 = eval_level_rev(0)
        if a < r0 < b:
            return 0, grad_evals

        if r0 <= a:
            # Halving branch: if k_fwd > 0, reverse cannot match k_fwd
            if k_fwd > 0:
                return -1, grad_evals
            for j in range(1, self.max_halvings + 1):
                r_j = eval_level_rev(-j)
                if r_j > a:
                    return -j, grad_evals
                if -j < k_fwd:
                    # Passed k_fwd without stopping; cannot match k_fwd
                    return -j, grad_evals
            return -self.max_halvings, grad_evals

        # Doubling branch (r0 >= b): if k_fwd < 0, reverse cannot match k_fwd
        if k_fwd < 0:
            return 0, grad_evals
        prev_k = 0
        for j in range(1, self.max_doublings + 1):
            r_j = eval_level_rev(j)
            if r_j < b:
                return prev_k, grad_evals
            prev_k = j
            if prev_k > k_fwd:
                # Exceeded k_fwd without stopping; cannot match k_fwd
                return prev_k, grad_evals
        return prev_k, grad_evals

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes a single Locally Adaptive autoMALA transition step."""
        d = target.d
        x = state.x
        lp_x = state.log_prob
        g_x = state.grad_log_prob
        if g_x is None:
            raise ValueError("AutoMALA step requires cached grad_log_prob in ChainState")

        s = (
            self._preconditioner
            if self._preconditioner is not None
            else np.ones(d, dtype=np.float64)
        )
        # Sample momentum p ~ N(0, M) where M = diag(1 / s^2)
        p = rng.normal(size=d) / s
        k_x = self._kinetic_energy(p, s)

        # Draw randomized energy-ratio bounds a ~ Uniform(0, 0.5), b = a + 0.5
        a = float(rng.uniform(0.0, 0.5))
        b = a + 0.5
        eps0 = self.step_size

        # 1. Forward step-size selector
        k_fwd, x_star, lp_star, g_star, p_star, delta_h, fwd_grads = self._select_step_size_forward(
            x, lp_x, g_x, p, k_x, eps0, a, b, s, target
        )
        eps_fwd = eps0 * (2.0**k_fwd)

        if not np.isfinite(delta_h) or not np.isfinite(lp_star):
            stats: dict[str, Any] = {
                "accept_prob": 0.0,
                "grad_evals": fwd_grads,
                "log_prob_evals": fwd_grads,
                "accepted": 0.0,
                "selected_step_size": float(eps_fwd),
                "reversibility_passed": 0.0,
            }
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=0.0,
                stats=stats,
                proposal=x_star,
                step_size=eps_fwd,
            )

        r_sym_fwd = 0.0 if abs(delta_h) > 700.0 else float(np.exp(-abs(delta_h)))

        # 2. Backward involutive reversibility check from (x*, -p*)
        k_rev, rev_grads = self._select_step_size_reverse(
            x_star,
            lp_star,
            g_star,
            -p_star,
            eps0,
            a,
            b,
            s,
            target,
            k_fwd=k_fwd,
            r_sym_at_k_fwd=r_sym_fwd,
        )
        total_grads = fwd_grads + rev_grads
        reversibility_ok = k_rev == k_fwd

        if not reversibility_ok:
            accept_prob = 0.0
        elif delta_h <= 0.0:
            accept_prob = 1.0
        elif delta_h > 700.0:
            accept_prob = 0.0
        else:
            accept_prob = float(np.exp(-delta_h))

        accepted = bool(rng.uniform() < accept_prob)
        new_state = ChainState(x=x_star, log_prob=lp_star, grad_log_prob=g_star) if accepted else state

        stats = {
            "accept_prob": float(accept_prob),
            "grad_evals": int(total_grads),
            "log_prob_evals": int(total_grads),
            "accepted": float(accepted),
            "selected_step_size": float(eps_fwd),
            "k_fwd": int(k_fwd),
            "k_rev": int(k_rev),
            "reversibility_passed": float(reversibility_ok),
        }
        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=float(accept_prob),
            stats=stats,
            proposal=x_star,
            step_size=eps_fwd,
        )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Warmup adaptation hook updating base step size and optional diagonal preconditioner."""
        if self.adapt_step_size:
            sel_eps = float(result.stats.get("selected_step_size", self.step_size))
            if sel_eps > 0.0 and np.isfinite(sel_eps):
                weight = max(0.05, 1.0 / (step_idx + 10.0))
                self._log_eps_ema = (1.0 - weight) * self._log_eps_ema + weight * float(np.log(sel_eps))
                self.step_size = float(np.clip(np.exp(self._log_eps_ema), 1e-5, 10.0))

        if self.adapt_preconditioner:
            d = result.state.x.shape[0]
            if self._welford is None:
                self._welford = WelfordPreconditioner(d=d)
            self._welford.update(result.state.x)
            if step_idx >= 20 and step_idx % 10 == 0:
                self.set_preconditioner(self._welford.diagonal_scale())
