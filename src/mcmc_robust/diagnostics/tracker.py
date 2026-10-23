"""Evaluation budget tracking and oracle instrumentation wrapper."""

from typing import Any

import numpy as np

from mcmc_robust.core.target import TargetDistribution


class BudgetExceededError(RuntimeError):
    """Exception raised when an evaluation budget limit is exceeded."""


# Alias for compatibility
BudgetExhaustedError = BudgetExceededError


class GradBudgetTracker(TargetDistribution):
    """TargetDistribution wrapper that strictly tracks and bounds evaluation budgets.

    Tracks every call to log_prob, grad_log_prob, and log_prob_and_grad.
    Enforces optional hard budget ceilings to terminate benchmarks cleanly.
    """

    def __init__(
        self,
        target: TargetDistribution,
        max_grad_evals: int | None = None,
        max_log_prob_evals: int | None = None,
        max_total_evals: int | None = None,
    ) -> None:
        """Wraps a TargetDistribution with budget tracking.

        Args:
            target: Underlying TargetDistribution instance to wrap.
            max_grad_evals: Optional ceiling on gradient evaluations.
            max_log_prob_evals: Optional ceiling on log-prob evaluations.
            max_total_evals: Optional ceiling on total (log_prob + grad) evaluations.
        """
        super().__init__(d=target.d)
        self._target: TargetDistribution = target
        self.target: TargetDistribution = target
        self.max_grad_evals: int | None = max_grad_evals
        self.max_log_prob_evals: int | None = max_log_prob_evals
        self.max_total_evals: int | None = max_total_evals

        self.reset()

    def reset(self) -> None:
        """Resets all evaluation counters to zero."""
        self.n_log_prob: int = 0
        self.n_grad: int = 0
        self.n_joint: int = 0

    @property
    def grad_evals(self) -> int:
        """Alias for n_grad."""
        return self.n_grad

    @property
    def log_prob_evals(self) -> int:
        """Alias for n_log_prob."""
        return self.n_log_prob

    @property
    def total_evals(self) -> int:
        """Total evaluations performed (log_prob + grad)."""
        return self.n_log_prob + self.n_grad

    @property
    def is_exhausted(self) -> bool:
        """Checks if any configured budget ceiling has been reached."""
        if self.max_grad_evals is not None and self.n_grad >= self.max_grad_evals:
            return True
        if self.max_log_prob_evals is not None and self.n_log_prob >= self.max_log_prob_evals:
            return True
        if self.max_total_evals is not None and self.total_evals >= self.max_total_evals:
            return True
        return False

    def _check_budget(self, log_prob_inc: int = 0, grad_inc: int = 0) -> None:
        """Validates that proposed evaluations do not exceed configured budgets."""
        if self.max_grad_evals is not None:
            if self.n_grad + grad_inc > self.max_grad_evals:
                raise BudgetExceededError(f"Gradient budget exceeded: {self.n_grad + grad_inc} > {self.max_grad_evals}")
        if self.max_log_prob_evals is not None:
            if self.n_log_prob + log_prob_inc > self.max_log_prob_evals:
                raise BudgetExceededError(
                    f"Log-prob budget exceeded: {self.n_log_prob + log_prob_inc} > {self.max_log_prob_evals}"
                )
        if self.max_total_evals is not None:
            new_total = self.total_evals + log_prob_inc + grad_inc
            if new_total > self.max_total_evals:
                raise BudgetExceededError(f"Total evaluation budget exceeded: {new_total} > {self.max_total_evals}")

    def log_prob(self, x: np.ndarray) -> float:
        """Evaluates target log_prob and increments counter."""
        self._check_budget(log_prob_inc=1)
        lp = self._target.log_prob(x)
        self.n_log_prob += 1
        return lp

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Evaluates target grad_log_prob and increments counter."""
        self._check_budget(grad_inc=1)
        grad = self._target.grad_log_prob(x)
        self.n_grad += 1
        return grad

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluates target log_prob_and_grad and increments joint counters."""
        self._check_budget(log_prob_inc=1, grad_inc=1)
        lp, grad = self._target.log_prob_and_grad(x)
        self.n_log_prob += 1
        self.n_grad += 1
        self.n_joint += 1
        return lp, grad

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Delegates ground truth sampling without consuming evaluation budget."""
        return self._target.sample_ground_truth(n=n, rng=rng)

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        """Delegates analytical moments without consuming evaluation budget."""
        return self._target.exact_moments_or_quantiles()

    def summary(self) -> dict[str, Any]:
        """Returns diagnostic dictionary of budget consumption metrics."""
        return {
            "n_log_prob": self.n_log_prob,
            "n_grad": self.n_grad,
            "n_joint": self.n_joint,
            "total_evals": self.total_evals,
            "max_grad_evals": self.max_grad_evals,
            "grad_budget_remaining": ((self.max_grad_evals - self.n_grad) if self.max_grad_evals is not None else None),
        }
