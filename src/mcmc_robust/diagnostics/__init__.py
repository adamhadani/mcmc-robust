"""Diagnostic evaluation tools, convergence diagnostics, metric calculators, and budget trackers."""

from mcmc_robust.diagnostics.convergence import (
    compute_folded_rhat,
    compute_split_rhat,
)
from mcmc_robust.diagnostics.ess import (
    compute_bulk_ess,
    compute_tail_ess,
)
from mcmc_robust.diagnostics.metrics import (
    compute_ks_test,
    compute_quantile_relative_error,
    compute_wasserstein1,
)
from mcmc_robust.diagnostics.tail_alignment import (
    TailAlignmentReport,
    TailAlignmentWarning,
    tail_alignment_diagnostic,
)
from mcmc_robust.diagnostics.tracker import (
    BudgetExceededError,
    BudgetExhaustedError,
    GradBudgetTracker,
)

__all__ = [
    # Budget tracking
    "GradBudgetTracker",
    "BudgetExceededError",
    "BudgetExhaustedError",
    # Effective Sample Size
    "compute_bulk_ess",
    "compute_tail_ess",
    # Convergence
    "compute_split_rhat",
    "compute_folded_rhat",
    # Statistical Metrics
    "compute_ks_test",
    "compute_wasserstein1",
    "compute_quantile_relative_error",
    # Heavy tails shared across coordinates
    "tail_alignment_diagnostic",
    "TailAlignmentReport",
    "TailAlignmentWarning",
]
