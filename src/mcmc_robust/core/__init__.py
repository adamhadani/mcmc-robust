"""Core architectural abstractions, state containers, and adaptation algorithms."""

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution

__all__ = [
    "ChainState",
    "DualAveragingAdaptation",
    "Sampler",
    "StepResult",
    "TargetDistribution",
    "WelfordPreconditioner",
]
