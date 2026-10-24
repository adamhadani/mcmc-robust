"""MCMC Robustness Research & Benchmarking Suite.

Novel gradient-based MCMC samplers resolving roughness and flatness pathologies
(UnifiedRobustNUTS, UnifiedRobustHMC, UnifiedRobustSampler, GentleMALA).
"""

from mcmc_robust.integrations import (
    BlackJAXAlgorithm,
    JaxArcsinhTransform,
    PyMCModelTarget,
    as_blackjax_algorithm,
    sample_pymc_model,
)
from mcmc_robust.samplers.automala import AutoMALA
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.nuts import ArcsinhTransformedNUTS, UnifiedRobustNUTS
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.samplers.novel.unified_hmc import UnifiedRobustHMC
from mcmc_robust.targets.roughness import NealsFunnelTarget
from mcmc_robust.targets.suite import get_target
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform
from mcmc_robust.transforms.hierarchical_arcsinh import HierarchicalArcsinhTransform

__version__ = "0.1.0"

__all__ = [
    "ArcsinhTransformedNUTS",
    "AutoMALA",
    "BlackJAXAlgorithm",
    "CoordinateWiseArcsinhTransform",
    "GentleMALA",
    "HierarchicalArcsinhTransform",
    "JaxArcsinhTransform",
    "NealsFunnelTarget",
    "PyMCModelTarget",
    "UnifiedRobustHMC",
    "UnifiedRobustNUTS",
    "UnifiedRobustSampler",
    "as_blackjax_algorithm",
    "get_target",
    "sample_pymc_model",
]
