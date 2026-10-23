"""Target distributions across five benchmark regimes."""

from mcmc_robust.targets.benign import (
    AnisotropicGaussianTarget,
    DiagonalGaussian,
    FullCovarianceGaussian,
    GaussianTarget,
    IsotropicGaussian,
)
from mcmc_robust.targets.dual import (
    ConfinementInducedRoughnessTarget,
    ConfinementRoughTarget,
    CoupledDualPathologyTarget,
    DualRoughFlatTarget,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    HeterogeneousStudentTTarget,
    MultivariateStudentTTarget,
    StudentTTarget,
)
from mcmc_robust.targets.roughness import (
    L0L1SmoothTarget,
    NealsFunnelTarget,
    OscillatingGaussianTarget,
    QuarticTarget,
    RippledGaussianTarget,
    SuperlinearPolynomialTarget,
    SuperlinearTarget,
)
from mcmc_robust.targets.suite import (
    TARGET_REGISTRY,
    get_target,
    list_targets,
)

__all__ = [
    # Benign
    "IsotropicGaussian",
    "DiagonalGaussian",
    "FullCovarianceGaussian",
    "GaussianTarget",
    "AnisotropicGaussianTarget",
    # Roughness
    "QuarticTarget",
    "SuperlinearPolynomialTarget",
    "SuperlinearTarget",
    "L0L1SmoothTarget",
    "NealsFunnelTarget",
    "RippledGaussianTarget",
    "OscillatingGaussianTarget",
    # Heavy-tailed
    "MultivariateStudentTTarget",
    "StudentTTarget",
    "HeterogeneousHeavyTailedTarget",
    "HeterogeneousStudentTTarget",
    # Dual
    "CoupledDualPathologyTarget",
    "DualRoughFlatTarget",
    "ConfinementInducedRoughnessTarget",
    "ConfinementRoughTarget",
    # Suite
    "get_target",
    "list_targets",
    "TARGET_REGISTRY",
]
