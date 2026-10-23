"""Target registry and unified instantiation factory for benchmark regimes."""

from collections.abc import Callable
from typing import Any

from mcmc_robust.core.target import TargetDistribution
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

TARGET_REGISTRY: dict[str, Callable[..., TargetDistribution]] = {
    # Benign Gaussian regime
    "gaussian_isotropic": lambda d, **kwargs: IsotropicGaussian(d=d, **kwargs),
    "gaussian_anisotropic": lambda d, **kwargs: DiagonalGaussian(d=d, **kwargs),
    "gaussian_correlated": lambda d, **kwargs: FullCovarianceGaussian(d=d, **kwargs),
    "gaussian": lambda d, **kwargs: GaussianTarget(d=d, **kwargs),
    "anisotropic_gaussian": lambda d, **kwargs: AnisotropicGaussianTarget(d=d, **kwargs),
    # Roughness regime
    "quartic": lambda d, **kwargs: QuarticTarget(d=d, **kwargs),
    "superlinear": lambda d, **kwargs: SuperlinearTarget(d=d, **kwargs),
    "superlinear_p6": lambda d, **kwargs: SuperlinearPolynomialTarget(d=d, p=6.0, **kwargs),
    "l0_l1_smooth": lambda d, **kwargs: L0L1SmoothTarget(d=d, **kwargs),
    "l0l1_smooth": lambda d, **kwargs: L0L1SmoothTarget(d=d, **kwargs),
    "funnel": lambda d, **kwargs: NealsFunnelTarget(d=d, **kwargs),
    "neals_funnel": lambda d, **kwargs: NealsFunnelTarget(d=d, **kwargs),
    "rippled_gaussian": lambda d, **kwargs: RippledGaussianTarget(d=d, **kwargs),
    "oscillating": lambda d, **kwargs: OscillatingGaussianTarget(d=d, **kwargs),
    "oscillating_gaussian": lambda d, **kwargs: OscillatingGaussianTarget(d=d, **kwargs),
    # Heavy-tailed regime
    "student_t": lambda d, **kwargs: StudentTTarget(d=d, **kwargs),
    "student_t_1.5": lambda d, **kwargs: MultivariateStudentTTarget(d=d, nu=1.5, **kwargs),
    "student_t_2.5": lambda d, **kwargs: MultivariateStudentTTarget(d=d, nu=2.5, **kwargs),
    "student_t_4.0": lambda d, **kwargs: MultivariateStudentTTarget(d=d, nu=4.0, **kwargs),
    "heterogeneous_heavy": lambda d, **kwargs: HeterogeneousHeavyTailedTarget(d=d, **kwargs),
    "heterogeneous_student_t": lambda d, **kwargs: HeterogeneousStudentTTarget(d=d, **kwargs),
    # Dual regime
    "dual_coupled": lambda d, **kwargs: CoupledDualPathologyTarget(d=d, **kwargs),
    "dual_rough_flat": lambda d, **kwargs: DualRoughFlatTarget(d=d, **kwargs),
    "dual_confinement": lambda d, **kwargs: ConfinementInducedRoughnessTarget(d=d, **kwargs),
    "confinement_rough": lambda d, **kwargs: ConfinementRoughTarget(d=d, **kwargs),
}


def get_target(name: str, dim: int = 2, **kwargs: Any) -> TargetDistribution:
    """Instantiates a standardized TargetDistribution by name and dimension.

    Args:
        name: Canonical target identifier string.
        dim: State dimension d (supports d in {2, 10, 50, 100} and arbitrary d >= 1).
        **kwargs: Overrides for regime-specific parameters.

    Returns:
        Configured TargetDistribution instance.

    Raises:
        ValueError: If name is unknown or dim is invalid.
    """
    if "d" in kwargs:
        dim = int(kwargs.pop("d"))
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"State dimension must be >= 1, got {dim}")

    canonical_name = name.strip().lower()
    if canonical_name not in TARGET_REGISTRY:
        available = sorted(list(TARGET_REGISTRY.keys()))
        raise ValueError(f"Unknown target '{name}'. Available canonical targets: {available}")

    factory = TARGET_REGISTRY[canonical_name]
    return factory(d=dim, **kwargs)


def list_targets() -> list[str]:
    """Returns sorted list of available target names in the registry."""
    return sorted(list(TARGET_REGISTRY.keys()))
