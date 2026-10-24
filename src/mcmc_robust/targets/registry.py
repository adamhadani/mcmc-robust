"""Unified target registry for the synthetic benchmark suite and the applied (PosteriorDB-inspired) models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.targets.applied import (
    HierarchicalItemResponse2PLTarget,
    RadonHierarchicalTarget,
    RobustStudentTRegressionTarget,
    SparseHorseshoeLogisticTarget,
    StochasticVolatilityLogReturnTarget,
)
from mcmc_robust.targets.suite import TARGET_REGISTRY, get_target

#: Applied target constructors keyed by name, including short aliases.
APPLIED_TARGET_REGISTRY: dict[str, Callable[..., TargetDistribution]] = {
    "radon_hierarchical": RadonHierarchicalTarget,
    "radon": RadonHierarchicalTarget,
    "sparse_horseshoe_logistic": SparseHorseshoeLogisticTarget,
    "horseshoe_logistic": SparseHorseshoeLogisticTarget,
    "robust_student_t_regression": RobustStudentTRegressionTarget,
    "student_t_regression": RobustStudentTRegressionTarget,
    "hierarchical_irt_2pl": HierarchicalItemResponse2PLTarget,
    "irt_2pl": HierarchicalItemResponse2PLTarget,
    "stochastic_volatility": StochasticVolatilityLogReturnTarget,
    "stochastic_volatility_log_return": StochasticVolatilityLogReturnTarget,
}

CANONICAL_APPLIED_TARGETS: tuple[str, ...] = (
    "radon_hierarchical",
    "sparse_horseshoe_logistic",
    "robust_student_t_regression",
    "hierarchical_irt_2pl",
    "stochastic_volatility",
)


class TargetRegistry:
    """Unified factory and registry for both synthetic and applied TargetDistribution models."""

    _custom_registry: ClassVar[dict[str, Callable[..., TargetDistribution]]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., TargetDistribution]) -> None:
        """Registers a custom target distribution constructor."""
        canonical = name.strip().lower()
        if not canonical:
            raise ValueError("Target name must be a non-empty string")
        cls._custom_registry[canonical] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TargetDistribution:
        """Instantiates a TargetDistribution by canonical identifier."""
        canonical = name.strip().lower()
        if canonical in cls._custom_registry:
            return cls._custom_registry[canonical](**kwargs)
        if canonical in APPLIED_TARGET_REGISTRY:
            return APPLIED_TARGET_REGISTRY[canonical](**kwargs)
        if canonical in TARGET_REGISTRY:
            dim = int(kwargs.pop("dim", kwargs.pop("d", 2)))
            return get_target(canonical, dim=dim, **kwargs)
        available = cls.list_targets()
        raise ValueError(f"Unknown target '{name}'. Available targets: {available}")

    @classmethod
    def list_targets(cls) -> list[str]:
        """Returns sorted list of all registered synthetic and applied target names."""
        names = set(TARGET_REGISTRY.keys()) | set(APPLIED_TARGET_REGISTRY.keys()) | set(cls._custom_registry.keys())
        return sorted(names)

    @classmethod
    def list_applied_targets(cls) -> list[str]:
        """Returns the default applied target names (one per model, no aliases)."""
        return list(CANONICAL_APPLIED_TARGETS)


__all__ = [
    "APPLIED_TARGET_REGISTRY",
    "CANONICAL_APPLIED_TARGETS",
    "TargetRegistry",
]
