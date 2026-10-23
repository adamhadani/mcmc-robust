"""Tier 5 Adversarial Tests: Non-finite Inputs, Boundary Robustness, and Adaptation Guards.

Verifies target non-finite handling, dimension contract enforcement, sampler resilience
to singular or non-finite oracle returns, and warmup adaptation edge cases.
"""

import numpy as np
import pytest

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.samplers.barker import BarkerLangevin
from mcmc_robust.samplers.hmc import HMC
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.novel.anisotropic_tail import AnisotropicTailMALA
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.samplers.rwm import RandomWalkMetropolis
from mcmc_robust.samplers.tamed_mala import GloballyTamedMALA
from mcmc_robust.targets.benign import (
    DiagonalGaussian,
    FullCovarianceGaussian,
    IsotropicGaussian,
)
from mcmc_robust.targets.dual import (
    ConfinementInducedRoughnessTarget,
    ConfinementRoughTarget,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)
from mcmc_robust.targets.roughness import (
    L0L1SmoothTarget,
    QuarticTarget,
    RippledGaussianTarget,
    SuperlinearTarget,
)

TARGET_CLASSES = [
    lambda d: IsotropicGaussian(d=d),
    lambda d: DiagonalGaussian(d=d),
    lambda d: FullCovarianceGaussian(d=d),
    lambda d: QuarticTarget(d=d),
    lambda d: SuperlinearTarget(d=d),
    lambda d: L0L1SmoothTarget(d=d),
    lambda d: RippledGaussianTarget(d=d),
    lambda d: MultivariateStudentTTarget(d=d),
    lambda d: HeterogeneousHeavyTailedTarget(d=d),
    lambda d: ConfinementInducedRoughnessTarget(d=d),
    lambda d: ConfinementRoughTarget(d=d),
]


class TestTargetNonFiniteAndShapeRobustness:
    """Verifies that all targets handle non-finite inputs and validate shapes cleanly."""

    @pytest.mark.parametrize("target_factory", TARGET_CLASSES)
    def test_target_non_finite_inputs(self, target_factory) -> None:
        d = 4
        target = target_factory(d)

        # 1. NaN input
        x_nan = np.array([0.0, np.nan, 1.0, -1.0])
        assert target.log_prob(x_nan) == -np.inf
        grad_nan = target.grad_log_prob(x_nan)
        assert np.all(np.isnan(grad_nan))
        lp_j, grad_j = target.log_prob_and_grad(x_nan)
        assert lp_j == -np.inf
        assert np.all(np.isnan(grad_j))

        # 2. Positive Inf input
        x_inf = np.array([0.0, np.inf, 1.0, -1.0])
        assert target.log_prob(x_inf) == -np.inf
        grad_inf = target.grad_log_prob(x_inf)
        assert np.all(np.isnan(grad_inf))
        lp_j, grad_j = target.log_prob_and_grad(x_inf)
        assert lp_j == -np.inf
        assert np.all(np.isnan(grad_j))

        # 3. Negative Inf input
        x_ninf = np.array([0.0, -np.inf, 1.0, -1.0])
        assert target.log_prob(x_ninf) == -np.inf
        grad_ninf = target.grad_log_prob(x_ninf)
        assert np.all(np.isnan(grad_ninf))
        lp_j, grad_j = target.log_prob_and_grad(x_ninf)
        assert lp_j == -np.inf
        assert np.all(np.isnan(grad_j))

    @pytest.mark.parametrize("target_factory", TARGET_CLASSES)
    def test_target_dimension_validation(self, target_factory) -> None:
        d = 4
        target = target_factory(d)

        with pytest.raises(ValueError, match="Expected 1D array"):
            target._validate_input(np.ones(d + 1))

        with pytest.raises(ValueError, match="Expected 1D array"):
            target._validate_input(np.ones(d - 1))

        with pytest.raises(ValueError, match="Expected 1D array"):
            target._validate_input(np.ones((d, 2)))


class MockSingularTarget(TargetDistribution):
    """Pathological target that returns non-finite values at specific regions."""

    def __init__(self, d: int = 2) -> None:
        super().__init__(d=d)

    def log_prob(self, x: np.ndarray) -> float:
        # Returns -inf whenever x[0] > 1.0
        return -np.inf if x[0] > 1.0 else float(-0.5 * np.dot(x, x))

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return np.full(self.d, np.nan) if x[0] > 1.0 else -x

    def sample_ground_truth(self, n: int, rng=None) -> np.ndarray:
        return np.zeros((n, self.d))

    def exact_moments_or_quantiles(self) -> dict:
        return {"mean": np.zeros(self.d)}


class TestSamplerSingularOracleResilience:
    """Verifies that samplers survive when the target returns non-finite values."""

    @pytest.mark.parametrize(
        "sampler",
        [
            RandomWalkMetropolis(step_size=0.5),
            MALA(step_size=0.5),
            HMC(step_size=0.5, n_leapfrog=3),
            GloballyTamedMALA(step_size=0.5),
            BarkerLangevin(step_size=0.5),
            GentleMALA(step_size=0.5),
            AnisotropicTailMALA(step_size=0.5, radii=1.0),
            UnifiedRobustSampler(step_size=0.5),
        ],
    )
    def test_rejection_of_singular_target_values(self, sampler) -> None:
        target = MockSingularTarget(d=2)
        rng = np.random.default_rng(42)
        x0 = np.array([0.5, 0.0])

        state = sampler.init_state(x0, target)
        assert np.isfinite(state.log_prob)

        # Force propose into singular region by taking steps
        for _ in range(10):
            res = sampler.step(state, target, rng)
            # If candidate was in singular region, it must be rejected
            if res.proposal is not None and res.proposal[0] > 1.0:
                assert not res.accepted
                assert res.accept_prob == 0.0
                assert np.array_equal(res.state.x, state.x)

            state = res.state
            assert np.all(np.isfinite(state.x))
            assert np.isfinite(state.log_prob)


class TestAdaptationEdgeCases:
    """Verifies edge cases in DualAveragingAdaptation and WelfordPreconditioner."""

    def test_dual_averaging_non_finite_accept_prob(self) -> None:
        da = DualAveragingAdaptation(initial_step_size=0.1)

        # Update with NaN -> treated as accept_prob=0.0
        h1 = da.update(np.nan)
        assert np.isfinite(h1)
        assert da.min_step_size <= h1 <= da.max_step_size

        # Update with Inf -> treated as accept_prob=1.0
        h2 = da.update(np.inf)
        assert np.isfinite(h2)
        assert da.min_step_size <= h2 <= da.max_step_size

        # Update with -Inf -> treated as accept_prob=0.0
        h3 = da.update(-np.inf)
        assert np.isfinite(h3)
        assert da.min_step_size <= h3 <= da.max_step_size

    def test_welford_constant_samples_clamping(self) -> None:
        d = 3
        wp = WelfordPreconditioner(d=d, min_variance=1e-3, max_variance=100.0)

        # Batch update with identical constant samples
        X_const = np.full((50, d), 42.0)
        wp.update_batch(X_const)

        # Sample variance is 0, regularized variance should clamp to min_variance
        var = wp.diagonal_preconditioner()
        assert np.all(var >= 1e-3)
        assert np.all(var <= 100.0)
        assert np.all(np.isfinite(var))

    def test_welford_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="must be strictly less than"):
            WelfordPreconditioner(d=2, min_variance=10.0, max_variance=5.0)

        with pytest.raises(ValueError, match="Expected shape"):
            wp = WelfordPreconditioner(d=3)
            wp.update(np.ones(4))
