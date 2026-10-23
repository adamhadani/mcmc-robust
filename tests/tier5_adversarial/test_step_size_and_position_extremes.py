"""Tier 5 Adversarial Tests: Step Sizes, Initialization Extremes, and Preconditioners.

Verifies sampler stability under near-zero step sizes (1e-16), huge step sizes (1e12),
extreme positions (||x0|| = 10*sqrt(d) and 50*sqrt(d)), and ill-conditioned metrics.
"""

import numpy as np
import pytest

from mcmc_robust.diagnostics.tracker import GradBudgetTracker
from mcmc_robust.samplers.barker import BarkerLangevin
from mcmc_robust.samplers.hmc import HMC
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.samplers.novel.anisotropic_tail import AnisotropicTailMALA
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.samplers.rwm import RandomWalkMetropolis
from mcmc_robust.samplers.tamed_mala import GloballyTamedMALA
from mcmc_robust.targets.benign import IsotropicGaussian
from mcmc_robust.targets.heavy_tailed import MultivariateStudentTTarget
from mcmc_robust.targets.roughness import QuarticTarget

ALL_SAMPLER_FACTORIES = [
    ("rwm", lambda h, p=None: RandomWalkMetropolis(step_size=h, preconditioner=p)),
    ("mala", lambda h, p=None: MALA(step_size=h, preconditioner=p)),
    ("hmc", lambda h, p=None: HMC(step_size=h, n_leapfrog=3, preconditioner=p)),
    ("tamed_mala", lambda h, p=None: GloballyTamedMALA(step_size=h, gamma=0.1, preconditioner=p)),
    ("barker", lambda h, p=None: BarkerLangevin(step_size=h, preconditioner=p)),
    ("gentle_mala", lambda h, p=None: GentleMALA(step_size=h, gamma=1.0, preconditioner=p)),
    ("anisotropic_tail_mala", lambda h, p=None: AnisotropicTailMALA(step_size=h, radii=1.0, preconditioner=p)),
    ("unified_robust", lambda h, p=None: UnifiedRobustSampler(step_size=h, gamma=0.1, radius=2.0, preconditioner=p)),
]


class TestStepSizeExtremes:
    """Verifies sampler robustness to near-zero and explosive step sizes."""

    @pytest.mark.parametrize("s_name, factory", ALL_SAMPLER_FACTORIES)
    @pytest.mark.parametrize("h", [1e-8, 1e-12, 1e-16])
    def test_near_zero_step_size(self, s_name: str, factory, h: float) -> None:
        d = 5
        sampler = factory(h)
        target = QuarticTarget(d=d)
        rng = np.random.default_rng(42)
        x0 = np.ones(d)

        state = sampler.init_state(x0, target)
        res = sampler.step(state, target, rng)

        assert np.isfinite(res.accept_prob)
        assert 0.0 <= res.accept_prob <= 1.0
        assert np.all(np.isfinite(res.state.x))
        assert np.isfinite(res.state.log_prob)

    @pytest.mark.parametrize("s_name, factory", ALL_SAMPLER_FACTORIES)
    @pytest.mark.parametrize("h", [1e4, 1e8, 1e12])
    def test_huge_step_size_defensive_overflow_guards(self, s_name: str, factory, h: float) -> None:
        d = 5
        sampler = factory(h)
        target = QuarticTarget(d=d)
        tracker = GradBudgetTracker(target)
        rng = np.random.default_rng(42)
        x0 = np.ones(d) * 2.0

        state = sampler.init_state(x0, tracker)
        tracker.reset()

        # Under huge step sizes, transformed space samplers cleanly reject without unsuppressed warnings
        res = sampler.step(state, tracker, rng)

        # Huge step sizes should either reject cleanly via overflow guard or zero accept_prob
        assert not res.accepted
        assert res.accept_prob <= 1e-300
        # Critical: ChainState must not be corrupted
        assert np.all(np.isfinite(res.state.x))
        assert np.array_equal(res.state.x, state.x)
        assert res.state.log_prob == state.log_prob


class TestExtremeInitialPositions:
    """Verifies sampler behavior from extreme tail positions ||x0|| = 10*sqrt(d) and 50*sqrt(d)."""

    @pytest.mark.parametrize("d", [10, 50, 100])
    @pytest.mark.parametrize("multiplier", [10.0, 50.0])
    def test_extreme_tail_quartic_contraction(self, d: int, multiplier: float) -> None:
        target = QuarticTarget(d=d)
        rng = np.random.default_rng(100 + d)
        direction = rng.normal(size=d)
        direction = direction / np.linalg.norm(direction)
        x0 = direction * (multiplier * np.sqrt(d))

        # GentleMALA and GloballyTamedMALA operate directly on rough quartic potential
        # UnifiedRobustSampler on rough targets is configured with mask=False (or gentle mode)
        for sampler in [
            GentleMALA(step_size=0.01, gamma=1.0),
            GloballyTamedMALA(step_size=0.01, gamma=0.1),
            UnifiedRobustSampler(step_size=0.01, gamma=0.1, radius=2.0, mask=np.zeros(d, dtype=bool)),
        ]:
            state = sampler.init_state(x0, target)
            assert np.isfinite(state.log_prob)
            assert np.all(np.isfinite(state.grad_log_prob))

            curr_state = state
            for _ in range(5):
                res = sampler.step(curr_state, target, rng)
                curr_state = res.state
                assert np.all(np.isfinite(curr_state.x))
                assert np.isfinite(curr_state.log_prob)

    @pytest.mark.parametrize("d", [10, 50])
    def test_extreme_tail_student_t(self, d: int) -> None:
        target = MultivariateStudentTTarget(d=d, nu=2.5)
        rng = np.random.default_rng(200 + d)
        x0 = np.ones(d) * (10.0 * np.sqrt(d))

        for sampler in [
            AnisotropicTailMALA(step_size=0.05, radii=1.0),
            UnifiedRobustSampler(step_size=0.05, gamma=0.1, radius=2.0),
            BarkerLangevin(step_size=0.05),
        ]:
            state = sampler.init_state(x0, target)
            assert np.isfinite(state.log_prob)
            curr = state
            for _ in range(5):
                res = sampler.step(curr, target, rng)
                curr = res.state
                assert np.all(np.isfinite(curr.x))


class TestPreconditionerExtremes:
    """Verifies preconditioner validation and performance under extreme condition numbers."""

    def test_invalid_preconditioner_validation(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            MALA(step_size=0.1, preconditioner=np.array([1.0, 0.0, 1.0]))

        with pytest.raises(ValueError, match="strictly positive"):
            MALA(step_size=0.1, preconditioner=np.array([1.0, -0.5, 1.0]))

        with pytest.raises(ValueError, match="1D array"):
            MALA(step_size=0.1, preconditioner=np.eye(3))

    @pytest.mark.parametrize("s_name, factory", ALL_SAMPLER_FACTORIES)
    def test_ill_conditioned_preconditioner_stability(self, s_name: str, factory) -> None:
        d = 4
        # Highly anisotropic condition number: 1e6 between coordinates
        p = np.array([1e-3, 1.0, 10.0, 1e3])
        sampler = factory(1e-4, p)
        target = IsotropicGaussian(d=d)
        rng = np.random.default_rng(42)

        state = sampler.init_state(np.zeros(d), target)
        res = sampler.step(state, target, rng)
        assert np.isfinite(res.accept_prob)
        assert np.all(np.isfinite(res.state.x))
