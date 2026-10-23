"""Tier 5 Adversarial Tests: Extreme Dimensionality (d=100 and d=1).

Verifies stability, oracle accounting, analytical moments, and sampling
behavior under extreme dimensionality across all targets and samplers.
"""

import numpy as np
import pytest

from mcmc_robust.bench.adapters import make_radial_mala, make_stereographic_mala
from mcmc_robust.diagnostics.tracker import GradBudgetTracker
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
    CoupledDualPathologyTarget,
)
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)
from mcmc_robust.targets.roughness import (
    L0L1SmoothTarget,
    NealsFunnelTarget,
    QuarticTarget,
    RippledGaussianTarget,
    SuperlinearTarget,
)


@pytest.mark.parametrize("d", [1, 100])
class TestExtremeDimensionsTargets:
    """Verifies target distributions at extreme dimensions d=1 and d=100."""

    def test_gaussian_targets_d100(self, d: int) -> None:
        iso = IsotropicGaussian(d=d, std=2.0)
        diag = DiagonalGaussian(d=d, condition_number=50.0)
        full = FullCovarianceGaussian(d=d, condition_number=20.0)

        for target in [iso, diag, full]:
            assert target.d == d
            x = np.ones(d, dtype=np.float64) * 0.5
            lp = target.log_prob(x)
            grad = target.grad_log_prob(x)
            lp_j, grad_j = target.log_prob_and_grad(x)

            assert np.isfinite(lp)
            assert np.all(np.isfinite(grad))
            assert grad.shape == (d,)
            assert np.isclose(lp, lp_j, rtol=1e-10)
            assert np.allclose(grad, grad_j, rtol=1e-10)

            # Ground truth sampling
            samples = target.sample_ground_truth(10, rng=np.random.default_rng(42))
            assert samples.shape == (10, d)
            assert np.all(np.isfinite(samples))

            # Analytical moments
            moments = target.exact_moments_or_quantiles()
            assert "mean" in moments
            assert len(moments["mean"]) == d

    def test_roughness_targets_d100(self, d: int) -> None:
        quartic = QuarticTarget(d=d, p=4.0)
        superlin = SuperlinearTarget(d=d, p=6.0)
        l0l1 = L0L1SmoothTarget(d=d, L0=1.0, lambda_param=0.5)
        rippled = RippledGaussianTarget(d=d, sigma=1.0, amplitude=0.2, omega=5.0)

        targets = [quartic, superlin, l0l1, rippled]
        if d >= 2:
            funnel = NealsFunnelTarget(d=d, sigma_v=2.0)
            targets.append(funnel)

        for target in targets:
            assert target.d == d
            x = np.ones(d, dtype=np.float64) * 0.2
            lp, grad = target.log_prob_and_grad(x)
            assert np.isfinite(lp)
            assert np.all(np.isfinite(grad))
            assert grad.shape == (d,)

            samples = target.sample_ground_truth(10, rng=np.random.default_rng(42))
            assert samples.shape == (10, d)
            assert np.all(np.isfinite(samples))

    def test_heavy_tailed_targets_d100(self, d: int) -> None:
        student = MultivariateStudentTTarget(d=d, nu=2.5)
        hetero = HeterogeneousHeavyTailedTarget(d=d)

        for target in [student, hetero]:
            assert target.d == d
            x = np.ones(d, dtype=np.float64) * 0.5
            lp, grad = target.log_prob_and_grad(x)
            assert np.isfinite(lp)
            assert np.all(np.isfinite(grad))
            assert grad.shape == (d,)

            samples = target.sample_ground_truth(10, rng=np.random.default_rng(42))
            assert samples.shape == (10, d)
            assert np.all(np.isfinite(samples))

    def test_dual_targets_d100(self, d: int) -> None:
        conf_ind = ConfinementInducedRoughnessTarget(d=d, base_scale=1.0)
        conf_rough = ConfinementRoughTarget(d=d, nu=2.5, radius=3.0)
        targets = [conf_ind, conf_rough]

        if d >= 2:
            coupled = CoupledDualPathologyTarget(d=d, nu=2.0, c=0.5)
            targets.append(coupled)

        for target in targets:
            assert target.d == d
            x = np.ones(d, dtype=np.float64) * 0.3
            lp, grad = target.log_prob_and_grad(x)
            assert np.isfinite(lp)
            assert np.all(np.isfinite(grad))
            assert grad.shape == (d,)

            samples = target.sample_ground_truth(10, rng=np.random.default_rng(42))
            assert samples.shape == (10, d)
            assert np.all(np.isfinite(samples))


class TestExtremeDimensionsSamplers:
    """Verifies all samplers running at d=100 under zero-redundancy oracle tracking."""

    @pytest.mark.parametrize(
        "sampler_factory",
        [
            lambda d: RandomWalkMetropolis(step_size=0.01),
            lambda d: MALA(step_size=0.01),
            lambda d: HMC(step_size=0.02, n_leapfrog=3),
            lambda d: GloballyTamedMALA(step_size=0.01, gamma=0.1, mode="global"),
            lambda d: GloballyTamedMALA(step_size=0.01, gamma=0.1, mode="coordinate"),
            lambda d: BarkerLangevin(step_size=0.01),
            lambda d: GentleMALA(step_size=0.01, gamma=0.5, mode="global"),
            lambda d: GentleMALA(step_size=0.01, gamma=0.5, mode="coordinate"),
            lambda d: AnisotropicTailMALA(step_size=0.01, radii=1.0),
            lambda d: UnifiedRobustSampler(step_size=0.01, gamma=0.1, radius=2.0, mode="coordinate"),
            lambda d: make_stereographic_mala(step_size=0.01, dim=d, scale=1.0),
            lambda d: make_radial_mala(step_size=0.01, dim=d, radius=3.0, stiffness=0.5),
        ],
    )
    def test_sampler_execution_d100(self, sampler_factory) -> None:
        d = 100
        sampler = sampler_factory(d)
        target = IsotropicGaussian(d=d)
        tracker = GradBudgetTracker(target)

        rng = np.random.default_rng(12345)
        x0 = rng.normal(size=d)
        state = sampler.init_state(x0, tracker)
        expected_len = d + 1 if hasattr(sampler, "spherical_retraction") and sampler.spherical_retraction else d
        assert len(state.x) == expected_len
        assert np.isfinite(state.log_prob)

        # Initial evaluation accounting: exactly 1 call
        if sampler.requires_grad:
            assert tracker.n_grad == 1
        else:
            assert tracker.n_log_prob == 1

        tracker.reset()

        # Step execution
        n_steps = 5
        curr_state = state
        for t in range(n_steps):
            res = sampler.step(curr_state, tracker, rng)
            assert isinstance(res.accepted, bool)
            assert 0.0 <= res.accept_prob <= 1.0
            assert np.isfinite(res.state.log_prob)
            if res.state.grad_log_prob is not None:
                assert np.all(np.isfinite(res.state.grad_log_prob))
                expected_shape = (
                    (d + 1,) if hasattr(sampler, "spherical_retraction") and sampler.spherical_retraction else (d,)
                )
                assert res.state.grad_log_prob.shape == expected_shape
                if hasattr(sampler, "spherical_retraction") and sampler.spherical_retraction:
                    assert res.state.auxiliary["primal_x"].shape == (d,)
            curr_state = res.state

        # Strict oracle accounting: MALA/Barker/Gentle/Tamed/Unified/Stereo/Radial = strictly n_steps grad evals
        if isinstance(sampler, HMC):
            assert tracker.n_grad == n_steps * sampler.n_leapfrog
        elif sampler.requires_grad:
            assert tracker.n_grad == n_steps
        else:
            assert tracker.n_grad == 0
            assert tracker.n_log_prob == n_steps
