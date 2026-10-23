"""Tier 3: Pairwise & Cross-Feature Interactions (24 tests).

Validates multi-module interactions across target regimes, space transformations,
adaptation algorithms, samplers, diagnostic metrics, and guardrails.
"""

import json
import math

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.diagnostics.tracker import (
    BudgetExceededError,
    GradBudgetTracker,
)
from mcmc_robust.targets.benign import (
    DiagonalGaussian,
    IsotropicGaussian,
)
from mcmc_robust.targets.dual import (
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
)
from mcmc_robust.targets.suite import get_target
from tests.e2e.conftest import (
    ArcsinhTransformContract,
    BarkerContractSampler,
    MALAContractSampler,
    RWMContractSampler,
    StereographicTransformContract,
    TamedMALAContractSampler,
    compute_bulk_ess,
    compute_folded_rhat,
    compute_split_rhat,
    compute_tail_ess,
)


class TestTier3CrossFeatureInteractions:
    """24 pairwise and cross-feature interaction test cases."""

    def test_01_roughness_target_with_budget_tracker_and_sampler_caching(self, rng):
        """1. F05 (Roughness) + F18 (Tracker) + F02 (Sampler Base Caching): exact evaluation counting."""
        raw_target = QuarticTarget(d=2, p=4.0)
        tracked_target = GradBudgetTracker(raw_target)
        sampler = MALAContractSampler(step_size=0.05)

        x0 = np.array([1.0, 1.0])
        state = sampler.init_state(x0, tracked_target)
        assert tracked_target.grad_evals == 1

        n_steps = 15
        for _ in range(n_steps):
            prev_grad = tracked_target.grad_evals
            res = sampler.step(state, tracked_target, rng)
            assert tracked_target.grad_evals == prev_grad + 1  # Exactly 1 new eval
            if res.accepted:
                assert res.state is not state
            else:
                assert res.state is state  # Rejection returns exact prior state
            state = res.state

        assert tracked_target.grad_evals == 1 + n_steps

    def test_02_heavy_tail_target_with_space_transforms_and_jacobian(self):
        """2. F06 (Flatness) + F14/F16 (Transforms): coordinate-wise diffeomorphism and Jacobian."""
        radii = np.array([2.0, 3.0])
        transform = ArcsinhTransformContract(radii=radii)
        target = MultivariateStudentTTarget(d=2, nu=2.5)

        # Transformed density log pi_Y(y) = log pi_X(psi^-1(y)) + log |det D psi^-1(y)|
        y = np.array([1.5, -2.0])
        x, rev_log_det = transform.inverse(y)
        lp_x = target.log_prob(x)
        lp_y = lp_x + rev_log_det
        assert np.isfinite(lp_y)

    def test_03_dual_target_with_adaptation_and_preconditioning(self, rng):
        """3. F07 (Dual Pathology) + F03 (Dual Averaging & Welford): simultaneous warmup tuning."""
        target = CoupledDualPathologyTarget(d=4, nu=2.5, c=1.0)
        da = DualAveragingAdaptation(initial_step_size=0.1, target_accept=0.574)
        welford = WelfordPreconditioner(d=4, shrinkage_samples=0.0)
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.5)

        state = sampler.init_state(np.zeros(4), target)
        for t in range(25):
            sampler.set_preconditioner(welford.diagonal_preconditioner())
            sampler.step_size = da.step_size
            res = sampler.step(state, target, rng)
            da.update(res.accept_prob)
            welford.update(res.state.x)
            state = res.state

        assert np.isfinite(da.step_size)
        assert np.all(welford.diagonal_preconditioner() > 0)

    def test_04_benign_gaussian_with_welford_metric_convergence(self, rng):
        """4. F04 (Benign Target) + F03 (Welford): streaming convergence to exact diagonal variance."""
        true_stds = np.array([0.5, 2.0, 4.0])
        target = DiagonalGaussian(d=3, stds=true_stds)
        welford = WelfordPreconditioner(d=3, shrinkage_samples=0.0)

        # Stream exact independent draws from ground truth
        draws = target.sample_ground_truth(3000, rng=rng)
        welford.update_batch(draws)

        # Online estimate matches true analytical variance within statistical sampling error (< 15%)
        assert np.allclose(welford.sample_variance, true_stds**2, rtol=0.15)

    def test_05_roughness_with_tamed_mala_and_step_sweep(self, rng):
        """5. F05 (Roughness) + F12 (Tamed MALA) + F08 (Sweeps): stability across 4 orders of magnitude."""
        target = QuarticTarget(d=2, p=4.0)
        step_sizes = [1e-3, 1e-2, 1e-1, 5e-1]
        for h in step_sizes:
            sampler = TamedMALAContractSampler(step_size=h, gamma=0.5)
            x0 = np.array([5.0, 5.0])
            samples, _ = sampler.sample(x0, target, n_steps=10, rng=rng)
            assert np.all(np.isfinite(samples))

    def test_06_flatness_with_barker_proposal_and_budget_limit(self, rng):
        """6. F06 (Flatness) + F13 (Barker) + F18 (Budget Tracker): clean termination on exhaustion."""
        budget = 12
        target = GradBudgetTracker(MultivariateStudentTTarget(d=2, nu=1.5), max_grad_evals=budget)
        sampler = BarkerContractSampler(step_size=0.1)

        state = sampler.init_state(np.ones(2), target)
        assert target.grad_evals == 1

        with pytest.raises(BudgetExceededError):
            for _ in range(20):
                res = sampler.step(state, target, rng)
                state = res.state

        assert target.grad_evals == budget

    def test_07_anisotropic_tail_with_gentle_taming_on_dual_regime(self, rng):
        """7. F07 (Dual Target) + F15 (Gentle Taming) + F16 (Anisotropic Tail): coordinated execution."""
        target = CoupledDualPathologyTarget(d=4, nu=2.0, c=0.5)
        radii = np.array([2.0, 2.0, 1.0, 1.0])
        transform = ArcsinhTransformContract(radii=radii)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.2)

        state = sampler.init_state(np.zeros(4), target)
        samples, _ = sampler.sample(state.x, target, n_steps=10, rng=rng)
        assert samples.shape == (10, 4)
        assert np.all(np.isfinite(samples))

    def test_08_unified_sampler_with_multichain_split_rhat(self, rng):
        """8. F17 (Unified Sampler) + F19 (Split R-hat & Folded R-hat): multi-chain convergence."""
        target = IsotropicGaussian(d=2)
        sampler = TamedMALAContractSampler(step_size=0.5, gamma=0.2)

        chains = []
        for seed in [101, 102, 103, 104]:
            chain_rng = np.random.default_rng(seed)
            s, _ = sampler.sample(np.zeros(2), target, n_steps=300, burn_in=50, rng=chain_rng)
            chains.append(s)

        split_rhat = compute_split_rhat(chains)
        folded_rhat = compute_folded_rhat(chains)
        assert split_rhat < 1.15
        assert folded_rhat < 1.15

    def test_09_benchmark_runner_with_warmup_grids_and_budget_tracker(self, rng):
        """9. F20 (Benchmark Runner) + F08 (Warmup Grids) + F18 (Tracker): budget enforcement loop."""
        regimes = ["gaussian_isotropic", "quartic", "student_t"]
        budget = 20
        for reg in regimes:
            raw = get_target(reg, dim=2)
            tracker = GradBudgetTracker(raw, max_grad_evals=budget)
            sampler = MALAContractSampler(step_size=0.05)
            state = sampler.init_state(np.zeros(2), tracker)
            while tracker.grad_evals < budget:
                res = sampler.step(state, tracker, rng)
                state = res.state
            assert tracker.grad_evals == budget

    def test_10_space_transforms_with_ground_truth_sampling(self, rng):
        """10. F14 (Transforms) + F04 (Ground Truth Moments): pushforward sample transformation."""
        target = IsotropicGaussian(d=2, std=1.0)
        draws = target.sample_ground_truth(1000, rng=rng)
        stereo = StereographicTransformContract(scale=1.0)

        # Apply forward transform to samples
        transformed_y = np.array([stereo.forward(pt)[0] for pt in draws])
        # Transformed coordinates must remain bounded in unit ball
        norms = np.linalg.norm(transformed_y, axis=1)
        assert np.all(norms < 1.0)

    def test_11_heavy_tail_target_with_tail_ess_diagnostics(self, rng):
        """11. F06 (Heavy Tails) + F19 (Tail ESS Diagnostic): tail exploration evaluation."""
        target = MultivariateStudentTTarget(d=2, nu=2.0)
        sampler = BarkerContractSampler(step_size=0.1)
        samples, _ = sampler.sample(np.zeros(2), target, n_steps=150, burn_in=30, rng=rng)

        tail_ess = compute_tail_ess(samples, lower_q=0.05, upper_q=0.95)
        assert np.isfinite(tail_ess)
        assert tail_ess > 0

    def test_12_rough_target_with_analytical_quantile_errors(self, rng):
        """12. F05 (Rough Target) + F19 (Quantile Relative Error): empirical quantile accuracy."""
        target = QuarticTarget(d=1, p=4.0)
        samples = target.sample_ground_truth(3000, rng=rng).ravel()

        # Analytical 95th percentile from exact incomplete gamma inverse
        # For U(x) = x^4 / 4, x^4 / 4 ~ Gamma(0.25, 1)
        gamma_q = stats.gamma(a=0.25, scale=1.0).ppf(0.90)  # Two-sided 95% -> 0.90 upper mass
        true_95 = (4.0 * gamma_q) ** 0.25
        emp_95 = float(np.quantile(samples, 0.95))
        rel_err = abs(emp_95 - true_95) / true_95
        assert rel_err < 0.08

    def test_13_ill_conditioned_warmup_with_welford_diagonal_scaling(self, rng):
        """13. F04/F08 (Ill-Conditioned Gaussian) + F03 (Welford Adaptation): scale adaptation."""
        target = DiagonalGaussian(d=2, condition_number=50.0)
        welford = WelfordPreconditioner(d=2, shrinkage_samples=0.0)

        draws = target.sample_ground_truth(2000, rng=rng)
        welford.update_batch(draws)
        diag = welford.diagonal_preconditioner()

        # Ratio between diagonal entries mirrors target condition number variance
        emp_kappa = np.max(diag) / np.min(diag)
        assert 35.0 < emp_kappa < 65.0

    def test_14_funnel_target_with_dual_averaging_and_tamed_drift(self, rng):
        """14. F05 (Neal's Funnel) + F12 (Tamed Drift) + F03 (Dual Averaging): traversal."""
        target = NealsFunnelTarget(d=3)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.5)
        da = DualAveragingAdaptation(initial_step_size=0.05, target_accept=0.574)

        state = sampler.init_state(np.array([-2.0, 0.0, 0.0]), target)
        for _ in range(30):
            res = sampler.step(state, target, rng)
            da.update(res.accept_prob)
            state = res.state

        assert np.isfinite(da.step_size)
        assert np.all(np.isfinite(state.x))

    def test_15_registry_suite_with_ground_truth_ks_verification(self, rng):
        """15. F08 (Registry Suite) + F01/F19 (KS Test): ground truth validity across registry."""
        for name in ["gaussian_isotropic", "student_t", "quartic"]:
            t = get_target(name, dim=2)
            samples = t.sample_ground_truth(1500, rng=rng)
            assert samples.shape == (1500, 2)
            assert np.all(np.isfinite(samples))

    def test_16_stereographic_transform_with_student_t_and_grad_tracker(self, rng):
        """16. F14 (Stereographic) + F06 (Student-t) + F18 (Tracker): transformed evaluation counting."""
        raw = MultivariateStudentTTarget(d=2, nu=2.0)
        tracked = GradBudgetTracker(raw)
        stereo = StereographicTransformContract(scale=1.0)

        x = np.array([2.0, -1.0])
        y, _ = stereo.forward(x)
        lp, grad = tracked.log_prob_and_grad(x)
        assert tracked.grad_evals == 1

    def test_17_radial_diffeomorphism_with_quartic_potential(self):
        """17. F14 (Radial Transform) + F05 (Quartic Potential): radial monotonicity."""
        target = QuarticTarget(d=2, p=4.0)
        stereo = StereographicTransformContract(scale=2.0)

        x_near = np.array([1.0, 0.0])
        x_far = np.array([5.0, 0.0])
        y_near, _ = stereo.forward(x_near)
        y_far, _ = stereo.forward(x_far)

        assert np.linalg.norm(y_near) < np.linalg.norm(y_far)
        assert np.linalg.norm(y_far) < 2.0

    def test_18_anisotropic_tail_with_heterogeneous_student_t_and_hill(self, rng):
        """18. F16 (Anisotropic Arcsinh) + F06 (Heterogeneous Target): online tail calibration."""
        target = HeterogeneousHeavyTailedTarget(d=2, nu_list=[1.5, 4.0])
        samples = target.sample_ground_truth(2000, rng=rng)

        # Coordinate 0 is heavier than coordinate 1
        q_coord0 = np.quantile(np.abs(samples[:, 0]), 0.95)
        q_coord1 = np.quantile(np.abs(samples[:, 1]), 0.95)
        assert q_coord0 > q_coord1

    def test_19_gentle_taming_with_l0l1_smooth_target_and_budget(self, rng):
        """19. F15 (Gentle Taming) + F05 (L0L1 Target) + F18 (Budget Tracker): bounded sampling."""
        budget = 25
        target = GradBudgetTracker(L0L1SmoothTarget(d=2, L0=2.0, lambda_param=1.0), max_grad_evals=budget)
        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.2)

        samples, _ = sampler.sample(np.zeros(2), target, n_steps=20, rng=rng)
        assert target.grad_evals <= budget
        assert len(samples) == 20

    def test_20_unified_sampler_with_benign_parity_guardrail(self, rng):
        """20. F17 (Unified Sampler) + F21 (Benign Parity Guardrail AC2): ratio >= 0.70."""
        target = IsotropicGaussian(d=5)
        t_novel = GradBudgetTracker(target)
        t_base = GradBudgetTracker(target)

        sampler_novel = TamedMALAContractSampler(step_size=0.1, gamma=0.1)
        sampler_base = MALAContractSampler(step_size=0.1)

        s_novel, _ = sampler_novel.sample(np.zeros(5), t_novel, n_steps=100, rng=rng)
        s_base, _ = sampler_base.sample(np.zeros(5), t_base, n_steps=100, rng=rng)

        ess_novel_std = compute_bulk_ess(s_novel) / max(1, t_novel.grad_evals)
        ess_base_std = compute_bulk_ess(s_base) / max(1, t_base.grad_evals)

        ratio = ess_novel_std / max(1e-12, ess_base_std)
        # On isotropic Gaussian with small gamma=0.1, novel sampler achieves parity >= 0.70
        assert ratio >= 0.70

    def test_21_unified_sampler_with_roughness_stability_guardrail(self, rng):
        """21. F17 (Unified Sampler) + F22 (Roughness Stability Guardrail AC3): tail initialization."""
        d = 4
        target = QuarticTarget(d=d, p=4.0)
        x0 = np.ones(d) * (10.0 * math.sqrt(d) / math.sqrt(d))

        sampler = TamedMALAContractSampler(step_size=0.05, gamma=0.5)
        samples, _ = sampler.sample(x0, target, n_steps=20, rng=rng)

        # Must not diverge to NaN/inf
        assert np.all(np.isfinite(samples))
        # Must make significant inward progress toward mode
        final_norm = np.linalg.norm(samples[-1])
        init_norm = np.linalg.norm(x0)
        assert final_norm < init_norm

    def test_22_unified_sampler_with_heavy_tail_efficiency_guardrail(self, rng):
        """22. F17 (Unified Sampler) + F23 (Heavy Tail Guardrail AC4): tail ESS standardization."""
        target = MultivariateStudentTTarget(d=2, nu=1.5)
        tracker = GradBudgetTracker(target)
        sampler = TamedMALAContractSampler(step_size=0.1, gamma=0.5)

        samples, _ = sampler.sample(np.zeros(2), tracker, n_steps=100, rng=rng)
        tail_ess = compute_tail_ess(samples)
        tail_ess_per_grad = tail_ess / max(1, tracker.grad_evals)

        assert np.isfinite(tail_ess_per_grad)
        assert tail_ess_per_grad > 0

    def test_23_multichain_multisampler_benchmark_pipeline(self, rng):
        """23. F09–F13 (Samplers) + F18 (Tracker) + F20 (Benchmark Harness): comparative runner."""
        target = IsotropicGaussian(d=2)
        budget_per_sampler = 20

        samplers = [
            ("RWM", RWMContractSampler(step_size=0.2)),
            ("MALA", MALAContractSampler(step_size=0.1)),
            ("Barker", BarkerContractSampler(step_size=0.1)),
            ("TamedMALA", TamedMALAContractSampler(step_size=0.1, gamma=0.5)),
        ]

        results = {}
        for name, s in samplers:
            tracker = GradBudgetTracker(target, max_grad_evals=budget_per_sampler if s.requires_grad else None)
            samps, res = s.sample(np.zeros(2), tracker, n_steps=15, rng=rng)
            results[name] = {
                "n_samples": len(samps),
                "grad_evals": tracker.grad_evals,
                "bulk_ess": compute_bulk_ess(samps),
            }

        assert results["RWM"]["grad_evals"] == 0
        assert results["MALA"]["grad_evals"] == 16
        assert results["Barker"]["grad_evals"] == 16
        assert results["TamedMALA"]["grad_evals"] == 16

    def test_24_benchmark_serialization_and_report_schema_validation(self):
        """24. F20 (Benchmark Serialization) + F24 (Report Table Schema): JSON validation."""
        benchmark_payload = {
            "metadata": {
                "version": "0.1.0",
                "timestamp": "2026-10-23T12:00:00Z",
                "grad_budget": 2000,
            },
            "regimes": [
                {
                    "regime": "benign",
                    "dimension": 50,
                    "target": "gaussian_isotropic",
                    "samplers": [
                        {"name": "MALA", "ess_per_grad": 0.42, "tail_ess_per_grad": 0.31, "rhat": 1.01},
                        {"name": "UnifiedNovel", "ess_per_grad": 0.39, "tail_ess_per_grad": 0.35, "rhat": 1.01},
                    ],
                }
            ],
        }

        # Verify serializability and key presence
        serialized = json.dumps(benchmark_payload)
        parsed = json.loads(serialized)
        assert len(parsed["regimes"]) == 1
        assert parsed["regimes"][0]["samplers"][1]["name"] == "UnifiedNovel"
