"""Unit tests for canonical PosteriorDB-style applied Bayesian targets and benchmark runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mcmc_robust.bench.applied_runner import (
    APPLIED_SAMPLER_NAMES,
    APPLIED_TARGET_NAMES,
    AppliedBenchmarkConfig,
    AppliedBenchmarkRunner,
)
from mcmc_robust.targets import (
    HierarchicalItemResponse2PLTarget,
    RadonHierarchicalTarget,
    RobustStudentTRegressionTarget,
    SparseHorseshoeLogisticTarget,
    StochasticVolatilityLogReturnTarget,
    TargetRegistry,
)
from tests.conftest import max_relative_error, numerical_gradient

APPLIED_TARGET_CLASSES = [
    RadonHierarchicalTarget,
    SparseHorseshoeLogisticTarget,
    RobustStudentTRegressionTarget,
    HierarchicalItemResponse2PLTarget,
    StochasticVolatilityLogReturnTarget,
]


@pytest.mark.unit
@pytest.mark.parametrize("target_cls", APPLIED_TARGET_CLASSES)
def test_applied_targets_default_dimensions_and_metadata(target_cls) -> None:
    """Verify default dimensions, sample_ground_truth, and moments metadata for all 5 applied targets."""
    target = target_cls()
    expected_dims = {
        RadonHierarchicalTarget: 18,
        SparseHorseshoeLogisticTarget: 21,
        RobustStudentTRegressionTarget: 12,
        HierarchicalItemResponse2PLTarget: 22,
        StochasticVolatilityLogReturnTarget: 22,
    }
    assert target.d == expected_dims[target_cls]
    assert target.dim == expected_dims[target_cls]

    rng = np.random.default_rng(123)
    samples = target.sample_ground_truth(16, rng=rng)
    assert samples.shape == (16, target.d)
    assert np.all(np.isfinite(samples))

    moments = target.exact_moments_or_quantiles()
    assert isinstance(moments, dict)
    assert "mean" in moments
    assert np.asarray(moments["mean"]).shape == (target.d,)


@pytest.mark.unit
@pytest.mark.parametrize("target_cls", APPLIED_TARGET_CLASSES)
def test_applied_targets_finite_difference_bulk_and_tail(target_cls) -> None:
    """Test 1: Central finite-difference gradient (eps=1e-6) matches grad_log_prob to < 1e-6 rel error."""
    target = target_cls()
    rng = np.random.default_rng(2026)

    # 1. Origin / canonical bulk initialization
    x_zero = np.zeros(target.d, dtype=np.float64)
    anal_zero = target.grad_log_prob(x_zero)
    num_zero = numerical_gradient(target, x_zero, h=1e-6)
    assert np.all(np.isfinite(anal_zero))
    assert max_relative_error(anal_zero, num_zero) < 1e-6

    # 2. Random bulk points
    for _ in range(3):
        x_bulk = rng.normal(0.0, 0.7, size=target.d)
        anal_bulk = target.grad_log_prob(x_bulk)
        num_bulk = numerical_gradient(target, x_bulk, h=1e-6)
        assert np.all(np.isfinite(anal_bulk))
        assert max_relative_error(anal_bulk, num_bulk) < 1e-6

    # 3. Far-tail / funnel-neck stress points
    x_tail = rng.normal(0.0, 2.5, size=target.d)
    if isinstance(target, RadonHierarchicalTarget):
        # Deep hierarchical county funnel neck v = log(sigma_alpha) = -2.2
        x_tail[target.n_counties + 1] = -2.2
    elif isinstance(target, SparseHorseshoeLogisticTarget):
        # Shrinkage funnel neck on inactive local log-scales and global log-scale
        x_tail[target.n_features : 2 * target.n_features] = -1.8
        x_tail[-1] = -2.0
    elif isinstance(target, RobustStudentTRegressionTarget):
        # Far-tail regression weights and tight log noise scale
        x_tail[1:4] = np.array([8.0, -7.5, 6.0])
        x_tail[-1] = -1.5
    elif isinstance(target, HierarchicalItemResponse2PLTarget):
        # Funnel neck on log discrimination scale v
        x_tail[-1] = -2.0
    elif isinstance(target, StochasticVolatilityLogReturnTarget):
        # High persistence u_phi = atanh(phi) and tight innovation scale v
        x_tail[-2] = 1.6
        x_tail[-1] = -1.8

    anal_tail = target.grad_log_prob(x_tail)
    num_tail = numerical_gradient(target, x_tail, h=1e-6)
    assert np.all(np.isfinite(anal_tail))
    assert max_relative_error(anal_tail, num_tail) < 1e-6

    # 4. Also verify JAX autodiff matches analytical gradient to < 1e-9
    jax_grad = np.asarray(jax.grad(target.jax_log_prob)(jnp.asarray(x_tail, dtype=jnp.float64)), dtype=np.float64)
    assert max_relative_error(anal_tail, jax_grad) < 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("target_cls", APPLIED_TARGET_CLASSES)
def test_applied_targets_log_prob_and_grad_bitwise_match(target_cls) -> None:
    """Test 2: log_prob_and_grad(x) matches (log_prob(x), grad_log_prob(x)) bitwise across all 5 targets."""
    target = target_cls()
    rng = np.random.default_rng(999)

    for scale in (0.0, 0.5, 2.0):
        x = np.zeros(target.d, dtype=np.float64) if scale == 0.0 else rng.normal(0.0, scale, size=target.d)
        lp_joint, grad_joint = target.log_prob_and_grad(x)
        lp_sep = target.log_prob(x)
        grad_sep = target.grad_log_prob(x)

        assert lp_joint == lp_sep
        assert np.array_equal(grad_joint, grad_sep)


@pytest.mark.unit
def test_applied_targets_input_validation_and_registry() -> None:
    """Test 3: Input validation and TargetRegistry.create(...) instantiation for all 5 targets."""
    expected_map = {
        "radon_hierarchical": (RadonHierarchicalTarget, 18),
        "sparse_horseshoe_logistic": (SparseHorseshoeLogisticTarget, 21),
        "robust_student_t_regression": (RobustStudentTRegressionTarget, 12),
        "hierarchical_irt_2pl": (HierarchicalItemResponse2PLTarget, 22),
        "stochastic_volatility": (StochasticVolatilityLogReturnTarget, 22),
    }

    registered = TargetRegistry.list_targets()
    for name in expected_map:
        assert name in registered

    for name, (cls, expected_d) in expected_map.items():
        target = TargetRegistry.create(name)
        assert isinstance(target, cls)
        assert target.d == expected_d

        # Wrong input shape raises ValueError in _validate_input / log_prob
        with pytest.raises(ValueError, match="Expected 1D array"):
            target.log_prob(np.zeros(expected_d + 1))
        with pytest.raises(ValueError, match="Expected 1D array"):
            target.grad_log_prob(np.zeros((2, expected_d)))
        with pytest.raises(ValueError, match="Expected 1D array"):
            target.log_prob_and_grad(np.zeros(expected_d - 1))

        # Non-finite input returns -inf log_prob and NaN grad
        x_nan = np.zeros(expected_d, dtype=np.float64)
        x_nan[0] = np.nan
        assert target.log_prob(x_nan) == -np.inf
        assert np.all(np.isnan(target.grad_log_prob(x_nan)))
        lp_nan, g_nan = target.log_prob_and_grad(x_nan)
        assert lp_nan == -np.inf
        assert np.all(np.isnan(g_nan))

    # Constructor parameter validation checks
    with pytest.raises(ValueError):
        RadonHierarchicalTarget(n_counties=0)
    with pytest.raises(ValueError):
        SparseHorseshoeLogisticTarget(n_features=1)
    with pytest.raises(ValueError):
        RobustStudentTRegressionTarget(nu=0.0)
    with pytest.raises(ValueError):
        HierarchicalItemResponse2PLTarget(n_students=1)
    with pytest.raises(ValueError):
        StochasticVolatilityLogReturnTarget(n_steps=2)
    with pytest.raises(ValueError, match="Unknown target"):
        TargetRegistry.create("nonexistent_applied_model")


@pytest.mark.unit
def test_applied_benchmark_runner_quick_end_to_end(tmp_path: Path) -> None:
    """Test 4: AppliedBenchmarkRunner(quick=True) runs all 5 models & 6 samplers and writes RFC 8259 JSON/CSV."""
    config = AppliedBenchmarkConfig(
        quick=True,
        output_dir=str(tmp_path),
        seed=42,
    )
    runner = AppliedBenchmarkRunner(config=config)
    result = runner.run_all()

    json_path = tmp_path / "applied_benchmark_results.json"
    csv_path = tmp_path / "applied_benchmark_results.csv"
    assert json_path.exists()
    assert csv_path.exists()

    # Validate strict RFC 8259 JSON compliance (no NaN / Infinity tokens)
    raw_json = json_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_json
    assert "Infinity" not in raw_json
    parsed = json.loads(raw_json)
    assert "metadata" in parsed
    assert "summary" in parsed
    assert "results" in parsed
    assert len(parsed["results"]) == len(APPLIED_TARGET_NAMES) * len(APPLIED_SAMPLER_NAMES)

    # Validate CSV output
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(APPLIED_TARGET_NAMES) * len(APPLIED_SAMPLER_NAMES)
    required_cols = {
        "target_name",
        "sampler_name",
        "dimension",
        "bulk_ess",
        "tail_ess",
        "ess_per_grad",
        "tail_ess_per_grad",
        "split_rhat",
        "divergences",
        "grad_evals",
        "wall_time_sec",
    }
    assert required_cols.issubset(set(rows[0].keys()))

    # Verify UnifiedRobustNUTS and BlackJAXNUTS + JaxArcsinhTransform achieve finite ESS and zero unhandled exceptions
    for target_name in APPLIED_TARGET_NAMES:
        for robust_sampler in ("UnifiedRobustNUTS", "BlackJAXNUTS + JaxArcsinhTransform"):
            matching = [r for r in result.records if r.target_name == target_name and r.sampler_name == robust_sampler]
            assert len(matching) == 1
            rec = matching[0]
            assert rec.error is None
            assert np.isfinite(rec.bulk_ess) and rec.bulk_ess > 0.0
            assert np.isfinite(rec.tail_ess) and rec.tail_ess > 0.0
            assert np.isfinite(rec.ess_per_grad) and rec.ess_per_grad > 0.0
            assert np.isfinite(rec.tail_ess_per_grad) and rec.tail_ess_per_grad > 0.0
            assert rec.grad_evals > 0
            assert rec.wall_time_sec > 0.0
