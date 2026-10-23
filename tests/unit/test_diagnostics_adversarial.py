"""Adversarial stress and oracle verification test suite for statistical diagnostics.

Author: Challenger M4 1 (Statistical Diagnostics Adversarial Verifier)
Target: src/mcmc_robust/diagnostics/ (ess.py, convergence.py, metrics.py)

Covers:
1. Pathological AR(1) chains (near-unit positive rho -> 1-, negative rho).
2. Heavy-tailed infinite-variance distributions (Cauchy, Pareto alpha <= 2, Student-t nu <= 2).
3. Folded R-hat scale detection vs standard split R-hat location detection (statistical duality).
4. Trending non-stationary chains (single-chain and multi-chain identical trend).
5. Multi-chain permutation invariance and time-shuffle oracle property.
6. Distribution metrics stress (extreme outliers, constant point masses, non-standard quantiles).
7. Degenerate boundaries (N=1, N=2, N=3, constant chains, d=100 high-dimensional scaling).
"""

import numpy as np
import pytest

from mcmc_robust.diagnostics.convergence import (
    compute_folded_rhat,
    compute_split_rhat,
)
from mcmc_robust.diagnostics.ess import (
    compute_bulk_ess,
    compute_tail_ess,
)
from mcmc_robust.diagnostics.metrics import (
    compute_ks_test,
    compute_quantile_relative_error,
    compute_wasserstein1,
)
from mcmc_robust.targets.benign import IsotropicGaussian


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


# =============================================================================
# 1. Pathological Autocorrelations (Near-Unit and Negative)
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("rho", [0.90, 0.95, 0.99])
def test_bulk_ess_ar1_near_unit_convergence(rho, rng):
    """Near-unit AR(1) chains have Bulk ESS matching theoretical asymptotic limits."""
    N = 10000
    eps = rng.normal(size=N) * np.sqrt(1.0 - rho**2)
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + eps[t]

    ess = compute_bulk_ess(x)[0]
    theo_tau = (1.0 + rho) / (1.0 - rho)
    theo_ess = N / theo_tau

    assert np.isfinite(ess)
    assert ess > 0.0
    # Relative error against asymptotic AR(1) effective sample size within 35%
    rel_err = abs(ess - theo_ess) / theo_ess
    assert rel_err < 0.35


@pytest.mark.unit
def test_bulk_ess_ar1_monotonicity_under_persistence(rng):
    """Bulk ESS monotonically decreases as persistence rho increases towards 1."""
    N = 8000
    rhos = [0.0, 0.5, 0.8, 0.95, 0.99]
    ess_vals = []

    for rho in rhos:
        eps = rng.normal(size=N) * np.sqrt(1.0 - rho**2)
        x = np.zeros(N)
        for t in range(1, N):
            x[t] = rho * x[t - 1] + eps[t]
        ess_vals.append(compute_bulk_ess(x)[0])

    for i in range(len(ess_vals) - 1):
        assert ess_vals[i] > ess_vals[i + 1], (
            f"ESS at rho={rhos[i]} ({ess_vals[i]}) not > rho={rhos[i + 1]} ({ess_vals[i + 1]})"
        )


@pytest.mark.unit
@pytest.mark.parametrize("rho", [-0.2, -0.5, -0.8])
def test_bulk_ess_negative_autocorrelation_bounded(rho, rng):
    """For anti-correlated chains, Bulk ESS is safely bounded by sample size N."""
    N = 5000
    eps = rng.normal(size=N) * np.sqrt(1.0 - rho**2)
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + eps[t]

    ess = compute_bulk_ess(x)[0]
    tail = compute_tail_ess(x)[0]

    assert np.isfinite(ess)
    assert np.isfinite(tail)
    assert ess <= N
    assert tail > 1000.0


# =============================================================================
# 2. Infinite Variance Heavy-Tailed Chains
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("gamma", [0.1, 1.0, 50.0])
def test_bulk_ess_cauchy_scale_invariance(gamma, rng):
    """Rank transformation ensures Bulk ESS on Cauchy is invariant to Cauchy scale."""
    N = 5000
    cauchy_samples = gamma * rng.standard_cauchy(size=(N, 1))

    bulk_ess = compute_bulk_ess(cauchy_samples)[0]
    tail_ess = compute_tail_ess(cauchy_samples)[0]

    assert np.isfinite(bulk_ess)
    assert np.isfinite(tail_ess)
    # Independent draws should achieve >= 70% efficiency
    assert bulk_ess > 0.70 * N
    assert tail_ess > 0.70 * N


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
def test_heavy_tailed_pareto_ess_stability(alpha, rng):
    """Pareto distributions across infinite-variance regimes (alpha <= 2) yield stable ESS."""
    N = 5000
    # Pareto Type II: X = (U^(-1/alpha) - 1)
    u = rng.uniform(0.0, 1.0, size=(N, 1))
    x = u ** (-1.0 / alpha) - 1.0

    bulk_ess = compute_bulk_ess(x)[0]
    tail_ess = compute_tail_ess(x)[0]

    assert np.isfinite(bulk_ess)
    assert np.isfinite(tail_ess)
    assert bulk_ess > 0.70 * N
    assert tail_ess > 0.70 * N


@pytest.mark.unit
@pytest.mark.parametrize("nu", [0.5, 1.0, 1.1, 1.5])
def test_heavy_tailed_student_t_infinite_variance_ess(nu, rng):
    """Student-t with fractional/sub-2 degrees of freedom preserves finite ESS."""
    N = 5000
    x = rng.standard_t(df=nu, size=(N, 1))

    bulk_ess = compute_bulk_ess(x)[0]
    tail_ess = compute_tail_ess(x)[0]

    assert np.isfinite(bulk_ess)
    assert np.isfinite(tail_ess)
    assert bulk_ess > 0.70 * N
    assert tail_ess > 0.70 * N


# =============================================================================
# 3. R-hat Duality: Folded (Scale) vs Split (Location)
# =============================================================================


@pytest.mark.unit
def test_rhat_statistical_duality(rng):
    """Proves the necessary complementarity of Split R-hat and Folded R-hat.

    Case A: Identical scale, separated symmetric modes (-3 vs +3).
            Standard Split R-hat detects separation, Folded R-hat is blind (~1.0).
    Case B: Identical location (mean 0), 5x variance discrepancy.
            Standard Split R-hat is blind (~1.0), Folded R-hat detects discrepancy.
    """
    N = 1500

    # Case A: Pure location separation
    c1_loc = rng.normal(-3.0, 1.0, size=(N, 1))
    c2_loc = rng.normal(+3.0, 1.0, size=(N, 1))
    r_split_A = compute_split_rhat([c1_loc, c2_loc])[0]
    r_folded_A = compute_folded_rhat([c1_loc, c2_loc])[0]

    assert r_split_A > 2.5, f"Split R-hat failed to detect separated modes: {r_split_A}"
    assert r_folded_A < 1.05, f"Folded R-hat unexpectedly triggered on symmetric modes: {r_folded_A}"

    # Case B: Pure scale discrepancy
    c1_scale = rng.normal(0.0, 1.0, size=(N, 1))
    c2_scale = rng.normal(0.0, 5.0, size=(N, 1))
    r_split_B = compute_split_rhat([c1_scale, c2_scale])[0]
    r_folded_B = compute_folded_rhat([c1_scale, c2_scale])[0]

    assert r_split_B < 1.05, f"Split R-hat falsely detected scale difference: {r_split_B}"
    assert r_folded_B > 1.25, f"Folded R-hat failed to detect scale discrepancy: {r_folded_B}"

    # Overall convergence guardrail must take max(split, folded)
    assert max(r_split_A, r_folded_A) > 2.0
    assert max(r_split_B, r_folded_B) > 1.20


@pytest.mark.unit
def test_split_rhat_trending_chains(rng):
    """Split R-hat detects non-stationarity even when all chains share the exact same trend."""
    N = 1000
    trend = np.linspace(0.0, 10.0, N)[:, np.newaxis]
    c1 = trend + rng.normal(0.0, 0.2, size=(N, 1))
    c2 = trend + rng.normal(0.0, 0.2, size=(N, 1))

    # Without splitting, cross-chain means are identical
    r_unsplit = compute_split_rhat([c1, c2], split_chains=False)[0]
    assert r_unsplit < 1.05

    # With splitting, early vs late half drift is caught immediately
    r_split = compute_split_rhat([c1, c2], split_chains=True)[0]
    assert r_split > 2.0


# =============================================================================
# 4. Invariance and Oracle Properties
# =============================================================================


@pytest.mark.unit
def test_multichain_permutation_invariance(rng):
    """Diagnostics are invariant under chain ordering permutation."""
    M, N, d = 4, 600, 3
    chains = rng.normal(size=(M, N, d))

    b_orig = compute_bulk_ess(chains)
    t_orig = compute_tail_ess(chains)
    r_orig = compute_split_rhat(chains)
    f_orig = compute_folded_rhat(chains)

    perm = [3, 1, 0, 2]
    chains_perm = chains[perm]

    b_perm = compute_bulk_ess(chains_perm)
    t_perm = compute_tail_ess(chains_perm)
    r_perm = compute_split_rhat(chains_perm)
    f_perm = compute_folded_rhat(chains_perm)

    np.testing.assert_allclose(b_orig, b_perm, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(t_orig, t_perm, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r_orig, r_perm, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(f_orig, f_perm, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_autocorrelation_destruction_by_time_shuffle(rng):
    """Oracle property: Shuffling time order in an AR(1) chain destroys correlation and restores ESS."""
    N = 4000
    rho = 0.95
    eps = rng.normal(size=N) * np.sqrt(1.0 - rho**2)
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + eps[t]

    ess_autocorr = compute_bulk_ess(x)[0]
    assert ess_autocorr < 250.0

    # Randomly shuffle temporal sequence
    x_shuffled = rng.permutation(x)
    ess_shuffled = compute_bulk_ess(x_shuffled)[0]
    assert ess_shuffled > 3000.0


# =============================================================================
# 5. Distribution Metrics Edge Cases
# =============================================================================


@pytest.mark.unit
def test_metrics_degenerate_constant_samples(rng):
    """Degenerate constant samples fail KS test with extreme significance."""
    target = IsotropicGaussian(d=2, std=1.0)
    const_samples = np.zeros((200, 2))

    res_ks = compute_ks_test(const_samples, target)
    assert res_ks["passed"] is False
    assert res_ks["min_pvalue"] < 1e-15

    w1 = compute_wasserstein1(const_samples, target, rng=rng)
    # E[|Z|] for standard normal is sqrt(2/pi) approx 0.7979
    np.testing.assert_allclose(w1, [np.sqrt(2.0 / np.pi)] * 2, rtol=0.10)


@pytest.mark.unit
def test_metrics_extreme_outlier_sensitivity(rng):
    """Outliers affect Wasserstein-1 linearly while KS test remains bounded by 1/N."""
    target = IsotropicGaussian(d=2, std=1.0)
    samples = rng.normal(size=(1000, 2))

    # Single extreme outlier at 10^9 along coordinate 0
    samples[0, 0] = 1e9

    ks_res = compute_ks_test(samples, target)
    w1 = compute_wasserstein1(samples, target, rng=rng)

    # KS statistic is bounded: 1 outlier shifts empirical CDF by <= 1/N
    assert ks_res["statistic"][0] < 0.05
    # Wasserstein-1 distance explodes with outlier: W1 approx 10^9 / 1000 = 10^6
    assert w1[0] > 5e5
    assert w1[1] < 0.10


@pytest.mark.unit
def test_quantile_relative_error_nonstandard_percentiles(rng):
    """Quantile relative errors on non-standard percentiles match ground truth."""
    target = IsotropicGaussian(d=2, std=1.0)
    samples = target.sample_ground_truth(10000, rng=rng)

    # 1st, 10th, 90th, 99th percentiles
    quantiles = (0.01, 0.10, 0.90, 0.99)
    q_errors = compute_quantile_relative_error(samples, target, quantiles=quantiles, rng=rng)

    for p in quantiles:
        assert p in q_errors
        assert np.all(q_errors[p] < 0.10)


# =============================================================================
# 6. Degenerate Boundaries and High Dimensions
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("n_samples", [1, 2, 3])
def test_short_chains_boundaries(n_samples):
    """Very short chains (N=1, 2, 3) do not crash or emit unhandled exceptions."""
    c = np.ones((2, n_samples, 2))
    b = compute_bulk_ess(c)
    t = compute_tail_ess(c)
    r = compute_split_rhat(c)
    f = compute_folded_rhat(c)

    assert b.shape == (2,)
    assert t.shape == (2,)
    assert r.shape == (2,)
    assert f.shape == (2,)


@pytest.mark.unit
def test_high_dimensional_d100_diagnostics(rng):
    """Diagnostics scale efficiently and correctly on high-dimensional d=100 chains."""
    M, N, d = 4, 500, 100
    chains = rng.normal(size=(M, N, d))

    b = compute_bulk_ess(chains)
    t = compute_tail_ess(chains)
    r = compute_split_rhat(chains)
    f = compute_folded_rhat(chains)

    assert b.shape == (100,)
    assert t.shape == (100,)
    assert r.shape == (100,)
    assert f.shape == (100,)

    assert np.all(b > 0.60 * (M * N))
    assert np.all(t > 0.60 * (M * N))
    assert np.all(np.abs(r - 1.0) < 0.05)
    assert np.all(np.abs(f - 1.0) < 0.05)
