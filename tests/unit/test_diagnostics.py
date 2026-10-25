"""Unit tests for statistical diagnostics: Bulk ESS, Tail ESS, R-hat, and metrics.

Tests Feature F19_DIAG_STATISTICAL_METRICS:
1. Rank-normalized Bulk ESS (Margossian et al. 2024 / Vehtari et al. 2021).
2. Indicator-based Tail ESS.
3. Split Gelman-Rubin R-hat and Folded R-hat.
4. 1D Marginal Kolmogorov-Smirnov test against analytical/empirical CDF.
5. 1D Marginal Wasserstein-1 distance.
6. Quantile relative estimation error against analytical ground truth.
7. Numerical edge cases (constant chains, short chains, zero variance, NaNs/Infs).
"""

import numpy as np
import pytest

from mcmc_robust.diagnostics.convergence import (
    compute_folded_rhat,
    compute_split_rhat,
)
from mcmc_robust.diagnostics.ess import (
    compute_bulk_ess,
    compute_mean_ess,
    compute_tail_ess,
)
from mcmc_robust.diagnostics.metrics import (
    compute_ks_test,
    compute_quantile_relative_error,
    compute_wasserstein1,
)
from mcmc_robust.targets.benign import IsotropicGaussian
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
)
from mcmc_robust.targets.roughness import QuarticTarget


@pytest.fixture
def rng():
    """Deterministic random generator."""
    return np.random.default_rng(42)


# =============================================================================
# 1. Bulk ESS Tests
# =============================================================================


@pytest.mark.unit
def test_bulk_ess_independent_samples(rng):
    """Bulk ESS on independent Gaussian samples approaches total sample count N."""
    N, d = 1000, 2
    samples = rng.normal(size=(N, d))
    ess = compute_bulk_ess(samples)
    assert isinstance(ess, np.ndarray)
    assert ess.shape == (d,)
    assert np.all(ess > 700.0)
    assert np.all(ess <= N + 50.0)


@pytest.mark.unit
def test_bulk_ess_correlated_ar1(rng):
    """Bulk ESS on AR(1) process matches theoretical effective sample size."""
    N = 4000
    rho = 0.8
    eps = rng.normal(size=N) * np.sqrt(1.0 - rho**2)
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + eps[t]

    ess = compute_bulk_ess(x)
    theo_tau = (1.0 + rho) / (1.0 - rho)  # 9.0
    theo_ess = N / theo_tau  # ~444.4

    # Allow 25% sampling variance
    rel_err = abs(ess[0] - theo_ess) / theo_ess
    assert rel_err < 0.25


@pytest.mark.unit
def test_bulk_ess_multichain(rng):
    """Bulk ESS across multiple independent chains pools correctly to M * N."""
    M, N, d = 4, 1000, 2
    chains = rng.normal(size=(M, N, d))
    ess = compute_bulk_ess(chains)
    assert isinstance(ess, np.ndarray)
    assert ess.shape == (d,)
    # Total samples = 4000
    assert np.all(ess > 2800.0)


@pytest.mark.unit
def test_bulk_ess_constant_zero_variance():
    """Constant chains have zero effective samples without division by zero."""
    x = np.full((500, 2), 3.14159)
    ess = compute_bulk_ess(x)
    assert np.all(ess == 0.0)


@pytest.mark.unit
def test_bulk_ess_short_chain():
    """Chains shorter than 4 iterations are safely returned without error."""
    x = np.array([1.0, 2.0, 3.0])
    ess = compute_bulk_ess(x)
    assert ess[0] == 3.0


@pytest.mark.unit
def test_bulk_ess_heavy_tailed_cauchy(rng):
    """Rank transformation prevents explosion on heavy-tailed Cauchy samples."""
    samples = rng.standard_cauchy(size=(2000, 1))
    ess = compute_bulk_ess(samples)
    assert np.isfinite(ess[0])
    assert ess[0] > 1000.0


def _ar1_chains(rng: np.random.Generator, phi: float, chains: int, draws: int) -> np.ndarray:
    """Stationary AR(1) chains with N(0, 1) marginals, shape (chains, draws)."""
    x = np.empty((chains, draws))
    x[:, 0] = rng.normal(size=chains)
    eps = rng.normal(size=(chains, draws)) * np.sqrt(1.0 - phi**2)
    for t in range(1, draws):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x


@pytest.mark.unit
def test_mean_ess_is_the_ess_of_the_mean_of_autocorrelated_chains(rng):
    """ESS for the mean (no rank normalisation) matches the AR(1) value n (1 - phi) / (1 + phi) and ArviZ.

    Over 20 replicates of 4 chains x 2,000 draws, the largest relative error against the AR(1) value was 7% at
    phi = 0 and 12% at phi = 0.5 (tolerance 20%; at phi = 0.9 the estimator's own noise reaches 32%, so only ArviZ
    is compared there), and against ArviZ's estimator on the same draws 5%, 2% and 10% (tolerance 15%).
    """
    import arviz as az

    for phi in (0.0, 0.5, 0.9):
        x = _ar1_chains(rng, phi, 4, 2000)
        ess = compute_mean_ess(x[:, :, np.newaxis])
        assert ess.shape == (1,)
        assert ess[0] <= x.size
        if phi < 0.9:
            assert ess[0] == pytest.approx(x.size * (1.0 - phi) / (1.0 + phi), rel=0.2)
        assert ess[0] == pytest.approx(float(az.ess(x, method="mean")), rel=0.15)


@pytest.mark.unit
def test_mean_ess_collapses_when_chains_disagree_or_drift(rng):
    """Split chains with different means (a chain elsewhere, or a trend) leave few effective draws.

    Over 20 replicates (AR(1), phi = 0.5, 4 chains x 200 draws), ESS / n was at most 0.009 with one chain shifted
    by 3 sd and at most 0.024 with a drift of 3 sd, against at least 0.26 for the same chains unshifted (the AR(1)
    value is 1/3).
    """
    x = _ar1_chains(rng, 0.5, 4, 200)
    shifted = x + np.array([0.0, 0.0, 0.0, 3.0])[:, np.newaxis]
    drifting = x + np.linspace(0.0, 3.0, 200)[np.newaxis, :]
    for bad in (shifted, drifting):
        assert compute_mean_ess(bad[:, :, np.newaxis])[0] < 0.05 * bad.size
    assert compute_mean_ess(x[:, :, np.newaxis])[0] > 0.2 * x.size


@pytest.mark.unit
def test_mean_ess_is_affine_invariant_and_handles_degenerate_input(rng):
    x = _ar1_chains(rng, 0.7, 4, 300)[:, :, np.newaxis]
    np.testing.assert_allclose(compute_mean_ess(-3.0 * x + 7.0), compute_mean_ess(x), rtol=1e-10)
    # It is not rank-normalised: a monotone map that is not affine changes it.
    assert compute_mean_ess(np.exp(3.0 * x))[0] != pytest.approx(compute_mean_ess(x)[0], rel=0.01)
    assert compute_mean_ess(np.full((2, 50, 1), 1.5))[0] == 0.0
    with_nan = x.copy()
    with_nan[0, 3, 0] = np.nan
    assert np.isnan(compute_mean_ess(with_nan)[0])
    assert isinstance(compute_mean_ess(x, reduction="min"), float)


# =============================================================================
# 2. Tail ESS Tests
# =============================================================================


@pytest.mark.unit
def test_tail_ess_independent_samples(rng):
    """Tail ESS on independent samples exceeds 500 for N=1000."""
    samples = rng.normal(size=(1000, 2))
    tail_ess = compute_tail_ess(samples)
    assert isinstance(tail_ess, np.ndarray)
    assert tail_ess.shape == (2,)
    assert np.all(tail_ess > 500.0)


@pytest.mark.unit
def test_tail_ess_custom_quantiles(rng):
    """Tail ESS with extreme tail thresholds (prob_low=0.01, prob_high=0.99)."""
    samples = rng.normal(size=(2000, 1))
    tail_ess = compute_tail_ess(samples, prob_low=0.01, prob_high=0.99)
    assert np.isfinite(tail_ess[0])
    assert tail_ess[0] > 500.0


@pytest.mark.unit
def test_tail_ess_constant_chain():
    """Tail ESS on constant chain returns 0.0 cleanly."""
    samples = np.ones((100, 2))
    tail_ess = compute_tail_ess(samples)
    assert np.all(tail_ess == 0.0)


# =============================================================================
# 3. Split R-hat and Folded R-hat Tests
# =============================================================================


@pytest.mark.unit
def test_split_rhat_identical_stationary_chains(rng):
    """Split R-hat on independent stationary chains is close to 1.0."""
    c1 = rng.normal(size=(500, 2))
    c2 = rng.normal(size=(500, 2))
    rhat = compute_split_rhat([c1, c2])
    assert isinstance(rhat, np.ndarray)
    assert rhat.shape == (2,)
    assert np.all(np.abs(rhat - 1.0) < 0.05)


@pytest.mark.unit
def test_split_rhat_separated_modes(rng):
    """Split R-hat on chains trapped in separated modes flags non-convergence."""
    c1 = rng.normal(loc=5.0, size=(500, 1))
    c2 = rng.normal(loc=-5.0, size=(500, 1))
    rhat = compute_split_rhat([c1, c2])
    assert rhat[0] > 2.0


@pytest.mark.unit
def test_split_rhat_single_chain_drift():
    """Single drifting chain split in half detects non-stationarity."""
    drifting_chain = np.linspace(-10.0, 10.0, 1000)[:, np.newaxis]
    rhat = compute_split_rhat(drifting_chain)
    assert rhat[0] > 1.5


@pytest.mark.unit
def test_split_rhat_constant_chains():
    """Constant chains edge cases: identical constant gives 1.0, distinct gives inf."""
    identical = [np.ones(100), np.ones(100)]
    assert compute_split_rhat(identical)[0] == 1.0

    distinct = [np.zeros(100), np.ones(100)]
    assert np.isinf(compute_split_rhat(distinct)[0])


@pytest.mark.unit
def test_folded_rhat_detects_scale_discrepancy(rng):
    """Folded split R-hat detects scale differences where standard R-hat fails."""
    # Chains with identical mean 0.0, but 3x difference in scale
    c1 = rng.normal(loc=0.0, scale=1.0, size=(1500, 1))
    c2 = rng.normal(loc=0.0, scale=3.0, size=(1500, 1))

    rhat_standard = compute_split_rhat([c1, c2])
    rhat_folded = compute_folded_rhat([c1, c2])

    # Standard R-hat fails to detect variance difference
    assert rhat_standard[0] < 1.05
    # Folded R-hat immediately detects variance discrepancy
    assert rhat_folded[0] > 1.15


# =============================================================================
# 4. Statistical Metrics (KS test, Wasserstein-1, Quantile Error)
# =============================================================================


@pytest.mark.unit
def test_ks_test_analytical_gaussian(rng):
    """1-sample KS test passes on exact Gaussian draws."""
    target = IsotropicGaussian(d=2, std=1.0)
    samples = target.sample_ground_truth(2000, rng=rng)
    res = compute_ks_test(samples, target)
    assert res["passed"] is True
    assert res["min_pvalue"] > 0.01
    assert res["statistic"].shape == (2,)


@pytest.mark.unit
def test_ks_test_biased_samples(rng):
    """1-sample KS test rejects biased samples with high significance."""
    target = IsotropicGaussian(d=2, std=1.0)
    biased_samples = rng.normal(loc=1.5, scale=1.0, size=(1000, 2))
    res = compute_ks_test(biased_samples, target)
    assert res["passed"] is False
    assert res["min_pvalue"] < 1e-10


@pytest.mark.unit
def test_ks_test_student_t(rng):
    """1-sample KS test against exact Student-t marginal CDF."""
    target = MultivariateStudentTTarget(d=2, nu=2.5)
    samples = target.sample_ground_truth(3000, rng=rng)
    res = compute_ks_test(samples, target)
    assert res["passed"] is True
    assert res["min_pvalue"] > 0.01


@pytest.mark.unit
def test_ks_test_heterogeneous(rng):
    """KS test on HeterogeneousHeavyTailedTarget validates mixed coordinates."""
    target = HeterogeneousHeavyTailedTarget(d=4, heavy_coords=[0, 1], nu_list=[2.5, 2.5])
    samples = target.sample_ground_truth(3000, rng=rng)
    res = compute_ks_test(samples, target)
    assert res["passed"] is True
    assert res["min_pvalue"] > 0.01


@pytest.mark.unit
def test_ks_test_two_sample_fallback(rng):
    """QuarticTarget without closed-form CDF uses 2-sample KS test."""
    target = QuarticTarget(d=2, scale=1.0)
    samples = target.sample_ground_truth(2000, rng=rng)
    res = compute_ks_test(samples, target, n_ref=5000, rng=rng)
    assert res["passed"] is True


@pytest.mark.unit
def test_wasserstein1_distance(rng):
    """Wasserstein-1 distance is near zero for true samples and ~offset for shifted."""
    target = IsotropicGaussian(d=2, std=1.0)
    samples = target.sample_ground_truth(3000, rng=rng)
    w1 = compute_wasserstein1(samples, target, n_ref=5000, rng=rng)
    assert isinstance(w1, np.ndarray)
    assert w1.shape == (2,)
    assert np.all(w1 < 0.06)

    # Shifted samples by 2.0 along dimension 0
    shifted = samples.copy()
    shifted[:, 0] += 2.0
    w1_shifted = compute_wasserstein1(shifted, target, n_ref=5000, rng=rng)
    assert abs(w1_shifted[0] - 2.0) < 0.10


@pytest.mark.unit
def test_quantile_relative_error_exact_moments(rng):
    """Quantile relative error calculation against analytical moments."""
    target = IsotropicGaussian(d=2, std=1.0)
    samples = target.sample_ground_truth(5000, rng=rng)
    q_errors = compute_quantile_relative_error(samples, target, quantiles=(0.05, 0.95, 0.99))

    assert 0.05 in q_errors
    assert 0.95 in q_errors
    assert 0.99 in q_errors
    assert np.all(q_errors[0.05] < 0.05)
    assert np.all(q_errors[0.95] < 0.05)
    assert np.all(q_errors[0.99] < 0.08)


# =============================================================================
# 5. Polymorphic Input and Reduction Tests
# =============================================================================


@pytest.mark.unit
def test_input_shape_polymorphism(rng):
    """Diagnostics accept 1D (N,), 2D (N, d), 3D (M, N, d), and lists of chains."""
    s1d = rng.normal(size=500)
    s2d = rng.normal(size=(500, 3))
    s3d = rng.normal(size=(2, 500, 3))
    slist = [rng.normal(size=(500, 3)), rng.normal(size=(500, 3))]

    for arr in [s1d, s2d, s3d, slist]:
        b = compute_bulk_ess(arr)
        t = compute_tail_ess(arr)
        r = compute_split_rhat(arr)
        f = compute_folded_rhat(arr)
        assert isinstance(b, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(f, np.ndarray)


@pytest.mark.unit
def test_reduction_options(rng):
    """Test scalar reductions 'min', 'max', 'mean'."""
    samples = rng.normal(size=(1000, 4))
    b_min = compute_bulk_ess(samples, reduction="min")
    assert isinstance(b_min, float)

    t_mean = compute_tail_ess(samples, reduction="mean")
    assert isinstance(t_mean, float)

    r_max = compute_split_rhat([samples, samples], reduction="max")
    assert isinstance(r_max, float)
    assert abs(r_max - 1.0) < 0.05
