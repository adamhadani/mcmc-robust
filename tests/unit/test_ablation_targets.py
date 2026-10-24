"""Tests for the ablation target registry and its Gaussian scale-mixture reparameterisations."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from mcmc_robust.bench.sota_adapters import SOTATargetAdapter, SparseSignalRegressionTarget
from mcmc_robust.jax_unified_nuts import sample_blackjax_unified_nuts
from mcmc_robust.targets.ablation import (
    ABLATION_TARGET_NAMES,
    AblationTarget,
    make_ablation_target,
    rotated_spd_matrix,
)
from mcmc_robust.targets.benign import DiagonalGaussian, FullCovarianceGaussian
from mcmc_robust.targets.heavy_tailed import (
    HeterogeneousHeavyTailedTarget,
    MultivariateStudentTTarget,
    RotatedProductCauchyTarget,
)
from mcmc_robust.targets.scale_mixture import ScaleMixtureModel

HEAVY = ("task1", "task3", "mvt_rot_nu1", "mvt_rot_nu2", "mvt_iso_nu2", "rot_prod_cauchy")
BENIGN = ("gauss_rot", "gauss_aniso")


@pytest.mark.unit
def test_registry_names_classes_and_dimensions() -> None:
    """Every name builds the specified target class, dimension and scale-mixture availability."""
    assert set(ABLATION_TARGET_NAMES) == set(HEAVY) | set(BENIGN)
    expected = {
        "task1": (SparseSignalRegressionTarget, 10),
        "task3": (HeterogeneousHeavyTailedTarget, 20),
        "mvt_rot_nu1": (MultivariateStudentTTarget, 10),
        "mvt_rot_nu2": (MultivariateStudentTTarget, 10),
        "mvt_iso_nu2": (MultivariateStudentTTarget, 10),
        "rot_prod_cauchy": (RotatedProductCauchyTarget, 10),
        "gauss_rot": (FullCovarianceGaussian, 20),
        "gauss_aniso": (DiagonalGaussian, 20),
    }
    for name in ABLATION_TARGET_NAMES:
        spec = make_ablation_target(name)
        assert isinstance(spec, AblationTarget)
        cls, d = expected[name]
        assert isinstance(spec.target, cls)
        assert spec.d == spec.target.d == d
        assert spec.heavy_tailed == (name in HEAVY)
        assert (spec.scale_mixture is not None) == (name in HEAVY)
        assert len(spec.marginals) == d
    with pytest.raises(KeyError, match="Unknown ablation target"):
        make_ablation_target("nope")


@pytest.mark.unit
def test_target_parameters_match_the_specification() -> None:
    """Task 1/3 settings, rotated scale spectra, the isotropic t and the Gaussian spectra are as specified."""
    t1 = make_ablation_target("task1").target
    assert isinstance(t1, SparseSignalRegressionTarget)
    assert (t1.n_obs, t1.n_active, t1.prior_scale) == (40, 3, 1.0)

    t3 = make_ablation_target("task3").target
    assert isinstance(t3, HeterogeneousHeavyTailedTarget)
    assert t3.heavy_coords == list(range(10))
    np.testing.assert_allclose(t3.nu_arr, np.linspace(1.2, 2.5, 10))

    for name, nu in (("mvt_rot_nu1", 1.0), ("mvt_rot_nu2", 2.0)):
        t = make_ablation_target(name).target
        assert isinstance(t, MultivariateStudentTTarget)
        assert t.nu == nu
        np.testing.assert_allclose(np.linalg.eigvalsh(t.cov), np.geomspace(1.0, 25.0, 10), rtol=1e-10)
        # Rotated: the scale matrix is far from diagonal.
        off = t.cov - np.diag(np.diag(t.cov))
        assert np.max(np.abs(off)) > 1.0
    # Both rotated t targets share the same fixed rotation.
    np.testing.assert_array_equal(
        make_ablation_target("mvt_rot_nu1").target.cov, make_ablation_target("mvt_rot_nu2").target.cov
    )

    iso = make_ablation_target("mvt_iso_nu2").target
    assert isinstance(iso, MultivariateStudentTTarget)
    assert iso.nu == 2.0
    np.testing.assert_array_equal(iso.cov, np.eye(10))

    g = make_ablation_target("gauss_rot").target
    eig = np.linalg.eigvalsh(g.cov)
    np.testing.assert_allclose(eig.max() / eig.min(), 100.0, rtol=1e-10)

    ga = make_ablation_target("gauss_aniso").target
    assert isinstance(ga, DiagonalGaussian)
    np.testing.assert_allclose(ga.stds, np.geomspace(0.1, 10.0, 20))


@pytest.mark.unit
def test_rotated_spd_matrix_is_reproducible_and_has_the_requested_spectrum() -> None:
    """The fixed-seed rotation gives the same matrix every time and the requested eigenvalues."""
    eig = np.geomspace(1.0, 7.0, 6)
    a = rotated_spd_matrix(eig, seed=0)
    b = rotated_spd_matrix(eig, seed=0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, a.T)
    np.testing.assert_allclose(np.linalg.eigvalsh(a), eig, rtol=1e-12)
    assert not np.allclose(rotated_spd_matrix(eig, seed=1), a)


@pytest.mark.unit
@pytest.mark.parametrize("name", ABLATION_TARGET_NAMES)
def test_exact_marginal_quantiles_agree_with_the_target(name: str) -> None:
    """marginal_ppf agrees with the target's own quantiles, inverts marginal_cdf, and matches exact draws."""
    spec = make_ablation_target(name)
    exact = spec.target.exact_moments_or_quantiles()
    for p, key in ((0.05, "q05"), (0.95, "q95"), (0.99, "q99")):
        np.testing.assert_allclose(spec.marginal_ppf(p), exact[key], rtol=1e-6, atol=1e-8)
    for p in (0.01, 0.2, 0.5, 0.99):
        q = spec.marginal_ppf(p)
        np.testing.assert_allclose(spec.marginal_cdf(q[np.newaxis, :])[0], np.full(spec.d, p), atol=1e-6)

    # Exact iid draws from the target: the empirical CDF at q_p is within 5 binomial SEs of p.
    n = 40_000
    draws = spec.target.sample_ground_truth(n, rng=np.random.default_rng(7))
    for p in (0.01, 0.99):
        frac = np.mean(draws < spec.marginal_ppf(p), axis=0)
        assert np.all(np.abs(frac - p) < 5.0 * np.sqrt(p * (1 - p) / n)), (name, p, frac)


@pytest.mark.unit
def test_student_t_marginals_are_univariate_t_with_scale_sqrt_sigma_ii() -> None:
    """Marginal i of a multivariate t(nu, mu, Sigma) is t(nu) with location mu_i and scale sqrt(Sigma_ii)."""
    spec = make_ablation_target("mvt_rot_nu2")
    t = spec.target
    assert isinstance(t, MultivariateStudentTTarget)
    scale = np.sqrt(np.diag(t.cov))
    np.testing.assert_allclose(spec.marginal_ppf(0.01), t.mu + scale * stats.t.ppf(0.01, df=2.0), rtol=1e-12)


# =============================================================================
# Scale-mixture reparameterisation
# =============================================================================


def _invgamma_log_u(u: np.ndarray, a: float, b: float) -> np.ndarray:
    """log density of u = log(lambda) when lambda ~ InvGamma(a, b), via scipy."""
    return stats.invgamma.logpdf(np.exp(u), a, scale=b) + u


@pytest.mark.unit
@pytest.mark.parametrize("name", HEAVY)
def test_scale_mixture_log_density_formula(name: str) -> None:
    """The augmented log density is N(eta) x log-InvGamma(u) (x the likelihood for Task 1), term by term."""
    spec = make_ablation_target(name)
    sm = spec.scale_mixture
    assert isinstance(sm, ScaleMixtureModel)
    assert sm.d == spec.d
    assert sm.d_aug == spec.d + sm.n_mix
    rng = np.random.default_rng(11)
    for _ in range(3):
        theta = rng.normal(size=sm.d_aug)
        eta, u = theta[: sm.d], theta[sm.d :]
        expected = float(np.sum(stats.norm.logpdf(eta)))
        expected += float(sum(_invgamma_log_u(u[k], sm.nu[k] / 2, sm.nu[k] / 2) for k in range(sm.n_mix)))
        if name == "task1":
            t1 = spec.target
            assert isinstance(t1, SparseSignalRegressionTarget)
            beta = np.asarray(sm.to_x(jnp.asarray(theta)))
            # Likelihood = full posterior minus the Cauchy prior (the target's own NumPy code).
            prior = float(np.sum(stats.cauchy.logpdf(beta, scale=t1.prior_scale)))
            expected += t1.log_prob(beta) - prior
        np.testing.assert_allclose(float(sm.logdensity(jnp.asarray(theta))), expected, rtol=1e-10)


@pytest.mark.unit
def test_scale_mixture_maps_and_groups() -> None:
    """Mixing groups follow the target structure and init_from_x maps x0 to an exact preimage with u = 0."""
    groups = {
        "task1": np.arange(10),
        "task3": np.r_[np.arange(10), np.full(10, -1)],
        "mvt_rot_nu1": np.zeros(10, dtype=int),
        "mvt_rot_nu2": np.zeros(10, dtype=int),
        "mvt_iso_nu2": np.zeros(10, dtype=int),
    }
    nus = {"task1": np.ones(10), "task3": np.linspace(1.2, 2.5, 10), "mvt_rot_nu1": [1.0], "mvt_rot_nu2": [2.0]}
    for name, grp in groups.items():
        sm = make_ablation_target(name).scale_mixture
        assert sm is not None
        np.testing.assert_array_equal(sm.group, grp)
        if name in nus:
            np.testing.assert_allclose(sm.nu, nus[name])
        x0 = np.random.default_rng(3).uniform(-2.0, 2.0, size=(4, sm.d))
        theta0 = sm.init_from_x(x0)
        assert theta0.shape == (4, sm.d_aug)
        np.testing.assert_array_equal(theta0[:, sm.d :], 0.0)
        np.testing.assert_allclose(np.asarray(jax.vmap(sm.to_x)(jnp.asarray(theta0))), x0, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("name", HEAVY)
def test_scale_mixture_prior_marginal_is_exact(name: str) -> None:
    """Direct draws of the augmented prior pushed to x have the exact t (or Cauchy-prior) marginals (KS test)."""
    spec = make_ablation_target(name)
    sm = spec.scale_mixture
    assert sm is not None
    theta = sm.sample_prior(20_000, rng=np.random.default_rng(2024))
    x = np.asarray(jax.vmap(sm.to_x)(jnp.asarray(theta)))
    if name == "task1":
        t1 = spec.target
        assert isinstance(t1, SparseSignalRegressionTarget)
        cdfs = [stats.cauchy(scale=t1.prior_scale).cdf] * spec.d
    else:
        cdfs = [m.cdf for m in spec.marginals]
    p_values = np.array([stats.kstest(x[:, i], cdfs[i]).pvalue for i in range(spec.d)])
    # Bonferroni over coordinates at family-wise level 1e-3.
    assert np.all(p_values > 1e-3 / spec.d), (name, p_values)


@pytest.mark.unit
def test_scale_mixture_invgamma_mixing_density_is_normalised() -> None:
    """exp(log-InvGamma(u)) integrates to 1 over u for the extreme Task 3 / Cauchy shape parameters."""
    sm = make_ablation_target("task3").scale_mixture
    assert sm is not None
    grid = np.linspace(-40.0, 120.0, 400_001)
    for nu in (1.0, 1.2, 2.5):
        a = b = nu / 2.0
        log_dens = a * np.log(b) - special.gammaln(a) - a * grid - b * np.exp(-grid)
        np.testing.assert_allclose(np.trapezoid(np.exp(log_dens), grid), 1.0, rtol=1e-6)
        np.testing.assert_allclose(log_dens[::40_000], _invgamma_log_u(grid[::40_000], a, b), rtol=1e-9, atol=1e-9)


def _tail_prob_z(x: np.ndarray, quantiles: dict[float, np.ndarray]) -> np.ndarray:
    """z-scores (P_hat - p) / MCSE of the indicator 1{x_i < q_p}, with ArviZ's MCSE."""
    zs = []
    for p, q in quantiles.items():
        ind = (x < q).astype(np.float64)
        zs.append((ind.mean(axis=(0, 1)) - p) / az.mcse(ind, method="mean"))
    return np.concatenate(zs)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["task1", "mvt_rot_nu1"])
def test_scale_mixture_long_run_nuts_matches_exact_marginals(name: str) -> None:
    """NUTS on the augmented model, mapped to x, hits the exact marginal tail and central probabilities.

    Tolerances are in Monte Carlo standard errors of the indicator means: with 40 to 60 z-scores
    the maximum |z| of a correct sampler exceeds 4.5 with probability below 1e-3.
    """
    spec = make_ablation_target(name)
    sm = spec.scale_mixture
    assert sm is not None
    x0 = np.random.default_rng(0).uniform(-2.0, 2.0, size=(4, sm.d))
    res = sample_blackjax_unified_nuts(
        sm.logdensity,
        d=sm.d_aug,
        draws=2500,
        tune=1000,
        chains=4,
        seed=5,
        x0=sm.init_from_x(x0),
        transform_kind="identity",
    )
    x = np.asarray(jax.vmap(jax.vmap(sm.to_x))(jnp.asarray(res.samples)))
    assert x.shape == (4, 2500, spec.d)
    z = _tail_prob_z(x, {p: spec.marginal_ppf(p) for p in (0.01, 0.1, 0.5, 0.9, 0.99)})
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z)) < 4.5, (name, z)
    assert np.mean(np.abs(z)) < 1.5, (name, z)


@pytest.mark.unit
def test_task1_scale_mixture_likelihood_matches_the_jax_posterior() -> None:
    """For Task 1, logdensity(theta) - mixing priors equals the JAX posterior minus the Cauchy prior."""
    spec = make_ablation_target("task1")
    sm = spec.scale_mixture
    assert sm is not None
    t1 = spec.target
    assert isinstance(t1, SparseSignalRegressionTarget)
    post = SOTATargetAdapter(t1).jax_logdensity_fn()
    theta = jnp.asarray(np.random.default_rng(9).normal(size=sm.d_aug))
    beta = sm.to_x(theta)
    eta, u = theta[: sm.d], theta[sm.d :]
    a = b = 0.5
    mix = jnp.sum(-0.5 * eta**2 - 0.5 * jnp.log(2 * jnp.pi)) + jnp.sum(
        a * jnp.log(b) - special.gammaln(a) - a * u - b * jnp.exp(-u)
    )
    cauchy = jnp.sum(-jnp.log(jnp.pi * t1.prior_scale) - jnp.log1p((beta / t1.prior_scale) ** 2))
    np.testing.assert_allclose(float(sm.logdensity(theta) - mix), float(post(beta) - cauchy), rtol=1e-10)


@pytest.mark.unit
def test_rotated_product_cauchy_density_gradient_and_jax_agree() -> None:
    """x = A u with u iid Cauchy: log density, its gradient and the JAX log density agree with the definition."""
    spec = make_ablation_target("rot_prod_cauchy")
    t = spec.target
    assert isinstance(t, RotatedProductCauchyTarget)
    rng = np.random.default_rng(3)
    for x in rng.standard_cauchy(size=(5, t.d)):
        u = np.linalg.solve(t.A, x)
        expected = float(np.sum(stats.cauchy.logpdf(u)) - np.linalg.slogdet(t.A)[1])
        lp, grad = t.log_prob_and_grad(x)
        np.testing.assert_allclose(lp, expected, rtol=1e-12)
        np.testing.assert_allclose(float(t.jax_log_prob(jnp.asarray(x))), expected, rtol=1e-12)
        np.testing.assert_allclose(grad, np.asarray(jax.grad(t.jax_log_prob)(jnp.asarray(x))), rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_rotated_product_cauchy_marginals_are_cauchy_with_l1_row_scales() -> None:
    """Marginal i of A u (u iid Cauchy) is Cauchy with scale sum_j |A_ij| (stability of the Cauchy law)."""
    spec = make_ablation_target("rot_prod_cauchy")
    t = spec.target
    for i, m in enumerate(spec.marginals):
        np.testing.assert_allclose(m.ppf(0.75), np.sum(np.abs(t.A[i])), rtol=1e-12)


@pytest.mark.unit
def test_scale_mixture_accepts_a_general_invertible_scale_matrix() -> None:
    """A non-triangular scale matrix is allowed, and init_from_x maps exactly back to x."""
    a = np.array([[2.0, 1.0], [-0.5, 1.5]])
    sm = ScaleMixtureModel(loc=[1.0, -1.0], chol=a, nu=[1.0, 2.0], group=[0, 1])
    x0 = np.array([[0.3, 4.0], [-7.0, 2.5]])
    theta = sm.init_from_x(x0)
    np.testing.assert_allclose(np.asarray(jax.vmap(sm.to_x)(jnp.asarray(theta))), x0, rtol=1e-12)
    with pytest.raises(ValueError, match="invertible"):
        ScaleMixtureModel(loc=[0.0, 0.0], chol=np.ones((2, 2)), nu=[1.0], group=[0, 0])
