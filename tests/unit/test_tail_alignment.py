"""Tests for the tail-alignment diagnostic (`mcmc_robust.diagnostics.tail_alignment`).

Exact iid draws (n = 4000) carry the statistical tests. Margins quoted below were measured with the
same construction over seeds 0-199: the rotated product of Cauchys gave a maximum co-exceedance of at
least 0.71, the independent product of Student-t and Gaussian coordinates at most 0.13 (with exactly
the ten t coordinates classed heavy on every seed), and a Gaussian with correlation 0.95 never had a
coordinate classed heavy (smallest Hill estimate 4.93, against the cut-off 4).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mcmc_robust.diagnostics.tail_alignment import (
    TailAlignmentReport,
    TailAlignmentWarning,
    tail_alignment_diagnostic,
)
from mcmc_robust.jax_unified_nuts import sample_blackjax_unified_nuts
from mcmc_robust.targets.heavy_tailed import HeterogeneousHeavyTailedTarget, RotatedProductCauchyTarget

N_DRAWS = 4000
T_DFS = np.linspace(1.2, 2.5, 10)


def _rotated_cauchy_product(seed: int) -> np.ndarray:
    return RotatedProductCauchyTarget(d=10).sample_ground_truth(N_DRAWS, rng=np.random.default_rng(seed))


def _product_t_and_gaussian(seed: int) -> np.ndarray:
    """Ten independent t coordinates (nu = 1.2 to 2.5) and ten independent N(0, 1) coordinates."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.standard_t(T_DFS, size=(N_DRAWS, 10)), rng.normal(size=(N_DRAWS, 10))], axis=1)


def _equicorrelated_gaussian(seed: int, rho: float = 0.95, d: int = 10) -> np.ndarray:
    corr = np.full((d, d), rho)
    np.fill_diagonal(corr, 1.0)
    return np.random.default_rng(seed).normal(size=(N_DRAWS, d)) @ np.linalg.cholesky(corr).T


# =============================================================================
# 1. Statistical behaviour on exact iid draws
# =============================================================================


@pytest.mark.unit
@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotated_product_of_cauchys_fires(seed: int) -> None:
    """Tails along rotated columns make every coordinate extreme together: the check must warn."""
    report = tail_alignment_diagnostic(_rotated_cauchy_product(seed))
    assert report.warn
    assert report.heavy_coordinates == tuple(range(10))
    # Threshold 0.3; iid values over 200 seeds were >= 0.71.
    assert report.max_chi > 0.6
    assert report.offending_pairs, "a warning must name the coordinate pairs"


@pytest.mark.unit
@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_independent_product_of_student_t_does_not_fire(seed: int) -> None:
    """Axis-aligned heavy tails (a product form) are what coordinate-wise transforms handle: no warning."""
    report = tail_alignment_diagnostic(_product_t_and_gaussian(seed))
    assert not report.warn
    assert report.heavy_coordinates == tuple(range(10))
    # Independence gives about 0.05; iid values over 200 seeds were <= 0.13.
    assert report.max_chi < 0.2
    assert report.offending_pairs == ()


@pytest.mark.unit
@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_correlated_gaussian_has_no_heavy_coordinates_and_does_not_fire(seed: int) -> None:
    """Strong linear correlation co-exceeds at a finite level, but light tails are not this check's concern."""
    x = _equicorrelated_gaussian(seed)
    report = tail_alignment_diagnostic(x)
    assert not report.warn
    assert report.heavy_coordinates == ()
    assert math.isnan(report.max_chi)
    assert np.all(report.tail_index > 4.5)
    # Without the heaviness gate this pair would be flagged: its raw co-exceedance is far above 0.3.
    a = np.abs(x[:, :2] - np.median(x[:, :2], axis=0))
    exceed = a > np.quantile(a, 0.95, axis=0)
    assert np.mean(exceed[:, 0] & exceed[:, 1]) / 0.05 > 0.5


@pytest.mark.unit
def test_identical_heavy_coordinates_have_co_exceedance_one() -> None:
    """Two copies of one Cauchy column exceed together every time, so chi = 1 exactly; a third independent one does not."""
    rng = np.random.default_rng(7)
    c = rng.standard_cauchy(N_DRAWS)
    x = np.column_stack([c, c, rng.standard_cauchy(N_DRAWS)])
    report = tail_alignment_diagnostic(x)
    assert report.chi[0, 1] == 1.0
    assert report.chi[1, 0] == 1.0
    assert report.max_chi == 1.0
    assert report.offending_pairs == ((0, 1),)
    assert report.warn


@pytest.mark.unit
@pytest.mark.statistical
def test_independent_heavy_coordinates_give_about_one_minus_level() -> None:
    """Under independence P(both exceed) / (1 - u) = 1 - u; across seeds 0-199 the n = 40000 value stays in [0.038, 0.069]."""
    x = np.random.default_rng(3).standard_cauchy(size=(40000, 2))
    report = tail_alignment_diagnostic(x)
    assert report.level == 0.95
    assert abs(report.chi[0, 1] - 0.05) < 0.025
    assert not report.warn


@pytest.mark.unit
@pytest.mark.statistical
def test_hill_tail_index_separates_cauchy_from_gaussian() -> None:
    """Hill estimates: near 1 for Cauchy coordinates, well above the cut-off 4 for Gaussian ones."""
    rng = np.random.default_rng(11)
    x = np.column_stack([rng.standard_cauchy(size=(N_DRAWS, 3)), rng.normal(size=(N_DRAWS, 3))])
    report = tail_alignment_diagnostic(x)
    assert report.tail_index.shape == (6,)
    assert np.all((report.tail_index[:3] > 0.7) & (report.tail_index[:3] < 1.4))
    assert np.all(report.tail_index[3:] > 4.5)
    assert report.heavy_coordinates == (0, 1, 2)


# =============================================================================
# 2. Report schema and message
# =============================================================================


@pytest.mark.unit
def test_report_schema() -> None:
    """Every documented field is present with the documented type and shape."""
    report = tail_alignment_diagnostic(_rotated_cauchy_product(0))
    assert isinstance(report, TailAlignmentReport)
    assert isinstance(report.warn, bool)
    assert isinstance(report.max_chi, float)
    assert report.chi_threshold == 0.3
    assert report.level == 0.95
    assert report.max_tail_index == 4.0
    assert report.n_draws == N_DRAWS
    assert report.n_exceedances == 200
    assert report.tail_index.shape == (10,)
    assert all(isinstance(i, int) for i in report.heavy_coordinates)
    assert report.chi.shape == (10, 10)
    np.testing.assert_array_equal(report.chi, report.chi.T)
    assert np.all(np.isnan(np.diag(report.chi)))
    assert report.max_chi == float(np.nanmax(report.chi))
    # Offending pairs: i < j, heavy, above the threshold, largest chi first, and every such pair listed.
    chis = [report.chi[i, j] for i, j in report.offending_pairs]
    assert all(i < j for i, j in report.offending_pairs)
    assert all(c > report.chi_threshold for c in chis)
    assert chis == sorted(chis, reverse=True)
    assert chis[0] == report.max_chi
    above = int(np.sum(np.triu(report.chi, 1) > report.chi_threshold))
    assert len(report.offending_pairs) == above
    assert report.independence_chi == pytest.approx(0.05)


@pytest.mark.unit
def test_chi_is_nan_outside_heavy_pairs() -> None:
    """Only pairs of heavy coordinates are scored; Gaussian coordinates get NaN rows and columns."""
    report = tail_alignment_diagnostic(_product_t_and_gaussian(0))
    assert np.all(np.isfinite(report.chi[:10, :10][~np.eye(10, dtype=bool)]))
    assert np.all(np.isnan(report.chi[10:, :]))
    assert np.all(np.isnan(report.chi[:, 10:]))


@pytest.mark.unit
def test_warning_message_names_the_pair_and_says_it_is_a_risk() -> None:
    """The message names the worst pair and its value, points to REPORT.md 5.1, and says a miss is not proven."""
    report = tail_alignment_diagnostic(_rotated_cauchy_product(0))
    i, j = report.offending_pairs[0]
    assert f"coordinates {i} and {j}" in report.message
    assert f"{report.max_chi:.2f}" in report.message
    assert "REPORT.md" in report.message and "5.1" in report.message
    assert "risk" in report.message and "not a proven miss" in report.message


@pytest.mark.unit
def test_chains_are_pooled() -> None:
    """(chains, draws, d) input gives the same report as the pooled (chains * draws, d) array."""
    x = _rotated_cauchy_product(1)
    pooled = tail_alignment_diagnostic(x)
    split = tail_alignment_diagnostic(x.reshape(4, N_DRAWS // 4, 10))
    np.testing.assert_array_equal(pooled.chi, split.chi)
    np.testing.assert_array_equal(pooled.tail_index, split.tail_index)
    assert pooled.n_draws == split.n_draws == N_DRAWS


# =============================================================================
# 3. Edge cases
# =============================================================================


@pytest.mark.unit
def test_too_few_draws_gives_no_verdict() -> None:
    """Below `min_exceedances` exceedances per coordinate the check does not warn and says why."""
    x = _rotated_cauchy_product(0)[:400]  # 20 exceedances at level 0.95, below the default 50
    report = tail_alignment_diagnostic(x)
    assert not report.warn
    assert math.isnan(report.max_chi)
    assert report.n_exceedances == 20
    assert "too few draws" in report.message.lower()
    # A lower minimum lets the same draws be scored.
    assert tail_alignment_diagnostic(x, min_exceedances=20).warn


@pytest.mark.unit
def test_non_finite_draws_are_dropped() -> None:
    """Rows with a NaN or infinity are removed before scoring; n_draws counts the rows kept."""
    x = _rotated_cauchy_product(0)
    x_bad = x.copy()
    x_bad[[5, 17], 3] = np.nan
    x_bad[99, 0] = np.inf
    report = tail_alignment_diagnostic(x_bad)
    assert report.n_draws == N_DRAWS - 3
    np.testing.assert_array_equal(report.chi, tail_alignment_diagnostic(np.delete(x, [5, 17, 99], axis=0)).chi)


@pytest.mark.unit
def test_constant_coordinate_is_not_heavy_and_raises_no_numpy_warning() -> None:
    """A coordinate stuck at one value (e.g. a frozen chain) has no tail; no log(0) warning escapes."""
    rng = np.random.default_rng(5)
    x = np.column_stack([np.full(N_DRAWS, 2.5), rng.standard_cauchy(size=(N_DRAWS, 2))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = tail_alignment_diagnostic(x)
    assert 0 not in report.heavy_coordinates
    assert report.heavy_coordinates == (1, 2)


@pytest.mark.unit
def test_single_coordinate_has_no_pairs() -> None:
    report = tail_alignment_diagnostic(np.random.default_rng(0).standard_cauchy(size=(N_DRAWS, 1)))
    assert not report.warn
    assert math.isnan(report.max_chi)
    assert report.offending_pairs == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"level": 1.0}, "level"),
        ({"level": 0.0}, "level"),
        ({"chi_threshold": 0.0}, "chi_threshold"),
        ({"max_tail_index": -1.0}, "max_tail_index"),
        ({"min_exceedances": 1}, "min_exceedances"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tail_alignment_diagnostic(_rotated_cauchy_product(0), **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(100,), (2, 3, 4, 5)])
def test_invalid_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        tail_alignment_diagnostic(np.zeros(shape))


@pytest.mark.unit
def test_warning_category_is_a_user_warning() -> None:
    """Users can filter it like any other UserWarning, or by its own class."""
    assert issubclass(TailAlignmentWarning, UserWarning)


# =============================================================================
# 4. Integration with sample_blackjax_unified_nuts
# =============================================================================
#
# Budget 4 chains x 500 draws (k = 100 exceedances). On seeds 0-5 the arcsinh pipeline gave a maximum
# co-exceedance of 0.48 to 0.69 on the rotated product of Cauchys and 0.09 to 0.14 on the product-form
# Task 3 target (threshold 0.3).

PIPELINE_BUDGET = {"draws": 500, "tune": 300, "chains": 4, "seed": 0}


@pytest.mark.unit
def test_pipeline_warns_and_attaches_the_report_on_rotated_cauchy_product() -> None:
    """The pipeline scores its primal post-warm-up draws, stores the report and emits the warning."""
    # Stock NUTS (transform_kind="identity") reaches these rotated tails reliably: all 10 coordinates heavy and
    # max co-exceedance 0.68 to 0.99 on seeds 0-9 of this budget. The arcsinh default under-explores them, and
    # how many coordinates still look heavy varies between runs (as few as one). The emission path is the same
    # for every transform kind.
    with pytest.warns(TailAlignmentWarning, match=r"REPORT\.md §5\.1"):
        res = sample_blackjax_unified_nuts(
            RotatedProductCauchyTarget(d=10), transform_kind="identity", **PIPELINE_BUDGET
        )
    report = res.tail_alignment
    assert isinstance(report, TailAlignmentReport)
    assert report.warn
    # Computed on the x-space draws (not z_samples), pooled over chains.
    reference = tail_alignment_diagnostic(res.samples)
    np.testing.assert_array_equal(report.chi, reference.chi)
    np.testing.assert_array_equal(report.tail_index, reference.tail_index)
    assert report.n_draws == 4 * 500


@pytest.mark.unit
def test_pipeline_check_can_be_switched_off() -> None:
    """check_tail_alignment=False skips the check: no report and no warning, on a target where it would fire."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sample_blackjax_unified_nuts(
            RotatedProductCauchyTarget(d=10), check_tail_alignment=False, **PIPELINE_BUDGET
        )
    assert res.tail_alignment is None
    assert not [w for w in caught if issubclass(w.category, TailAlignmentWarning)]


@pytest.mark.unit
def test_pipeline_attaches_a_silent_report_on_a_product_form_target() -> None:
    """Axis-aligned heavy tails (Task 3): the report is attached and no warning is emitted (tests run with -W error)."""
    target = HeterogeneousHeavyTailedTarget(d=20, heavy_coords=list(range(10)), nu_list=list(T_DFS))
    res = sample_blackjax_unified_nuts(target, **PIPELINE_BUDGET)
    assert res.tail_alignment is not None
    assert not res.tail_alignment.warn
    assert res.tail_alignment.heavy_coordinates == tuple(range(10))
