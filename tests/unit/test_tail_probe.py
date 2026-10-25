"""Tests for the tail probe (`probe_tail_decay`), the decision rule of the Tukey-h guard.

The probe evaluates U(t) = -log pi(c + s t R_i e_i) at t = r_j for each coordinate i and side s, and
compares the log-slopes t U'(t) a decade apart: constant for a polynomial tail, growing tenfold per
decade for an exponential one and a hundredfold for a Gaussian one.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mcmc_robust.jax_unified_nuts import TAIL_PROBE_RADII, TAIL_PROBE_RATIO, TailProbe, probe_tail_decay


def _log_t(r: jnp.ndarray, nu: float) -> jnp.ndarray:
    return -0.5 * (nu + 1.0) * jnp.log1p(r**2 / nu)


def _log_normal(x: jnp.ndarray, scale: jnp.ndarray | float) -> jnp.ndarray:
    return -0.5 * (x / scale) ** 2 - jnp.log(scale)


def _log_gumbel(r: jnp.ndarray) -> jnp.ndarray:
    return -r - jnp.exp(-r)


#: One-dimensional log densities (up to constants) and whether each side, (minus, plus), is polynomial.
CANONICAL_TAILS = (
    ("gaussian", lambda r: -0.5 * r**2, (False, False)),
    ("laplace", lambda r: -jnp.abs(r), (False, False)),
    ("logistic", lambda r: -jnp.abs(r) - 2.0 * jnp.log1p(jnp.exp(-jnp.abs(r))), (False, False)),
    ("quartic", lambda r: -(r**4), (False, False)),
    ("gumbel", _log_gumbel, (False, False)),
    ("cauchy", lambda r: _log_t(r, 1.0), (True, True)),
    ("t3", lambda r: _log_t(r, 3.0), (True, True)),
    ("t30", lambda r: _log_t(r, 30.0), (True, True)),
    ("t2 right, gaussian left", lambda r: jnp.where(r > 0.0, _log_t(r, 2.0), -0.5 * r**2), (False, True)),
)
_FNS = tuple(fn for _, fn, _ in CANONICAL_TAILS)
_EXPECTED = np.array([sides for _, _, sides in CANONICAL_TAILS])
_LOC = np.linspace(-4.0, 4.0, len(CANONICAL_TAILS))
_SCALE = np.geomspace(0.05, 20.0, len(CANONICAL_TAILS))


def _product_logdensity(x: jnp.ndarray) -> jnp.ndarray:
    r = (x - _LOC) / _SCALE
    return sum(fn(r[i]) for i, fn in enumerate(_FNS))


@pytest.mark.unit
def test_the_probe_classifies_each_side_of_canonical_tails() -> None:
    """Polynomial on a side exactly for the Student-t sides; Gaussian, exponential and lighter sides are not."""
    probe = probe_tail_decay(_product_logdensity, _LOC, _SCALE)
    assert isinstance(probe, TailProbe)
    np.testing.assert_array_equal(probe.polynomial, _EXPECTED)


@pytest.mark.unit
def test_the_probe_definition() -> None:
    """U is -log pi on the rays; the slopes are half-decade differences of U; the rule compares them a decade apart."""
    center = _LOC + 0.3 * _SCALE
    scale = 2.0 * _SCALE
    probe = probe_tail_decay(_product_logdensity, center, scale)
    d, radii = len(_LOC), np.asarray(TAIL_PROBE_RADII)
    assert probe.potential.shape == (d, 2, len(radii))
    for i in range(d):
        for k, sign in enumerate((-1.0, 1.0)):
            for j, r in enumerate(radii):
                x = center.copy()
                x[i] += sign * r * scale[i]
                assert probe.potential[i, k, j] == pytest.approx(-float(_product_logdensity(jnp.asarray(x))), rel=1e-13)
    np.testing.assert_allclose(radii[1:] / radii[:-1], np.sqrt(10.0), rtol=1e-15)
    half_decade = 0.5 * np.log(10.0)
    u = probe.potential
    assert np.isinf(u).any()  # the Gumbel side's density underflows to 0 far out
    with np.errstate(invalid="ignore"):
        near, far = (u[..., 1] - u[..., 0]) / half_decade, (u[..., 3] - u[..., 2]) / half_decade
    np.testing.assert_allclose(probe.slope_near, near, rtol=1e-13)
    np.testing.assert_allclose(probe.slope_far, far, rtol=1e-13)
    with np.errstate(invalid="ignore"):
        expected = (
            np.all(np.isfinite(u), axis=-1)
            & (probe.slope_near > 0.0)
            & (probe.slope_far > 1.0)
            & (probe.slope_far < TAIL_PROBE_RATIO * probe.slope_near)
        )
    np.testing.assert_array_equal(probe.polynomial, expected)
    assert np.sqrt(10.0) == pytest.approx(TAIL_PROBE_RATIO)


@pytest.mark.unit
def test_the_far_slope_is_the_tail_exponent_of_a_polynomial_side() -> None:
    """For density ~ |x|^-beta the far slope is beta: nu + 1 for t_nu, up to O(nu / r^2) at r >= 1e4 scales."""
    nus = np.array([1.0, 2.5, 7.0, 30.0])

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(_log_t(x, nus))

    probe = probe_tail_decay(logdensity, np.zeros(4), np.ones(4))
    np.testing.assert_allclose(probe.slope_far, np.column_stack([nus + 1.0, nus + 1.0]), rtol=1e-5)
    assert probe.polynomial.all()


@pytest.mark.unit
def test_the_decision_does_not_depend_on_where_the_probe_starts() -> None:
    """Centres up to 3 scales off and scales 10 times too large or too small leave every decision unchanged."""
    for shift in (-3.0, 0.0, 3.0):
        for factor in (0.1, 1.0, 10.0):
            probe = probe_tail_decay(_product_logdensity, _LOC + shift * _SCALE, factor * _SCALE)
            np.testing.assert_array_equal(probe.polynomial, _EXPECTED, err_msg=f"shift={shift}, factor={factor}")


@pytest.mark.unit
def test_the_probe_reads_conditional_tails_funnel_and_elliptical_t() -> None:
    """The rays are conditionals: Neal's funnel has none polynomial, a multivariate t_nu all, with exponent nu + d."""

    def funnel(x: jnp.ndarray) -> jnp.ndarray:
        v, rest = x[0], x[1:]
        return _log_normal(v, 3.0) + jnp.sum(_log_normal(rest, jnp.exp(0.5 * v)))

    probe = probe_tail_decay(funnel, np.zeros(5), np.array([3.0, 1.0, 1.0, 1.0, 1.0]))
    assert not probe.polynomial.any()

    nu, d = 1.0, 5
    q = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))[0]
    prec = jnp.asarray(q @ np.diag(1.0 / np.geomspace(1.0, 25.0, d)) @ q.T)

    def mvt(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * (nu + d) * jnp.log1p(x @ prec @ x / nu)

    probe = probe_tail_decay(mvt, np.zeros(d), np.ones(d))
    assert probe.polynomial.all()
    np.testing.assert_allclose(probe.slope_far, nu + d, rtol=1e-4)


@pytest.mark.unit
def test_non_finite_flat_increasing_and_improper_sides_are_not_polynomial() -> None:
    """No polynomial verdict where the density vanishes (beyond a point, or at one probe radius) or is NaN,
    is flat or grows, or decays too slowly to be proper (exponent at most 1)."""
    hole = TAIL_PROBE_RADII[1]

    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        cauchy = _log_t(x, 1.0)
        support = jnp.where(jnp.abs(x[0]) < 50.0, cauchy[0], -jnp.inf)  # density 0 beyond 50
        nan_side = jnp.where(x[1] > 0.0, jnp.nan, cauchy[1])
        flat = 0.0 * x[2]
        increasing = jnp.where(x[3] > 0.0, jnp.log1p(x[3] ** 2), cauchy[3])  # improper on the right
        improper = jnp.where(x[4] > 0.0, -0.25 * jnp.log1p(x[4] ** 2), cauchy[4])  # |x|^(-1/2) on the right
        holed = jnp.where(jnp.abs(x[5] - hole) < 1.0, -jnp.inf, cauchy[5])  # density 0 at the second radius only
        return support + nan_side + flat + increasing + improper + holed

    probe = probe_tail_decay(logdensity, np.zeros(6), np.ones(6))
    np.testing.assert_array_equal(
        probe.polynomial,
        [[False, False], [True, False], [False, False], [True, False], [True, False], [True, False]],
    )
    assert probe.slope_far[4, 1] == pytest.approx(0.5, rel=1e-6)


@pytest.mark.unit
def test_the_probe_validates_its_inputs() -> None:
    def logdensity(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(_log_t(x, 1.0))

    for center, scale in (
        (np.zeros(3), np.ones(2)),
        (np.zeros((2, 3)), np.ones((2, 3))),
        (np.zeros(3), np.array([1.0, 0.0, 1.0])),
        (np.zeros(3), np.array([1.0, np.inf, 1.0])),
        (np.array([0.0, np.nan, 0.0]), np.ones(3)),
    ):
        with pytest.raises(ValueError):
            probe_tail_decay(logdensity, center, scale)
