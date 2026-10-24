"""Tests for the Tukey h-map coordinate transform (`JaxTukeyHTransform`) and its Lambert W inverse.

Per coordinate, x = mu + R z exp(h z^2 / 2) with h >= 0 (h = 0 is the affine map). The inverse
map is z = u exp(-W(h u^2) / 2) with u = (x - mu) / R and W the principal Lambert W.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from mcmc_robust.integrations import JaxAffineTransform, JaxTukeyHTransform, lambertw_from_log

LOCATION = np.array([2.0, -1.5, 0.25, 7.0, -3.0], dtype=np.float64)
RADII = np.array([0.3, 1.0, 4.0, 12.5, 0.07], dtype=np.float64)
H = np.array([1.0, 0.0, 0.5, 2.0, 0.1], dtype=np.float64)
MASK = np.array([True, True, False, True, True])


def _draw_z(rng: np.random.Generator, scale: float) -> np.ndarray:
    """Normal z-space points, clipped so that h z^2 / 2 <= 600 (exp stays far from float64 overflow)."""
    limit = np.sqrt(1200.0 / np.where(H > 0.0, H, 1e-300))
    return np.clip(rng.normal(size=5) * scale, -limit, limit)


def _correlated_t_logdensity(x: jnp.ndarray) -> jnp.ndarray:
    """A non-separable, heavy-tailed test density so cross-terms in the gradient matter."""
    a = jnp.array([[1.0, 0.3, 0.0, 0.1, 0.0]] * 5) + jnp.eye(5)
    r = a @ (x - 1.0)
    return -2.5 * jnp.log1p(jnp.sum(r**2) / 3.0)


# =============================================================================
# Lambert W, principal branch, evaluated from log y
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("xp", [jnp, np], ids=["jax", "numpy"])
def test_lambertw_from_log_matches_scipy(xp: object) -> None:
    """W(exp(log_y)) agrees with scipy's principal-branch Lambert W to a few ulps over log y in [-700, 700]."""
    log_y = np.concatenate([np.linspace(-700.0, 700.0, 2801), [-40.0, -39.999, 0.999, 1.0, 1.001]])
    expected = special.lambertw(np.exp(log_y)).real
    got = np.asarray(lambertw_from_log(xp.asarray(log_y), xp=xp))  # type: ignore[attr-defined]
    np.testing.assert_allclose(got, expected, rtol=2e-15, atol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("xp", [jnp, np], ids=["jax", "numpy"])
def test_lambertw_from_log_huge_arguments_and_edge_values(xp: object) -> None:
    """For y far beyond float64 range W solves w + log w = log y to machine precision; y = 0, inf and NaN map exactly."""
    log_y = np.geomspace(700.0, 1e300, 400)
    w = np.asarray(lambertw_from_log(xp.asarray(log_y), xp=xp))  # type: ignore[attr-defined]
    assert np.all(np.isfinite(w)) and np.all(w > 0.0)
    assert np.all(np.diff(w) > 0.0)
    np.testing.assert_allclose(w + np.log(w), log_y, rtol=4e-16, atol=0.0)

    edge = np.asarray(lambertw_from_log(xp.asarray([-np.inf, np.inf, np.nan, -1e300]), xp=xp))  # type: ignore[attr-defined]
    assert edge[0] == 0.0
    assert edge[1] == np.inf
    assert np.isnan(edge[2])
    assert edge[3] == 0.0


@pytest.mark.unit
def test_lambertw_from_log_is_jit_and_vmap_compatible() -> None:
    """The JAX version compiles and vectorises, with the same values as eager evaluation."""
    log_y = jnp.linspace(-50.0, 900.0, 77)
    eager = lambertw_from_log(log_y)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambertw_from_log)(log_y)), np.asarray(eager))
    np.testing.assert_array_equal(np.asarray(jax.vmap(lambertw_from_log)(log_y)), np.asarray(eager))


# =============================================================================
# The transform
# =============================================================================


@pytest.mark.unit
def test_tukey_h_interface_and_parameter_validation() -> None:
    """Constructor aliases and broadcasting follow the other JAX transforms; h must be finite and non-negative."""
    tr = JaxTukeyHTransform(radii=RADII, location=LOCATION, h=H, mask=MASK)
    assert tr.d == 5
    np.testing.assert_array_equal(np.asarray(tr.h), H)
    np.testing.assert_array_equal(np.asarray(tr.radii), RADII)
    np.testing.assert_array_equal(np.asarray(tr.location), LOCATION)
    np.testing.assert_array_equal(np.asarray(tr.mask), MASK)

    scalar = JaxTukeyHTransform(radius=2.0, dim=3, h=0.5)
    assert scalar.d == 3
    np.testing.assert_array_equal(np.asarray(scalar.h), np.full(3, 0.5))
    np.testing.assert_array_equal(np.asarray(JaxTukeyHTransform(radius=1.0, dim=2).h), np.zeros(2))

    with pytest.raises(ValueError, match="h"):
        JaxTukeyHTransform(radii=[1.0, 1.0], h=[0.5, -0.1])
    with pytest.raises(ValueError, match="h"):
        JaxTukeyHTransform(radii=[1.0, 1.0], h=[0.5, np.inf])
    with pytest.raises(ValueError, match="h"):
        JaxTukeyHTransform(radii=[1.0, 1.0], h=[0.5, 0.5, 0.5])


@pytest.mark.unit
def test_tukey_h_inverse_formula_and_round_trip() -> None:
    """inverse is mu + R z exp(h z^2 / 2) on masked coordinates; forward inverts it in the bulk and far tails."""
    tr = JaxTukeyHTransform(radii=RADII, location=LOCATION, h=H, mask=MASK)
    rng = np.random.default_rng(0)
    for scale in (0.5, 3.0, 12.0):
        z = _draw_z(rng, scale)
        x = np.asarray(tr.inverse(jnp.asarray(z)))
        expected = np.where(MASK, LOCATION + RADII * z * np.exp(0.5 * H * z**2), z)
        np.testing.assert_allclose(x, expected, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(tr.forward(jnp.asarray(x))), z, rtol=1e-13, atol=1e-14)

    # x -> z -> x from the centre out to |x| = 1e300. The x round trip loses at most the conditioning
    # factor |d log x / d log z| = 1 + h z^2 of the inverse map.
    for x_mag in (0.0, 1e-8, 1.0, 1e3, 1e30, 1e150, 1e300):
        x = LOCATION + np.array([1.0, -1.0, 1.0, -1.0, 1.0]) * x_mag
        z = np.asarray(tr.forward(jnp.asarray(x)))
        assert np.all(np.isfinite(z))
        x_back = np.asarray(tr.inverse(jnp.asarray(z)))
        cond = 1.0 + np.where(MASK, H, 0.0) * np.minimum(np.abs(z), 1e100) ** 2
        rel_err = np.abs(x_back - x) / np.maximum(np.abs(x), 1.0)
        assert np.all(rel_err <= 8e-14 * cond), (x_mag, rel_err, cond)
    # Unmasked coordinates pass through unchanged in both directions.
    assert float(tr.forward(jnp.asarray(LOCATION + 5.0))[2]) == LOCATION[2] + 5.0


@pytest.mark.unit
def test_tukey_h_log_jacobian_matches_slogdet_of_autodiff_jacobian() -> None:
    """log|det dx/dz| = sum over masked coordinates of log R + h z^2 / 2 + log1p(h z^2), equal to slogdet(jacfwd(inverse))."""
    tr = JaxTukeyHTransform(radii=RADII, location=LOCATION, h=H, mask=MASK)
    rng = np.random.default_rng(1)
    for scale in (0.3, 2.0, 12.0):
        z = jnp.asarray(_draw_z(rng, scale))
        sign, logabsdet = jnp.linalg.slogdet(jax.jacfwd(tr.inverse)(z))
        assert float(sign) == 1.0
        np.testing.assert_allclose(float(tr.log_jacobian_det(z)), float(logabsdet), rtol=1e-13, atol=1e-13)
        zn = np.asarray(z)
        terms = np.log(RADII) + 0.5 * H * zn**2 + np.log1p(H * zn**2)
        np.testing.assert_allclose(float(tr.log_jacobian_det(z)), float(np.sum(terms[MASK])), rtol=1e-14)


@pytest.mark.unit
def test_tukey_h_pullback_gradient_matches_autodiff() -> None:
    """forward_grad(x, grad_x) equals jax.grad of the wrapped log density at z = forward(x), bulk and tails."""
    tr = JaxTukeyHTransform(radii=RADII, location=LOCATION, h=H, mask=MASK)
    wrapped = tr.wrap_logdensity(_correlated_t_logdensity)
    grad_x_fn = jax.grad(_correlated_t_logdensity)
    grad_z_fn = jax.grad(wrapped)
    rng = np.random.default_rng(2)
    for scale in (0.5, 4.0, 1e3, 1e8):
        x = jnp.asarray(LOCATION + rng.standard_cauchy(size=5) * scale)
        z = tr.forward(x)
        analytic = np.asarray(tr.forward_grad(x, grad_x_fn(x)))
        autodiff = np.asarray(grad_z_fn(z))
        assert np.all(np.isfinite(analytic))
        np.testing.assert_allclose(analytic, autodiff, rtol=1e-9, atol=1e-12)
        # The wrapped density is the primal density plus the log-Jacobian.
        np.testing.assert_allclose(
            float(wrapped(z)), float(_correlated_t_logdensity(tr.inverse(z)) + tr.log_jacobian_det(z)), rtol=1e-14
        )


@pytest.mark.unit
def test_tukey_h_with_zero_h_is_the_affine_transform() -> None:
    """With h = 0 every method coincides with JaxAffineTransform for the same (mu, R, mask)."""
    tukey = JaxTukeyHTransform(radii=RADII, location=LOCATION, h=0.0, mask=MASK)
    affine = JaxAffineTransform(radii=RADII, location=LOCATION, mask=MASK)
    grad_x_fn = jax.grad(_correlated_t_logdensity)
    rng = np.random.default_rng(3)
    for scale in (0.5, 50.0, 1e6):
        x = jnp.asarray(LOCATION + rng.standard_cauchy(size=5) * scale)
        z = affine.forward(x)
        np.testing.assert_allclose(np.asarray(tukey.forward(x)), np.asarray(z), rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(np.asarray(tukey.inverse(z)), np.asarray(affine.inverse(z)), rtol=1e-15, atol=0.0)
        assert float(tukey.log_jacobian_det(z)) == pytest.approx(float(affine.log_jacobian_det(z)), rel=1e-15)
        np.testing.assert_allclose(
            np.asarray(tukey.forward_grad(x, grad_x_fn(x))),
            np.asarray(affine.forward_grad(x, grad_x_fn(x))),
            rtol=1e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            float(tukey.wrap_logdensity(_correlated_t_logdensity)(z)),
            float(affine.wrap_logdensity(_correlated_t_logdensity)(z)),
            rtol=1e-15,
        )


def _gauss(x: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.sum(x**2)


def _cauchy_product(x: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(jnp.log1p(x**2))


def _rotated(x: jnp.ndarray) -> jnp.ndarray:
    # An infinite coordinate times a zero matrix entry gives NaN here, not -inf.
    a = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    return -jnp.sum(jnp.log1p((a @ x) ** 2))


@pytest.mark.unit
@pytest.mark.parametrize("logdensity", [_gauss, _cauchy_product, _rotated], ids=["gauss", "cauchy", "rotated"])
def test_tukey_h_overflow_gives_minus_inf_log_density_with_finite_gradient(
    logdensity: Callable[[jnp.ndarray], jnp.ndarray],
) -> None:
    """Where exp(h z^2 / 2) overflows the pulled-back log density is -inf (never NaN) and its gradient is finite."""
    h0 = 1.0
    tr = JaxTukeyHTransform(radii=[1.0, 2.0], location=[0.0, 1.0], h=[h0, 0.5])
    wrapped = tr.wrap_logdensity(logdensity)
    edge = float(np.sqrt(2.0 * np.log(np.finfo(np.float64).max) / h0))
    for fn in (wrapped, jax.jit(wrapped)):
        for z0 in (edge + 0.5, -(edge + 3.0), 1e160, -1e300):
            z = jnp.asarray([z0, 0.3])
            assert not np.isfinite(float(tr.inverse(z)[0]))
            assert float(fn(z)) == -np.inf
            grad = np.asarray(jax.grad(fn)(z))
            assert np.all(np.isfinite(grad)), grad
        # Inside the representable range the value is the exact, finite pullback.
        z_in = jnp.asarray([21.0, 0.3])
        assert np.isfinite(float(fn(z_in)))
