"""Numerical checks for Section E of THEORY.md: how much Gaussian mass a tail map puts where the
leapfrog integrator is linearly unstable.

Exploration snapshot (branch explore/tail-transform-theory). Run with:

    uv run python explore/theory/check_stability.py

For a light (Gaussian) coordinate x ~ N(m, sigma^2) pulled back through a tail map, the
linear-stability condition of the leapfrog integrator at step size eps is eps^2 U''(z) < 4.
We report the Gaussian probability of the region where it fails.

  E1  arcsinh, x = m + R sinh z, r = R/sigma: U'' = r^2 + 2 s^2 - sech^2 z with s = (x - m)/sigma,
      so the failure set is |s| > s*, with s*^2 in [2/eps^2 - r^2/2, 2/eps^2 - r^2/2 + 1/2]
      (Lean, Section 12). Mass = 2 Phi(-s*), approximately 2 Phi(-sqrt(2)/eps) for every r.
  E2  h-map, x = m + R z exp(z^2/(2c)): failure mass as a function of c (it shrinks as c grows).
  E3  Bulk shape: U''(0) for arcsinh and h-map pullbacks of Gaussian and Student-t marginals
      (U''(0) < 0 means the pullback is bimodal).
"""

import numpy as np
from scipy import optimize, special

RESULTS: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ----------------------------------------------------------------------------- E1
print("E1 arcsinh pullback of N(m, sigma^2): Gaussian mass where eps^2 U'' >= 4")
print(f"{'r':>5s} {'eps':>5s} {'s*':>8s} {'mass':>11s} {'Lean bracket on mass':>27s} {'2Phi(-sqrt2/eps)':>17s}")
for r in [1.0, 1.5, 2.0]:
    for eps in [1.0, 0.7, 0.5, 0.3, 0.2]:

        def curv_gap(z, r=r, eps=eps):
            return r * r * np.cosh(2 * z) - 1 / np.cosh(z) ** 2 - 4 / eps**2

        if curv_gap(0.0) >= 0:
            print(f"{r:5.2f} {eps:5.2f}  unstable at z = 0")
            continue
        zs = optimize.brentq(curv_gap, 0.0, 60.0)
        s_star = r * np.sinh(zs)
        mass = 2 * special.ndtr(-s_star)
        s_lo2 = 2 / eps**2 - r * r / 2
        s_hi = np.sqrt(s_lo2 + 0.5)
        s_lo = np.sqrt(max(s_lo2, 0.0))
        m_lo, m_hi = 2 * special.ndtr(-s_hi), 2 * special.ndtr(-s_lo)
        ok = s_lo - 1e-9 <= s_star <= s_hi + 1e-9 and m_lo <= mass <= m_hi
        RESULTS.append((f"E1 r={r} eps={eps}", ok))
        print(
            f"{r:5.2f} {eps:5.2f} {s_star:8.4f} {mass:11.3e}   [{m_lo:9.3e}, {m_hi:9.3e}] "
            f"{2 * special.ndtr(-np.sqrt(2) / eps):17.3e}  {'PASS' if ok else 'FAIL'}"
        )

# ----------------------------------------------------------------------------- E2


def hmap_gauss_curv(z, r, c):
    """U'' for U = r^2 z^2 e^{z^2/c}/2 - z^2/(2c) - log(1 + z^2/c) (h-map pullback of N(m, sigma^2), mu = m)."""
    q = z * z / c
    return r * r * np.exp(q) * (1 + 5 * q + 2 * q * q) - 1 / c - 2 * (c - z * z) / (c + z * z) ** 2


print("\nE2 h-map pullback of N(m, sigma^2), r = R/sigma = 1: Gaussian mass where eps^2 U'' >= 4")
print("cells: mass (threshold s* in standard deviations)")
print(f"{'c':>6s} " + " ".join(f"{'eps=' + str(e):>18s}" for e in [0.7, 0.5, 0.3]))
for c in [1.0, 4.0, 16.0, 64.0]:
    row = []
    for eps in [0.7, 0.5, 0.3]:
        zg = np.linspace(0.0, np.sqrt(c * 40.0), 400_001)
        unstable = eps**2 * hmap_gauss_curv(zg, 1.0, c) >= 4
        if not unstable.any():
            row.append(f"{'0':>18s}")
            continue
        first = np.argmax(unstable)
        # failure set must be a half-line on z >= 0 for the 2 Phi formula to apply
        if not unstable[first:].all():
            row.append(f"{'non-monotone':>18s}")
            continue
        s_star = zg[first] * np.exp(zg[first] ** 2 / (2 * c))
        row.append(f"{2 * special.ndtr(-s_star):10.3e} ({s_star:5.2f})")
    print(f"{c:6.1f} " + " ".join(row))

# ----------------------------------------------------------------------------- E3
print("\nE3 curvature at the centre, U''(0) (negative => bimodal pullback)")
for r in [0.5, 1 / np.sqrt(2), 1.0, 1.5]:
    gauss_as = r * r - 1.0  # arcsinh, Gaussian: r^2 cosh 0 - sech^2 0
    cauchy_as = 2 * r * r - 1.0  # arcsinh, Cauchy: (nu+1)/nu r^2 - 1 with nu = 1
    print(f"  r={r:.3f}: arcsinh-Gaussian U''(0)={gauss_as:+.3f}, arcsinh-Cauchy U''(0)={cauchy_as:+.3f}")
# finite-difference confirmation of the closed forms
for r in [0.5, 1.5]:
    h = 1e-4

    def u_g(z, r=r):
        return r * r * np.sinh(z) ** 2 / 2 - np.log(np.cosh(z))

    def u_c(z, r=r):
        return np.log1p((r * np.sinh(z)) ** 2) - np.log(np.cosh(z))

    fd_g = (u_g(h) - 2 * u_g(0.0) + u_g(-h)) / h**2
    fd_c = (u_c(h) - 2 * u_c(0.0) + u_c(-h)) / h**2
    report(
        f"E3 closed forms r={r}",
        abs(fd_g - (r * r - 1)) < 1e-5 and abs(fd_c - (2 * r * r - 1)) < 1e-5,
        f"finite differences {fd_g:.6f}, {fd_c:.6f} vs {r * r - 1:.6f}, {2 * r * r - 1:.6f}",
    )

n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
raise SystemExit(1 if n_fail else 0)
