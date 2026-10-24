"""Numerical checks for Section A of THEORY.md: pullback tail classification and envelopes.

Exploration snapshot (branch explore/tail-transform-theory). Run with:

    uv run python explore/theory/check_pullback_envelopes.py

Checks, each printed with PASS/FAIL:
  A1  Cauchy under arcsinh: exact sech law at perfect calibration; the Lean envelope constant
      K = (r^2 + 1 + delta^2)/r and the sharp constant K* for arbitrary calibration errors.
  A2  Student-t under arcsinh: Laplace tail with rate nu (slope of the potential).
  A3  Cauchy under the h-map: the Lean two-sided Gaussian envelope constants.
  A4  Student-t under the h-map: U = (nu/c) z^2/2 + Ut with Ut' and Ut'' bounded and
      |Ut| <= A(1+|z|) (the H4/H7/A4 "Gaussian plus perturbation" class), U'' -> nu/c.
  A5  Local growth exponent of the pullback hazard psi_Z(t) = psi_F(g(t)) for several maps and
      marginals (the "tail window" is 1 <= beta <= 2).
"""

import numpy as np
from scipy import stats

RESULTS: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ----------------------------------------------------------------------------- A1
z = np.linspace(-40.0, 40.0, 400_001)
pz = (1.0 / np.pi) * np.cosh(z) / (1.0 + np.sinh(z) ** 2)
report("A1 sech law", np.allclose(pz * np.pi * np.cosh(z), 1.0, rtol=1e-12), "pi*p_z(z)*cosh z == 1 on [-40, 40]")

for delta, r in [(0.0, 3.0), (0.0, 0.2), (2.0, 1.0), (-3.0, 0.25), (10.0, 7.0), (0.5, 1.5)]:
    # pi p_z(z) cosh z = r (1 + y^2) / (1 + (delta + r y)^2), y = sinh z
    ratio = r * np.cosh(z) ** 2 / (1.0 + (delta + r * np.sinh(z)) ** 2)
    K = (r**2 + 1 + delta**2) / r
    S = 1 + delta**2 + r**2
    Kstar = (S + np.sqrt(S**2 - 4 * r**2)) / (2 * r)
    ok = ratio.min() >= 1 / K * (1 - 1e-12) and ratio.max() <= K * (1 + 1e-12)
    sharp = abs(ratio.max() - Kstar) / Kstar < 1e-3 and abs(ratio.min() - 1 / Kstar) * Kstar < 1e-3
    report(
        f"A1 envelope delta={delta} r={r}",
        ok and sharp,
        f"range of pi p_z cosh z = [{ratio.min():.4g}, {ratio.max():.4g}]; sharp K*={Kstar:.4g}; Lean K={K:.4g}",
    )

# ----------------------------------------------------------------------------- A2


def student_grad(x, nu):
    return (nu + 1) * x / (nu + x * x)


for nu in [0.5, 1.0, 3.0, 10.0]:
    for R in [0.3, 1.0, 5.0]:
        zz = np.array([15.0, 25.0])
        slope = student_grad(R * np.sinh(zz), nu) * R * np.cosh(zz) - np.tanh(zz)
        report(
            f"A2 arcsinh t nu={nu} R={R}",
            np.allclose(slope, nu, rtol=1e-5),
            f"U'(15), U'(25) = {slope[0]:.6f}, {slope[1]:.6f} (expected nu)",
        )

# ----------------------------------------------------------------------------- A3
for r, c in [(1.0, 1.0), (1.0, 4.0), (0.1, 1.0), (5.0, 0.5), (1.0, 0.05), (0.3, 10.0)]:
    zz = np.linspace(-30.0, 30.0, 600_001)
    # pi p_z(z) exp(z^2/(2c)) = r (1 + z^2/c) / (exp(-z^2/c) + r^2 z^2)
    val = r * (1 + zz**2 / c) / (np.exp(-(zz**2) / c) + r**2 * zz**2)
    m = min(1.0, 1.0 / (r**2 * c))
    M = max(1.0, 1.0 / (r**2 * c))
    rho = min(np.exp(-1.0), r**2 * c / (1 + r**2 * c))
    lo, hi = r * m, r * M / rho
    report(
        f"A3 h-map Cauchy r={r} c={c}",
        val.min() >= lo * (1 - 1e-12) and val.max() <= hi * (1 + 1e-12),
        f"pi p_z e^(z^2/2c) in [{val.min():.4g}, {val.max():.4g}] within Lean bounds [{lo:.4g}, {hi:.4g}]",
    )

# ----------------------------------------------------------------------------- A4


def hmap_student(zv, nu, mu, R, c, m=0.0, s=1.0):
    """Potential, gradient and curvature of the h-map pullback of t_nu(m, s) under x = mu + R z e^{z^2/2c}."""
    E = np.exp(zv * zv / (2 * c))
    g = mu + R * zv * E
    g1 = R * E * (1 + zv * zv / c)
    g2 = R * E * (zv / c) * (3 + zv * zv / c)
    u = (g - m) / s
    V = 0.5 * (nu + 1) * np.log1p(u * u / nu)
    V1 = (nu + 1) * u / (nu + u * u) / s
    V2 = (nu + 1) * (nu - u * u) / (nu + u * u) ** 2 / s**2
    U = V - (np.log(R) + zv * zv / (2 * c) + np.log1p(zv * zv / c))
    dU = V1 * g1 - (zv / c + 2 * zv / (c + zv * zv))
    d2U = V2 * g1**2 + V1 * g2 - (1 / c + 2 * (c - zv * zv) / (c + zv * zv) ** 2)
    return U, dU, d2U


for nu, mu, R, c, m, s in [
    (1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    (0.5, 0.0, 1.0, 1.0, 0.0, 1.0),
    (3.0, 0.0, 1.0, 4.0, 0.0, 1.0),
    (1.0, 0.7, 2.0, 2.0, 0.0, 1.0),
    (2.0, -3.0, 0.3, 0.5, 1.0, 2.0),
    (10.0, 0.0, 1.0, 10.0, 0.0, 1.0),
]:
    zmax = np.sqrt(2 * c * 150)  # keeps (x - m)^4 finite in float64
    zv = np.linspace(-zmax, zmax, 400_001)
    U, dU, d2U = hmap_student(zv, nu, mu, R, c, m, s)
    kappa = nu / c
    Ut, dUt, d2Ut = U - 0.5 * kappa * zv**2, dU - kappa * zv, d2U - kappa
    tail = np.abs(zv) > 0.8 * zmax
    # finite-difference check of the analytic derivatives
    zc = np.array([-2.3, -0.4, 0.0, 0.9, 3.1])
    h = 1e-5
    Up, dUp, _ = hmap_student(zc + h, nu, mu, R, c, m, s)
    Um, dUm, _ = hmap_student(zc - h, nu, mu, R, c, m, s)
    _, dUc, d2Uc = hmap_student(zc, nu, mu, R, c, m, s)
    fd_ok = np.allclose((Up - Um) / (2 * h), dUc, rtol=1e-5, atol=1e-6) and np.allclose(
        (dUp - dUm) / (2 * h), d2Uc, rtol=1e-4, atol=1e-5
    )
    ok = fd_ok and np.all(np.abs(d2U[tail] - kappa) < 0.05 * max(kappa, 1.0))
    report(
        f"A4 h-map t nu={nu} mu={mu} R={R} c={c} m={m} s={s}",
        ok,
        f"sup|Ut'|={np.abs(dUt).max():.3g}, sup|Ut''|={np.abs(d2Ut).max():.3g}, "
        f"sup|Ut|/(1+|z|)={np.max(np.abs(Ut) / (1 + np.abs(zv))):.3g}, "
        f"U'' on |z|>{0.8 * zmax:.1f}: [{d2U[tail].min():.4f}, {d2U[tail].max():.4f}] vs nu/c={kappa:.4f}",
    )

# ----------------------------------------------------------------------------- A5
dists = {
    "Gaussian": stats.norm(),
    "Laplace": stats.laplace(),
    "Weibull(k=1/2)": stats.weibull_min(0.5),
    "lognormal": stats.lognorm(1.0),
    "Student t3": stats.t(3),
    "Cauchy": stats.cauchy(),
}


def log_g(name: str, t: float) -> float:
    if name == "identity":
        return float(np.log(t))
    if name == "sinh":
        return float(t + np.log1p(-np.exp(-2 * t)) - np.log(2))
    if name == "h-map(c=1)":
        return float(np.log(t) + t * t / 2)
    raise ValueError(name)


print("\nA5 local growth exponent beta(t) = d log psi_Z / d log t of psi_Z(t) = -log P(X > g(t))")
print(f"{'map':12s} {'marginal':15s} {'beta(8)':>9s} {'beta(16)':>9s}")
for gname in ["identity", "sinh", "h-map(c=1)"]:
    for dname, d in dists.items():
        cells = []
        for t in [8.0, 16.0]:
            lg1, lg2 = log_g(gname, t), log_g(gname, t * 1.01)
            if max(lg1, lg2) > 700:
                cells.append("overflow")
                continue
            p1, p2 = -d.logsf(np.exp(lg1)), -d.logsf(np.exp(lg2))
            # "underflow": P(X > g(t)) underflows double precision, i.e. psi_Z grows super-polynomially
            cells.append(f"{np.log(p2 / p1) / np.log(1.01):9.3f}" if np.isfinite([p1, p2]).all() else "underflow")
        print(f"{gname:12s} {dname:15s} {cells[0]:>9s} {cells[1]:>9s}")

n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
raise SystemExit(1 if n_fail else 0)
