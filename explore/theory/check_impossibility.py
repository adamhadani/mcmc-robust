"""Numerical checks for Section B of THEORY.md: no common tail map for light and heavy marginals.

Exploration snapshot (branch explore/tail-transform-theory). Run with:

    uv run python explore/theory/check_impossibility.py

Checks:
  B1  Sharpness of the trade-off exponent 2a/alpha: for g = R sinh(lam z) the Student-t_alpha
      pullback potential has slope a = alpha*lam and the Gaussian pullback potential grows like
      exp(2 lam z) = exp(2 a z / alpha).
  B2  The Lean lower bound (Section 11, gaussPullback_lower_of_cauchy_envelope) evaluated for
      g = sinh with the constants a = 1, b = log 2 proved in arcsinh_cauchy_pullback_linear_lower.
  B3  The window criterion psi_1 <~ psi_2^2 on pairs of marginals (bounded ratio = a common map
      exists; growing ratio = none), and the constructive map g = phi^{-1}(t^2),
      phi = max(psi_1, psi_2), for an admissible pair.
"""

import numpy as np
from scipy import optimize, special, stats

RESULTS: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ----------------------------------------------------------------------------- B1
def log_U_gauss_sinh(z, R, lam):
    """log of the Gaussian pullback potential U_1 = (R sinh(lam z))^2/2 - log(R lam cosh(lam z)), large z."""
    x = R * np.sinh(lam * z)
    U = 0.5 * x * x - np.log(R * lam * np.cosh(lam * z))
    return np.log(U)


for alpha in [0.5, 1.0, 3.0]:
    for lam in [0.5, 1.0, 2.0]:
        R = 1.0
        z = np.array([12.0 / lam, 14.0 / lam])
        x = R * np.sinh(lam * z)
        # Student-t_alpha potential gradient through g: V'(g) g' - (log g')'
        slope = (alpha + 1) * x / (alpha + x * x) * R * lam * np.cosh(lam * z) - lam * np.tanh(lam * z)
        a = slope.mean()
        growth = (log_U_gauss_sinh(z[1], R, lam) - log_U_gauss_sinh(z[0], R, lam)) / (z[1] - z[0])
        ok = abs(a - alpha * lam) < 1e-6 * alpha * lam + 1e-9 and abs(growth - 2 * a / alpha) < 1e-3 * (2 * a / alpha)
        report(
            f"B1 alpha={alpha} lam={lam}",
            ok,
            f"t-pullback slope a={a:.6f} (alpha*lam={alpha * lam}); Gaussian pullback growth rate {growth:.5f} "
            f"vs 2a/alpha={2 * a / alpha:.5f}",
        )

# ----------------------------------------------------------------------------- B2
a, b = 1.0, np.log(2.0)
zs = np.linspace(np.arcsinh(1.0), 20.0, 20_001)  # region where sinh z >= 1
UG = 0.5 * np.sinh(zs) ** 2 - np.log(np.cosh(zs))
lower = (a / (2 * np.exp(b))) ** 2 / 4 * np.exp(2 * a * zs) - b - np.log(4.0)
UC = np.log1p(np.sinh(zs) ** 2) - np.log(np.cosh(zs))
report(
    "B2 Cauchy envelope for sinh",
    bool(np.all(a * zs - b <= UC + 1e-12)),
    f"min over z of U_C(z) - (z - log 2) = {np.min(UC - (a * zs - b)):.3e} (>= 0 required)",
)
report(
    "B2 Gaussian lower bound for sinh",
    bool(np.all(lower <= UG + 1e-9)),
    f"min over z of U_G(z) - bound = {np.min(UG - lower):.3e}; ratio U_G/bound at z=20: {UG[-1] / lower[-1]:.2f}",
)

# ----------------------------------------------------------------------------- B3
# right hazards psi(x) = -log P(X > x), x >= 1
xs = np.logspace(0.5, 300, 4000)
haz = {
    "Gaussian": lambda x: -special.log_ndtr(-x),
    "Laplace": lambda x: x + np.log(2.0),
    "Weibull(k=1/2)": lambda x: np.sqrt(x),
    "Weibull(k=3/2)": lambda x: x**1.5,
    "lognormal": lambda x: -special.log_ndtr(-np.log(x)),
    "Cauchy": lambda x: -np.log(stats.cauchy.sf(x)),
    "log-Cauchy": lambda x: -np.log(stats.cauchy.sf(np.log(x))),
}
print("\nB3 window criterion: max over x in [x_lo, x] of psi_light / psi_heavy^2 (bounded <=> a common map exists)")
pairs = [
    ("Gaussian", "Laplace", True),
    ("Gaussian", "Weibull(k=3/2)", True),
    ("Gaussian", "Weibull(k=1/2)", False),
    ("Gaussian", "Cauchy", False),
    ("lognormal", "Cauchy", True),
    ("Cauchy", "log-Cauchy", False),
]
for light, heavy, expect in pairs:
    with np.errstate(all="ignore"):
        p1, p2 = haz[light](xs), haz[heavy](xs)
    finite = np.isfinite(p1) & np.isfinite(p2) & (p2 > 1.0)
    ratio = np.exp(np.log(p1[finite]) - 2 * np.log(p2[finite]))
    xf = xs[finite]
    checkpoints = [1e1, 1e3, 1e8, 1e30, 1e100]
    vals = [np.max(ratio[xf <= c]) if np.any(xf <= c) else np.nan for c in checkpoints]
    # heuristic on a finite range: the running maximum must stop growing between 1e8 and 1e100
    bounded = vals[-1] < 1.5 * vals[2]
    report(
        f"B3 {light} vs {heavy}",
        bounded == expect,
        "running max of ratio at x=1e1,1e3,1e8,1e30,1e100: " + ", ".join(f"{v:.3g}" for v in vals),
    )

# constructive map for an admissible pair: Gaussian and Laplace, g = phi^{-1}(t^2)
psi_g, psi_l = haz["Gaussian"], haz["Laplace"]


def phi(x):
    return max(psi_g(x), psi_l(x))


ts = np.linspace(2.0, 30.0, 57)
gt = np.array([optimize.brentq(lambda x, t=t: phi(x) - t * t, 1e-9, 1e3) for t in ts])
hz_g = np.array([psi_g(x) for x in gt])
hz_l = np.array([psi_l(x) for x in gt])
lo_g, lo_l = np.min(hz_g / ts), np.min(hz_l / ts)
ok = np.all(hz_g <= ts**2 + 1e-9) and np.all(hz_l <= ts**2 + 1e-9) and lo_g > 0.5 and lo_l > 0.5
report(
    "B3 constructive common map (Gaussian, Laplace)",
    bool(ok),
    f"psi_i(g(t)) <= t^2 for both; min psi_i(g(t))/t = {lo_g:.3f} (Gaussian), {lo_l:.3f} (Laplace) on t in [2, 30]",
)

n_fail = sum(not ok for _, ok in RESULTS)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
raise SystemExit(1 if n_fail else 0)
