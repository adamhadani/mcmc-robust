"""Illustration for Section D of THEORY.md: HMC acceptance far in the tail of 1-D pullbacks.

Exploration snapshot (branch explore/tail-transform-theory). Run with:

    uv run python explore/theory/check_hmc_acceptance.py

Livingstone et al. (2019, corrected Thm 5.14) show that when the potential's gradient grows faster
than linearly, HMC acceptance tends to 0 far in the tail, which rules out geometric ergodicity.
This script shows the effect for the arcsinh pullback of a Gaussian (gradient ~ e^{2z}) and its
absence for pullbacks inside the linear-to-quadratic window. It is an illustration, not a proof.

Pullbacks (r = R / scale, location matched):
  arcsinh-Gaussian  U = r^2 sinh^2 z / 2 - log cosh z                 (r = 1.5, repo default multiplier)
  arcsinh-Cauchy    U = log(1 + r^2 sinh^2 z) - log cosh z            (r = 2.2)
  h-Cauchy (c=1)    U = log(1 + z^2 e^{z^2}) - z^2/2 - log(1 + z^2)    (r = 1)
  h-Gaussian (c=16) U = z^2 e^{z^2/16}/2 - z^2/32 - log(1 + z^2/16)    (misclassified light coordinate)
"""

import numpy as np

rng = np.random.default_rng(7)


def leapfrog_dH(z0, p0, U, G, eps, L):
    z, p = z0, p0
    H0 = U(z0) + 0.5 * p0 * p0
    with np.errstate(over="ignore", invalid="ignore"):
        p = p - 0.5 * eps * G(z)
        for i in range(L):
            z = z + eps * p
            if i < L - 1:
                p = p - eps * G(z)
        p = p - 0.5 * eps * G(z)
        return U(z) + 0.5 * p * p - H0


def ag(r):
    return (
        lambda z: r * r * np.sinh(z) ** 2 / 2 - np.log(np.cosh(z)),
        lambda z: r * r * np.sinh(z) * np.cosh(z) - np.tanh(z),
    )


def ac(r):
    return (
        lambda z: np.log1p((r * np.sinh(z)) ** 2) - np.log(np.cosh(z)),
        lambda z: 2 * r * r * np.sinh(z) * np.cosh(z) / (1 + (r * np.sinh(z)) ** 2) - np.tanh(z),
    )


def hc(c):
    def U(z):
        return np.logaddexp(0.0, 2 * np.log(abs(z) + 1e-300) + z * z / c) - z * z / (2 * c) - np.log1p(z * z / c)

    def G(z):
        e = np.exp(z * z / c)
        return 2 * z * e * (1 + z * z / c) / (1 + z * z * e) - z / c - 2 * z / (c + z * z)

    return U, G


def hg(c):
    def U(z):
        return z * z * np.exp(z * z / c) / 2 - z * z / (2 * c) - np.log1p(z * z / c)

    def G(z):
        e = np.exp(z * z / c)
        return z * e * (1 + z * z / c) - z / c - 2 * z / (c + z * z)

    return U, G


targets = {
    "arcsinh-Gaussian r=1.5": ag(1.5),
    "arcsinh-Cauchy r=2.2": ac(2.2),
    "h-Cauchy c=1": hc(1.0),
    "h-Gaussian c=16": hg(16.0),
}
momenta = rng.standard_normal(2000)
z0s = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
for eps, L in [(0.2, 10), (0.5, 10)]:
    print(f"\nmean acceptance over 2000 momenta, eps={eps}, L={L}")
    print(f"{'pullback':24s} " + " ".join(f"{'z0=' + str(z0):>8s}" for z0 in z0s))
    for name, (U, G) in targets.items():
        row = []
        for z0 in z0s:
            dH = np.array([leapfrog_dH(z0, p, U, G, eps, L) for p in momenta])
            acc = np.where(np.isfinite(dH), np.exp(-np.clip(dH, 0.0, 700.0)), 0.0)
            row.append(f"{acc.mean():8.3f}")
        print(f"{name:24s} " + " ".join(row))
