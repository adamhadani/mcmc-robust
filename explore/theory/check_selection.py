"""Numerical check for Section C of THEORY.md: how well a Hill-estimator threshold separates
light-tailed (gamma = 0) from polynomial-tailed (gamma = 1/alpha > 0) coordinates.

Exploration snapshot (branch explore/tail-transform-theory). Run with:

    uv run python explore/theory/check_selection.py

Draws are i.i.d. (an idealisation: warm-up draws from a sampler are dependent and, for heavy
tails, under-explore the tail). The Hill estimator is computed on |x - median| with the k largest
order statistics. Asymptotically Hill -> max(gamma, 0) (Dekkers, Einmahl & de Haan 1989), so a
fixed threshold tau separates gamma = 0 from gamma > tau; the table shows how slowly the
gamma = 0 cases approach 0.
"""

import numpy as np

rng = np.random.default_rng(20261024)
REPS = 400


def hill(sample: np.ndarray, k: int) -> float:
    a = np.sort(np.abs(sample - np.median(sample)))[::-1]
    return float(np.mean(np.log(a[:k])) - np.log(a[k]))


samplers = {
    "Gaussian (g=0)": lambda m: rng.standard_normal(m),
    "Laplace (g=0)": lambda m: rng.laplace(size=m),
    "lognormal (g=0)": lambda m: rng.lognormal(size=m),
    "t10 (g=0.1)": lambda m: rng.standard_t(10, size=m),
    "t5 (g=0.2)": lambda m: rng.standard_t(5, size=m),
    "t2 (g=0.5)": lambda m: rng.standard_t(2, size=m),
    "Cauchy (g=1)": lambda m: rng.standard_cauchy(m),
}

print(f"{REPS} replicates per cell; P(>tau) is the fraction of replicates classified heavy at threshold tau")
print(
    f"{'n':>6s} {'k':>5s} {'marginal':16s} {'mean':>6s} {'sd':>6s} "
    f"{'P(>0.1)':>8s} {'P(>0.2)':>8s} {'P(>0.3)':>8s} {'P(>0.4)':>8s}"
)
for n in [500, 2000, 20000]:
    for k in [int(np.sqrt(n)), n // 10]:
        for name, draw in samplers.items():
            est = np.array([hill(draw(n), k) for _ in range(REPS)])
            print(
                f"{n:6d} {k:5d} {name:16s} {est.mean():6.3f} {est.std():6.3f} "
                f"{np.mean(est > 0.1):8.2f} {np.mean(est > 0.2):8.2f} {np.mean(est > 0.3):8.2f} "
                f"{np.mean(est > 0.4):8.2f}"
            )
        print()
