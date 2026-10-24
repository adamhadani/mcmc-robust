# mcmc-robust

[![CI](https://github.com/adamhadani/mcmc-robust/actions/workflows/ci.yml/badge.svg)](https://github.com/adamhadani/mcmc-robust/actions/workflows/ci.yml)
[![Lean 4 Formal Proofs](https://github.com/adamhadani/mcmc-robust/actions/workflows/lean.yml/badge.svg)](https://github.com/adamhadani/mcmc-robust/actions/workflows/lean.yml)
[![Python Version](https://img.shields.io/badge/python->=3.13-blue.svg)](https://github.com/adamhadani/mcmc-robust)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A research library for NUTS on heavy-tailed targets. It calibrates a coordinate-wise arcsinh transform during warm-up and samples the exact transformed density. It is benchmarked against PyMC and BlackJAX NUTS.**

## What it does

**The problem.** Gradient-based samplers struggle when a target has polynomial (Student-*t*, Cauchy) tails. The gradient vanishes far out, so NUTS builds very long trajectories there and mixes poorly in the tails.

**The approach.** `mcmc-robust` maps each coordinate through $z_i = \mathop{\mathrm{arsinh}}((x_i - \mu_i)/R_i)$:
- $\mu_i$ and $R_i$ are estimated from early warm-up draws, using the median and a scaled IQR.
- NUTS then runs on the exact pullback density in $z$.
- In $z$, a polynomial tail becomes an exponential one with a constant restoring force.
- Because the map acts on each coordinate separately, one coordinate's excursion leaves the others unchanged.
- After warm-up the transform, metric and step size are frozen, so the retained draws come from a fixed, $\pi$-invariant kernel.

**Prior work.** Similar transforms have been used before, either radially (Johnson & Geyer, 2012) or for a single coordinate (Power & Vasdekis, 2025). Power & Vasdekis list coordinate-wise, learnable transformations as an open problem (§4.3). This library is a practical, automatically calibrated take on that problem. [REPORT.md](REPORT.md) sets out what is new and what is not.

Two implementations:
- `sample_blackjax_unified_nuts` (JAX/BlackJAX, recommended): Phase 1 calibration, then BlackJAX window adaptation and jit-compiled sampling.
- `UnifiedRobustNUTS` (NumPy): the reference implementation used in most of the benchmarks.

## Status of each component

| Component | Status | Evidence and limits |
| :--- | :--- | :--- |
| Calibrated arcsinh transform + NUTS | Main contribution | Gains are largest on heavy tails aligned with the coordinate axes and smaller on rotated ones. It costs a little on a rotated Gaussian. Where the model is known to be a Student-t scale mixture, rewriting it by hand is better (REPORT §5.1). |
| Funnels and hierarchical models | Not addressed | A coordinate-wise map cannot remove the coupling between a scale parameter and its children. Non-centre the model instead (REPORT §3.3, Table 2). |
| `GentleMALA` (soft-capped drift damping for stiff targets) | Experimental | Faster burn-in on a quartic target, but a narrower stable step-size range than tamed MALA. No guarantee for the coordinate-wise mode it runs in (REPORT §3.1). |
| First-order samplers with the transform (`UnifiedRobustSampler`, `AnisotropicTailMALA`) | Experimental | Not converged within the benchmark budgets (REPORT §3.2). |
| Lean 4 formalisation (`formal/`) | Supporting lemmas | Scalar facts about the transform, the damping and the calibration clip. It does not verify the samplers (REPORT §4). |

## Results at a glance

**Multi-seed ablation** (REPORT §5.1, Table A). The arms differ only in the coordinate transform or the model's parameterisation, and are paired by seed. This is the strongest evidence in the repository:

<!-- BEGIN GENERATED: readme-ablation -->
- **Arcsinh vs stock NUTS** (paired tail-ESS per gradient, median over seeds): Task 1 7.1× (wins on 10 of 10 seeds); Task 3 3.8× (wins on 10 of 10 seeds); rotated t (ν = 1) 5.2× (wins on 10 of 10 seeds); rotated t (ν = 2) 1.7× (wins on 10 of 10 seeds); isotropic t (ν = 2) 1.6× (wins on 10 of 10 seeds); rotated Gaussian 0.9× (wins on 0 of 10 seeds); anisotropic Gaussian 1.5× (wins on 10 of 10 seeds).
- **Rescaling alone** (`affine`: the same calibration without the arcsinh) vs stock: Task 1 0.9× (wins on 4 of 10 seeds); Task 3 0.9× (wins on 4 of 10 seeds); rotated t (ν = 1) 1.9× (wins on 7 of 10 seeds); rotated t (ν = 2) 1.0× (wins on 6 of 10 seeds); isotropic t (ν = 2) 1.1× (wins on 8 of 10 seeds); rotated Gaussian 1.0× (wins on 4 of 10 seeds); anisotropic Gaussian 1.2× (wins on 10 of 10 seeds).
- **Reparameterising by hand** (`scale_mixture`, where the model is a Student-t scale mixture) vs `arcsinh`: Task 1 0.2× (better than arcsinh on 0 of 10 seeds); Task 3 0.9× (better than arcsinh on 1 of 10 seeds); rotated t (ν = 1) 10.8× (better than arcsinh on 10 of 10 seeds); rotated t (ν = 2) 5.3× (better than arcsinh on 10 of 10 seeds); isotropic t (ν = 2) 2.5× (better than arcsinh on 10 of 10 seeds).
- **Seeds converged** (max R-hat at most 1.01) with `arcsinh`: Task 1: 9 of 10 seeds; Task 3: 10 of 10 seeds; rotated t (ν = 1): 2 of 10 seeds; rotated t (ν = 2): 10 of 10 seeds; isotropic t (ν = 2): 9 of 10 seeds; rotated Gaussian: 10 of 10 seeds; anisotropic Gaussian: 10 of 10 seeds; with `stock`: Task 1: 0 of 10 seeds; Task 3: 1 of 10 seeds; rotated t (ν = 1): 0 of 10 seeds; rotated t (ν = 2): 2 of 10 seeds; isotropic t (ν = 2): 4 of 10 seeds; rotated Gaussian: 10 of 10 seeds; anisotropic Gaussian: 10 of 10 seeds.
- **Seeds with any post-warm-up divergence** with `arcsinh`: Task 1: 0 of 10 seeds; Task 3: 0 of 10 seeds; rotated t (ν = 1): 0 of 10 seeds; rotated t (ν = 2): 0 of 10 seeds; isotropic t (ν = 2): 0 of 10 seeds; rotated Gaussian: 0 of 10 seeds; anisotropic Gaussian: 4 of 10 seeds.
- **Runs in which some coordinate never visits a tail** (no draw beyond its exact 0.01 or 0.99 quantile, which R-hat cannot detect) with `arcsinh`: none; with `stock`: Task 1: 1 of 10 seeds.
<!-- END GENERATED: readme-ablation -->

**Single-seed comparisons with PyMC NUTS and other samplers** (REPORT §5.2):

<!-- BEGIN GENERATED: readme-headline -->
- **Heavy tails, product form (REPORT Table 1).** On a d=10 regression in which 7 of the 10 coefficients sit at their independent Cauchy prior, `UnifiedRobustNUTS` has 23.5× the tail-ESS per gradient of BlackJAX NUTS and 4.3× that of PyMC NUTS (max R-hat 1.002 vs 1.037 and 1.013); its 99% quantile error is 8.1% against 45.3% and 41.6%.
- **Mixed tails (REPORT Table 3, d=20).** 2.3× the tail-ESS per gradient of BlackJAX NUTS, using 71,545 gradients against 133,000; 99% quantile error 5.8% against 11.3%.
- **Funnels are not solved (REPORT Table 2).** On Eight-Schools `UnifiedRobustNUTS` has 7 divergences against 68 (BlackJAX) and 148 (PyMC), but only 1.5% of its draws reach log τ < −1 (exact: 7.49%), so it has fewer divergences because it stays out of the neck. On Neal's funnel it reaches 22.2% neck mass (exact: 25.25%) at max R-hat 1.079, with 34 divergences against 3 for BlackJAX NUTS. Non-centre funnel models instead.
- **Applied models on simulated data (REPORT Table 5).** 4 of 30 runs converge; see the table before reading anything into efficiency differences.
- **Scope.** Each number is one seed (42) of 4 chains × 1,000 draws, with no error bars.
<!-- END GENERATED: readme-headline -->

Full tables, setup and limitations are in [REPORT.md](REPORT.md) §5 and §6.

## Installation

Requires Python 3.13 or newer (CI runs 3.13, 3.14 and 3.15):

```bash
git clone https://github.com/adamhadani/mcmc-robust.git
cd mcmc-robust
uv sync --all-extras --dev
```

## Usage

### JAX / BlackJAX (recommended)

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import t as student_t

from mcmc_robust import sample_blackjax_unified_nuts


def logdensity_fn(x):
    # Five Cauchy coordinates and five standard normal coordinates.
    return jnp.sum(student_t.logpdf(x[:5], df=1.0)) - 0.5 * jnp.sum(x[5:] ** 2)


result = sample_blackjax_unified_nuts(logdensity_fn, d=10, draws=500, tune=500, chains=4, seed=0)
print(result.samples.shape)  # (chains, draws, d), in the original coordinates
print("calibrated location:", result.transform.location)
print("calibrated radii:", result.transform.radii)
print("gradient evaluations:", result.grad_evals, "divergences:", result.divergences)
```

### NumPy reference sampler

```python
import numpy as np

from mcmc_robust import UnifiedRobustNUTS
from mcmc_robust.targets import HeterogeneousHeavyTailedTarget

# 10 Student-t coordinates (nu from 1.2 to 2.5) and 10 Gaussian coordinates.
target = HeterogeneousHeavyTailedTarget(d=20, heavy_coords=list(range(10)), nu_list=list(np.linspace(1.2, 2.5, 10)))

# burn_in must be at least warmup_steps, so the calibration phase runs and finishes before sampling.
sampler = UnifiedRobustNUTS(step_size=0.15, max_depth=10, adapt_step_size=True, adapt_tail_scale=True, warmup_steps=300)
samples, results = sampler.sample(
    x0=np.zeros(target.d), target=target, n_steps=800, burn_in=300, rng=np.random.default_rng(0)
)
print("draws:", samples.shape, "post-warm-up divergences:", sum(r.stats["divergent"] > 0 for r in results[300:]))
```

### PyMC models

```python
import numpy as np
import pymc as pm

from mcmc_robust import sample_pymc_model

rng = np.random.default_rng(1)
x = rng.normal(size=50)
y = 1.0 + 2.0 * x + rng.standard_t(df=2, size=50)

with pm.Model() as model:
    beta = pm.Cauchy("beta", alpha=0.0, beta=1.0, shape=2)
    pm.StudentT("y", nu=2, mu=beta[0] + beta[1] * x, sigma=1.0, observed=y)

idata = sample_pymc_model(model, sampler="unified_nuts", draws=300, tune=300, chains=2, random_seed=42)
print(idata.posterior["beta"].mean(dim=("chain", "draw")).values)
```

`uv run python example.py` runs a short head-to-head comparison between stock BlackJAX NUTS and the calibrated transform. It prints what it measured.

## Development

```bash
uv run pytest -v -W error            # full test suite, warnings as errors
uv run mypy src/mcmc_robust
uv run ruff check . && uv run ruff format --check .
cd formal && lake build              # Lean 4 proofs
```

Pull requests run only the tests a change can reach (`pytest-impacted`); see [AGENTS.md](AGENTS.md).

**Where the numbers come from.**
- Every number in this README and in REPORT.md is rendered from a results file in `reports/`, each written by a benchmark runner.
- `tests/unit/test_docs_consistency.py` fails when a rendered block is out of date, or when a ratio or percentage is typed by hand outside one.
- To regenerate the results and the docs:

```bash
uv run python -m mcmc_robust.bench.ablation_runner --full --workers 2
uv run python -m mcmc_robust.bench.sota_runner --full
uv run python -m mcmc_robust.bench.applied_runner
uv run python -m mcmc_robust.bench.render_docs --write
```

## License

MIT. See [LICENSE](LICENSE).
