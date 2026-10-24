# AGENTS.md — Engineering Guide for AI & Human Contributors

## Overview
`mcmc-robust` is a research and production-grade library providing novel gradient-based MCMC and No-U-Turn Samplers (NUTS) designed to eliminate the dual geometric pathologies of target roughness (stiff super-linear gradients) and target flatness (sub-exponential polynomial heavy tails) without compromising benign efficiency.

---

## 1. Environment & Architecture Invariants

- **Python Version**: Requires Python `>= 3.13`. Fully tested and supported on Python 3.13, 3.14, and 3.15.
- **Package Manager**: Use `uv` exclusively (`uv sync`, `uv run`).
- **JAX 64-bit Precision**: When JAX is imported, 64-bit float precision must be activated before importing `jax.numpy`:
  ```python
  import jax

  jax.config.update("jax_enable_x64", True)
  import jax.numpy as jnp
  ```
- **Formal Verification**: The Lean 4 formalization lives in `formal/` and builds with zero errors, zero warnings, and zero `sorry`/`admit` statements via `cd formal && lake build`.
- **Integrity Mandate**: No hardcoded numbers, mock facades, or dummy outputs in source code. All reported empirical results and unit tests must be genuine and reproducible.

---

## 2. Directory Layout

```
.
├── src/mcmc_robust/
│   ├── core/                  # Core abstractions (Sampler, ChainState, StepResult, TargetDistribution)
│   ├── targets/               # Benchmark distributions (benign, rough, heavy-tailed, funnels, dual)
│   ├── samplers/
│   │   ├── base/              # Reference samplers (MALA, TamedMALA, Barker, RWM)
│   │   └── novel/             # Novel algorithms (GentleMALA, AnisotropicTailMALA, UnifiedRobustNUTS)
│   ├── transforms/            # Diffeomorphisms (CoordinateWiseArcsinh, Stereographic, Radial)
│   ├── diagnostics/           # Diagnostics (metrics, tracker, ArviZ evaluation pipeline)
│   ├── integrations.py        # Framework bridges (PyMCModelTarget, sample_pymc_model, JaxArcsinhTransform)
│   └── bench/                 # Benchmark runners, CLI entry point, publication plotting
├── tests/unit/                # Unit test suite (>790 tests passing with -W error)
├── benchmarks/                # Standalone CLI runner script (run_benchmarks.py)
├── example.py                 # Minimal, runnable head-to-head comparison script (~10s)
├── formal/                    # Lean 4 formal proofs (McmcRobustFormal.lean, lakefile.lean)
├── reports/                   # Empirical benchmark datasets (CSV, JSON) and publication figures (PNG, PDF)
├── pyproject.toml             # Project metadata, dependencies, tool configurations (ruff, mypy)
└── .pre-commit-config.yaml    # Pre-commit hooks enforcing formatting, linting, typing, and tests
```

---

## 3. Standard Verification Commands

Contributors must verify the following checks pass cleanly before opening pull requests or committing code:

```bash
# 1. Full unit and integration test suite with strict warnings-as-errors:
uv run pytest -v -W error

# 2. Static type checking:
uv run mypy src/mcmc_robust

# 3. Code formatting and linting:
uv run ruff check .
uv run ruff format --check .

# 4. Lean 4 formal verification:
cd formal && lake build

# 5. Quick smoke test of example script:
uv run python example.py

# 6. Pre-commit suite:
uv run pre-commit run --all-files
```

**Impacted-test selection.** `pytest-impacted` (configured under `[tool.pytest.ini_options]` in `pyproject.toml`) selects the tests whose imports reach a changed file and marks the rest skipped:

- `uv run pytest -v -W error --impacted` covers uncommitted changes. This is what the pre-commit hook runs.
- `--impacted-git-mode=branch --impacted-base-branch=main` covers a whole branch. CI runs this on pull requests (against `origin/main`), and the full suite on every push to `main`.
- Changes to `uv.lock`, `pyproject.toml`, any `conftest.py` or `reports/*.json` select every test. Documentation and Lean-only changes select none.
- Import analysis cannot see data-file or subprocess dependencies. If a test starts reading a new generated file, add its glob to `impacted_invalidate_all`.

---

## 4. Key Sampler APIs & Integration Contracts

- **`UnifiedRobustNUTS`**: Symplectic Hamiltonian No-U-Turn Sampler operating in warped diffeomorphism coordinates with stiffness damping and 3-phase automated warmup (scale detection $\to$ metric preconditioner $\to$ dual averaging step size).
- **`sample_pymc_model(model, sampler="unified_nuts", draws=1000, tune=500, ...)`**: Wraps continuous PyMC models with `DictToArrayBijection` unflattening and exact leapfrog gradient evaluation accounting. Returns an `arviz.DataTree` InferenceData object.
- **`JaxArcsinhTransform`**: Coordinate-wise $\text{arcsinh}$ diffeomorphism with `wrap_logdensity` for JAX/BlackJAX models, compatible with `jax.jit` and `jax.lax.scan`.
