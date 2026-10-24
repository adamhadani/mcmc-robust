# AGENTS.md — Engineering Guide for AI & Human Contributors

## Overview
`mcmc-robust` is a research and production-grade library providing novel gradient-based MCMC and No-U-Turn Samplers (NUTS) designed to address the dual geometric pathologies of target roughness (stiff super-linear gradients) and target flatness (sub-exponential polynomial heavy tails) without compromising benign efficiency.

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
- **Integrity Mandate**: No hardcoded numbers, mock facades, or dummy outputs in source code. All reported empirical results and unit tests must be genuine and reproducible. Every number quoted in `README.md` or `REPORT.md` must come from a committed results file in `reports/` written by a runner in `src/mcmc_robust/bench/`, or from a test. Ratios must be computable from the table they summarise, and claims must state their scope (seeds, chains, targets). A new feature's benchmark goes into a runner, not a one-off script.

---

## 2. Directory Layout

```
.
├── src/mcmc_robust/
│   ├── core/                  # Core abstractions (Sampler, ChainState, StepResult, TargetDistribution, adaptation)
│   ├── targets/               # Synthetic targets (benign, roughness, heavy-tailed, dual, funnels), applied
│   │                          #   models on simulated data (applied.py) and the target registry (registry.py)
│   ├── samplers/              # Reference samplers (MALA, GloballyTamedMALA, Barker, RWM, HMC, AutoMALA)
│   │   └── novel/             # GentleMALA, AnisotropicTailMALA, UnifiedRobustSampler, UnifiedRobustHMC, UnifiedRobustNUTS
│   ├── transforms/            # Diffeomorphisms (CoordinateWiseArcsinh, Stereographic, Radial)
│   ├── diagnostics/           # Rank-normalised ESS and R-hat, accuracy metrics (KS, W1, quantile error), tracker
│   ├── integrations.py        # PyMC bridge (PyMCModelTarget, sample_pymc_model) and JaxArcsinhTransform
│   ├── jax_unified_nuts.py    # Pure-JAX 3-phase NUTS (sample_blackjax_unified_nuts, calibrate_jax_arcsinh_transform)
│   └── bench/                 # Runners (sota_runner, applied_runner, sweep), CLI (python -m mcmc_robust.bench), plots
├── tests/                     # unit/, e2e/ (tiers 1-4) and tier5_adversarial/, all run with -W error
├── benchmarks/                # Standalone CLI runner script (run_benchmarks.py)
├── example.py                 # Minimal, runnable head-to-head comparison script (~10s)
├── formal/                    # Lean 4 formal proofs (McmcRobustFormal.lean, lakefile.toml)
├── reports/                   # Committed benchmark results (CSV, JSON) and figures: the source of every number in the docs
├── README.md, REPORT.md       # User-facing results and technical report (GitHub math rules: Section 5)
├── SOTA_ANALYSIS.md           # Plain-language comparison with the state of the art
├── .github/workflows/         # ci.yml (lint; tests on Python 3.13-3.15, impacted-only on PRs), lean.yml (lake build)
├── pyproject.toml             # Project metadata, dependencies, tool configuration (ruff, mypy, pytest, pytest-impacted)
└── .pre-commit-config.yaml    # Pre-commit hooks enforcing formatting, linting, typing, and impacted tests
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

**Regenerating reported results.** The tables in `README.md` and `REPORT.md` come from these commands. Rerun them whenever a sampler change affects a reported number, and commit the regenerated `reports/` files together with the doc edits:

```bash
uv run python -m mcmc_robust.bench.sota_runner --full   # Tables 1-4 -> reports/sota_comparison_results.{json,csv}
uv run python -m mcmc_robust.bench.applied_runner       # Table 5   -> reports/applied_benchmark_results.{json,csv}
uv run python -m mcmc_robust.bench.plot                 # figures   -> reports/figures/
```

---

## 4. Key Sampler APIs & Integration Contracts

- **`UnifiedRobustNUTS`**: No-U-Turn Sampler in calibrated coordinate-wise $\text{arcsinh}$ coordinates with a 3-phase warm-up (Phase 1 location and scale calibration $\to$ Phase 2 diagonal metric $\to$ Phase 3 dual-averaging step size). Stiffness damping is off by default (`gamma=0.0`); pass `gamma > 0` to enable it. `UnifiedRobustHMC` and `UnifiedRobustSampler` follow the same schedule.
- **Warm-up contract**: `Sampler.sample(x0, target, n_steps, burn_in, adapt=True)` calls `adapt()` only while `t < burn_in` and freezes the step size at `t = burn_in - 1`. Pass `burn_in >= warmup_steps`, or the 3-phase schedule is cut short and Phase 1 calibration may never run. After warm-up the kernel is fixed, so the retained draws come from a π-invariant chain.
- **`sample_blackjax_unified_nuts(logdensity_fn, d, draws=1000, tune=500, chains=4, seed=42, ...)`**: Pure-JAX pipeline. Phase 1 calibrates a `JaxArcsinhTransform` from pooled draws, BlackJAX window adaptation then runs per chain in $z$-space, and sampling uses `jax.jit` + `jax.vmap` + `jax.lax.scan`. It returns `JaxUnifiedNUTSResult`, whose `step_size` has shape `(chains,)` and `inverse_mass_matrix` shape `(chains, d)`, because every chain samples with its own adapted parameters.
- **`sample_pymc_model(model, sampler="unified_nuts", draws=1000, tune=500, ...)`**: Wraps continuous PyMC models with `DictToArrayBijection` unflattening and exact leapfrog gradient evaluation accounting. Returns an `arviz.DataTree` InferenceData object.
- **`JaxArcsinhTransform`**: Coordinate-wise $\text{arcsinh}$ diffeomorphism with `wrap_logdensity` for JAX/BlackJAX models, compatible with `jax.jit` and `jax.lax.scan`.

---

## 5. Writing Math in Markdown (GitHub Rendering)

GitHub renders math in `README.md` and `REPORT.md` with its own rules, and a formula that looks fine in a local preview or parses under MathJax can still show an error or raw TeX on github.com. These pitfalls were all observed on this repo's pages (October 2026, PR #5):

1. **No `\operatorname`.** GitHub rejects it ("The following macros are not allowed: operatorname"). Write `\mathop{\mathrm{median}}(x)`. `\phantom` is reported to be blocked too.
2. **Backslash escapes are applied inside plain `$...$`.** `\%` becomes `%`, a TeX comment, so `$7.49\%$` renders as "7.49". `\|` becomes `|`, so norms render as absolute values, and `\{ \}` become grouping braces, `\,` a comma, `\\` a single backslash. Use `\Vert` for norms. For any other backslash-punctuation command, use the ``$`...`$`` form, which GitHub parses like a code span and passes to TeX verbatim.
3. **`_` or `*` after punctuation can become emphasis.** In plain `$...$`, `\|x\|_2` or `(a, b)_i` can pair with a later `_` in the same paragraph, and the emphasis swallows the formula. Use ``$`...`$``.
4. **No math inside italics.** `$...$` inside `*...*` is never rendered. Do not italicise captions that contain math (this repo uses a bold label, `**Target:** ...`), and split the emphasis around a formula: `*An improved* $\widehat{R}$ *for ...*`.
5. **No `$` straight after a hyphen or dash.** `Student-$t$` and `$37$–$42$` stay as raw text. Write `Student-*t*`, and put a range in one span: `$37\text{–}42$`.
6. **Display math.** A `$$...$$` line inside a paragraph is not recognised, and its subscripts turn into italics. At the top level, use a fenced block (three backticks followed by `math`) with blank lines around it. Inside a list item GitHub shows that fence as a code block, so use ``$`\displaystyle ...`$`` there.
7. **Tables.** `|` ends a cell even inside math or code spans, so write `\vert` or `\Vert`, never a bare `|`.

To verify, open the file on its branch on github.com and look for red error text, leftover `$` or `\command` text, and `|x|` where `‖x‖` was meant. A local MathJax parse catches TeX errors, but none of the GitHub-specific behaviour above.
