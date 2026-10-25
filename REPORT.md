# Calibrated Coordinate-Wise Arcsinh Transforms for NUTS on Heavy-Tailed Targets

This is the technical record for the code in this repository. Every result number in it is rendered from `reports/*.json` by `python -m mcmc_robust.bench.render_docs`. The prose outside the rendered blocks states no results; `tests/unit/test_docs_consistency.py` enforces both rules.

## Abstract

**The problem.** Gradient-based samplers mix slowly on targets with polynomial tails. The gradient vanishes far from the mode, so NUTS trajectories in the tails become long and expensive, and tail quantiles are estimated poorly.

**The remedy studied here.**
- A coordinate-wise map $z_i = \mathop{\mathrm{arsinh}}((x_i - \mu_i)/R_i)$.
- Its location and scale are calibrated from early warm-up draws.
- Standard NUTS adaptation and sampling follow, on the exact pullback density.

**Results.**
- **Multi-seed ablation (§5.1).** It compares the transform with three alternatives on heavy-tailed and Gaussian targets, with ratios paired by seed:
  - stock NUTS;
  - a rescaling-only control;
  - a by-hand scale-mixture reparameterisation.

  It shows where the gain comes from and where the transform is not the best choice.
- **Comparison with nutpie (§5.2).** nutpie is the closest existing tool. The comparison covers its default (a gradient-based mass matrix) and its normalizing-flow adaptation. The transform is ahead per gradient on axis-aligned heavy tails, and behind the flow on rotated targets.
- **Real posteriors (§5.3).** Every posteriordb posterior with reference draws, compared with stock NUTS on Stan's unconstrained space. There neither transform is a uniform improvement over stock NUTS, both add divergences on posteriors where stock NUTS has none, and the Tukey h-map's losses come where its guard sets h > 0.
- **Single-seed comparisons (§5.4).** On two synthetic targets whose heavy tails are aligned with the coordinate axes, the transform raises tail-ESS per gradient over stock BlackJAX and PyMC NUTS and lowers the error of the estimated 0.99 quantile (Tables 1 and 3).
- It does not resolve funnel geometry (Table 2).
- The repository's other components remain experimental (§3): a damped Langevin drift for stiff targets, and first-order samplers that use the transform.

§6 lists what the results do and do not establish.

## 1. Problem and positioning

### 1.1 Heavy tails and gradient-based MCMC

**Why the gradient vanishes.** If a coordinate has a polynomial tail, $`\pi(x_i) \propto \lvert x_i\rvert^{-\alpha_i}`$, its score $`\partial_{x_i} \log \pi \approx -\alpha_i / x_i`$ vanishes far out.

**What this does to each sampler.**
- MALA and the Barker proposal degenerate into random walks there.
- HMC and NUTS trajectories travel far before turning.
- NUTS trees then hit their depth cap.
- The tails are explored slowly: tail-ESS is low and extreme quantiles are noisy.

### 1.2 Related work

**Variable transformations for heavy tails.**
- **Johnson & Geyer (2012), Thm 3:** a radial map of exponential type turns multivariate-*t* tails into light tails for random-walk Metropolis. They note that a map of exponential type is necessary for such tails.
- **Radial maps:**
  - stereographic MCMC (Yang, Łatuszyński & Roberts, 2024) and its adaptive version (Bell, Łatuszyński & Roberts, 2024);
  - the transformed unadjusted Langevin algorithm (He, Balasubramanian & Erdogdu, 2024);
  - diffeomorphic MCMC (Brešar & Mijatović, 2026), which reports beating NUTS on heavy-tailed posteriordb posteriors.

  All of these couple the coordinates through $`\Vert x\Vert`$.
- **Power & Vasdekis (2025):**
  - In §4.2.1 they apply the signed log $`\mathop{\mathrm{sign}}(x)\log(1 + \lvert x\rvert)`$ to a one-dimensional Cauchy target.
  - In §4.3 they list as open problems:
    - (iii) "reliable", possibly coordinate-wise and learnable, transformations;
    - (iv) choosing the transformation's scale without inducing roughness.

**Learned transformations.**
- NeuTra (Hoffman et al., 2019) runs HMC through a normalizing flow.
- Lipschitz triangular flows cannot change the tail class of their base distribution (Jaini et al., 2020). This motivates tail-adaptive flows:
  - Laszkiewicz, Lederer & Fischer (2022);
  - Liang, Mahoney & Hodgkinson (2022);
  - Hickling & Prangle (2025), whose tail-transform layer acts coordinate by coordinate.
- nutpie's experimental normalizing-flow adaptation includes per-coordinate sinh-arcsinh layers, trained during warm-up, and then runs NUTS in the transformed space.

**Reparameterisation by hand.** When the tail family is known, a Cauchy coordinate can be rewritten as a scale mixture of normals or through a tangent transform (Betancourt, 2018).

**Theory.**
- Livingstone, Betancourt, Byrne & Girolami (2019, Cor. 2.3): HMC with a fixed integration time is geometrically ergodic for targets with tails like $`\exp(-\lvert x\rvert^\beta)`$ when $`1 \le \beta \le 2`$, and not when $`\beta < 1`$ or $`\beta > 2`$.
- This is the main reference point for §2.2.

### 1.3 What is and is not new here

**Not new:**
- using an exponential-type map to lighten polynomial tails (Johnson & Geyer);
- a signed-log-like map for a single coordinate (Power & Vasdekis);
- coordinate-wise tail layers inside flows (Hickling & Prangle; nutpie).

**New, as far as a literature check found:**
- **The method:** a closed-form arcsinh map per coordinate, calibrated by median and IQR during warm-up, followed by standard NUTS adaptation on the exact pullback. This is a concrete default answer to Power & Vasdekis's §4.3(iii)–(iv).
- **Two theorems, machine-checked in Lean (§4):**
  - No single coordinate map can bring both a Gaussian and a Cauchy coordinate into the potential class that published ergodicity results for HMC and NUTS assume.
  - Choosing the map per coordinate (a Tukey h-map for Student-*t* coordinates, an affine map for Gaussian ones) brings every coordinate into that class.
  - So the per-coordinate choice is both necessary and sufficient.
- **A matched-map option (§2.6):** `transform_kind="tukey_h"` fits that map per coordinate during warm-up, and keeps it affine unless a test finds a heavy tail.
- **A risk warning (§2.7)** for the regime in which coordinate-wise maps can miss tails silently.

**Not yet done:** a comparison with diffeomorphic MCMC. No public code for it was found (§5.2, §6).

The comparisons with reparameterisation by hand and with nutpie are in §5.1 and §5.2.

## 2. Method

### 2.1 Transform and pullback

For each coordinate $i$ with location $\mu_i$, scale $R_i > 0$ and mask $`m_i \in \{0, 1\}`$ (coordinates with $m_i = 0$ are left unchanged):

```math
z_i = \mathop{\mathrm{arsinh}}\left(\frac{x_i - \mu_i}{R_i}\right), \qquad x_i = \mu_i + R_i \sinh(z_i).
```

The sampler targets the exact pullback density, whose Jacobian is diagonal:

```math
\log \pi_Z(z) = \log \pi_X(x(z)) + \sum_{i:\, m_i = 1} \left[\log R_i + \log\cosh(z_i)\right],
```

```math
\nabla_{z_i} \log \pi_Z(z) = \sqrt{R_i^2 + (x_i - \mu_i)^2}\; \partial_{x_i} \log \pi_X(x) + \frac{x_i - \mu_i}{\sqrt{R_i^2 + (x_i - \mu_i)^2}}.
```

**Numerics.**
- $`\log\cosh z = \lvert z\rvert - \log 2 + \mathop{\mathrm{log1p}}(e^{-2\lvert z\rvert})`$ avoids overflow.
- The square root is computed with `np.hypot`.
- Each gradient of $\pi_Z$ costs one gradient of $\pi_X$ plus $O(d)$ work.

### 2.2 What the transform does to tails

**Heavy coordinates.** For a polynomial tail $`\pi(x_i) \propto \lvert x_i - \mu_i\rvert^{-\alpha_i}`$, the pullback score tends to a constant:

```math
\nabla_{z_i} \log \pi_Z(z) \to -(\alpha_i - 1)\,\mathop{\mathrm{sgn}}(z_i), \qquad \lvert z_i\rvert \to \infty.
```

The pullback tail is therefore Laplace-like. That is $\beta = 1$ in the class of §1.2, its lower edge.

**Light coordinates.** For a Gaussian coordinate $`x_i \sim \mathcal{N}(\mu_i, \sigma_i^2)`$:

```math
\log \pi_Z(z) = -\frac{R_i^2 \sinh^2(z_i)}{2\sigma_i^2} + \log\cosh(z_i) + \text{const},
```

which falls like $`-R_i^2 e^{2\lvert z_i\rvert} / (8\sigma_i^2)`$. That is doubly exponential, outside the class ($\beta > 2$).

**This is unavoidable for any single map.** By `no_common_map_gaussian_cauchy` (§4), no map that gives a Cauchy coordinate an exponential tail can keep a Gaussian coordinate's pullback potential quadratically bounded.

**Consequences for the implementation.**
- The NUTS samplers transform every coordinate unless a mask is passed.
- Light coordinates therefore get stiffer tails in $z$.
- The committed benchmarks do not yet measure the cost on light-tailed targets (§6).

### 2.3 Calibration

**Estimates.** From the Phase 1 draws $`x_{\cdot i}`$:

```math
\hat\mu_i = \mathrm{median}(x_{\cdot i}), \qquad
\hat\sigma_i = \frac{q_{0.75}(x_{\cdot i} - \hat\mu_i) - q_{0.25}(x_{\cdot i} - \hat\mu_i)}{1.3489795}, \qquad
R_i = \min\left(R_{\max}, \max\left(R_{\min}, c\,\hat\sigma_i\right)\right),
```

The defaults are:
- $c = 1.5$;
- $R_{\min} = 0.1$;
- $R_{\max}$ = 10⁴ in NumPy and 100 in the JAX pipeline.

**Fallback.** If $`\hat\sigma_i \le R_{\min}`$, the scale falls back to the larger of the normalised MAD and the 0.9 quantile of $`\lvert x_{\cdot i} - \hat\mu_i\rvert`$.

**Translation equivariance.** Shifting a coordinate by $c$ shifts $\hat\mu_i$ by $c$ and leaves the map unchanged; `shiftedForwardMap_translation_invariant` in Lean.

**Where it lives.** `CoordinateWiseArcsinhTransform.compute_phase1_calibration` implements this rule, and every sampler uses it.

### 2.4 Warm-up and exactness

**`UnifiedRobustNUTS` (NumPy)** splits warm-up into three phases:
1. **Phase 1, the first 15% (at least 10 steps):** collects draws and calibrates $(\mu, R)$ at its last step.
2. **Phase 2, the next 65%:** optionally adapts a diagonal metric in $z$ with Welford variance updates (`adapt_preconditioner`).
3. **Phase 3, the remaining 20%:** tunes the step size by dual averaging. The averaged step size is frozen at the last warm-up step.

The dual-averaging state is reset at each phase boundary.

**`sample_blackjax_unified_nuts` (JAX):**
- **Phase 1:** BlackJAX window adaptation on the initial transform for the first 15% of tuning steps. The transform is calibrated from the second half of those draws, pooled across chains.
- **Then:** each chain runs BlackJAX window adaptation in $z$ for the rest of the tuning budget, and samples with its own step size and diagonal metric.

**Exactness.** After warm-up, the transform, metric and step size are fixed. The retained draws are then NUTS draws targeting $\pi_Z$, mapped through $x = x(z)$, so their law is $\pi_X$.
- This rests on the standard correctness of NUTS and on the change-of-variables formula. Neither is machine-checked here.
- `Sampler.sample` adapts only while $t$ < `burn_in`. Pass `burn_in` at least `warmup_steps`, or calibration may never run.

### 2.5 Configurations used in the benchmarks

The benchmarked `UnifiedRobustNUTS` is not a single configuration.

**SOTA Tasks 1 and 3** (`bench/sota_runner.py`):
- `step_size=0.15, max_depth=10, adapt_tail_scale=True`;
- `gamma = 0`, so no drift damping;
- no metric adaptation, i.e. an identity metric in $z$;
- the step size is still adapted by dual averaging.

**Tasks 2 and 4:** `gamma` = 0.2 and 0.5, which switches on the damped leapfrog force of §3.1.

**Applied runner** (`bench/applied_runner.py`): `gamma = 0.15` with metric adaptation.

**Stock baselines:**
- BlackJAX NUTS uses window adaptation;
- PyMC NUTS uses its defaults;
- both target an acceptance rate of 0.8.

**Own implementations, not the published methods:**
- `AutoMALA` differs from the published method in its step-selection rule and tuning.
- "Stereographic MALA" is a MALA in stereographic coordinates. It is not the published stereographic random-walk or bouncy-particle samplers.

### 2.6 The Tukey h-map option (`transform_kind="tukey_h"`)

**The map.** For coordinate i, with $h_i \ge 0$:

```math
x_i = \mu_i + R_i\, z_i \exp\left(\tfrac{1}{2} h_i z_i^2\right), \qquad
\log \frac{dx_i}{dz_i} = \log R_i + \tfrac{1}{2} h_i z_i^2 + \log\left(1 + h_i z_i^2\right).
```

- With $h_i = 0$ the map is affine.
- Its inverse is $`z = u \exp(-\tfrac{1}{2} W(h u^2))`$, with $`u = (x - \mu)/R`$ and $W$ the principal Lambert W function. W is computed from $\log y$, so it stays exact for very large arguments.
- Where the forward map overflows, the wrapped log density is $-\infty$ rather than NaN.

**What it does to tails.**
- A polynomial tail of index α pulls back to a Gaussian tail with variance $`1/(\alpha h)`$, whatever the true α. Misspecifying α only rescales that variance.
- §4 machine-checks that the pullback potential then lies in the window that published HMC/NUTS ergodicity results assume (`hMapStudentT_inTailWindow`).
- With $h = 0$, a Gaussian coordinate stays inside the window (`affineGauss_inTailWindow`).

**Calibration**, from the Phase 1 draws:
- **Location:** $\mu_i$ is the median.
- **Shape and scale:** $(R_i, h_i)$ are fitted by maximum likelihood of the Tukey h-transformed Gaussian, i.e. Lambert W × Gaussian (Goerg, 2015), with $h_i$ capped at `TUKEY_H_MAX_H`.
- **The guard:** a coordinate keeps $h_i = 0$, with the Gaussian scale estimate, unless a likelihood-ratio test rejects $h_i = 0$ at level `tukey_h_level`.
  - The p-value uses the boundary null mixture $`\tfrac12\chi^2_0 + \tfrac12\chi^2_1`$ (Self & Liang, 1987).
  - **Dependent draws.** Phase 1's draws are autocorrelated and come from several chains. Near the null, the statistic is the squared standardised sum of the efficient score of h, $`\tfrac12 He_4(z)`$ with $`He_4(z) = z^4 - 6z^2 + 3`$, and dependence inflates the variance of that sum by n/ESS. The test therefore multiplies the statistic by $`\min(1, \mathrm{ESS}/n)`$, where ESS is the split-chain effective sample size of $`He_4(z)`$ (`diagnostics.ess.compute_mean_ess`).
  - The guard makes the map's choice per coordinate. That is what the impossibility theorem of §4 says is necessary.
  - `tukey_h_level=0` never rejects, so every coordinate keeps h = 0. The `tukey_h0` arm of Tables A and C runs that setting as the control for the guard's decisions.
- **The R clip:** R is clipped to `[min_radius, max_radius]`, and h is rescaled to keep $h/R^2$ fixed, so the clip does not change the map's shape.

**Limits of the guard.**
- It separates Gaussian tails from Tukey-h tails, not polynomial tails from exponential ones. A Laplace-tailed coordinate is often warped, which makes its pullback lighter than Gaussian.
- The ESS correction covers autocorrelation, and chains that disagree about where the mass is. It does not see a transient that every chain shares, nor chains that differ only in spread. If Phase 1 ends before the chains reach the posterior, its pooled draws can look heavy-tailed with no sign of it in any chain. Table C measures this by running the guard on exact posterior draws as well (§5.3).

**Default.** arcsinh remains the default `transform_kind`.
- Table A (§5.1) compares the two maps on synthetic targets.
- Table C (§5.3) compares them on real posteriors. There `tukey_h` is close to stock NUTS where the guard keeps h = 0 everywhere, and loses where it sets h > 0.

### 2.7 Tail-alignment warning

**What it does.** `sample_blackjax_unified_nuts` checks its post-warm-up draws with `tail_alignment_diagnostic`. It stores the report as `result.tail_alignment`, and emits `TailAlignmentWarning` when heavy tails appear to be shared across coordinates. Pass `check_tail_alignment=False` to skip the check.

**How it works:**
1. A coordinate is *heavy* when its Hill tail index, estimated from the draws above the 0.95 quantile of $`\lvert x_i - \mathop{\mathrm{median}}\rvert`$, is below 4.
2. For each pair of heavy coordinates it computes the co-exceedance $`\chi_{ij}`$: the fraction of a coordinate's top exceedances during which the other coordinate also exceeds its own 0.95 quantile. Under independence this is about $1 - u$ at level $u$, and it equals 1 for identical coordinates.
3. It warns when the largest $`\chi_{ij}`$ exceeds 0.3.
4. Below 50 exceedances per coordinate, i.e. fewer than 1,000 draws at this level, it gives no verdict.
5. The thresholds were chosen on pilot seeds that are not among the seeds of Table A.

**What it does not do:**
- **A risk flag, not a detector.** It flags the condition under which coordinate-wise maps can miss tails without any R-hat warning (§5.1): heavy tails along directions that are not coordinate axes. It is not a detector of missed tails.
- **Fires on elliptical targets too.** It also fires on elliptical targets, where the arcsinh pipeline may still converge.
- **Silent when the bulk is never left.** It stays silent if a sampler never leaves the bulk.

Table A reports, for each target and arm, the seeds on which it fired. The findings also report how often it fired on runs that never visited a tail.

## 3. Experimental components

### 3.1 `GentleMALA`: damped drift for stiff targets

**Definition.** With step size $h$, diagonal scale $s$ and sensitivity $\gamma$, the stiffness measure and damping factor are:

```math
\kappa_h(x) = \gamma h \Vert s \odot \nabla \log \pi(x)\Vert_2 \;\; \text{(global mode)}, \qquad
\kappa_{h,i}(x) = \gamma h s_i \lvert \partial_i \log \pi(x)\rvert \;\; \text{(coordinate mode)},
```

```math
\rho(\kappa) = \left(1 + \kappa + (\kappa/\kappa_{\max})^2\right)^{-1/2}, \qquad
y = x + h\, s^2 \odot \rho(\kappa_h(x)) \nabla \log \pi(x) + \sqrt{2h}\, s \odot \xi,
```

followed by an exact Metropolis–Hastings correction, at one gradient per step.

**What is proved** (Lean, §4):
- for any finite $\kappa_{\max}$, the deterministic step is bounded by $`s\kappa_{\max}/\gamma`$;
- in the global (radial) mode, the chain contracts without overshooting the origin from radius $`r \ge s\kappa_{\max}/\gamma`$;
- for pure square-root damping ($\kappa_{\max} = \infty$) on $`U(x) = \Vert x\Vert^4/4`$, a step does not overshoot when $`0 < r < 4\gamma/h`$. That is a sufficient condition, not a characterisation.

**What is not proved or not supported:**
- **No guarantee for the coordinate mode.** The benchmarks run the coordinate mode, for which no result is stated.
- **Still bounded drift.** For finite $\kappa_{\max}$ the drift is bounded, so this is a member of the tamed-drift family (Roberts & Tweedie, 1996; Brosse et al., 2019; Lytras & Sabanis, 2025). It does not escape the over-conservatism that Power & Vasdekis describe in §3.3.

**Step-size sweep on the quartic target.**

<!-- BEGIN GENERATED: langevin-sweep -->
Quartic target U(x) = ‖x‖⁴/4 started at ‖x₀‖ = 10√d; 25 step sizes log-spaced on [0.0001, 10]. A step size is stable when the run neither diverges nor accepts fewer than 1 in 100 proposals. Source: `reports/sweep_results.json`.

| d | Sampler | Largest stable h | h_max / h_min | Median burn-in steps |
| :--- | :--- | ---: | ---: | ---: |
| 2 | MALA | 0.00464 | 46 | 91 |
| 2 | Tamed MALA | 1.47 | 14,678 | 44 |
| 2 | Barker | 10 | 100,000 | 38 |
| 2 | GentleMALA | 0.562 | 5,623 | 16 |
| 2 | UnifiedRobustSampler | 10 | 100,000 | 11.5 |
| 10 | MALA | 0.000681 | 7 | 377.5 |
| 10 | Tamed MALA | 0.909 | 9,085 | 104 |
| 10 | Barker | 10 | 100,000 | 102.5 |
| 10 | GentleMALA | 0.0825 | 825 | 36.5 |
| 10 | UnifiedRobustSampler | 0.562 | 5,623 | 16 |
| 50 | MALA | 0.000162 | 2 | not reached |
| 50 | Tamed MALA | 2.37 | 23,714 | not reached |
| 50 | Barker | 10 | 100,000 | 317 |
| 50 | GentleMALA | 0.0121 | 121 | 160 |
| 50 | UnifiedRobustSampler | 0.562 | 5,623 | 92 |
<!-- END GENERATED: langevin-sweep -->

**Benign Gaussians.** At this budget no sampler reaches the convergence threshold on these targets, so the efficiency ratios below are indicative only.

<!-- BEGIN GENERATED: langevin-benign -->
d = 50, budget 3,000 gradients per chain, 4 chains, seed 42; "tail" starts at ‖x₀‖ = 10√d. Source: `reports/benchmark_results.json`.

| Gaussian | Start | Sampler | ESS/grad | Tail-ESS/grad | R-hat | Converged | ESS/grad vs MALA |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| isotropic | bulk | MALA | 0.0536 | 0.0705 | 1.037 | no | 100.0% |
| isotropic | bulk | Barker | 0.0244 | 0.0186 | 1.054 | no | 45.6% |
| isotropic | bulk | Tamed MALA | 0.016 | 0.0149 | 1.104 | no | 29.8% |
| isotropic | bulk | GentleMALA | 0.034 | 0.029 | 1.057 | no | 63.5% |
| isotropic | bulk | UnifiedRobustSampler | 0.0606 | 0.0465 | 1.037 | no | 113.1% |
| isotropic | tail | MALA | 0.0607 | 0.0312 | 1.032 | no | 100.0% |
| isotropic | tail | Barker | 0.0274 | 0.0244 | 1.068 | no | 45.1% |
| isotropic | tail | Tamed MALA | 0.0223 | 0.0251 | 1.077 | no | 36.7% |
| isotropic | tail | GentleMALA | 0.032 | 0.0358 | 1.066 | no | 52.7% |
| isotropic | tail | UnifiedRobustSampler | 0.0529 | 0.0472 | 1.040 | no | 87.1% |
| anisotropic | bulk | MALA | 0.0022 | 0.00542 | 1.891 | no | 100.0% |
| anisotropic | bulk | Barker | 0.00229 | 0.00494 | 2.109 | no | 104.0% |
| anisotropic | bulk | Tamed MALA | 0.00167 | 0.00487 | 3.134 | no | 76.1% |
| anisotropic | bulk | GentleMALA | 0.00203 | 0.00558 | 1.915 | no | 92.2% |
| anisotropic | bulk | UnifiedRobustSampler | 0.00201 | 0.00577 | 1.885 | no | 91.5% |
| anisotropic | tail | MALA | 0.00196 | 0.00493 | 2.124 | no | 100.0% |
| anisotropic | tail | Barker | 0.00203 | 0.00489 | 2.006 | no | 103.7% |
| anisotropic | tail | Tamed MALA | 0.0015 | 0.00488 | 6.997 | no | 76.5% |
| anisotropic | tail | GentleMALA | 0.00221 | 0.00487 | 2.077 | no | 112.9% |
| anisotropic | tail | UnifiedRobustSampler | 0.00255 | 0.00579 | 1.649 | no | 130.3% |
<!-- END GENERATED: langevin-benign -->

### 3.2 First-order samplers with the transform

- **What they are:**
  - `UnifiedRobustSampler` combines the transform with `GentleMALA`'s damping;
  - `AnisotropicTailMALA` applies the transform only to coordinates classified as heavy-tailed by a Hill-type test.
- **Why their numbers are not evidence:** in Tables 1–5 these samplers mostly do not converge within their budgets. Their ESS and efficiency ratios therefore don't support any claim.

### 3.3 Funnels

**Why the transform can't help.** A coordinate-wise map has a diagonal Jacobian, so it cannot remove the dependence of $`x_i \mid v \sim \mathcal{N}(0, e^{v})`$ on $v$ in a centred funnel. Table 2 shows the effect.

**Eight-Schools.** `UnifiedRobustNUTS` records fewer divergences than stock NUTS, but it also places far less mass in the neck than the exact posterior does.

**Neal's funnel.** It reaches deeper into the neck than BlackJAX NUTS, without converging, and with more divergences.

**A learned partial non-centring** was explored and not merged (PR #3).
- With a linear scale function, it is the variationally inferred parameterisation of Gorinova, Moore & Hoffman (2020), with the parameter fitted by regression.
- Its accuracy depended on the funnel's apex being at the origin.

**Recommendation.** Use a non-centred parameterisation when the model structure is known (Papaspiliopoulos, Roberts & Sköld, 2007). Otherwise, consider samplers built for multiscale geometry: WALNUTS, GIST, or delayed-rejection HMC.

## 4. Formal verification (Lean 4)

**Build.** `cd formal && lake build` checks the proofs. The build has no `sorry` and no `admit`, and the axiom audit at the end of the file pins each theorem to `propext`, `Classical.choice` and `Quot.sound`.

**Scope.** The theorems are about scalar building blocks:
- the arcsinh map's invertibility, derivative and translation equivariance;
- the damping inequalities of §3.1;
- the calibration clip of §2.3;
- Student-*t* score bounds;
- the algebraic identity behind Metropolis–Hastings detailed balance.

**Not covered:** no transition kernel, no change-of-variables argument and no NUTS property is formalised. `mh_detailed_balance` states $`a \min(1, b/a) = b \min(1, a/b)`$ for positive reals, the algebra behind detailed balance, not a fact about any sampler here.

**What the tail-map sections prove.** These sections concern the coordinate maps themselves, not the samplers:

- **The arcsinh envelope does not depend on calibration** (`arcsinhCauchyPullback_exp_envelope`). For a Cauchy marginal, every location error and every scale ratio give a pullback density between two multiples of $`e^{-\lvert z\rvert}`$. Only the constants depend on the calibration; the tail rate does not.
- **The Tukey h-map gives a Gaussian envelope.** Let $`x = \mu + R z e^{z^2/(2c)}`$.
  - The map is a smooth increasing bijection with an explicit log-Jacobian (`hMap_bijective`, `log_hMapDeriv`).
  - It pulls a Cauchy marginal back to a density between two multiples of $`e^{-z^2/(2c)}`$ (`hMapCauchyPullback_gauss_lower`, `hMapCauchyPullback_gauss_upper`).
  - Rescaling $z$ leaves the map unchanged as long as $`R^2 c`$ is fixed (`hMap_rescale_invariant`). So once the metric adapts, only $`R^2 c`$ matters.
- **No single map can serve both a Gaussian and a Cauchy coordinate** (`no_common_map_gaussian_cauchy`).
  - The theorem: no increasing differentiable map $g$ with $g \to \infty$ makes both the Cauchy pullback potential at least linear ($`U_C(z) \ge a z - b`$ with $a > 0$) and the Gaussian pullback potential at most quadratic ($`U_G(z) \le A z^2 + B`$).
  - Why it matters: published geometric-ergodicity results for HMC and NUTS assume a potential between these two bounds. So the transform's shape must be chosen per coordinate.
  - The arcsinh case: arcsinh satisfies the Cauchy side, and its Gaussian pullback has no quadratic bound (`arcsinh_gauss_pullback_not_quadratic`).
- **Matched maps put every coordinate inside the window** (`hMapStudentT_inTailWindow`, `tailMatched_inTailWindow`).
  - Under the h-map, the pullback potential of a Student-*t* marginal lies in the window $`a z - b \le U(z) \le A z^2 + B`$, for any $\nu > 0$ and any location offset.
  - Under an affine map, a Gaussian's pullback potential does too.
  - So choosing the map per coordinate (h-map for heavy-tailed coordinates, affine for Gaussian ones) puts every coordinate of such a product target inside the window.
  - Swapping the two maps takes both coordinates out of it (`tail_matched_selection_forced`).
  - Together with `no_common_map_gaussian_cauchy`, this says the selection is both necessary and sufficient for the window.
- **Where leapfrog is unstable under arcsinh** (`leapfrog_stable_of_standardized_le`, `leapfrog_unstable_of_standardized_ge`). For a Gaussian coordinate pulled back by arcsinh, these bracket the region where the linear leapfrog stability condition $`\varepsilon^2 U'' \le 4`$ fails, in terms of the standardised coordinate.

**Paper-level results, not machine-checked.** A write-up on the exploration branch, [`explore/tail-transform-theory`](https://github.com/adamhadani/mcmc-robust/blob/explore/tail-transform-theory/explore/theory/THEORY.md), relies on cited theorems. It covers:
- which marginal tails each map handles;
- the step from the machine-checked window condition to geometric ergodicity of HMC and NUTS on the product target, via the published theorems (Durmus, Moulines & Saksman 2020; Gruffaz et al. 2026), which are cited, not formalised;
- an open problem: the Laplace boundary, where arcsinh places heavy coordinates and no NUTS theorem applies.

<!-- BEGIN GENERATED: lean-inventory -->
103 theorems in `formal/McmcRobustFormal.lean`, listed under the file's own section headings:

| Section (as titled in the Lean file) | Theorems |
| :--- | :--- |
| Exact Invertibility of Shift-Equivariant Arcsinh/Sinh Diffeomorphism | `arcsinh_inv_right`, `arcsinh_inv_left`, `scaled_invertibility_roundtrip_forward`, `scaled_invertibility_roundtrip_inverse`, `shifted_invertibility_roundtrip_forward`, `shifted_invertibility_roundtrip_inverse`, `shiftedForwardMap_translation_invariant`, `inverseMap_jacobian_factor_pos` |
| Derivative Identity and Metric Scale Strict Positivity | `arsinh_deriv_eq`, `metric_scale_pos`, `metric_scale_le_one`, `hasDerivAt_forwardMap`, `forwardMap_metric_pos` |
| Pure Square-Root Attenuation & Exact Discrete Basin Boundary | `rho_pos`, `rho_le_one`, `rho_strictAntiOn`, `pureSqrtStepSq_lt_div_gamma`, `pureSqrt_quartic_discrete_basin_bound` |
| Soft-Capped Gentle Attenuation Global Drift & Non-Overshoot | `rhoSoft_pos`, `rhoSoft_le_one`, `softCap_dimensionless_drift_sq_lt`, `softCap_dimensionless_drift_lt`, `softCappedDriftStep_lt_bound`, `softCapped_global_contraction_no_overshoot` |
| Metropolis-Hastings Kernel Reversibility (Detailed Balance) | `mhAccept_bounds`, `mh_detailed_balance` |
| Applied Heavy-Tailed & Shrinkage Posterior Score Bounds | `studentTNegLogKernel_nonneg`, `hasDerivAt_studentTNegLogKernel`, `studentTScore_mul_le`, `cauchyShrinkagePenalty_nonneg`, `studentTScore_zero` |
| Pure-JAX 3-Phase Scale & Location Calibration Bounds | `calibratedRadius_ge_min`, `calibratedRadius_pos`, `calibratedRadius_le_max`, `calibratedRadius_mono`, `calibrated_shifted_roundtrip_forward`, `calibrated_shifted_roundtrip_inverse` |
| Tail Envelopes of Arcsinh Pullbacks of Cauchy Marginals | `one_add_sinh_sq`, `arcsinhCauchyPullback_eq_sech`, `half_exp_abs_le_cosh`, `cosh_le_exp_abs`, `sech_envelope`, `calibration_kernel_upper`, `calibration_kernel_lower`, `arcsinhCauchyPullback_envelope`, `arcsinhCauchyPullback_exp_envelope` |
| Tukey h-Map: Derivative, Log-Jacobian, Bijectivity and Gaussian Envelopes | `hasDerivAt_hMap`, `hMapDeriv_pos`, `log_hMapDeriv`, `hMap_strictMono`, `hMap_surjective`, `hMap_bijective`, `hMap_rescale`, `hMap_rescale_invariant`, `hMapCauchyPullback_eq`, `hMapCauchyPullback_gauss_lower`, `hMapCauchyPullback_gauss_upper` |
| Impossibility of a Common Tail Map for Gaussian and Cauchy Marginals | `pullback_potential_difference`, `deriv_bound_of_cauchy_envelope`, `div_one_add_sq_le_arctan`, `cauchy_pullback_tail_bound`, `exp_growth_of_cauchy_pullback_envelope`, `gaussPullback_lower_of_cauchy_envelope`, `no_common_map_gaussian_cauchy`, `arcsinh_cauchy_pullback_linear_lower`, `arcsinh_gauss_pullback_not_quadratic`, `window_necessity` |
| Leapfrog Stability Region of the Arcsinh Pullback of a Gaussian | `arcsinhGaussPotential_one_eq`, `hasDerivAt_arcsinhGaussPotential`, `hasDerivAt_arcsinhGaussGrad`, `arcsinhGaussCurv_eq`, `inv_cosh_sq_pos_le_one`, `leapfrog_stable_of_standardized_le`, `leapfrog_unstable_of_standardized_ge`, `arcsinhGaussCurv_mono_abs`, `arcsinhGaussCurv_zero` |
| Tail Window of the h-Map Pullback of a Student-t Marginal | `inTailWindow_of_quadratic_bounds`, `inTailWindow_of_abs_sub_le`, `inTailWindow_comp_mul`, `hMapStudentTPotential_eq_pullback`, `hMapStudentTPotential_neg`, `hMapStudentTPotential_rescale`, `exp_le_one_add_mul_exp`, `hMap_sq`, `hMapStudentTPotential_quadratic_lower`, `hMapStudentTPotential_quadratic_upper`, `hMapStudentT_inTailWindow`, `abs_log_one_add_sq_shift_le`, `hMapStudentTOffsetPotential_eq_pullback`, `hMapStudentTOffsetPotential_sub_le`, `hMapStudentTOffset_inTailWindow` |
| Tail-Matched Map Selection: Gaussian, Cauchy and Student-t Coordinates | `affineGauss_inTailWindow`, `identityGauss_inTailWindow`, `hMapCauchyPullback_eq_exp_neg`, `hMapCauchyPotential_sandwich`, `hMapCauchy_inTailWindow`, `hMapStudentTPotential_one_eq`, `not_inTailWindow_gauss_and_cauchy`, `tendsto_hMap_atTop`, `hMapGauss_not_inTailWindow`, `affineCauchy_not_inTailWindow`, `tailMatched_inTailWindow`, `tail_matched_selection_forced` |
<!-- END GENERATED: lean-inventory -->

## 5. Results

### 5.1 Multi-seed ablation (Table A)

`bench/ablation_runner.py --full` writes `reports/ablation_results.json`.

**Arms.** It runs every arm through the same BlackJAX pipeline:
- `stock`;
- `stock_ta95`: stock with a higher target acceptance rate;
- `arcsinh`: the transform;
- `affine`: the same Phase 1 calibration of μ and R with a linear map, which separates rescaling from tail warping;
- `scale_mixture`, on the Student-t-family targets: the practitioner's rewrite of a Student-t or Cauchy component as a non-centred Gaussian scale mixture, x = μ + √λ·L·η with an inverse-gamma λ;
- `tukey_h`: the Tukey h-map with its guard (§2.6);
- `tukey_h0`: `tukey_h` with the guard never rejecting, so h = 0 on every coordinate. It differs from `tukey_h` only in the guard's h > 0 decisions.

**Targets.**
- Tasks 1 and 3 of §5.4;
- rotated and isotropic multivariate *t* targets, whose heavy tails are not aligned with the axes;
- a rotated product of Cauchys, whose heavy tails lie along rotated axes without being elliptical;
- two Gaussians, which measure the cost on targets stock NUTS already handles.

**Metrics.**
- Each arm's tail-ESS per gradient is divided by stock's on the same seed.
- Tail accuracy is the error in the probability below the exact 0.01 and 0.99 marginal quantiles.

<!-- BEGIN GENERATED: ablation-table -->
10 seeds per cell; each run has 4 chains × 1,000 draws after 500 warm-up steps. Every arm runs BlackJAX NUTS with window adaptation through `sample_blackjax_unified_nuts`; arms differ only in the coordinate transform (or, for `scale_mixture`, in the model's parameterisation), and share each seed's initial positions and random key, so ratios are paired by seed. Source: `reports/ablation_results.json` (run 2026-10-25, commit e80f4a7).

Tail-probability error is the mean over coordinates and p ∈ {0.01, 0.99} of |P̂(x_i < q_p) − p| at the exact marginal quantile q_p; it is less noisy than a relative error on the 0.99 quantile.

Targets: **Task 1**: Sparse regression, Cauchy(0, 1) priors, 3 active of 10 coefficients (SOTA Task 1); **Task 3**: Product form: t(nu) on 10 coordinates (nu = 1.2 to 2.5), N(0, 1) on 10 (SOTA Task 3); **rotated t (ν = 1)**: Multivariate Cauchy (t, nu = 1), d = 10, rotated scale with condition number 25; **rotated t (ν = 2)**: Multivariate t, nu = 2, d = 10, rotated scale with condition number 25; **isotropic t (ν = 2)**: Isotropic multivariate t, nu = 2, d = 10; **rotated product of Cauchys**: Rotated product of Cauchys, x = Q diag(s) u with u iid Cauchy and s = geomspace(1, 5), d = 10; **rotated Gaussian**: Rotated Gaussian, d = 20, covariance condition number 100; **anisotropic Gaussian**: Axis-aligned Gaussian, d = 20, standard deviations 0.1 to 10.

Arms: `stock`: BlackJAX NUTS, window adaptation, target acceptance 0.8; `stock_ta95`: The same with target acceptance 0.95; `arcsinh`: NUTS on the calibrated coordinate-wise arcsinh pullback; `affine`: Same pipeline and Phase 1 calibration, z = (x - mu) / R, no arcsinh; `scale_mixture`: Stock NUTS on the non-centred Gaussian scale-mixture reparameterisation; `tukey_h`: Same pipeline and Phase 1, Tukey h-map x = mu + R z exp(h z^2 / 2); (R, h) by Tukey-h maximum likelihood, h = 0 unless a likelihood-ratio test (deflated by the effective sample size of the Phase 1 draws) rejects it; `tukey_h0`: tukey_h with the guard never rejecting (level 0): h = 0 on every coordinate.

| Target | Arm | Tail-ESS/grad vs stock, median [IQR] | Seeds better than stock | Seeds converged | Median gradients | Median divergences | Median tail-probability error | Seeds with tail-alignment warning |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task 1 | `stock` | 1 (reference) | – | 0 of 10 | 635,440 | 0 | 0.0034 | 0 of 10 |
| Task 1 | `stock_ta95` | 0.55× [0.45, 1.06] | 3 of 10 | 1 of 10 | 1,124,198 | 0 | 0.0044 | 0 of 10 |
| Task 1 | `arcsinh` | 7.14× [6.61, 13.13] | 10 of 10 | 9 of 10 | 148,826 | 0 | 0.0018 | 0 of 10 |
| Task 1 | `affine` | 0.89× [0.61, 1.24] | 4 of 10 | 0 of 10 | 683,634 | 0 | 0.0049 | 0 of 10 |
| Task 1 | `scale_mixture` | 1.09× [0.94, 1.89] | 7 of 10 | 8 of 10 | 1,061,258 | 205.5 | 0.0015 | 0 of 10 |
| Task 1 | `tukey_h` | 17.21× [14.78, 32.84] | 10 of 10 | 10 of 10 | 91,530 | 0 | 0.0014 | 0 of 10 |
| Task 1 | `tukey_h0` | 0.87× [0.50, 1.38] | 4 of 10 | 0 of 10 | 1,006,456 | 0 | 0.0042 | 0 of 10 |
| Task 3 | `stock` | 1 (reference) | – | 1 of 10 | 197,496 | 0 | 0.0030 | 0 of 10 |
| Task 3 | `stock_ta95` | 0.55× [0.52, 0.96] | 2 of 10 | 3 of 10 | 297,914 | 0 | 0.0026 | 0 of 10 |
| Task 3 | `arcsinh` | 3.80× [3.12, 4.07] | 10 of 10 | 10 of 10 | 74,596 | 0 | 0.0015 | 0 of 10 |
| Task 3 | `affine` | 0.92× [0.63, 1.17] | 4 of 10 | 3 of 10 | 242,389 | 0 | 0.0030 | 0 of 10 |
| Task 3 | `scale_mixture` | 3.45× [2.81, 3.94] | 10 of 10 | 10 of 10 | 82,176 | 0 | 0.0014 | 0 of 10 |
| Task 3 | `tukey_h` | 5.84× [4.85, 6.62] | 10 of 10 | 10 of 10 | 50,356 | 0 | 0.0015 | 0 of 10 |
| Task 3 | `tukey_h0` | 0.90× [0.74, 1.17] | 3 of 10 | 1 of 10 | 189,387 | 0 | 0.0030 | 0 of 10 |
| rotated t (ν = 1) | `stock` | 1 (reference) | – | 0 of 10 | 195,961 | 0 | 0.0043 | 10 of 10 |
| rotated t (ν = 1) | `stock_ta95` | 0.75× [0.55, 1.47] | 4 of 10 | 0 of 10 | 314,761 | 0 | 0.0035 | 10 of 10 |
| rotated t (ν = 1) | `arcsinh` | 5.24× [2.16, 8.88] | 10 of 10 | 2 of 10 | 121,078 | 0 | 0.0046 | 10 of 10 |
| rotated t (ν = 1) | `affine` | 1.92× [0.97, 2.88] | 7 of 10 | 0 of 10 | 178,581 | 0 | 0.0044 | 10 of 10 |
| rotated t (ν = 1) | `scale_mixture` | 44.71× [30.11, 75.54] | 10 of 10 | 10 of 10 | 51,242 | 0 | 0.0015 | 10 of 10 |
| rotated t (ν = 1) | `tukey_h` | 4.10× [1.96, 5.30] | 10 of 10 | 4 of 10 | 133,994 | 0 | 0.0033 | 10 of 10 |
| rotated t (ν = 1) | `tukey_h0` | 1.23× [0.68, 1.61] | 5 of 10 | 0 of 10 | 196,524 | 0 | 0.0039 | 10 of 10 |
| rotated t (ν = 2) | `stock` | 1 (reference) | – | 2 of 10 | 117,246 | 0 | 0.0030 | 10 of 10 |
| rotated t (ν = 2) | `stock_ta95` | 0.63× [0.53, 0.76] | 0 of 10 | 4 of 10 | 178,166 | 0 | 0.0024 | 10 of 10 |
| rotated t (ν = 2) | `arcsinh` | 1.73× [1.44, 1.86] | 10 of 10 | 10 of 10 | 99,168 | 0 | 0.0019 | 10 of 10 |
| rotated t (ν = 2) | `affine` | 1.03× [0.95, 1.22] | 6 of 10 | 3 of 10 | 114,478 | 0 | 0.0031 | 10 of 10 |
| rotated t (ν = 2) | `scale_mixture` | 8.99× [7.62, 10.17] | 10 of 10 | 10 of 10 | 43,884 | 0 | 0.0013 | 10 of 10 |
| rotated t (ν = 2) | `tukey_h` | 1.54× [1.15, 1.69] | 9 of 10 | 10 of 10 | 106,958 | 0 | 0.0019 | 10 of 10 |
| rotated t (ν = 2) | `tukey_h0` | 1.06× [0.78, 1.09] | 6 of 10 | 4 of 10 | 111,123 | 0 | 0.0037 | 10 of 10 |
| isotropic t (ν = 2) | `stock` | 1 (reference) | – | 4 of 10 | 64,814 | 0 | 0.0028 | 10 of 10 |
| isotropic t (ν = 2) | `stock_ta95` | 0.77× [0.68, 0.79] | 1 of 10 | 4 of 10 | 104,074 | 0 | 0.0017 | 10 of 10 |
| isotropic t (ν = 2) | `arcsinh` | 1.59× [1.46, 1.78] | 10 of 10 | 9 of 10 | 58,962 | 0 | 0.0019 | 10 of 10 |
| isotropic t (ν = 2) | `affine` | 1.13× [1.03, 1.19] | 8 of 10 | 4 of 10 | 65,280 | 0 | 0.0018 | 10 of 10 |
| isotropic t (ν = 2) | `scale_mixture` | 4.24× [3.87, 4.50] | 10 of 10 | 10 of 10 | 43,618 | 0 | 0.0012 | 10 of 10 |
| isotropic t (ν = 2) | `tukey_h` | 1.20× [1.05, 1.51] | 8 of 10 | 10 of 10 | 73,107 | 0 | 0.0023 | 10 of 10 |
| isotropic t (ν = 2) | `tukey_h0` | 1.27× [0.88, 1.42] | 6 of 10 | 6 of 10 | 61,646 | 0 | 0.0024 | 10 of 10 |
| rotated product of Cauchys | `stock` | 1 (reference) | – | 0 of 10 | 520,939 | 0 | 0.0068 | 10 of 10 |
| rotated product of Cauchys | `stock_ta95` | 0.69× [0.49, 1.02] | 3 of 10 | 0 of 10 | 739,482 | 0 | 0.0074 | 10 of 10 |
| rotated product of Cauchys | `arcsinh` | 1.58× [0.93, 3.24] | 7 of 10 | 0 of 10 | 342,392 | 0 | 0.0100 | 10 of 10 |
| rotated product of Cauchys | `affine` | 0.82× [0.62, 1.61] | 4 of 10 | 0 of 10 | 532,026 | 0 | 0.0068 | 10 of 10 |
| rotated product of Cauchys | `scale_mixture` | 72.59× [56.99, 87.81] | 10 of 10 | 10 of 10 | 92,334 | 0 | 0.0017 | 10 of 10 |
| rotated product of Cauchys | `tukey_h` | 1.44× [0.92, 2.08] | 7 of 10 | 0 of 10 | 360,266 | 0 | 0.0100 | 10 of 10 |
| rotated product of Cauchys | `tukey_h0` | 1.27× [0.55, 1.98] | 6 of 10 | 0 of 10 | 526,338 | 0 | 0.0083 | 10 of 10 |
| rotated Gaussian | `stock` | 1 (reference) | – | 10 of 10 | 155,246 | 0 | 0.0013 | 0 of 10 |
| rotated Gaussian | `stock_ta95` | 0.71× [0.69, 0.73] | 0 of 10 | 10 of 10 | 209,912 | 0 | 0.0013 | 0 of 10 |
| rotated Gaussian | `arcsinh` | 0.85× [0.84, 0.90] | 0 of 10 | 10 of 10 | 177,917 | 0 | 0.0013 | 0 of 10 |
| rotated Gaussian | `affine` | 1.00× [0.95, 1.02] | 4 of 10 | 10 of 10 | 156,798 | 0 | 0.0015 | 0 of 10 |
| rotated Gaussian | `tukey_h` | 0.98× [0.96, 1.00] | 3 of 10 | 10 of 10 | 159,620 | 0 | 0.0012 | 0 of 10 |
| rotated Gaussian | `tukey_h0` | 0.99× [0.96, 1.00] | 3 of 10 | 10 of 10 | 159,620 | 0 | 0.0013 | 0 of 10 |
| anisotropic Gaussian | `stock` | 1 (reference) | – | 10 of 10 | 121,870 | 0 | 0.0014 | 0 of 10 |
| anisotropic Gaussian | `stock_ta95` | 0.72× [0.72, 0.74] | 0 of 10 | 10 of 10 | 164,086 | 0 | 0.0015 | 0 of 10 |
| anisotropic Gaussian | `arcsinh` | 1.55× [1.52, 1.58] | 10 of 10 | 10 of 10 | 90,602 | 0 | 0.0014 | 0 of 10 |
| anisotropic Gaussian | `affine` | 1.23× [1.20, 1.24] | 10 of 10 | 10 of 10 | 99,100 | 0 | 0.0013 | 0 of 10 |
| anisotropic Gaussian | `tukey_h` | 1.76× [1.73, 1.80] | 10 of 10 | 10 of 10 | 69,622 | 0 | 0.0014 | 0 of 10 |
| anisotropic Gaussian | `tukey_h0` | 1.76× [1.73, 1.77] | 10 of 10 | 10 of 10 | 69,507 | 0 | 0.0014 | 0 of 10 |
<!-- END GENERATED: ablation-table -->

<!-- BEGIN GENERATED: ablation-findings -->
- **Arcsinh vs stock NUTS** (paired tail-ESS per gradient, median over seeds): Task 1 7.1× (wins on 10 of 10 seeds); Task 3 3.8× (wins on 10 of 10 seeds); rotated t (ν = 1) 5.2× (wins on 10 of 10 seeds); rotated t (ν = 2) 1.7× (wins on 10 of 10 seeds); isotropic t (ν = 2) 1.6× (wins on 10 of 10 seeds); rotated product of Cauchys 1.6× (wins on 7 of 10 seeds); rotated Gaussian 0.9× (wins on 0 of 10 seeds); anisotropic Gaussian 1.5× (wins on 10 of 10 seeds).
- **Rescaling alone** (`affine`: the same calibration without the arcsinh) vs stock: Task 1 0.9× (wins on 4 of 10 seeds); Task 3 0.9× (wins on 4 of 10 seeds); rotated t (ν = 1) 1.9× (wins on 7 of 10 seeds); rotated t (ν = 2) 1.0× (wins on 6 of 10 seeds); isotropic t (ν = 2) 1.1× (wins on 8 of 10 seeds); rotated product of Cauchys 0.8× (wins on 4 of 10 seeds); rotated Gaussian 1.0× (wins on 4 of 10 seeds); anisotropic Gaussian 1.2× (wins on 10 of 10 seeds).
- **Reparameterising by hand** (`scale_mixture`, where the model is a Student-t scale mixture) vs `arcsinh`: Task 1 0.2× (better than arcsinh on 0 of 10 seeds); Task 3 0.9× (better than arcsinh on 1 of 10 seeds); rotated t (ν = 1) 10.8× (better than arcsinh on 10 of 10 seeds); rotated t (ν = 2) 5.3× (better than arcsinh on 10 of 10 seeds); isotropic t (ν = 2) 2.5× (better than arcsinh on 10 of 10 seeds); rotated product of Cauchys 45.6× (better than arcsinh on 10 of 10 seeds).
- **Tukey h-map** (`tukey_h`) vs `arcsinh`: Task 1 2.3× (better than arcsinh on 10 of 10 seeds); Task 3 1.6× (better than arcsinh on 10 of 10 seeds); rotated t (ν = 1) 1.0× (better than arcsinh on 5 of 10 seeds); rotated t (ν = 2) 0.9× (better than arcsinh on 4 of 10 seeds); isotropic t (ν = 2) 0.8× (better than arcsinh on 2 of 10 seeds); rotated product of Cauchys 0.9× (better than arcsinh on 5 of 10 seeds); rotated Gaussian 1.1× (better than arcsinh on 10 of 10 seeds); anisotropic Gaussian 1.1× (better than arcsinh on 10 of 10 seeds); vs stock: Task 1 17.2× (wins on 10 of 10 seeds); Task 3 5.8× (wins on 10 of 10 seeds); rotated t (ν = 1) 4.1× (wins on 10 of 10 seeds); rotated t (ν = 2) 1.5× (wins on 9 of 10 seeds); isotropic t (ν = 2) 1.2× (wins on 8 of 10 seeds); rotated product of Cauchys 1.4× (wins on 7 of 10 seeds); rotated Gaussian 1.0× (wins on 3 of 10 seeds); anisotropic Gaussian 1.8× (wins on 10 of 10 seeds).
- **The guard's h > 0 decisions** (`tukey_h` vs `tukey_h0`, the same pipeline with h = 0 on every coordinate): Task 1 27.6× (better than h = 0 on 10 of 10 seeds); Task 3 5.0× (better than h = 0 on 10 of 10 seeds); rotated t (ν = 1) 3.6× (better than h = 0 on 10 of 10 seeds); rotated t (ν = 2) 1.5× (better than h = 0 on 9 of 10 seeds); isotropic t (ν = 2) 1.1× (better than h = 0 on 5 of 10 seeds); rotated product of Cauchys 1.1× (better than h = 0 on 6 of 10 seeds); rotated Gaussian 1.0× (better than h = 0 on 0 of 10 seeds); anisotropic Gaussian 1.0× (better than h = 0 on 2 of 10 seeds).
- **Seeds converged** (max R-hat at most 1.01) with `arcsinh`: Task 1: 9 of 10 seeds; Task 3: 10 of 10 seeds; rotated t (ν = 1): 2 of 10 seeds; rotated t (ν = 2): 10 of 10 seeds; isotropic t (ν = 2): 9 of 10 seeds; rotated product of Cauchys: 0 of 10 seeds; rotated Gaussian: 10 of 10 seeds; anisotropic Gaussian: 10 of 10 seeds; with `stock`: Task 1: 0 of 10 seeds; Task 3: 1 of 10 seeds; rotated t (ν = 1): 0 of 10 seeds; rotated t (ν = 2): 2 of 10 seeds; isotropic t (ν = 2): 4 of 10 seeds; rotated product of Cauchys: 0 of 10 seeds; rotated Gaussian: 10 of 10 seeds; anisotropic Gaussian: 10 of 10 seeds.
- **Seeds with any post-warm-up divergence** with `arcsinh`: Task 1: 0 of 10 seeds; Task 3: 0 of 10 seeds; rotated t (ν = 1): 0 of 10 seeds; rotated t (ν = 2): 0 of 10 seeds; isotropic t (ν = 2): 0 of 10 seeds; rotated product of Cauchys: 4 of 10 seeds; rotated Gaussian: 0 of 10 seeds; anisotropic Gaussian: 4 of 10 seeds.
- **Runs in which some coordinate never visits a tail** (no draw beyond its exact 0.01 or 0.99 quantile, which R-hat cannot detect) with `arcsinh`: rotated product of Cauchys: 10 of 10 seeds; with `tukey_h`: rotated product of Cauchys: 10 of 10 seeds; with `stock`: Task 1: 1 of 10 seeds; rotated product of Cauchys: 8 of 10 seeds.
- **Tail-alignment warning** (`TailAlignmentWarning`, computed on each run's own draws; it flags heavy tails shared across coordinates) with `arcsinh`: Task 1: 0 of 10 seeds; Task 3: 0 of 10 seeds; rotated t (ν = 1): 10 of 10 seeds; rotated t (ν = 2): 10 of 10 seeds; isotropic t (ν = 2): 10 of 10 seeds; rotated product of Cauchys: 10 of 10 seeds; rotated Gaussian: 0 of 10 seeds; anisotropic Gaussian: 0 of 10 seeds. Among runs of any arm in which some coordinate never visits a tail, it fired on 52 of 57.
<!-- END GENERATED: ablation-findings -->

**How to read Table A.**
- **Tail warping vs rescaling.** Comparing the `affine` and `arcsinh` rows separates the two, because both arms use the same calibrated μ and R.
- **Why the scale mixture is strong on pure *t* targets.** Without a likelihood, the scale-mixture rewrite of a multivariate *t* is a Gaussian in η times a one-dimensional log-inverse-gamma factor, which NUTS samples easily.
- **Why it struggles on Task 1.** There the strongly identified coefficients make the non-centred rewrite a narrow ridge, hence its divergences.
- **When to prefer which.** The scale mixture needs the model to be known as a scale mixture, while the arcsinh pipeline is a black box.
- **Tails that are never visited.** On the rotated product of Cauchys, a tail excursion runs along a direction that is not a coordinate axis.
  - Under the coordinate-wise map, that direction becomes a ridge in $z$ that narrows exponentially, so NUTS cannot follow it.
  - R-hat stays low because every chain misses the same region.
  - The last finding above counts the affected runs for each arm. The scale-mixture rewrite samples the independent Cauchy factors directly, so this does not happen to it.

### 5.2 Comparison with nutpie (Table B)

`bench/competitor_runner.py` writes `reports/competitor_results.json`. It needs the optional `competitors` extra.

**How the arms run.**
- nutpie runs through `nutpie.compiled_pyfunc.from_pyfunc` on the same JAX log density as the other arms.
- Gradient evaluations are counted by wrapping the log density.
- Flow training reuses the gradients already stored from warm-up, so it costs wall time but no target evaluations.

**Caveats when reading the table.**
- nutpie's flow mode counts a transition as divergent at an energy error of 20, against 1000 for the other arms, so its divergence counts are not comparable.
- Diffeomorphic MCMC (Brešar & Mijatović, 2026) was not run. No official code was found, and the paper does not specify its tuning fully enough to re-implement it faithfully; the results file records what was searched.

<!-- BEGIN GENERATED: competitor-table -->
5 seeds; each run has 4 chains × 1,000 draws. Ratios are paired by seed against stock BlackJAX NUTS; gradient counts include warm-up, and nutpie's flow training reuses warm-up gradients, so it adds wall time but no gradient evaluations. Wall-time ratios favour the jit-compiled BlackJAX arms over nutpie's Python callbacks. Source: `reports/competitor_results.json`.

Arms: `stock_nuts`: BlackJAX NUTS, Stan-style window adaptation (reference); `arcsinh_nuts`: sample_blackjax_unified_nuts: BlackJAX NUTS on the calibrated coordinate-wise arcsinh pullback (this repo); `nutpie_diag`: nutpie default: NUTS with gradient-based diagonal mass-matrix adaptation; `nutpie_flow`: nutpie normalizing-flow adaptation (default flow settings), same warm-up length as the other arms; `nutpie_flow_tune1500`: nutpie normalizing-flow adaptation with nutpie's default flow warm-up of 1500 draws (all counted).

Excluded: `dcs_bresar_mijatovic_2026` (not run: no official code found).

| Target | Arm | Tail-ESS/grad vs stock, median [IQR] | Tail-ESS/s vs stock, median | Seeds converged | Max R-hat | Divergences, all seeds | Tail-probability error at 0.99, median (MCSE) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task 1 | `stock_nuts` | 1 (reference) | 1 | 0 of 5 | 1.062 | 0 | 0.0056 (0.0045) |
| Task 1 | `arcsinh_nuts` | 14.86× [10.30, 18.35] | 3.35× | 4 of 5 | 1.010 | 0 | 0.0016 (0.0021) |
| Task 1 | `nutpie_diag` | 5.61× [5.37, 5.96] | 0.87× | 0 of 5 | 1.030 | 0 | 0.0043 (0.0045) |
| Task 1 | `nutpie_flow` | 6.45× [5.90, 9.56] | 0.07× | 1 of 5 | 1.029 | 302 | 0.0054 (0.0046) |
| Task 1 | `nutpie_flow_tune1500` | 6.45× [6.01, 8.54] | 0.04× | 2 of 5 | 1.057 | 855 | 0.0030 (0.0036) |
| rotated Gaussian | `stock_nuts` | 1 (reference) | 1 | 5 of 5 | 1.004 | 0 | 0.0014 (0.0017) |
| rotated Gaussian | `arcsinh_nuts` | 0.84× [0.81, 0.84] | 0.85× | 5 of 5 | 1.002 | 0 | 0.0014 (0.0017) |
| rotated Gaussian | `nutpie_diag` | 1.29× [1.26, 1.32] | 1.32× | 5 of 5 | 1.003 | 0 | 0.0013 (0.0018) |
| rotated Gaussian | `nutpie_flow` | 2.50× [2.43, 2.64] | 0.05× | 5 of 5 | 1.005 | 0 | 0.0013 (0.0018) |
| rotated Gaussian | `nutpie_flow_tune1500` | 1.83× [1.78, 1.87] | 0.02× | 5 of 5 | 1.007 | 0 | 0.0015 (0.0017) |
| Task 3 | `stock_nuts` | 1 (reference) | 1 | 2 of 5 | 1.091 | 0 | 0.0026 (0.0031) |
| Task 3 | `arcsinh_nuts` | 5.99× [4.19, 6.25] | 2.05× | 5 of 5 | 1.008 | 0 | 0.0014 (0.0019) |
| Task 3 | `nutpie_diag` | 4.45× [2.95, 5.01] | 2.01× | 0 of 5 | 1.030 | 0 | 0.0026 (0.0028) |
| Task 3 | `nutpie_flow` | 4.50× [3.23, 4.63] | 0.06× | 3 of 5 | 1.013 | 67 | 0.0023 (0.0025) |
| Task 3 | `nutpie_flow_tune1500` | 4.19× [2.61, 4.50] | 0.03× | 5 of 5 | 1.004 | 344 | 0.0019 (0.0019) |
| rotated t (ν = 1) | `stock_nuts` | 1 (reference) | 1 | 0 of 5 | 1.805 | 0 | 0.0063 (0.0024) |
| rotated t (ν = 1) | `arcsinh_nuts` | 1.83× [1.61, 13.91] | 1.57× | 1 of 5 | 1.022 | 0 | 0.0023 (0.0032) |
| rotated t (ν = 1) | `nutpie_diag` | 1.05× [0.77, 5.78] | 1.37× | 0 of 5 | 1.650 | 0 | 0.0037 (0.0046) |
| rotated t (ν = 1) | `nutpie_flow` | 9.02× [5.98, 36.21] | 0.23× | 4 of 5 | 1.021 | 12 | 0.0021 (0.0036) |
| rotated t (ν = 1) | `nutpie_flow_tune1500` | 9.88× [7.25, 75.75] | 0.14× | 5 of 5 | 1.006 | 5 | 0.0020 (0.0019) |
<!-- END GENERATED: competitor-table -->

<!-- BEGIN GENERATED: competitor-findings -->
- **Arcsinh vs nutpie's default** (gradient-based mass matrix), tail-ESS per gradient: Task 1 2.1× (better on 5 of 5 seeds); rotated Gaussian 0.6× (better on 0 of 5 seeds); Task 3 1.4× (better on 5 of 5 seeds); rotated t (ν = 1) 2.1× (better on 5 of 5 seeds).
- **Arcsinh vs nutpie's normalizing-flow adaptation** (same warm-up length), tail-ESS per gradient: Task 1 1.9× (better on 5 of 5 seeds); rotated Gaussian 0.3× (better on 0 of 5 seeds); Task 3 1.3× (better on 5 of 5 seeds); rotated t (ν = 1) 0.3× (better on 0 of 5 seeds).
- **Arcsinh vs the flow with nutpie's own default warm-up** (1,500 draws, all counted): Task 1 1.9× (better on 5 of 5 seeds); rotated Gaussian 0.4× (better on 0 of 5 seeds); Task 3 1.4× (better on 5 of 5 seeds); rotated t (ν = 1) 0.2× (better on 0 of 5 seeds).
- **Seeds converged** (max R-hat at most 1.01): `arcsinh_nuts` Task 1: 4 of 5; rotated Gaussian: 5 of 5; Task 3: 5 of 5; rotated t (ν = 1): 1 of 5; `nutpie_flow` Task 1: 1 of 5; rotated Gaussian: 5 of 5; Task 3: 3 of 5; rotated t (ν = 1): 4 of 5.
<!-- END GENERATED: competitor-findings -->

### 5.3 Real posteriors from posteriordb (Table C)

`bench/posteriordb_runner.py` writes `reports/posteriordb_results.json`. It needs the optional `posteriordb` extra (BridgeStan) and a checkout of posteriordb, which `--fetch` makes at the pinned commit.

**Why this table.** Tables A and B use synthetic targets built to have heavy tails. Table C asks what the transforms do on posteriors that people fit. It runs every posterior in posteriordb (Magnusson et al., 2025) that has reference draws; none is selected or excluded by its results. It is the evidence for or against making `tukey_h` the default (§2.6).

**How the arms run.**
- BridgeStan (Roualdes et al., 2023) compiles each Stan model. `stan_bridge.bridgestan_logdensity_fn` passes its log density and gradient on Stan's unconstrained space to `sample_blackjax_unified_nuts`. Every arm samples the same density; only `transform_kind` differs.
- Initial positions follow Stan: uniform on (−2, 2) for each unconstrained coordinate, redrawn until the density is finite.
- Draws are mapped to the parameters that have reference draws, transformed parameters included. ESS, R-hat and accuracy are measured on those.
- The arms are stock NUTS, `arcsinh`, `tukey_h` and `tukey_h0`. `tukey_h0` is `tukey_h` with h held at 0, so the paired ratio of the two is the effect of the guard's h > 0 decisions on that posterior.

**Accuracy against the reference.** posteriordb's reference draws are themselves a Monte Carlo sample. `diagnostics/tail_accuracy.reference_agreement` compares tail probabilities at the reference's quantiles, and means, as z-scores that count the Monte Carlo error of both the run and the reference.

**Three diagnostics in each row.**
- *Where Phase 1 calibrated:* the distance between Phase 1's location and scale and the run's own post-warm-up draws. Phase 1 calibrates on the first part of warm-up (§2.4); when the chains have not reached the posterior by then, the transform is centred and scaled on the transient.
- *What the `tukey_h` guard decided:* how many unconstrained coordinates it gave h > 0.
- *What the guard decides on exact draws:* the same guard on as many of posteriordb's reference draws as Phase 1 calibrates from, mapped to the unconstrained space. A coordinate with h > 0 from Phase 1's draws but not from exact draws was warped because of where Phase 1's chains were, not because of the posterior's tails.

<!-- BEGIN GENERATED: posteriordb-table -->
Every posterior of posteriordb (commit `91027f9cdd13`) with reference draws: 58 posteriors, 10 seeds each. Each run has 4 chains × 1,000 draws after 500 warm-up steps, on Stan's unconstrained space through BridgeStan; arms share each seed's Stan-style initial positions and random key, so ratios are paired by seed. ESS and accuracy are measured on the parameters that have reference draws; the ratios compare the worst parameter's tail-ESS per gradient. Source: `reports/posteriordb_results.json` (run 2026-10-24, commit 6a860fd).

A run disagrees with the reference when a tail-probability z-score (at the reference's 1%, 5%, 95% and 99% quantiles) or a mean z-score exceeds 5 in absolute value; z counts the Monte Carlo error of the run and of the reference. Not converged: max R-hat above 1.01. d is the unconstrained dimension. The last column runs `tukey_h`'s guard on as many reference draws as Phase 1 calibrates from (unconstrained; n/a where the reference lacks a parameter of the model).

| Posterior | d | Stock min-tail-ESS/grad, median | `arcsinh` ÷ stock (seeds better) | `tukey_h` ÷ stock (seeds better) | `tukey_h` ÷ `arcsinh` (seeds better) | `tukey_h` ÷ `tukey_h0` (seeds better) | Median divergences (stock / arcsinh / tukey_h) | Seeds disagreeing with reference (stock / arcsinh / tukey_h) | Seeds not converged (stock / arcsinh / tukey_h) | `tukey_h` coordinates with h > 0, median | The same guard on exact draws, median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | 1 | 0.111 | 0.57× (0 of 10) | 0.93× (5 of 10) | 1.45× (10 of 10) | n/a | 0 / 6 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| Rate_2_data-Rate_2_model | 2 | 0.105 | 0.78× (1 of 10) | 0.90× (2 of 10) | 1.24× (10 of 10) | n/a | 0 / 2 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| Rate_3_data-Rate_3_model | 1 | 0.113 | 0.61× (0 of 10) | 0.99× (4 of 10) | 1.77× (10 of 10) | n/a | 0 / 10.5 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| Rate_5_data-Rate_5_model | 1 | 0.106 | 0.68× (0 of 10) | 1.04× (6 of 10) | 1.49× (10 of 10) | n/a | 0 / 13.5 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| arK-arK | 7 | 0.0125 | 0.98× (4 of 10) | 0.99× (5 of 10) | 1.03× (6 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| arma-arma11 | 4 | 0.000113 | 1.21× (8 of 10) | 0.52× (3 of 10) | 0.38× (2 of 10) | n/a | 122 / 0 / 49.5 | 5 / 2 / 3 | 7 / 6 / 8 | 3 | n/a |
| bball_drive_event_0-hmm_drive_0 | 6 | 0.000334 | 95.00× (8 of 10) | 47.48× (8 of 10) | 0.50× (4 of 10) | n/a | 19 / 4.5 / 3 | 3 / 1 / 2 | 7 / 3 / 2 | 2 | n/a |
| bball_drive_event_1-hmm_drive_1 | 6 | 0.016 | 1.05× (6 of 10) | 0.63× (1 of 10) | 0.54× (2 of 10) | n/a | 0 / 0 / 94 | 0 / 0 / 0 | 0 / 0 / 0 | 5 | n/a |
| diamonds-diamonds | 26 | 0.000328 | 0.99× (4 of 10) | 1.06× (8 of 10) | 1.06× (6 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 20.5 | n/a |
| dogs-dogs | 3 | 0.026 | 0.89× (2 of 10) | 0.98× (3 of 10) | 1.05× (6 of 10) | n/a | 0 / 0 / 0 | 1 / 0 / 0 | 0 / 1 / 0 | 0 | n/a |
| dogs-dogs_hierarchical | 2 | 0.0683 | 0.78× (0 of 10) | 0.98× (3 of 10) | 1.23× (10 of 10) | n/a | 0 / 1.5 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| earnings-earn_height | 3 | 0.00229 | 0.77× (4 of 10) | 0.03× (1 of 10) | 0.10× (3 of 10) | n/a | 0 / 0 / 120 | 0 / 2 / 6 | 0 / 3 / 9 | 1 | n/a |
| earnings-log10earn_height | 3 | 0.00246 | 1.32× (9 of 10) | 1.12× (9 of 10) | 0.89× (4 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 0 | n/a |
| earnings-logearn_height | 3 | 0.00276 | 1.07× (6 of 10) | 1.07× (7 of 10) | 1.07× (6 of 10) | n/a | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 1 / 0 | 1 | n/a |
| earnings-logearn_height_male | 4 | 0.00219 | 1.01× (7 of 10) | 0.91× (3 of 10) | 0.84× (2 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 1 | 2 | n/a |
| earnings-logearn_interaction | 5 | 0.00111 | 0.95× (3 of 10) | 0.88× (3 of 10) | 0.92× (4 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 1 / 1 | 2 | n/a |
| earnings-logearn_interaction_z | 5 | 0.0291 | 0.93× (1 of 10) | 0.59× (0 of 10) | 0.65× (0 of 10) | n/a | 0 / 0 / 7.5 | 0 / 0 / 0 | 0 / 0 / 1 | 5 | n/a |
| earnings-logearn_logheight_male | 4 | 0.000546 | 1.01× (5 of 10) | 1.03× (6 of 10) | 0.98× (5 of 10) | n/a | 0 / 16 / 0 | 0 / 3 / 1 | 0 / 2 / 1 | 0 | n/a |
| eight_schools-eight_schools_centered | 10 | 0.000229 | 5.23× (9 of 10) | 1.49× (7 of 10) | 0.29× (4 of 10) | n/a | 150.5 / 57.5 / 141.5 | 10 / 10 / 10 | 10 / 10 / 10 | 5 | n/a |
| eight_schools-eight_schools_noncentered | 10 | 0.0342 | 1.12× (6 of 10) | 1.23× (7 of 10) | 1.06× (6 of 10) | n/a | 0 / 25 / 77 | 0 / 0 / 0 | 0 / 0 / 0 | 1 | n/a |
| garch-garch11 | 4 | 0.0295 | 0.98× (4 of 10) | 0.95× (2 of 10) | 0.92× (2 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 | n/a |
| gp_pois_regr-gp_pois_regr | 13 | 0.00163 | 0.88× (2 of 10) | 1.01× (5 of 10) | 1.15× (9 of 10) | n/a | 11.5 / 43 / 16 | 0 / 0 / 0 | 0 / 1 / 0 | 0 | n/a |
| gp_pois_regr-gp_regr | 3 | 0.0858 | 0.79× (0 of 10) | 1.08× (7 of 10) | 1.31× (10 of 10) | n/a | 0 / 11 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| hmm_example-hmm_example | 4 | 0.0253 | 1.01× (5 of 10) | 0.97× (3 of 10) | 0.94× (3 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 1 | 1 | n/a |
| hudson_lynx_hare-lotka_volterra | 8 | 0.00539 | 1.13× (6 of 10) | 1.08× (7 of 10) | 1.02× (5 of 10) | n/a | 0 / 0 / 1.5 | 0 / 0 / 0 | 4 / 2 / 1 | 8 | n/a |
| kidiq-kidscore_interaction | 5 | 0.00165 | 0.92× (5 of 10) | 0.62× (2 of 10) | 0.86× (3 of 10) | n/a | 0 / 10 / 127.5 | 0 / 1 / 5 | 0 / 3 / 6 | 3 | n/a |
| kidiq-kidscore_momhs | 3 | 0.0209 | 1.26× (10 of 10) | 0.90× (3 of 10) | 0.77× (2 of 10) | n/a | 0 / 0 / 2 | 0 / 0 / 0 | 0 / 0 / 0 | 3 | n/a |
| kidiq-kidscore_momhsiq | 4 | 0.00577 | 1.40× (10 of 10) | 1.32× (9 of 10) | 0.94× (4 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 | n/a |
| kidiq-kidscore_momiq | 3 | 0.00649 | 1.35× (8 of 10) | 1.28× (10 of 10) | 1.05× (7 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.5 | n/a |
| kidiq_with_mom_work-kidscore_interaction_c | 5 | 0.0526 | 1.10× (8 of 10) | 0.90× (5 of 10) | 0.89× (2 of 10) | n/a | 0 / 2 / 18.5 | 0 / 0 / 0 | 0 / 0 / 0 | 3 | n/a |
| kidiq_with_mom_work-kidscore_interaction_c2 | 5 | 0.0374 | 0.94× (4 of 10) | 0.58× (0 of 10) | 0.58× (0 of 10) | n/a | 0 / 0 / 23 | 0 / 0 / 0 | 0 / 0 / 1 | 5 | n/a |
| kidiq_with_mom_work-kidscore_interaction_z | 5 | 0.0328 | 1.43× (9 of 10) | 1.00× (5 of 10) | 0.68× (1 of 10) | n/a | 0 / 2 / 24.5 | 0 / 0 / 0 | 0 / 0 / 0 | 3 | n/a |
| kidiq_with_mom_work-kidscore_mom_work | 5 | 0.013 | 1.24× (10 of 10) | 1.06× (6 of 10) | 0.90× (4 of 10) | n/a | 0 / 1.5 / 21.5 | 0 / 0 / 0 | 0 / 0 / 1 | 2 | n/a |
| kilpisjarvi_mod-kilpisjarvi | 3 | 0.000205 | 0.11× (1 of 10) | 1.32× (6 of 10) | 7.22× (8 of 10) | n/a | 0 / 106.5 / 0 | 0 / 8 / 2 | 3 / 9 / 2 | 0.5 | n/a |
| low_dim_gauss_mix-low_dim_gauss_mix | 5 | 0.0602 | 1.00× (5 of 10) | 0.56× (2 of 10) | 0.55× (1 of 10) | n/a | 0 / 0 / 22.5 | 0 / 0 / 0 | 0 / 0 / 1 | 5 | n/a |
| mcycle_gp-accel_gp | 66 | 0.000398 | 1.01× (6 of 10) | 0.93× (5 of 10) | 0.91× (4 of 10) | n/a | 40.5 / 102 / 97.5 | 1 / 0 / 0 | 0 / 0 / 2 | 6 | n/a |
| mesquite-logmesquite | 8 | 0.0113 | 0.89× (3 of 10) | 0.95× (4 of 10) | 1.18× (7 of 10) | n/a | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| mesquite-logmesquite_logva | 5 | 0.0099 | 0.99× (4 of 10) | 0.95× (3 of 10) | 1.02× (6 of 10) | n/a | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| mesquite-logmesquite_logvas | 8 | 0.0058 | 0.90× (2 of 10) | 0.98× (3 of 10) | 1.11× (7 of 10) | n/a | 0 / 12 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| mesquite-logmesquite_logvash | 7 | 0.00567 | 0.83× (0 of 10) | 1.00× (5 of 10) | 1.17× (10 of 10) | n/a | 0 / 4.5 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| mesquite-logmesquite_logvolume | 3 | 0.0544 | 0.92× (3 of 10) | 1.01× (6 of 10) | 1.04× (7 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| mesquite-mesquite | 8 | 0.00385 | 0.59× (0 of 10) | 1.09× (5 of 10) | 1.76× (9 of 10) | n/a | 0 / 11 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 | n/a |
| nes1972-nes | 10 | 0.00756 | 0.96× (4 of 10) | 1.00× (5 of 10) | 1.03× (7 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes1976-nes | 10 | 0.00741 | 1.01× (5 of 10) | 1.03× (7 of 10) | 1.06× (6 of 10) | n/a | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes1980-nes | 10 | 0.00733 | 0.99× (5 of 10) | 0.98× (4 of 10) | 0.97× (3 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes1984-nes | 10 | 0.0073 | 1.10× (8 of 10) | 1.06× (6 of 10) | 0.97× (2 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes1988-nes | 10 | 0.00746 | 0.93× (3 of 10) | 1.00× (5 of 10) | 1.07× (8 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes1992-nes | 10 | 0.00779 | 0.98× (4 of 10) | 0.99× (5 of 10) | 1.01× (6 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes1996-nes | 10 | 0.00686 | 0.93× (4 of 10) | 0.99× (5 of 10) | 1.08× (6 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes2000-nes | 10 | 0.00665 | 0.94× (3 of 10) | 0.96× (3 of 10) | 0.94× (3 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| nes_logit_data-nes_logit_model | 2 | 0.0168 | 0.85× (1 of 10) | 0.88× (1 of 10) | 1.08× (5 of 10) | n/a | 0 / 0 / 0.5 | 0 / 0 / 1 | 0 / 1 / 1 | 1 | n/a |
| one_comp_mm_elim_abs-one_comp_mm_elim_abs | 4 | 0.00842 | 0.83× (4 of 10) | 0.70× (2 of 10) | 1.14× (5 of 10) | n/a | 12.5 / 138.5 / 171 | 5 / 10 / 5 | 2 / 3 / 2 | 1 | n/a |
| radon_all-radon_pooled | 3 | 0.0898 | 0.96× (4 of 10) | 0.98× (4 of 10) | 1.02× (6 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| radon_mn-radon_pooled | 3 | 0.0861 | 0.99× (4 of 10) | 1.00× (5 of 10) | 1.02× (6 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
| sblrc-blr | 6 | 0.00755 | 2.87× (10 of 10) | 2.67× (10 of 10) | 0.89× (2 of 10) | n/a | 0 / 0 / 0 | 1 / 0 / 1 | 1 / 0 / 0 | 5 | n/a |
| sblri-blr | 6 | 0.0143 | 4.63× (10 of 10) | 4.49× (10 of 10) | 0.93× (3 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 5 | n/a |
| wells_data-wells_dist | 2 | 0.00761 | 3.27× (10 of 10) | 1.85× (10 of 10) | 0.57× (2 of 10) | n/a | 0 / 0 / 8 | 0 / 0 / 0 | 1 / 0 / 2 | 2 | n/a |
| wells_data-wells_dist100_model | 2 | 0.0365 | 0.85× (3 of 10) | 1.01× (6 of 10) | 1.19× (8 of 10) | n/a | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | n/a |
<!-- END GENERATED: posteriordb-table -->

<!-- BEGIN GENERATED: posteriordb-findings -->
- **`arcsinh` vs stock NUTS** (worst-parameter tail-ESS per gradient, per-posterior median over seeds): `arcsinh` better on 24 of 58 posteriors, worse by more than a fifth on 9; median over posteriors 0.98× (range 0.11× to 95.00×).
- **`tukey_h` vs stock NUTS** (worst-parameter tail-ESS per gradient, per-posterior median over seeds): `tukey_h` better on 24 of 58 posteriors, worse by more than a fifth on 8; median over posteriors 0.99× (range 0.03× to 47.48×).
- **`tukey_h` vs `arcsinh`** (worst-parameter tail-ESS per gradient, per-posterior median over seeds): `tukey_h` better on 31 of 58 posteriors, worse by more than a fifth on 11; median over posteriors 1.02× (range 0.10× to 7.22×).
- **Posteriors where most runs disagree with the reference** (a z-score above 5 on at least half the seeds): `stock`: 3 of 58 (arma-arma11, eight_schools-eight_schools_centered, one_comp_mm_elim_abs-one_comp_mm_elim_abs); `arcsinh`: 3 of 58 (eight_schools-eight_schools_centered, kilpisjarvi_mod-kilpisjarvi, one_comp_mm_elim_abs-one_comp_mm_elim_abs); `tukey_h`: 4 of 58 (earnings-earn_height, eight_schools-eight_schools_centered, kidiq-kidscore_interaction, one_comp_mm_elim_abs-one_comp_mm_elim_abs).
- **Posteriors with divergences on at least half the seeds**: `stock`: 6 of 58 (arma-arma11, bball_drive_event_0-hmm_drive_0, eight_schools-eight_schools_centered, gp_pois_regr-gp_pois_regr, mcycle_gp-accel_gp, one_comp_mm_elim_abs-one_comp_mm_elim_abs); `arcsinh`: 24 of 58; `tukey_h`: 20 of 58.
- **Posteriors not converged on at least half the seeds** (max R-hat above 1.01): `stock`: 3 of 58 (arma-arma11, bball_drive_event_0-hmm_drive_0, eight_schools-eight_schools_centered); `arcsinh`: 3 of 58 (arma-arma11, eight_schools-eight_schools_centered, kilpisjarvi_mod-kilpisjarvi); `tukey_h`: 4 of 58 (arma-arma11, earnings-earn_height, eight_schools-eight_schools_centered, kidiq-kidscore_interaction).
- **Phase 1 calibrated off the posterior** (median over seeds of the largest distance between the calibrated location and the run's final median, above 3 posterior scales): `arcsinh`: 10 of 58 posteriors, worse than stock by more than a fifth on 2 of them; `tukey_h`: 10 of 58 posteriors, worse than stock by more than a fifth on 1 of them.
- **The `tukey_h` guard** (median over seeds) set h > 0 on 103 of 394 unconstrained coordinates (31 of 58 posteriors); everywhere else the h-map is the affine map.
- **`TailAlignmentWarning`** on the default `arcsinh` runs' unconstrained draws fired, on at least half the seeds, on 0 of 58 posteriors.
<!-- END GENERATED: posteriordb-findings -->

**What the table does and does not show.**
- It compares the transforms with stock NUTS on Stan's unconstrained parameterisation of each model, as posteriordb defines it. Most of these posteriors are regressions that stock NUTS already samples well, so Table C tests whether the transforms are safe as a default, not whether they help on heavy tails (Tables A and B).
- The split by the guard's decision is observational: the guard's decision and the posterior are not varied independently. Running `tukey_h` with h held at 0, which is the `affine` arm, on the same posteriors would test whether the h > 0 decisions cause the losses.
- Each run has the protocol of Table A; stock NUTS on posteriordb is usually run with a longer warm-up.

### 5.4 Single-seed comparisons (Tables 1–5)

**Two runners produce the tables.**

`bench/sota_runner.py --full` writes `reports/sota_comparison_results.json`, with four tasks:
1. a sparse regression with Cauchy priors, d = 10;
2. Eight-Schools and Neal's funnel;
3. mixed heavy and Gaussian tails, d = 20;
4. a quartic target started far in the tail.

`bench/applied_runner.py` writes `reports/applied_benchmark_results.json`. It covers five applied models fitted to simulated data.

**How to read the tables:**
- A run is converged when its largest split R-hat is at most 1.01.
- Efficiency ratios involving unconverged runs are not evidence of an advantage.
- Langevin samplers build no trajectories, so they have no divergence count ("–").

### Table 1: Sparse regression with Cauchy priors (Task 1, d = 10)

**Target.** A Bayesian regression with $`\beta_j \sim \mathrm{Cauchy}(0, 1)`$ and true signal $`(5, -4, 3, 0, \dots, 0)`$.
- Only the three active coefficients have non-zero design columns.
- The other seven posteriors therefore equal their independent Cauchy prior.

**Why it matters.** This is a product of Cauchy and near-Gaussian coordinates: the most favourable case for a coordinate-wise map.

<!-- BEGIN GENERATED: sota-task1 -->
Seed 42 only, 4 chains × 1,000 draws after 500 warm-up steps per chain; gradient counts include warm-up. Source: `reports/sota_comparison_results.json` (run 2026-10-23). A run is converged when its largest rank-normalised split R-hat is at most 1.01.

| Sampler | Gradients | Bulk-ESS | Tail-ESS | Max R-hat | Converged | Tail-ESS/grad | vs BlackJAX NUTS | Divergences | q99 error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyMC NUTS | 450,980 | 1,965 | 1,161 | 1.013 | no | 0.00257 | 5.5× | 0 | 41.6% |
| BlackJAX NUTS | 2,109,517 | 1,684 | 985 | 1.037 | no | 0.000467 | 1.0× | 0 | 45.3% |
| Barker | 6,004 | 335 | 378 | 2.323 | no | 0.0629 | 134.7× | – | 62.3% |
| Stereographic MALA (own implementation) | 6,004 | 331 | 269 | 2.465 | no | 0.0448 | 96.0× | – | 65.7% |
| AutoMALA (own re-implementation) | 11,126 | 396 | 524 | 2.438 | no | 0.0471 | 100.8× | – | 68.6% |
| `UnifiedRobustSampler` | 6,004 | 347 | 365 | 2.312 | no | 0.0607 | 130.1× | – | 48.5% |
| `UnifiedRobustHMC` | 29,892 | 1,021 | 551 | 1.040 | no | 0.0184 | 39.5× | 0 | 54.2% |
| `UnifiedRobustNUTS` | 174,625 | 3,442 | 1,917 | 1.002 | yes | 0.011 | 23.5× | 0 | 8.1% |
<!-- END GENERATED: sota-task1 -->

### Table 2a: Centred Eight-Schools (Task 2)

**Target.** Rubin's (1981) model with $`\tau \sim \text{Half-Cauchy}(0, 5)`$, sampled in $(\mu, \log\tau, \theta)$. The reference neck statistics come from exact one-dimensional marginalisation (`EightSchoolsTarget.exact_moments_or_quantiles`).

<!-- BEGIN GENERATED: sota-eight-schools -->
Seed 42 only, 4 chains × 1,000 draws after 500 warm-up steps per chain; gradient counts include warm-up. Source: `reports/sota_comparison_results.json` (run 2026-10-23). A run is converged when its largest rank-normalised split R-hat is at most 1.01. Exact reference: P(log τ < −1) = 7.49%, q05(log τ) = −1.40.

| Sampler | Gradients | Bulk-ESS | Tail-ESS | Max R-hat | Converged | Divergences | P(log τ < −1) | q05(log τ) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyMC NUTS | 117,717 | 540 | 677 | 1.032 | no | 148 | 0.0% | −0.11 |
| BlackJAX NUTS | 149,818 | 659 | 934 | 1.012 | no | 68 | 0.0% | −0.07 |
| Tamed MALA | 6,004 | 7 | 10 | 2.053 | no | – | 30.2% | −2.58 |
| AutoMALA (own re-implementation) | 14,036 | 7 | 10 | 1.900 | no | – | 7.2% | −1.08 |
| `GentleMALA` | 6,004 | 6 | 8 | 2.445 | no | – | 6.7% | −1.07 |
| `UnifiedRobustSampler` | 6,004 | 5 | 7 | 3.040 | no | – | 16.6% | −1.49 |
| `UnifiedRobustNUTS` | 291,573 | 516 | 733 | 1.017 | no | 7 | 1.5% | −0.53 |
<!-- END GENERATED: sota-eight-schools -->

### Table 2b: Neal's funnel (Task 2, d = 10)

**Target.** $`v \sim \mathcal{N}(0, 3^2)`$ and $`x_i \mid v \sim \mathcal{N}(0, e^{v})`$ for the other nine coordinates.

<!-- BEGIN GENERATED: sota-neal -->
Seed 42 only, 4 chains × 1,000 draws after 500 warm-up steps per chain; gradient counts include warm-up. Source: `reports/sota_comparison_results.json` (run 2026-10-23). A run is converged when its largest rank-normalised split R-hat is at most 1.01. Exact reference (v ~ N(0, 3²)): P(v < −2) = 25.25%, q05(v) = −4.93.

| Sampler | Gradients | Bulk-ESS | Tail-ESS | Max R-hat | Converged | Divergences | P(v < −2) | q05(v) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BlackJAX NUTS | 160,066 | 4,131 | 301 | 1.151 | no | 3 | 5.0% | −2.01 |
| `UnifiedRobustSampler` | 6,004 | 12 | 10 | 2.092 | no | – | 29.1% | −4.47 |
| `UnifiedRobustNUTS` | 160,757 | 274 | 209 | 1.079 | no | 34 | 22.2% | −4.40 |
<!-- END GENERATED: sota-neal -->

### Table 3: Mixed heavy and Gaussian tails (Task 3, d = 20)

**Target.** Ten Student-*t* coordinates, with $\nu$ evenly spaced from 1.2 to 2.5, and ten standard Gaussian coordinates, all independent.

<!-- BEGIN GENERATED: sota-task3 -->
Seed 42 only, 4 chains × 1,000 draws after 500 warm-up steps per chain; gradient counts include warm-up. Source: `reports/sota_comparison_results.json` (run 2026-10-23). A run is converged when its largest rank-normalised split R-hat is at most 1.01.

| Sampler | Gradients | Bulk-ESS | Tail-ESS | Max R-hat | Converged | Tail-ESS/grad | vs BlackJAX NUTS | Divergences | q99 error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BlackJAX NUTS | 133,000 | 3,877 | 2,230 | 1.015 | no | 0.0168 | 1.0× | 0 | 11.3% |
| Barker | 6,004 | 276 | 381 | 1.824 | no | 0.0635 | 3.8× | – | 20.3% |
| AutoMALA (own re-implementation) | 11,008 | 127 | 184 | 1.237 | no | 0.0167 | 1.0× | – | 18.6% |
| `AnisotropicTailMALA` | 6,004 | 318 | 465 | 1.285 | no | 0.0775 | 4.6× | – | 15.8% |
| `UnifiedRobustSampler` | 6,004 | 94 | 170 | 1.130 | no | 0.0283 | 1.7× | – | 13.0% |
| `UnifiedRobustNUTS` | 71,545 | 4,609 | 2,810 | 1.003 | yes | 0.0393 | 2.3× | 4 | 5.8% |
<!-- END GENERATED: sota-task3 -->

### Table 4: Quartic target started in the tail (Task 4, d = 10)

<!-- BEGIN GENERATED: sota-task4 -->
Seed 42 only, 4 chains × 1,000 draws after 500 warm-up steps per chain; gradient counts include warm-up. Source: `reports/sota_comparison_results.json` (run 2026-10-23). A run is converged when its largest rank-normalised split R-hat is at most 1.01. Chains start at ‖x₀‖ = 10√d; "Gradients to reach the bulk" is the median over chains of the gradient evaluations until ‖x‖ ≤ 3.5, or the chain's whole budget when it never gets there.

| Sampler | Gradients | Bulk-ESS | Tail-ESS | Max R-hat | Converged | Tail-ESS/grad | Divergences | Gradients to reach the bulk |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MALA | 6,004 | 4 | 4 | n/a | no | 0.000672 | – | 1,501 |
| BlackJAX HMC | 60,000 | 4,994 | 2,208 | 1.003 | yes | 0.0368 | 0 | 50 |
| BlackJAX NUTS | 45,978 | 7,188 | 3,171 | 1.006 | yes | 0.069 | 0 | 40 |
| Tamed MALA | 6,004 | 203 | 346 | 1.036 | no | 0.0577 | – | 17 |
| AutoMALA (own re-implementation) | 19,625 | 4 | 5 | 4.558 | no | 0.000233 | – | 5,002 |
| `GentleMALA` | 6,004 | 612 | 921 | 1.010 | no | 0.153 | – | 5 |
| `UnifiedRobustSampler` | 6,004 | 404 | 765 | 1.019 | no | 0.127 | – | 10 |
| `UnifiedRobustNUTS` | 62,131 | 4,697 | 3,503 | 1.002 | yes | 0.0564 | 0 | 15 |
<!-- END GENERATED: sota-task4 -->

### Table 5: Applied models on simulated data

**The models.** They have the structure of standard applied posteriors but are fitted to simulated data, so they are not the posteriordb posteriors:
- a radon-style hierarchical regression;
- a sparse horseshoe logistic regression;
- a robust Student-*t* regression;
- a hierarchical 2PL item-response model;
- AR(1) stochastic volatility.

**The arcsinh row.** "BlackJAX NUTS + fixed arcsinh" uses a fixed transform centred at the origin. It is not the calibrated JAX pipeline of §2.4.

<!-- BEGIN GENERATED: applied -->
Seed 42 only, 4 chains × 1,000 draws after 1,000 warm-up steps, NUTS tree depth at most 10. Min ESS is over coordinates; Tail-ESS/grad is the coordinate mean divided by all gradients, warm-up included. Wall time includes JIT compilation for the BlackJAX rows, while `UnifiedRobustNUTS` runs in NumPy. Source: `reports/applied_benchmark_results.json`.

| Model | Sampler | Max R-hat | Converged | Min Bulk-ESS | Min Tail-ESS | Tail-ESS/grad (×10⁻³) | Divergences | Gradients | Wall time (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Radon-style hierarchical (d=18) | MALA | 1.385 | no | 9 | 10 | 21.40 | – | 8,004 | 0.6 |
| Radon-style hierarchical (d=18) | AutoMALA (own re-implementation) | 1.096 | no | 34 | 63 | 14.34 | – | 16,604 | 1.1 |
| Radon-style hierarchical (d=18) | UnifiedRobustSampler | 1.304 | no | 11 | 11 | 11.52 | – | 8,004 | 1.1 |
| Radon-style hierarchical (d=18) | BlackJAX NUTS | 1.019 | no | 295 | 250 | 12.14 | 10 | 151,503 | 12.5 |
| Radon-style hierarchical (d=18) | BlackJAX NUTS + fixed arcsinh (R = 2.5) | 1.022 | no | 227 | 154 | 16.61 | 30 | 106,174 | 12.3 |
| Radon-style hierarchical (d=18) | UnifiedRobustNUTS | 1.011 | no | 475 | 417 | 21.05 | 3 | 114,269 | 15.0 |
| Horseshoe logistic (d=21) | MALA | 2.108 | no | 5 | 6 | 3.77 | – | 8,004 | 0.8 |
| Horseshoe logistic (d=21) | AutoMALA (own re-implementation) | 1.584 | no | 7 | 6 | 1.40 | – | 21,012 | 2.0 |
| Horseshoe logistic (d=21) | UnifiedRobustSampler | 1.412 | no | 8 | 11 | 3.39 | – | 8,004 | 1.5 |
| Horseshoe logistic (d=21) | BlackJAX NUTS | 1.029 | no | 179 | 119 | 1.64 | 559 | 361,983 | 13.3 |
| Horseshoe logistic (d=21) | BlackJAX NUTS + fixed arcsinh (R = 2.5) | 1.088 | no | 38 | 17 | 1.47 | 851 | 375,654 | 14.2 |
| Horseshoe logistic (d=21) | UnifiedRobustNUTS | 1.011 | no | 365 | 306 | 2.59 | 122 | 455,219 | 73.5 |
| Student-t regression (d=12) | MALA | 1.045 | no | 151 | 211 | 43.06 | – | 8,004 | 0.6 |
| Student-t regression (d=12) | AutoMALA (own re-implementation) | 1.038 | no | 161 | 182 | 23.07 | – | 15,281 | 1.2 |
| Student-t regression (d=12) | UnifiedRobustSampler | 1.024 | no | 199 | 306 | 49.19 | – | 8,004 | 1.3 |
| Student-t regression (d=12) | BlackJAX NUTS | 1.002 | yes | 1,744 | 2,497 | 41.14 | 0 | 67,334 | 11.2 |
| Student-t regression (d=12) | BlackJAX NUTS + fixed arcsinh (R = 2.5) | 1.004 | yes | 1,637 | 2,300 | 37.35 | 0 | 70,234 | 11.2 |
| Student-t regression (d=12) | UnifiedRobustNUTS | 1.001 | yes | 2,267 | 2,793 | 41.61 | 0 | 71,254 | 11.0 |
| IRT-2PL (d=22) | MALA | 1.259 | no | 13 | 14 | 7.20 | – | 8,004 | 0.8 |
| IRT-2PL (d=22) | AutoMALA (own re-implementation) | 1.113 | no | 30 | 41 | 8.68 | – | 17,127 | 1.7 |
| IRT-2PL (d=22) | UnifiedRobustSampler | 1.412 | no | 9 | 12 | 10.23 | – | 8,004 | 1.5 |
| IRT-2PL (d=22) | BlackJAX NUTS | 1.013 | no | 337 | 460 | 9.54 | 38 | 185,666 | 12.7 |
| IRT-2PL (d=22) | BlackJAX NUTS + fixed arcsinh (R = 2.5) | 1.124 | no | 22 | 8 | 7.21 | 106 | 211,194 | 13.9 |
| IRT-2PL (d=22) | UnifiedRobustNUTS | 1.008 | yes | 397 | 556 | 11.84 | 38 | 175,595 | 28.6 |
| Stochastic volatility (d=22) | MALA | 1.893 | no | 6 | 7 | 3.49 | – | 8,004 | 0.6 |
| Stochastic volatility (d=22) | AutoMALA (own re-implementation) | 1.580 | no | 7 | 9 | 2.03 | – | 17,953 | 1.2 |
| Stochastic volatility (d=22) | UnifiedRobustSampler | 1.551 | no | 7 | 7 | 2.07 | – | 8,004 | 1.6 |
| Stochastic volatility (d=22) | BlackJAX NUTS | 1.018 | no | 137 | 225 | 3.65 | 0 | 263,446 | 11.5 |
| Stochastic volatility (d=22) | BlackJAX NUTS + fixed arcsinh (R = 2.5) | 1.019 | no | 114 | 194 | 3.40 | 1 | 207,953 | 11.6 |
| Stochastic volatility (d=22) | UnifiedRobustNUTS | 1.022 | no | 134 | 200 | 3.93 | 0 | 232,745 | 31.0 |

- 4 of 30 runs converge (largest split R-hat at most 1.01).
- First-order samplers converge in 0 of 15 runs.
- Student-t regression is a model on which every NUTS variant converges: `UnifiedRobustNUTS` reaches 1.0× the tail-ESS per gradient of stock BlackJAX NUTS.
<!-- END GENERATED: applied -->

## 6. Limitations and open work

**About the evidence:**
- **One seed per cell in Tables 1–5, no error bars.** JAX gradient counts for stock BlackJAX also vary across platforms, so single-seed ratios can shift between machines. Table A is the multi-seed evidence.
- **Heavy tails along rotated directions.**
  - On the rotated multivariate Cauchy, the arcsinh arm converges on few seeds (Table A), so its ratio there is not a supported gain.
  - On the rotated product of Cauchys it misses tails without any R-hat warning.
  - Do not rely on the transform when heavy tails may lie along directions that are not coordinate axes.
- **Real posteriors.** On posteriordb (§5.3) neither transform is a uniform improvement over stock NUTS, and both add divergences on posteriors that stock NUTS samples without any. Phase 1 calibrating on draws that have not yet reached the posterior happens there too, but does not account for most of the losses.
- **Unconverged denominators.** Several ratios compare against baselines that are not converged; for example, BlackJAX NUTS on Task 1. An unconverged denominator makes a ratio unreliable in either direction.
- **The best case for the method.** Task 1 is built so that most coordinates are independent and heavy-tailed. Task 3 is also product-form.
- **A noisy accuracy metric.** For a Cauchy marginal at an effective sample size of about a thousand, the Monte Carlo error of an estimated 0.99 quantile is comparable to the quantile errors reported in Tables 1 and 3. Single-seed differences in that column are therefore weak evidence. A tail-probability error with its MCSE would be a better metric.

**Not yet evaluated:**
- elliptical heavy tails with several scales, beyond the rotated *t* targets of Table A;
- diffeomorphic MCMC (Brešar & Mijatović, 2026), for which no public code was found;
- the `affine` arm on posteriordb, which would test whether the `tukey_h` guard's h > 0 decisions cause its losses there (§5.3).

**Alternative transforms under exploration** (exploration branches, not part of the library): whitened radial and block-structured maps for elliptical heavy tails.

**The new options:**
- The `tukey_h` guard and the tail-alignment warning were calibrated on synthetic targets, and both can be wrong in the ways described in §2.6 and §2.7.
- On posteriordb (§5.3) the guard sets h > 0 on coordinates of many posteriors, and those are the posteriors where `tukey_h` loses. Its test treats autocorrelated, early warm-up draws as independent; a guard that accounts for that is open work.

**Engineering:**
- **Speed.** `UnifiedRobustNUTS` is a NumPy implementation and is slower per gradient than BlackJAX; the JAX pipeline is the faster path.
- **Figures.** `bench/plot.py` writes figures to `reports/figures/`. They are not embedded here until their annotations are revised to match the tables.

## 7. Practitioner guide

**When to use it.** Use `sample_blackjax_unified_nuts` when the posterior has heavy-tailed coordinates that roughly align with the parameters. Typical cases:
- Cauchy or half-Cauchy priors on weakly identified coefficients;
- Student-*t* components.

Check R-hat and ESS as usual.

**Reparameterise by hand first if you can.** If the heavy-tailed part of the model is known to be a Student-t or Cauchy component and is weakly informed by the data, writing it as a non-centred scale mixture can beat the transform. Table A compares the two.

**When not to expect help:**
- **Funnels:** non-centre the model (§3.3).
- **Heavy tails that are strongly rotated relative to the coordinates:** gains are smaller and less reliable, and a tail can be missed without warning (Table A). nutpie's normalizing-flow adaptation is worth trying there (Table B).
- **Targets that stock NUTS already samples well:** the transform is not expected to help there, and on real posteriors it can add divergences (Table C). Use it when you expect heavy tails, not as a default for every model.

**Try the Tukey h-map.** On axis-aligned heavy tails, try `transform_kind="tukey_h"` (Table A compares it with arcsinh). Check `result.transform.h` to see which coordinates were warped. If it is positive on coordinates you expect to be light-tailed, prefer arcsinh or stock NUTS: on real posteriors, the guard's h > 0 decisions are where `tukey_h` lost (Table C).

**Heed the warning.** If `TailAlignmentWarning` fires, treat tail estimates with caution. Reparameterise the model, or try a method that follows tail directions, such as nutpie's flow (Table B).

**If you use `UnifiedRobustNUTS` directly:** pass `adapt_step_size=True` and set `burn_in` to at least `warmup_steps`.

## 8. References

1. Atchadé, Y. F. (2006). An adaptive version for the Metropolis adjusted Langevin algorithm with a truncated drift. *Methodology and Computing in Applied Probability*, 8(2), 235–254.
2. Bell, C., Łatuszyński, K., & Roberts, G. O. (2024). Adaptive stereographic MCMC. arXiv:2408.11780.
3. Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
4. Betancourt, M. (2018). Fitting the Cauchy. Stan case study.
5. Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A. (2024). autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm. *AISTATS*, PMLR 238. arXiv:2310.16782.
6. Bou-Rabee, N., Carpenter, B., & Marsden, M. (2024). GIST: Gibbs self-tuning for locally adaptive Hamiltonian Monte Carlo. arXiv:2404.15253.
7. Bou-Rabee, N., Carpenter, B., Kleppe, T. S., & Liu, S. (2025). The within-orbit adaptive leapfrog no-U-turn sampler (WALNUTS). arXiv:2506.18746.
8. Brešar, M., & Mijatović, A. (2026). Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions. arXiv:2608.04284.
9. Brosse, N., Durmus, A., Moulines, É., & Sabanis, S. (2019). The tamed unadjusted Langevin algorithm. *Stochastic Processes and their Applications*, 129(10), 3638–3663. doi:10.1016/j.spa.2018.10.002.
10. Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
11. Goerg, G. M. (2015). The Lambert way to Gaussianize heavy-tailed data with the inverse of Tukey's h transformation as a special case. *The Scientific World Journal*, 2015, 909231. arXiv:1010.2265.
12. Gorinova, M. I., Moore, D., & Hoffman, M. D. (2020). Automatic reparameterisation of probabilistic programs. *ICML*, PMLR 119, 3648–3657. arXiv:1906.03028.
13. He, Y., Balasubramanian, K., & Erdogdu, M. A. (2024). An analysis of transformed unadjusted Langevin algorithm for heavy-tailed sampling. *IEEE Transactions on Information Theory*, 70(1), 571–593. arXiv:2201.08349.
14. Hickling, T., & Prangle, D. (2025). Flexible tails for normalizing flows. *ICML*, PMLR 267, 23155–23178. arXiv:2406.16971.
15. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15, 1593–1623.
16. Hoffman, M., Sountsov, P., Dillon, J. V., Langmore, I., Tran, D., & Vasudevan, S. (2019). NeuTra-lizing bad geometry in Hamiltonian Monte Carlo using neural transport. arXiv:1903.03704.
17. Jaini, P., Kobyzev, I., Yu, Y., & Brubaker, M. (2020). Tails of Lipschitz triangular flows. *ICML*. arXiv:1907.04481.
18. Johnson, L. T., & Geyer, C. J. (2012). Variable transformation to obtain geometric ergodicity in the random-walk Metropolis algorithm. *The Annals of Statistics*, 40(6), 3050–3076. doi:10.1214/12-AOS1048.
19. Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *The Review of Economic Studies*, 65(3), 361–393.
20. Laszkiewicz, M., Lederer, J., & Fischer, A. (2022). Marginal tail-adaptive normalizing flows. *ICML*, PMLR 162, 12020–12048. arXiv:2206.10311.
21. Liang, F., Mahoney, M. W., & Hodgkinson, L. (2022). Fat-tailed variational inference with anisotropic tail adaptive flows. *ICML*, PMLR 162, 13257–13270. arXiv:2205.07918.
22. Livingstone, S., & Zanella, G. (2022). The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. *Journal of the Royal Statistical Society: Series B*, 84(2), 496–523.
23. Livingstone, S., Betancourt, M., Byrne, S., & Girolami, M. (2019). On the geometric ergodicity of Hamiltonian Monte Carlo. *Bernoulli*, 25(4A), 3109–3138. doi:10.3150/18-BEJ1083.
24. Lytras, I., & Sabanis, S. (2025). Taming under isoperimetry. *Stochastic Processes and their Applications*, 188, 104684. arXiv:2311.09003.
25. Magnusson, M., Torgander, J., Bürkner, P.-C., Zhang, L., Carpenter, B., & Vehtari, A. (2025). posteriordb: Testing, benchmarking and developing Bayesian inference algorithms. *AISTATS*, PMLR 258, 1198–1206. arXiv:2407.04967.
26. Modi, C., Barnett, A., & Carpenter, B. (2024). Delayed rejection Hamiltonian Monte Carlo for sampling multiscale distributions. *Bayesian Analysis*, 19(3), 815–842. doi:10.1214/23-BA1360.
27. Neal, R. M. (2003). Slice sampling. *The Annals of Statistics*, 31(3), 705–767.
28. Papaspiliopoulos, O., Roberts, G. O., & Sköld, M. (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 22(1), 59–73.
29. Piironen, J., & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018–5051.
30. Power, S., & Vasdekis, G. (2025). Some aspects of robustness in modern Markov chain Monte Carlo. arXiv:2511.21563.
31. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341–363.
32. Roualdes, E. A., Ward, B., Carpenter, B., Seyboldt, A., & Axen, S. D. (2023). BridgeStan: Efficient in-memory access to the methods of a Stan model. *Journal of Open Source Software*, 8(87), 5236. doi:10.21105/joss.05236.
33. Rubin, D. B. (1981). Estimation in parallel randomized experiments. *Journal of Educational Statistics*, 6(4), 377–401.
34. Self, S. G., & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *Journal of the American Statistical Association*, 82(398), 605–610.
35. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved $\widehat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
36. Yang, J., Łatuszyński, K., & Roberts, G. O. (2024). Stereographic Markov chain Monte Carlo. *The Annals of Statistics*, 52(6), 2692–2713. doi:10.1214/24-AOS2426.
37. nutpie: a NUTS implementation with gradient-based and normalizing-flow adaptation. https://github.com/pymc-devs/nutpie.
38. The Mathlib Community. The Lean 4 mathematical library (mathlib4). https://github.com/leanprover-community/mathlib4.
