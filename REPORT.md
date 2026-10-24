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
- **Single-seed comparisons (§5.2).** On two synthetic targets whose heavy tails are aligned with the coordinate axes, the transform raises tail-ESS per gradient over stock BlackJAX and PyMC NUTS and lowers the error of the estimated 0.99 quantile (Tables 1 and 3).
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

**New, as far as a literature check found:** a closed-form arcsinh map per coordinate, calibrated by median and IQR during warm-up, followed by standard NUTS adaptation on the exact pullback. This is a concrete default answer to Power & Vasdekis's §4.3(iii)–(iv).

**Not yet done:**
- comparison with nutpie's flow adaptation;
- comparison with diffeomorphic MCMC.

§6 lists these. The comparison with reparameterisation by hand is in §5.1.

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

<!-- BEGIN GENERATED: lean-inventory -->
37 theorems in `formal/McmcRobustFormal.lean`, listed under the file's own section headings:

| Section (as titled in the Lean file) | Theorems |
| :--- | :--- |
| Exact Invertibility of Shift-Equivariant Arcsinh/Sinh Diffeomorphism | `arcsinh_inv_right`, `arcsinh_inv_left`, `scaled_invertibility_roundtrip_forward`, `scaled_invertibility_roundtrip_inverse`, `shifted_invertibility_roundtrip_forward`, `shifted_invertibility_roundtrip_inverse`, `shiftedForwardMap_translation_invariant`, `inverseMap_jacobian_factor_pos` |
| Derivative Identity and Metric Scale Strict Positivity | `arsinh_deriv_eq`, `metric_scale_pos`, `metric_scale_le_one`, `hasDerivAt_forwardMap`, `forwardMap_metric_pos` |
| Pure Square-Root Attenuation & Exact Discrete Basin Boundary | `rho_pos`, `rho_le_one`, `rho_strictAntiOn`, `pureSqrtStepSq_lt_div_gamma`, `pureSqrt_quartic_discrete_basin_bound` |
| Soft-Capped Gentle Attenuation Global Drift & Non-Overshoot | `rhoSoft_pos`, `rhoSoft_le_one`, `softCap_dimensionless_drift_sq_lt`, `softCap_dimensionless_drift_lt`, `softCappedDriftStep_lt_bound`, `softCapped_global_contraction_no_overshoot` |
| Metropolis-Hastings Kernel Reversibility (Detailed Balance) | `mhAccept_bounds`, `mh_detailed_balance` |
| Applied Heavy-Tailed & Shrinkage Posterior Score Bounds | `studentTNegLogKernel_nonneg`, `hasDerivAt_studentTNegLogKernel`, `studentTScore_mul_le`, `cauchyShrinkagePenalty_nonneg`, `studentTScore_zero` |
| Pure-JAX 3-Phase Scale & Location Calibration Bounds | `calibratedRadius_ge_min`, `calibratedRadius_pos`, `calibratedRadius_le_max`, `calibratedRadius_mono`, `calibrated_shifted_roundtrip_forward`, `calibrated_shifted_roundtrip_inverse` |
<!-- END GENERATED: lean-inventory -->

## 5. Results

### 5.1 Multi-seed ablation (Table A)

`bench/ablation_runner.py --full` writes `reports/ablation_results.json`.

**Arms.** It runs every arm through the same BlackJAX pipeline:
- `stock`;
- `stock_ta95`: stock with a higher target acceptance rate;
- `arcsinh`: the transform;
- `affine`: the same Phase 1 calibration of μ and R with a linear map, which separates rescaling from tail warping;
- `scale_mixture`, on the Student-t-family targets: the practitioner's rewrite of a Student-t or Cauchy component as a non-centred Gaussian scale mixture, x = μ + √λ·L·η with an inverse-gamma λ.

**Targets.**
- Tasks 1 and 3 of §5.2;
- rotated and isotropic multivariate *t* targets, whose heavy tails are not aligned with the axes;
- two Gaussians, which measure the cost on targets stock NUTS already handles.

**Metrics.**
- Each arm's tail-ESS per gradient is divided by stock's on the same seed.
- Tail accuracy is the error in the probability below the exact 0.01 and 0.99 marginal quantiles.

<!-- BEGIN GENERATED: ablation-table -->
10 seeds per cell; each run has 4 chains × 1,000 draws after 500 warm-up steps. Every arm runs BlackJAX NUTS with window adaptation through `sample_blackjax_unified_nuts`; arms differ only in the coordinate transform (or, for `scale_mixture`, in the model's parameterisation), and share each seed's initial positions and random key, so ratios are paired by seed. Source: `reports/ablation_results.json` (run 2026-10-24, commit cdde709).

Tail-probability error is the mean over coordinates and p ∈ {0.01, 0.99} of |P̂(x_i < q_p) − p| at the exact marginal quantile q_p; it is less noisy than a relative error on the 0.99 quantile.

Targets: **Task 1**: Sparse regression, Cauchy(0, 1) priors, 3 active of 10 coefficients (SOTA Task 1); **Task 3**: Product form: t(nu) on 10 coordinates (nu = 1.2 to 2.5), N(0, 1) on 10 (SOTA Task 3); **rotated t (ν = 1)**: Multivariate Cauchy (t, nu = 1), d = 10, rotated scale with condition number 25; **rotated t (ν = 2)**: Multivariate t, nu = 2, d = 10, rotated scale with condition number 25; **isotropic t (ν = 2)**: Isotropic multivariate t, nu = 2, d = 10; **rotated Gaussian**: Rotated Gaussian, d = 20, covariance condition number 100; **anisotropic Gaussian**: Axis-aligned Gaussian, d = 20, standard deviations 0.1 to 10.

Arms: `stock`: BlackJAX NUTS, window adaptation, target acceptance 0.8; `stock_ta95`: The same with target acceptance 0.95; `arcsinh`: NUTS on the calibrated coordinate-wise arcsinh pullback; `affine`: Same pipeline and Phase 1 calibration, z = (x - mu) / R, no arcsinh; `scale_mixture`: Stock NUTS on the non-centred Gaussian scale-mixture reparameterisation.

| Target | Arm | Tail-ESS/grad vs stock, median [IQR] | Seeds better than stock | Seeds converged | Median gradients | Median divergences | Median tail-probability error |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task 1 | `stock` | 1 (reference) | – | 0 of 10 | 635,440 | 0 | 0.0034 |
| Task 1 | `stock_ta95` | 0.55× [0.45, 1.06] | 3 of 10 | 1 of 10 | 1,124,198 | 0 | 0.0044 |
| Task 1 | `arcsinh` | 7.14× [6.61, 13.13] | 10 of 10 | 9 of 10 | 148,826 | 0 | 0.0018 |
| Task 1 | `affine` | 0.89× [0.61, 1.24] | 4 of 10 | 0 of 10 | 683,634 | 0 | 0.0049 |
| Task 1 | `scale_mixture` | 1.09× [0.94, 1.89] | 7 of 10 | 8 of 10 | 1,061,258 | 205.5 | 0.0015 |
| Task 3 | `stock` | 1 (reference) | – | 1 of 10 | 197,496 | 0 | 0.0030 |
| Task 3 | `stock_ta95` | 0.55× [0.52, 0.96] | 2 of 10 | 3 of 10 | 297,914 | 0 | 0.0026 |
| Task 3 | `arcsinh` | 3.80× [3.12, 4.07] | 10 of 10 | 10 of 10 | 74,596 | 0 | 0.0015 |
| Task 3 | `affine` | 0.92× [0.63, 1.17] | 4 of 10 | 3 of 10 | 242,389 | 0 | 0.0030 |
| Task 3 | `scale_mixture` | 3.45× [2.81, 3.94] | 10 of 10 | 10 of 10 | 82,176 | 0 | 0.0014 |
| rotated t (ν = 1) | `stock` | 1 (reference) | – | 0 of 10 | 195,961 | 0 | 0.0043 |
| rotated t (ν = 1) | `stock_ta95` | 0.75× [0.55, 1.47] | 4 of 10 | 0 of 10 | 314,761 | 0 | 0.0035 |
| rotated t (ν = 1) | `arcsinh` | 5.24× [2.16, 8.88] | 10 of 10 | 2 of 10 | 121,078 | 0 | 0.0046 |
| rotated t (ν = 1) | `affine` | 1.92× [0.97, 2.88] | 7 of 10 | 0 of 10 | 178,581 | 0 | 0.0044 |
| rotated t (ν = 1) | `scale_mixture` | 44.71× [30.11, 75.54] | 10 of 10 | 10 of 10 | 51,242 | 0 | 0.0015 |
| rotated t (ν = 2) | `stock` | 1 (reference) | – | 2 of 10 | 117,246 | 0 | 0.0030 |
| rotated t (ν = 2) | `stock_ta95` | 0.63× [0.53, 0.76] | 0 of 10 | 4 of 10 | 178,166 | 0 | 0.0024 |
| rotated t (ν = 2) | `arcsinh` | 1.73× [1.44, 1.86] | 10 of 10 | 10 of 10 | 99,168 | 0 | 0.0019 |
| rotated t (ν = 2) | `affine` | 1.03× [0.95, 1.22] | 6 of 10 | 3 of 10 | 114,478 | 0 | 0.0031 |
| rotated t (ν = 2) | `scale_mixture` | 8.99× [7.62, 10.17] | 10 of 10 | 10 of 10 | 43,884 | 0 | 0.0013 |
| isotropic t (ν = 2) | `stock` | 1 (reference) | – | 4 of 10 | 64,814 | 0 | 0.0028 |
| isotropic t (ν = 2) | `stock_ta95` | 0.77× [0.68, 0.79] | 1 of 10 | 4 of 10 | 104,074 | 0 | 0.0017 |
| isotropic t (ν = 2) | `arcsinh` | 1.59× [1.46, 1.78] | 10 of 10 | 9 of 10 | 58,962 | 0 | 0.0019 |
| isotropic t (ν = 2) | `affine` | 1.13× [1.03, 1.19] | 8 of 10 | 4 of 10 | 65,280 | 0 | 0.0018 |
| isotropic t (ν = 2) | `scale_mixture` | 4.24× [3.87, 4.50] | 10 of 10 | 10 of 10 | 43,618 | 0 | 0.0012 |
| rotated Gaussian | `stock` | 1 (reference) | – | 10 of 10 | 155,246 | 0 | 0.0013 |
| rotated Gaussian | `stock_ta95` | 0.71× [0.69, 0.73] | 0 of 10 | 10 of 10 | 209,912 | 0 | 0.0013 |
| rotated Gaussian | `arcsinh` | 0.85× [0.84, 0.90] | 0 of 10 | 10 of 10 | 177,917 | 0 | 0.0013 |
| rotated Gaussian | `affine` | 1.00× [0.95, 1.02] | 4 of 10 | 10 of 10 | 156,798 | 0 | 0.0015 |
| anisotropic Gaussian | `stock` | 1 (reference) | – | 10 of 10 | 121,870 | 0 | 0.0014 |
| anisotropic Gaussian | `stock_ta95` | 0.72× [0.72, 0.74] | 0 of 10 | 10 of 10 | 164,086 | 0 | 0.0015 |
| anisotropic Gaussian | `arcsinh` | 1.55× [1.52, 1.58] | 10 of 10 | 10 of 10 | 90,602 | 0 | 0.0014 |
| anisotropic Gaussian | `affine` | 1.23× [1.20, 1.24] | 10 of 10 | 10 of 10 | 99,100 | 0 | 0.0013 |
<!-- END GENERATED: ablation-table -->

<!-- BEGIN GENERATED: ablation-findings -->
- **Arcsinh vs stock NUTS** (paired tail-ESS per gradient, median over seeds): Task 1 7.1× (wins on 10 of 10 seeds); Task 3 3.8× (wins on 10 of 10 seeds); rotated t (ν = 1) 5.2× (wins on 10 of 10 seeds); rotated t (ν = 2) 1.7× (wins on 10 of 10 seeds); isotropic t (ν = 2) 1.6× (wins on 10 of 10 seeds); rotated Gaussian 0.9× (wins on 0 of 10 seeds); anisotropic Gaussian 1.5× (wins on 10 of 10 seeds).
- **Rescaling alone** (`affine`: the same calibration without the arcsinh) vs stock: Task 1 0.9× (wins on 4 of 10 seeds); Task 3 0.9× (wins on 4 of 10 seeds); rotated t (ν = 1) 1.9× (wins on 7 of 10 seeds); rotated t (ν = 2) 1.0× (wins on 6 of 10 seeds); isotropic t (ν = 2) 1.1× (wins on 8 of 10 seeds); rotated Gaussian 1.0× (wins on 4 of 10 seeds); anisotropic Gaussian 1.2× (wins on 10 of 10 seeds).
- **Reparameterising by hand** (`scale_mixture`, where the model is a Student-t scale mixture) vs `arcsinh`: Task 1 0.2× (better than arcsinh on 0 of 10 seeds); Task 3 0.9× (better than arcsinh on 1 of 10 seeds); rotated t (ν = 1) 10.8× (better than arcsinh on 10 of 10 seeds); rotated t (ν = 2) 5.3× (better than arcsinh on 10 of 10 seeds); isotropic t (ν = 2) 2.5× (better than arcsinh on 10 of 10 seeds).
- **Seeds converged** (max R-hat at most 1.01) with `arcsinh`: Task 1: 9 of 10 seeds; Task 3: 10 of 10 seeds; rotated t (ν = 1): 2 of 10 seeds; rotated t (ν = 2): 10 of 10 seeds; isotropic t (ν = 2): 9 of 10 seeds; rotated Gaussian: 10 of 10 seeds; anisotropic Gaussian: 10 of 10 seeds; with `stock`: Task 1: 0 of 10 seeds; Task 3: 1 of 10 seeds; rotated t (ν = 1): 0 of 10 seeds; rotated t (ν = 2): 2 of 10 seeds; isotropic t (ν = 2): 4 of 10 seeds; rotated Gaussian: 10 of 10 seeds; anisotropic Gaussian: 10 of 10 seeds.
- **Seeds with any post-warm-up divergence** with `arcsinh`: Task 1: 0 of 10 seeds; Task 3: 0 of 10 seeds; rotated t (ν = 1): 0 of 10 seeds; rotated t (ν = 2): 0 of 10 seeds; isotropic t (ν = 2): 0 of 10 seeds; rotated Gaussian: 0 of 10 seeds; anisotropic Gaussian: 4 of 10 seeds.
<!-- END GENERATED: ablation-findings -->

**How to read Table A.**
- **Tail warping vs rescaling.** Comparing the `affine` and `arcsinh` rows separates the two, because both arms use the same calibrated μ and R.
- **Why the scale mixture is strong on pure *t* targets.** Without a likelihood, the scale-mixture rewrite of a multivariate *t* is a Gaussian in η times a one-dimensional log-inverse-gamma factor, which NUTS samples easily.
- **Why it struggles on Task 1.** There the strongly identified coefficients make the non-centred rewrite a narrow ridge, hence its divergences.
- **When to prefer which.** The scale mixture needs the model to be known as a scale mixture, while the arcsinh pipeline is a black box.

### 5.2 Single-seed comparisons (Tables 1–5)

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
- **Rotated multivariate Cauchy.** The arcsinh arm converges on few seeds (Table A), so its ratio there is not a supported gain.
- **Unconverged denominators.** Several ratios compare against baselines that are not converged; for example, BlackJAX NUTS on Task 1. An unconverged denominator makes a ratio unreliable in either direction.
- **The best case for the method.** Task 1 is built so that most coordinates are independent and heavy-tailed. Task 3 is also product-form.
- **A noisy accuracy metric.** For a Cauchy marginal at an effective sample size of about a thousand, the Monte Carlo error of an estimated 0.99 quantile is comparable to the quantile errors reported in Tables 1 and 3. Single-seed differences in that column are therefore weak evidence. A tail-probability error with its MCSE would be a better metric.

**Not yet evaluated:**
- elliptical heavy tails with several scales, beyond the rotated *t* targets of Table A;
- nutpie's flow adaptation;
- diffeomorphic MCMC (Brešar & Mijatović, 2026);
- posteriordb posteriors with reference draws.

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
- **Heavy tails that are strongly rotated relative to the coordinates:** gains are smaller and less reliable (Table A).
- **Targets that stock NUTS already samples well:** the transform is not expected to help there.

**If you use `UnifiedRobustNUTS` directly:** pass `adapt_step_size=True` and set `burn_in` to at least `warmup_steps`.

## 8. References

1. Atchadé, Y. F. (2006). An adaptive version for the Metropolis adjusted Langevin algorithm with a truncated drift. *Methodology and Computing in Applied Probability*, 8(2), 235–254.
2. Bell, C., Łatuszyński, K., & Roberts, G. O. (2024). Adaptive stereographic MCMC. arXiv:2408.11780.
3. Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
4. Betancourt, M. (2018). Fitting the Cauchy. Stan case study.
5. Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A. (2024). autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm. *AISTATS*, PMLR 238. arXiv:2310.16782.
6. Bou-Rabee, N., Carpenter, B., Kleppe, T. S., & Liu, S. (2025). The within-orbit adaptive leapfrog no-U-turn sampler (WALNUTS). arXiv:2506.18746.
7. Bou-Rabee, N., Carpenter, B., & Marsden, M. (2024). GIST: Gibbs self-tuning for locally adaptive Hamiltonian Monte Carlo. arXiv:2404.15253.
8. Brešar, M., & Mijatović, A. (2026). Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions. arXiv:2608.04284.
9. Brosse, N., Durmus, A., Moulines, É., & Sabanis, S. (2019). The tamed unadjusted Langevin algorithm. *Stochastic Processes and their Applications*, 129(10), 3638–3663. doi:10.1016/j.spa.2018.10.002.
10. Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
11. Gorinova, M. I., Moore, D., & Hoffman, M. D. (2020). Automatic reparameterisation of probabilistic programs. *ICML*, PMLR 119, 3648–3657. arXiv:1906.03028.
12. He, Y., Balasubramanian, K., & Erdogdu, M. A. (2024). An analysis of transformed unadjusted Langevin algorithm for heavy-tailed sampling. *IEEE Transactions on Information Theory*, 70(1), 571–593. arXiv:2201.08349.
13. Hickling, T., & Prangle, D. (2025). Flexible tails for normalizing flows. *ICML*, PMLR 267, 23155–23178. arXiv:2406.16971.
14. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15, 1593–1623.
15. Hoffman, M., Sountsov, P., Dillon, J. V., Langmore, I., Tran, D., & Vasudevan, S. (2019). NeuTra-lizing bad geometry in Hamiltonian Monte Carlo using neural transport. arXiv:1903.03704.
16. Jaini, P., Kobyzev, I., Yu, Y., & Brubaker, M. (2020). Tails of Lipschitz triangular flows. *ICML*. arXiv:1907.04481.
17. Johnson, L. T., & Geyer, C. J. (2012). Variable transformation to obtain geometric ergodicity in the random-walk Metropolis algorithm. *The Annals of Statistics*, 40(6), 3050–3076. doi:10.1214/12-AOS1048.
18. Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *The Review of Economic Studies*, 65(3), 361–393.
19. Laszkiewicz, M., Lederer, J., & Fischer, A. (2022). Marginal tail-adaptive normalizing flows. *ICML*, PMLR 162, 12020–12048. arXiv:2206.10311.
20. Liang, F., Mahoney, M. W., & Hodgkinson, L. (2022). Fat-tailed variational inference with anisotropic tail adaptive flows. *ICML*, PMLR 162, 13257–13270. arXiv:2205.07918.
21. Livingstone, S., Betancourt, M., Byrne, S., & Girolami, M. (2019). On the geometric ergodicity of Hamiltonian Monte Carlo. *Bernoulli*, 25(4A), 3109–3138. doi:10.3150/18-BEJ1083.
22. Livingstone, S., & Zanella, G. (2022). The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. *Journal of the Royal Statistical Society: Series B*, 84(2), 496–523.
23. Lytras, I., & Sabanis, S. (2025). Taming under isoperimetry. *Stochastic Processes and their Applications*, 188, 104684. arXiv:2311.09003.
24. Modi, C., Barnett, A., & Carpenter, B. (2024). Delayed rejection Hamiltonian Monte Carlo for sampling multiscale distributions. *Bayesian Analysis*, 19(3), 815–842. doi:10.1214/23-BA1360.
25. Neal, R. M. (2003). Slice sampling. *The Annals of Statistics*, 31(3), 705–767.
26. nutpie: a NUTS implementation with gradient-based and normalizing-flow adaptation. https://github.com/pymc-devs/nutpie.
27. Papaspiliopoulos, O., Roberts, G. O., & Sköld, M. (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 22(1), 59–73.
28. Piironen, J., & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018–5051.
29. Power, S., & Vasdekis, G. (2025). Some aspects of robustness in modern Markov chain Monte Carlo. arXiv:2511.21563.
30. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341–363.
31. Rubin, D. B. (1981). Estimation in parallel randomized experiments. *Journal of Educational Statistics*, 6(4), 377–401.
32. The Mathlib Community. The Lean 4 mathematical library (mathlib4). https://github.com/leanprover-community/mathlib4.
33. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved $\widehat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
34. Yang, J., Łatuszyński, K., & Roberts, G. O. (2024). Stereographic Markov chain Monte Carlo. *The Annals of Statistics*, 52(6), 2692–2713. doi:10.1214/24-AOS2426.
