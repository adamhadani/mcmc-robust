# Tukey-h pullback for NUTS on heavy tails: findings

> **Exploration snapshot, branch `explore/tukey-h-transform`, not maintained docs.** Nothing here is a library claim. Every number is copied from a committed file in `explore/tukey_h/results/`. `tables.md` and `summary.json` are generated from the run files by `summarise.py`. The math is plain text.

## Bottom line

1. **The h-map works, and the transform is exact.** Its log-det and gradient match JAX autodiff to 5e-15 and 2e-15 (relative) for c in {0.5, 1, 4, 16, inf}, out to the float64 overflow edge.
2. **On axis-aligned heavy tails it beats the repo's arcsinh pipeline by a wide, seed-robust margin, once (R, c) are calibrated jointly** by the Tukey-h (Lambert W × Gaussian) MLE. Paired tail-ESS/grad against arcsinh, main seeds 0–4 / held-out seeds 10–14:
   - product t with ν = 0.5: 21.9× [13.3, 22.7] / 21.8× [16.4, 24.0];
   - ν = 1: 4.4× / 4.6×;
   - Task 1: 2.2× / 2.6×;
   - Task 3: 1.7× / 1.5×;
   - ν = 2: 1.4× / 1.3×.
3. **On Gaussian targets it removes arcsinh's benign tax.** It runs at 0.99× stock, which is 1.15–1.28× arcsinh.
4. **It does not help where arcsinh is weak.**
   - Rotated multivariate t: 1.00–1.04× arcsinh (main) and 0.82–0.85× (held-out).
   - Product t with ν = 5: 0.77–0.94×.
5. **The pre-registered rule (c = t-MLE ν̂, R from the arcsinh IQR rule) leaves most of the gain on the table.** It gives 5.8×, 2.3× and 1.8× on ν = 0.5, ν = 1 and Task 1. The map's shape depends only on R²·c, and the IQR R is 1.5–2× too large for this family.
6. **Misspecifying c by ×1/4 or ×4 never breaks the sampler.** There were 0 divergences and R-hat stayed ≤ 1.033. c×4 drifts back to about arcsinh level (0.61–1.90× arcsinh).
7. **The light-tail cost is small.** Forcing c ≈ 5 onto Gaussian coordinates costs 6–7% of tail-ESS/grad, plus up to 11 divergences per 4,000 draws.
8. **Cost scaling follows the predicted ordering, but not arcsinh's √E law.**
   - x-space: cost grows like exp(E/2) (slope 0.47, predicted 0.5).
   - h-map: flat.
   - arcsinh: nearly flat in 1-D (slope 0.21, not 0.5), and only 1.6× growth across the 10-d energy bins.
9. **Novelty:** no prior use as an exact HMC/NUTS/MALA pullback was found. The combination appears new; the ingredients are known.
10. **Recommendation: iterate** (§10).

## 1. What I ran

- **Transform** (`tukey_h.py`): x = μ + R·z·exp(z²/(2c)).
  - log-det term: log R + z²/(2c) + log1p(z²/c).
  - Forward map: z = u·exp(−W(u²/c)/2), with u = (x − μ)/R. This form is exact at c = ∞ and never squares u.
  - W is the principal Lambert W, computed from log y by Newton iteration.
  - c = ∞ gives the affine map.
- **Pipeline:**
  - **Phase 1** is identical to the repo's `sample_blackjax_unified_nuts`: same key splits, initial arcsinh with R = 3, 75 of 500 tuning steps, (μ, R) calibrated from 152 pooled draws.
  - The h-map arms reuse the arcsinh arm's Phase-1 draws. Their (μ, R) matched the repo's to 0.0 in every row. Each arm still pays for Phase 1 in its gradient count.
  - **Phases 2–3:** BlackJAX `window_adaptation` per chain, then jit + vmap + scan on the exact pullback.
- **Settings:** tune 500, draws 1000, 4 chains, max depth 10.
- **Seeds:** main grid seeds 0–4; held-out seeds 10–14.
- **Targets:** product t with ν ∈ {0.5, 1, 2, 5} (d = 10); Task 1; Task 3; rotated multivariate t with ν ∈ {1, 2} (d = 10, condition number 25); isotropic Gaussian and rotated Gaussian (d = 20, condition number 100).
- **Arms in the pre-registered main grid:**
  - `stock` (`run_blackjax_sampler`);
  - `arcsinh`;
  - `hmap_tmle`: heavy iff ν̂ ≤ 5, c = ν̂, light coordinates get c = ∞;
  - `hmap_qr`: heavy iff q99/q75 > 3, c = Hill α̂;
  - `hmap_tmle_c0.25` and `hmap_tmle_c4`: c misspecified by ×1/4 and ×4;
  - `hmap_forced`: Hill c on every coordinate.
- **Arms added after the main grid:**
  - `hmap_lwmle`: t-MLE heavy test, then joint (R, c) by Tukey-h MLE with μ = median (Goerg 2015, Definition 2);
  - `hmap_lwmle_all`: the same MLE on every coordinate;
  - the κ sweep: c = κ·ν̂.
- **Metrics:** from `evaluate_chains_with_arviz`, plus the tail-probability error abs(P̂(x_i < q_p) − p)/0.01 at exact quantiles for p ∈ {0.01, 0.99}. Its MCSE comes from the indicator's ESS.
- **Volume:** 350 + 100 + 305 sampler runs, 50 Phase-1 runs and 18 cost-scaling runs.
- **Provenance:** `git_sha` is HEAD at run start. grid_A/B record `92a2596`; their code is committed in `bcca868` (reformatting only). grid_lwmle and kappa_sweep record `bcca868`; their code is in `18ef83f` and `c8983ec`.

## 2. Correctness (`results/verify_results.json`, `all_pass: true`)

| Check | Max relative error |
|---|---|
| Lambert W vs scipy, log y ∈ [−700, 690] | 2.8e-15 |
| Lambert W identity residual, log y up to 1e300 | 0.0 |
| z round trip, and x round trip divided by the condition number 1 + z²/c | 1.1e-13 |
| log-det vs `jax.grad(inverse)` | ≤ 2.2e-16 |
| log-det vs log-form autodiff, all z | 5.0e-15 |
| log-det vs `slogdet(jacfwd)`, d = 5 | 4.2e-16 |
| pullback gradient vs `jax.grad(wrap_logdensity)` | 1.6e-15 |
| pullback gradient, via the x → z round trip | 3.4e-12 |
| normalisation (absolute) | 2.5e-10 |

Tail curvature at z = 3√c, 6√c and 10√c:
- ν = 1, c = 1: −0.938, −0.996, −0.999 (target −1);
- ν = 3, c = 1: −2.72, −2.94, −2.98 (target −3).

## 3. Calibration rules (`results/calib_results.json`)

| Rule | Light flagged heavy | Heavy missed | Missed on t₅ | Median abs(log(α̂/α)) |
|---|---|---|---|---|
| Hill ≤ 3.5 | 8.7% | 8.8% | 46% | 0.24–0.35 |
| Hill or q-ratio (repo's rule) | 8.7% | 7.3% | 46% | 0.24–0.35 |
| q99/q75 > 3 | 1.1% | 11.7% | 68% | 0.21–0.35 |
| **t-MLE ν̂ ≤ 5** | **0.0%** | 10.4% | 72% | **0.12–0.29** |

Hill flags 20% of Task 1's Gaussian coordinates: at n ≈ 150 a Gaussian sample gives α̂ ≈ 4–5. t-MLE was chosen before the grid. It made one false heavy flag on a held-out run (Task 3, seed 10, c = 8.2).

## 4. Does it beat arcsinh?

Sources: `tables.md` Table 2 (main seeds) and Table 8 (held-out). All ratios vs arcsinh unless stated.

| Target | arcsinh/stock | tmle (pre-reg.) | lwmle main | lwmle held-out | lwmle_all main | lwmle_all held-out |
|---|---|---|---|---|---|---|
| prod_t0.5 | 31.44× [12.89, 62.23] | 5.76× [4.99, 6.24] | 21.87× [13.31, 22.65] | 21.81× [16.43, 23.98] | 21.87× [13.31, 22.65] | 21.81× [16.43, 23.98] |
| prod_t1 | 16.99× [8.05, 26.65] | 2.29× [2.01, 2.62] | 4.43× [3.53, 5.28] | 4.60× [4.37, 5.35] | 4.43× [3.53, 5.28] | 4.60× [4.37, 5.35] |
| prod_t2 | 6.11× [3.94, 7.19] | 1.19× [0.76, 1.21] | 1.36× [1.09, 1.54] | 1.28× [0.77, 1.41] | 1.36× [1.25, 1.54] | 1.33× [1.28, 1.41] |
| prod_t5 | 1.54× [1.40, 1.69] | 0.78× [0.70, 0.85] | 0.77× [0.73, 0.85] | 0.86× [0.76, 0.88] | 0.91× [0.87, 0.97] | 0.94× [0.92, 0.99] |
| task1 | 12.32× [7.40, 20.42] | 1.76× [1.55, 1.82] | 2.17× [2.04, 2.38] | 2.62× [2.07, 2.99] | 2.23× [2.20, 2.59] | 2.54× [2.09, 3.14] |
| task3 | 3.42× [2.09, 5.20] | 1.26× [1.22, 1.70] | 1.68× [1.30, 1.75] | 1.48× [1.37, 2.03] | 1.66× [1.49, 1.81] | 1.66× [1.50, 2.01] |
| corr_t1 | 6.90× [1.49, 47.71] | 1.09× [0.90, 2.06] | 1.04× [0.77, 1.77] | 0.82× [0.57, 1.17] | 1.04× [0.77, 1.77] | 0.82× [0.57, 1.17] |
| corr_t2 | 1.59× [0.92, 3.34] | 1.16× [0.93, 1.34] | 1.00× [0.90, 1.35] | 0.85× [0.69, 1.01] | 1.00× [0.90, 1.35] | 0.85× [0.68, 1.01] |
| iso_gauss | 0.83× [0.78, 0.91] | 1.19× [1.08, 1.31] | 1.19× [1.08, 1.31] | 1.28× [1.24, 1.39] | 1.19× [1.08, 1.30] | 1.28× [1.26, 1.45] |
| corr_gauss | 0.85× [0.80, 0.94] | 1.15× [1.06, 1.23] | 1.15× [1.06, 1.23] | 1.19× [1.10, 1.28] | 1.12× [1.07, 1.18] | 1.17× [1.09, 1.26] |

**Against stock,** `hmap_lwmle` reaches 653× on ν = 0.5, 72× on ν = 1, 25× on Task 1, 4.9× on Task 3 and 0.99× on the Gaussians.

**Worst-coordinate ratios hold up.** For `lwmle_all` vs arcsinh on held-out seeds: 22.0×, 4.7×, 3.4× and 1.9× on ν = 0.5, ν = 1, Task 1 and Task 3.

**The hard ν̂ ≤ 5 cut can leave a t₂ coordinate affine.** That gives `lwmle` a held-out `prod_t2` worst-coordinate ratio of 0.71× [0.46, 1.39]. `lwmle_all` gives 1.22× [0.86, 1.39].

**R-hat:** h-map arms stay ≤ 1.033 everywhere. Stock reaches 1.323 on `prod_t0.5` and 1.579 on `corr_t1`.

## 5. Misspecification and the R²·c effect (Tables 4 and 7)

| Target | c×1/4 / tmle | c×4 / tmle | c×4 / arcsinh |
|---|---|---|---|
| prod_t0.5 | 2.70× [2.20, 2.92] | 0.33× [0.33, 0.37] | 1.90× [1.83, 2.08] |
| prod_t1 | 1.99× [1.66, 2.48] | 0.39× [0.38, 0.47] | 0.89× [0.84, 1.07] |
| prod_t2 | 1.31× [1.20, 1.36] | 0.55× [0.51, 0.61] | 0.61× [0.46, 0.66] |
| task1 | 1.37× [1.31, 1.48] | 0.52× [0.51, 0.55] | 0.92× [0.82, 1.01] |
| task3 | 1.16× [0.89, 1.53] | 0.58× [0.55, 0.64] | 0.79× [0.72, 0.94] |
| corr_t1 | 0.72× [0.24, 0.78] | 0.88× [0.65, 0.90] | 0.94× [0.81, 1.71] |
| corr_t2 | 0.73× [0.68, 0.82] | 0.96× [0.78, 1.03] | 0.97× [0.91, 1.29] |

All 70 misspecified heavy-target runs had 0 divergences and R-hat ≤ 1.033. A too-large c degrades toward arcsinh, not stock: on Task 1 the median tail-ESS/grad is 0.0131 for c×4, against 0.0142 for arcsinh and 0.0012 for stock.

**Held-out κ sweep** (c = κ·ν̂, vs arcsinh):

| Target | κ=1/8 | κ=1/4 | κ=1/2 | κ=1 |
|---|---|---|---|---|
| prod_t0.5 | 22.37× [18.45, 23.58] | 14.23× [13.79, 15.99] | 9.26× [8.09, 9.73] | 5.41× [4.91, 5.94] |
| prod_t1 | 4.82× [3.57, 5.57] | 5.21× [4.51, 6.30] | 4.25× [3.47, 5.07] | 2.57× [2.22, 3.09] |
| prod_t2 | 1.13× [0.96, 1.25] | 1.19× [1.00, 1.45] | 1.25× [1.08, 1.41] | 1.13× [0.59, 1.25] |
| task1 | 2.80× [2.11, 3.29] | 2.75× [2.40, 3.26] | 2.50× [2.08, 2.98] | 1.77× [1.64, 2.14] |
| task3 | 1.18× [1.00, 1.46] | 1.46× [1.11, 2.04] | 1.44× [1.34, 2.00] | 1.19× [1.12, 1.44] |
| corr_t1 | 0.72× [0.48, 1.01] | 0.79× [0.57, 1.18] | 0.74× [0.67, 0.96] | 0.83× [0.76, 1.68] |
| corr_t2 | 0.60× [0.54, 0.73] | 0.80× [0.70, 0.86] | 0.84× [0.76, 0.95] | 1.01× [0.68, 1.21] |

**Why κ = 1 is wrong:** rescaling z by λ maps (R, c) to (Rλ, c/λ²), and the diagonal metric absorbs that rescaling. So the map's shape depends only on R²·c.
- In an uncommitted check on 4,000 exact t draws, the Tukey-h MLE gives R = 1.50, 1.18, 1.10 and 1.04 for ν = 0.5, 1, 2 and 5. The IQR rule gives R = 3.38, 2.16, 1.80 and 1.62.
- The per-run R and c are committed in `grid_lwmle.json` and `kappa_sweep.json`.

**The MLE removes the constant.**
- Held-out, it matches or beats the best κ on ν = 0.5, ν = 2 and Task 3, and falls slightly short on ν = 1 and Task 1.
- It avoids the small-κ penalty on rotated targets.
- It was designed post hoc and confirmed on new seeds only, not new targets.

## 6. Accuracy (Table 6)

**Tail-probability error divided by its MCSE:**
- Transformed arms: medians of 0.54–1.11 over targets and seeds, consistent with unbiased sampling (about 0.8 is expected).
- Stock NUTS: 5.19 [2.82, 6.11] on `prod_t0.5`, and 1.38–1.44 on the rotated targets.

The errors themselves are noise-dominated. On Task 1 the error is 12.7% for `hmap_tmle` (MCSE 18.6%) against 20.1% for arcsinh (MCSE 22.0%).

## 7. Light-tail cost (Table 5)

**`hmap_forced`** gives light coordinates a median c of about 5 (2.7–9.7).
- Against the affine map on Gaussians: 0.94× [0.81, 1.05] and 0.93× [0.84, 1.00]; worst coordinate 0.92× and 0.86×.
- Divergences: up to 11 per run on `iso_gauss` and 5 on Task 3, against 0 for `hmap_tmle`.
- Still 1.12× and 1.08× arcsinh.

**`hmap_lwmle_all`** gives Gaussian coordinates c ≈ 9–1,000.
- Efficiency: 0.99× and 0.97× stock.
- Divergences: up to 3 per run on the main seeds and 12 on a held-out run.
- The single t-MLE false positive (c = 8.2) caused 22 divergences.

**On `prod_t5`,** forcing the map onto every coordinate helps: 1.24× `hmap_tmle`. The hard cut is too conservative for mild tails.

## 8. Cost scaling (`results/cost_scaling.json`, `results/cost_scaling.png`)

**Prediction** (Betancourt 2017, arXiv:1701.02434, §4.3: T_opt ∝ H^((2−β)/(2β))): exp(E/2) in x-space, √E for arcsinh, O(1) for the h-map.

**Fixed start, 1-D Cauchy, 400 momenta per x0:**

| x0 | 3 | 10 | 100 | 1e3 | 1e4 | 1e6 |
|---|---|---|---|---|---|---|
| stock, mean steps (fraction at max depth) | 4.8 | 14.2 | 115.6 | 613 (0.31) | 970 (0.90) | 1023 (1.00) |
| arcsinh | 3.3 | 4.0 | 5.0 | 5.1 | 5.3 | 5.6 |
| Tukey-h | 3.1 | 3.2 | 3.0 | 3.0 | 3.0 | 2.9 |

**Fits:**
- stock: slope of log(steps) on E is 0.47 for median steps and 0.42 for mean steps (prediction 0.5);
- arcsinh: log-log slope 0.21 (prediction 0.5);
- h-map: −0.03 (prediction 0).

In 1-D, NUTS's U-turn fires at the first momentum reversal, after a time of about abs(p)/abs(U′(z)). That time is constant on a Laplace tail, so arcsinh shows no √E growth per transition.

**10-d product Cauchy, mean steps by energy-quantile bin (median of 3 seeds):**

| Bin | 0–50% | 50–75% | 75–90% | 90–99% | 99–99.9% | 99.9–100% |
|---|---|---|---|---|---|---|
| stock | 106 | 167 | 235 | 332 | 430 | 394 |
| arcsinh | 17.0 | 20.6 | 22.8 | 24.5 | 26.8 | 28.0 |
| Tukey-h | 10.5 | 11.7 | 12.1 | 11.8 | 11.3 | 12.1 |

Stock's fraction of transitions at max depth rises from 0.4% in the bulk to 33% in the top bin.

**Where the h-map's gain over arcsinh comes from (10-d):**
- step size 0.32–0.38 against 0.19–0.24;
- energy ESS per gradient 0.023–0.034 against 0.009–0.011;
- tail-excursion return time 3–4 transitions against 5–6 (stock: 12–23).

In 1-D the return times are 4–5 against 7–9 (stock: 15–17).

## 9. Novelty

**Sources checked:**
- **Goerg (arXiv:1010.2265; Sci. World J. 2015):** Lambert W × F, Y = U·exp(δU²/2)·σ + μ, which is exactly this map with δ = 1/c. Estimated by MLE or IGMM. No MCMC use.
- **Hickling & Prangle (ICML 2025, arXiv:2406.16971):** the tail-transform flow R(z) = μ + σ(s/λ)(erfc(|z|/√2)^(−λ) − 1) maps Gaussian tails to GPD tails, so its inverse Gaussianises polynomial tails. Used only for density estimation and VI.
- **nutpie flows** (`normalizing_flow.py`, read 2026-10-24): `Contract` and `Contract2` grow like a power of abs(x), so by my reading they cannot turn a polynomial tail into a Gaussian one. Both are off by default in `make_flow`.
- **NeuTra (arXiv:1903.03704)** and **transport-map MCMC (arXiv:1412.5492):** learned transports, and the latter's maps revert to linear beyond a radius, preserving the tail class.
- **Johnson & Geyer (2012):** radial exponential maps for random-walk Metropolis, aiming for super-exponential tails. That is the opposite of HMC's β ≤ 2 window (Livingstone et al. 2019, Cor. 2.3), which this map hits at β = 2.
- **Pachebat (arXiv:2605.20068):** Hill-selected signed log for flow matching.
- **Power & Vasdekis (§4.3(iii)):** list coordinate-wise transformations as an open problem.

**Verdict:**
- **Not found:** any use of Tukey's h, Lambert W × F or the TTF inverse as an exact pullback for HMC, NUTS or MALA, or any automatic per-coordinate calibration of such a map for a sampler.
- **How a referee would read it:** a closed-form, tail-class-changing instance of NeuTra-style transport.
- **What would make it a contribution:**
  - the β = 2 ergodicity theory for regularly varying product targets;
  - the scale-matched calibration;
  - benchmarks against nutpie and Brešar & Mijatović (2026).

## 10. Recommendation: iterate

**Don't drop it.** On axis-aligned heavy tails it gives 1.3–22× over arcsinh, which itself gives 6–31× over stock there. That holds across 10 seeds, with no convergence problems on heavy targets, and it removes the ~15% tax on Gaussians.

**Don't productionise yet:**
- the MLE calibration was found post hoc and is confirmed on held-out seeds only;
- there is no gain on rotated tails (0.8–1.0×), and ν ≈ 5 loses 6–23%;
- the heavy/light decision causes occasional divergences.

**Next steps:**
- A guarded MLE: set c = ∞ unless a likelihood-ratio test rejects h = 0.
- Unseen targets: horseshoe, mixtures, skewed tails (using h± per tail) and posteriordb.
- Head-to-head runs against nutpie.
- Theory for the Lean agent: β = 2 tails of the scale-matched pullback, so Livingstone et al. Cor. 2.3 applies; the R²·c invariance lemma.
- Then, possibly, a `transform="tukey_h"` option in `sample_blackjax_unified_nuts`, with arcsinh kept as the default and these autodiff checks as unit tests.

## Independent spot-check (coordinating session)

`results/spotcheck_seed20.json` re-runs `run_grid.py` on seed 20, which is in neither seed set above, with arms `stock`, `arcsinh` and `hmap_lwmle`.

| Target | `hmap_lwmle` / `arcsinh` | `hmap_lwmle` / `stock` | Notes |
|---|---|---|---|
| Task 1 | 2.1× | 22× | The heavy test picked exactly the 7 Cauchy coordinates. |
| `prod_t1` | 3.1× | – | Below the 3.5–5.3 range above, but the same direction. |
| `corr_gauss` | 1.32× | 1.0× | |

There were no divergences, and R-hat was at most 1.014 throughout. One unseen seed is consistent with §4.

## Files

| File | Contents |
|---|---|
| `tukey_h.py` | transform, Lambert W, calibration rules, pipeline |
| `verify_transform.py` | correctness checks (§2) |
| `calib_compare.py` | calibration-rule study (§3) |
| `run_grid.py` | arm/target/seed runner (§4, §5, §7) |
| `summarise.py` | generates `tables.md` and `summary.json` |
| `cost_scaling.py`, `plot_cost_scaling.py` | cost-scaling study and figure (§8) |
| `common.py` | targets, exact quantiles, metrics |

Run each with `uv run python explore/tukey_h/<script>.py`.
