import Mathlib.Analysis.SpecialFunctions.Arsinh
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic.Linter.Lint


/-!
# Lean 4 Formal Verification: Robust MCMC Mathematical Foundations

Formal proofs verifying core geometric, dynamical, and Markov kernel invariants of novel
gradient-based MCMC samplers resolving roughness and flatness pathologies.

## Verified Properties:
1. **Shift-Equivariant Coordinate Diffeomorphism Roundtrip Invertibility**:
   - `sinh (arsinh x) = x` and `arsinh (sinh y) = y`
   - Parameterized location $\mu \in \mathbb{R}$ and scale $R > 0$:
     $\psi_{\mu, R}(\phi_{\mu, R}(x)) = x$ and $\phi_{\mu, R}(\psi_{\mu, R}(z)) = z$
   - Translation invariance: $\phi_{\mu + c, R}(x + c) = \phi_{\mu, R}(x)$

2. **Metric Scale Positivity & Derivative Identity**:
   - $\frac{d}{dx} \operatorname{arsinh}(x) = \frac{1}{\sqrt{1 + x^2}}$
   - Positivity: $\forall x \in \mathbb{R}, \frac{1}{\sqrt{1 + x^2}} > 0$
   - Boundedness: $\forall x \in \mathbb{R}, \frac{1}{\sqrt{1 + x^2}} \le 1$
   - Scaled chain rule: $\frac{d}{dx} \operatorname{arsinh}(x / R) = \frac{1}{R \sqrt{1 + (x/R)^2}} > 0$

3. **Gentle Stiffness Attenuation & Exact Discrete Basin of Attraction**:
   - Pure square-root attenuation: $\rho(\kappa) = (1 + \kappa)^{-1/2}$
   - Exact discrete basin bound for pure square-root damping:
     For radial gradient $g(r) = r^3$ (Quartic potential) and step size $h > 0$,
     if $0 < r < 4 \gamma / h$, then the squared damped step satisfies
     $\Delta(r)^2 = \frac{(h r^3)^2}{1 + h \gamma r^3} < \frac{h r^3}{\gamma} < (2r)^2$,
     establishing non-overshoot ($\Delta(r) < 2r$) inside the basin $r < 4\gamma/h$.

4. **Soft-Capped Gentle Attenuation Global Drift Bound & Non-Overshoot Contraction**:
   - Soft-capped attenuation: $\rho_{\text{soft}}(\kappa, \kappa_{\max}) = (1 + \kappa + (\kappa / \kappa_{\max})^2)^{-1/2}$
   - Global dimensionless drift bound:
     $\forall \kappa \ge 0, \kappa_{\max} > 0, \quad (\kappa \cdot \rho_{\text{soft}}(\kappa, \kappa_{\max}))^2 < \kappa_{\max}^2$
   - Global Euclidean displacement cap:
     For $\Delta = h g \cdot \rho_{\text{soft}}(h \gamma g, \kappa_{\max})$, we have $\Delta^2 < (\kappa_{\max} / \gamma)^2$
     and $\Delta < \kappa_{\max} / \gamma$.
   - Global non-overshoot radial contraction:
     For any initial radius $r > \kappa_{\max} / \gamma$ and $g > 0$, the post-drift radius $r - \Delta$
     satisfies $0 < r - \kappa_{\max} / \gamma < r - \Delta < r$, proving that soft-capped GentleMALA
     strictly contracts toward the origin and never overshoots from any arbitrarily distant initialization.

5. **Metropolis-Hastings Kernel Reversibility (Detailed Balance)**:
   - For any positive target density $\pi(x), \pi(y) > 0$ and positive proposal kernel $q(x, y), q(y, x) > 0$,
     the Metropolis-Hastings acceptance probability $\alpha(x, y) = \min\!\left(1, \frac{\pi(y) q(y, x)}{\pi(x) q(x, y)}\right)$
     satisfies $0 \le \alpha(x, y) \le 1$ and exact detailed balance:
     $$\pi(x) q(x, y) \alpha(x, y) = \pi(y) q(y, x) \alpha(y, x).$$

6. **Applied Heavy-Tailed & Shrinkage Posterior Score Bounds (§7)**:
   - Student-$t$ negative log-kernel $U_t(\nu, r) = \frac{\nu + 1}{2} \log(1 + r^2 / \nu) \ge 0$ for $\nu > 0$.
   - Exact score identity $\frac{d}{dr} U_t(\nu, r) = \frac{(\nu + 1) r}{\nu + r^2}$ and zero at $r = 0$.
   - Global virial score bound $|r \cdot s_t(\nu, r)| \le \nu + 1$ for all $r \in \mathbb{R}$, proving outlier robustness.
   - Non-negativity of the Cauchy / Horseshoe local log-prior penalty $\log(1 + (\lambda / \tau)^2) \ge 0$.

7. **Pure-JAX 3-Phase Scale & Location Calibration Bounds (§8)**:
   - Clipped normal-consistent coordinate scale calibration map:
     $$R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c) = \min(R_{\max}, \max(R_{\min}, c \cdot \sigma))$$
   - Strict interval bounds $0 < R_{\min} \le R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c) \le R_{\max}$,
     monotonicity in $\sigma$ for $c \ge 0$, and unconditional roundtrip invertibility of
     $\psi_{\mu, R_{\text{cal}}} \circ \phi_{\mu, R_{\text{cal}}}$ for any sample median $\mu \in \mathbb{R}$
     and dispersion $\sigma \in \mathbb{R}$ (including zero-variance $\sigma \le 0$).

8. **Tail Envelopes of Arcsinh Pullbacks of Cauchy Marginals (§9)**:
   - Perfect calibration gives the hyperbolic secant law: $\pi\, p_z(z) = \mathrm{sech}\, z$.
   - For any location error $\delta$ and scale ratio $r > 0$, with $K = (r^2 + 1 + \delta^2)/r$:
     $e^{-|z|} / K \le \pi\, p_z(z) \le 2 K e^{-|z|}$ for all $z$ (Laplace tail, rate 1).

9. **Tukey h-Map (§10)**: $x = \mu + R z e^{z^2/(2c)}$ has derivative
   $R e^{z^2/(2c)} (1 + z^2/c) > 0$, log-Jacobian $\log R + z^2/(2c) + \log(1 + z^2/c)$, and is a
   bijection of $\mathbb{R}$; rescaling $z = \lambda w$ maps $(R, c)$ to $(R\lambda, c/\lambda^2)$,
   so only $R^2 c$ is identifiable. Its Cauchy pullback has explicit two-sided Gaussian envelopes
   proportional to $e^{-z^2/(2c)}$, valid for all $z$.

10. **Impossibility of a Common Tail Map (§11)**: if a differentiable increasing map $g \to \infty$
    gives the Cauchy pullback potential a linear lower envelope $a z - b$ on $[0, \infty)$, then
    $g(z) \ge (a / (2 e^b)) e^{a z}$ and the Gaussian pullback potential is at least
    $c\, e^{2 a z} - C$; hence no such $g$ also keeps the Gaussian pullback potential below
    $A z^2 + B$. Arcsinh is an instance ($a = 1$).

11. **Leapfrog Stability Region of the Arcsinh Gaussian Pullback (§12)**: for
    $U(z) = r^2 \sinh^2 z / 2 - \log\cosh z$, $U''(z) = r^2 + 2 s^2 - \mathrm{sech}^2 z$ with
    $s = r \sinh z$; $\varepsilon^2 U'' \le 4$ when $s^2 \le 2/\varepsilon^2 - r^2/2$, and
    $\varepsilon^2 U'' \ge 4$ when $s^2 \ge 2/\varepsilon^2 - r^2/2 + 1/2$. $U''$ is nondecreasing in
    $|z|$, so the unstable set is a symmetric tail region, and $U''(0) = r^2 - 1$.

Author: MCMC Robustness Research Team
Toolchain: Lean 4 + Mathlib
-/

open Real

namespace McmcRobust

noncomputable section

/-! ============================================================================
    PROPERTY 1: Exact Invertibility of Shift-Equivariant Arcsinh/Sinh Diffeomorphism
    ============================================================================ -/

/-- Unscaled right-inverse identity: sinh(arsinh(x)) = x. -/
theorem arcsinh_inv_right (x : ℝ) : sinh (arsinh x) = x :=
  sinh_arsinh x

/-- Unscaled left-inverse identity: arsinh(sinh(y)) = y. -/
theorem arcsinh_inv_left (y : ℝ) : arsinh (sinh y) = y :=
  arsinh_sinh y

/-- Forward coordinate diffeomorphism mapping primal coordinate x to transformed coordinate z:
    z = \operatorname{arsinh}(x / R) for scale parameter R > 0. -/
def forwardMap (R : ℝ) (x : ℝ) : ℝ :=
  arsinh (x / R)

/-- Inverse coordinate diffeomorphism mapping transformed coordinate z back to primal coordinate x:
    x = R * \sinh(z) for scale parameter R > 0. -/
def inverseMap (R : ℝ) (z : ℝ) : ℝ :=
  R * sinh z

/-- Roundtrip invertibility: inverseMap after forwardMap recovers primal coordinate exactly:
    \forall x \in \mathbb{R}, R \ne 0 \implies \psi_R(\phi_R(x)) = x. -/
theorem scaled_invertibility_roundtrip_forward (R x : ℝ) (hR : R ≠ 0) :
    inverseMap R (forwardMap R x) = x := by
  dsimp [forwardMap, inverseMap]
  rw [sinh_arsinh]
  exact mul_div_cancel₀ x hR

/-- Roundtrip invertibility: forwardMap after inverseMap recovers transformed coordinate exactly:
    \forall z \in \mathbb{R}, R \ne 0 \implies \phi_R(\psi_R(z)) = z. -/
theorem scaled_invertibility_roundtrip_inverse (R z : ℝ) (hR : R ≠ 0) :
    forwardMap R (inverseMap R z) = z := by
  dsimp [forwardMap, inverseMap]
  rw [mul_div_cancel_left₀ (sinh z) hR]
  exact arsinh_sinh z

/-- Shift-equivariant forward coordinate diffeomorphism with location \mu and scale R:
    z = \operatorname{arsinh}((x - \mu) / R). -/
def shiftedForwardMap (mu R x : ℝ) : ℝ :=
  arsinh ((x - mu) / R)

/-- Shift-equivariant inverse coordinate diffeomorphism with location \mu and scale R:
    x = \mu + R * \sinh(z). -/
def shiftedInverseMap (mu R z : ℝ) : ℝ :=
  mu + R * sinh z

/-- Shift-equivariant roundtrip invertibility (forward then inverse):
    \psi_{\mu, R}(\phi_{\mu, R}(x)) = x for all R \ne 0. -/
theorem shifted_invertibility_roundtrip_forward (mu R x : ℝ) (hR : R ≠ 0) :
    shiftedInverseMap mu R (shiftedForwardMap mu R x) = x := by
  dsimp [shiftedForwardMap, shiftedInverseMap]
  rw [sinh_arsinh, mul_div_cancel₀ (x - mu) hR]
  ring

/-- Shift-equivariant roundtrip invertibility (inverse then forward):
    \phi_{\mu, R}(\psi_{\mu, R}(z)) = z for all R \ne 0. -/
theorem shifted_invertibility_roundtrip_inverse (mu R z : ℝ) (hR : R ≠ 0) :
    shiftedForwardMap mu R (shiftedInverseMap mu R z) = z := by
  dsimp [shiftedForwardMap, shiftedInverseMap]
  have hsub : mu + R * sinh z - mu = R * sinh z := by ring
  rw [hsub, mul_div_cancel_left₀ (sinh z) hR]
  exact arsinh_sinh z

/-- Exact translation invariance of shifted coordinate diffeomorphism:
    shifting both x and location \mu by c leaves transformed coordinate z invariant. -/
theorem shiftedForwardMap_translation_invariant (mu R x c : ℝ) :
    shiftedForwardMap (mu + c) R (x + c) = shiftedForwardMap mu R x := by
  dsimp [shiftedForwardMap]
  have hsub : x + c - (mu + c) = x - mu := by ring
  rw [hsub]

/-- Strict positivity of the inverse coordinate diffeomorphism Jacobian factor:
    R > 0 \implies R * \cosh(z) > 0 for all z \in \mathbb{R}. -/
theorem inverseMap_jacobian_factor_pos (R z : ℝ) (hR : 0 < R) : 0 < R * Real.cosh z := by
  have hcosh : 0 < Real.cosh z := Real.cosh_pos z
  exact mul_pos hR hcosh


/-! ============================================================================
    PROPERTY 2: Derivative Identity and Metric Scale Strict Positivity
    ============================================================================ -/

/-- Derivative identity: \frac{d}{dx} \operatorname{arsinh}(x) = (1 + x^2)^{-1/2}. -/
theorem arsinh_deriv_eq (x : ℝ) : deriv arsinh x = (√(1 + x ^ 2))⁻¹ :=
  (hasDerivAt_arsinh x).deriv

/-- Strict positivity of unscaled Riemannian metric factor:
    \forall x \in \mathbb{R}, \frac{1}{\sqrt{1 + x^2}} > 0. -/
theorem metric_scale_pos (x : ℝ) : 0 < (√(1 + x ^ 2))⁻¹ := by
  have h1 : 0 < 1 + x ^ 2 := by linarith [sq_nonneg x]
  have h2 : 0 < √(1 + x ^ 2) := sqrt_pos.mpr h1
  exact inv_pos.mpr h2

/-- Upper bound of unscaled Riemannian metric factor:
    \forall x \in \mathbb{R}, \frac{1}{\sqrt{1 + x^2}} \le 1. -/
theorem metric_scale_le_one (x : ℝ) : (√(1 + x ^ 2))⁻¹ ≤ 1 := by
  have h1 : 1 ≤ 1 + x ^ 2 := by linarith [sq_nonneg x]
  have h2 : 1 ≤ √(1 + x ^ 2) := by
    have h_sqrt : √1 ≤ √(1 + x ^ 2) := sqrt_le_sqrt h1
    rwa [sqrt_one] at h_sqrt
  exact inv_le_one_of_one_le₀ h2

/-- Scaled derivative identity via chain rule:
    \frac{d}{dx} \operatorname{arsinh}(x / R) = \frac{1}{R \sqrt{1 + (x/R)^2}}. -/
theorem hasDerivAt_forwardMap (R x : ℝ) :
    HasDerivAt (forwardMap R) ((R * √(1 + (x / R) ^ 2))⁻¹) x := by
  have h1 : HasDerivAt (fun y : ℝ => y / R) (R⁻¹) x := by
    simpa using (hasDerivAt_id x).div_const R
  have h2 : HasDerivAt arsinh (√(1 + (x / R) ^ 2))⁻¹ (x / R) :=
    hasDerivAt_arsinh (x / R)
  have hcomp := HasDerivAt.comp x h2 h1
  have hcomm : (√(1 + (x / R) ^ 2))⁻¹ * R⁻¹ = R⁻¹ * (√(1 + (x / R) ^ 2))⁻¹ := mul_comm _ _
  have hinv : R⁻¹ * (√(1 + (x / R) ^ 2))⁻¹ = (R * √(1 + (x / R) ^ 2))⁻¹ :=
    (mul_inv R (√(1 + (x / R) ^ 2))).symm
  rw [hcomm, hinv] at hcomp
  exact hcomp

/-- Strict positivity of scaled metric scale factor:
    R > 0 \implies \frac{1}{R \sqrt{1 + (x/R)^2}} > 0. -/
theorem forwardMap_metric_pos (R x : ℝ) (hR : 0 < R) :
    0 < (R * √(1 + (x / R) ^ 2))⁻¹ := by
  have h1 : 0 < 1 + (x / R) ^ 2 := by linarith [sq_nonneg (x / R)]
  have h2 : 0 < √(1 + (x / R) ^ 2) := sqrt_pos.mpr h1
  have hprod : 0 < R * √(1 + (x / R) ^ 2) := mul_pos hR h2
  exact inv_pos.mpr hprod


/-! ============================================================================
    PROPERTY 3: Pure Square-Root Attenuation & Exact Discrete Basin Boundary
    ============================================================================ -/

/-- Dimensionless gentle stiffness attenuation factor:
    \rho(\kappa) = (1 + \kappa)^{-1/2}. -/
def rho (kappa : ℝ) : ℝ :=
  (√(1 + kappa))⁻¹

/-- Strict positivity of gentle stiffness attenuation factor:
    \forall \kappa \ge 0, \rho(\kappa) > 0. -/
theorem rho_pos {kappa : ℝ} (h : 0 ≤ kappa) : 0 < rho kappa := by
  dsimp [rho]
  have h1 : 0 < 1 + kappa := by linarith
  have h2 : 0 < √(1 + kappa) := sqrt_pos.mpr h1
  exact inv_pos.mpr h2

/-- Non-expansion bound of gentle stiffness attenuation factor:
    \forall \kappa \ge 0, \rho(\kappa) \le 1. -/
theorem rho_le_one {kappa : ℝ} (h : 0 ≤ kappa) : rho kappa ≤ 1 := by
  dsimp [rho]
  have h1 : 1 ≤ 1 + kappa := by linarith
  have h2 : 1 ≤ √(1 + kappa) := by
    have h_sqrt : √1 ≤ √(1 + kappa) := sqrt_le_sqrt h1
    rwa [sqrt_one] at h_sqrt
  exact inv_le_one_of_one_le₀ h2

/-- Strict antimonotonicity: attenuation factor \rho(\kappa) strictly decreases as stiffness grows:
    \forall 0 \le \kappa_1 < \kappa_2, \rho(\kappa_2) < \rho(\kappa_1). -/
theorem rho_strictAntiOn : StrictAntiOn rho (Set.Ici 0) := by
  intro a ha b _ hab
  have ha_ge : 0 ≤ a := Set.mem_Ici.mp ha
  dsimp [rho]
  have ha_pos : 0 < 1 + a := by linarith
  have hab_add : 1 + a < 1 + b := by linarith
  have h_sqrt : √(1 + a) < √(1 + b) := (sqrt_lt_sqrt_iff ha_pos.le).mpr hab_add
  have h_sqrta_pos : 0 < √(1 + a) := sqrt_pos.mpr ha_pos
  have h_sqrtb_pos : 0 < √(1 + b) := h_sqrta_pos.trans h_sqrt
  exact (inv_lt_inv₀ h_sqrtb_pos h_sqrta_pos).mpr h_sqrt

/-- Squared drift displacement under pure square-root damping:
    \Delta^2 = (h g)^2 / (1 + h \gamma g). -/
def pureSqrtStepSq (h gamma g : ℝ) : ℝ :=
  (h * g) ^ 2 / (1 + h * gamma * g)

/-- Upper bound on pure square-root damped squared displacement:
    For h > 0, \gamma > 0, g > 0, we have (h g)^2 / (1 + h \gamma g) < h g / \gamma. -/
theorem pureSqrtStepSq_lt_div_gamma {h gamma g : ℝ}
    (hh : 0 < h) (hgamma : 0 < gamma) (hg : 0 < g) :
    pureSqrtStepSq h gamma g < h * g / gamma := by
  dsimp [pureSqrtStepSq]
  have h_hgg_pos : 0 < h * gamma * g := mul_pos (mul_pos hh hgamma) hg
  have h_denom_pos : 0 < 1 + h * gamma * g := by linarith
  rw [div_lt_div_iff₀ h_denom_pos hgamma]
  have h_hg_pos : 0 < h * g := mul_pos hh hg
  have h_diff : (1 + h * gamma * g) * (h * g) - (h * g) ^ 2 * gamma = h * g := by ring
  linarith

/-- Exact Discrete Basin of Attraction for Pure Square-Root Damping on Quartic Potential:
    For g(r) = r^3 (quartic gradient), if 0 < r < 4 * \gamma / h, then the squared
    damped displacement satisfies \Delta(r)^2 < (2 * r)^2, guaranteeing that the deterministic
    drift step cannot overshoot the opposite side of the ball of radius r. -/
theorem pureSqrt_quartic_discrete_basin_bound {h gamma r : ℝ}
    (hh : 0 < h) (hgamma : 0 < gamma) (hr : 0 < r) (hbasin : r < 4 * gamma / h) :
    pureSqrtStepSq h gamma (r ^ 3) < (2 * r) ^ 2 := by
  have hr3_pos : 0 < r ^ 3 := pow_pos hr 3
  have h1 : pureSqrtStepSq h gamma (r ^ 3) < h * r ^ 3 / gamma :=
    pureSqrtStepSq_lt_div_gamma hh hgamma hr3_pos
  have h_rh_lt : r * h < 4 * gamma := (lt_div_iff₀ hh).mp hbasin
  have h_hr_lt : h * r < 4 * gamma := by linarith
  have hr2_pos : 0 < r ^ 2 := sq_pos_of_pos hr
  have h_mul : (h * r) * r ^ 2 < (4 * gamma) * r ^ 2 :=
    mul_lt_mul_of_pos_right h_hr_lt hr2_pos
  have h_div : h * r ^ 3 / gamma < (2 * r) ^ 2 := by
    rw [div_lt_iff₀ hgamma]
    calc
      h * r ^ 3 = (h * r) * r ^ 2 := by ring
      _ < (4 * gamma) * r ^ 2 := h_mul
      _ = (2 * r) ^ 2 * gamma := by ring
  exact h1.trans h_div


/-! ============================================================================
    PROPERTY 4: Soft-Capped Gentle Attenuation Global Drift & Non-Overshoot
    ============================================================================ -/

/-- Squared soft-capped gentle attenuation factor:
    \rho_{\text{soft}}^2(\kappa, \kappa_{\max}) = (1 + \kappa + (\kappa / \kappa_{\max})^2)^{-1}. -/
def rhoSoftSq (kappa kappaMax : ℝ) : ℝ :=
  (1 + kappa + (kappa / kappaMax) ^ 2)⁻¹

/-- Soft-capped gentle attenuation factor:
    \rho_{\text{soft}}(\kappa, \kappa_{\max}) = (1 + \kappa + (\kappa / \kappa_{\max})^2)^{-1/2}. -/
def rhoSoft (kappa kappaMax : ℝ) : ℝ :=
  (√(1 + kappa + (kappa / kappaMax) ^ 2))⁻¹

/-- Strict positivity of soft-capped gentle attenuation for all \kappa \ge 0. -/
theorem rhoSoft_pos {kappa kappaMax : ℝ} (h : 0 ≤ kappa) : 0 < rhoSoft kappa kappaMax := by
  dsimp [rhoSoft]
  have hsq : 0 ≤ (kappa / kappaMax) ^ 2 := sq_nonneg (kappa / kappaMax)
  have h1 : 0 < 1 + kappa + (kappa / kappaMax) ^ 2 := by linarith
  have h2 : 0 < √(1 + kappa + (kappa / kappaMax) ^ 2) := sqrt_pos.mpr h1
  exact inv_pos.mpr h2

/-- Non-expansion bound of soft-capped gentle attenuation:
    \forall \kappa \ge 0, \rho_{\text{soft}}(\kappa, \kappa_{\max}) \le 1. -/
theorem rhoSoft_le_one {kappa kappaMax : ℝ} (h : 0 ≤ kappa) : rhoSoft kappa kappaMax ≤ 1 := by
  dsimp [rhoSoft]
  have hsq : 0 ≤ (kappa / kappaMax) ^ 2 := sq_nonneg (kappa / kappaMax)
  have h1 : 1 ≤ 1 + kappa + (kappa / kappaMax) ^ 2 := by linarith
  have h2 : 1 ≤ √(1 + kappa + (kappa / kappaMax) ^ 2) := by
    have h_sqrt : √1 ≤ √(1 + kappa + (kappa / kappaMax) ^ 2) := sqrt_le_sqrt h1
    rwa [sqrt_one] at h_sqrt
  exact inv_le_one_of_one_le₀ h2

/-- Global Dimensionless Drift Cap Theorem (Squared):
    For all \kappa \ge 0 and \kappa_{\max} > 0, the squared dimensionless effective drift
    satisfies \kappa^2 * \rho_{\text{soft}}^2(\kappa, \kappa_{\max}) < \kappa_{\max}^2. -/
theorem softCap_dimensionless_drift_sq_lt {kappa kappaMax : ℝ}
    (hk : 0 ≤ kappa) (hmax : 0 < kappaMax) :
    kappa ^ 2 * rhoSoftSq kappa kappaMax < kappaMax ^ 2 := by
  dsimp [rhoSoftSq]
  have hsq : 0 ≤ (kappa / kappaMax) ^ 2 := sq_nonneg (kappa / kappaMax)
  have h_denom_pos : 0 < 1 + kappa + (kappa / kappaMax) ^ 2 := by linarith
  rw [mul_inv_lt_iff₀ h_denom_pos]
  have hmax_ne : kappaMax ≠ 0 := ne_of_gt hmax
  have h_id : kappaMax ^ 2 * (kappa / kappaMax) ^ 2 = kappa ^ 2 := by
    rw [div_pow, mul_div_cancel₀ _ (pow_ne_zero 2 hmax_ne)]
  have hmax2_pos : 0 < kappaMax ^ 2 := sq_pos_of_pos hmax
  have h_kmax2_k : 0 ≤ kappaMax ^ 2 * kappa := mul_nonneg (le_of_lt hmax2_pos) hk
  calc
    kappa ^ 2 = kappaMax ^ 2 * (kappa / kappaMax) ^ 2 := h_id.symm
    _ < kappaMax ^ 2 * 1 + kappaMax ^ 2 * kappa + kappaMax ^ 2 * (kappa / kappaMax) ^ 2 := by linarith
    _ = kappaMax ^ 2 * (1 + kappa + (kappa / kappaMax) ^ 2) := by ring

/-- Global Dimensionless Drift Cap Theorem:
    For all \kappa \ge 0 and \kappa_{\max} > 0, the dimensionless effective drift
    satisfies \kappa * \rho_{\text{soft}}(\kappa, \kappa_{\max}) < \kappa_{\max}. -/
theorem softCap_dimensionless_drift_lt {kappa kappaMax : ℝ}
    (hk : 0 ≤ kappa) (hmax : 0 < kappaMax) :
    kappa * rhoSoft kappa kappaMax < kappaMax := by
  dsimp [rhoSoft]
  have hsq : 0 ≤ (kappa / kappaMax) ^ 2 := sq_nonneg (kappa / kappaMax)
  have h_denom_pos : 0 < 1 + kappa + (kappa / kappaMax) ^ 2 := by linarith
  have h_sqrt_pos : 0 < √(1 + kappa + (kappa / kappaMax) ^ 2) := sqrt_pos.mpr h_denom_pos
  rw [mul_inv_lt_iff₀ h_sqrt_pos]
  have h_sq_lt : kappa ^ 2 < (kappaMax * √(1 + kappa + (kappa / kappaMax) ^ 2)) ^ 2 := by
    rw [mul_pow, sq_sqrt (le_of_lt h_denom_pos)]
    have hmax_ne : kappaMax ≠ 0 := ne_of_gt hmax
    have h_id : kappaMax ^ 2 * (kappa / kappaMax) ^ 2 = kappa ^ 2 := by
      rw [div_pow, mul_div_cancel₀ _ (pow_ne_zero 2 hmax_ne)]
    have hmax2_pos : 0 < kappaMax ^ 2 := sq_pos_of_pos hmax
    have h_kmax2_k : 0 ≤ kappaMax ^ 2 * kappa := mul_nonneg (le_of_lt hmax2_pos) hk
    calc
      kappa ^ 2 = kappaMax ^ 2 * (kappa / kappaMax) ^ 2 := h_id.symm
      _ < kappaMax ^ 2 * (1 + kappa + (kappa / kappaMax) ^ 2) := by linarith
  have h_rhs_nonneg : 0 ≤ kappaMax * √(1 + kappa + (kappa / kappaMax) ^ 2) :=
    mul_nonneg (le_of_lt hmax) (le_of_lt h_sqrt_pos)
  exact lt_of_pow_lt_pow_left₀ 2 h_rhs_nonneg h_sq_lt

/-- Soft-capped Euclidean drift displacement:
    \Delta(h, \gamma, g, \kappa_{\max}) = h * g * \rho_{\text{soft}}(h * \gamma * g, \kappa_{\max}). -/
def softCappedDriftStep (h gamma g kappaMax : ℝ) : ℝ :=
  h * g * rhoSoft (h * gamma * g) kappaMax

/-- Global Euclidean Step Cap Theorem:
    For any h > 0, \gamma > 0, g \ge 0, and \kappa_{\max} > 0, the soft-capped drift step
    is strictly bounded above by \kappa_{\max} / \gamma, independent of gradient magnitude g! -/
theorem softCappedDriftStep_lt_bound {h gamma g kappaMax : ℝ}
    (hh : 0 < h) (hgamma : 0 < gamma) (hg : 0 ≤ g) (hmax : 0 < kappaMax) :
    softCappedDriftStep h gamma g kappaMax < kappaMax / gamma := by
  dsimp [softCappedDriftStep]
  rw [lt_div_iff₀ hgamma]
  have hk_nonneg : 0 ≤ h * gamma * g := mul_nonneg (mul_nonneg (le_of_lt hh) (le_of_lt hgamma)) hg
  have h_cap := softCap_dimensionless_drift_lt hk_nonneg hmax
  calc
    h * g * rhoSoft (h * gamma * g) kappaMax * gamma
      = (h * gamma * g) * rhoSoft (h * gamma * g) kappaMax := by ring
    _ < kappaMax := h_cap

/-- Global Non-Overshoot Contraction Theorem for Soft-Capped GentleMALA:
    For any initial radius r > \kappa_{\max} / \gamma and positive gradient g > 0,
    the post-drift radial distance r - \Delta satisfies:
    0 < r - \kappa_{\max} / \gamma < r - \Delta < r,
    proving that soft-capped GentleMALA strictly contracts toward the mode and never
    overshoots the origin from any arbitrarily large initialization r > \kappa_{\max} / \gamma. -/
theorem softCapped_global_contraction_no_overshoot {h gamma g kappaMax r : ℝ}
    (hh : 0 < h) (hgamma : 0 < gamma) (hg : 0 < g) (hmax : 0 < kappaMax)
    (hr : kappaMax / gamma < r) :
    0 < r - softCappedDriftStep h gamma g kappaMax ∧
    r - softCappedDriftStep h gamma g kappaMax < r := by
  have h_step_bound : softCappedDriftStep h gamma g kappaMax < kappaMax / gamma :=
    softCappedDriftStep_lt_bound hh hgamma (le_of_lt hg) hmax
  have hk_nonneg : 0 ≤ h * gamma * g :=
    mul_nonneg (mul_nonneg (le_of_lt hh) (le_of_lt hgamma)) (le_of_lt hg)
  have h_rho_pos : 0 < rhoSoft (h * gamma * g) kappaMax := rhoSoft_pos hk_nonneg
  have h_step_pos : 0 < softCappedDriftStep h gamma g kappaMax := by
    dsimp [softCappedDriftStep]
    exact mul_pos (mul_pos hh hg) h_rho_pos
  constructor
  · linarith
  · linarith


/-! ============================================================================
    PROPERTY 5: Metropolis-Hastings Kernel Reversibility (Detailed Balance)
    ============================================================================ -/

/-- Metropolis-Hastings acceptance probability:
    \alpha(x, y) = \min(1, (\pi(y) q(y, x)) / (\pi(x) q(x, y))). -/
def mhAccept (pi_x pi_y q_xy q_yx : ℝ) : ℝ :=
  min 1 ((pi_y * q_yx) / (pi_x * q_xy))

/-- Metropolis-Hastings unnormalized transition flux from state x to state y:
    \Phi(x, y) = \pi(x) q(x, y) \alpha(x, y). -/
def mhFlux (pi_x pi_y q_xy q_yx : ℝ) : ℝ :=
  pi_x * q_xy * mhAccept pi_x pi_y q_xy q_yx

/-- Metropolis-Hastings acceptance probability is a valid probability in [0, 1]. -/
theorem mhAccept_bounds {pi_x pi_y q_xy q_yx : ℝ}
    (hpx : 0 < pi_x) (hpy : 0 < pi_y) (hqxy : 0 < q_xy) (hqyx : 0 < q_yx) :
    0 ≤ mhAccept pi_x pi_y q_xy q_yx ∧ mhAccept pi_x pi_y q_xy q_yx ≤ 1 := by
  dsimp [mhAccept]
  have h_num : 0 < pi_y * q_yx := mul_pos hpy hqyx
  have h_den : 0 < pi_x * q_xy := mul_pos hpx hqxy
  have h_div : 0 < (pi_y * q_yx) / (pi_x * q_xy) := div_pos h_num h_den
  constructor
  · exact le_min (by norm_num) (le_of_lt h_div)
  · exact min_le_left 1 _

/-- Exact Metropolis-Hastings Detailed Balance (Reversibility) Theorem:
    For any positive target density \pi_x, \pi_y > 0 and positive proposal density q_{xy}, q_{yx} > 0,
    \pi(x) q(x, y) \min(1, \frac{\pi(y) q(y, x)}{\pi(x) q(x, y)}) =
    \pi(y) q(y, x) \min(1, \frac{\pi(x) q(x, y)}{\pi(y) q(y, x)}). -/
theorem mh_detailed_balance {pi_x pi_y q_xy q_yx : ℝ}
    (hpx : 0 < pi_x) (hpy : 0 < pi_y) (hqxy : 0 < q_xy) (hqyx : 0 < q_yx) :
    mhFlux pi_x pi_y q_xy q_yx = mhFlux pi_y pi_x q_yx q_xy := by
  dsimp [mhFlux, mhAccept]
  have h_xy_pos : 0 < pi_x * q_xy := mul_pos hpx hqxy
  have h_yx_pos : 0 < pi_y * q_yx := mul_pos hpy hqyx
  have h_xy_ne : pi_x * q_xy ≠ 0 := ne_of_gt h_xy_pos
  have h_yx_ne : pi_y * q_yx ≠ 0 := ne_of_gt h_yx_pos
  rcases le_total (pi_x * q_xy) (pi_y * q_yx) with hle | hle
  · have h1 : 1 ≤ (pi_y * q_yx) / (pi_x * q_xy) := (one_le_div₀ h_xy_pos).mpr hle
    have h2 : (pi_x * q_xy) / (pi_y * q_yx) ≤ 1 := (div_le_one₀ h_yx_pos).mpr hle
    rw [min_eq_left h1, min_eq_right h2, mul_one, mul_div_cancel₀ _ h_yx_ne]
  · have h1 : (pi_y * q_yx) / (pi_x * q_xy) ≤ 1 := (div_le_one₀ h_xy_pos).mpr hle
    have h2 : 1 ≤ (pi_x * q_xy) / (pi_y * q_yx) := (one_le_div₀ h_yx_pos).mpr hle
    rw [min_eq_right h1, min_eq_left h2, mul_div_cancel₀ _ h_xy_ne, mul_one]


/-! ============================================================================
    §7. Applied Heavy-Tailed & Shrinkage Posterior Score Bounds
    ============================================================================ -/

/-- Student-t negative log-kernel for residual r ∈ ℝ and degrees of freedom ν > 0:
    U_t(ν, r) = ((ν + 1) / 2) * log(1 + r^2 / ν). -/
def studentTNegLogKernel (ν r : ℝ) : ℝ :=
  ((ν + 1) / 2) * Real.log (1 + r ^ 2 / ν)

/-- Analytical score (derivative of studentTNegLogKernel with respect to residual r):
    s_t(ν, r) = ((ν + 1) * r) / (ν + r^2). -/
def studentTScore (ν r : ℝ) : ℝ :=
  ((ν + 1) * r) / (ν + r ^ 2)

/-- Cauchy / Horseshoe local log-prior shrinkage penalty:
    P_C(τ, λ) = log(1 + (λ / τ)^2). -/
def cauchyShrinkagePenalty (τ lam : ℝ) : ℝ :=
  Real.log (1 + (lam / τ) ^ 2)

/-- The Student-t negative log-kernel is globally non-negative for all ν > 0 and r ∈ ℝ. -/
theorem studentTNegLogKernel_nonneg {ν : ℝ} (hν : 0 < ν) (r : ℝ) :
    0 ≤ studentTNegLogKernel ν r := by
  dsimp [studentTNegLogKernel]
  have h1 : 0 ≤ (ν + 1) / 2 := by linarith
  have h_div : 0 ≤ r ^ 2 / ν := div_nonneg (sq_nonneg r) (le_of_lt hν)
  have h_arg : 1 ≤ 1 + r ^ 2 / ν := by linarith
  have h_log : 0 ≤ Real.log (1 + r ^ 2 / ν) := Real.log_nonneg h_arg
  exact mul_nonneg h1 h_log

/-- Exact derivative identity: the derivative of studentTNegLogKernel ν at r is studentTScore ν r. -/
theorem hasDerivAt_studentTNegLogKernel {ν : ℝ} (hν : 0 < ν) (r : ℝ) :
    HasDerivAt (studentTNegLogKernel ν) (studentTScore ν r) r := by
  have h_sq : HasDerivAt (fun x : ℝ => x ^ 2) (2 * r) r := by
    have h_mul : HasDerivAt (fun x : ℝ => x * x) (1 * r + r * 1) r :=
      (hasDerivAt_id' r).mul (hasDerivAt_id' r)
    have h_fn : (fun x : ℝ => x * x) = (fun x : ℝ => x ^ 2) := by
      funext x
      ring
    have h_der : 1 * r + r * 1 = 2 * r := by ring
    rwa [h_fn, h_der] at h_mul
  have h_div : HasDerivAt (fun x : ℝ => x ^ 2 / ν) ((2 * r) / ν) r :=
    h_sq.div_const ν
  have h_inner : HasDerivAt (fun x : ℝ => 1 + x ^ 2 / ν) ((2 * r) / ν) r :=
    h_div.const_add 1
  have h_arg_pos : 0 < 1 + r ^ 2 / ν := by
    have h_nonneg : 0 ≤ r ^ 2 / ν := div_nonneg (sq_nonneg r) (le_of_lt hν)
    linarith
  have h_log : HasDerivAt (fun x : ℝ => Real.log (1 + x ^ 2 / ν))
      (((2 * r) / ν) / (1 + r ^ 2 / ν)) r :=
    h_inner.log (ne_of_gt h_arg_pos)
  have h_scaled : HasDerivAt (studentTNegLogKernel ν)
      (((ν + 1) / 2) * (((2 * r) / ν) / (1 + r ^ 2 / ν))) r :=
    h_log.const_mul ((ν + 1) / 2)
  have h_denom_pos : 0 < ν + r ^ 2 := by linarith [sq_nonneg r]
  have h_simp : ((ν + 1) / 2) * (((2 * r) / ν) / (1 + r ^ 2 / ν)) = studentTScore ν r := by
    dsimp [studentTScore]
    have hν_ne : ν ≠ 0 := ne_of_gt hν
    have h_denom_ne : ν + r ^ 2 ≠ 0 := ne_of_gt h_denom_pos
    have h_arg_ne : 1 + r ^ 2 / ν ≠ 0 := ne_of_gt h_arg_pos
    field_simp [hν_ne, h_denom_ne, h_arg_ne]
  rwa [h_simp] at h_scaled

/-- Global virial score bound: |studentTScore ν r * r| ≤ ν + 1 for all r ∈ ℝ and ν > 0,
    proving outlier robustness even as |r| → ∞. -/
theorem studentTScore_mul_le {ν : ℝ} (hν : 0 < ν) (r : ℝ) :
    |studentTScore ν r * r| ≤ ν + 1 := by
  dsimp [studentTScore]
  have h_denom_pos : 0 < ν + r ^ 2 := by linarith [sq_nonneg r]
  have h_num_nonneg : 0 ≤ (ν + 1) * r * r := by
    have h1 : 0 ≤ ν + 1 := by linarith
    have h2 : (ν + 1) * r * r = (ν + 1) * r ^ 2 := by ring
    rw [h2]
    exact mul_nonneg h1 (sq_nonneg r)
  have h_prod_nonneg : 0 ≤ (ν + 1) * r / (ν + r ^ 2) * r := by
    have h_eq : (ν + 1) * r / (ν + r ^ 2) * r = ((ν + 1) * r * r) / (ν + r ^ 2) := by ring
    rw [h_eq]
    exact div_nonneg h_num_nonneg (le_of_lt h_denom_pos)
  rw [abs_of_nonneg h_prod_nonneg]
  have h_eq : (ν + 1) * r / (ν + r ^ 2) * r = ((ν + 1) * r ^ 2) / (ν + r ^ 2) := by ring
  rw [h_eq, div_le_iff₀ h_denom_pos]
  have h_pos_term : 0 < (ν + 1) * ν := mul_pos (by linarith) hν
  nlinarith

/-- The Cauchy / Horseshoe local log-prior shrinkage penalty is globally non-negative. -/
theorem cauchyShrinkagePenalty_nonneg (τ lam : ℝ) :
    0 ≤ cauchyShrinkagePenalty τ lam := by
  dsimp [cauchyShrinkagePenalty]
  have h_sq : 0 ≤ (lam / τ) ^ 2 := sq_nonneg (lam / τ)
  have h_arg : 1 ≤ 1 + (lam / τ) ^ 2 := by linarith
  exact Real.log_nonneg h_arg

/-- The Student-t score vanishes at zero residual r = 0. -/
theorem studentTScore_zero (ν : ℝ) :
    studentTScore ν 0 = 0 := by
  dsimp [studentTScore]
  ring


/-! ============================================================================
    §8. Pure-JAX 3-Phase Scale & Location Calibration Bounds
    ============================================================================ -/

/-- Clipped normal-consistent coordinate scale calibration map used in Phase 1 warmup:
    R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c) = \min(R_{\max}, \max(R_{\min}, c \cdot \sigma)). -/
def calibratedRadius (σ Rmin Rmax c : ℝ) : ℝ :=
  min Rmax (max Rmin (c * σ))

/-- Lower bound of calibrated coordinate radius:
    Assuming 0 < R_{\min} and R_{\min} \le R_{\max}, R_{\min} \le R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c). -/
theorem calibratedRadius_ge_min (σ Rmin Rmax c : ℝ) (_hmin_pos : 0 < Rmin) (hle : Rmin ≤ Rmax) :
    Rmin ≤ calibratedRadius σ Rmin Rmax c := by
  dsimp [calibratedRadius]
  exact le_min hle (le_max_left Rmin (c * σ))

/-- Strict positivity of calibrated coordinate radius:
    Assuming 0 < R_{\min} and R_{\min} \le R_{\max}, 0 < R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c)
    for any raw dispersion estimate \sigma \in \mathbb{R} (even \sigma \le 0). -/
theorem calibratedRadius_pos (σ Rmin Rmax c : ℝ) (hmin_pos : 0 < Rmin) (hle : Rmin ≤ Rmax) :
    0 < calibratedRadius σ Rmin Rmax c :=
  lt_of_lt_of_le hmin_pos (calibratedRadius_ge_min σ Rmin Rmax c hmin_pos hle)

/-- Upper bound of calibrated coordinate radius:
    Assuming R_{\min} \le R_{\max}, R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c) \le R_{\max}. -/
theorem calibratedRadius_le_max (σ Rmin Rmax c : ℝ) (_hle : Rmin ≤ Rmax) :
    calibratedRadius σ Rmin Rmax c ≤ Rmax := by
  dsimp [calibratedRadius]
  exact min_le_left Rmax (max Rmin (c * σ))

/-- Monotonicity of calibrated coordinate radius with respect to raw dispersion \sigma:
    For c \ge 0, \sigma_1 \le \sigma_2 \implies R_{\text{cal}}(\sigma_1) \le R_{\text{cal}}(\sigma_2). -/
theorem calibratedRadius_mono {σ₁ σ₂ Rmin Rmax c : ℝ} (hc : 0 ≤ c) (hσ : σ₁ ≤ σ₂) :
    calibratedRadius σ₁ Rmin Rmax c ≤ calibratedRadius σ₂ Rmin Rmax c := by
  dsimp [calibratedRadius]
  have hmul : c * σ₁ ≤ c * σ₂ := mul_le_mul_of_nonneg_left hσ hc
  have hmax : max Rmin (c * σ₁) ≤ max Rmin (c * σ₂) := max_le_max_left Rmin hmul
  exact min_le_min_left Rmax hmax

/-- Exact roundtrip invertibility (forward then inverse) of the Phase 1 calibrated
    shift-equivariant diffeomorphism for any median \mu \in \mathbb{R} and dispersion \sigma \in \mathbb{R}. -/
theorem calibrated_shifted_roundtrip_forward (μ σ Rmin Rmax c x : ℝ)
    (hmin_pos : 0 < Rmin) (hle : Rmin ≤ Rmax) :
    shiftedInverseMap μ (calibratedRadius σ Rmin Rmax c)
      (shiftedForwardMap μ (calibratedRadius σ Rmin Rmax c) x) = x := by
  have hpos : 0 < calibratedRadius σ Rmin Rmax c :=
    calibratedRadius_pos σ Rmin Rmax c hmin_pos hle
  exact shifted_invertibility_roundtrip_forward μ (calibratedRadius σ Rmin Rmax c) x (ne_of_gt hpos)

/-- Exact roundtrip invertibility (inverse then forward) of the Phase 1 calibrated
    shift-equivariant diffeomorphism for any median \mu \in \mathbb{R} and dispersion \sigma \in \mathbb{R}. -/
theorem calibrated_shifted_roundtrip_inverse (μ σ Rmin Rmax c z : ℝ)
    (hmin_pos : 0 < Rmin) (hle : Rmin ≤ Rmax) :
    shiftedForwardMap μ (calibratedRadius σ Rmin Rmax c)
      (shiftedInverseMap μ (calibratedRadius σ Rmin Rmax c) z) = z := by
  have hpos : 0 < calibratedRadius σ Rmin Rmax c :=
    calibratedRadius_pos σ Rmin Rmax c hmin_pos hle
  exact shifted_invertibility_roundtrip_inverse μ (calibratedRadius σ Rmin Rmax c) z (ne_of_gt hpos)


/-! ============================================================================
    §9. Tail Envelopes of Arcsinh Pullbacks of Cauchy Marginals
    ============================================================================ -/

open Filter Topology

/-- Unnormalised Cauchy kernel `(1 + x²)⁻¹`; the standard Cauchy density is `cauchyKernel x / π`. -/
def cauchyKernel (x : ℝ) : ℝ :=
  (1 + x ^ 2)⁻¹

/-- `π` times the density of `z = arsinh((x - μ) / R)` when `x ~ Cauchy(m, s)`, written in the
    dimensionless calibration errors `δ = (μ - m) / s` and `r = R / s`:
    `π p_z(z) = r cosh z / (1 + (δ + r sinh z)²)`. -/
def arcsinhCauchyPullback (δ r z : ℝ) : ℝ :=
  r * cosh z * cauchyKernel (δ + r * sinh z)

/-- Hyperbolic Pythagoras in the form used for Jacobians: `1 + sinh² z = cosh² z`. -/
theorem one_add_sinh_sq (z : ℝ) : 1 + sinh z ^ 2 = cosh z ^ 2 := by
  rw [cosh_sq]; ring

/-- Exact pullback for a perfectly calibrated transform (`μ = m`, `R = s`): the arcsinh pullback
    of a Cauchy marginal is the hyperbolic secant law, `π p_z(z) = sech z`. -/
theorem arcsinhCauchyPullback_eq_sech (z : ℝ) :
    arcsinhCauchyPullback 0 1 z = (cosh z)⁻¹ := by
  unfold arcsinhCauchyPullback cauchyKernel
  have hc : cosh z ≠ 0 := (cosh_pos z).ne'
  rw [zero_add, one_mul, one_mul, one_add_sinh_sq]
  field_simp

/-- Lower exponential envelope of `cosh`: `exp |z| / 2 ≤ cosh z`. -/
theorem half_exp_abs_le_cosh (z : ℝ) : exp |z| / 2 ≤ cosh z := by
  rw [← cosh_abs, cosh_eq]
  have := exp_pos (-|z|)
  linarith

/-- Upper exponential envelope of `cosh`: `cosh z ≤ exp |z|`. -/
theorem cosh_le_exp_abs (z : ℝ) : cosh z ≤ exp |z| := by
  rw [← cosh_abs, cosh_eq]
  have : exp (-|z|) ≤ exp |z| := exp_le_exp.mpr (by linarith [abs_nonneg z])
  linarith

/-- Two-sided Laplace envelope of the hyperbolic secant:
    `exp(-|z|) ≤ sech z ≤ 2 exp(-|z|)` for every `z ∈ ℝ`. -/
theorem sech_envelope (z : ℝ) :
    exp (-|z|) ≤ (cosh z)⁻¹ ∧ (cosh z)⁻¹ ≤ 2 * exp (-|z|) := by
  have hc := cosh_pos z
  have he := exp_pos |z|
  rw [exp_neg]
  constructor
  · exact inv_anti₀ hc (cosh_le_exp_abs z)
  · have h := half_exp_abs_le_cosh z
    rw [← div_eq_mul_inv, le_div_iff₀ he, inv_mul_le_iff₀ hc]
    linarith

/-- Calibration-error kernel bound (upper half): for all `δ, r, y`,
    `r² (1 + y²) ≤ (r² + 1 + δ²) (1 + (δ + r y)²)`.
    The gap is the sum of squares `(1 + δ w)² + r² w²` with `w = δ + r y`. -/
theorem calibration_kernel_upper (δ r y : ℝ) :
    r ^ 2 * (1 + y ^ 2) ≤ (r ^ 2 + 1 + δ ^ 2) * (1 + (δ + r * y) ^ 2) := by
  nlinarith [sq_nonneg (1 + δ * (δ + r * y)), sq_nonneg (r * (δ + r * y))]

/-- Calibration-error kernel bound (lower half): for all `δ, r, y`,
    `1 + (δ + r y)² ≤ (r² + 1 + δ²) (1 + y²)`.
    The gap is the sum of squares `(r - δ y)² + y²`. -/
theorem calibration_kernel_lower (δ r y : ℝ) :
    1 + (δ + r * y) ^ 2 ≤ (r ^ 2 + 1 + δ ^ 2) * (1 + y ^ 2) := by
  nlinarith [sq_nonneg (r - δ * y), sq_nonneg y]

/-- Calibration robustness of the arcsinh pullback of a Cauchy marginal: for any location error
    `δ` and scale ratio `r > 0`, the pullback stays within the factor
    `K = (r² + 1 + δ²) / r` of the hyperbolic secant, `sech z / K ≤ π p_z(z) ≤ K sech z`. -/
theorem arcsinhCauchyPullback_envelope {r : ℝ} (hr : 0 < r) (δ z : ℝ) :
    (cosh z)⁻¹ * (r / (r ^ 2 + 1 + δ ^ 2)) ≤ arcsinhCauchyPullback δ r z ∧
      arcsinhCauchyPullback δ r z ≤ (cosh z)⁻¹ * ((r ^ 2 + 1 + δ ^ 2) / r) := by
  have hc := cosh_pos z
  have hS : 0 < r ^ 2 + 1 + δ ^ 2 := by positivity
  have hw : 0 < 1 + (δ + r * sinh z) ^ 2 := by positivity
  have hcs : cosh z * cosh z = 1 + sinh z ^ 2 := by rw [one_add_sinh_sq]; ring
  have hP : arcsinhCauchyPullback δ r z = r * cosh z / (1 + (δ + r * sinh z) ^ 2) := by
    unfold arcsinhCauchyPullback cauchyKernel; rw [div_eq_mul_inv]
  have hlo := calibration_kernel_lower δ r (sinh z)
  have hhi := calibration_kernel_upper δ r (sinh z)
  rw [hP]
  constructor
  · rw [inv_mul_eq_div, div_div, div_le_div_iff₀ (by positivity) hw]
    have : r * (1 + (δ + r * sinh z) ^ 2) ≤ r * ((r ^ 2 + 1 + δ ^ 2) * (1 + sinh z ^ 2)) :=
      mul_le_mul_of_nonneg_left hlo hr.le
    have e : r * cosh z * ((r ^ 2 + 1 + δ ^ 2) * cosh z)
        = r * ((r ^ 2 + 1 + δ ^ 2) * (1 + sinh z ^ 2)) := by rw [← hcs]; ring
    linarith
  · rw [inv_mul_eq_div, div_div, div_le_div_iff₀ hw (by positivity)]
    have e : r * cosh z * (cosh z * r) = r ^ 2 * (1 + sinh z ^ 2) := by rw [← hcs]; ring
    linarith

/-- Non-asymptotic Laplace envelope with tail rate exactly `1` (the Cauchy tail index) for every
    calibration `(δ, r)`:
    `exp(-|z|) r / (r² + 1 + δ²) ≤ π p_z(z) ≤ 2 exp(-|z|) (r² + 1 + δ²) / r`. -/
theorem arcsinhCauchyPullback_exp_envelope {r : ℝ} (hr : 0 < r) (δ z : ℝ) :
    exp (-|z|) * (r / (r ^ 2 + 1 + δ ^ 2)) ≤ arcsinhCauchyPullback δ r z ∧
      arcsinhCauchyPullback δ r z ≤ 2 * exp (-|z|) * ((r ^ 2 + 1 + δ ^ 2) / r) := by
  have hS : 0 < r ^ 2 + 1 + δ ^ 2 := by positivity
  obtain ⟨hlo, hhi⟩ := arcsinhCauchyPullback_envelope hr δ z
  obtain ⟨hslo, hshi⟩ := sech_envelope z
  constructor
  · exact le_trans (mul_le_mul_of_nonneg_right hslo (by positivity)) hlo
  · exact le_trans hhi (mul_le_mul_of_nonneg_right hshi (by positivity))


/-! ============================================================================
    §10. Tukey h-Map: Derivative, Log-Jacobian, Bijectivity and Gaussian Envelopes
    ============================================================================ -/

/-- Tukey h-map used as a coordinate pullback: `x = μ + R z exp(z² / (2c))`
    (Tukey's h-transform with `h = 1 / c`; `c = ∞` is the affine map). -/
def hMap (μ R c z : ℝ) : ℝ :=
  μ + R * z * exp (z ^ 2 / (2 * c))

/-- Jacobian of the h-map: `dx/dz = R exp(z² / (2c)) (1 + z² / c)`. -/
def hMapDeriv (R c z : ℝ) : ℝ :=
  R * exp (z ^ 2 / (2 * c)) * (1 + z ^ 2 / c)

/-- Derivative identity for the h-map: `d/dz [μ + R z exp(z²/(2c))] = R exp(z²/(2c)) (1 + z²/c)`. -/
theorem hasDerivAt_hMap (μ R : ℝ) {c : ℝ} (hc : c ≠ 0) (z : ℝ) :
    HasDerivAt (hMap μ R c) (hMapDeriv R c z) z := by
  have h1 : HasDerivAt (fun w : ℝ => w ^ 2 / (2 * c)) (2 * z / (2 * c)) z := by
    simpa using (hasDerivAt_pow 2 z).div_const (2 * c)
  have h2 : HasDerivAt (fun w : ℝ => w * exp (w ^ 2 / (2 * c)))
      (1 * exp (z ^ 2 / (2 * c)) + z * (exp (z ^ 2 / (2 * c)) * (2 * z / (2 * c)))) z :=
    (hasDerivAt_id' z).mul h1.exp
  have h3 := (h2.const_mul R).const_add μ
  have hfun : hMap μ R c = fun w => μ + R * (w * exp (w ^ 2 / (2 * c))) := by
    funext w; unfold hMap; ring
  have hd : hMapDeriv R c z
      = R * (1 * exp (z ^ 2 / (2 * c)) + z * (exp (z ^ 2 / (2 * c)) * (2 * z / (2 * c)))) := by
    unfold hMapDeriv; field_simp
  rw [hfun, hd]
  exact h3

/-- Strict positivity of the h-map Jacobian for `R > 0`, `c > 0`. -/
theorem hMapDeriv_pos {R c : ℝ} (hR : 0 < R) (hc : 0 < c) (z : ℝ) : 0 < hMapDeriv R c z := by
  have h : 0 ≤ z ^ 2 / c := div_nonneg (sq_nonneg z) hc.le
  unfold hMapDeriv
  exact mul_pos (mul_pos hR (exp_pos _)) (by linarith)

/-- Log-determinant of the h-map:
    `log (dx/dz) = log R + z² / (2c) + log(1 + z² / c)`. -/
theorem log_hMapDeriv {R c : ℝ} (hR : 0 < R) (hc : 0 < c) (z : ℝ) :
    log (hMapDeriv R c z) = log R + z ^ 2 / (2 * c) + log (1 + z ^ 2 / c) := by
  have h : 0 < 1 + z ^ 2 / c := by have := div_nonneg (sq_nonneg z) hc.le; linarith
  unfold hMapDeriv
  rw [log_mul (mul_pos hR (exp_pos _)).ne' h.ne', log_mul hR.ne' (exp_pos _).ne', log_exp]

/-- The h-map is strictly increasing for `R > 0`, `c > 0`. -/
theorem hMap_strictMono (μ : ℝ) {R c : ℝ} (hR : 0 < R) (hc : 0 < c) :
    StrictMono (hMap μ R c) :=
  strictMono_of_deriv_pos fun z => by
    rw [(hasDerivAt_hMap μ R hc.ne' z).deriv]; exact hMapDeriv_pos hR hc z

/-- The h-map is onto `ℝ` for `R > 0`, `c > 0` (continuity plus `±∞` limits). -/
theorem hMap_surjective (μ : ℝ) {R c : ℝ} (hR : 0 < R) (hc : 0 < c) :
    Function.Surjective (hMap μ R c) := by
  have hcont : Continuous (hMap μ R c) := by unfold hMap; fun_prop
  apply hcont.surjective
  · have hlin : Tendsto (fun z : ℝ => μ + R * z) atTop atTop :=
      tendsto_atTop_add_const_left _ _ (tendsto_id.const_mul_atTop hR)
    refine tendsto_atTop_mono' _ ?_ hlin
    filter_upwards [eventually_ge_atTop 0] with z hz
    have he : 1 ≤ exp (z ^ 2 / (2 * c)) := one_le_exp (by positivity)
    unfold hMap
    nlinarith [mul_nonneg hR.le hz]
  · have hlin : Tendsto (fun z : ℝ => μ + R * z) atBot atBot :=
      tendsto_atBot_add_const_left _ _ (tendsto_id.const_mul_atBot hR)
    refine tendsto_atBot_mono' _ ?_ hlin
    filter_upwards [eventually_le_atBot 0] with z hz
    have he : 1 ≤ exp (z ^ 2 / (2 * c)) := one_le_exp (by positivity)
    unfold hMap
    nlinarith [mul_nonpos_iff.mpr (Or.inl ⟨hR.le, hz⟩)]

/-- The h-map is a bijection of `ℝ` for `R > 0`, `c > 0`, so the pullback is exact. -/
theorem hMap_bijective (μ : ℝ) {R c : ℝ} (hR : 0 < R) (hc : 0 < c) :
    Function.Bijective (hMap μ R c) :=
  ⟨(hMap_strictMono μ hR hc).injective, hMap_surjective μ hR hc⟩

/-- Rescaling invariance of the h-map: `hMap μ R c (λ w) = hMap μ (R λ) (c / λ²) w` for `λ ≠ 0`.
    A linear change of the latent coordinate (e.g. an adapted diagonal mass matrix) maps
    `(R, c)` to `(R λ, c / λ²)`, so only `R² c` is identifiable. -/
theorem hMap_rescale (μ R c : ℝ) {l : ℝ} (hl : l ≠ 0) (w : ℝ) :
    hMap μ R c (l * w) = hMap μ (R * l) (c / l ^ 2) w := by
  unfold hMap
  have harg : (l * w) ^ 2 / (2 * c) = w ^ 2 / (2 * (c / l ^ 2)) := by
    rcases eq_or_ne c 0 with h0 | h0
    · simp [h0]
    · field_simp
  rw [harg]; ring

/-- The invariant of `hMap_rescale`: `(R λ)² (c / λ²) = R² c`. -/
theorem hMap_rescale_invariant (R c : ℝ) {l : ℝ} (hl : l ≠ 0) :
    (R * l) ^ 2 * (c / l ^ 2) = R ^ 2 * c := by
  field_simp

/-- `π` times the density of `z` when `x = r z exp(z² / (2c)) ~ Cauchy(0, 1)`:
    `π p_z(z) = (dx/dz) / (1 + x²)`. -/
def hMapCauchyPullback (r c z : ℝ) : ℝ :=
  hMapDeriv r c z * cauchyKernel (hMap 0 r c z)

/-- Closed form of the h-map Cauchy pullback with `E = exp(z² / (2c))`:
    `π p_z(z) = r E (1 + z² / c) / (1 + r² z² E²)`. -/
theorem hMapCauchyPullback_eq (r c z : ℝ) :
    hMapCauchyPullback r c z = r * exp (z ^ 2 / (2 * c)) * (1 + z ^ 2 / c)
      / (1 + r ^ 2 * z ^ 2 * (exp (z ^ 2 / (2 * c)) * exp (z ^ 2 / (2 * c)))) := by
  unfold hMapCauchyPullback hMapDeriv cauchyKernel hMap
  rw [div_eq_mul_inv]
  ring_nf

/-- Gaussian lower envelope (tail variance `c`) of the h-map Cauchy pullback, for all `z ∈ ℝ`:
    `r m exp(-z² / (2c)) ≤ π p_z(z)` whenever `m ≤ 1` and `m r² c ≤ 1`
    (e.g. `m = min 1 (1 / (r² c))`). -/
theorem hMapCauchyPullback_gauss_lower {r c m : ℝ} (hr : 0 < r) (hc : 0 < c)
    (hm1 : m ≤ 1) (hmc : m * r ^ 2 * c ≤ 1) (z : ℝ) :
    r * m * exp (-(z ^ 2 / (2 * c))) ≤ hMapCauchyPullback r c z := by
  set E := exp (z ^ 2 / (2 * c)) with hE
  set q := z ^ 2 / c with hqdef
  have hEpos : 0 < E := exp_pos _
  have hE1 : 1 ≤ E := one_le_exp (by positivity)
  have hq : 0 ≤ q := by positivity
  have hzq : z ^ 2 = c * q := by rw [hqdef]; field_simp
  rw [hMapCauchyPullback_eq, exp_neg, ← hE, ← hqdef, ← div_eq_mul_inv,
    div_le_div_iff₀ hEpos (by positivity)]
  have hF : 1 ≤ E * E := by nlinarith
  have key : m * (1 + r ^ 2 * z ^ 2 * (E * E)) ≤ E * E * (1 + q) := by
    have h1 : m * r ^ 2 * z ^ 2 ≤ q := by
      rw [hzq]; nlinarith [mul_le_mul_of_nonneg_right hmc hq]
    nlinarith [mul_le_mul_of_nonneg_right h1 (by positivity : (0:ℝ) ≤ E * E)]
  nlinarith [mul_le_mul_of_nonneg_left key hr.le]

/-- Gaussian upper envelope (tail variance `c`) of the h-map Cauchy pullback, for all `z ∈ ℝ`:
    `π p_z(z) ≤ (r M / ρ) exp(-z² / (2c))` whenever `1 ≤ M`, `1 ≤ M r² c`, `0 < ρ`,
    `ρ e ≤ 1` and `ρ (1 + r² c) ≤ r² c`
    (e.g. `M = max 1 (1 / (r² c))`, `ρ = min (exp (-1)) (r² c / (1 + r² c))`). -/
theorem hMapCauchyPullback_gauss_upper {r c M ρ : ℝ} (hr : 0 < r) (hc : 0 < c)
    (hM1 : 1 ≤ M) (hMc : 1 ≤ M * r ^ 2 * c) (hρ : 0 < ρ) (hρe : ρ * exp 1 ≤ 1)
    (hρc : ρ * (1 + r ^ 2 * c) ≤ r ^ 2 * c) (z : ℝ) :
    hMapCauchyPullback r c z ≤ r * M / ρ * exp (-(z ^ 2 / (2 * c))) := by
  set E := exp (z ^ 2 / (2 * c)) with hE
  set q := z ^ 2 / c with hqdef
  have hEpos : 0 < E := exp_pos _
  have hq : 0 ≤ q := by positivity
  have hzq : z ^ 2 = c * q := by rw [hqdef]; field_simp
  have hF : E * E = exp q := by rw [hE, ← exp_add]; congr 1; rw [hqdef]; field_simp; ring
  have hρ1 : ρ ≤ 1 := by nlinarith [add_one_le_exp (1:ℝ)]
  have hy : r ^ 2 * z ^ 2 = r ^ 2 * c * q := by rw [hzq]; ring
  have core : ρ * (E * E) * (1 + r ^ 2 * z ^ 2) ≤ 1 + r ^ 2 * z ^ 2 * (E * E) := by
    have hy0 : 0 ≤ r ^ 2 * z ^ 2 := by positivity
    rcases le_total q 1 with hq1 | hq1
    · have hFe : E * E ≤ exp 1 := by rw [hF]; exact exp_le_exp.mpr hq1
      have hρF : ρ * (E * E) ≤ 1 := le_trans (mul_le_mul_of_nonneg_left hFe hρ.le) hρe
      nlinarith [mul_le_mul_of_nonneg_right hρ1 (mul_nonneg hy0 (mul_self_nonneg E))]
    · have hyc : r ^ 2 * c ≤ r ^ 2 * z ^ 2 := by rw [hy]; nlinarith [sq_nonneg r]
      have hρy : ρ * (1 + r ^ 2 * z ^ 2) ≤ r ^ 2 * z ^ 2 := by nlinarith
      nlinarith [mul_le_mul_of_nonneg_left hρy (mul_self_nonneg E)]
  have hM : 1 + q ≤ M * (1 + r ^ 2 * z ^ 2) := by
    rw [hy]; nlinarith [mul_le_mul_of_nonneg_right hMc hq]
  rw [hMapCauchyPullback_eq, exp_neg, ← hE, ← hqdef, ← div_eq_mul_inv, div_div,
    div_le_div_iff₀ (by positivity) (by positivity)]
  have h1 : ρ * (E * E) * (1 + q) ≤ M * (1 + r ^ 2 * z ^ 2 * (E * E)) := by
    calc ρ * (E * E) * (1 + q) ≤ ρ * (E * E) * (M * (1 + r ^ 2 * z ^ 2)) :=
          mul_le_mul_of_nonneg_left hM (by positivity)
      _ = M * (ρ * (E * E) * (1 + r ^ 2 * z ^ 2)) := by ring
      _ ≤ M * (1 + r ^ 2 * z ^ 2 * (E * E)) :=
          mul_le_mul_of_nonneg_left core (by linarith)
  nlinarith [mul_le_mul_of_nonneg_left h1 hr.le]


/-! ============================================================================
    §11. Impossibility of a Common Tail Map for Gaussian and Cauchy Marginals
    ============================================================================ -/

/-- Pullback potential of a standard Gaussian marginal under `x = g(z)` with Jacobian `g'`,
    up to an additive constant: `U_G(z) = g(z)² / 2 - log g'(z)`. -/
def gaussPullbackPotential (g g' : ℝ → ℝ) (z : ℝ) : ℝ :=
  g z ^ 2 / 2 - log (g' z)

/-- Pullback potential of a standard Cauchy marginal under `x = g(z)` with Jacobian `g'`,
    up to an additive constant: `U_C(z) = log(1 + g(z)²) - log g'(z)`. -/
def cauchyPullbackPotential (g g' : ℝ → ℝ) (z : ℝ) : ℝ :=
  log (1 + g z ^ 2) - log (g' z)

/-- Jacobian cancellation: the difference of the two pullback potentials depends on `g(z)` only,
    `U_G(z) - U_C(z) = g(z)² / 2 - log(1 + g(z)²)`, whatever the map `g`. -/
theorem pullback_potential_difference (g g' : ℝ → ℝ) (z : ℝ) :
    gaussPullbackPotential g g' z - cauchyPullbackPotential g g' z
      = g z ^ 2 / 2 - log (1 + g z ^ 2) := by
  unfold gaussPullbackPotential cauchyPullbackPotential; ring

/-- A linear lower envelope `a z - b ≤ U_C(z)` is the Jacobian bound
    `g'(z) ≤ exp(b - a z) (1 + g(z)²)`. -/
theorem deriv_bound_of_cauchy_envelope {g g' : ℝ → ℝ} {a b z : ℝ} (hpos : 0 < g' z)
    (h : a * z - b ≤ cauchyPullbackPotential g g' z) :
    g' z ≤ exp (b - a * z) * (1 + g z ^ 2) := by
  unfold cauchyPullbackPotential at h
  have h1 : 0 < 1 + g z ^ 2 := by positivity
  have hlog : log (g' z) ≤ log (exp (b - a * z) * (1 + g z ^ 2)) := by
    rw [log_mul (exp_pos _).ne' h1.ne', log_exp]; linarith
  exact (log_le_log_iff hpos (by positivity)).mp hlog

/-- Elementary arctangent bound: `t / (1 + t²) ≤ arctan t` for `t ≥ 0`. -/
theorem div_one_add_sq_le_arctan {t : ℝ} (ht : 0 ≤ t) : t / (1 + t ^ 2) ≤ arctan t := by
  let F : ℝ → ℝ := fun s => arctan s - s / (1 + s ^ 2)
  have hFd : ∀ s, HasDerivAt F
      (1 / (1 + s ^ 2) - (1 * (1 + s ^ 2) - s * (2 * s)) / (1 + s ^ 2) ^ 2) s := by
    intro s
    have h2 : HasDerivAt (fun s : ℝ => 1 + s ^ 2) (2 * s) s := by
      simpa using (hasDerivAt_pow 2 s).const_add 1
    exact (hasDerivAt_arctan s).sub ((hasDerivAt_id' s).div h2 (by positivity))
  have hmono : Monotone F := by
    apply monotone_of_deriv_nonneg (fun s => (hFd s).differentiableAt)
    intro s
    rw [(hFd s).deriv]
    have hs : 0 < 1 + s ^ 2 := by positivity
    have e : 1 / (1 + s ^ 2) - (1 * (1 + s ^ 2) - s * (2 * s)) / (1 + s ^ 2) ^ 2
        = 2 * s ^ 2 / (1 + s ^ 2) ^ 2 := by field_simp; ring
    rw [e]; positivity
  have h0 := hmono ht
  simp only [F, arctan_zero] at h0
  norm_num at h0
  linarith

/-- Exact tail transfer for a Cauchy marginal: if `g' ≤ exp(b - a z) (1 + g²)` on `[0, ∞)` and
    `g → ∞`, then the pullback tail mass `P(Z > z) = (π/2 - arctan g(z)) / π` is exponentially
    small: `π/2 - arctan g(z) ≤ (exp b / a) exp(-a z)`. -/
theorem cauchy_pullback_tail_bound {g g' : ℝ → ℝ} {a b : ℝ} (ha : 0 < a)
    (hderiv : ∀ z, HasDerivAt g (g' z) z) (htop : Tendsto g atTop atTop)
    (hbound : ∀ z, 0 ≤ z → g' z ≤ exp (b - a * z) * (1 + g z ^ 2)) (z : ℝ) (hz : 0 ≤ z) :
    π / 2 - arctan (g z) ≤ exp b / a * exp (-(a * z)) := by
  set Φ : ℝ → ℝ := fun w => arctan (g w) + exp b / a * exp (-(a * w)) with hΦ
  have hΦd : ∀ w, HasDerivAt Φ
      (1 / (1 + g w ^ 2) * g' w + exp b / a * (exp (-(a * w)) * (-a))) w := by
    intro w
    have h2 : HasDerivAt (fun w : ℝ => -(a * w)) (-a) w := by
      have hf : (fun w : ℝ => -(a * w)) = fun w => (-a) * w := by funext x; ring
      rw [hf]; simpa using (hasDerivAt_id' w).const_mul (-a)
    exact (hderiv w).arctan.add (h2.exp.const_mul (exp b / a))
  have hanti : AntitoneOn Φ (Set.Ici 0) := by
    apply antitoneOn_of_deriv_nonpos (convex_Ici 0)
    · exact fun w _ => (hΦd w).continuousAt.continuousWithinAt
    · exact fun w _ => (hΦd w).differentiableAt.differentiableWithinAt
    · intro w hw
      rw [interior_Ici] at hw
      rw [(hΦd w).deriv]
      have hb := hbound w (le_of_lt hw)
      have h1 : 0 < 1 + g w ^ 2 := by positivity
      have hle : 1 / (1 + g w ^ 2) * g' w ≤ exp b * exp (-(a * w)) := by
        rw [one_div, inv_mul_le_iff₀ h1]
        calc g' w ≤ exp (b - a * w) * (1 + g w ^ 2) := hb
          _ = (1 + g w ^ 2) * (exp b * exp (-(a * w))) := by
            rw [sub_eq_add_neg, exp_add]; ring
      have e2 : exp b / a * (exp (-(a * w)) * (-a)) = -(exp b * exp (-(a * w))) := by
        field_simp
      linarith
  have hlim : Tendsto Φ atTop (𝓝 (π / 2)) := by
    have h1 : Tendsto (fun w => arctan (g w)) atTop (𝓝 (π / 2)) :=
      (tendsto_nhds_of_tendsto_nhdsWithin tendsto_arctan_atTop).comp htop
    have h3 : Tendsto (fun w => exp (-(a * w))) atTop (𝓝 0) :=
      tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_id.const_mul_atTop ha)
    have h2 : Tendsto (fun w => exp b / a * exp (-(a * w))) atTop (𝓝 0) := by
      simpa using h3.const_mul (exp b / a)
    simpa using h1.add h2
  have hle : π / 2 ≤ Φ z := by
    apply le_of_tendsto hlim
    filter_upwards [eventually_ge_atTop z] with w hw
    exact hanti (Set.mem_Ici.mpr hz) (Set.mem_Ici.mpr (le_trans hz hw)) hw
  simp only [hΦ] at hle
  linarith

/-- Exponential growth forced on the map: under the hypotheses of `cauchy_pullback_tail_bound`,
    `g(z) ≥ (a / (2 exp b)) exp(a z)` wherever `g(z) ≥ 1`. -/
theorem exp_growth_of_cauchy_pullback_envelope {g g' : ℝ → ℝ} {a b : ℝ} (ha : 0 < a)
    (hderiv : ∀ z, HasDerivAt g (g' z) z) (htop : Tendsto g atTop atTop)
    (hbound : ∀ z, 0 ≤ z → g' z ≤ exp (b - a * z) * (1 + g z ^ 2)) (z : ℝ) (hz : 0 ≤ z)
    (hg1 : 1 ≤ g z) :
    a / (2 * exp b) * exp (a * z) ≤ g z := by
  have htail := cauchy_pullback_tail_bound ha hderiv htop hbound z hz
  have hgpos : 0 < g z := by linarith
  rw [← arctan_inv_of_pos hgpos] at htail
  have ht0 : 0 ≤ (g z)⁻¹ := inv_nonneg.mpr hgpos.le
  have ht1 : (g z)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hg1
  have harc := div_one_add_sq_le_arctan ht0
  have h2 : (g z)⁻¹ / 2 ≤ (g z)⁻¹ / (1 + (g z)⁻¹ ^ 2) :=
    div_le_div_of_nonneg_left ht0 (by positivity) (by nlinarith)
  have key : (g z)⁻¹ / 2 ≤ exp b / a * exp (-(a * z)) := by linarith
  have hE := exp_pos (a * z)
  have hb := exp_pos b
  rw [exp_neg] at key
  have key2 : a * exp (a * z) ≤ 2 * exp b * g z := by
    have e1 : (g z)⁻¹ / 2 = 1 / (2 * g z) := by field_simp
    have e2 : exp b / a * (exp (a * z))⁻¹ = exp b / (a * exp (a * z)) := by field_simp
    rw [e1, e2, div_le_div_iff₀ (by positivity) (by positivity)] at key
    linarith
  rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
  linarith

/-- Quantitative trade-off: if the Cauchy pullback potential has a linear lower envelope
    `a z - b` on `[0, ∞)` (exponential tails), then the Gaussian pullback potential through the
    same map grows at least like `exp(2 a z)`:
    `(a / (2 exp b))² / 4 · exp(2 a z) - b - log 4 ≤ U_G(z)` wherever `g(z) ≥ 1`. -/
theorem gaussPullback_lower_of_cauchy_envelope {g g' : ℝ → ℝ} {a b : ℝ} (ha : 0 < a)
    (hderiv : ∀ z, HasDerivAt g (g' z) z) (hpos : ∀ z, 0 < g' z) (htop : Tendsto g atTop atTop)
    (hcauchy : ∀ z, 0 ≤ z → a * z - b ≤ cauchyPullbackPotential g g' z) (z : ℝ) (hz : 0 ≤ z)
    (hg1 : 1 ≤ g z) :
    (a / (2 * exp b)) ^ 2 / 4 * exp (2 * (a * z)) - b - log 4 ≤ gaussPullbackPotential g g' z := by
  have hbound : ∀ w, 0 ≤ w → g' w ≤ exp (b - a * w) * (1 + g w ^ 2) :=
    fun w hw => deriv_bound_of_cauchy_envelope (hpos w) (hcauchy w hw)
  have hgrow := exp_growth_of_cauchy_pullback_envelope ha hderiv htop hbound z hz hg1
  have hc := hcauchy z hz
  unfold cauchyPullbackPotential at hc
  unfold gaussPullbackPotential
  have h1 : 0 < 1 + g z ^ 2 := by positivity
  have hlog4 : log (1 + g z ^ 2) ≤ g z ^ 2 / 4 + log 4 - 3 / 4 := by
    have := log_le_sub_one_of_pos (show 0 < (1 + g z ^ 2) / 4 by positivity)
    rw [log_div h1.ne' (by norm_num)] at this
    linarith
  have hsq : (a / (2 * exp b) * exp (a * z)) ^ 2 ≤ g z ^ 2 :=
    pow_le_pow_left₀ (by positivity) hgrow 2
  have hexp : (a / (2 * exp b) * exp (a * z)) ^ 2
      = (a / (2 * exp b)) ^ 2 * exp (2 * (a * z)) := by
    rw [mul_pow, sq (exp (a * z)), ← exp_add]; ring_nf
  have haz : 0 ≤ a * z := mul_nonneg ha.le hz
  nlinarith

/-- Impossibility theorem (Gaussian versus Cauchy). No increasing differentiable map `g` with
    `g → ∞` gives both an exponential-tailed Cauchy pullback (`U_C(z) ≥ a z - b` on `[0, ∞)`,
    `a > 0`) and an at-most-quadratic Gaussian pullback (`U_G(z) ≤ A z² + B` on `[0, ∞)`). -/
theorem no_common_map_gaussian_cauchy {g g' : ℝ → ℝ} {a b A B : ℝ} (ha : 0 < a)
    (hderiv : ∀ z, HasDerivAt g (g' z) z) (hpos : ∀ z, 0 < g' z) (htop : Tendsto g atTop atTop)
    (hcauchy : ∀ z, 0 ≤ z → a * z - b ≤ cauchyPullbackPotential g g' z)
    (hgauss : ∀ z, 0 ≤ z → gaussPullbackPotential g g' z ≤ A * z ^ 2 + B) : False := by
  obtain ⟨z₁, hz₁⟩ := eventually_atTop.mp (htop.eventually_ge_atTop 1)
  set k := (a / (2 * exp b)) ^ 2 / 4 with hk
  have hkpos : 0 < k := by positivity
  set C := b + log 4 with hC
  set D := |A| + |B| + |C| + 1 with hD
  set κ := k * (2 * a) ^ 3 / 6 with hκ
  have hκpos : 0 < κ := by positivity
  set T := max (max z₁ 1) (D / κ) with hT
  have hT1 : 1 ≤ T := le_trans (le_max_right _ _) (le_max_left _ _)
  have hTz : z₁ ≤ T := le_trans (le_max_left _ _) (le_max_left _ _)
  have hTD : D / κ ≤ T := le_max_right _ _
  have hT0 : 0 ≤ T := by linarith
  have hlow := gaussPullback_lower_of_cauchy_envelope ha hderiv hpos htop hcauchy T hT0 (hz₁ T hTz)
  have hup := hgauss T hT0
  have hcube : (2 * (a * T)) ^ 3 / 6 ≤ exp (2 * (a * T)) := by
    have := pow_div_factorial_le_exp (2 * (a * T)) (by positivity) 3
    simpa [Nat.factorial] using this
  have hκT : D ≤ κ * T := by rwa [div_le_iff₀ hκpos, mul_comm] at hTD
  have hkE : κ * T ^ 3 ≤ k * exp (2 * (a * T)) := by
    have e : κ * T ^ 3 = k * ((2 * (a * T)) ^ 3 / 6) := by rw [hκ]; ring
    rw [e]; exact mul_le_mul_of_nonneg_left hcube hkpos.le
  have hDT : D * T ^ 2 ≤ κ * T ^ 3 := by
    have e : κ * T ^ 3 = (κ * T) * T ^ 2 := by ring
    rw [e]; exact mul_le_mul_of_nonneg_right hκT (by positivity)
  have hT2 : 1 ≤ T ^ 2 := by nlinarith
  have hAbs : A * T ^ 2 + B + C < D * T ^ 2 := by
    have h1 : A * T ^ 2 ≤ |A| * T ^ 2 :=
      mul_le_mul_of_nonneg_right (le_abs_self A) (by positivity)
    have h2 : B ≤ |B| * T ^ 2 := by nlinarith [le_abs_self B, abs_nonneg B]
    have h3 : C ≤ |C| * T ^ 2 := by nlinarith [le_abs_self C, abs_nonneg C]
    rw [hD]; nlinarith
  have : k * exp (2 * (a * T)) - b - log 4 ≤ A * T ^ 2 + B := le_trans hlow hup
  linarith

/-- The arcsinh map (`g = sinh`, `g' = cosh`) meets the Cauchy hypothesis of the impossibility
    theorem with `a = 1`, `b = log 2`: `z - log 2 ≤ U_C(z) = log cosh z` on `[0, ∞)`. -/
theorem arcsinh_cauchy_pullback_linear_lower (z : ℝ) (hz : 0 ≤ z) :
    1 * z - log 2 ≤ cauchyPullbackPotential sinh cosh z := by
  unfold cauchyPullbackPotential
  have hc := cosh_pos z
  rw [one_add_sinh_sq, sq, log_mul hc.ne' hc.ne']
  have h := half_exp_abs_le_cosh z
  rw [abs_of_nonneg hz] at h
  have hl : log (exp z / 2) ≤ log (cosh z) := log_le_log (by positivity) h
  rw [log_div (exp_pos z).ne' two_ne_zero, log_exp] at hl
  linarith

/-- Consequence for arcsinh: the Gaussian pullback potential `sinh² z / 2 - log cosh z` admits no
    quadratic upper bound `A z² + B` on `[0, ∞)`, for any constants `A`, `B`. -/
theorem arcsinh_gauss_pullback_not_quadratic (A B : ℝ) :
    ¬ ∀ z, 0 ≤ z → gaussPullbackPotential sinh cosh z ≤ A * z ^ 2 + B := by
  intro h
  have htop : Tendsto sinh atTop atTop := by
    refine tendsto_atTop_mono' _ ?_ tendsto_id
    filter_upwards [eventually_ge_atTop 0] with x hx
    exact self_le_sinh_iff.mpr hx
  exact no_common_map_gaussian_cauchy one_pos hasDerivAt_sinh cosh_pos htop
    arcsinh_cauchy_pullback_linear_lower h

/-- Algebraic core of the tail-window necessity: if a common map puts marginal 1 below a quadratic
    hazard and marginal 2 above a linear one at `t`, then at `x = g(t)` the hazards satisfy
    `ψ₁(x) ≤ A ((ψ₂(x) + b) / a)² + B`, a relation free of `t`. -/
theorem window_necessity {ψ₁ ψ₂ g : ℝ → ℝ} {a b A B t : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (ht : 0 ≤ t) (h₁ : ψ₁ (g t) ≤ A * t ^ 2 + B) (h₂ : a * t - b ≤ ψ₂ (g t)) :
    ψ₁ (g t) ≤ A * ((ψ₂ (g t) + b) / a) ^ 2 + B := by
  have hlt : t ≤ (ψ₂ (g t) + b) / a := by rw [le_div_iff₀ ha]; linarith
  have hsq : t ^ 2 ≤ ((ψ₂ (g t) + b) / a) ^ 2 := pow_le_pow_left₀ ht hlt 2
  nlinarith [mul_le_mul_of_nonneg_left hsq hA]


/-! ============================================================================
    §12. Leapfrog Stability Region of the Arcsinh Pullback of a Gaussian
    ============================================================================ -/

/-- Pullback potential of `N(μ, σ²)` under `x = μ + R sinh z`, with `r = R / σ`, up to an
    additive constant: `U(z) = r² sinh² z / 2 - log cosh z`. -/
def arcsinhGaussPotential (r z : ℝ) : ℝ :=
  r ^ 2 * sinh z ^ 2 / 2 - log (cosh z)

/-- Gradient of `arcsinhGaussPotential`: `U'(z) = r² sinh z cosh z - tanh z`. -/
def arcsinhGaussGrad (r z : ℝ) : ℝ :=
  r ^ 2 * sinh z * cosh z - sinh z / cosh z

/-- Curvature of `arcsinhGaussPotential`: `U''(z) = r² cosh 2z - sech² z`. -/
def arcsinhGaussCurv (r z : ℝ) : ℝ :=
  r ^ 2 * cosh (2 * z) - (cosh z ^ 2)⁻¹

/-- With `r = 1` the §12 potential is the §11 Gaussian pullback potential through `sinh`. -/
theorem arcsinhGaussPotential_one_eq (z : ℝ) :
    arcsinhGaussPotential 1 z = gaussPullbackPotential sinh cosh z := by
  unfold arcsinhGaussPotential gaussPullbackPotential; ring

/-- Derivative identity: `d/dz [r² sinh² z / 2 - log cosh z] = r² sinh z cosh z - tanh z`. -/
theorem hasDerivAt_arcsinhGaussPotential (r z : ℝ) :
    HasDerivAt (arcsinhGaussPotential r) (arcsinhGaussGrad r z) z := by
  have h1 : HasDerivAt (fun w => r ^ 2 * sinh w ^ 2 / 2)
      (r ^ 2 * (2 * sinh z * cosh z) / 2) z := by
    have := (((hasDerivAt_sinh z).pow 2).const_mul (r ^ 2)).div_const 2
    simpa using this
  have h2 : HasDerivAt (fun w => log (cosh w)) (sinh z / cosh z) z :=
    (hasDerivAt_cosh z).log (cosh_pos z).ne'
  have h := h1.sub h2
  have e : r ^ 2 * (2 * sinh z * cosh z) / 2 - sinh z / cosh z = arcsinhGaussGrad r z := by
    unfold arcsinhGaussGrad; ring
  rw [e] at h
  exact h

/-- Second-derivative identity: `d/dz [r² sinh z cosh z - tanh z] = r² cosh 2z - sech² z`. -/
theorem hasDerivAt_arcsinhGaussGrad (r z : ℝ) :
    HasDerivAt (arcsinhGaussGrad r) (arcsinhGaussCurv r z) z := by
  have hc := (cosh_pos z).ne'
  have h1 := ((hasDerivAt_sinh z).mul (hasDerivAt_cosh z)).const_mul (r ^ 2)
  have h2 := (hasDerivAt_sinh z).div (hasDerivAt_cosh z) hc
  have h := h1.sub h2
  have e : r ^ 2 * (cosh z * cosh z + sinh z * sinh z)
      - (cosh z * cosh z - sinh z * sinh z) / cosh z ^ 2 = arcsinhGaussCurv r z := by
    unfold arcsinhGaussCurv
    rw [cosh_two_mul]
    have hcs : cosh z ^ 2 = sinh z ^ 2 + 1 := cosh_sq z
    field_simp
    rw [hcs]; ring
  have hfun : arcsinhGaussGrad r = (fun y => r ^ 2 * (sinh * cosh) y) - sinh / cosh := by
    funext w; simp only [arcsinhGaussGrad, Pi.sub_apply, Pi.mul_apply, Pi.div_apply]; ring
  rw [hfun, ← e]
  exact h

/-- Curvature in standardised units `s = r sinh z = (x - μ) / σ`:
    `U''(z) = r² + 2 s² - sech² z`. -/
theorem arcsinhGaussCurv_eq (r z : ℝ) :
    arcsinhGaussCurv r z = r ^ 2 + 2 * (r * sinh z) ^ 2 - (cosh z ^ 2)⁻¹ := by
  unfold arcsinhGaussCurv
  rw [cosh_two_mul, cosh_sq]; ring

/-- `0 < sech² z ≤ 1`. -/
theorem inv_cosh_sq_pos_le_one (z : ℝ) : 0 < (cosh z ^ 2)⁻¹ ∧ (cosh z ^ 2)⁻¹ ≤ 1 := by
  have hc := cosh_pos z
  have h1 : 1 ≤ cosh z ^ 2 := by rw [cosh_sq]; nlinarith [sq_nonneg (sinh z)]
  exact ⟨by positivity, inv_le_one_of_one_le₀ h1⟩

/-- Linear leapfrog stability (`ε² U'' ≤ 4`) holds wherever the standardised Gaussian coordinate
    satisfies `s² ≤ 2 / ε² - r² / 2`. -/
theorem leapfrog_stable_of_standardized_le {r ε z : ℝ} (hε : 0 < ε)
    (h : (r * sinh z) ^ 2 ≤ 2 / ε ^ 2 - r ^ 2 / 2) :
    ε ^ 2 * arcsinhGaussCurv r z ≤ 4 := by
  rw [arcsinhGaussCurv_eq]
  have hs := (inv_cosh_sq_pos_le_one z).1
  have hε2 : 0 < ε ^ 2 := by positivity
  have hcurv : r ^ 2 + 2 * (r * sinh z) ^ 2 - (cosh z ^ 2)⁻¹ ≤ 4 / ε ^ 2 := by
    have e : 4 / ε ^ 2 = 2 * (2 / ε ^ 2) := by ring
    rw [e]; linarith
  calc ε ^ 2 * (r ^ 2 + 2 * (r * sinh z) ^ 2 - (cosh z ^ 2)⁻¹) ≤ ε ^ 2 * (4 / ε ^ 2) :=
        mul_le_mul_of_nonneg_left hcurv hε2.le
    _ = 4 := by field_simp

/-- Linear leapfrog stability fails (`ε² U'' ≥ 4`) wherever the standardised Gaussian coordinate
    satisfies `s² ≥ 2 / ε² - r² / 2 + 1 / 2`. -/
theorem leapfrog_unstable_of_standardized_ge {r ε z : ℝ} (hε : 0 < ε)
    (h : 2 / ε ^ 2 - r ^ 2 / 2 + 1 / 2 ≤ (r * sinh z) ^ 2) :
    4 ≤ ε ^ 2 * arcsinhGaussCurv r z := by
  rw [arcsinhGaussCurv_eq]
  have hs := (inv_cosh_sq_pos_le_one z).2
  have hε2 : 0 < ε ^ 2 := by positivity
  have hcurv : 4 / ε ^ 2 ≤ r ^ 2 + 2 * (r * sinh z) ^ 2 - (cosh z ^ 2)⁻¹ := by
    have e : 4 / ε ^ 2 = 2 * (2 / ε ^ 2) := by ring
    rw [e]; linarith
  calc (4 : ℝ) = ε ^ 2 * (4 / ε ^ 2) := by field_simp
    _ ≤ ε ^ 2 * (r ^ 2 + 2 * (r * sinh z) ^ 2 - (cosh z ^ 2)⁻¹) :=
        mul_le_mul_of_nonneg_left hcurv hε2.le

/-- The curvature is nondecreasing in `|z|`, so the set where `ε² U'' ≥ 4` is exactly
    `{|z| ≥ z*}` for some threshold `z*`, i.e. a symmetric tail region of the Gaussian. -/
theorem arcsinhGaussCurv_mono_abs (r : ℝ) {z₁ z₂ : ℝ} (h : |z₁| ≤ |z₂|) :
    arcsinhGaussCurv r z₁ ≤ arcsinhGaussCurv r z₂ := by
  rw [arcsinhGaussCurv_eq, arcsinhGaussCurv_eq]
  have hc : cosh z₁ ≤ cosh z₂ := cosh_le_cosh.mpr h
  have hc1 := cosh_pos z₁
  have hsq : cosh z₁ ^ 2 ≤ cosh z₂ ^ 2 := pow_le_pow_left₀ hc1.le hc 2
  have hinv : (cosh z₂ ^ 2)⁻¹ ≤ (cosh z₁ ^ 2)⁻¹ := inv_anti₀ (by positivity) hsq
  have hs1 : sinh z₁ ^ 2 = cosh z₁ ^ 2 - 1 := by rw [cosh_sq]; ring
  have hs2 : sinh z₂ ^ 2 = cosh z₂ ^ 2 - 1 := by rw [cosh_sq]; ring
  have hr : 0 ≤ r ^ 2 := sq_nonneg r
  have hss : r ^ 2 * sinh z₁ ^ 2 ≤ r ^ 2 * sinh z₂ ^ 2 := by
    rw [hs1, hs2]; exact mul_le_mul_of_nonneg_left (by linarith) hr
  nlinarith [hss, hinv]

/-- Curvature at the centre: `U''(0) = r² - 1`, so the arcsinh pullback of a Gaussian is bimodal
    exactly when the transform scale is below the Gaussian scale (`R < σ`). -/
theorem arcsinhGaussCurv_zero (r : ℝ) : arcsinhGaussCurv r 0 = r ^ 2 - 1 := by
  unfold arcsinhGaussCurv
  simp


/-! ============================================================================
    PROPERTY 6: Axiom Audit & Linter Verification
    ============================================================================ -/

#print axioms arcsinh_inv_right
#print axioms arcsinh_inv_left
#print axioms scaled_invertibility_roundtrip_forward
#print axioms scaled_invertibility_roundtrip_inverse
#print axioms shifted_invertibility_roundtrip_forward
#print axioms shifted_invertibility_roundtrip_inverse
#print axioms shiftedForwardMap_translation_invariant
#print axioms inverseMap_jacobian_factor_pos
#print axioms arsinh_deriv_eq
#print axioms metric_scale_pos
#print axioms metric_scale_le_one
#print axioms hasDerivAt_forwardMap
#print axioms forwardMap_metric_pos
#print axioms rho_pos
#print axioms rho_le_one
#print axioms rho_strictAntiOn
#print axioms pureSqrtStepSq_lt_div_gamma
#print axioms pureSqrt_quartic_discrete_basin_bound
#print axioms rhoSoft_pos
#print axioms rhoSoft_le_one
#print axioms softCap_dimensionless_drift_sq_lt
#print axioms softCap_dimensionless_drift_lt
#print axioms softCappedDriftStep_lt_bound
#print axioms softCapped_global_contraction_no_overshoot
#print axioms mhAccept_bounds
#print axioms mh_detailed_balance
/-- info: 'McmcRobust.studentTNegLogKernel_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms studentTNegLogKernel_nonneg
/-- info: 'McmcRobust.hasDerivAt_studentTNegLogKernel' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hasDerivAt_studentTNegLogKernel
/-- info: 'McmcRobust.studentTScore_mul_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms studentTScore_mul_le
/-- info: 'McmcRobust.cauchyShrinkagePenalty_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms cauchyShrinkagePenalty_nonneg
/-- info: 'McmcRobust.studentTScore_zero' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms studentTScore_zero

/-- info: 'McmcRobust.calibratedRadius_ge_min' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibratedRadius_ge_min

/-- info: 'McmcRobust.calibratedRadius_pos' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibratedRadius_pos

/-- info: 'McmcRobust.calibratedRadius_le_max' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibratedRadius_le_max

/-- info: 'McmcRobust.calibratedRadius_mono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibratedRadius_mono

/-- info: 'McmcRobust.calibrated_shifted_roundtrip_forward' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibrated_shifted_roundtrip_forward

/-- info: 'McmcRobust.calibrated_shifted_roundtrip_inverse' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibrated_shifted_roundtrip_inverse

/-- info: 'McmcRobust.one_add_sinh_sq' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms one_add_sinh_sq

/-- info: 'McmcRobust.arcsinhCauchyPullback_eq_sech' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinhCauchyPullback_eq_sech

/-- info: 'McmcRobust.half_exp_abs_le_cosh' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms half_exp_abs_le_cosh

/-- info: 'McmcRobust.cosh_le_exp_abs' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms cosh_le_exp_abs

/-- info: 'McmcRobust.sech_envelope' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms sech_envelope

/-- info: 'McmcRobust.calibration_kernel_upper' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibration_kernel_upper

/-- info: 'McmcRobust.calibration_kernel_lower' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms calibration_kernel_lower

/-- info: 'McmcRobust.arcsinhCauchyPullback_envelope' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinhCauchyPullback_envelope

/-- info: 'McmcRobust.arcsinhCauchyPullback_exp_envelope' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinhCauchyPullback_exp_envelope

/-- info: 'McmcRobust.hasDerivAt_hMap' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hasDerivAt_hMap

/-- info: 'McmcRobust.hMapDeriv_pos' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMapDeriv_pos

/-- info: 'McmcRobust.log_hMapDeriv' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms log_hMapDeriv

/-- info: 'McmcRobust.hMap_strictMono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMap_strictMono

/-- info: 'McmcRobust.hMap_surjective' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMap_surjective

/-- info: 'McmcRobust.hMap_bijective' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMap_bijective

/-- info: 'McmcRobust.hMap_rescale' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMap_rescale

/-- info: 'McmcRobust.hMap_rescale_invariant' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMap_rescale_invariant

/-- info: 'McmcRobust.hMapCauchyPullback_eq' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMapCauchyPullback_eq

/-- info: 'McmcRobust.hMapCauchyPullback_gauss_lower' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMapCauchyPullback_gauss_lower

/-- info: 'McmcRobust.hMapCauchyPullback_gauss_upper' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hMapCauchyPullback_gauss_upper

/-- info: 'McmcRobust.pullback_potential_difference' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms pullback_potential_difference

/-- info: 'McmcRobust.deriv_bound_of_cauchy_envelope' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms deriv_bound_of_cauchy_envelope

/-- info: 'McmcRobust.div_one_add_sq_le_arctan' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms div_one_add_sq_le_arctan

/-- info: 'McmcRobust.cauchy_pullback_tail_bound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms cauchy_pullback_tail_bound

/-- info: 'McmcRobust.exp_growth_of_cauchy_pullback_envelope' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms exp_growth_of_cauchy_pullback_envelope

/-- info: 'McmcRobust.gaussPullback_lower_of_cauchy_envelope' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms gaussPullback_lower_of_cauchy_envelope

/-- info: 'McmcRobust.no_common_map_gaussian_cauchy' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms no_common_map_gaussian_cauchy

/-- info: 'McmcRobust.arcsinh_cauchy_pullback_linear_lower' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinh_cauchy_pullback_linear_lower

/-- info: 'McmcRobust.arcsinh_gauss_pullback_not_quadratic' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinh_gauss_pullback_not_quadratic

/-- info: 'McmcRobust.window_necessity' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms window_necessity

/-- info: 'McmcRobust.arcsinhGaussPotential_one_eq' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinhGaussPotential_one_eq

/-- info: 'McmcRobust.hasDerivAt_arcsinhGaussPotential' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hasDerivAt_arcsinhGaussPotential

/-- info: 'McmcRobust.hasDerivAt_arcsinhGaussGrad' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms hasDerivAt_arcsinhGaussGrad

/-- info: 'McmcRobust.arcsinhGaussCurv_eq' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinhGaussCurv_eq

/-- info: 'McmcRobust.inv_cosh_sq_pos_le_one' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms inv_cosh_sq_pos_le_one

/-- info: 'McmcRobust.leapfrog_stable_of_standardized_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms leapfrog_stable_of_standardized_le

/-- info: 'McmcRobust.leapfrog_unstable_of_standardized_ge' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms leapfrog_unstable_of_standardized_ge

/-- info: 'McmcRobust.arcsinhGaussCurv_mono_abs' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinhGaussCurv_mono_abs

/-- info: 'McmcRobust.arcsinhGaussCurv_zero' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms arcsinhGaussCurv_zero

#lint

end

end McmcRobust
