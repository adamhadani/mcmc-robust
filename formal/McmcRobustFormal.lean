import Mathlib.Analysis.SpecialFunctions.Arsinh
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Comp
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

6. **Pure-JAX 3-Phase Scale & Location Calibration Bounds (§7)**:
   - Clipped normal-consistent coordinate scale calibration map:
     $$R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c) = \min(R_{\max}, \max(R_{\min}, c \cdot \sigma))$$
   - Strict interval bounds $0 < R_{\min} \le R_{\text{cal}}(\sigma, R_{\min}, R_{\max}, c) \le R_{\max}$,
     monotonicity in $\sigma$ for $c \ge 0$, and unconditional roundtrip invertibility of
     $\psi_{\mu, R_{\text{cal}}} \circ \phi_{\mu, R_{\text{cal}}}$ for any sample median $\mu \in \mathbb{R}$
     and dispersion $\sigma \in \mathbb{R}$ (including zero-variance $\sigma \le 0$).

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
    §7. Pure-JAX 3-Phase Scale & Location Calibration Bounds
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

#lint

end

end McmcRobust
