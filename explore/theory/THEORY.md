# Coordinate-wise tail transforms for HMC and NUTS: what can be proved

> **Exploration snapshot, branch `explore/tail-transform-theory`, not maintained docs.**
> Written 2026-10-24 on top of `main` at `92a2596`. Nothing here is a claim of `README.md` or `REPORT.md`. The numbers in this file come from the scripts in `explore/theory/` (outputs saved in `explore/theory/outputs/`), not from `reports/`, so AGENTS.md §1 forbids quoting them in the maintained docs until a runner produces them.

**Status labels used below.**

- **Lean**: machine-checked in `formal/McmcRobustFormal.lean` (§9–§12, theorem names given). `lake build` passes with zero errors, zero warnings, no `sorry`, and every theorem is in the `#guard_msgs` axiom audit.
- **Proved**: complete proof written here, using only cited published theorems.
- **Sketch**: the argument is given, but some routine steps are only indicated.
- **Numerical**: checked numerically only.
- **Conjecture** / **Open**: not proved.

## Summary

1. **Tail classification (Section A).** A coordinate map `x = g(z)` transfers tails exactly: the pullback hazard is the marginal hazard composed with `g`. Each map shape covers one "window" of marginal tails, and each window spans exactly a square in the hazard scale. Arcsinh covers polynomial to lognormal tails. Tukey's h-map covers polynomial and heavier tails. The identity covers exponential to Gaussian tails. Explicit non-asymptotic envelopes are machine-checked for Cauchy marginals under arcsinh (Laplace tail, rate 1, for every calibration) and under the h-map (Gaussian tail, variance `c`).
2. **Impossibility (Section B).** No single map, even with per-coordinate location and scale calibration, puts a Gaussian coordinate and a polynomial-tailed coordinate into the potential class assumed by any published geometric-ergodicity theorem for HMC or NUTS. The core is machine-checked for the Gaussian–Cauchy pair. Quantitatively, if the Cauchy pullback potential grows at least like `a z`, the Gaussian pullback potential grows at least like `exp(2 a z)`, and arcsinh attains this rate. A tail-level characterisation holds as well: two marginals admit a common map iff their hazards are within a square of each other.
3. **Positive result (Section C).** Use the affine map on Gaussian coordinates and an h-map with any `c > 0` on Student-*t* coordinates. The pullback of a product target is then "Gaussian plus a perturbation with bounded, Lipschitz gradient", the class in which HMC (Durmus, Moulines & Saksman 2020, Thm 9(c)) and NUTS (Gruffaz et al. 2026, Thm 19(b); Durmus et al. 2023, Thm 16(b)) are proved geometrically ergodic. This holds for every value of the tail index, so misspecifying it only changes the Gaussian variance. The arcsinh pullback of the same coordinates sits at the Laplace edge, which no NUTS theorem covers.
4. **Implications (Section D).** Fixed-length HMC on a product target is not geometrically ergodic if any coordinate is light-tailed and pulled back by arcsinh or by an h-map with finite `c`. It is also not geometrically ergodic if a Student-*t* coordinate is left untransformed or pulled back by a map of polynomial growth. The same holds for NUTS in the heavy-tail case. For NUTS in the light-tail case the question is open.
5. **Practical size of the light-tail damage (Section E).** Under arcsinh, the Gaussian mass of the region where leapfrog is linearly unstable at step size `ε` is about `2Φ(−√2/ε)`, whatever the scale `R`. The bracket is machine-checked. Under an h-map with constant `c` the same mass decays roughly like `exp(−const·c/ε²)` (numerical).
6. **Selection is the bottleneck.** Theorem B makes per-coordinate selection necessary. A Hill-estimator threshold is consistent only asymptotically. At warm-up sample sizes it cannot tell Laplace from Student-*t* with $\nu = 5$, and it often labels lognormal coordinates heavy (numerical, Section C.4).

---

## 0. Setup and notation

A target coordinate `x` has marginal density `f = e^{-V}` with right tail function `F̄(x) = P(X > x)` and right hazard `ψ(x) = −log F̄(x)`. A coordinate map is a `C²` increasing bijection `g : ℝ → ℝ`. NUTS runs on the exact pullback

```math
\pi_z(z) = f(g(z)) g'(z), \qquad U(z) = -\log \pi_z(z) = V(g(z)) - \log g'(z).
```

The maps studied are listed below; `μ` and `R` are the calibrated location and scale.

| name | `x = g(z)` | growth |
|---|---|---|
| affine | $\mu + R z$ | linear |
| arcsinh (the repo's map) | $\mu + R\sinh z$ | $e^{z}$ |
| Tukey h-map, $h = 1/c$ | $\mu + R z e^{z^2/(2c)}$ | $e^{z^2/(2c)}$ |
| power | $\mu + R \mathrm{sign}(z) \vert z\vert^{p}$ | polynomial |

For a product target $\pi(x) = \prod_i f_i(x_i)$ with coordinate maps $g_i$, the pullback is again a product, and $U(z) = \sum_i U_i(z_i)$.

**The "window".** Every published geometric-ergodicity (GE) result for gradient-based samplers that we know of assumes, along each coordinate axis, a potential between linear and quadratic growth:

```math
a\vert z\vert - b \;\le\; U(z) \;\le\; A z^2 + B .
```

This covers Livingstone, Betancourt, Byrne & Girolami (2019) A2 and Cor. 2.3(i); Durmus, Moulines & Saksman (2020) A2(m)+A3(m) with $1 < m \le 2$ and A4; Durmus, Gruffaz, Kailas, Saksman & Vihola (2023) H6(m) and H7; and Gruffaz, Kim, Guehtar, Duval-Decaix & Trautmann (2026) H3(m) and H4. Each of these hypotheses implies the window: (SC1.1)+(SC1.2) or A2(m) give the linear lower bound, and (SC1.3b), A1(m−1) or a sub-quadratic perturbation give the quadratic upper bound. Outside the window there are matching negative results:

- **Light side:** gradient growing faster than linearly, plus regularity conditions. Fixed-length HMC is not GE (Livingstone et al. 2019, Thm 5.14 as corrected in their 2026 corrigendum, arXiv:1601.08057v5, App. D).
- **Heavy side:** bounded gradient and $E e^{s\vert X\vert} = \infty$ for every $s > 0$. HMC is not GE (Livingstone et al., Thm 5.18). Neither is NUTS (Gruffaz et al., Thm 13 with H2-(i); see the caveat in D.4).

So "which coordinate ends up where in the window" is the right question to ask of a coordinate map.

---

## A. Pullback tail classification

### A.1 Exact tail transfer (Proved; elementary, known)

For an increasing bijection `g`, $P(Z > t) = P(X > g(t))$. Hence

```math
\psi_Z = \psi_X \circ g .
```

Lower and upper potential envelopes transfer to hazard envelopes:

- If $U(z) \ge a z - b$ for $z \ge 0$, then $P(Z > t) \le (e^{b}/(a \mathcal{Z})) e^{-a t}$, where $\mathcal{Z}$ is the normalising constant.
- If $U(z) \le A z^2 + B$ for $z \ge 0$, then $P(Z>t) \ge \int_t^{t+1} e^{-A s^2 - B} ds / \mathcal{Z}$.

So the potential-level window is contained in the hazard-level window

```math
\mathcal{W} = \lbrace \psi_Z : \ a t - b \le \psi_Z(t) \le A t^2 + B \text{ for } t \ge 0 \rbrace .
```

### A.2 Which marginals each map handles (Proved by substitution; Numerical table)

Substituting the growth of `g` into $\psi_X \circ g$ gives the window of marginal hazards that each map sends into $\mathcal{W}$. Here $\psi \lesssim \phi$ means $\psi \le C\phi + C$.

| map | marginal hazards it sends into $\mathcal{W}$ | lower edge (pullback at $\beta = 1$) | upper edge ($\beta = 2$) |
|---|---|---|---|
| affine | $x \lesssim \psi \lesssim x^2$ | exponential (Laplace) | Gaussian |
| power $z^{p}$ | $x^{1/p} \lesssim \psi \lesssim x^{2/p}$ | Weibull $k = 1/p$ | Weibull $k = 2/p$ |
| arcsinh | $\log x \lesssim \psi \lesssim (\log x)^2$ | polynomial (Student-*t*, Cauchy) | lognormal |
| h-map | $(\log x)^{1/2} \lesssim \psi \lesssim \log x$ | $e^{-\sqrt{\log x}}$ tails | polynomial |

The local growth exponent $\beta(t) = d\log\psi_Z/d\log t$ was computed at $t = 8$ and $t = 16$ (`check_pullback_envelopes.py`, A5):

| marginal | affine | arcsinh | h-map ($c = 1$) |
|---|---|---|---|
| Gaussian | 1.86, 1.95 | 16.1, 32.2 (doubly exponential) | 131, 519 |
| Laplace | 0.92, 0.96 | super-polynomial | super-polynomial |
| lognormal | 0.61, 0.52 | 2.01, 2.04 | 3.79, 3.93 |
| Student-*t*, $\nu = 3$ | 0.47, 0.36 | 1.10, 1.05 | 1.91, 1.97 |
| Cauchy | 0.31, 0.25 | 0.95, 0.97 | 1.85, 1.95 |

Two consequences:

- **Each window spans exactly a square in the hazard scale.** Theorem B.3 shows this is forced.
- **A two-member family {affine, h-map} leaves gaps.** Lognormal-type tails need arcsinh, and Weibull tails with $k$ outside $[1, 2]$ need a power map.

### A.3 Cauchy under arcsinh: exact law and calibration-robust envelopes (Lean)

Let $x \sim \mathrm{Cauchy}(m, s)$ and $z = \mathrm{arsinh}((x-\mu)/R)$. Write $\delta = (\mu - m)/s$ for the location error and $r = R/s$ for the scale ratio. Then

```math
\pi p_z(z) = \frac{r\cosh z}{1 + (\delta + r\sinh z)^2}.
```

**Theorem A.3.**

1. Perfect calibration gives the hyperbolic secant law, $\pi p_z(z) = \mathop{\mathrm{sech}} z$. (`arcsinhCauchyPullback_eq_sech`)
2. For every $\delta \in \mathbb{R}$, every $r > 0$ and every $z$, with $K = (r^2 + 1 + \delta^2)/r$ (`arcsinhCauchyPullback_envelope`, `arcsinhCauchyPullback_exp_envelope`):

   $`\displaystyle \frac{\mathop{\mathrm{sech}} z}{K} \le \pi p_z(z) \le K \mathop{\mathrm{sech}} z, \qquad \frac{e^{-\vert z\vert}}{K} \le \pi p_z(z) \le 2K e^{-\vert z\vert}.`$
3. The sharp constant is $K^{\star} = (S + \sqrt{S^2 - 4r^2})/(2r)$ with $S = 1 + \delta^2 + r^2$, and $K^{\star} \le K$. (Proved on paper; Numerical.)

**Proof.**

- The density follows from the change of variables.
- With $y = \sinh z$ and $w = \delta + r y$, $\pi p_z(z)\cosh z = r(1+y^2)/(1+w^2)$. The two bounds are the polynomial identities

  $`\displaystyle (r^2+1+\delta^2)(1+w^2) - r^2(1+y^2) = (1 + \delta w)^2 + r^2 w^2 \ge 0`$ (`calibration_kernel_upper`)

  and

  $`\displaystyle (r^2+1+\delta^2)(1+y^2) - (1+w^2) = (r - \delta y)^2 + y^2 \ge 0`$ (`calibration_kernel_lower`).
- The exponential form follows from $e^{\vert z\vert}/2 \le \cosh z \le e^{\vert z\vert}$ (`sech_envelope`).
- Sharpness: $r(1+y^2)/(1+w^2)$ is a ratio of two positive-definite quadratic forms in $(1, y)$. Its range is $[1/K^{\star}, K^{\star}]$, where $1/K^{\star}$ and $K^{\star}$ are the roots of $r\lambda^2 - S\lambda + r = 0$ (the generalised eigenvalues, rescaled). ∎

**Reading.** The tail rate of the pullback is exactly 1, the Cauchy tail index, for every calibration. Calibration error moves only the constant, by at most $r + (1+\delta^2)/r$. This is the "calibration robustness" theorem that Track 2.3 of the state assessment asked for. Numerically, the constant $K^{\star}$ is attained for all six $(\delta, r)$ pairs tested (A1).

### A.4 Student-*t* under arcsinh (Proved; Numerical)

Let $x \sim t_\nu(m, s)$ and use the same notation, with $c_\nu = \Gamma((\nu+1)/2)/(\sqrt{\nu\pi}\Gamma(\nu/2))$. Set $r' = r/\sqrt\nu$, $\delta' = \delta/\sqrt\nu$ and $S' = r'^2 + 1 + \delta'^2$. Then for all $z$:

```math
c_\nu r S'^{-(\nu+1)/2} e^{-\nu\vert z\vert} \;\le\; p_z(z) \;\le\; c_\nu r 2^{\nu} (S'/r'^2)^{(\nu+1)/2} e^{-\nu\vert z\vert}.
```

**Proof.**

- $p_z(z) = c_\nu r\cosh z (1 + (\delta' + r'\sinh z)^2)^{-(\nu+1)/2}$.
- The two kernel bounds of A.3, applied to $(\delta', r')$, sandwich $1 + (\delta' + r' y)^2$ between $(r'^2/S')(1+y^2)$ and $S'(1+y^2)$.
- Raise to the power $-(\nu+1)/2$ and use $1 + y^2 = \cosh^2 z$ and the cosh envelopes. ∎

For $\nu = 1$ this is exactly the Lean statement. Numerically, $U'(z) \to \nu$ to six digits at $z = 15$ and $z = 25$ for $\nu \in \lbrace 0.5, 1, 3, 10 \rbrace$ and $R \in \lbrace 0.3, 1, 5\rbrace$ (A2). The pullback is therefore at $\beta = 1$, the Laplace edge, for every $\nu$.

### A.5 Tukey h-map: identities and Gaussian envelopes for Cauchy marginals (Lean)

For $R, c > 0$ the map $g(z) = \mu + R z e^{z^2/(2c)}$ satisfies:

- $g'(z) = R e^{z^2/(2c)}(1 + z^2/c) > 0$ (`hasDerivAt_hMap`, `hMapDeriv_pos`);
- $\log g'(z) = \log R + z^2/(2c) + \log(1 + z^2/c)$ (`log_hMapDeriv`);
- `g` is a strictly increasing bijection of $\mathbb{R}$ (`hMap_strictMono`, `hMap_surjective`, `hMap_bijective`). The inverse is $z = \mathrm{sign}(y)\sqrt{c W(y^2/(cR^2))}$ with $y = x - \mu$ and $W$ the Lambert function; this inverse is not formalised.
- **Only $R^2c$ is identifiable.** Rescaling the latent coordinate, $z = \lambda w$, maps the parameters $(R, c)$ to $(R\lambda, c/\lambda^2)$: `hMap μ R c (λ w) = hMap μ (R λ) (c / λ²) w` (`hMap_rescale`). The product $R^2 c$ is invariant (`hMap_rescale_invariant`). An adapted diagonal mass matrix absorbs any such rescaling, so NUTS sees only $R^2c$.
  - This observation came from the Tukey-h exploration branch (`explore/tukey-h-transform`).
  - It agrees with the constants below, which depend on $(r, c)$ only through an overall factor $r$ and through $r^2c$, once $z$ is rescaled.

**Theorem A.5.** Let $x = r z e^{z^2/(2c)} \sim \mathrm{Cauchy}(0,1)$, with $r, c > 0$. Set

- $m = \min(1, 1/(r^2 c))$,
- $M = \max(1, 1/(r^2 c))$,
- $\rho = \min(e^{-1}, r^2c/(1+r^2c))$.

Then for all $z \in \mathbb{R}$:

```math
r m e^{-z^2/(2c)} \;\le\; \pi p_z(z) \;\le\; \frac{r M}{\rho} e^{-z^2/(2c)} .
```

The Lean theorems (`hMapCauchyPullback_gauss_lower`, `hMapCauchyPullback_gauss_upper`) take $m$, $M$ and $\rho$ as any constants satisfying $m \le 1$, $m r^2 c \le 1$, $1 \le M$, $1 \le M r^2 c$, $\rho e \le 1$ and $\rho(1 + r^2 c) \le r^2 c$; the choices above satisfy these.

**Proof idea.**

- Write $E = e^{z^2/(2c)}$, so $\pi p_z E = r(1 + z^2/c)/(E^{-2} + r^2z^2)$.
- The lower bound uses $E^2 \ge 1$.
- The upper bound splits into two cases. If $z^2 \le c$, then $E^{-2} \ge e^{-1} \ge \rho$. If $z^2 \ge c$, then $r^2 z^2 \ge r^2 c$ and $\rho(1+r^2z^2) \le r^2z^2$. ∎

A location error $\delta$ multiplies both constants by at most $2(1+\delta^2)$, because $1 + (\delta + w)^2 \le 2(1+\delta^2)(1+w^2)$ and symmetrically (Proved). The envelopes hold numerically, with the true range strictly inside the bounds for six $(r, c)$ pairs (A3).

### A.6 Student-*t* under the h-map is "Gaussian plus bounded-gradient perturbation" (Proved; Numerical)

**Proposition A.6.** Take any $\nu > 0$, $m$, $s > 0$ for the marginal $t_\nu(m,s)$ and any $\mu$, $R > 0$, $c > 0$ for the map. The pullback potential of $t_\nu(m,s)$ under $x = \mu + R z e^{z^2/(2c)}$ decomposes as

```math
U(z) = \frac{\kappa}{2} z^2 + \tilde U(z), \qquad \kappa = \frac{\nu}{c},
```

where $\tilde U \in C^\infty$ has $\sup\vert\tilde U'\vert < \infty$ and $\sup\vert\tilde U''\vert < \infty$, and satisfies $\vert\tilde U(z)\vert \le A(1 + \vert z\vert)$. As $\vert z\vert \to \infty$, $\tilde U(z) = (\nu - 1)\log\vert z\vert + \mathrm{const} + o(1)$.

**Proof.**

1. Write $\delta = (\mu - m)/s$, $r = R/s$, $E = e^{z^2/(2c)}$ and $u = \delta + r z E$. Up to an additive constant,

   $`\displaystyle U(z) = \tfrac{\nu+1}{2}\log(1 + u^2/\nu) - \tfrac{z^2}{2c} - \log(1 + z^2/c).`$
2. For $\vert z\vert \ge z_0$, with $z_0$ large enough that $\vert rzE\vert > 2\vert\delta\vert$, write $u = r z E(1+\eta)$ with $\eta = \delta/(rzE)$. Then

   $`\displaystyle \log(1+u^2/\nu) = 2\log r + 2\log\vert z\vert + z^2/c + 2\log\vert 1+\eta\vert - \log\nu + \log(1 + \nu/u^2).`$
3. Hence

   $`\displaystyle \tilde U(z) = (\nu+1)\log\vert z\vert - \log(1 + z^2/c) + C_0 + \varepsilon(z),`$

   with $\varepsilon = (\nu+1)\left[\log\vert 1+\eta\vert + \tfrac12\log(1+\nu/u^2)\right]$.
4. Each of $\eta$ and $1/u^2$ is a rational function of $z$ times $E^{-1}$ or $E^{-2}$, and so are their first two derivatives. So $\varepsilon$, $\varepsilon'$ and $\varepsilon''$ tend to 0, and the explicit terms have derivatives $O(1/\vert z\vert)$ and $O(1/z^2)$. Therefore $\tilde U'$ and $\tilde U''$ tend to 0 at $\pm\infty$.
5. On $[-z_0, z_0]$, $\tilde U$ is $C^\infty$, because $g$ is a $C^\infty$ diffeomorphism with $g' > 0$ and the *t* log-density is $C^\infty$. A continuous function with finite limits at both ends is bounded, so $\tilde U'$ and $\tilde U''$ are bounded.
6. Finally, $\vert\tilde U(z)\vert \le \vert\tilde U(0)\vert + \sup\vert\tilde U'\vert \vert z\vert$. ∎

Numerically, $\sup\vert\tilde U'\vert$ ranges from 0.10 to 13.2 and $\sup\vert\tilde U''\vert$ from 0.5 to 111 over six parameter sets. These include location and scale errors, $\nu \in \lbrace 0.5, 1, 2, 3, 10 \rbrace$ and $c \in \lbrace 0.5, 1, 2, 4, 10 \rbrace$. In every set $U''$ in the far tail (outer 20% of the grid) is within 0.5% of $\nu/c$ (A4). Analytic derivatives were cross-checked by finite differences.

**Robustness to the tail index.** The proposition holds for every $c > 0$, whatever the true $\nu$. Choosing $c$ from an estimate $\hat\nu$ (for example $c = \hat\nu$) gives a $z$-space tail variance of $\hat\nu/\nu$. A wrong tail index rescales the Gaussian tail and never brings a heavy tail back.

**Proposition A.7 (general regularly varying marginals; Sketch).** Suppose that for $x \to \pm\infty$

```math
x V'(x) = \alpha_\pm + 1 + \epsilon_1(x), \qquad x^2 V''(x) = -(\alpha_\pm + 1) + \epsilon_2(x), \qquad \epsilon_1, \epsilon_2 = O(1/\log\vert x\vert).
```

This is second-order smooth regular variation at a logarithmic rate. It covers Student-*t*, Pareto-type and log-gamma-type tails, but not all regularly varying laws. Then the conclusion of A.6 holds with $\kappa_\pm = \alpha_\pm/c$; the two tails may have different curvatures, so the perturbation absorbs the difference.

**Sketch.** $U' = (gV'(g))(g'/g) - (\log g')'$ with $g'/g = z/c + 1/z + o(1)$. Since $\log g \sim z^2/(2c)$, the error $\epsilon_1(g) z/c = O(1/z)$, and the second derivative is handled similarly. A rate $O((\log x)^{-\delta})$ still gives the weaker A4 bound $\vert\nabla\tilde U\vert \le A(1 + \vert z\vert^{\varrho - 1})$ with $\varrho = 2 - 2\delta$, provided $\varrho \ge 1$.

---

## B. Impossibility of a common map

### B.1 Gaussian versus Cauchy, quantitative (Lean)

Let `g` be differentiable with $g' > 0$ and $g(z) \to \infty$. Write

- $U_G(z) = g(z)^2/2 - \log g'(z)$ for the Gaussian pullback potential, and
- $U_C(z) = \log(1 + g(z)^2) - \log g'(z)$ for the Cauchy pullback potential,

both up to constants.

**Jacobian cancellation** (`pullback_potential_difference`): $U_G - U_C = g^2/2 - \log(1+g^2)$, whatever `g` is.

**Theorem B.1.** Suppose $U_C(z) \ge a z - b$ on $[0, \infty)$ with $a > 0$. Then:

1. **Exponentially small tail.** $\pi/2 - \arctan g(z) \le (e^{b}/a) e^{-az}$, so the Cauchy pullback has an exponentially small tail mass. (`cauchy_pullback_tail_bound`)
2. **Exponential growth of the map.** $g(z) \ge (a/(2e^{b})) e^{az}$ wherever $g(z) \ge 1$. (`exp_growth_of_cauchy_pullback_envelope`)
3. **Super-quadratic Gaussian potential.** $U_G(z) \ge \tfrac14 (a/(2e^{b}))^2 e^{2az} - b - \log 4$ wherever $g(z) \ge 1$. (`gaussPullback_lower_of_cauchy_envelope`)
4. **Conclusion.** No such `g` also satisfies $U_G(z) \le A z^2 + B$ on $[0,\infty)$. (`no_common_map_gaussian_cauchy`)

Arcsinh is the boundary instance: $g = \sinh$ satisfies the hypothesis with $a = 1$, $b = \log 2$ (`arcsinh_cauchy_pullback_linear_lower`), so its Gaussian pullback admits no quadratic bound (`arcsinh_gauss_pullback_not_quadratic`).

**Proof.**

- $\Phi(z) = \arctan g(z) + (e^b/a)e^{-az}$ has $\Phi' \le 0$ on $[0,\infty)$ by the hypothesis, and $\Phi \to \pi/2$. So $\Phi \ge \pi/2$.
- Then use $\arctan(1/y) = \pi/2 - \arctan y$ and $\arctan t \ge t/(1+t^2)$.
- Then use $\log(1+y^2) \le y^2/4 + \log 4 - 3/4$.
- Finally $e^{2az} \ge (2az)^3/6$ beats any quadratic. ∎

### B.2 General light versus polynomial marginals (Proved)

**Theorem B.2.** Let $V_1(x) \ge \kappa x^2 - C_1$ (sub-Gaussian potential) and $V_2(x) \le (\alpha+1)\log(1+\vert x\vert) + C_2$ (tail at least as heavy as polynomial index $\alpha$), for $x \ge x_0$. Let `g` be $C^1$ with $g' > 0$ and $g \to \infty$, and let $U_i = V_i\circ g - \log g'$. If $U_2(z) \ge az - b$ on $[0,\infty)$, then for $z$ large:

```math
g(z) \ge c e^{az/\alpha}, \qquad U_1(z) \ge c' e^{2az/\alpha} - C'.
```

**Proof.**

- For $G = 1 + g$, the hypothesis gives $G' \le e^{b + C_2 - az} G^{\alpha+1}$.
- So $(G^{-\alpha})' \ge -\alpha e^{b+C_2} e^{-az}$. Integrating over $[z, \infty)$, using $G \to \infty$, gives $G(z)^{-\alpha} \le (\alpha e^{b+C_2}/a) e^{-az}$.
- Then $U_1 = V_1(g) - \log g' \ge \kappa g^2 - C_1 - (b + C_2 - az) - (\alpha+1)\log(1+g) \ge \tfrac{\kappa}{2} g^2 - C'$. ∎

**Sharpness.** The exponent $2a/\alpha$ cannot be improved. For $g = R\sinh(\lambda z)$ and a Student-*t* marginal with index $\alpha$, the *t* pullback has slope exactly $a = \alpha\lambda$, and the Gaussian pullback grows at rate exactly $2\lambda = 2a/\alpha$. This was confirmed for $\alpha \in \lbrace 0.5, 1, 3\rbrace$ and $\lambda \in \lbrace 0.5, 1, 2\rbrace$ (`check_impossibility.py`, B1), together with the Lean constants for $g = \sinh$ (B2).

**Per-coordinate calibration does not help.** Suppose the two coordinates use $\mu_1 + R_1 g$ and $\mu_2 + R_2 g$. The second is an affine function of the first, and location-scale images of $V_1$ and $V_2$ satisfy the same hypotheses with new constants. Only the shape of the map matters.

### B.3 Tail-level characterisation (Proved; the algebraic core is Lean `window_necessity`)

Let $F_1$ and $F_2$ have positive continuous densities, so that their right hazards $\psi_1$ and $\psi_2$ are continuous, strictly increasing and unbounded. For $0 < \beta_1 \le \beta_2$, let $\mathcal{W}[\beta_1,\beta_2]$ be the set of hazards with $a t^{\beta_1} - b \le \psi(t) \le A t^{\beta_2} + B$ on $t \ge 0$.

**Theorem B.3.** The following are equivalent:

1. There is an increasing homeomorphism `g` of $\mathbb{R}$ with $\psi_1\circ g$ and $\psi_2\circ g$ both in $\mathcal{W}[\beta_1,\beta_2]$.
2. There is a $C$ such that for all large `x`:

   $`\displaystyle \psi_1(x) \le C\psi_2(x)^{\beta_2/\beta_1} + C \quad\text{and}\quad \psi_2(x) \le C\psi_1(x)^{\beta_2/\beta_1} + C.`$

For the HMC/NUTS window, $\beta_2/\beta_1 = 2$: two marginals can share a map iff their hazards are within a square of each other.

**Proof.**

- **(1 ⇒ 2).** At $x = g(t)$, combine $\psi_1(x) \le At^{\beta_2} + B$ with $t \le ((\psi_2(x)+b)/a)^{1/\beta_1}$, and then absorb constants once $\psi_2 \ge 1$. The case $\beta_1 = 1$, $\beta_2 = 2$ is `window_necessity`.
- **(2 ⇒ 1).** Let $\phi = \max(\psi_1,\psi_2)$ and $g(t) = \phi^{-1}(t^{\beta_2})$ for large $t$, extended increasingly to all of $\mathbb{R}$. Then $\psi_i(g(t)) \le t^{\beta_2}$. By (2), $\min(\psi_1,\psi_2) \ge ((\phi - C)/C)^{\beta_1/\beta_2}$, so $\psi_i(g(t)) \ge (2C)^{-\beta_1/\beta_2} t^{\beta_1}$ once $t^{\beta_2} \ge 2C$. Left tails are handled the same way, independently. ∎

**Examples** (`check_impossibility.py`, B3, running maximum of $\psi_1/\psi_2^2$):

| pair | common map? | witness |
|---|---|---|
| Gaussian and Laplace | yes | ratio stays at 0.5; identity, or the explicit $\phi^{-1}(t^2)$ checked numerically |
| Gaussian and Weibull $k = 3/2$ | yes | ratio 0.226 |
| lognormal and Cauchy | yes (arcsinh) | ratio below 0.5 up to $x = 10^{100}$ |
| Gaussian and Cauchy | no | ratio $7.7\times10^{194}$ by $x = 10^{100}$ |
| Gaussian and Weibull $k = 1/2$ | no | ratio grows like $x$ |
| Cauchy and log-Cauchy | no | ratio grows like $\log x/(\log\log x)^2$ |

**Caveat.** Theorem B.3 is a statement about tails, not potentials. Its necessity direction does transfer to potentials, because potential envelopes imply hazard envelopes (A.1). Its sufficiency direction would need a smooth `g` with regular pullbacks, which is not proved here (Open problem 3).

### B.4 Corollary: selection is necessary for gradient-based samplers (Proved, from B.1, B.2 and A.1)

Take any fixed map shape `g`, with arbitrary per-coordinate location $\mu_i$ and scale $R_i$. Take the product of a Gaussian coordinate and a Student-*t* coordinate, the simplest heterogeneous target. Then the pullback does not satisfy the hypotheses of any of these:

- Livingstone et al. A2 or Cor. 2.3(i);
- Durmus–Moulines–Saksman A2(m)+A3(m) or A4;
- Durmus et al. H6(m) or H7;
- Gruffaz et al. H3(m) or H4.

Restricted to the coordinate axes, each of these hypotheses implies the window of Section 0. Theorem B.2 shows that no shape gives both coordinates the window. A universal coordinate-wise transform for HMC or NUTS therefore has to choose its shape per coordinate, which requires a data-dependent selection or calibration step beyond location and scale. This gives theoretical backing to the "coordinate-wise, learnable transformations" open problem of Power & Vasdekis (arXiv:2511.21563, §4.3(iii)).

**Remark (random-walk Metropolis is different).** RWM has no upper limit on tail lightness. Johnson & Geyer (2012, §4) note that their transformation "will do no harm" on super-exponentially light targets. Concluding that one exponential-type map serves RWM for all tail classes is our inference, not their claim; they also write that users "must understand the tail behavior … to select the correct transformation". The upper edge of the window, and with it Theorem B, is specific to gradient-based samplers. Brešar & Mijatović (arXiv:2608.04284, §3.3) say informally that the Johnson–Geyer route fails for ULA, MALA and HMC when the transformation makes tails lighter than Gaussian.

---

## C. Positive result for a tail-matched family

### C.1 Product targets with affine and h-maps (Proved, modulo the cited GE theorems)

**Theorem C.1.** Let $\pi(x) = \prod_{i=1}^d f_i(x_i)$, where each coordinate is one of:

- **light:** $f_i = N(m_i, \sigma_i^2)$, pulled back by $x_i = \mu_i + R_i z_i$;
- **heavy:** $f_i = t_{\nu_i}(m_i, s_i)$, pulled back by $x_i = \mu_i + R_i z_i e^{z_i^2/(2c_i)}$.

Here $\mu_i \in \mathbb{R}$, $R_i > 0$ and $c_i > 0$ are arbitrary, and the momentum may use any fixed positive-definite mass matrix. Then the pullback potential satisfies:

- H1: $C^2$ with globally Lipschitz gradient;
- A4 / H7 / H8 / H4 (the same "Gaussian plus perturbation" condition, under different names in the four papers): $\Sigma = \mathrm{diag}(\kappa_i)$, with $\kappa_i = R_i^2/\sigma_i^2$ for light coordinates and $\kappa_i = \nu_i/c_i$ for heavy ones, and $\varrho = 1$.

Consequently:

1. **Fixed-length HMC** is $V_a$-uniformly GE for every $a > 0$, $T \in \mathbb{N}$ and $h \in (0, \bar S/T]$, where $V_a(z) = e^{a\Vert z\Vert}$. Source: Durmus, Moulines & Saksman, *Ann. Statist.* 48(6), 2020, arXiv:1705.00166, Thm 9(c). Theorem 10(c) covers a random number of steps.
2. **NUTS with multinomial index selection** (Stan, BlackJAX) is $V_a$-uniformly GE when $h 2^{K_m} \le \bar S$. Source: Gruffaz et al. 2026, arXiv:2603.18640, Thm 19(b); irreducibility needs only H1 (their Thm 17).
3. **NUTS with biased progressive sampling** is GE under the same restriction, provided H2(h, $K_m$) holds. Source: Durmus et al. 2023, arXiv:2307.03460, Thm 16(b) with H5 from Thm 8.
4. **Fixed-length HMC is ergodic** for Lebesgue-almost every `h` and every `T`. Source: Durmus et al. 2023, Thm 17.

**Proof.**

- $U(z) = \sum_i U_i(z_i)$, and the perturbation is $\tilde U = \sum_i\tilde U_i(z_i)$.
- **Light coordinates.** $U_i$ is quadratic plus linear, so $\tilde U_i'$ is constant.
- **Heavy coordinates.** Proposition A.6 applies.
- **Combining.** Then $\vert\nabla\tilde U\vert \le (\sum_i\sup\vert\tilde U_i'\vert^2)^{1/2}$, the Hessian of $\tilde U$ is diagonal and bounded, and $\vert\tilde U(z)\vert \le \sum_i A_i(1+\vert z_i\vert) \le A(1 + \Vert z\Vert)$.
- **Mass matrix.** A mass matrix is equivalent to a linear change of variables $z = Lw$, which keeps the class: $\Sigma \mapsto L^\top\Sigma L$, and $\tilde U\circ L$ still has a bounded, Lipschitz gradient. ∎

GE in `z` is the same as GE in `x`: the two chains are conjugate through the bijection `g`, so total-variation distances agree. The Lyapunov function becomes $V_a(g^{-1}(x))$, which on heavy coordinates grows only like $e^{a\sqrt{2c\log\vert x\vert}}$.

**What is not claimed.**

- The constants $\bar S$ are not explicit.
- The NUTS results need the maximum trajectory length $h2^{K_m}$ to be small, while practical NUTS uses $K_m = 10$.
- Warm-up adaptation is excluded. The theorems apply to the frozen post-warm-up kernel, as in the repo's warm-up contract.

### C.2 What misclassification costs

| true coordinate | map applied | pullback | consequence |
|---|---|---|---|
| Student-*t*, any $\nu$ | h-map, any $c$ | Gaussian (variance $c/\nu$) plus perturbation | in the C.1 class |
| Student-*t* | affine, or any map with $\vert g(z)\vert \le C(1+\vert z\vert)^p$ | polynomial tail; bounded gradient for affine and power maps | HMC and NUTS not GE when the gradient is bounded (Prop D.2) |
| Gaussian | affine | Gaussian | in the C.1 class |
| Gaussian | h-map, $c < \infty$ | $U \sim r^2 z^2 e^{z^2/c}/2$ | HMC not GE (Prop D.1); unstable mass $e^{-\Theta(c/\varepsilon^2)}$ (E.2) |
| Gaussian | arcsinh | $U \sim r^2 e^{2\vert z\vert}/8$ | HMC not GE (Prop D.1); unstable mass $\approx 2\Phi(-\sqrt2/\varepsilon)$ (E.1) |
| lognormal | h-map | $\psi_Z \sim t^4/(8c^2)$ ($\beta = 4$) | outside the window (needs arcsinh) |
| Laplace | h-map | super-Gaussian | outside the window (needs the affine map) |

### C.3 nutpie's flow layers cannot fix polynomial tails (Proved for any polynomial-growth map; code reading for nutpie)

If $\Vert T(w)\Vert \le C(1 + \Vert w\Vert)^p$, then $P(\Vert W\Vert > t) \ge P(\Vert X\Vert > C(1+t)^p)$. A polynomially tailed $X$ therefore keeps a polynomial tail in the latent space, and $E e^{s\Vert W\Vert} = \infty$ for every $s$.

In the nutpie 0.16.11 source, the per-coordinate layer `Contract2` is

```math
T(x) = \sigma'\frac{\delta^2 \zeta^\gamma - \delta^{-2}\zeta^{-\gamma}}{\gamma} + \mu, \qquad \zeta = x/2 + \sqrt{1 + x^2/4},
```

which grows like $\vert x\vert^\gamma$. A finite composition of such layers, together with the affine, Householder and planar layers, grows polynomially. So by the statement above, nutpie's normalizing-flow adaptation cannot give a Student-*t* coordinate exponential tails, and its latent-space NUTS faces the heavy-tail obstruction of Prop D.2 whenever the gradient stays bounded. This is a code reading; we did not run nutpie. It does not affect nutpie's other strengths, such as correlations and funnels.

### C.4 Selecting the class per coordinate (cited theory; Numerical)

**What is known.**

- The Hill estimator $\hat\gamma_k$ is weakly consistent for $\gamma = 1/\alpha > 0$ under regular variation for any $k \to \infty$ with $k/n \to 0$ (Mason 1982, *Ann. Probab.* 10(3)).
- For any distribution with a positive right endpoint it converges to $\gamma_+ = \max(\gamma, 0)$ (Dekkers, Einmahl & de Haan 1989, *Ann. Statist.* 17(4), eq. (2.13) in the proof of Thm 2.1). Gaussian, Laplace and lognormal all have $\gamma = 0$.
- So the rule "finite $c$ iff $\hat\gamma_k > \tau$" is consistent for every fixed $\tau$ that separates 0 from the smallest heavy $\gamma$ present.
- Consistency extends to stationary dependent data: Hsing 1991, *Ann. Statist.* 19(3); Resnick & Stărică 1998, *Ann. Appl. Probab.* 8(4).

**What the numbers say.** We used 400 replicates of i.i.d. draws (`check_selection.py`). Convergence for $\gamma = 0$ is logarithmically slow:

| n | k | Gaussian | Laplace | lognormal | $t_{10}$ ($\gamma=0.1$) | $t_5$ (0.2) | $t_2$ (0.5) | Cauchy (1) |
|---|---|---|---|---|---|---|---|---|
| 2,000 | 44 | 0.13 | 0.21 | 0.42 | 0.19 | 0.25 | 0.51 | 1.01 |
| 2,000 | 200 | 0.21 | 0.33 | 0.59 | 0.26 | 0.32 | 0.54 | 1.00 |
| 20,000 | 141 | 0.10 | 0.17 | 0.35 | 0.16 | 0.23 | 0.50 | 1.01 |

The table gives the mean of $\hat\gamma$ on $\vert x - \mathrm{median}\vert$.

- **Two-way choice.** Take $k = \sqrt n$ and $\tau = 0.4$:
  - Gaussian, Laplace, $t_5$ and $t_{10}$ are labelled light in at least 98% of replicates at every $n$ tested.
  - $t_2$ is labelled heavy in 86%, 93% and 100% of replicates at $n = 500$, 2,000 and 20,000.
  - Cauchy is always labelled heavy.
- **The lognormal problem.** With $k = \sqrt n$, lognormal coordinates are labelled heavy in 79% and 62% of replicates at $n = 500$ and 2,000, and in 4% at $n = 20{,}000$. With $k = n/10$ they are labelled heavy in every replicate. The h-map would then put them at $\beta = 4$ (C.2), where arcsinh would have been right.
- **Laplace versus $t_5$.** The mean of $\hat\gamma$ is nearly the same for the two in every setting (0.17–0.33 for Laplace, 0.23–0.32 for $t_5$). At $n \le 2{,}000$ no threshold separates them. At $n = 20{,}000$ with $k = \sqrt n$, $\tau = 0.2$ does (2% against 94%), but it then labels $t_{10}$ light.
- **MCMC draws.** Warm-up draws are dependent and non-stationary, and a heavy-tailed chain under-explores its tails. So $\hat\gamma$ from warm-up is biased toward light.

**Related practice.**

- Pachebat (arXiv:2605.20068) applies a Hill threshold to choose a per-coordinate signed log for flow matching.
- Hickling & Prangle's TTFfix (ICML 2025) sets a per-marginal tail parameter with Hill's double bootstrap.

We found no use inside MCMC.

---

## D. Implications for HMC and NUTS

### D.1 A light coordinate with a super-linear pullback gradient breaks fixed-length HMC (Sketch)

**Proposition D.1.** Let $U(z) = \sum_i U_i(z_i)$ with a diagonal mass matrix. Suppose $U_1$ satisfies conditions (17), (18) and (19) of Livingstone et al. (2019, Thm 5.14), with (18) as corrected in their 2026 corrigendum ($\vert y\vert \ge 3\vert x\vert$). Then HMC with step size $\varepsilon$ and $L$ leapfrog steps is not GE, whether $L$ is fixed or drawn independently of the state.

**Sketch.**

1. **The leapfrog factorises.** For product targets with diagonal mass, the leapfrog map acts coordinate-wise, and the energy error is $\Delta H = \sum_i\Delta H_i$.
2. **The first coordinate.** Their Lemma 5.17 (corrected) gives $\Delta H_1 \to +\infty$ uniformly over $\vert p_1\vert \le \vert z_1\vert^{\delta}$ as $\vert z_1\vert \to \infty$.
3. **The other coordinates.** For $z_{2:d}$ in a compact set $K$, the errors $\Delta H_{i\ge2}$ are continuous functions of $(z_i, p_i)$, hence tight.
4. **Rejection.** So the acceptance probability tends to 0 uniformly on $\lbrace z_1 > t\rbrace\times K$. This set has positive $\pi$-mass, so the essential supremum of the rejection probability is 1.
5. **Conclusion.** By Roberts & Tweedie (1996), Thm 5.1 (Livingstone et al. Prop 3.5), the chain is not GE. Random $L$ is handled by dominated convergence, as in the corrigendum.

**Instances.**

- **Arcsinh pullback of $N(m, \sigma^2)$, any $\mu$ and $R$.** $U_1 = (\mu - m + R\sinh z)^2/(2\sigma^2) - \log\cosh z$ has $\vert U_1'\vert \sim (R^2/4\sigma^2)e^{2\vert z\vert}$. Condition (17) is immediate. For (18), $\vert U'(y)\vert/\vert U'(x)\vert \approx e^{2(\vert y\vert - \vert x\vert)} \ge e^{4\vert x\vert}$. Condition (19) holds because $U(y) \gtrsim e^{2\vert y\vert}$.
- **h-map pullback of a Gaussian, any finite $c$.** Here $U_1' \sim r^2 z e^{z^2/c}(1+z^2/c)$, and the same checks apply.

`check_hmc_acceptance.py` illustrates the mechanism. With $\varepsilon = 0.2$ and $L = 10$, mean acceptance for the arcsinh pullback of $N(0,1)$ with $r = 1.5$ falls from 0.98 at $z_0 = 1$ to 0.82 at $z_0 = 2$ and 0.000 at $z_0 \ge 3$. For the arcsinh pullback of Cauchy it stays at or above 0.98 out to $z_0 = 8$. This is an illustration, not evidence for the theorem.

### D.2 A heavy coordinate with bounded pullback gradient breaks HMC and NUTS (Sketch)

**Proposition D.2.** Let $U = \sum_i U_i$, and suppose $\sup\vert U_1'\vert < \infty$ and $\int e^{s\vert z_1\vert}\pi(dz) = \infty$ for every $s > 0$. Assume the chain is irreducible and aperiodic with compact sets small, which holds under H1 (Durmus et al. 2023, Thm 8; Gruffaz et al. 2026, Thm 17). Then neither fixed-length HMC nor NUTS with maximum depth $K_m$ is GE.

**Sketch.** This is the proof of Jarner & Tweedie (2003, Thm 2.2), applied to the coordinate projection $z \mapsto z_1$ instead of $\Vert z\Vert$.

1. **Bounded increments.** In one transition, $z_1$ moves by at most $N\varepsilon(\vert p_1\vert + N\varepsilon\sup\vert U_1'\vert)$, with $N = L$ for HMC and $N = 2^{K_m}$ for NUTS, since the selected point lies on the trajectory. These increments are uniformly small in probability.
2. **Geometric return times.** GE gives a compact small set $C$ and $\kappa > 1$ with $\pi(E_\cdot\kappa^{\sigma_C}) < \infty$.
3. **Exponential moments.** Leaving $\lbrace\vert z_1\vert > r_C + n\delta\rbrace$ takes at least $n$ steps with probability at least $(1-\eta)^n$. So $E_z\kappa^{\sigma_C} \ge e^{s\vert z_1\vert}$, which contradicts $\pi(e^{s\vert z_1\vert}) = \infty$.

**Instances.**

- A Student-*t* coordinate left in `x`-space.
- A Student-*t* coordinate under any polynomial-growth map: the pullback density is $\sim\vert z\vert^{-p\nu-1}$ and its gradient $\sim (p\nu+1)/z$.

In one dimension, the two cases are Livingstone et al. Thm 5.18 for HMC and Gruffaz et al. Thm 13 with H2-(i) for NUTS.

### D.3 What is proved, by coordinate type and map

GE of a product target holds when every coordinate falls in a GE row, and fails when any coordinate falls in a "not GE" row.

| coordinate | map | fixed-length HMC | NUTS |
|---|---|---|---|
| Gaussian | affine | GE (C.1; DMS Thm 9(c)) | GE (C.1; Gruffaz Thm 19(b)) |
| Gaussian | arcsinh | **not GE** (D.1) | open; conjectured not GE (no NUTS light-tail result beyond exact $C\vert q\vert^\beta$) |
| Gaussian | h-map, $c<\infty$ | **not GE** (D.1) | open; conjectured not GE |
| Student-*t* | affine or power | **not GE** (D.2) | **not GE** (D.2) |
| Student-*t* | arcsinh | open in this form: Livingstone Cor. 2.3 covers only exact $e^{-\alpha\vert z\vert}$ tails in 1-D; conjectured GE | open: every NUTS theorem needs $m > 1$ |
| Student-*t* | h-map, any $c$ | GE (C.1) | GE (C.1) |

### D.4 Caveats about the cited theorems

1. **Gruffaz et al. H2-(i).** It is printed as "there exists $b$ with $E e^{b\vert X\vert} = \infty$". Their Remark 12 says it covers $\beta \in (0,1]$, but the Jarner–Tweedie argument behind it needs $E e^{b\vert X\vert} = \infty$ for every $b > 0$, as in Livingstone et al. Thm 2.2(ii). We use it only in that form. Under that form it says nothing about the Laplace edge $\beta = 1$.
2. **The original Livingstone et al. Thm 5.14 had a gap.** Gruffaz et al. App. B.1 found it, and the 2026 corrigendum fixes condition (18). We cite the corrected version.
3. **Laplace edge.** Durmus, Moulines & Saksman's Prop. 6 allows $m \in [1,2]$, but their Props 5 and 7 and Thms 9–10 need $m \in (1,2]$. The Laplace edge $m = 1$ is therefore uncovered for HMC with general potentials as well as for NUTS.

### D.5 Trajectory cost: fixed-time HMC is not NUTS (Heuristic)

Geometric ergodicity says nothing about cost per transition. The state assessment predicted that the time to U-turn grows like $\sqrt{E}$ under arcsinh. That holds for the full orbit of the dynamics, but **it is not the right prediction for NUTS**, so we do not state a $\sqrt E$ cost law.

**Fixed integration time.** On a tail $U \sim \vert z\vert^\beta$, the orbit at energy $H$ has half-period proportional to $H^{(2-\beta)/(2\beta)}$. This is the scaling Betancourt (2017, arXiv:1701.02434, §4.3; quoted, not re-derived here) gives for the optimal integration time of HMC.

- It grows like $\sqrt H$ at $\beta = 1$ (arcsinh on polynomial tails).
- It is constant at $\beta = 2$ (h-map).

**NUTS.** In one dimension the U-turn criterion fires at the first momentum reversal. Starting from $z_0$ with momentum $p$, that takes time about $\vert p\vert/\vert U'(z_0)\vert$:

- **Laplace tail** (arcsinh on Student-*t*): constant, since $\vert U'\vert \to \nu$.
- **Gaussian tail** (h-map): at most about $\pi/(2\sqrt\kappa)$.
- **Cauchy in $x$-space**: proportional to $\vert x_0\vert$, that is, to $e^{E/2}$.

**Measurements (not reproduced here).** The Tukey-h exploration branch measured mean leapfrog steps per NUTS transition on 1-D Cauchy, with starting $\vert x_0\vert$ up to $10^6$:

- stock: growth like $e^{E/2}$ (fitted slope 0.47);
- arcsinh: nearly flat (log-log slope 0.21);
- h-map: flat.

In a 10-dimensional product of Cauchy coordinates, arcsinh rises from 17 to 28 steps between the bulk and the top 0.1% energy bin, while the h-map stays near 12. We have not reproduced these numbers. A quantitative NUTS cost theory remains open (Open problem 9).

---

## E. Practical size of the light-tail damage

### E.1 Arcsinh on a Gaussian coordinate (Lean bracket; Numerical mass)

Let $x \sim N(m,\sigma^2)$, $\mu = m$ and $r = R/\sigma$. Then $U(z) = r^2\sinh^2 z/2 - \log\cosh z$, and in standardised units $s = r\sinh z = (x-m)/\sigma$:

```math
U''(z) = r^2\cosh 2z - \mathop{\mathrm{sech}}^2 z = r^2 + 2s^2 - \mathop{\mathrm{sech}}^2 z .
```

(`hasDerivAt_arcsinhGaussPotential`, `hasDerivAt_arcsinhGaussGrad`, `arcsinhGaussCurv_eq`)

$U''$ is nondecreasing in $\vert z\vert$ (`arcsinhGaussCurv_mono_abs`), so the set where leapfrog is linearly unstable, $\varepsilon^2U'' \ge 4$, is $\lbrace\vert s\vert \ge s^{\star}\rbrace$. Its threshold is bracketed by

```math
\frac{2}{\varepsilon^2} - \frac{r^2}{2} \;\le\; s^{\star 2} \;\le\; \frac{2}{\varepsilon^2} - \frac{r^2}{2} + \frac12
```

(`leapfrog_stable_of_standardized_le`, `leapfrog_unstable_of_standardized_ge`). The Gaussian mass of the unstable set is $2\Phi(-s^{\star})$, which is approximately $2\Phi(-\sqrt2/\varepsilon)$ for every $r$ to leading order. Increasing $R$ does not shrink it.

Exact masses (`check_stability.py`, E1), all inside the Lean bracket:

| $r$ | $\varepsilon = 1$ | 0.7 | 0.5 | 0.3 | 0.2 |
|---|---|---|---|---|---|
| 1.0 | 0.19 | 0.055 | $6.0\times10^{-3}$ | $3.1\times10^{-6}$ | $2.0\times10^{-12}$ |
| 1.5 (the repo's `scale_multiplier`) | 0.27 | 0.075 | $8.2\times10^{-3}$ | $4.3\times10^{-6}$ | $2.7\times10^{-12}$ |
| 2.0 | 0.50 | 0.12 | $1.3\times10^{-2}$ | $6.6\times10^{-6}$ | $4.2\times10^{-12}$ |

**Reading.** Step-size adaptation on an arcsinh-transformed Gaussian coordinate trades $\varepsilon$ against the Gaussian quantile that remains stable. Keeping the unstable mass below $10^{-5}$ needs $\varepsilon \lesssim 0.3$, against a bulk curvature of $r^2 - 1$. This mechanism fits the benign-target cost reported in the state assessment (§3.1, about 0.85× on a rotated Gaussian), but we have not tested it.

**Bulk shape** (`arcsinhGaussCurv_zero`). $U''(0) = r^2 - 1$, so the pullback of a Gaussian is bimodal iff $R < \sigma$. For Student-*t*, $U''(0) = (\nu+1)r^2/\nu - 1$; for Cauchy this is $2r^2 - 1$ (Numerical, E3). The repo's $R = 1.5\hat\sigma_{\mathrm{IQR}}$ gives $r \approx 1.5$ for a Gaussian and $r \approx 2.2$ for a Cauchy, so neither pullback is bimodal.

### E.2 h-map on a Gaussian coordinate (Numerical)

Here $r = 1$ and $U(z) = z^2e^{z^2/c}/2 - z^2/(2c) - \log(1+z^2/c)$. The Gaussian mass where $\varepsilon^2U''\ge4$ is (`check_stability.py`, E2):

| $c$ | $\varepsilon = 0.7$ | 0.5 | 0.3 |
|---|---|---|---|
| 1 | 0.27 | 0.14 | 0.021 |
| 4 | 0.038 | $4.1\times10^{-3}$ | $4.3\times10^{-6}$ |
| 16 | $4.3\times10^{-5}$ | $1.1\times10^{-8}$ | $4.3\times10^{-20}$ |
| 64 | $3.6\times10^{-16}$ | $4.0\times10^{-30}$ | $3.2\times10^{-75}$ |

The threshold $s^{\star}$ grows like $\sqrt c$; for example $s^{\star} \approx 2.9$, 5.7 and 11.4 at $\varepsilon = 0.5$ for $c = 4$, 16 and 64.

**Design consequence (Conjecture, to be tested).** A single h-map with large $c$ (say $c = 16$) on every coordinate behaves very differently on the two coordinate types:

- **Light coordinates** are not GE in theory (D.1), but the unstable mass is about $10^{-8}$ at $\varepsilon = 0.5$. That is far below arcsinh's $8\times10^{-3}$.
- **Heavy coordinates** are in the C.1 class for every $c$. They behave like the identity for $\vert x\vert \lesssim R\sqrt{c}$ and are Gaussianised beyond.

So an h-map with a large $c$ might look like a safer universal default than arcsinh, with the class decision softened into a single scale $c$. Theorem B shows this cannot be fully universal, and the trade-off is real on the heavy side. On the Tukey-h exploration branch, a fixed large $c$ lowers tail-ESS per gradient on product-*t* targets: for `prod_t1`, $c = 4$ gives 0.012 against 0.060 at $c = 0.25$ (median of 5 seeds, `explore/tukey_h/results/tables.md` on that branch, not reproduced here). Because only $R^2c$ is identifiable, "large $c$" means large $R^2c$ relative to the coordinate's scale.

That branch also reports two light-coordinate results:

- Gaussian coordinates given a finite $c \approx 5$ cost 6–7% efficiency.
- They also show occasional divergences, up to 11 per 4,000 draws.

Both are consistent in order of magnitude with the $c = 4$, $r = 1$ row of the table, about $4\times10^{-3}$ unstable mass at $\varepsilon = 0.5$. So a soft default does not remove the need for selection; it only makes misclassifying a light coordinate cheap.

---

## F. Novelty verdicts

| result | status | verdict | closest prior work |
|---|---|---|---|
| A.1 tail transfer | Proved | **known**, elementary | implicit in Johnson & Geyer 2012 and in Jaini et al. 2020 (ICML) |
| A.2 window table | Proved / Numerical | **probably new as an explicit table**; individual entries known | Power & Vasdekis Ex. 4.3 (log map: Cauchy to Laplace); Goerg 2015 (Lambert W × F); Livingstone et al. §7.2 ($\mathrm{sgn}(x)\vert x\vert^{\beta/2}$ Gaussianises $E(\beta,\alpha)$) |
| A.3 Cauchy–arcsinh sech law | Lean | **known** (hyperbolic secant arises from Cauchy, e.g. via $\log\vert X\vert$) | — |
| A.3 calibration-robust envelope, sharp $K^{\star}$ | Lean / Proved | **probably new**, minor | — |
| A.5 explicit Gaussian envelopes, h-map Cauchy | Lean | **probably new**, minor | — |
| A.5 only $R^2c$ identifiable | Lean | **elementary**; first noted on the Tukey-h branch | — |
| A.6 h-map Student-*t* lies in the A4/H4 class for every $c$ | Proved | **probably new** as a verified statement; the mechanism ($\exp(r^2)$-type maps Gaussianise polynomial tails) is known | He, Balasubramanian & Erdogdu (IEEE TIT 70(1), 2024): radial $e^{b r^\beta}$, $\beta \in (1,2]$, for ULA; Hickling & Prangle TTF (ICML 2025): $e^{\lambda z^2/2}$-order flows for density estimation |
| B.1/B.2 no common map, sharp exponent $2a/\alpha$ | Lean / Proved | **probably new** as a theorem; informal antecedents only | Johnson & Geyer 2012 §4; Brešar & Mijatović 2026 §3.3; Power & Vasdekis §4.2.1 |
| B.3 within-a-square criterion | Proved | **probably new** | none found |
| B.4 selection is necessary, even with per-coordinate $(\mu,R)$ | Proved | **probably new** | backs Power & Vasdekis §4.3(iii) |
| C.1 product HMC/NUTS GE via tail-matched maps | Proved, modulo cited theorems | **new as an assembled result**; all GE machinery is cited | DMS 2020; Durmus et al. 2023; Gruffaz et al. 2026 |
| C.3 polynomial-growth flows keep polynomial tails | Proved | **known** in spirit | Jaini et al. 2020 (Lipschitz flows preserve tail class) |
| C.4 Hill selection | cited / Numerical | **known** theory; the finite-sample warning is standard extreme-value-theory folklore | Mason 1982; Dekkers, Einmahl & de Haan 1989 |
| D.1/D.2 product extensions | Sketch | **probably known to specialists**; not found stated for product targets or NUTS (D.2) | Livingstone et al. 2019; Jarner & Tweedie 2003 |
| E.1 unstable mass $\approx 2\Phi(-\sqrt2/\varepsilon)$, independent of $R$ | Lean bracket / Numerical | **probably new**, minor | — |
| E.2 h-map unstable mass decays in $c$ | Numerical | **probably new**, minor | — |

**Dropped or downgraded.**

- **"Arcsinh puts heavy coordinates where Livingstone et al. prove GE."** Cor. 2.3 covers only exact Laplace tails in one dimension. The actual pullback is a perturbation of that, and no NUTS theorem covers $m = 1$. Downgraded to "open; conjectured GE" (D.3).
- **"Hill-based selection is reliable."** It is only asymptotically consistent. At $n \le 2{,}000$ it cannot tell Laplace from $t_5$, and it often labels lognormal heavy (C.4).
- **"NUTS cost grows like $\sqrt E$ under arcsinh"** (Track 2.4 of the assessment). Dropped. It describes fixed-time HMC orbits, not NUTS, whose U-turn time is about $\vert p\vert/\vert U'\vert$ (D.5).
- **NUTS non-GE for light-tail pullbacks.** Not provable with existing tools, which cover only exact power potentials; left as a conjecture.

---

## G. Machine-checked content

The new content is in `formal/McmcRobustFormal.lean`, sections §9–§12: 39 new theorems, bringing the file from 37 to 76, each in the `#guard_msgs` axiom audit. They depend only on `propext`, `Classical.choice` and `Quot.sound`. `lake build` reports zero errors and zero warnings, `#lint` passes, and there is no `sorry` or `admit`.

| section | theorems |
|---|---|
| §9 arcsinh–Cauchy envelopes | `one_add_sinh_sq`, `arcsinhCauchyPullback_eq_sech`, `half_exp_abs_le_cosh`, `cosh_le_exp_abs`, `sech_envelope`, `calibration_kernel_upper`, `calibration_kernel_lower`, `arcsinhCauchyPullback_envelope`, `arcsinhCauchyPullback_exp_envelope` |
| §10 h-map | `hasDerivAt_hMap`, `hMapDeriv_pos`, `log_hMapDeriv`, `hMap_strictMono`, `hMap_surjective`, `hMap_bijective`, `hMap_rescale`, `hMap_rescale_invariant`, `hMapCauchyPullback_eq`, `hMapCauchyPullback_gauss_lower`, `hMapCauchyPullback_gauss_upper` |
| §11 impossibility | `pullback_potential_difference`, `deriv_bound_of_cauchy_envelope`, `div_one_add_sq_le_arctan`, `cauchy_pullback_tail_bound`, `exp_growth_of_cauchy_pullback_envelope`, `gaussPullback_lower_of_cauchy_envelope`, `no_common_map_gaussian_cauchy`, `arcsinh_cauchy_pullback_linear_lower`, `arcsinh_gauss_pullback_not_quadratic`, `window_necessity` |
| §12 leapfrog stability | `arcsinhGaussPotential_one_eq`, `hasDerivAt_arcsinhGaussPotential`, `hasDerivAt_arcsinhGaussGrad`, `arcsinhGaussCurv_eq`, `inv_cosh_sq_pos_le_one`, `leapfrog_stable_of_standardized_le`, `leapfrog_unstable_of_standardized_ge`, `arcsinhGaussCurv_mono_abs`, `arcsinhGaussCurv_zero` |

**Not formalised.**

- The general tail index $\alpha$ in B.2 and A.4; this needs `rpow` calculus, which is routine.
- The asymptotic analysis behind A.6, which needs a substantial asymptotics development.
- Theorem B.3 beyond its algebraic core.
- Everything about Markov chains (C.1, D.1, D.2). Mathlib has no geometric-ergodicity theory.

## H. Numerical evidence

All scripts use numpy and scipy only and are seeded. Run them with `uv run python explore/theory/<script>`. If `uv` cannot resolve the project, the synced `.venv/bin/python` works too.

| script | checks | result |
|---|---|---|
| `check_pullback_envelopes.py` | A.3 (sharp and Lean constants), A.4 slopes, A.5 envelopes, A.6 perturbation bounds and finite differences, A.2 growth exponents | 31/31 PASS |
| `check_impossibility.py` | B.2 sharp exponent, B.1 Lean constants for sinh, B.3 square criterion on six pairs, constructive common map | 18/18 PASS |
| `check_stability.py` | E.1 masses inside the Lean bracket for 15 $(r,\varepsilon)$ pairs, E.2 table, E.3 $U''(0)$ by finite differences | 17/17 PASS |
| `check_selection.py` | C.4 Hill table (400 replicates per cell) | table |
| `check_hmc_acceptance.py` | D.1 illustration | table |

## I. Open problems

1. **NUTS at the Laplace edge.** Prove (or refute) GE of HMC and NUTS for $U = a\vert z\vert + \tilde U$ with $\tilde U', \tilde U'' \to 0$. This would settle the arcsinh rows of D.3. Livingstone et al.'s $\beta = 1$ argument (the proposal becomes a random walk with inward drift, and acceptance tends to 1) looks extendable to HMC. NUTS needs control of the U-turn time on near-linear potentials.
2. **NUTS non-GE for light tails beyond exact powers.** Extend Gruffaz et al. Thm 13 (H2-(ii)) to the regularity class of Livingstone's (17)–(19), so that D.1 also covers NUTS.
3. **Potential-level sufficiency in B.3.** When two hazards are within a square, construct a smooth `g` whose pullbacks satisfy A4 or H6(m), not just the tail window.
4. **Finite-sample selection.** Find a statistic that places a coordinate in the right window (affine, arcsinh, h-map or power), with guarantees on warm-up draws. Hill thresholds cannot tell Laplace from $t_5$ and often mislabel lognormal tails (C.4). The window theorem suggests estimating the growth of $\log\hat\psi$ against $\log\log x$, but the finite range of $x$ that a warm-up sees makes this hard.
5. **Quantitative constants.** Find how $\bar S$ (step size times maximum depth) depends on $c$ and $\nu$, and close the gap between the $h2^{K_m}\le\bar S$ regime and practical NUTS with $K_m = 10$.
6. **Non-product targets.** Coordinate-wise maps of rotated or elliptical heavy tails do not give product pullbacks. The A4 class is closed under linear maps, which suggests whitening followed by a radial h-map for the elliptical part (assessment Track 2.2).
7. **The soft default.** Benchmark an h-map with a large $c$ on all coordinates against arcsinh and against selected maps in the Track 1 ablation runner. Then find a theorem that trades the light-tail unstable mass $e^{-\Theta(c/\varepsilon^2)}$ against the heavy-tail bulk region $\vert x\vert\lesssim R\sqrt c$.
8. **Lean.** Cover general $\alpha$ in B.2 and A.4 (`rpow`), and the location-error factor $2(1+\delta^2)$ for A.5.
9. **NUTS cost.** Bound the expected number of leapfrog steps per NUTS transition as a function of position in the tail, for each map. D.5 gives the one-dimensional heuristic; the product case needs the multivariate U-turn criterion.

## References

- Brešar, M. & Mijatović, A. (2026). Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions. arXiv:2608.04284.
- Dekkers, A. L. M., Einmahl, J. H. J. & de Haan, L. (1989). A moment estimator for the index of an extreme-value distribution. *Ann. Statist.* 17(4), 1833–1855.
- Durmus, A., Gruffaz, S., Kailas, M., Saksman, E. & Vihola, M. (2023). On the convergence of dynamic implementations of Hamiltonian Monte Carlo and No U-Turn Samplers. arXiv:2307.03460 (v2, 2024).
- Durmus, A., Moulines, É. & Saksman, E. (2020). Irreducibility and geometric ergodicity of Hamiltonian Monte Carlo. *Ann. Statist.* 48(6), 3545–3564; arXiv:1705.00166 (theorem numbers from v2).
- Goerg, G. M. (2015). The Lambert way to Gaussianize heavy-tailed data with the inverse of Tukey's h transformation as a special case. *Sci. World J.* 2015, 909231.
- Gruffaz, S., Kim, K., Guehtar, F., Duval-Decaix, H. & Trautmann, P. (2026). A theoretical comparison of No-U-Turn Sampler variants. arXiv:2603.18640 (v3).
- He, Y., Balasubramanian, K. & Erdogdu, M. A. (2024). An analysis of transformed unadjusted Langevin algorithm for heavy-tailed sampling. *IEEE Trans. Inf. Theory* 70(1), 571–593; arXiv:2201.08349.
- Hickling, T. & Prangle, D. (2025). Flexible tails for normalizing flows. ICML 2025; arXiv:2406.16971.
- Hsing, T. (1991). On tail index estimation using dependent data. *Ann. Statist.* 19(3), 1547–1569.
- Jaini, P., Kobyzev, I., Yu, Y. & Brubaker, M. (2020). Tails of Lipschitz triangular flows. ICML 2020.
- Jarner, S. F. & Tweedie, R. L. (2003). Necessary conditions for geometric and polynomial ergodicity of random-walk-type Markov chains. *Bernoulli* 9(4), 559–578.
- Johnson, L. T. & Geyer, C. J. (2012). Variable transformation to obtain geometric ergodicity in the random-walk Metropolis algorithm. *Ann. Statist.* 40(6), 3050–3076.
- Livingstone, S., Betancourt, M., Byrne, S. & Girolami, M. (2019). On the geometric ergodicity of Hamiltonian Monte Carlo. *Bernoulli* 25(4A), 3109–3138; corrigendum in arXiv:1601.08057v5, App. D (2026).
- Mason, D. M. (1982). Laws of large numbers for sums of extreme values. *Ann. Probab.* 10(3), 754–764.
- nutpie 0.16.11, `python/nutpie/normalizing_flow.py` (class `Contract2`).
- Pachebat, J. (2026). arXiv:2605.20068.
- Power, S. & Vasdekis, G. (2025). arXiv:2511.21563.
- Resnick, S. & Stărică, C. (1998). Tail index estimation for dependent data. *Ann. Appl. Probab.* 8(4), 1156–1183.
- Roberts, G. O. & Tweedie, R. L. (1996). Geometric convergence and central limit theorems for multidimensional Hastings and Metropolis algorithms. *Biometrika* 83(1), 95–110.

Unverified details (from memory or secondary reading only): the page range of Jarner & Tweedie (2003); the Roberts & Tweedie (1996) theorem number (cited as Livingstone et al. Prop. 3.5); the full titles of Pachebat (2026) and Power & Vasdekis (2025).
