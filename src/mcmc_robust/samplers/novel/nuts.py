"""Unified Robust No-U-Turn Sampler (NUTS) with Arcsinh Tail Diffeomorphism (Feature F26)."""

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.transforms.base import SpaceTransformation
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform
from mcmc_robust.transforms.hierarchical_arcsinh import HierarchicalArcsinhTransform


class _Candidate(NamedTuple):
    z: np.ndarray
    p: np.ndarray
    x: np.ndarray
    lp_x: float
    grad_x: np.ndarray
    lp_z: float
    grad_z: np.ndarray
    log_det_inv: float


class UnifiedRobustNUTS(Sampler):
    r"""Unified Robust No-U-Turn Sampler (NUTS) with Coordinate-Wise Arcsinh Diffeomorphism.

    Feature F26 resolves the dual pathologies of target flatness and roughness in
    multi-step Hamiltonian MCMC:
    1. Transformed Phase Space (z, p):
       Coordinate-wise \operatorname{arcsinh} diffeomorphism:
           z_i = \operatorname{arcsinh}(x_i / R_i) \quad (x_i = R_i \sinh(z_i))
       converts polynomial heavy tails into asymptotic linear potentials V(z) \sim \nu_i |z_i|
       (exponential tails). This guarantees constant restoring deceleration a_i = -\nu_i / M_i \operatorname{sgn}(z_i),
       ensuring the No-U-Turn criterion \langle z^+ - z^-, M^{-1} p^\pm \rangle < 0 triggers in
       finite, small leapfrog steps, eliminating NUTS tree-depth blowup to max_depth.
    2. Recursive Binary Tree Building:
       Builds trajectory trees of depth j \in [0, \text{max\_depth}] with forward/backward
       direction doubling.
    3. Betancourt (2017) Multinomial Trajectory Sampling:
       Selects candidate states with exact Boltzmann weight w = \exp(-\Delta H), eliminating
       the inefficiency of Hoffman-Gelman (2014) slice sampling.
    4. Generalized No-U-Turn Stopping Criterion:
       Monitors endpoint velocities in transformed metric:
           \langle z^+ - z^-, s_z^2 \odot p^+ \rangle \ge 0 \quad \text{and} \quad
           \langle z^+ - z^-, s_z^2 \odot p^- \rangle \ge 0
    5. Divergence Detection & Numerical Stability:
       Detects energy divergence (\Delta H > \Delta_{\max}, default 1000.0) or non-finite evaluations,
       terminating tree exploration safely.
    6. Three-Phase Warmup Scheduling:
       - Phase 1: Coordinate tail scale R_i calibration and freeze.
       - Phase 2: Diagonal covariance / mass matrix M adaptation via Welford in z-space.
       - Phase 3: Leapfrog step size \epsilon adaptation targeting \alpha^\star = 0.80.
    7. Zero-Redundancy Oracle Caching:
       Evaluates target strictly once per leapfrog step, caching log_prob and grad_log_prob
       of the accepted state without redundant evaluation.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        max_depth: int = 10,
        gamma: float = 0.0,
        radius: float | Sequence[float] | np.ndarray = 3.0,
        mode: str = "coordinate",
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.80,
        adapt_step_size: bool = False,
        adapt_preconditioner: bool = False,
        adapt_tail_scale: bool = False,
        warmup_steps: int = 0,
        taming_power: float = 0.5,
        delta_max: float = 1000.0,
        min_radius: float = 0.1,
        max_radius: float = 1e4,
        mask: Sequence[bool] | np.ndarray | None = None,
        adaptation: DualAveragingAdaptation | None = None,
        funnel_coupling: bool = False,
        funnel_parent_idx: int = 0,
    ) -> None:
        r"""Initializes UnifiedRobustNUTS sampler.

        Args:
            step_size: Base leapfrog step size \epsilon > 0.
            max_depth: Maximum binary tree doubling depth (default: 10).
            gamma: Gentle stiffness attenuation parameter \gamma >= 0 (default: 0.0).
            radius: Confinement tail scale R > 0 (scalar or coordinate vector).
            mode: Stiffness attenuation mode, either 'coordinate' or 'global'.
            preconditioner: Optional initial diagonal scale vector s_z in transformed space.
            target_accept: Dual averaging target acceptance rate (default: 0.80 for NUTS).
            adapt_step_size: Whether to adapt step size during warmup.
            adapt_preconditioner: Whether to adapt diagonal mass matrix via Welford during Phase 2.
            adapt_tail_scale: Whether to calibrate coordinate scale R_i online during Phase 1.
            warmup_steps: Total planned warmup iterations.
            taming_power: Exponent \beta in (0, 1] for gentle taming divisor (default: 0.5).
            delta_max: Maximum allowable Hamiltonian error \Delta H before declaring divergence.
            min_radius: Minimum allowable value for coordinate scale R_i.
            max_radius: Maximum allowable value for coordinate scale R_i.
            mask: Optional boolean mask of shape (d,) indicating transformed coordinates.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if mode not in ("coordinate", "global"):
            raise ValueError(f"mode must be 'coordinate' or 'global', got {mode}")
        if delta_max <= 0.0:
            raise ValueError(f"delta_max must be positive, got {delta_max}")
        if min_radius <= 0.0 or min_radius >= max_radius:
            raise ValueError(f"Invalid radius bounds: min_radius={min_radius}, max_radius={max_radius}")
        if not (0.0 < target_accept < 1.0):
            raise ValueError(f"target_accept must be in (0, 1), got {target_accept}")
        if not (0.0 < taming_power <= 1.0):
            raise ValueError(f"taming_power must be in (0, 1], got {taming_power}")
        if funnel_parent_idx < 0:
            raise ValueError(f"funnel_parent_idx must be non-negative, got {funnel_parent_idx}")

        self.max_depth: int = int(max_depth)
        self.gamma: float = float(gamma)
        self.mode: str = mode
        self.target_accept: float = float(target_accept)
        self.adapt_step_size: bool = bool(adapt_step_size)
        self.adapt_preconditioner: bool = bool(adapt_preconditioner)
        self.adapt_tail_scale: bool = bool(adapt_tail_scale)
        self.warmup_steps: int = int(warmup_steps)
        self.taming_power: float = float(taming_power)
        self.delta_max: float = float(delta_max)
        self.min_radius: float = float(min_radius)
        self.max_radius: float = float(max_radius)
        self.funnel_coupling: bool = bool(funnel_coupling)
        self.funnel_parent_idx: int = int(funnel_parent_idx)

        # Coordinate tail scale configuration
        self._initial_radius: float | np.ndarray = (
            np.asarray(radius, dtype=np.float64) if hasattr(radius, "__len__") else float(radius)
        )
        self.radius: np.ndarray | None = None
        self._mask: np.ndarray | None = np.asarray(mask, dtype=bool) if mask is not None else None
        self._transform: CoordinateWiseArcsinhTransform | HierarchicalArcsinhTransform | None = None

        # Adaptation components
        eff_target_accept = max(self.target_accept, 0.85) if self.funnel_coupling else self.target_accept
        if adaptation is not None:
            self.adaptation: DualAveragingAdaptation | None = adaptation
        elif self.adapt_step_size:
            self.adaptation = DualAveragingAdaptation(
                initial_step_size=self.step_size,
                target_accept=eff_target_accept,
            )
        else:
            self.adaptation = None

        self.welford: WelfordPreconditioner | None = None

        # Warmup phase tracking
        self._phase1_samples: list[np.ndarray] = []
        self._tail_scale_frozen: bool = False
        self._preconditioner_frozen: bool = False

    @property
    def requires_grad(self) -> bool:
        """UnifiedRobustNUTS requires first-order gradients."""
        return True

    @property
    def transform(self) -> CoordinateWiseArcsinhTransform | HierarchicalArcsinhTransform | None:
        """Underlying coordinate or hierarchical arcsinh transform instance."""
        return self._transform

    def _ensure_transform(self, d: int) -> CoordinateWiseArcsinhTransform | HierarchicalArcsinhTransform:
        """Initializes or retrieves the arcsinh diffeomorphism instance."""
        if self._transform is None:
            if self.radius is None:
                if isinstance(self._initial_radius, np.ndarray):
                    if self._initial_radius.shape != (d,):
                        raise ValueError(f"Initial radius shape {self._initial_radius.shape} does not match d={d}")
                    self.radius = np.clip(self._initial_radius, self.min_radius, self.max_radius)
                else:
                    self.radius = np.full(d, float(self._initial_radius), dtype=np.float64)
            if self.funnel_coupling:
                init_radii = self.radius.copy()
                if not isinstance(self._initial_radius, np.ndarray) and float(self._initial_radius) == 3.0:
                    init_radii = np.full(d, 6.0, dtype=np.float64)
                    init_radii[self.funnel_parent_idx] = 18.0
                self._transform = HierarchicalArcsinhTransform(
                    d=d,
                    radii=init_radii,
                    coupling=1.5,
                    parent_idx=self.funnel_parent_idx,
                    parent_scale=3.0,
                    max_log_scale=15.0,
                    mask=self._mask,
                )
            else:
                self._transform = CoordinateWiseArcsinhTransform(
                    d=d,
                    radii=self.radius,
                    mask=self._mask,
                )
        return self._transform

    def _kinetic_energy(self, p: np.ndarray, s_z: np.ndarray) -> float:
        """Computes kinetic energy K(p) = 0.5 * p^T M^{-1} p."""
        with np.errstate(over="ignore", invalid="ignore"):
            return float(0.5 * np.sum((s_z * p) ** 2))

    def _velocity(self, p: np.ndarray, s_z: np.ndarray) -> np.ndarray:
        """Computes velocity v = M^{-1} p = s_z^2 * p."""
        return (s_z**2) * p

    def _sample_momentum(self, d: int, s_z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Samples momentum p ~ N(0, M) where M = diag(s_z^{-2})."""
        xi = rng.normal(size=d)
        return xi / s_z

    def _gentle_force(
        self,
        grad_z: np.ndarray,
        s_z: np.ndarray,
        epsilon: float,
    ) -> tuple[np.ndarray, float]:
        r"""Computes gentle stiffness-attenuated leapfrog force F(z) and stiffness indicator \kappa."""
        eff_gamma = max(self.gamma, 0.10) if self.funnel_coupling else self.gamma
        if eff_gamma <= 0.0:
            return grad_z.copy(), 0.0

        with np.errstate(over="ignore", invalid="ignore"):
            if self.mode == "coordinate":
                kappa_coord = epsilon * s_z * np.abs(grad_z)
                force = grad_z / ((1.0 + eff_gamma * kappa_coord) ** self.taming_power)
                kappa_max = float(np.max(kappa_coord))
                return force, kappa_max
            else:
                kappa_global = epsilon * float(np.linalg.norm(s_z * grad_z))
                force = grad_z / ((1.0 + eff_gamma * kappa_global) ** self.taming_power)
                return force, kappa_global

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        """Initializes ChainState with strictly 1 target evaluation."""
        x0_arr = target._validate_input(x0)
        d = target.d
        transform = self._ensure_transform(d)

        lp_x, grad_x = target.log_prob_and_grad(x0_arr)

        z0, _ = transform.forward(x0_arr)
        _, log_det_inv = transform.inverse(z0)
        lp_z = float(lp_x + log_det_inv)
        grad_z = transform.forward_grad(x0_arr, grad_x)

        aux = {
            "z": z0,
            "lp_z": lp_z,
            "grad_z": grad_z,
            "log_det_inv": log_det_inv,
            "transform_id": id(transform),
        }
        return ChainState(x=x0_arr, log_prob=lp_x, grad_log_prob=grad_x, auxiliary=aux)

    def _build_tree(
        self,
        z: np.ndarray,
        p: np.ndarray,
        g_z: np.ndarray,
        v: int,
        j: int,
        epsilon: float,
        h0: float,
        s_z: np.ndarray,
        transform: SpaceTransformation,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,  # z_minus, p_minus, g_minus
        np.ndarray,
        np.ndarray,
        np.ndarray,  # z_plus, p_plus, g_plus
        _Candidate,  # candidate state
        float,  # weight w
        int,  # stopping flag s (1 or 0)
        float,  # alpha_sum
        int,  # n_alpha
        bool,  # divergent
        int,  # n_steps
    ]:
        """Recursively builds binary leapfrog tree."""
        d = len(z)

        # Base case: single leapfrog step in direction v * epsilon
        if j == 0:
            force, _ = self._gentle_force(g_z, s_z, epsilon)
            p_half = p + (0.5 * v * epsilon) * force
            z_next = z + (v * epsilon) * self._velocity(p_half, s_z)

            # Defensive numerical checks
            if not np.all(np.isfinite(z_next)) or not np.all(np.isfinite(p_half)):
                cand_dummy = _Candidate(z, p, z, -np.inf, np.zeros(d), -np.inf, np.zeros(d), 0.0)
                return (z, p, g_z, z, p, g_z, cand_dummy, 0.0, 0, 0.0, 1, True, 1)

            x_next, log_det_inv_next = transform.inverse(z_next)
            if not np.all(np.isfinite(x_next)):
                cand_dummy = _Candidate(z, p, z, -np.inf, np.zeros(d), -np.inf, np.zeros(d), 0.0)
                return (z, p, g_z, z, p, g_z, cand_dummy, 0.0, 0, 0.0, 1, True, 1)

            lp_x_next, grad_x_next = target.log_prob_and_grad(x_next)
            if not np.isfinite(lp_x_next) or not np.all(np.isfinite(grad_x_next)):
                cand_dummy = _Candidate(
                    z_next, p_half, x_next, -np.inf, np.zeros(d), -np.inf, np.zeros(d), log_det_inv_next
                )
                return (z_next, p_half, g_z, z_next, p_half, g_z, cand_dummy, 0.0, 0, 0.0, 1, True, 1)

            grad_z_next = transform.forward_grad(x_next, grad_x_next)
            force_next, _ = self._gentle_force(grad_z_next, s_z, epsilon)
            p_next = p_half + (0.5 * v * epsilon) * force_next

            if not np.all(np.isfinite(p_next)):
                cand_dummy = _Candidate(
                    z_next, p_half, x_next, lp_x_next, grad_x_next, -np.inf, grad_z_next, log_det_inv_next
                )
                return (z_next, p_half, grad_z_next, z_next, p_half, grad_z_next, cand_dummy, 0.0, 0, 0.0, 1, True, 1)

            lp_z_next = float(lp_x_next + log_det_inv_next)
            k_next = self._kinetic_energy(p_next, s_z)
            h_next = -lp_z_next + k_next
            delta_h = h_next - h0

            is_divergent = bool(delta_h > self.delta_max or not np.isfinite(delta_h))
            if is_divergent:
                w_next = 0.0
                s_next = 0
                alpha_next = 0.0
            else:
                neg_delta_h = float(np.clip(-delta_h, -709.0, 709.0))
                w_next = float(np.exp(neg_delta_h))
                s_next = 1
                alpha_next = float(np.exp(min(0.0, -delta_h)))

            cand = _Candidate(
                z=z_next,
                p=p_next,
                x=x_next,
                lp_x=lp_x_next,
                grad_x=grad_x_next,
                lp_z=lp_z_next,
                grad_z=grad_z_next,
                log_det_inv=log_det_inv_next,
            )
            return (
                z_next,
                p_next,
                grad_z_next,
                z_next,
                p_next,
                grad_z_next,
                cand,
                w_next,
                s_next,
                alpha_next,
                1,
                is_divergent,
                1,
            )

        # Recursive case: build subtree 1
        (
            z_m,
            p_m,
            g_m,
            z_p,
            p_p,
            g_p,
            cand1,
            w1,
            s1,
            alpha1,
            n_alpha1,
            div1,
            steps1,
        ) = self._build_tree(z, p, g_z, v, j - 1, epsilon, h0, s_z, transform, target, rng)

        if s1 == 0:
            return (
                z_m,
                p_m,
                g_m,
                z_p,
                p_p,
                g_p,
                cand1,
                w1,
                0,
                alpha1,
                n_alpha1,
                div1,
                steps1,
            )

        # Build subtree 2 from the outer boundary of subtree 1
        if v == -1:
            (
                z_mm,
                p_mm,
                g_mm,
                z_mp,
                p_mp,
                g_mp,
                cand2,
                w2,
                s2,
                alpha2,
                n_alpha2,
                div2,
                steps2,
            ) = self._build_tree(z_m, p_m, g_m, v, j - 1, epsilon, h0, s_z, transform, target, rng)
            z_new_m, p_new_m, g_new_m = z_mm, p_mm, g_mm
            z_new_p, p_new_p, g_new_p = z_p, p_p, g_p
        else:
            (
                z_pm,
                p_pm,
                g_pm,
                z_pp,
                p_pp,
                g_pp,
                cand2,
                w2,
                s2,
                alpha2,
                n_alpha2,
                div2,
                steps2,
            ) = self._build_tree(z_p, p_p, g_p, v, j - 1, epsilon, h0, s_z, transform, target, rng)
            z_new_m, p_new_m, g_new_m = z_m, p_m, g_m
            z_new_p, p_new_p, g_new_p = z_pp, p_pp, g_pp

        # Multinomial candidate selection between subtree 1 and subtree 2
        w_total = w1 + w2
        if w_total > 0.0 and rng.uniform() < (w2 / w_total):
            cand_comb = cand2
        else:
            cand_comb = cand1

        # Generalized No-U-Turn stopping criterion on outermost boundaries
        delta_z = z_new_p - z_new_m
        v_plus = self._velocity(p_new_p, s_z)
        v_minus = self._velocity(p_new_m, s_z)
        no_u_turn = bool(np.dot(delta_z, v_plus) >= 0.0 and np.dot(delta_z, v_minus) >= 0.0)

        # Cross-subtree inner-seam U-turn checks (time-reversal symmetric)
        if no_u_turn:
            if v == -1:
                delta_left_inner = z_m - z_mm
                delta_inner_right = z_p - z_mp
                cross_ok = bool(
                    np.dot(delta_left_inner, self._velocity(p_m, s_z)) >= 0.0
                    and np.dot(delta_left_inner, self._velocity(p_mm, s_z)) >= 0.0
                    and np.dot(delta_inner_right, self._velocity(p_p, s_z)) >= 0.0
                    and np.dot(delta_inner_right, self._velocity(p_mp, s_z)) >= 0.0
                )
            else:
                delta_left_inner = z_pm - z_m
                delta_inner_right = z_pp - z_p
                cross_ok = bool(
                    np.dot(delta_left_inner, self._velocity(p_pm, s_z)) >= 0.0
                    and np.dot(delta_left_inner, self._velocity(p_m, s_z)) >= 0.0
                    and np.dot(delta_inner_right, self._velocity(p_pp, s_z)) >= 0.0
                    and np.dot(delta_inner_right, self._velocity(p_p, s_z)) >= 0.0
                )
            no_u_turn = no_u_turn and cross_ok

        s_comb = int(s2 == 1 and no_u_turn)
        alpha_comb = alpha1 + alpha2
        n_alpha_comb = n_alpha1 + n_alpha2
        div_comb = div1 or div2
        steps_comb = steps1 + steps2

        return (
            z_new_m,
            p_new_m,
            g_new_m,
            z_new_p,
            p_new_p,
            g_new_p,
            cand_comb,
            w_total,
            s_comb,
            alpha_comb,
            n_alpha_comb,
            div_comb,
            steps_comb,
        )

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes one Unified Robust NUTS transition with multinomial sampling."""
        d = target.d
        transform = self._ensure_transform(d)
        s_z = self._preconditioner if self._preconditioner is not None else np.ones(d, dtype=np.float64)

        # 1. Retrieve or recompute current state in transformed phase space
        x = state.x
        transform_id = id(transform)
        if (
            "z" in state.auxiliary
            and "grad_z" in state.auxiliary
            and "lp_z" in state.auxiliary
            and state.auxiliary.get("transform_id") == transform_id
        ):
            z = np.asarray(state.auxiliary["z"], dtype=np.float64)
            lp_z = float(state.auxiliary["lp_z"])
            grad_z = np.asarray(state.auxiliary["grad_z"], dtype=np.float64)
            log_det_inv = float(state.auxiliary.get("log_det_inv", 0.0))
            current_aux = state.auxiliary
        else:
            z, _ = transform.forward(x)
            _, log_det_inv = transform.inverse(z)
            lp_z = float(state.log_prob + log_det_inv)
            assert state.grad_log_prob is not None
            grad_z = transform.forward_grad(x, state.grad_log_prob)
            current_aux = {
                "z": z,
                "lp_z": lp_z,
                "grad_z": grad_z,
                "log_det_inv": log_det_inv,
                "transform_id": transform_id,
            }

        # 2. Sample momentum p0 ~ N(0, M)
        p0 = self._sample_momentum(d, s_z, rng)
        k0 = self._kinetic_energy(p0, s_z)
        h0 = -lp_z + k0

        # Initialize tree boundaries and initial candidate
        z_minus, p_minus, g_minus = z.copy(), p0.copy(), grad_z.copy()
        z_plus, p_plus, g_plus = z.copy(), p0.copy(), grad_z.copy()

        assert state.grad_log_prob is not None
        best_cand = _Candidate(
            z=z.copy(),
            p=p0.copy(),
            x=x.copy(),
            lp_x=float(state.log_prob),
            grad_x=state.grad_log_prob.copy(),
            lp_z=lp_z,
            grad_z=grad_z.copy(),
            log_det_inv=log_det_inv,
        )

        w_tree = 1.0  # Weight of initial state
        n_leapfrog_total = 0
        alpha_sum_total = 0.0
        n_alpha_total = 0
        divergent_step = False
        depth = 0
        s = 1

        epsilon = self.step_size

        # 3. Main NUTS tree-doubling loop
        while s == 1 and depth < self.max_depth:
            v = int(rng.choice([-1, 1]))

            if v == -1:
                (
                    z_new_m,
                    p_new_m,
                    g_new_m,
                    _,
                    _,
                    _,
                    cand_sub,
                    w_sub,
                    s_sub,
                    alpha_sub,
                    n_alpha_sub,
                    div_sub,
                    steps_sub,
                ) = self._build_tree(z_minus, p_minus, g_minus, v, depth, epsilon, h0, s_z, transform, target, rng)
                z_minus, p_minus, g_minus = z_new_m, p_new_m, g_new_m
            else:
                (
                    _,
                    _,
                    _,
                    z_new_p,
                    p_new_p,
                    g_new_p,
                    cand_sub,
                    w_sub,
                    s_sub,
                    alpha_sub,
                    n_alpha_sub,
                    div_sub,
                    steps_sub,
                ) = self._build_tree(z_plus, p_plus, g_plus, v, depth, epsilon, h0, s_z, transform, target, rng)
                z_plus, p_plus, g_plus = z_new_p, p_new_p, g_new_p

            n_leapfrog_total += steps_sub
            alpha_sum_total += alpha_sub
            n_alpha_total += n_alpha_sub
            if div_sub:
                divergent_step = True

            # Betancourt (2017) multinomial proposal acceptance from the new subtree
            if s_sub == 1:
                prob_accept = min(1.0, w_sub / w_tree) if w_tree > 0.0 else 1.0
                if rng.uniform() < prob_accept:
                    best_cand = cand_sub
                w_tree += w_sub

            # Check generalized No-U-Turn stopping criterion on outer boundaries
            delta_z = z_plus - z_minus
            v_plus = self._velocity(p_plus, s_z)
            v_minus = self._velocity(p_minus, s_z)
            no_u_turn = bool(np.dot(delta_z, v_plus) >= 0.0 and np.dot(delta_z, v_minus) >= 0.0)

            if s_sub == 0 or not no_u_turn:
                s = 0
            else:
                depth += 1

        # 4. Energy error and transition diagnostics
        mean_accept_prob = alpha_sum_total / max(1, n_alpha_total)
        cand_energy = -best_cand.lp_z + self._kinetic_energy(best_cand.p, s_z)
        hamiltonian_error = cand_energy - h0

        # State transition decision
        accepted = not np.array_equal(best_cand.x, x)

        if accepted:
            aux_cand = {
                "z": best_cand.z,
                "lp_z": best_cand.lp_z,
                "grad_z": best_cand.grad_z,
                "log_det_inv": best_cand.log_det_inv,
                "transform_id": transform_id,
            }
            new_state = ChainState(
                x=best_cand.x,
                log_prob=best_cand.lp_x,
                grad_log_prob=best_cand.grad_x,
                auxiliary=aux_cand,
            )
        else:
            new_state = state.copy_with(auxiliary=current_aux)

        stats: dict[str, Any] = {
            "accept_prob": float(mean_accept_prob),
            "grad_evals": n_leapfrog_total,
            "n_leapfrog": n_leapfrog_total,
            "tree_depth": depth,
            "divergent": float(divergent_step),
            "hamiltonian_error": float(hamiltonian_error),
            "energy_error": float(hamiltonian_error),
            "step_size": float(self.step_size),
            "epsilon": float(self.step_size),
            "accepted": float(accepted),
        }

        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=float(mean_accept_prob),
            stats=stats,
            proposal=best_cand.x,
            step_size=self.step_size,
        )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Executes multi-phase warmup adaptation targeting alpha* = 0.80."""
        W = self.warmup_steps
        if W <= 0:
            if self.adaptation is None:
                self.adaptation = DualAveragingAdaptation(
                    initial_step_size=self.step_size,
                    target_accept=self.target_accept,
                )
            self.step_size = self.adaptation.update(result.accept_prob)
            return

        W1 = max(20, int(0.30 * W)) if self.funnel_coupling else max(10, int(0.15 * W))
        W2 = int(0.80 * W)
        d = len(result.state.x)
        calibrate_phase1 = self.adapt_tail_scale or self.funnel_coupling
        eff_accept = (
            0.0 if (self.funnel_coupling and float(result.stats.get("divergent", 0.0)) > 0.0) else result.accept_prob
        )

        # Phase 1: Tail Scale, Location & Funnel Coupling Calibration (0 <= step_idx < W1)
        if step_idx < W1:
            if calibrate_phase1:
                self._phase1_samples.append(result.state.x.copy())
            if self.adaptation is None:
                eff_target = max(self.target_accept, 0.85) if self.funnel_coupling else self.target_accept
                self.adaptation = DualAveragingAdaptation(
                    initial_step_size=self.step_size,
                    target_accept=eff_target,
                )
            self.step_size = self.adaptation.update(eff_accept)

            if step_idx == W1 - 1 and calibrate_phase1 and len(self._phase1_samples) > 0:
                stacked = np.vstack(self._phase1_samples)
                if self.funnel_coupling:
                    self._transform = HierarchicalArcsinhTransform.calibrate_from_samples(
                        stacked,
                        parent_idx=self.funnel_parent_idx,
                        min_radius=self.min_radius,
                        max_radius=self.max_radius,
                        mask=self._mask,
                    )
                    self.radius = self._transform.radii.copy()
                else:
                    med, self.radius = CoordinateWiseArcsinhTransform.compute_phase1_calibration(
                        stacked,
                        min_radius=self.min_radius,
                        max_radius=self.max_radius,
                        scale_multiplier=1.5,
                    )
                    self._transform = CoordinateWiseArcsinhTransform(
                        d=d,
                        radii=self.radius,
                        mask=self._mask,
                        location=med,
                    )
                self._tail_scale_frozen = True
                self._phase1_samples.clear()
                if self.adaptation is not None:
                    self.adaptation.reset()

        # Phase 2: Diagonal Metric Adaptation (W1 <= step_idx < W2)
        elif step_idx < W2:
            if self.adapt_preconditioner or self.funnel_coupling:
                if self.welford is None:
                    self.welford = WelfordPreconditioner(d=d)
                z = result.state.auxiliary.get("z")
                if z is not None:
                    self.welford.update(np.asarray(z))
                if (step_idx - W1 + 1) % 25 == 0 or step_idx == W2 - 1:
                    self.set_preconditioner(self.welford.diagonal_scale())

            if self.adaptation is not None:
                self.step_size = self.adaptation.update(eff_accept)

            if step_idx == W2 - 1:
                self._preconditioner_frozen = True
                if self.adaptation is not None:
                    self.adaptation.reset()

        # Phase 3: Step-Size Fine Tuning (W2 <= step_idx < W)
        elif step_idx < W:
            if self.adaptation is not None:
                self.step_size = self.adaptation.update(eff_accept)
            if step_idx == W - 1:
                self.finalize_adaptation()

    def finalize_adaptation(self) -> float:
        """Freezes smoothed step size at end of warmup."""
        if self.adaptation is not None:
            self.step_size = self.adaptation.finalize()
        return self.step_size


# Exact alias
ArcsinhTransformedNUTS = UnifiedRobustNUTS
