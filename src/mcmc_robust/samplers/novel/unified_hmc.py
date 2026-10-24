"""Unified Multi-Step Robust Hamiltonian Monte Carlo Sampler (Feature F25)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform
from mcmc_robust.transforms.hierarchical_arcsinh import HierarchicalArcsinhTransform


class UnifiedRobustHMC(Sampler):
    r"""Unified Multi-Step Robust Hamiltonian Monte Carlo Sampler (Feature F25).

    Extends Hamiltonian dynamics beyond single-step Langevin diffusions by operating
    in a transformed phase space (z, p) induced by a coordinate-wise arcsinh diffeomorphism:
        z_i = \operatorname{arcsinh}(x_i / R_i) \quad (x_i = R_i \sinh(z_i))

    Properties:
    1. Canonical Separable Hamiltonian on Transformed Phase Space:
           H(z, p) = V(z) + K(p) = -\log \pi_Z(z) + \frac{1}{2} p^T M^{-1} p
       where M = \operatorname{diag}(s_z^{-2}) is a position-independent diagonal metric.
    2. Symplectic Explicit Leapfrog:
       Avoids expensive and fragile implicit Riemannian Newton-Raphson solvers while
       preserving the canonical symplectic 2-form \omega = dz \wedge dp and exact
       phase-space volume (\det J \equiv 1.0 bit-for-bit).
    3. Momentum Inversion and Skew-Reversibility:
       The momentum flip R(z, p) = (z, -p) satisfies R \circ \Phi_\epsilon \circ R \circ \Phi_\epsilon = I,
       guaranteeing exact detailed balance under the Metropolis-Hastings filter:
           \alpha = \min(1, \exp(-\Delta H))
    4. Optional Gentle Stiffness Attenuation:
       Leapfrog force attenuation \rho(\kappa) = (1 + \gamma \kappa)^{-\beta} (default \gamma = 0)
       tames numerical instability in superlinear / stiff geometries while preserving exact
       \pi-invariance (via Green 1995 deterministic MH theorem).
    5. Three-Phase Warmup Scheduling:
       - Phase 1: Coordinate tail scale R_i calibration and freeze.
       - Phase 2: Diagonal covariance / mass matrix M adaptation via Welford in z-space.
       - Phase 3: Trajectory step size adaptation via Hoffman-Gelman Dual Averaging.
    6. Exact Zero-Redundancy Oracle Caching Protocol:
       Consumes strictly L gradient evaluations per step, 1 joint log_prob on final step,
       and 0 extra target evaluations on rejection.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        gamma: float = 0.0,
        radius: float | Sequence[float] | np.ndarray = 3.0,
        mode: str = "coordinate",
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.65,
        adapt_step_size: bool = False,
        adapt_preconditioner: bool = False,
        adapt_tail_scale: bool = False,
        warmup_steps: int = 0,
        taming_power: float = 0.5,
        min_radius: float = 0.1,
        max_radius: float = 1e4,
        mask: Sequence[bool] | np.ndarray | None = None,
        adaptation: DualAveragingAdaptation | None = None,
        funnel_coupling: bool = False,
        funnel_parent_idx: int = 0,
    ) -> None:
        r"""Initializes UnifiedRobustHMC sampler.

        Args:
            step_size: Trajectory length h > 0 (leapfrog step size is \epsilon = h / L).
            n_leapfrog: Number of leapfrog integration steps L >= 1.
            gamma: Gentle stiffness attenuation parameter \gamma >= 0 (default: 0.0).
            radius: Confinement tail scale R > 0 (scalar or coordinate vector).
            mode: Stiffness attenuation mode, either 'coordinate' or 'global'.
            preconditioner: Optional initial diagonal scale vector s_z in transformed space.
            target_accept: Dual averaging target acceptance rate (default: 0.65 for HMC).
            adapt_step_size: Whether to adapt step size during warmup.
            adapt_preconditioner: Whether to adapt diagonal mass matrix via Welford during Phase 2.
            adapt_tail_scale: Whether to calibrate coordinate scale R_i online during Phase 1.
            warmup_steps: Total planned warmup iterations.
            taming_power: Exponent \beta in (0, 1] for gentle taming divisor (default: 0.5).
            min_radius: Minimum allowable value for coordinate scale R_i.
            max_radius: Maximum allowable value for coordinate scale R_i.
            mask: Optional boolean mask of shape (d,) indicating transformed coordinates.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if n_leapfrog < 1:
            raise ValueError(f"n_leapfrog must be >= 1, got {n_leapfrog}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if mode not in ("coordinate", "global"):
            raise ValueError(f"mode must be 'coordinate' or 'global', got {mode}")
        if min_radius <= 0.0 or min_radius >= max_radius:
            raise ValueError(f"Invalid radius bounds: min_radius={min_radius}, max_radius={max_radius}")
        if not (0.0 < target_accept < 1.0):
            raise ValueError(f"target_accept must be in (0, 1), got {target_accept}")
        if not (0.0 < taming_power <= 1.0):
            raise ValueError(f"taming_power must be in (0, 1], got {taming_power}")
        if funnel_parent_idx < 0:
            raise ValueError(f"funnel_parent_idx must be non-negative, got {funnel_parent_idx}")

        self.n_leapfrog: int = int(n_leapfrog)
        self.gamma: float = float(gamma)
        self.mode: str = mode
        self.target_accept: float = float(target_accept)
        self.adapt_step_size: bool = bool(adapt_step_size)
        self.adapt_preconditioner: bool = bool(adapt_preconditioner)
        self.adapt_tail_scale: bool = bool(adapt_tail_scale)
        self.warmup_steps: int = int(warmup_steps)
        self.taming_power: float = float(taming_power)
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
        """UnifiedRobustHMC requires first-order gradients."""
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

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes one multi-step Hamiltonian transition in transformed phase space."""
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
            current_aux = state.auxiliary
        else:
            z, _ = transform.forward(x)
            _, log_det_inv_x = transform.inverse(z)
            lp_z = float(state.log_prob + log_det_inv_x)
            assert state.grad_log_prob is not None
            grad_z = transform.forward_grad(x, state.grad_log_prob)
            current_aux = {
                "z": z,
                "lp_z": lp_z,
                "grad_z": grad_z,
                "log_det_inv": log_det_inv_x,
                "transform_id": transform_id,
            }

        # 2. Sample initial momentum p0 ~ N(0, M)
        p0 = self._sample_momentum(d, s_z, rng)
        p = p0.copy()
        k0 = self._kinetic_energy(p0, s_z)
        h0 = -lp_z + k0

        L = self.n_leapfrog
        epsilon = self.step_size / L

        grad_evals = 0
        lp_evals = 0
        diverged = False
        kappa_max_step = 0.0

        # Current force at position z
        force, kappa = self._gentle_force(grad_z, s_z, epsilon)
        kappa_max_step = max(kappa_max_step, kappa)

        # Variables for candidate state
        z_curr = z.copy()
        x_curr = x.copy()
        lp_x_cand = -np.inf
        grad_x_cand = np.zeros(d, dtype=np.float64)
        lp_z_cand = -np.inf
        grad_z_cand = np.zeros(d, dtype=np.float64)
        log_det_inv_cand = 0.0

        # 3. Symplectic explicit leapfrog integration in (z, p) space
        for ell in range(L):
            # Half-step momentum kick
            p += 0.5 * epsilon * force

            # Full-step position drift in z-space
            z_curr += epsilon * self._velocity(p, s_z)

            if not np.all(np.isfinite(z_curr)) or not np.all(np.isfinite(p)):
                diverged = True
                break

            # Map back to primal space for target evaluation
            x_curr, log_det_inv_cand = transform.inverse(z_curr)
            if not np.all(np.isfinite(x_curr)):
                diverged = True
                break

            # Evaluate target
            if ell == L - 1:
                lp_x_cand, grad_x_cand = target.log_prob_and_grad(x_curr)
                grad_evals += 1
                lp_evals += 1
            else:
                grad_x_cand = target.grad_log_prob(x_curr)
                grad_evals += 1

            if not np.all(np.isfinite(grad_x_cand)):
                diverged = True
                break

            # Pull back gradient to z-space
            grad_z_cand = transform.forward_grad(x_curr, grad_x_cand)
            force, kappa = self._gentle_force(grad_z_cand, s_z, epsilon)
            kappa_max_step = max(kappa_max_step, kappa)

            # Half-step momentum kick
            p += 0.5 * epsilon * force

        # 4. Energy error and Metropolis-Hastings filter
        if diverged or not np.isfinite(lp_x_cand) or not np.all(np.isfinite(grad_z_cand)):
            delta_H = float("inf")
            accept_prob = 0.0
            divergent = 1.0
        else:
            lp_z_cand = float(lp_x_cand + log_det_inv_cand)
            kL = self._kinetic_energy(p, s_z)
            hL = -lp_z_cand + kL
            delta_H = hL - h0

            if not np.isfinite(delta_H):
                accept_prob = 0.0
                divergent = 1.0
            elif delta_H > 1000.0:
                accept_prob = float(np.exp(-min(709.0, delta_H)))
                divergent = 1.0
            elif delta_H <= 0.0:
                accept_prob = 1.0
                divergent = 0.0
            else:
                accept_prob = float(np.exp(-delta_H))
                divergent = 0.0

        # 5. Accept / Reject decision
        u = rng.uniform()
        accepted = bool(u < accept_prob)

        if accepted:
            aux_cand = {
                "z": z_curr,
                "lp_z": lp_z_cand,
                "grad_z": grad_z_cand,
                "log_det_inv": log_det_inv_cand,
                "transform_id": transform_id,
            }
            new_state = ChainState(
                x=x_curr,
                log_prob=lp_x_cand,
                grad_log_prob=grad_x_cand,
                auxiliary=aux_cand,
            )
        else:
            new_state = state.copy_with(auxiliary=current_aux)

        stats: dict[str, Any] = {
            "accept_prob": float(accept_prob),
            "grad_evals": grad_evals,
            "log_prob_evals": lp_evals,
            "n_leapfrog": L,
            "epsilon": float(epsilon),
            "step_size": float(self.step_size),
            "hamiltonian_error": float(delta_H),
            "energy_error": float(delta_H),
            "divergent": float(divergent),
            "accepted": float(accepted),
            "kappa": float(kappa_max_step),
        }

        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=float(accept_prob),
            stats=stats,
            proposal=x_curr,
            step_size=self.step_size,
        )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Executes multi-phase warmup adaptation without destructive interference."""
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
