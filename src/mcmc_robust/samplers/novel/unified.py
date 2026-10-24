"""Unified Dual-Pathology Robust MCMC Sampler (Feature F17)."""

from collections.abc import Sequence

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation, WelfordPreconditioner
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform


class UnifiedRobustSampler(Sampler):
    r"""Unified Dual-Pathology Robust MCMC Sampler (Feature F17).

    Synthesizes:
    1. Coordinate-wise C^\infty arcsinh diffeomorphism:
           z_i = \operatorname{arcsinh}(x_i / R_i) \quad (x_i = R_i \sinh(z_i))
       converting polynomial heavy tails into non-vanishing constant restoring drift,
       resolving target flatness (§4.3 of Power & Vasdekis 2025).
    2. Localized gentle taming:
           b_i(z) = [\nabla_z \log \pi_z(z)]_i / (1 + \gamma \kappa_{h, i}(z))^\beta
       mitigating superlinear gradient explosion while preserving unbounded super-exponential
       tail return, resolving target roughness (§3.3).
    3. Three-phase warmup scheduling (Phase 1: tail scale freeze -> Phase 2: metric
       freeze -> Phase 3: step-size freeze).
    4. Exact zero-redundancy oracle caching protocol: strictly 1 target evaluation
       per step, 0 on rejection.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        gamma: float = 0.1,
        radius: float | Sequence[float] | np.ndarray = 3.0,
        mode: str = "coordinate",
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.574,
        adapt_step_size: bool = False,
        adapt_preconditioner: bool = False,
        adapt_tail_scale: bool = False,
        warmup_steps: int = 0,
        taming_power: float = 0.5,
        min_radius: float = 0.1,
        max_radius: float = 1e4,
        mask: Sequence[bool] | np.ndarray | None = None,
        adaptation: DualAveragingAdaptation | None = None,
        kappa_max: float = 1.5,
    ) -> None:
        r"""Initializes UnifiedRobustSampler.

        Args:
            step_size: Base exploration step size h > 0.
            gamma: Gentle taming parameter \gamma >= 0 (0 disables taming).
            radius: Confinement tail scale R > 0 (scalar or coordinate-wise vector).
            mode: Taming mode, either 'coordinate' or 'global'.
            preconditioner: Optional initial diagonal standard deviation scale vector s_z.
            target_accept: Hoffman-Gelman dual averaging target acceptance rate (default 0.574).
            adapt_step_size: Whether to adapt step size during warmup.
            adapt_preconditioner: Whether to adapt diagonal covariance via Welford during Phase 2.
            adapt_tail_scale: Whether to calibrate coordinate scale R_i online during Phase 1.
            warmup_steps: Total planned warmup iterations (governs phase boundaries).
            taming_power: Exponent \beta in (0, 1] for gentle taming divisor (default: 0.5).
            min_radius: Minimum allowable value for coordinate scale R_i.
            max_radius: Maximum allowable value for coordinate scale R_i.
            mask: Optional boolean mask of shape (d,) indicating transformed coordinates.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if mode not in ("coordinate", "global"):
            raise ValueError(f"mode must be 'coordinate' or 'global', got {mode}")
        if min_radius <= 0.0 or min_radius >= max_radius:
            raise ValueError(f"Invalid radius bounds: min_radius={min_radius}, max_radius={max_radius}")

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
        self.kappa_max: float = float(kappa_max)

        # Coordinate tail scale storage
        self._initial_radius: float | np.ndarray = (
            np.asarray(radius, dtype=np.float64) if hasattr(radius, "__len__") else float(radius)
        )
        self.radius: np.ndarray | None = None
        self._mask: np.ndarray | None = np.asarray(mask, dtype=bool) if mask is not None else None
        self._transform: CoordinateWiseArcsinhTransform | None = None

        # Adaptation components
        if adaptation is not None:
            self.adaptation: DualAveragingAdaptation | None = adaptation
        elif self.adapt_step_size:
            self.adaptation = DualAveragingAdaptation(
                initial_step_size=self.step_size,
                target_accept=self.target_accept,
            )
        else:
            self.adaptation = None

        self.welford: WelfordPreconditioner | None = None

        # Phase tracking state
        self._phase1_samples: list[np.ndarray] = []
        self._tail_scale_frozen: bool = False
        self._preconditioner_frozen: bool = False

    @property
    def requires_grad(self) -> bool:
        """UnifiedRobustSampler requires first-order gradients."""
        return True

    @property
    def transform(self) -> CoordinateWiseArcsinhTransform | None:
        """Underlying CoordinateWiseArcsinhTransform instance."""
        return self._transform

    def _ensure_transform(self, d: int) -> CoordinateWiseArcsinhTransform:
        """Initializes or retrieves the CoordinateWiseArcsinhTransform instance."""
        if self._transform is None:
            if self.radius is None:
                if isinstance(self._initial_radius, np.ndarray):
                    if self._initial_radius.shape != (d,):
                        raise ValueError(f"Initial radius shape {self._initial_radius.shape} does not match d={d}")
                    self.radius = np.clip(self._initial_radius, self.min_radius, self.max_radius)
                else:
                    self.radius = np.full(d, float(self._initial_radius), dtype=np.float64)
            self._transform = CoordinateWiseArcsinhTransform(
                d=d,
                radii=self.radius,
                mask=self._mask,
            )
        return self._transform

    def _gentle_tamed_drift(
        self,
        grad_z: np.ndarray,
        s_z: np.ndarray,
        h: float,
    ) -> tuple[np.ndarray, float]:
        """Computes gentle tamed drift b(z) and stiffness indicator kappa."""
        if self.gamma <= 0.0:
            return grad_z.copy(), 0.0

        if self.mode == "coordinate":
            kappa_coord = h * s_z * np.abs(grad_z)
            stiff = self.gamma * kappa_coord
            if np.isinf(self.kappa_max):
                b = grad_z / ((1.0 + stiff) ** self.taming_power)
            else:
                excess = np.maximum(0.0, kappa_coord - 1.0)
                b = grad_z / (
                    (1.0 + stiff + (excess / self.kappa_max) ** (1.0 / self.taming_power)) ** self.taming_power
                )
            kappa_max = float(np.max(kappa_coord))
            return b, kappa_max
        else:
            kappa_global = h * float(np.linalg.norm(s_z * grad_z))
            stiff_g = self.gamma * kappa_global
            if np.isinf(self.kappa_max):
                b = grad_z / ((1.0 + stiff_g) ** self.taming_power)
            else:
                excess_g = max(0.0, kappa_global - 1.0)
                b = grad_z / (
                    (1.0 + stiff_g + (excess_g / self.kappa_max) ** (1.0 / self.taming_power)) ** self.taming_power
                )
            return b, kappa_global

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
        """Executes one unified robust MCMC step adhering strictly to 1 grad / step."""
        d = target.d
        h = self.step_size
        transform = self._ensure_transform(d)
        s_z = self._preconditioner if self._preconditioner is not None else np.ones(d, dtype=np.float64)
        var_z = s_z**2

        # 1. Retrieve or compute current state in transformed space
        x = state.x
        transform_id = id(transform)
        if (
            "z" in state.auxiliary
            and "grad_z" in state.auxiliary
            and "lp_z" in state.auxiliary
            and state.auxiliary.get("transform_id") == transform_id
        ):
            z = np.asarray(state.auxiliary["z"])
            lp_z = float(state.auxiliary["lp_z"])
            grad_z = np.asarray(state.auxiliary["grad_z"])
            current_aux = state.auxiliary
        else:
            # Recompute transformed quantities under current transform using cached primal quantities (0 oracle calls)
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

        with np.errstate(over="ignore", invalid="ignore"):
            # 2. Forward gentle tamed drift
            b_z, kappa_fwd = self._gentle_tamed_drift(grad_z, s_z, h)

            # 3. Propose candidate in transformed space
            xi = rng.normal(size=d)
            drift_z = h * (var_z * b_z)
            diff_z = np.sqrt(2.0 * h) * (s_z * xi)
            z_cand = z + drift_z + diff_z

            # Defensive guard against overflow in z
            if not np.all(np.isfinite(z_cand)) or np.any(np.abs(z_cand) > 1e70):
                return StepResult(
                    state=state.copy_with(auxiliary=current_aux),
                    accepted=False,
                    accept_prob=0.0,
                    stats={"grad_evals": 0, "log_prob_evals": 0, "overflow": 1.0, "accepted": 0.0},
                    proposal=state.x,
                    step_size=h,
                )

            # 4. Map candidate back to primal space
            x_cand, log_det_inv_cand = transform.inverse(z_cand)

            # Defensive guard against overflow in x
            if not np.all(np.isfinite(x_cand)) or np.any(np.abs(x_cand) > 1e70):
                return StepResult(
                    state=state.copy_with(auxiliary=current_aux),
                    accepted=False,
                    accept_prob=0.0,
                    stats={"grad_evals": 0, "log_prob_evals": 0, "overflow": 1.0, "accepted": 0.0},
                    proposal=x,
                    step_size=h,
                )

            # 5. Evaluate target at candidate x_cand (strictly 1 gradient evaluation)
            lp_x_cand, grad_x_cand = target.log_prob_and_grad(x_cand)

            # Defensive guard against non-finite evaluations in candidate target
            if not np.isfinite(lp_x_cand) or not np.all(np.isfinite(grad_x_cand)):
                return StepResult(
                    state=state.copy_with(auxiliary=current_aux),
                    accepted=False,
                    accept_prob=0.0,
                    stats={"grad_evals": 1, "log_prob_evals": 1, "non_finite": 1.0, "accepted": 0.0},
                    proposal=x_cand,
                    step_size=h,
                )

            # 6. Compute candidate transformed quantities
            lp_z_cand = float(lp_x_cand + log_det_inv_cand)
            grad_z_cand = transform.forward_grad(x_cand, grad_x_cand)
            b_z_cand, _ = self._gentle_tamed_drift(grad_z_cand, s_z, h)

            # 7. Exact proposal log-ratio in z-space
            delta_z = z_cand - z
            quad_cand = float(np.sum(var_z * (b_z_cand**2)))
            quad_curr = float(np.sum(var_z * (b_z**2)))
            log_q_diff = -0.5 * float(np.dot(delta_z, b_z + b_z_cand)) - (h / 4.0) * (quad_cand - quad_curr)

            # 8. Metropolis-Hastings acceptance ratio
            log_alpha = (lp_z_cand - lp_z) + log_q_diff
            if not np.isfinite(log_alpha) or log_alpha < -709.0:
                accept_prob = 0.0
            elif log_alpha >= 0.0:
                accept_prob = 1.0
            else:
                accept_prob = float(np.exp(log_alpha))

        # 9. Accept / Reject decision
        accepted = bool(rng.uniform() < accept_prob)
        if accepted:
            aux_cand = {
                "z": z_cand,
                "lp_z": lp_z_cand,
                "grad_z": grad_z_cand,
                "log_det_inv": log_det_inv_cand,
                "transform_id": transform_id,
            }
            new_state = ChainState(
                x=x_cand,
                log_prob=lp_x_cand,
                grad_log_prob=grad_x_cand,
                auxiliary=aux_cand,
            )
        else:
            new_state = state.copy_with(auxiliary=current_aux)

        stats = {
            "accept_prob": float(accept_prob),
            "grad_evals": 1,
            "log_prob_evals": 1,
            "accepted": float(accepted),
            "kappa": float(kappa_fwd),
            "step_size": float(h),
        }
        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=float(accept_prob),
            stats=stats,
            proposal=x_cand,
            step_size=h,
        )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Executes multi-phase warmup adaptation without destructive interference."""
        W = self.warmup_steps
        if W <= 0:
            # Fallback to pure dual averaging if no warmup horizon configured
            if self.adaptation is None:
                self.adaptation = DualAveragingAdaptation(
                    initial_step_size=self.step_size,
                    target_accept=self.target_accept,
                )
            self.step_size = self.adaptation.update(result.accept_prob)
            return

        W1 = max(10, int(0.15 * W))
        W2 = int(0.80 * W)
        d = len(result.state.x)

        # Phase 1: Tail Scale & Location Calibration (0 <= step_idx < W1)
        if step_idx < W1:
            if self.adapt_tail_scale:
                self._phase1_samples.append(result.state.x.copy())
            if self.adaptation is None:
                self.adaptation = DualAveragingAdaptation(
                    initial_step_size=self.step_size,
                    target_accept=self.target_accept,
                )
            self.step_size = self.adaptation.update(result.accept_prob)

            if step_idx == W1 - 1 and self.adapt_tail_scale and len(self._phase1_samples) > 0:
                stacked = np.vstack(self._phase1_samples)
                med = np.median(stacked, axis=0)
                centered = stacked - med
                q75, q25 = np.quantile(centered, [0.75, 0.25], axis=0)
                iqr_scale = (q75 - q25) / 1.3489795
                mad_scale = np.median(np.abs(centered), axis=0) / 0.67448975
                q90_abs = np.quantile(np.abs(centered), 0.90, axis=0)
                base_scale = np.where(iqr_scale > self.min_radius, iqr_scale, np.maximum(mad_scale, q90_abs))
                self.radius = np.clip(base_scale * 1.5, self.min_radius, self.max_radius)
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
            if self.adapt_preconditioner:
                if self.welford is None:
                    self.welford = WelfordPreconditioner(d=d)
                z = result.state.auxiliary.get("z")
                if z is not None:
                    self.welford.update(np.asarray(z))
                # Periodically update preconditioner
                if (step_idx - W1 + 1) % 25 == 0 or step_idx == W2 - 1:
                    self.set_preconditioner(self.welford.diagonal_scale())

            if self.adaptation is not None:
                self.step_size = self.adaptation.update(result.accept_prob)

            if step_idx == W2 - 1:
                self._preconditioner_frozen = True
                if self.adaptation is not None:
                    self.adaptation.reset()

        # Phase 3: Step-Size Fine Tuning (W2 <= step_idx < W)
        elif step_idx < W:
            if self.adaptation is not None:
                self.step_size = self.adaptation.update(result.accept_prob)
            if step_idx == W - 1:
                self.finalize_adaptation()

    def finalize_adaptation(self) -> float:
        """Freezes smoothed step size at end of warmup."""
        if self.adaptation is not None:
            self.step_size = self.adaptation.finalize()
        return self.step_size
