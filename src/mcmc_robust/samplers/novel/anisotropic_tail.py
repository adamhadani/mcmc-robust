"""Anisotropic Tail MALA Sampler (Feature F16)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from mcmc_robust.core.adaptation import DualAveragingAdaptation
from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform


class AnisotropicTailMALA(Sampler):
    r"""Anisotropic Tail Metropolis-Adjusted Langevin Algorithm (Feature F16).

    Resolves target flatness (§4.3 of Power & Vasdekis 2025 arXiv:2511.21563v1)
    via coordinate-wise arcsinh diffeomorphism:
        z_i = \operatorname{arcsinh}(x_i / R_i) \quad \text{for heavy coordinates}
        z_i = x_i \quad \text{for light coordinates}

    Converts vanishing polynomial tail gradients \nabla_{x_i} \log \pi \propto -1 / x_i
    into non-vanishing constant restoring drift in transformed space:
        \lim_{z_i \to \pm \infty} \nabla_{z_i} \log \pi_z(z) = -(\alpha_i - 1) \operatorname{sgn}(z_i)

    Guarantees:
    - Strictly 1 gradient evaluation per step via ChainState caching.
    - Decoupled coordinate scaling preserving benign light-tailed directions.
    - Online warmup calibration of scale parameters R_i and tail index \alpha_i.
    - Machine-precision invertibility (< 1e-12) and exact detailed balance.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        radii: float | Sequence[float] | np.ndarray | None = None,
        mask: Sequence[bool] | np.ndarray | None = None,
        preconditioner: np.ndarray | None = None,
        target_accept: float = 0.574,
        adapt_step_size: bool = False,
        calib_steps: int = 0,
        warmup_steps: int = 0,
        alpha_threshold: float = 3.5,
        quantile_ratio_threshold: float = 3.0,
        adaptation: DualAveragingAdaptation | None = None,
    ) -> None:
        r"""Initializes AnisotropicTailMALA.

        Args:
            step_size: Base step size h > 0 in transformed space.
            radii: Optional coordinate scale vector R or scalar. If None, calibrated
                during warmup or defaulted.
            mask: Optional boolean mask of shape (d,) indicating transformed coordinates.
            preconditioner: Optional 1D array of positive diagonal scale factors s.
            target_accept: Target acceptance rate for dual averaging (default: 0.574).
            adapt_step_size: Whether to adapt step size during warmup.
            calib_steps: Number of initial exploratory steps for online calibration.
            warmup_steps: Total planned warmup steps.
            alpha_threshold: Hill estimator tail index cutoff for heaviness classification.
            quantile_ratio_threshold: q99/q75 ratio threshold for heaviness classification.
            adaptation: Optional pre-configured DualAveragingAdaptation instance.
        """
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        self.target_accept: float = float(target_accept)
        self.adapt_step_size: bool = bool(adapt_step_size)
        self.calib_steps: int = int(calib_steps)
        self.warmup_steps: int = int(warmup_steps)
        self.alpha_threshold: float = float(alpha_threshold)
        self.quantile_ratio_threshold: float = float(quantile_ratio_threshold)

        self._pre_radii = radii
        self._pre_mask = mask
        if radii is not None and hasattr(radii, "__len__"):
            self._transform: CoordinateWiseArcsinhTransform | None = CoordinateWiseArcsinhTransform(
                d=len(radii),  # type: ignore[arg-type]
                radii=radii,
                mask=mask,
            )
        else:
            self._transform = None

        if adaptation is not None:
            self.adaptation: DualAveragingAdaptation | None = adaptation
        elif self.adapt_step_size:
            self.adaptation = DualAveragingAdaptation(
                initial_step_size=self.step_size,
                target_accept=self.target_accept,
            )
        else:
            self.adaptation = None

        self._warmup_buffer: list[np.ndarray] = []
        self._step_counter: int = 0

    @property
    def requires_grad(self) -> bool:
        """AnisotropicTailMALA requires first-order gradients."""
        return True

    @property
    def transform(self) -> CoordinateWiseArcsinhTransform | None:
        """Current CoordinateWiseArcsinhTransform instance."""
        return self._transform

    def _init_transform(self, d: int) -> None:
        """Instantiates transform from pre-configured parameters or defaults."""
        if self._transform is not None:
            return

        if self._pre_radii is not None:
            self._transform = CoordinateWiseArcsinhTransform(
                d=d,
                radii=self._pre_radii,
                mask=self._pre_mask,
            )
        elif self.calib_steps <= 0 and self.warmup_steps <= 0:
            # Default to unit scale transform across all coordinates
            self._transform = CoordinateWiseArcsinhTransform(
                d=d,
                radii=1.0,
                mask=self._pre_mask,
            )

    def init_state(
        self,
        x0: np.ndarray,
        target: TargetDistribution,
    ) -> ChainState:
        """Initializes ChainState with strictly 1 target evaluation."""
        x0_arr = target._validate_input(x0)
        d = target.d
        self._init_transform(d)

        lp_x, grad_x = target.log_prob_and_grad(x0_arr)

        if self._transform is not None:
            z0, _ = self._transform.forward(x0_arr)
            _, log_det_inv = self._transform.inverse(z0)
            lp_z = float(lp_x + log_det_inv)
            grad_z = self._transform.forward_grad(x0_arr, grad_x)
            aux = {"z": z0, "lp_z": lp_z, "grad_z": grad_z}
        else:
            aux = {}

        return ChainState(x=x0_arr, log_prob=lp_x, grad_log_prob=grad_x, auxiliary=aux)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        """Executes one transition step adhering strictly to 1 grad / step."""
        d = target.d
        h = self.step_size
        x = state.x
        g_x = state.grad_log_prob
        if g_x is None:
            raise RuntimeError("AnisotropicTailMALA requires cached grad_log_prob in ChainState")

        self._step_counter += 1

        # Check if online calibration is needed
        if self._transform is None:
            self._warmup_buffer.append(x.copy())
            target_calib = self.calib_steps if self.calib_steps > 0 else max(20, int(0.3 * self.warmup_steps))
            if len(self._warmup_buffer) >= target_calib:
                stacked = np.array(self._warmup_buffer)
                self._transform = CoordinateWiseArcsinhTransform.calibrate_from_samples(
                    stacked,
                    alpha_threshold=self.alpha_threshold,
                    quantile_ratio_threshold=self.quantile_ratio_threshold,
                )
                self._warmup_buffer.clear()

        # If transform is still uncalibrated, take an exploratory unconstrained step
        if self._transform is None:
            with np.errstate(over="ignore", invalid="ignore"):
                xi = rng.normal(size=d)
                s = self._preconditioner if self._preconditioner is not None else np.ones(d, dtype=np.float64)
                var = s**2
                drift_x = h * (var * g_x)
                diff_x = np.sqrt(2.0 * h) * (s * xi)
                y = x + drift_x + diff_x

                if not np.all(np.isfinite(y)) or np.any(np.abs(y) > 1e70):
                    return StepResult(
                        state=state,
                        accepted=False,
                        accept_prob=0.0,
                        stats={"grad_evals": 0, "log_prob_evals": 0, "overflow": 1.0, "accepted": 0.0},
                        proposal=y,
                        step_size=h,
                    )

                lp_y, g_y = target.log_prob_and_grad(y)
                if not np.isfinite(lp_y) or not np.all(np.isfinite(g_y)):
                    return StepResult(
                        state=state,
                        accepted=False,
                        accept_prob=0.0,
                        stats={"grad_evals": 1, "log_prob_evals": 1, "non_finite": 1.0, "accepted": 0.0},
                        proposal=y,
                        step_size=h,
                    )

                drift_y = h * (var * g_y)
                delta = y - x
                sq_fwd = float(np.sum(((delta - drift_x) ** 2) / var))
                sq_rev = float(np.sum(((-delta - drift_y) ** 2) / var))
                log_q_diff = -0.25 * (sq_rev - sq_fwd) / h
                log_alpha = (lp_y - state.log_prob) + log_q_diff

                if not np.isfinite(log_alpha) or log_alpha < -709.0:
                    accept_prob = 0.0
                elif log_alpha >= 0.0:
                    accept_prob = 1.0
                else:
                    accept_prob = float(np.exp(log_alpha))
                accepted = bool(rng.uniform() < accept_prob)
                new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=g_y) if accepted else state
                return StepResult(
                    state=new_state,
                    accepted=accepted,
                    accept_prob=accept_prob,
                    stats={"grad_evals": 1, "log_prob_evals": 1, "accepted": float(accepted)},
                    proposal=y,
                    step_size=h,
                )

        # Transformed space sampling:
        with np.errstate(over="ignore", invalid="ignore"):
            # 1. Retrieve or compute current state in transformed space
            if "z" in state.auxiliary and "lp_z" in state.auxiliary and "grad_z" in state.auxiliary:
                z = np.asarray(state.auxiliary["z"])
                lp_z = float(state.auxiliary["lp_z"])
                grad_z = np.asarray(state.auxiliary["grad_z"])
            else:
                z, _ = self._transform.forward(x)
                _, log_det_inv = self._transform.inverse(z)
                lp_z = float(state.log_prob + log_det_inv)
                grad_z = self._transform.forward_grad(x, g_x)

            s = self._preconditioner if self._preconditioner is not None else np.ones(d, dtype=np.float64)
            var = s**2

            # 2. Propose candidate in transformed space: z_cand = z + h D grad_z + sqrt(2h) s xi
            xi = rng.normal(size=d)
            drift_z = h * (var * grad_z)
            diff_z = np.sqrt(2.0 * h) * (s * xi)
            z_cand = z + drift_z + diff_z

            if not np.all(np.isfinite(z_cand)) or np.any(np.abs(z_cand) > 1e70):
                return StepResult(
                    state=state,
                    accepted=False,
                    accept_prob=0.0,
                    stats={"grad_evals": 0, "log_prob_evals": 0, "overflow": 1.0, "accepted": 0.0},
                    proposal=state.x,
                    step_size=h,
                )

            # 3. Map candidate to primal space
            x_cand, log_det_inv_cand = self._transform.inverse(z_cand)

            if not np.all(np.isfinite(x_cand)) or np.any(np.abs(x_cand) > 1e70):
                return StepResult(
                    state=state,
                    accepted=False,
                    accept_prob=0.0,
                    stats={"grad_evals": 0, "log_prob_evals": 0, "overflow": 1.0, "accepted": 0.0},
                    proposal=state.x,
                    step_size=h,
                )

            # 4. Evaluate primal target at candidate x_cand (strictly 1 gradient evaluation)
            lp_x_cand, grad_x_cand = target.log_prob_and_grad(x_cand)

            if not np.isfinite(lp_x_cand) or not np.all(np.isfinite(grad_x_cand)):
                return StepResult(
                    state=state,
                    accepted=False,
                    accept_prob=0.0,
                    stats={"grad_evals": 1, "log_prob_evals": 1, "non_finite": 1.0, "accepted": 0.0},
                    proposal=x_cand,
                    step_size=h,
                )

            # 5. Push forward candidate evaluations to z-space
            lp_z_cand = float(lp_x_cand + log_det_inv_cand)
            grad_z_cand = self._transform.forward_grad(x_cand, grad_x_cand)

            # 6. Proposal log-ratio in z-space
            drift_z_cand = h * (var * grad_z_cand)
            delta_z = z_cand - z
            r_fwd = delta_z - drift_z
            r_rev = -delta_z - drift_z_cand
            sq_fwd = float(np.sum((r_fwd**2) / var))
            sq_rev = float(np.sum((r_rev**2) / var))
            log_q_diff = -0.25 * (sq_rev - sq_fwd) / h

            # 7. Metropolis-Hastings acceptance ratio
            log_alpha = (lp_z_cand - lp_z) + log_q_diff
            if not np.isfinite(log_alpha) or log_alpha < -709.0:
                accept_prob = 0.0
            elif log_alpha >= 0.0:
                accept_prob = 1.0
            else:
                accept_prob = float(np.exp(log_alpha))

        # 8. Accept / Reject decision
        accepted = bool(rng.uniform() < accept_prob)
        if accepted:
            aux_cand = {
                "z": z_cand,
                "lp_z": lp_z_cand,
                "grad_z": grad_z_cand,
            }
            new_state = ChainState(
                x=x_cand,
                log_prob=lp_x_cand,
                grad_log_prob=grad_x_cand,
                auxiliary=aux_cand,
            )
        else:
            new_state = state

        stats: dict[str, Any] = {
            "accept_prob": float(accept_prob),
            "grad_evals": 1,
            "log_prob_evals": 1,
            "accepted": float(accepted),
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
        """Warmup adaptation hook tuning step size via Dual Averaging."""
        if self.adaptation is None:
            self.adaptation = DualAveragingAdaptation(
                initial_step_size=self.step_size,
                target_accept=self.target_accept,
            )
        self.step_size = self.adaptation.update(result.accept_prob)

    def finalize_adaptation(self) -> float:
        """Freezes and sets the smoothed step size."""
        if self.adaptation is not None:
            self.step_size = self.adaptation.finalize()
        return self.step_size
