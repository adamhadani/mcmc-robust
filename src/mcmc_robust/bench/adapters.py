"""Adapters for space-transformed samplers (Stereographic & Radial MALA)."""

import numpy as np

from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution
from mcmc_robust.samplers.mala import MALA
from mcmc_robust.transforms.base import SpaceTransformation, TransformedTarget
from mcmc_robust.transforms.radial import RadialDiffeomorphicTransform
from mcmc_robust.transforms.stereographic import StereographicTransform


class TransformedSamplerAdapter(Sampler):
    r"""Adapts any base Sampler to operate in transformed space \mathcal{Y} = \psi(\mathcal{X}).

    Maintains the zero-redundancy oracle caching protocol and pulls back candidate
    states to primal space \mathcal{X} for telemetry and ground-truth evaluation.
    """

    def __init__(
        self,
        base_sampler: Sampler,
        transform: SpaceTransformation,
        spherical_retraction: bool = False,
    ) -> None:
        super().__init__(step_size=base_sampler.step_size)
        self.base_sampler = base_sampler
        self.transform = transform
        self.spherical_retraction = spherical_retraction

    @property
    def requires_grad(self) -> bool:
        return self.base_sampler.requires_grad

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        tt = TransformedTarget(target, self.transform)
        # Map initial point to transformed space
        y0, _ = self.transform.forward(x0)
        state_y = self.base_sampler.init_state(y0, tt)
        # Store primal x0 in auxiliary dictionary
        aux = dict(state_y.auxiliary)
        aux["primal_x"] = np.asarray(x0, dtype=np.float64)
        return state_y.copy_with(auxiliary=aux)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        tt = TransformedTarget(target, self.transform)

        if self.spherical_retraction:
            # Custom projected Riemannian step on S^d
            res = self._step_spherical(state, tt, rng)
        else:
            res = self.base_sampler.step(state, tt, rng)

        # Pull back accepted/current state to primal space
        y_curr = res.state.x
        try:
            x_curr, _ = self.transform.inverse(y_curr)
        except Exception:
            x_curr = state.auxiliary.get("primal_x", np.full(self.transform.input_dim, np.nan))

        aux = dict(res.state.auxiliary)
        aux["primal_x"] = x_curr
        new_state = res.state.copy_with(auxiliary=aux)

        return StepResult(
            state=new_state,
            accepted=res.accepted,
            accept_prob=res.accept_prob,
            stats=res.stats,
            proposal=res.proposal,
            step_size=res.step_size,
        )

    def _step_spherical(
        self,
        state: ChainState,
        tt: TransformedTarget,
        rng: np.random.Generator,
    ) -> StepResult:
        """Projected tangent MALA step on S^d satisfying exact detailed balance.

        Forward proposal in tangent space T_z S^d:
            v = (h / 2) g_z^tan + sqrt(h) xi_tan, where xi_tan in T_z S^d.
        Candidate state via spherical exponential map:
            z_cand = cos(||v||) z + sin(||v||) (v / ||v||).
        Exact reverse tangent vector in T_{z_cand} S^d via spherical logarithmic map:
            v_rev = theta * (z - cos(theta) z_cand) / sin(theta),
            where cos(theta) = clip(<z_cand, z>, -1, 1), theta = arccos(cos(theta)).
        Tangent space proposal density log-ratio:
            log q(z | z_cand) - log q(z_cand | z) =
                - 1 / (2h) * [ ||v_rev - (h / 2) g_cand^tan||^2 - ||v - (h / 2) g_z^tan||^2 ].
        """
        z = state.x
        d_out = len(z)
        h = self.base_sampler.step_size
        gz = state.grad_log_prob
        if gz is None:
            raise RuntimeError("Gradient required for spherical MALA")

        # Project ambient gradient to tangent space T_z S^d
        gz_tan = gz - float(np.dot(gz, z)) * z

        # Sample ambient Gaussian noise and project to tangent space T_z S^d
        xi = rng.normal(size=d_out)
        xi_tan = xi - float(np.dot(xi, z)) * z

        # Tangent proposal velocity v in T_z S^d
        v = 0.5 * h * gz_tan + np.sqrt(h) * xi_tan

        # Geodesic spherical retraction Exp_z(v)
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-12:
            z_cand = np.cos(v_norm) * z + np.sin(v_norm) * (v / v_norm)
        else:
            z_cand = z.copy()

        # Renormalize to ensure candidate lies strictly on S^d
        cand_norm = float(np.linalg.norm(z_cand))
        if cand_norm > 1e-12:
            z_cand = z_cand / cand_norm

        # North-pole safeguard: z_{d+1} <= 1 - 1e-12
        if z_cand[-1] >= 1.0 - 1e-12:
            z_cand[-1] = 1.0 - 1e-12
            rem_sq = max(0.0, 1.0 - z_cand[-1] ** 2)
            norm_sub = float(np.linalg.norm(z_cand[:-1]))
            if norm_sub > 1e-15:
                z_cand[:-1] = z_cand[:-1] * (np.sqrt(rem_sq) / norm_sub)

        lp_cand, grad_cand = tt.log_prob_and_grad(z_cand)

        if not np.isfinite(lp_cand) or not np.all(np.isfinite(grad_cand)):
            accept_prob = 0.0
            accepted = False
        else:
            # Geodesic angle theta and reverse logarithmic map Log_{z_cand}(z)
            cos_theta = float(np.clip(np.dot(z_cand, z), -1.0, 1.0))
            theta = float(np.arccos(cos_theta))
            sin_theta = float(np.sin(theta))

            if sin_theta > 1e-12:
                v_rev = theta * (z - cos_theta * z_cand) / sin_theta
            else:
                v_rev = np.zeros_like(z)

            # Defensive projection to tangent space T_{z_cand} S^d
            v_rev = v_rev - float(np.dot(v_rev, z_cand)) * z_cand

            # Project candidate gradient to tangent space T_{z_cand} S^d
            gcand_tan = grad_cand - float(np.dot(grad_cand, z_cand)) * z_cand

            # Evaluate forward and reverse tangent Gaussian residuals
            diff_fwd = v - 0.5 * h * gz_tan
            diff_rev = v_rev - 0.5 * h * gcand_tan

            sq_fwd = float(np.dot(diff_fwd, diff_fwd))
            sq_rev = float(np.dot(diff_rev, diff_rev))

            # Tangent proposal density log-ratio (Jacobian determinants cancel out)
            log_prop_ratio = -0.5 * (sq_rev - sq_fwd) / h
            log_alpha = (lp_cand - state.log_prob) + log_prop_ratio

            if not np.isfinite(log_alpha):
                accept_prob = 0.0
            elif log_alpha >= 0.0:
                accept_prob = 1.0
            else:
                accept_prob = float(np.exp(np.clip(log_alpha, -700.0, 0.0)))

            accepted = bool(rng.uniform() < accept_prob)

        new_state = ChainState(x=z_cand, log_prob=lp_cand, grad_log_prob=grad_cand) if accepted else state
        return StepResult(
            state=new_state,
            accepted=accepted,
            accept_prob=accept_prob,
            stats={"grad_evals": 1, "log_prob_evals": 1, "accepted": float(accepted)},
            proposal=z_cand,
            step_size=h,
        )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        """Delegates step adaptation to base sampler and synchronizes step size."""
        self.base_sampler.adapt(step_idx, result)
        self.step_size = self.base_sampler.step_size

    def finalize_adaptation(self) -> float:
        """Delegates adaptation finalization to base sampler if supported."""
        if hasattr(self.base_sampler, "finalize_adaptation"):
            new_h = self.base_sampler.finalize_adaptation()
            self.step_size = new_h
            return new_h
        return self.step_size


def make_stereographic_mala(
    step_size: float = 0.1,
    dim: int = 2,
    scale: float | None = None,
) -> Sampler:
    """Instantiates Stereographic MALA with spherical retraction.

    By default, sets projection radius ``scale = sqrt(dim)`` following
    Yang, Latuszynski, & Roberts (2024, Annals of Statistics).
    """
    resolved_scale = float(np.sqrt(dim)) if scale is None else float(scale)
    transform = StereographicTransform(d=dim, scale=resolved_scale)
    base_mala = MALA(step_size=step_size)
    return TransformedSamplerAdapter(base_mala, transform, spherical_retraction=True)


def make_radial_mala(
    step_size: float = 0.1,
    dim: int = 2,
    radius: float = 3.0,
    stiffness: float = 0.5,
) -> Sampler:
    """Instantiates Radial Diffeomorphic MALA."""
    transform = RadialDiffeomorphicTransform(d=dim, radius=radius, stiffness=stiffness)
    base_mala = MALA(step_size=step_size)
    return TransformedSamplerAdapter(base_mala, transform, spherical_retraction=False)
