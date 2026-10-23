"""Shared fixtures, numerical helpers, and contract models for E2E testing."""

import math

import numpy as np
import pytest
from scipy import stats

from mcmc_robust.core.base_sampler import Sampler
from mcmc_robust.core.state import ChainState, StepResult
from mcmc_robust.core.target import TargetDistribution

# -----------------------------------------------------------------------------
# Fixtures: Random Generators and Targets
# -----------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic NumPy RNG generator fixture."""
    return np.random.default_rng(42)


@pytest.fixture
def tolerance_constants() -> dict[str, float]:
    """Standardized numerical and statistical tolerance thresholds."""
    return {
        "diffeomorphism_precision": 1e-10,
        "finite_diff_grad_rel_err": 1e-5,
        "ks_min_pvalue": 0.01,
        "wasserstein_max_dist": 0.05,
        "quantile_rel_err_max": 0.08,
        "benign_parity_ratio": 0.70,
        "roughness_step_ratio": 10.0,
        "heavy_tail_ess_ratio": 2.0,
    }


# -----------------------------------------------------------------------------
# Numerical Gradient and Metric Helpers
# -----------------------------------------------------------------------------


def numerical_gradient(
    target: TargetDistribution,
    x: np.ndarray,
    h: float = 1e-6,
) -> np.ndarray:
    """Computes central finite-difference gradient of target.log_prob at x."""
    x_arr = np.asarray(x, dtype=np.float64)
    d = len(x_arr)
    num_grad = np.empty(d, dtype=np.float64)
    for i in range(d):
        e_i = np.zeros(d, dtype=np.float64)
        e_i[i] = h
        lp_plus = target.log_prob(x_arr + e_i)
        lp_minus = target.log_prob(x_arr - e_i)
        num_grad[i] = (lp_plus - lp_minus) / (2.0 * h)
    return num_grad


def max_relative_error(anal_grad: np.ndarray, num_grad: np.ndarray) -> float:
    """Computes maximum standardized relative error between analytical and numerical gradients."""
    diff = np.abs(anal_grad - num_grad)
    scale = np.maximum(np.abs(anal_grad), 1.0)
    return float(np.max(diff / scale))


def compute_bulk_ess(samples: np.ndarray) -> float:
    """Computes rank-normalized Bulk Effective Sample Size (ESS) for an MCMC chain."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    n, d = s.shape
    if n < 4:
        return float(n)

    ess_per_dim = []
    for j in range(d):
        col = s[:, j]
        # Rank-normalize
        ranks = stats.rankdata(col)
        normalized = stats.norm.ppf((ranks - 0.375) / (n + 0.25))
        var_norm = np.var(normalized, ddof=1)
        if var_norm < 1e-12:
            ess_per_dim.append(float(n))
            continue

        # Autocorrelation estimates via FFT or lag sums
        max_lag = min(n - 1, 100)
        norm_centered = normalized - np.mean(normalized)
        autocorr = np.correlate(norm_centered, norm_centered, mode="full")[n - 1 : n + max_lag]
        autocorr = autocorr / (autocorr[0] + 1e-15)

        # Truncate at first negative pair or monotonic cutoff
        tau = 1.0
        for lag in range(1, len(autocorr)):
            if autocorr[lag] < 0.05:
                break
            tau += 2.0 * autocorr[lag]
        ess_dim = n / max(1.0, tau)
        ess_per_dim.append(ess_dim)

    return float(np.min(ess_per_dim))


def compute_tail_ess(samples: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95) -> float:
    """Computes indicator-based Tail ESS (Margossian et al. 2024 / Vehtari et al. 2021)."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, np.newaxis]
    n, d = s.shape
    if n < 4:
        return float(n)

    ess_tails = []
    for j in range(d):
        col = s[:, j]
        q_low, q_high = np.quantile(col, [lower_q, upper_q])
        # Indicator transformed chains
        ind_low = (col <= q_low).astype(np.float64)
        ind_high = (col >= q_high).astype(np.float64)

        for ind in [ind_low, ind_high]:
            var_ind = np.var(ind, ddof=1)
            if var_ind < 1e-12:
                ess_tails.append(float(n))
                continue
            centered = ind - np.mean(ind)
            max_lag = min(n - 1, 80)
            ac = np.correlate(centered, centered, mode="full")[n - 1 : n + max_lag]
            ac = ac / (ac[0] + 1e-15)
            tau = 1.0
            for lag in range(1, len(ac)):
                if ac[lag] < 0.05:
                    break
                tau += 2.0 * ac[lag]
            ess_tails.append(n / max(1.0, tau))

    return float(np.min(ess_tails))


def compute_split_rhat(chains: list[np.ndarray]) -> float:
    """Computes multi-chain split Gelman-Rubin R-hat convergence diagnostic."""
    m = len(chains)
    if m < 2:
        return 1.0
    # Split each chain in half
    split_chains = []
    for c in chains:
        n = len(c)
        half = n // 2
        split_chains.append(c[:half])
        split_chains.append(c[half : 2 * half])

    m_split = len(split_chains)
    n_split = len(split_chains[0])
    if n_split < 4:
        return 1.0

    # Ensure 2D arrays
    arrs = [np.asarray(sc, dtype=np.float64) for sc in split_chains]
    d = arrs[0].shape[1] if arrs[0].ndim > 1 else 1
    if arrs[0].ndim == 1:
        arrs = [a[:, np.newaxis] for a in arrs]

    rhats = []
    for j in range(d):
        chain_means = [np.mean(a[:, j]) for a in arrs]
        chain_vars = [np.var(a[:, j], ddof=1) for a in arrs]

        # Between-chain variance B/n
        B_over_n = np.var(chain_means, ddof=1)
        # Within-chain variance W
        W = np.mean(chain_vars)
        if W < 1e-15:
            if B_over_n > 1e-12:
                rhats.append(999.0)  # Infinitely separated chains
            else:
                rhats.append(1.0)
            continue
        var_plus = ((n_split - 1) / n_split) * W + B_over_n
        rhat = math.sqrt(max(1.0, var_plus / W))
        rhats.append(rhat)

    return float(np.max(rhats))


def compute_folded_rhat(chains: list[np.ndarray]) -> float:
    """Computes folded split R-hat for detecting tail and scale variance discrepancies."""
    if len(chains) < 2:
        return 1.0
    # Fold chains around overall median: |x - median|
    all_stacked = np.vstack(chains)
    medians = np.median(all_stacked, axis=0)
    folded_chains = [np.abs(c - medians) for c in chains]
    return compute_split_rhat(folded_chains)


# -----------------------------------------------------------------------------
# Reference Sampler Contracts (Progressive Testability for M2–M5)
# -----------------------------------------------------------------------------


class RWMContractSampler(Sampler):
    """Reference Random Walk Metropolis conforming to F09 contract."""

    def __init__(self, step_size: float = 0.5, preconditioner: np.ndarray | None = None) -> None:
        super().__init__(step_size=step_size, preconditioner=preconditioner)

    @property
    def requires_grad(self) -> bool:
        return False

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        x0_arr = target._validate_input(x0)
        lp = target.log_prob(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=None)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        d = len(state.x)
        scale = np.sqrt(self.preconditioner) if self.preconditioner is not None else np.ones(d)
        noise = rng.normal(size=d) * scale * self.step_size
        y = state.x + noise

        # Exact MH ratio: log alpha = log pi(y) - log pi(x) (proposal is symmetric)
        lp_y = target.log_prob(y)
        log_alpha = lp_y - state.log_prob
        accept_prob = min(1.0, float(np.exp(min(0.0, log_alpha))))
        accepted = bool(rng.uniform() < accept_prob)

        if accepted:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=None)
            return StepResult(
                state=new_state,
                accepted=True,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha},
                proposal=y,
                step_size=self.step_size,
            )
        else:
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha},
                proposal=y,
                step_size=self.step_size,
            )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        pass


class MALAContractSampler(Sampler):
    """Reference MALA conforming to F10 contract with zero-redundancy caching."""

    def __init__(self, step_size: float = 0.1, preconditioner: np.ndarray | None = None) -> None:
        super().__init__(step_size=step_size, preconditioner=preconditioner)

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        x0_arr = target._validate_input(x0)
        lp, grad = target.log_prob_and_grad(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=grad)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        assert state.grad_log_prob is not None
        d = len(state.x)
        h2 = self.step_size**2
        p_scale = self.preconditioner if self.preconditioner is not None else np.ones(d)

        drift = 0.5 * h2 * (p_scale * state.grad_log_prob)
        noise = rng.normal(size=d) * self.step_size * np.sqrt(p_scale)
        y = state.x + drift + noise

        # Forward log transition: log q(y|x) = -1/(2 h^2) || y - x - drift ||^2_{M^-1}
        fwd_diff = (y - state.x - drift) / np.sqrt(p_scale)
        log_q_fwd = -0.5 * float(np.sum(fwd_diff**2)) / h2

        # Candidate evaluation: exactly ONE joint call
        lp_y, grad_y = target.log_prob_and_grad(y)

        # Backward log transition: log q(x|y)
        rev_drift = 0.5 * h2 * (p_scale * grad_y)
        rev_diff = (state.x - y - rev_drift) / np.sqrt(p_scale)
        log_q_rev = -0.5 * float(np.sum(rev_diff**2)) / h2

        log_alpha = (lp_y - state.log_prob) + (log_q_rev - log_q_fwd)
        accept_prob = min(1.0, float(np.exp(min(0.0, log_alpha))))
        accepted = bool(rng.uniform() < accept_prob)

        if accepted:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=grad_y)
            return StepResult(
                state=new_state,
                accepted=True,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha},
                proposal=y,
                step_size=self.step_size,
            )
        else:
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha},
                proposal=y,
                step_size=self.step_size,
            )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        pass


class TamedMALAContractSampler(Sampler):
    """Reference Globally Tamed MALA conforming to F12 contract."""

    def __init__(
        self,
        step_size: float = 0.1,
        gamma: float = 0.5,
        preconditioner: np.ndarray | None = None,
    ) -> None:
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma: float = float(gamma)

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        x0_arr = target._validate_input(x0)
        lp, grad = target.log_prob_and_grad(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=grad)

    def _tamed_drift(self, grad: np.ndarray) -> np.ndarray:
        norm_g = np.linalg.norm(grad)
        return grad / (1.0 + self.gamma * norm_g)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        assert state.grad_log_prob is not None
        d = len(state.x)
        h2 = self.step_size**2
        p_scale = self.preconditioner if self.preconditioner is not None else np.ones(d)

        b_x = self._tamed_drift(state.grad_log_prob)
        drift = 0.5 * h2 * (p_scale * b_x)
        noise = rng.normal(size=d) * self.step_size * np.sqrt(p_scale)
        y = state.x + drift + noise

        fwd_diff = (y - state.x - drift) / np.sqrt(p_scale)
        log_q_fwd = -0.5 * float(np.sum(fwd_diff**2)) / h2

        lp_y, grad_y = target.log_prob_and_grad(y)
        b_y = self._tamed_drift(grad_y)
        rev_drift = 0.5 * h2 * (p_scale * b_y)
        rev_diff = (state.x - y - rev_drift) / np.sqrt(p_scale)
        log_q_rev = -0.5 * float(np.sum(rev_diff**2)) / h2

        log_alpha = (lp_y - state.log_prob) + (log_q_rev - log_q_fwd)
        accept_prob = min(1.0, float(np.exp(min(0.0, log_alpha))))
        accepted = bool(rng.uniform() < accept_prob)

        if accepted:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=grad_y)
            return StepResult(
                state=new_state,
                accepted=True,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha, "tamed_drift_norm": np.linalg.norm(b_x)},
                proposal=y,
                step_size=self.step_size,
            )
        else:
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha, "tamed_drift_norm": np.linalg.norm(b_x)},
                proposal=y,
                step_size=self.step_size,
            )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        pass


class BarkerContractSampler(Sampler):
    """Reference Barker-Langevin Proposal conforming to F13 contract."""

    def __init__(self, step_size: float = 0.1, preconditioner: np.ndarray | None = None) -> None:
        super().__init__(step_size=step_size, preconditioner=preconditioner)

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        x0_arr = target._validate_input(x0)
        lp, grad = target.log_prob_and_grad(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=grad)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        assert state.grad_log_prob is not None
        d = len(state.x)
        p_scale = self.preconditioner if self.preconditioner is not None else np.ones(d)
        z = rng.normal(size=d)

        # Softplus skew probability p_i = 1 / (1 + exp(-h * sqrt(m_i) * z_i * grad_i))
        logits = self.step_size * np.sqrt(p_scale) * z * state.grad_log_prob
        # Numerically stable sigmoid: 1 / (1 + exp(-logits))
        p_skew = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        b = np.where(rng.uniform(size=d) < p_skew, 1.0, -1.0)
        y = state.x + self.step_size * np.sqrt(p_scale) * b * np.abs(z)

        # Transition kernel log_q ratio
        # log q(y|x) = sum_i log p_i(b_i z_i) + log_norm(z)
        lp_y, grad_y = target.log_prob_and_grad(y)

        # Reverse logits for candidate y returning to x:
        z_rev = (state.x - y) / (self.step_size * np.sqrt(p_scale))
        logits_rev = self.step_size * np.sqrt(p_scale) * z_rev * grad_y
        log_prob_fwd = np.sum(np.log(np.where(b > 0, p_skew, 1.0 - p_skew) + 1e-15))
        p_skew_rev = 1.0 / (1.0 + np.exp(-np.clip(logits_rev, -50.0, 50.0)))
        log_prob_rev = np.sum(np.log(p_skew_rev + 1e-15))

        log_alpha = (lp_y - state.log_prob) + (log_prob_rev - log_prob_fwd)
        accept_prob = min(1.0, float(np.exp(min(0.0, log_alpha))))
        accepted = bool(rng.uniform() < accept_prob)

        if accepted:
            new_state = ChainState(x=y, log_prob=lp_y, grad_log_prob=grad_y)
            return StepResult(
                state=new_state,
                accepted=True,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha},
                proposal=y,
                step_size=self.step_size,
            )
        else:
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=accept_prob,
                stats={"log_alpha": log_alpha},
                proposal=y,
                step_size=self.step_size,
            )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        pass


class HMCContractSampler(Sampler):
    """Reference HMC Sampler conforming to F11 contract."""

    def __init__(
        self,
        step_size: float = 0.05,
        n_leapfrog: int = 5,
        preconditioner: np.ndarray | None = None,
    ) -> None:
        super().__init__(step_size=step_size, preconditioner=preconditioner)
        self.n_leapfrog = max(1, int(n_leapfrog))

    def init_state(self, x0: np.ndarray, target: TargetDistribution) -> ChainState:
        x0_arr = target._validate_input(x0)
        lp, grad = target.log_prob_and_grad(x0_arr)
        return ChainState(x=x0_arr, log_prob=lp, grad_log_prob=grad)

    def step(
        self,
        state: ChainState,
        target: TargetDistribution,
        rng: np.random.Generator,
    ) -> StepResult:
        assert state.grad_log_prob is not None
        d = len(state.x)
        inv_mass = self.preconditioner if self.preconditioner is not None else np.ones(d)
        mass = 1.0 / inv_mass

        # Sample initial momentum p ~ N(0, M)
        p = rng.normal(size=d) * np.sqrt(mass)
        init_kinetic = 0.5 * float(np.sum(p**2 * inv_mass))
        init_energy = -state.log_prob + init_kinetic

        # Symplectic Leapfrog integration
        q = state.x.copy()
        current_grad = state.grad_log_prob.copy()
        eps = self.step_size

        for step_i in range(self.n_leapfrog):
            # Half step for momentum
            p = p + 0.5 * eps * current_grad
            # Full step for position
            q = q + eps * (p * inv_mass)
            # Full gradient step
            lp_q, current_grad = target.log_prob_and_grad(q)
            # Final half step for momentum
            p = p + 0.5 * eps * current_grad

        # Negate momentum for reversible involution
        p = -p
        final_kinetic = 0.5 * float(np.sum(p**2 * inv_mass))
        final_energy = -lp_q + final_kinetic
        delta_h = final_energy - init_energy

        accept_prob = min(1.0, float(np.exp(min(0.0, -delta_h))))
        accepted = bool(rng.uniform() < accept_prob)

        if accepted:
            new_state = ChainState(x=q, log_prob=lp_q, grad_log_prob=current_grad)
            return StepResult(
                state=new_state,
                accepted=True,
                accept_prob=accept_prob,
                stats={"delta_h": delta_h, "energy": final_energy},
                proposal=q,
                step_size=self.step_size,
            )
        else:
            return StepResult(
                state=state,
                accepted=False,
                accept_prob=accept_prob,
                stats={"delta_h": delta_h, "energy": init_energy},
                proposal=q,
                step_size=self.step_size,
            )

    def adapt(self, step_idx: int, result: StepResult) -> None:
        pass


# -----------------------------------------------------------------------------
# Reference Diffeomorphism Contracts (F14 & F16)
# -----------------------------------------------------------------------------


class StereographicTransformContract:
    """Exact Stereographic Projection between R^d and sphere/ball (Yang et al. 2024 / Power & Vasdekis)."""

    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        r"""Maps x \in R^d to bounded space with exact log-det Jacobian."""
        x_arr = np.asarray(x, dtype=np.float64)
        norm_sq = float(np.sum((x_arr / self.scale) ** 2))
        factor = 1.0 / math.sqrt(1.0 + norm_sq)
        y = factor * x_arr
        d = len(x_arr)
        log_det = -0.5 * (d + 2) * math.log1p(norm_sq)
        return y, log_det

    def inverse(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        r"""Inverts transformation with exact reverse log-det Jacobian."""
        y_arr = np.asarray(y, dtype=np.float64)
        norm_y_sq = float(np.sum((y_arr / self.scale) ** 2))
        norm_y_sq = min(norm_y_sq, 0.999999999999)
        factor = 1.0 / math.sqrt(1.0 - norm_y_sq)
        x = factor * y_arr
        d = len(y_arr)
        log_det = -0.5 * (d + 2) * math.log(1.0 - norm_y_sq)
        return x, log_det


class ArcsinhTransformContract:
    """Coordinate-wise arcsinh diffeomorphism for heavy tails (F16 contract)."""

    def __init__(self, radii: np.ndarray) -> None:
        self.radii = np.asarray(radii, dtype=np.float64)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        y = np.arcsinh(x / self.radii)
        # Jacobian determinant: prod_i 1 / (R_i * sqrt(1 + (x_i/R_i)^2))
        log_det = -float(np.sum(np.log(self.radii) + 0.5 * np.log1p((x / self.radii) ** 2)))
        return y, log_det

    def inverse(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        x = self.radii * np.sinh(y)
        log_det = float(np.sum(np.log(self.radii) + np.log(np.cosh(y))))
        return x, log_det
