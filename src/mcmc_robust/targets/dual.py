"""Dual roughness-flatness pathology target distributions."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from mcmc_robust.core.target import TargetDistribution


class CoupledDualPathologyTarget(TargetDistribution):
    """Coupled target combining heavy-tailed directions and quartic rough directions.

    Coordinates 0 ... d1 - 1 are heavy-tailed Student-t directions.
    Coordinates d1 ... d - 1 are quartic rough directions coupled to heavy directions.
    """

    def __init__(
        self,
        d: int,
        nu: float = 2.0,
        c: float = 1.0,
        nu_list: Sequence[float] | None = None,
        coupling_list: Sequence[float] | None = None,
    ) -> None:
        if d < 2:
            raise ValueError(f"CoupledDualPathologyTarget requires dimension d >= 2, got {d}")
        super().__init__(d=d)

        self.d1: int = max(1, self.d // 2)
        self.d2: int = self.d - self.d1

        if nu_list is not None:
            self.nu_arr: np.ndarray = np.asarray(nu_list, dtype=np.float64)
        else:
            self.nu_arr = np.full(self.d1, float(nu), dtype=np.float64)

        if coupling_list is not None:
            self.c_arr: np.ndarray = np.asarray(coupling_list, dtype=np.float64)
        else:
            self.c_arr = np.full(self.d2, float(c), dtype=np.float64)

        # Mapping for coupling: rough coordinate j (index d1 + k) couples to heavy coordinate pi(k)
        self.pi_map: list[int] = [k % self.d1 for k in range(self.d2)]

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf

        # 1. Heavy potential components
        u_heavy = 0.0
        for i in range(self.d1):
            nu_i = self.nu_arr[i]
            x_i = x_arr[i]
            u_heavy += 0.5 * (nu_i + 1.0) * np.log1p((x_i**2) / nu_i)

        # 2. Quartic and quadratic coupling components
        u_rough = 0.0
        for k in range(self.d2):
            j = self.d1 + k
            x_j = x_arr[j]
            x_coupled = x_arr[self.pi_map[k]]
            c_j = self.c_arr[k]
            # Safeguard quartic term against extreme overflow
            if abs(x_j) > 1e25:
                return -np.inf
            u_rough += 0.25 * (x_j**4) + 0.5 * c_j * ((x_j - x_coupled) ** 2)

        return float(-(u_heavy + u_rough))

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)

        grad = np.zeros(self.d, dtype=np.float64)

        # 1. Heavy coordinates drift
        for i in range(self.d1):
            nu_i = self.nu_arr[i]
            x_i = x_arr[i]
            grad[i] = -((nu_i + 1.0) / (nu_i + (x_i**2))) * x_i

        # 2. Rough coordinates drift & coupled back-forces
        for k in range(self.d2):
            j = self.d1 + k
            i_coupled = self.pi_map[k]
            x_j = x_arr[j]
            x_i = x_arr[i_coupled]
            c_j = self.c_arr[k]

            diff = x_j - x_i
            # Force on heavy coord from rough coord
            grad[i_coupled] += c_j * diff

            # Force on rough coord
            if abs(x_j) > 1e30:
                q_grad = np.sign(x_j) * 1e90
            else:
                q_grad = x_j**3
            grad[j] = -(q_grad + c_j * diff)

        return grad

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return self.log_prob(x), self.grad_log_prob(x)

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
        max_batches: int = 100,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        if n <= 0:
            return np.empty((0, self.d), dtype=np.float64)

        # If coupling is zero, sample exactly from uncoupled independent marginals
        all_zero_coupling = np.all(self.c_arr == 0.0)
        if all_zero_coupling:
            samples = np.empty((n, self.d), dtype=np.float64)
            for i in range(self.d1):
                samples[:, i] = rng.standard_t(df=self.nu_arr[i], size=n)
            for k in range(self.d2):
                j = self.d1 + k
                y = rng.gamma(shape=0.25, scale=1.0, size=n)
                r = (4.0 * y) ** 0.25
                signs = rng.choice([-1.0, 1.0], size=n)
                samples[:, j] = signs * r
            return samples

        samples = np.empty((n, self.d), dtype=np.float64)

        # Block-wise rejection sampling across independent coupled blocks
        for i in range(self.d1):
            rough_k = [k for k in range(self.d2) if self.pi_map[k] == i]
            n_rough = len(rough_k)
            if n_rough == 0:
                samples[:, i] = rng.standard_t(df=self.nu_arr[i], size=n)
                continue

            count = 0
            block_heavy = np.empty(n, dtype=np.float64)
            block_rough = np.empty((n, n_rough), dtype=np.float64)
            batches = 0

            while count < n and batches < max_batches:
                batches += 1
                batch_size = max(int((n - count) * 2.5), 100)

                # 1. Propose heavy coordinate from Student-t
                cand_heavy = rng.standard_t(df=self.nu_arr[i], size=batch_size)

                # 2. Propose rough coordinates from Quartic distribution
                cand_rough = np.empty((batch_size, n_rough), dtype=np.float64)
                block_pen = np.zeros(batch_size, dtype=np.float64)

                for idx, k in enumerate(rough_k):
                    y = rng.gamma(shape=0.25, scale=1.0, size=batch_size)
                    r = (4.0 * y) ** 0.25
                    signs = rng.choice([-1.0, 1.0], size=batch_size)
                    cand_rough[:, idx] = signs * r
                    diff = cand_rough[:, idx] - cand_heavy
                    block_pen += 0.5 * self.c_arr[k] * (diff**2)

                # 3. Accept/reject on block coupling penalty
                prob = np.exp(-np.clip(block_pen, 0.0, 100.0))
                acc = rng.uniform(size=batch_size) <= prob

                n_acc = int(np.count_nonzero(acc))
                if n_acc > 0:
                    take = min(n - count, n_acc)
                    block_heavy[count : count + take] = cand_heavy[acc][:take]
                    block_rough[count : count + take] = cand_rough[acc][:take]
                    count += take

            # 4. Graceful fallback if max_batches reached before n accepted (extreme coupling)
            if count < n:
                deficit = n - count
                cand_heavy_fb = rng.standard_t(df=self.nu_arr[i], size=deficit)
                cand_rough_fb = np.empty((deficit, n_rough), dtype=np.float64)
                for idx, k in enumerate(rough_k):
                    c_k = self.c_arr[k]
                    if c_k > 0:
                        cand_rough_fb[:, idx] = cand_heavy_fb + rng.normal(scale=1.0 / np.sqrt(c_k), size=deficit)
                    else:
                        y = rng.gamma(shape=0.25, scale=1.0, size=deficit)
                        r = (4.0 * y) ** 0.25
                        signs = rng.choice([-1.0, 1.0], size=deficit)
                        cand_rough_fb[:, idx] = signs * r
                block_heavy[count:n] = cand_heavy_fb
                block_rough[count:n] = cand_rough_fb

            # Store accepted draws into samples array
            samples[:, i] = block_heavy
            for idx, k in enumerate(rough_k):
                samples[:, self.d1 + k] = block_rough[:, idx]

        return samples

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {
            "mean": np.zeros(self.d, dtype=np.float64),
            "d1": self.d1,
            "d2": self.d2,
            "nu": self.nu_arr,
        }


# Alias
DualRoughFlatTarget = CoupledDualPathologyTarget


class ConfinementInducedRoughnessTarget(TargetDistribution):
    """Target modeling pushforward gradient explosion under aggressive space transformation.

    Transformed potential corresponding to base X ~ N(0, I_d) under
    y_i = sign(x_i) * log(1 + |x_i|).
    """

    def __init__(
        self,
        d: int,
        base_scale: float = 1.0,
    ) -> None:
        super().__init__(d=d)
        if base_scale <= 0.0:
            raise ValueError(f"base_scale must be positive, got {base_scale}")
        self.base_scale: float = float(base_scale)

    MAX_Y: float = 350.0
    LARGE_GRAD: float = 1e300

    def log_prob(self, y: np.ndarray) -> float:
        y_arr = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y_arr)):
            return -np.inf
        abs_y = np.abs(y_arr)
        if np.any(abs_y > self.MAX_Y):
            return -np.inf

        # x_i = sign(y_i) * (exp(|y_i|) - 1) using high-precision expm1
        x = np.sign(y_arr) * np.expm1(abs_y)
        sq_norm = float(np.dot(x, x))
        lp_x = (
            -0.5 * self.d * np.log(2.0 * np.pi)
            - self.d * np.log(self.base_scale)
            - 0.5 * sq_norm / (self.base_scale**2)
        )
        lp = lp_x + float(np.sum(abs_y))
        return float(lp)

    def grad_log_prob(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        abs_y = np.abs(y_arr)

        grad = np.empty(self.d, dtype=np.float64)
        normal_mask = abs_y <= self.MAX_Y
        overflow_mask = ~normal_mask

        if np.any(normal_mask):
            ay = abs_y[normal_mask]
            exp_y = np.exp(ay)
            drift = ((exp_y**2 - exp_y) / (self.base_scale**2)) - 1.0
            grad[normal_mask] = -np.sign(y_arr[normal_mask]) * drift

        if np.any(overflow_mask):
            grad[overflow_mask] = -np.sign(y_arr[overflow_mask]) * self.LARGE_GRAD

        return grad

    def log_prob_and_grad(self, y: np.ndarray) -> tuple[float, np.ndarray]:
        y_arr = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y_arr)):
            return -np.inf, np.full(self.d, np.nan, dtype=np.float64)
        abs_y = np.abs(y_arr)

        grad = np.empty(self.d, dtype=np.float64)
        normal_mask = abs_y <= self.MAX_Y
        overflow_mask = ~normal_mask

        if np.any(overflow_mask):
            grad[overflow_mask] = -np.sign(y_arr[overflow_mask]) * self.LARGE_GRAD
            if np.any(normal_mask):
                ay = abs_y[normal_mask]
                exp_y = np.exp(ay)
                drift = ((exp_y**2 - exp_y) / (self.base_scale**2)) - 1.0
                grad[normal_mask] = -np.sign(y_arr[normal_mask]) * drift
            return -np.inf, grad

        x = np.sign(y_arr) * np.expm1(abs_y)
        sq_norm = float(np.dot(x, x))
        lp_x = (
            -0.5 * self.d * np.log(2.0 * np.pi)
            - self.d * np.log(self.base_scale)
            - 0.5 * sq_norm / (self.base_scale**2)
        )
        lp = lp_x + float(np.sum(abs_y))

        exp_y = np.exp(abs_y)
        drift = ((exp_y**2 - exp_y) / (self.base_scale**2)) - 1.0
        grad = -np.sign(y_arr) * drift
        return float(lp), grad

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Exact pushforward: sample X ~ N(0, base_scale^2 I), y = sign(x) * log(1 + |x|)
        x = rng.normal(loc=0.0, scale=self.base_scale, size=(n, self.d))
        return np.sign(x) * np.log1p(np.abs(x))

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        # Analytical marginal CDF: F_Y(y) = Phi( sign(y) * (exp(|y|) - 1) / base_scale )
        # Analytical quantiles: q_p = sign(z_p) * log(1 + |z_p| * base_scale)
        z05 = stats.norm.ppf(0.05)
        z95 = stats.norm.ppf(0.95)
        z99 = stats.norm.ppf(0.99)
        q05 = np.sign(z05) * np.log1p(np.abs(z05) * self.base_scale)
        q95 = np.sign(z95) * np.log1p(np.abs(z95) * self.base_scale)
        q99 = np.sign(z99) * np.log1p(np.abs(z99) * self.base_scale)
        return {
            "mean": np.zeros(self.d, dtype=np.float64),
            "base_scale": self.base_scale,
            "q05": np.full(self.d, q05, dtype=np.float64),
            "q50": np.zeros(self.d, dtype=np.float64),
            "q95": np.full(self.d, q95, dtype=np.float64),
            "q99": np.full(self.d, q99, dtype=np.float64),
        }


class ConfinementRoughTarget(TargetDistribution):
    """Heavy-tailed target with softplus boundary confinement penalty."""

    def __init__(
        self,
        d: int,
        nu: float = 2.0,
        lambda_param: float = 1.0,
        beta: float = 2.0,
        radius: float = 3.0,
    ) -> None:
        super().__init__(d=d)
        self.nu: float = float(nu)
        self.lambda_param: float = float(lambda_param)
        self.beta: float = float(beta)
        self.radius: float = float(radius)

    def log_prob(self, x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return -np.inf
        # Heavy-tailed Student-t base
        u_base = 0.5 * (self.nu + 1.0) * np.sum(np.log1p((x_arr**2) / self.nu))
        # Softplus penalty: softplus(beta * (|x_i| - R))^2
        arg = self.beta * (np.abs(x_arr) - self.radius)
        # softplus(z) = log1p(exp(z))
        sp = np.where(arg > 30.0, arg, np.log1p(np.exp(np.clip(arg, -50.0, 30.0))))
        u_pen = self.lambda_param * float(np.sum(sp**2))
        return float(-(u_base + u_pen))

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            return np.full(self.d, np.nan, dtype=np.float64)
        # Base gradient
        grad_base = -((self.nu + 1.0) / (self.nu + (x_arr**2))) * x_arr
        # Softplus gradient: 2 * lambda * sp * sigmoid(arg) * beta * sign(x)
        arg = self.beta * (np.abs(x_arr) - self.radius)
        sp = np.where(arg > 30.0, arg, np.log1p(np.exp(np.clip(arg, -50.0, 30.0))))
        sig = 1.0 / (1.0 + np.exp(-np.clip(arg, -50.0, 50.0)))
        grad_pen = -2.0 * self.lambda_param * self.beta * np.sign(x_arr) * sp * sig
        return grad_base + grad_pen

    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return self.log_prob(x), self.grad_log_prob(x)

    def sample_ground_truth(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if n <= 0:
            return np.empty((0, self.d), dtype=np.float64)

        # Potential is coordinate-wise separable; sample coordinate-by-coordinate in 1D
        samples = np.empty((n, self.d), dtype=np.float64)
        for j in range(self.d):
            coord: list[float] = []
            while len(coord) < n:
                batch_size = max(n - len(coord), 200)
                cand_1d = rng.standard_t(df=self.nu, size=batch_size)
                arg = self.beta * (np.abs(cand_1d) - self.radius)
                sp = np.where(arg > 30.0, arg, np.log1p(np.exp(np.clip(arg, -50.0, 30.0))))
                prob = np.exp(-self.lambda_param * (sp**2))
                accepted = cand_1d[rng.uniform(size=batch_size) <= prob]
                coord.extend(accepted)
            samples[:, j] = coord[:n]
        return samples

    def exact_moments_or_quantiles(self) -> dict[str, Any]:
        return {"mean": np.zeros(self.d, dtype=np.float64)}
