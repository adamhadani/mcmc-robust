"""Tests for the posteriordb benchmark runner (`mcmc_robust.bench.posteriordb_runner`).

`ScaleModel` is a NumPy object with the `bridgestan.StanModel` methods the runner calls, for
mu ~ N(0, 1), tau ~ LogNormal(0, 1) and a transformed parameter tau2 = tau^2. Its unconstrained
density is N(0, I) in (mu, log tau), and its reference draws are exact, so no Stan is needed.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mcmc_robust.bench import posteriordb_runner as pr
from mcmc_robust.diagnostics.tail_accuracy import reference_agreement
from mcmc_robust.diagnostics.tail_alignment import tail_alignment_diagnostic


class ScaleModel:
    def __init__(self, bound: float = np.inf) -> None:
        self.bound = bound

    def param_unc_num(self) -> int:
        return 2

    def param_names(self, include_tp: bool = False, include_gq: bool = False) -> list[str]:
        return ["mu", "tau", "tau2"] if include_tp else ["mu", "tau"]

    def _check(self, theta: np.ndarray) -> None:
        if theta[0] >= self.bound:
            raise RuntimeError("domain error")

    # Stan's C++ overflows to inf silently; NumPy would warn, and tests run with -W error.
    def log_density(self, theta: np.ndarray, *, propto: bool = True, jacobian: bool = True) -> float:
        self._check(theta)
        with np.errstate(over="ignore"):
            return float(-0.5 * np.sum(theta**2))

    def log_density_gradient(
        self, theta: np.ndarray, *, propto: bool = True, jacobian: bool = True
    ) -> tuple[float, np.ndarray]:
        self._check(theta)
        with np.errstate(over="ignore"):
            return float(-0.5 * np.sum(theta**2)), -np.asarray(theta, dtype=np.float64)

    def param_constrain(self, theta_unc: np.ndarray, include_tp: bool = False, include_gq: bool = False) -> np.ndarray:
        with np.errstate(over="ignore"):
            tau = np.exp(theta_unc[1])
            return np.array([theta_unc[0], tau, tau**2] if include_tp else [theta_unc[0], tau])


def exact_reference(seed: int = 0, names: tuple[str, ...] = ("tau2", "mu")) -> np.ndarray:
    """Exact iid draws, 10 chains x 1000, of the named constrained quantities."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(10, 1000))
    tau = np.exp(rng.normal(size=(10, 1000)))
    cols = {"mu": mu, "tau": tau, "tau2": tau**2}
    return np.stack([cols[n] for n in names], axis=-1)


def scale_posterior(name: str = "toy-scale", bound: float = np.inf) -> pr.LoadedPosterior:
    return pr.LoadedPosterior.from_model(name, ScaleModel(bound), ("tau2", "mu"), exact_reference())


QUICK = {"seeds": (0,), "tune": 100, "draws": 300, "chains": 2}


@pytest.mark.unit
def test_reference_columns_are_matched_by_name_including_transformed_parameters() -> None:
    post = scale_posterior()
    assert post.columns == (2, 0)
    assert post.d == 2
    assert post.k == 2
    with pytest.raises(ValueError, match="sigma"):
        pr.LoadedPosterior.from_model("x", ScaleModel(), ("mu", "sigma"), exact_reference(names=("mu", "tau")))


@pytest.mark.unit
def test_constrain_maps_unconstrained_draws_to_the_reference_columns() -> None:
    post = scale_posterior()
    unc = np.random.default_rng(1).normal(size=(2, 5, 2))
    out = post.constrain(unc)
    assert out.shape == (2, 5, 2)
    np.testing.assert_allclose(out[..., 0], np.exp(unc[..., 1]) ** 2)
    np.testing.assert_allclose(out[..., 1], unc[..., 0])


@pytest.mark.unit
def test_initial_positions_are_stan_style_shared_and_finite() -> None:
    """Uniform(-2, 2) per chain from the seed, redrawn until the log density is finite (Stan's rule)."""
    post = scale_posterior(bound=0.0)
    x0 = pr.initial_positions(post.model, seed=3, chains=4)
    assert x0.shape == (4, 2)
    assert np.all(np.abs(x0) <= 2.0)
    assert np.all(x0[:, 0] < 0.0)
    np.testing.assert_array_equal(x0, pr.initial_positions(post.model, seed=3, chains=4))
    assert not np.array_equal(x0, pr.initial_positions(post.model, seed=4, chains=4))
    with pytest.raises(RuntimeError, match="100"):
        pr.initial_positions(ScaleModel(bound=-5.0), seed=0, chains=1)


@pytest.mark.unit
def test_initial_positions_skip_a_non_finite_density_that_does_not_raise() -> None:
    """A Stan model can return -inf (target += negative_infinity()) instead of throwing; such points are redrawn."""

    class MinusInfinity(ScaleModel):
        def log_density_gradient(self, theta, **kwargs):  # type: ignore[no-untyped-def,override]
            value, grad = super().log_density_gradient(theta, **kwargs)
            return (-np.inf if theta[0] >= 0.0 else value), grad

    x0 = pr.initial_positions(MinusInfinity(), seed=3, chains=4)
    assert np.all(x0[:, 0] < 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("arm", ["stock", "arcsinh", "tukey_h"])
def test_rows_score_constrained_draws_against_the_reference(monkeypatch: pytest.MonkeyPatch, arm: str) -> None:
    """Accuracy and ESS use the constrained reference columns; the tail-alignment check uses the sampled draws."""
    results: list[Any] = []
    real = pr.sample_blackjax_unified_nuts

    def spy(*args: Any, **kwargs: Any) -> Any:
        results.append(real(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(pr, "sample_blackjax_unified_nuts", spy)
    post = scale_posterior()
    row = pr.run_single(post, arm, 0, pr.PosteriorDBConfig.quick(**QUICK))
    assert row["error"] is None, row["error"]
    assert list(row) == list(pr.ROW_FIELDS)
    (res,) = results
    assert res.tail_alignment is None

    constrained = post.constrain(res.samples)
    agreement = reference_agreement(constrained, post.reference_draws).summary()
    for key, value in agreement.items():
        assert row[key] == pytest.approx(value, rel=1e-12), key
    alignment = tail_alignment_diagnostic(res.samples)
    assert row["tail_alignment_n_heavy"] == len(alignment.heavy_coordinates)
    assert row["tail_alignment_warning"] is alignment.warn
    assert row["total_grad_evals"] == res.grad_evals
    assert row["divergences"] == res.divergences
    assert (row["d"], row["k"], row["posterior"], row["arm"], row["seed"]) == (2, 2, "toy-scale", arm, 0)
    if arm == "stock":
        assert row["calib_location_offset_max"] is None and row["calib_scale_ratio_max"] is None
    else:
        # How far Phase 1's calibration sits from the run's own post-warm-up draws, per unconstrained coordinate.
        x = res.samples.reshape(-1, 2)
        med = np.median(x, axis=0)
        scale = (np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)) / pr.IQR_TO_SD
        mu, radius = np.asarray(res.transform.location), np.asarray(res.transform.radius)
        assert row["calib_location_offset_max"] == pytest.approx(float(np.max(np.abs(mu - med) / scale)))
        assert row["calib_scale_ratio_max"] == pytest.approx(float(np.max(np.maximum(radius / scale, scale / radius))))
    if arm == "tukey_h":
        h = np.asarray(res.transform.h)
        assert row["tukey_h_n_positive"] == int(np.sum(h > 0.0))
        assert row["tukey_h_max"] == pytest.approx(float(np.max(h)))
    else:
        assert row["tukey_h_n_positive"] is None and row["tukey_h_max"] is None


class HeavyNuisanceModel(ScaleModel):
    """mu ~ N(0, 1) and an unconstrained nuisance nu ~ Cauchy; only mu has reference draws."""

    def param_names(self, include_tp: bool = False, include_gq: bool = False) -> list[str]:
        return ["mu", "nu"]

    def log_density_gradient(
        self, theta: np.ndarray, *, propto: bool = True, jacobian: bool = True
    ) -> tuple[float, np.ndarray]:
        mu, nu = theta
        return float(-0.5 * mu**2 - np.log1p(nu**2)), np.array([-mu, -2.0 * nu / (1.0 + nu**2)])

    def log_density(self, theta: np.ndarray, *, propto: bool = True, jacobian: bool = True) -> float:
        return self.log_density_gradient(theta)[0]

    def param_constrain(self, theta_unc: np.ndarray, include_tp: bool = False, include_gq: bool = False) -> np.ndarray:
        return np.asarray(theta_unc, dtype=np.float64)


@pytest.mark.unit
@pytest.mark.statistical
def test_the_tail_alignment_check_sees_the_sampled_draws_not_only_the_scored_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check runs on every sampled coordinate, as the library's warning would, not only the reference ones.

    The Cauchy nuisance (true tail index 1, cut-off 4) is classed heavy; the scored Gaussian mu is not.
    """
    results: list[Any] = []
    real = pr.sample_blackjax_unified_nuts

    def spy(*args: Any, **kwargs: Any) -> Any:
        results.append(real(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(pr, "sample_blackjax_unified_nuts", spy)
    reference = np.random.default_rng(0).normal(size=(10, 1000, 1))
    post = pr.LoadedPosterior.from_model("heavy-nuisance", HeavyNuisanceModel(), ("mu",), reference)
    row = pr.run_single(post, "stock", 0, pr.PosteriorDBConfig.quick(seeds=(0,), tune=200, draws=600, chains=2))
    assert row["error"] is None, row["error"]
    (res,) = results
    assert row["tail_alignment_n_heavy"] == len(tail_alignment_diagnostic(res.samples).heavy_coordinates) >= 1


@pytest.mark.unit
def test_a_failed_run_is_an_error_row_not_an_exception() -> None:
    """A posterior that cannot be initialised yields a row with its error message and converged False."""
    post = scale_posterior(bound=-5.0)
    row = pr.run_single(post, "stock", 0, pr.PosteriorDBConfig.quick(**QUICK))
    assert row["error"] is not None and "RuntimeError" in row["error"]
    assert row["converged"] is False


def _row(posterior: str, arm: str, seed: int, **metrics: float) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(pr.ROW_FIELDS)
    row.update(posterior=posterior, arm=arm, seed=seed, max_rhat=1.0, error=None, **metrics)
    return row


@pytest.mark.unit
def test_aggregates_pair_ratios_by_seed_against_stock_and_arcsinh() -> None:
    rows = [
        _row("p", arm, seed, min_tail_ess_per_grad=value, tail_ess_per_grad=value)
        for arm, values in (("stock", [1.0, 2.0, 4.0]), ("arcsinh", [2.0, 2.0, 2.0]), ("tukey_h", [4.0, 1.0, 8.0]))
        for seed, value in enumerate(values)
    ]
    agg = {(a["posterior"], a["arm"]): a for a in pr.aggregate_rows(rows)}
    assert agg[("p", "stock")]["metrics"]["min_tail_ess_per_grad"]["median"] == 2.0
    tukey = agg[("p", "tukey_h")]
    # Per seed: tukey/stock = 4, 0.5, 2 and tukey/arcsinh = 2, 0.5, 4.
    assert tukey["ratio_vs_stock"]["min_tail_ess_per_grad"]["median"] == 2.0
    assert tukey["ratio_vs_arcsinh"]["min_tail_ess_per_grad"]["median"] == 2.0
    assert tukey["ratio_vs_arcsinh"]["min_tail_ess_per_grad"]["n"] == 3
    assert "ratio_vs_stock" not in agg[("p", "stock")]
    assert "ratio_vs_arcsinh" not in agg[("p", "arcsinh")]


@pytest.mark.unit
def test_run_benchmark_writes_results_with_provenance(tmp_path: Path) -> None:
    config = pr.PosteriorDBConfig.quick(
        posteriors=("toy-scale",), arms=("stock", "tukey_h"), output_dir=str(tmp_path), **QUICK
    )
    result = pr.run_benchmark(config, loader=lambda name: scale_posterior(name), posteriordb_sha="0" * 40)
    data = json.loads((tmp_path / "posteriordb_results.json").read_text())
    assert data == json.loads(json.dumps(result))
    assert set(data) >= {"schema_version", "provenance", "config", "rows", "aggregates", "notes"}
    assert [(r["posterior"], r["arm"]) for r in data["rows"]] == [("toy-scale", "stock"), ("toy-scale", "tukey_h")]
    assert all(r["error"] is None for r in data["rows"])
    prov = data["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_sha"])
    assert prov["posteriordb_sha"] == "0" * 40
    assert "bridgestan" in prov["versions"]
    assert data["config"]["posteriors"] == ["toy-scale"]
    assert data["config"]["reference_probs"] == list(pr.REFERENCE_PROBS)
    with (tmp_path / "posteriordb_results.csv").open() as fh:
        assert list(csv.DictReader(fh).fieldnames or []) == list(pr.ROW_FIELDS)


@pytest.mark.unit
def test_full_config_is_the_reported_protocol() -> None:
    full = pr.PosteriorDBConfig.full()
    assert (full.tune, full.draws, full.chains, full.max_num_doublings) == (500, 1000, 4, 10)
    assert full.arms == ("stock", "arcsinh", "tukey_h")
    assert full.posteriors is None
    assert full.mode == "full"
    with pytest.raises(ValueError, match="scale_mixture"):
        pr.PosteriorDBConfig.full(arms=("scale_mixture",)).validate()


@pytest.mark.unit
def test_cli_arguments() -> None:
    cfg = pr.config_from_args(
        pr.build_parser().parse_args(
            ["--quick", "--seeds", "3", "4", "--posteriors", "a-b", "--arms", "stock", "--workers", "3"]
        )
    )
    assert cfg.mode == "quick"
    assert cfg.seeds == (3, 4)
    assert cfg.posteriors == ("a-b",)
    assert cfg.arms == ("stock",)
    assert cfg.workers == 3
