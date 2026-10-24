"""Unit tests for the competitor benchmark runner (bench/competitor_runner.py).

Most runner tests swap the real samplers for a fast i.i.d. fake so they check bookkeeping
(schema, aggregation, provenance, checkpoints) rather than sampler behaviour. One integration test
runs the real BlackJAX arms at smoke-test sizes, and one runs the nutpie arms when the optional
``competitors`` extra is installed. The last test checks the committed report, if present.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from scipy import stats

from mcmc_robust.bench import competitor_runner as cr
from mcmc_robust.bench.competitors import CompetitorUnavailableError, nutpie_flow_available
from mcmc_robust.bench.sota_adapters import SparseSignalRegressionTarget
from mcmc_robust.targets.benign import FullCovarianceGaussian
from mcmc_robust.targets.heavy_tailed import HeterogeneousHeavyTailedTarget, MultivariateStudentTTarget

REPORT_JSON = Path(__file__).resolve().parents[2] / "reports" / "competitor_results.json"
REPORT_CSV = REPORT_JSON.with_suffix(".csv")


# =============================================================================
# 1. Targets and exact marginal quantiles
# =============================================================================


@pytest.mark.unit
def test_rotated_covariance_has_requested_condition_number() -> None:
    cov = cr.rotated_covariance(10, 25.0, seed=0)
    np.testing.assert_allclose(cov, cov.T)
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > 0
    np.testing.assert_allclose(eig.max() / eig.min(), 25.0, rtol=1e-8)
    # Rotated: not diagonal.
    assert np.max(np.abs(cov - np.diag(np.diag(cov)))) > 0.1


@pytest.mark.unit
def test_make_target_builds_the_four_specified_targets() -> None:
    task1 = cr.make_target("task1")
    assert isinstance(task1, SparseSignalRegressionTarget)
    assert (task1.d, task1.n_obs, task1.n_active) == (10, 40, 3)

    task3 = cr.make_target("task3")
    assert isinstance(task3, HeterogeneousHeavyTailedTarget)
    assert task3.d == 20 and task3.heavy_coords == list(range(10))
    np.testing.assert_allclose(task3.nu_arr, np.linspace(1.2, 2.5, 10))

    corr_t1 = cr.make_target("corr_t1")
    assert isinstance(corr_t1, MultivariateStudentTTarget)
    assert corr_t1.d == 10 and corr_t1.nu == 1.0
    eig = np.linalg.eigvalsh(corr_t1.cov)
    np.testing.assert_allclose(eig.max() / eig.min(), 25.0, rtol=1e-6)

    corr_gauss = cr.make_target("corr_gauss")
    assert isinstance(corr_gauss, FullCovarianceGaussian)
    eig = np.linalg.eigvalsh(corr_gauss.cov)
    assert corr_gauss.d == 20
    np.testing.assert_allclose(eig.max() / eig.min(), 100.0, rtol=1e-6)

    with pytest.raises(KeyError):
        cr.make_target("no_such_target")


@pytest.mark.unit
@pytest.mark.statistical
@pytest.mark.parametrize("name", cr.TARGET_NAMES)
def test_exact_marginal_quantiles_match_ground_truth_cdf(name: str) -> None:
    target = cr.make_target(name)
    probs = (0.01, 0.25, 0.5, 0.99)
    quantiles = cr.exact_marginal_quantiles(target, probs)
    n = 200_000
    draws = target.sample_ground_truth(n, rng=np.random.default_rng(2024))
    for p in probs:
        q = quantiles[p]
        assert q.shape == (target.d,)
        cdf_hat = np.mean(draws <= q, axis=0)
        se = np.sqrt(p * (1.0 - p) / n)
        assert np.all(np.abs(cdf_hat - p) <= 5.0 * se), (name, p, cdf_hat)


@pytest.mark.unit
def test_exact_marginal_quantiles_rejects_unknown_target() -> None:
    from mcmc_robust.targets.roughness import QuarticTarget

    with pytest.raises(NotImplementedError):
        cr.exact_marginal_quantiles(QuarticTarget(d=2), (0.01,))


# =============================================================================
# 2. Tail-probability error with MCSE
# =============================================================================


def _normal_quantiles(d: int) -> dict[float, np.ndarray]:
    return {p: np.full(d, stats.norm.ppf(p)) for p in cr.TAIL_PROBS}


@pytest.mark.unit
@pytest.mark.statistical
def test_tail_probability_errors_are_within_mcse_for_iid_draws() -> None:
    draws = np.random.default_rng(7).standard_normal((4, 2000, 6))
    out = cr.tail_probability_errors(draws, _normal_quantiles(6))
    iid_se = np.sqrt(0.01 * 0.99 / draws[..., 0].size)
    for tag in ("p01", "p99"):
        assert 0.5 * iid_se < out[f"tail_prob_mcse_{tag}"] < 2.0 * iid_se
        assert out[f"tail_prob_abs_err_{tag}"] < 3.0 * iid_se
        assert out[f"tail_prob_max_abs_z_{tag}"] < 4.5
        assert out[f"tail_prob_rms_z_{tag}"] < 2.5


@pytest.mark.unit
def test_tail_probability_errors_flag_draws_that_miss_the_tails() -> None:
    # Draws with half the true scale almost never cross the true 1% and 99% quantiles.
    draws = 0.5 * np.random.default_rng(8).standard_normal((4, 1000, 3))
    out = cr.tail_probability_errors(draws, _normal_quantiles(3))
    for tag in ("p01", "p99"):
        assert out[f"tail_prob_abs_err_{tag}"] > 0.009
        assert out[f"tail_prob_max_abs_z_{tag}"] > 5.0


@pytest.mark.unit
def test_tail_probability_uses_iid_floor_when_tail_is_never_visited() -> None:
    draws = np.zeros((2, 100, 2))
    out = cr.tail_probability_errors(draws, _normal_quantiles(2))
    floor = np.sqrt(0.01 * 0.99 / 200)
    np.testing.assert_allclose(out["tail_prob_abs_err_p01"], 0.01)
    np.testing.assert_allclose(out["tail_prob_mcse_p01"], floor)
    np.testing.assert_allclose(out["tail_prob_max_abs_z_p01"], 0.01 / floor)


# =============================================================================
# 3. Aggregation and paired ratios
# =============================================================================


def _row(target: str, arm: str, seed: int, tess_per_grad: float, rhat: float = 1.001, status: str = "ok") -> dict:
    grads = 100_000
    row = {f: None for f in cr.ROW_FIELDS}
    row.update(
        {
            "target": target,
            "arm": arm,
            "seed": seed,
            "status": status,
            "total_grad_evals": grads,
            "tail_ess": tess_per_grad * grads,
            "min_tail_ess": 0.5 * tess_per_grad * grads,
            "tail_ess_per_grad": tess_per_grad,
            "min_tail_ess_per_grad": 0.5 * tess_per_grad,
            "tail_ess_per_sec": 10.0 * tess_per_grad,
            "split_rhat": rhat,
            "combined_rhat": rhat,
            "converged": rhat <= cr.RHAT_CONVERGED,
            "divergences": seed,
            "wall_clock_time": 1.0,
            "tail_prob_abs_err_p01": 0.001 * (seed + 1),
            "tail_prob_abs_err_p99": 0.002,
            "tail_prob_mcse_p01": 0.001,
            "tail_prob_mcse_p99": 0.001,
            "tail_prob_max_abs_z_p01": 1.0,
            "tail_prob_max_abs_z_p99": 2.0,
        }
    )
    return row


@pytest.mark.unit
def test_aggregate_rows_reports_median_iqr_and_paired_ratios() -> None:
    stock = [0.001, 0.002, 0.003, 0.004, 0.005]
    ratios = [2.0, 4.0, 1.0, 3.0, 5.0]
    rows = [_row("t", "stock_nuts", s, v) for s, v in enumerate(stock)]
    rows += [
        _row("t", "arcsinh_nuts", s, v * r, rhat=1.02 if s == 0 else 1.0) for s, (v, r) in enumerate(zip(stock, ratios))
    ]
    rows.append(_row("t", "nutpie_diag", 0, 0.01, status="error"))

    aggs = {(a["target"], a["arm"]): a for a in cr.aggregate_rows(rows, reference_arm="stock_nuts")}

    ref = aggs[("t", "stock_nuts")]
    assert ref["n_ok"] == 5 and ref["n_converged"] == 5
    tpg = ref["metrics"]["tail_ess_per_grad"]
    assert tpg["median"] == pytest.approx(0.003)
    assert tpg["q25"] == pytest.approx(0.002)
    assert tpg["q75"] == pytest.approx(0.004)
    assert ref["paired_ratios_vs_reference"] == {}

    arc = aggs[("t", "arcsinh_nuts")]
    assert arc["n_converged"] == 4
    assert arc["max_split_rhat"] == pytest.approx(1.02)
    ratio = arc["paired_ratios_vs_reference"]["tail_ess_per_grad"]
    assert ratio["n_pairs"] == 5
    assert ratio["seeds"] == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(ratio["values"], ratios)
    assert ratio["median"] == pytest.approx(3.0)
    assert ratio["q25"] == pytest.approx(2.0)
    assert ratio["q75"] == pytest.approx(4.0)
    assert ratio["min"] == pytest.approx(1.0) and ratio["max"] == pytest.approx(5.0)

    failed = aggs[("t", "nutpie_diag")]
    assert failed["n_seeds"] == 1 and failed["n_ok"] == 0
    assert failed["metrics"]["tail_ess_per_grad"]["median"] is None
    assert failed["paired_ratios_vs_reference"]["tail_ess_per_grad"]["n_pairs"] == 0


# =============================================================================
# 4. End-to-end runs with a fast fake sampler
# =============================================================================


def _fake_adapter(logdensity, d=None, seed=0, tune=10, draws=10, chains=2, init=None, **options) -> tuple:
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((chains, draws, d))
    telemetry: dict[str, Any] = {
        "arm": "fake",
        "seed": seed,
        "tune": tune,
        "draws": draws,
        "chains": chains,
        "max_depth": 10,
        "target_accept": 0.8,
        "total_grad_evals": 7 * chains * (tune + draws),
        "warmup_grad_evals": 7 * chains * tune,
        "draw_grad_evals": 7 * chains * draws,
        "divergences": 0,
        "warmup_divergences": 0,
        "wall_clock_time": 0.01,
        "grad_eval_accounting": "fake",
        "init_shape": None if init is None else list(np.shape(init)),
    }
    _fake_adapter.calls.append((seed, None if init is None else np.array(init)))
    return samples, telemetry


_fake_adapter.calls = []  # type: ignore[attr-defined]


@pytest.fixture
def fake_arms(monkeypatch: pytest.MonkeyPatch) -> list:
    _fake_adapter.calls.clear()  # type: ignore[attr-defined]
    arms = {
        "stock_nuts": cr.ArmSpec("stock_nuts", _fake_adapter, "fake reference"),
        "fake_b": cr.ArmSpec("fake_b", _fake_adapter, "fake competitor", tune=40),
    }
    monkeypatch.setattr(cr, "ARMS", arms)
    return _fake_adapter.calls  # type: ignore[attr-defined]


def _config(tmp_path: Path, **overrides: Any) -> cr.CompetitorBenchmarkConfig:
    base = {
        "targets": ("corr_gauss",),
        "arms": ("stock_nuts", "fake_b"),
        "seeds": (0, 1),
        "tune": 20,
        "draws": 30,
        "chains": 2,
        "output_dir": str(tmp_path),
    }
    base.update(overrides)
    return cr.CompetitorBenchmarkConfig(**base)


@pytest.mark.unit
def test_runner_writes_schema_provenance_and_shared_inits(tmp_path: Path, fake_arms: list) -> None:
    config = _config(tmp_path)
    result = cr.run_competitor_benchmark(config)

    json_path = tmp_path / "competitor_results.json"
    csv_path = tmp_path / "competitor_results.csv"
    assert json_path.exists() and csv_path.exists()
    data = json.loads(json_path.read_text())
    assert data == json.loads(json.dumps(result))
    for key in ("provenance", "config", "targets", "arms", "excluded_arms", "metric_definitions", "rows", "aggregates"):
        assert key in data

    prov = data["provenance"]
    for key in ("git_sha", "git_dirty", "timestamp_utc", "platform", "python", "packages", "command", "cpu_count"):
        assert key in prov
    for pkg in ("numpy", "jax", "blackjax", "arviz", "nutpie", "flowjax"):
        assert pkg in prov["packages"]
    assert data["config"]["seeds"] == [0, 1]
    assert "dcs" in json.dumps(data["excluded_arms"]).lower()

    rows = data["rows"]
    assert len(rows) == 1 * 2 * 2
    assert all(r["status"] == "ok" for r in rows)
    # The arm-specific warm-up length is applied and recorded.
    assert {r["tune"] for r in rows if r["arm"] == "fake_b"} == {40}
    assert {r["tune"] for r in rows if r["arm"] == "stock_nuts"} == {20}
    for r in rows:
        assert r["tail_ess_per_grad"] == pytest.approx(r["tail_ess"] / r["total_grad_evals"])
        assert r["tail_ess_per_sec"] == pytest.approx(r["tail_ess"] / r["wall_clock_time"])
        assert r["telemetry"]["grad_eval_accounting"] == "fake"

    # Same per-seed initial positions for every arm.
    by_seed: dict[int, list[np.ndarray]] = {}
    for seed, init in fake_arms:
        by_seed.setdefault(seed, []).append(init)
    assert set(by_seed) == {0, 1}
    for inits in by_seed.values():
        assert len(inits) == 2
        np.testing.assert_array_equal(inits[0], inits[1])
        assert inits[0].shape == (2, 20)
    assert not np.array_equal(by_seed[0][0], by_seed[1][0])

    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == cr.ROW_FIELDS
        assert len(list(reader)) == len(rows)

    recomputed = cr.aggregate_rows(rows, reference_arm=cr.REFERENCE_ARM)
    assert json.loads(json.dumps(recomputed)) == data["aggregates"]


@pytest.mark.unit
def test_runner_records_errors_instead_of_crashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> tuple:
        raise RuntimeError("sampler blew up")

    monkeypatch.setattr(
        cr,
        "ARMS",
        {"stock_nuts": cr.ArmSpec("stock_nuts", _fake_adapter, "ok"), "broken": cr.ArmSpec("broken", broken, "x")},
    )
    result = cr.run_competitor_benchmark(_config(tmp_path, arms=("stock_nuts", "broken"), seeds=(0,)))
    bad = [r for r in result["rows"] if r["arm"] == "broken"]
    assert len(bad) == 1 and bad[0]["status"] == "error"
    assert "sampler blew up" in bad[0]["error"]
    assert bad[0]["tail_ess_per_grad"] is None


@pytest.mark.unit
def test_checkpoint_resume_and_merge(tmp_path: Path, fake_arms: list) -> None:
    ckpt_a = tmp_path / "a.jsonl"
    ckpt_b = tmp_path / "b.jsonl"
    cr.run_competitor_benchmark(_config(tmp_path, arms=("stock_nuts",)), checkpoint=ckpt_a, write_report=False)
    n_calls = len(fake_arms)
    assert n_calls == 2
    # Resuming with a larger arm set only runs the missing (target, arm, seed) cells.
    cr.run_competitor_benchmark(_config(tmp_path), checkpoint=ckpt_a, write_report=False)
    assert len(fake_arms) == n_calls + 2
    assert not (tmp_path / "competitor_results.json").exists()

    cr.run_competitor_benchmark(_config(tmp_path, targets=("task3",)), checkpoint=ckpt_b, write_report=False)
    merged = cr.merge_checkpoints([ckpt_a, ckpt_b], output_dir=tmp_path)
    assert len(merged["rows"]) == 8
    assert merged["config"]["targets"] == ["corr_gauss", "task3"]
    assert len(merged["provenance"]["shards"]) == 2
    assert (tmp_path / "competitor_results.json").exists()


@pytest.mark.unit
def test_checkpoint_rejects_a_different_config(tmp_path: Path, fake_arms: list) -> None:
    ckpt = tmp_path / "a.jsonl"
    cr.run_competitor_benchmark(_config(tmp_path, arms=("stock_nuts",)), checkpoint=ckpt, write_report=False)
    with pytest.raises(ValueError, match="draws"):
        cr.run_competitor_benchmark(_config(tmp_path, draws=31), checkpoint=ckpt, write_report=False)


@pytest.mark.unit
def test_unavailable_competitor_fails_before_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nutpie", None)
    with pytest.raises(CompetitorUnavailableError, match="competitors"):
        cr.run_competitor_benchmark(_config(tmp_path, arms=("stock_nuts", "nutpie_diag")))
    assert not (tmp_path / "competitor_results.json").exists()


# =============================================================================
# 5. Integration with the real samplers at smoke-test sizes
# =============================================================================


@pytest.mark.unit
def test_quick_run_with_blackjax_arms(tmp_path: Path) -> None:
    config = cr.CompetitorBenchmarkConfig(
        targets=("task3",), arms=("stock_nuts", "arcsinh_nuts"), seeds=(0,), output_dir=str(tmp_path), quick=True
    )
    result = cr.run_competitor_benchmark(config)
    assert [r["status"] for r in result["rows"]] == ["ok", "ok"]
    for r in result["rows"]:
        assert r["total_grad_evals"] > 0
        assert r["tail_prob_abs_err_p01"] is not None


@pytest.mark.unit
@pytest.mark.skipif(not nutpie_flow_available(), reason="nutpie flow adaptation not installed (competitors extra)")
def test_quick_run_with_nutpie_arms(tmp_path: Path) -> None:
    config = cr.CompetitorBenchmarkConfig(
        targets=("corr_t1",),
        arms=("stock_nuts", "nutpie_diag", "nutpie_flow"),
        seeds=(1,),
        output_dir=str(tmp_path),
        quick=True,
    )
    result = cr.run_competitor_benchmark(config)
    assert [r["status"] for r in result["rows"]] == ["ok", "ok", "ok"]
    assert result["provenance"]["packages"]["nutpie"]
    flow = next(r for r in result["rows"] if r["arm"] == "nutpie_flow")
    assert flow["telemetry"]["flow_adapter_evals"] > 0


# =============================================================================
# 6. The committed report
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not REPORT_JSON.exists(), reason="reports/competitor_results.json not generated yet")
def test_committed_report_is_complete_and_self_consistent() -> None:
    data = json.loads(REPORT_JSON.read_text())
    config = data["config"]
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert (config["tune"], config["draws"], config["chains"], config["max_depth"]) == (500, 1000, 4, 10)
    assert config["quick"] is False
    assert set(config["targets"]) == set(cr.TARGET_NAMES)
    assert set(config["arms"]) == set(cr.ARM_NAMES)

    rows = data["rows"]
    cells = {(r["target"], r["arm"], r["seed"]) for r in rows}
    assert len(cells) == len(rows) == len(cr.TARGET_NAMES) * len(cr.ARM_NAMES) * 5

    prov = data["provenance"]
    assert len(prov["git_sha"]) == 40
    assert prov["packages"]["nutpie"] and prov["packages"]["blackjax"]

    # Aggregates must be recomputable from the per-seed rows they summarise.
    recomputed = cr.aggregate_rows(rows, reference_arm=data["reference_arm"])
    assert json.loads(json.dumps(recomputed)) == data["aggregates"]

    with REPORT_CSV.open() as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(rows)
    for csv_row, row in zip(csv_rows, rows):
        assert (csv_row["target"], csv_row["arm"], int(csv_row["seed"])) == (row["target"], row["arm"], row["seed"])
        assert int(csv_row["total_grad_evals"]) == row["total_grad_evals"]
