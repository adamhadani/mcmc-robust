"""Tests for the multi-seed arcsinh ablation runner (`mcmc_robust.bench.ablation_runner`)."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mcmc_robust.bench import ablation_runner as ar
from mcmc_robust.targets.ablation import make_ablation_target

QUICK_TARGETS = ("mvt_iso_nu2", "gauss_aniso")


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    """One quick-mode run on a heavy-tailed and a Gaussian target, every arm, written to a temp dir."""
    out = tmp_path_factory.mktemp("ablation_quick")
    config = ar.AblationConfig.quick(targets=QUICK_TARGETS, output_dir=str(out))
    result = ar.run_ablation(config)
    return result, out


@pytest.mark.unit
def test_quick_config_is_small_and_full_config_matches_the_protocol() -> None:
    """--quick uses 2 seeds and small draws; --full uses 10 seeds and the specified sampler settings."""
    quick = ar.AblationConfig.quick()
    assert quick.mode == "quick"
    assert len(quick.seeds) == 2
    assert quick.draws < 1000 and quick.tune < 500
    full = ar.AblationConfig.full()
    assert full.mode == "full"
    assert full.seeds == tuple(range(10))
    assert (full.tune, full.draws, full.chains, full.max_num_doublings) == (500, 1000, 4, 10)
    assert full.targets == ar.ABLATION_TARGET_NAMES
    assert set(full.arms) == {"stock", "stock_ta95", "arcsinh", "affine", "scale_mixture"}
    assert full.tail_probs == (0.01, 0.99)
    assert ar.ARMS["stock"].target_accept == 0.80
    assert ar.ARMS["stock_ta95"].target_accept == 0.95
    assert ar.ARMS["arcsinh"].transform_kind == "arcsinh"
    assert ar.ARMS["affine"].transform_kind == "affine"
    assert ar.ARMS["stock"].transform_kind == ar.ARMS["scale_mixture"].transform_kind == "identity"


@pytest.mark.unit
def test_quick_mode_schema(quick_run: tuple[dict[str, Any], Path]) -> None:
    """JSON and CSV outputs hold one row per applicable (target, arm, seed) with every field, and no errors."""
    result, out = quick_run
    data = json.loads((out / "ablation_results.json").read_text())
    assert data == json.loads(json.dumps(result))
    assert set(data) >= {"schema_version", "provenance", "config", "rows", "aggregates"}
    rows = data["rows"]
    # scale_mixture applies to the t target only.
    expected = {(t, a, s) for t in QUICK_TARGETS for a in ar.ARM_NAMES for s in (0, 1)}
    expected -= {("gauss_aniso", "scale_mixture", s) for s in (0, 1)}
    assert {(r["target"], r["arm"], r["seed"]) for r in rows} == expected
    assert len(rows) == len(expected)
    for r in rows:
        assert list(r) == list(ar.ROW_FIELDS)
        assert r["error"] is None
        assert r["total_grad_evals"] == r["warmup_grad_evals"] + r["sampling_grad_evals"] > 0
        for key in ("tail_ess_per_grad", "ess_per_grad", "max_rhat", "tail_prob_abs_err", "tail_prob_abs_z"):
            assert isinstance(r[key], float) and np.isfinite(r[key]), (key, r)
        assert r["converged"] == (r["max_rhat"] <= ar.RHAT_CONVERGED)
        assert r["d"] == make_ablation_target(r["target"]).d
        assert r["d_sampled"] == (r["d"] + 1 if r["arm"] == "scale_mixture" else r["d"])

    with (out / "ablation_results.csv").open() as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(rows)
    assert list(csv_rows[0]) == list(ar.ROW_FIELDS)
    with (out / "ablation_summary.csv").open() as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == len(data["aggregates"]) == len({(t, a) for t, a, _ in expected})


@pytest.mark.unit
def test_provenance_fields(quick_run: tuple[dict[str, Any], Path]) -> None:
    """Provenance records the git SHA and dirty flag, platform, library versions, config, seeds and a timestamp."""
    result, _ = quick_run
    prov = result["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_sha"])
    assert isinstance(prov["git_dirty"], bool)
    for key in ("platform", "machine", "python_implementation", "cpu_count", "jax_backend"):
        assert prov[key], key
    assert prov["jax_enable_x64"] is True
    for lib in ("python", "numpy", "scipy", "jax", "jaxlib", "blackjax", "arviz", "mcmc_robust"):
        assert isinstance(prov["versions"][lib], str) and prov["versions"][lib], lib
    datetime.fromisoformat(prov["timestamp_utc"])
    assert prov["total_wall_time_sec"] > 0.0

    cfg = result["config"]
    assert cfg["mode"] == "quick"
    assert cfg["seeds"] == [0, 1]
    assert cfg["targets"] == list(QUICK_TARGETS)
    assert cfg["arms"] == list(ar.ARM_NAMES)
    assert cfg["arm_settings"]["stock_ta95"]["target_accept"] == 0.95
    for key in ("tune", "draws", "chains", "max_num_doublings", "tail_probs", "rhat_converged", "init"):
        assert key in cfg, key


def _iqr_stats(values: list[float]) -> dict[str, float]:
    q25, q50, q75 = np.percentile(np.asarray(values, dtype=np.float64), [25.0, 50.0, 75.0])
    return {"median": float(q50), "q25": float(q25), "q75": float(q75), "n": len(values)}


def _recompute_aggregates(data: dict[str, Any]) -> None:
    """Independently recomputes every aggregate from the per-seed rows and compares."""
    rows = data["rows"]
    assert data["aggregates"], "no aggregates"
    for agg in data["aggregates"]:
        t, a = agg["target"], agg["arm"]
        mine = [r for r in rows if r["target"] == t and r["arm"] == a]
        assert agg["n_seeds"] == len(mine)
        assert agg["n_errors"] == sum(r["error"] is not None for r in mine)
        conv = [r["max_rhat"] is not None and r["max_rhat"] <= 1.01 for r in mine]
        assert agg["frac_converged"] == pytest.approx(sum(conv) / len(mine))
        for m in ar.AGG_METRICS:
            vals = [r[m] for r in mine if r[m] is not None]
            expected = _iqr_stats(vals) if vals else None
            got = agg["metrics"][m]
            if expected is None:
                assert got is None
            else:
                assert got == pytest.approx(expected, rel=1e-12), (t, a, m)
        if a == "stock":
            assert "ratio_vs_stock" not in agg
            continue
        base = {r["seed"]: r for r in rows if r["target"] == t and r["arm"] == "stock"}
        for m in ar.RATIO_METRICS:
            ratios = [
                r[m] / base[r["seed"]][m]
                for r in mine
                if r["seed"] in base and r[m] is not None and base[r["seed"]][m] not in (None, 0.0)
            ]
            got = agg["ratio_vs_stock"][m]
            if not ratios:
                assert got is None
            else:
                assert got == pytest.approx(_iqr_stats(ratios), rel=1e-12), (t, a, m)


@pytest.mark.unit
def test_aggregates_are_recomputable_from_rows(quick_run: tuple[dict[str, Any], Path]) -> None:
    """Medians, IQRs, paired per-seed ratios vs stock and the converged fraction follow from the rows alone."""
    result, out = quick_run
    _recompute_aggregates(json.loads((out / "ablation_results.json").read_text()))
    # The module's own aggregation is a pure function of the rows.
    assert ar.aggregate_rows(result["rows"]) == result["aggregates"]


@pytest.mark.unit
def test_aggregation_on_handmade_rows() -> None:
    """Hand-checkable rows: ratios are paired by seed, missing values are skipped, errors count as unconverged."""

    def row(arm: str, seed: int, tepg: float | None, rhat: float | None, error: str | None = None) -> dict[str, Any]:
        r: dict[str, Any] = dict.fromkeys(ar.ROW_FIELDS)
        r.update(target="t", arm=arm, seed=seed, tail_ess_per_grad=tepg, max_rhat=rhat, error=error)
        return r

    rows = [
        row("stock", 0, 1.0, 1.00),
        row("stock", 1, 2.0, 1.20),
        row("stock", 2, 4.0, 1.00),
        row("arcsinh", 2, 40.0, 1.00),
        row("arcsinh", 0, 3.0, 1.005),
        row("arcsinh", 1, None, None, error="boom"),
    ]
    aggs = {a["arm"]: a for a in ar.aggregate_rows(rows)}
    assert aggs["stock"]["frac_converged"] == pytest.approx(2 / 3)
    assert aggs["arcsinh"]["frac_converged"] == pytest.approx(2 / 3)
    assert aggs["arcsinh"]["n_errors"] == 1
    assert aggs["arcsinh"]["metrics"]["tail_ess_per_grad"] == pytest.approx(
        {"median": 21.5, "q25": 12.25, "q75": 30.75, "n": 2}
    )
    # Paired ratios: seed 0 -> 3.0, seed 2 -> 10.0 (seed 1 has no arcsinh value).
    assert aggs["arcsinh"]["ratio_vs_stock"]["tail_ess_per_grad"] == pytest.approx(
        {"median": 6.5, "q25": 4.75, "q75": 8.25, "n": 2}
    )
    assert aggs["arcsinh"]["metrics"]["total_grad_evals"] is None


@pytest.mark.unit
def test_arms_share_paired_initial_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every arm of a seed starts from the same x-space positions; the scale mixture starts at their preimage."""
    calls: list[tuple[str, np.ndarray]] = []
    real = ar.sample_blackjax_unified_nuts

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs["transform_kind"], np.asarray(kwargs["x0"])))
        return real(*args, **kwargs)

    monkeypatch.setattr(ar, "sample_blackjax_unified_nuts", spy)
    config = ar.AblationConfig.quick(targets=("mvt_iso_nu2",), seeds=(3,), tune=30, draws=20)
    spec = make_ablation_target("mvt_iso_nu2")
    sm = spec.scale_mixture
    assert sm is not None
    x0 = ar.initial_positions(3, config.chains, spec.d)
    np.testing.assert_array_equal(x0, ar.initial_positions(3, config.chains, spec.d))
    assert not np.allclose(x0, ar.initial_positions(4, config.chains, spec.d))
    assert np.all(np.abs(x0) <= 2.0)

    for arm in ar.ARM_NAMES:
        row = ar.run_single(spec, arm, 3, config)
        assert row["error"] is None, row["error"]
    assert len(calls) == len(ar.ARM_NAMES)
    for (kind, start), arm in zip(calls, ar.ARM_NAMES):
        if arm == "scale_mixture":
            np.testing.assert_allclose(np.asarray(jax.vmap(sm.to_x)(jnp.asarray(start))), x0, atol=1e-12)
        else:
            np.testing.assert_array_equal(start, x0)


@pytest.mark.unit
def test_cli_arguments() -> None:
    """The CLI selects mode, seeds, targets, arms and output dir, and rejects bad values."""
    cfg = ar.config_from_args(
        ar.build_parser().parse_args(
            ["--quick", "--seeds", "3", "4", "--targets", "task1", "--arms", "stock", "arcsinh", "--output-dir", "x"]
        )
    )
    assert cfg.mode == "quick"
    assert cfg.seeds == (3, 4)
    assert cfg.targets == ("task1",)
    assert cfg.arms == ("stock", "arcsinh")
    assert cfg.output_dir == "x"
    assert ar.config_from_args(ar.build_parser().parse_args([])).mode == "full"
    assert ar.config_from_args(ar.build_parser().parse_args(["--full", "--workers", "2"])).workers == 2
    for bad in (["--quick", "--full"], ["--workers", "3"], ["--targets", "nope"], ["--arms", "nope"]):
        with pytest.raises(SystemExit):
            ar.config_from_args(ar.build_parser().parse_args(bad))


# =============================================================================
# The committed results (reports/ablation_*), the source of every quoted ablation number
# =============================================================================

REPORTS = Path(__file__).resolve().parents[2] / "reports"


@pytest.mark.unit
def test_committed_results_follow_the_full_protocol() -> None:
    """The committed run is the clean-tree, 10-seed --full grid with every applicable cell and no errors."""
    data = json.loads((REPORTS / "ablation_results.json").read_text())
    assert data["schema_version"] == ar.SCHEMA_VERSION
    prov = data["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_sha"])
    assert prov["git_dirty"] is False
    cfg = data["config"]
    full = ar.AblationConfig.full()
    assert cfg["mode"] == "full"
    assert cfg["seeds"] == list(full.seeds)
    assert (cfg["tune"], cfg["draws"], cfg["chains"], cfg["max_num_doublings"]) == (500, 1000, 4, 10)
    assert cfg["targets"] == list(ar.ABLATION_TARGET_NAMES)
    assert cfg["arms"] == list(ar.ARM_NAMES)
    expected = {
        (t, a, s)
        for t in ar.ABLATION_TARGET_NAMES
        for a in ar.ARM_NAMES
        for s in full.seeds
        if full.applies(make_ablation_target(t), a)
    }
    rows = data["rows"]
    assert {(r["target"], r["arm"], r["seed"]) for r in rows} == expected
    assert len(rows) == len(expected)
    assert all(r["error"] is None for r in rows)


@pytest.mark.unit
def test_committed_aggregates_are_recomputable_from_rows() -> None:
    """Every committed aggregate follows from the committed per-seed rows, and the CSVs mirror the JSON."""
    data = json.loads((REPORTS / "ablation_results.json").read_text())
    _recompute_aggregates(data)
    assert ar.aggregate_rows(data["rows"]) == data["aggregates"]

    with (REPORTS / "ablation_results.csv").open() as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(data["rows"])
    for c, r in zip(csv_rows, data["rows"]):
        assert (c["target"], c["arm"], int(c["seed"])) == (r["target"], r["arm"], r["seed"])
        for key in ("total_grad_evals", "tail_ess_per_grad", "max_rhat", "tail_prob_abs_z"):
            assert float(c[key]) == pytest.approx(r[key], rel=1e-12)

    with (REPORTS / "ablation_summary.csv").open() as fh:
        summary = list(csv.DictReader(fh))
    assert len(summary) == len(data["aggregates"])
    for s, agg in zip(summary, data["aggregates"]):
        assert (s["target"], s["arm"]) == (agg["target"], agg["arm"])
        assert float(s["frac_converged"]) == pytest.approx(agg["frac_converged"])
        assert float(s["tail_ess_per_grad_median"]) == pytest.approx(agg["metrics"]["tail_ess_per_grad"]["median"])
        if agg["arm"] != "stock":
            ratio = agg["ratio_vs_stock"]["tail_ess_per_grad"]
            assert float(s["ratio_tail_ess_per_grad_median"]) == pytest.approx(ratio["median"])
            assert float(s["ratio_tail_ess_per_grad_q25"]) == pytest.approx(ratio["q25"])
            assert float(s["ratio_tail_ess_per_grad_q75"]) == pytest.approx(ratio["q75"])
