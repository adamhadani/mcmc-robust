"""Unit tests for Unified 5-Benchmark Suite CLI Runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcmc_robust.bench.__main__ import build_arg_parser, main


@pytest.mark.unit
def test_cli_argument_parsing():
    """Verify CLI parser parses all supported arguments."""
    parser = build_arg_parser()

    # Default arguments
    args = parser.parse_args([])
    assert args.benchmark == "all"
    assert not args.quick
    assert not args.plot
    assert args.output_dir == "reports"
    assert args.seed == 42

    # Custom options
    args5 = parser.parse_args(
        [
            "--benchmark",
            "5",
            "--quick",
            "--plot",
            "--output-dir",
            "custom_reports",
            "--seed",
            "999",
        ]
    )
    assert args5.benchmark == "5"
    assert args5.quick
    assert args5.plot
    assert args5.output_dir == "custom_reports"
    assert args5.seed == 999


@pytest.mark.unit
def test_cli_execution_benchmark_5_quick(tmp_path: Path):
    """Verify CLI executes Benchmark 5 in quick smoke mode with plotting."""
    ret = main(
        [
            "--benchmark",
            "5",
            "--quick",
            "--plot",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "101",
        ]
    )
    assert ret == 0

    csv_path = tmp_path / "sota_comparison_results.csv"
    json_path = tmp_path / "sota_comparison_results.json"
    assert csv_path.exists()
    assert json_path.exists()

    fig_png = tmp_path / "figures" / "figure5_sota_nuts_comparison.png"
    fig_pdf = tmp_path / "figures" / "figure5_sota_nuts_comparison.pdf"
    assert fig_png.exists()
    assert fig_pdf.exists()
