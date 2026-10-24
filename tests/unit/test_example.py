"""example.py must report what it measured: every sentence it prints is derived from its numbers."""

from __future__ import annotations

import example

STOCK = {
    "tail_ess_per_grad": 0.02,
    "total_grad_evals": 100_000,
    "split_rhat": 1.004,
    "divergences": 3,
    "q99_rel_error": 0.12,
}


def _robust(**overrides: float) -> dict[str, float]:
    return {**STOCK, **overrides}


def test_describe_comparison_reports_a_gain_as_higher() -> None:
    text = " ".join(example.describe_comparison(STOCK, _robust(tail_ess_per_grad=0.05)))
    assert "2.5× higher" in text


def test_describe_comparison_reports_a_loss_as_lower() -> None:
    text = " ".join(example.describe_comparison(STOCK, _robust(tail_ess_per_grad=0.01)))
    assert "0.5× (lower)" in text
    assert "higher" not in text


def test_describe_comparison_flags_an_unconverged_run() -> None:
    text = " ".join(example.describe_comparison(STOCK, _robust(split_rhat=1.08)))
    assert "transformed run has not converged" in text


def test_describe_comparison_states_divergences_without_spin() -> None:
    text = " ".join(example.describe_comparison(STOCK, _robust(divergences=13)))
    assert "13 vs 3" in text
    assert "eliminat" not in text.lower()


def test_example_runs_end_to_end(capsys) -> None:  # type: ignore[no-untyped-def]
    example.main(draws=60, tune=60, chains=2, seed=0)
    out = capsys.readouterr().out
    assert "Tail-ESS per gradient" in out
    assert "Max R-hat" in out
