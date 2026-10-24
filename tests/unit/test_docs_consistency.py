"""README.md and REPORT.md must agree with the committed results files.

Every result number in the docs lives in a block between ``<!-- BEGIN GENERATED: name -->`` and
``<!-- END GENERATED: name -->`` markers, rendered from ``reports/*.json`` by
``python -m mcmc_robust.bench.render_docs``. These tests fail when a block has drifted from the
data, when a ratio or percentage is typed by hand outside a block, or when a relative link is dead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mcmc_robust.bench import render_docs

REPO_ROOT = Path(__file__).resolve().parents[2]
FIX_HINT = "Regenerate with: uv run python -m mcmc_robust.bench.render_docs --write"


# --- the mechanism, on small synthetic documents ---------------------------------------------


def test_update_text_replaces_only_block_bodies() -> None:
    text = "intro 1.5x stays\n<!-- BEGIN GENERATED: demo -->\nold body\n<!-- END GENERATED: demo -->\noutro\n"
    updated = render_docs.update_text(text, {"demo": "new body\n"})
    assert (
        updated == "intro 1.5x stays\n<!-- BEGIN GENERATED: demo -->\nnew body\n<!-- END GENERATED: demo -->\noutro\n"
    )


def test_update_text_is_idempotent() -> None:
    text = "<!-- BEGIN GENERATED: demo -->\nold\n<!-- END GENERATED: demo -->\n"
    once = render_docs.update_text(text, {"demo": "new\n"})
    assert render_docs.update_text(once, {"demo": "new\n"}) == once


def test_update_text_rejects_unknown_block() -> None:
    text = "<!-- BEGIN GENERATED: nope -->\nx\n<!-- END GENERATED: nope -->\n"
    with pytest.raises(KeyError, match="nope"):
        render_docs.update_text(text, {"demo": "y\n"})


def test_unsourced_numbers_ignores_generated_blocks_and_code() -> None:
    text = (
        "Prose with a 12.3× gain and 7.49% mass.\n"
        "<!-- BEGIN GENERATED: demo -->\n2.5x and 40% inside a block\n<!-- END GENERATED: demo -->\n"
        "```python\nratio = '3x'  # 50%\n```\n"
        "Inline `4x` code is fine.\n"
    )
    assert render_docs.find_unsourced_numbers(text) == ["12.3×", "7.49%"]


def test_unsourced_numbers_catches_latex_percent_and_ascii_x() -> None:
    text = "Error fell to $`8.1\\%`$ with a 23.5x speed-up.\n"
    assert render_docs.find_unsourced_numbers(text) == ["8.1\\%", "23.5x"]


# --- the committed docs -----------------------------------------------------------------------


def test_generated_blocks_match_committed_reports() -> None:
    problems = render_docs.check(REPO_ROOT)
    assert problems == [], "\n".join(problems) + "\n" + FIX_HINT


def test_every_registered_block_appears_exactly_once_across_docs() -> None:
    counts = dict.fromkeys(render_docs.BLOCKS, 0)
    for name in render_docs.DOC_FILES:
        for block in render_docs.find_blocks((REPO_ROOT / name).read_text()):
            counts[block] += 1
    assert {k: v for k, v in counts.items() if v != 1} == {}


@pytest.mark.parametrize("doc", render_docs.DOC_FILES)
def test_docs_contain_no_result_numbers_outside_generated_blocks(doc: str) -> None:
    found = render_docs.find_unsourced_numbers((REPO_ROOT / doc).read_text())
    assert found == [], f"{doc}: hand-typed ratios/percentages {found}; move them into a generated block"


@pytest.mark.parametrize("doc", [*render_docs.DOC_FILES, "AGENTS.md"])
def test_relative_links_in_docs_resolve(doc: str) -> None:
    text = (REPO_ROOT / doc).read_text()
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    local = [t.split("#")[0] for t in targets if not re.match(r"[a-z]+:", t) and not t.startswith("#")]
    missing = sorted({t for t in local if t and not (REPO_ROOT / t).exists()})
    assert missing == [], f"{doc} links to missing files: {missing}"


def test_reference_list_lives_only_in_report() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    assert not re.search(r"^#+ .*References", readme, flags=re.MULTILINE), "keep the reference list in REPORT.md only"


# --- ablation renderers, on a small synthetic results file -----------------------------------


def _synthetic_ablation(tmp_path: Path) -> render_docs.Reports:
    """Two targets x three seeds. On 'heavy', arcsinh beats stock on seeds 0 and 1 only."""
    tpg = {
        ("heavy", "stock"): [1.0, 1.0, 1.0],
        ("heavy", "arcsinh"): [4.0, 2.0, 0.5],
        ("heavy", "affine"): [1.0, 1.0, 1.0],
        ("heavy", "scale_mixture"): [8.0, 8.0, 8.0],
        ("gauss_x", "stock"): [1.0, 1.0, 1.0],
        ("gauss_x", "arcsinh"): [0.8, 0.9, 0.7],
        ("gauss_x", "affine"): [1.0, 1.0, 1.0],
        ("heavy", "tukey_h"): [8.0, 4.0, 1.0],
        ("gauss_x", "tukey_h"): [1.0, 1.0, 1.0],
    }
    # The tail-alignment warning fired on these runs only.
    warned = {("heavy", "arcsinh"): [True, True, False], ("heavy", "stock"): [True, False, False]}
    divs = {("gauss_x", "arcsinh"): [0, 2, 0]}
    # Cells (coordinate, p) where the chain never drew beyond the exact quantile, per seed.
    unvisited = {("heavy", "arcsinh"): [0, 0, 3], ("heavy", "stock"): [0, 0, 0]}
    rows = [
        {
            "target": t,
            "arm": a,
            "seed": s,
            "tail_ess_per_grad": v,
            "max_rhat": 1.0 if a != "arcsinh" or s != 2 else 1.05,
            "converged": a != "arcsinh" or s != 2,
            "divergences": divs.get((t, a), [0, 0, 0])[s],
            "total_grad_evals": 1000,
            "tail_prob_abs_err": 0.002,
            "tail_prob_n_degenerate": unvisited.get((t, a), [0, 0, 0])[s],
            "tail_alignment_warning": warned.get((t, a), [False, False, False])[s],
        }
        for (t, a), vals in tpg.items()
        for s, v in enumerate(vals)
    ]
    (tmp_path / "reports").mkdir()
    payload = {"rows": rows, "config": {"seeds": [0, 1, 2], "chains": 4, "draws": 1000, "tune": 500}}
    (tmp_path / "reports" / "ablation_results.json").write_text(json.dumps(payload))
    return render_docs.Reports(tmp_path)


def test_ablation_findings_count_paired_seed_wins(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "heavy 2.0× (wins on 2 of 3 seeds)" in text
    assert "gauss_x 0.8× (wins on 0 of 3 seeds)" in text


def test_ablation_findings_compare_scale_mixture_with_arcsinh(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "heavy 4.0× (better than arcsinh on 3 of 3 seeds)" in text


def test_ablation_findings_report_divergent_seeds_and_convergence(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "gauss_x: 1 of 3 seeds" in text
    assert "heavy: 2 of 3 seeds" in text


def test_ablation_findings_flag_tails_that_are_never_visited(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "never visits a tail" in text
    assert "with `arcsinh`: heavy: 1 of 3 seeds" in text


# --- competitor renderers, on a small synthetic results file ---------------------------------


def _synthetic_competitors(tmp_path: Path) -> render_docs.Reports:
    """One target x three seeds: arcsinh beats nutpie_flow on seeds 0 and 2 only."""
    tpg = {
        "stock_nuts": [1.0, 1.0, 1.0],
        "arcsinh_nuts": [6.0, 3.0, 6.0],
        "nutpie_diag": [2.0, 2.0, 2.0],
        "nutpie_flow": [3.0, 4.0, 3.0],
    }
    rows = [
        {
            "target": "axis",
            "arm": arm,
            "seed": s,
            "status": "ok",
            "tail_ess_per_grad": v,
            "tail_ess_per_sec": 10.0 * v,
            "converged": True,
            "split_rhat": 1.0,
            "divergences": 0,
            "total_grad_evals": 1000,
            "tail_prob_abs_err_p01": 0.001,
            "tail_prob_abs_err_p99": 0.001,
            "tail_prob_mcse_p99": 0.002,
        }
        for arm, vals in tpg.items()
        for s, v in enumerate(vals)
    ]
    (tmp_path / "reports").mkdir(exist_ok=True)
    payload = {
        "rows": rows,
        "reference_arm": "stock_nuts",
        "arms": {a: {"description": a, "tune": 500} for a in tpg},
        "config": {"seeds": [0, 1, 2], "chains": 4, "draws": 1000, "tune": 500},
        "excluded_arms": {"dcs": {"status": "not run: no official code found"}},
    }
    (tmp_path / "reports" / "competitor_results.json").write_text(json.dumps(payload))
    return render_docs.Reports(tmp_path)


def test_competitor_findings_compare_arcsinh_with_nutpie_per_seed(tmp_path: Path) -> None:
    text = render_docs.render_competitor_findings(_synthetic_competitors(tmp_path))
    assert "axis 2.0× (better on 2 of 3 seeds)" in text  # arcsinh / nutpie_flow = 2, 0.75, 2
    assert "axis 3.0× (better on 3 of 3 seeds)" in text  # arcsinh / nutpie_diag = 3, 1.5, 3


def test_competitor_table_lists_excluded_arms(tmp_path: Path) -> None:
    text = render_docs.render_competitor_table(_synthetic_competitors(tmp_path))
    assert "not run: no official code found" in text
    assert "| axis | `arcsinh_nuts` | 6.00× [4.50, 6.00] |" in text


def test_ablation_findings_compare_tukey_h_with_arcsinh_and_stock(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "heavy 2.0× (better than arcsinh on 3 of 3 seeds)" in text  # tukey_h / arcsinh = 2, 2, 2
    assert "gauss_x 1.0× (wins on 0 of 3 seeds)" in text  # tukey_h / stock on the Gaussian


def test_ablation_findings_report_tail_alignment_warnings(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "**Tail-alignment warning**" in text
    assert "with `arcsinh`: heavy: 2 of 3 seeds" in text
    # One run never visits a tail (heavy, arcsinh, seed 2), and the warning did not fire on it.
    assert "fired on 0 of 1" in text
