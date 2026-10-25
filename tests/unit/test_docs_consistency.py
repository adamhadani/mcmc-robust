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
        ("heavy", "tukey_h0"): [2.0, 2.0, 2.0],
        ("gauss_x", "tukey_h0"): [1.0, 1.0, 1.0],
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
    assert "with `tukey_h`: none" in text  # every arm that ran is listed


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


def test_ablation_findings_compare_tukey_h_with_its_h0_control(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "**The guard's h > 0 decisions**" in text
    assert "heavy 2.0× (better than h = 0 on 2 of 3 seeds)" in text  # tukey_h / tukey_h0 = 4, 2, 0.5
    assert "gauss_x 1.0× (better than h = 0 on 0 of 3 seeds)" in text


def test_ablation_findings_report_tail_alignment_warnings(tmp_path: Path) -> None:
    text = render_docs.render_ablation_findings(_synthetic_ablation(tmp_path))
    assert "**Tail-alignment warning**" in text
    assert "with `arcsinh`: heavy: 2 of 3 seeds" in text
    # One run never visits a tail (heavy, arcsinh, seed 2), and the warning did not fire on it.
    assert "fired on 0 of 1" in text


# --- posteriordb renderers, on a small synthetic results file --------------------------------


def _synthetic_posteriordb(tmp_path: Path) -> render_docs.Reports:
    """Two posteriors x three seeds x four arms.

    On 'a-heavy' tukey_h beats arcsinh on seeds 0 and 1 (min-tail-ESS/grad ratios 2, 2, 0.5) and arcsinh
    beats stock on every seed (ratio 2); against its h = 0 control the ratios are 2, 0.5, 1. On 'b-light' both
    transforms lose to stock (ratio 0.5) and tukey_h ties arcsinh and tukey_h0. tukey_h disagrees with the
    reference on two seeds of 'b-light' only. The guard on exact draws gives h > 0 on 1 coordinate of 'a-heavy'.
    """
    mtpg = {
        ("a-heavy", "stock"): [1.0, 1.0, 1.0],
        ("a-heavy", "arcsinh"): [2.0, 2.0, 2.0],
        ("a-heavy", "tukey_h"): [4.0, 4.0, 1.0],
        ("b-light", "stock"): [2.0, 2.0, 2.0],
        ("b-light", "arcsinh"): [1.0, 1.0, 1.0],
        ("b-light", "tukey_h"): [1.0, 1.0, 1.0],
        ("a-heavy", "tukey_h0"): [2.0, 8.0, 1.0],
        ("b-light", "tukey_h0"): [1.0, 1.0, 1.0],
    }
    max_z = {("b-light", "tukey_h"): [9.0, 7.0, 1.0]}
    divs = {("a-heavy", "tukey_h"): [3, 0, 5]}
    h_pos = {("a-heavy", "tukey_h"): [2, 2, 1], ("b-light", "tukey_h"): [0, 1, 0]}
    h_ref = {("a-heavy", "tukey_h"): [1, 0, 1], ("b-light", "tukey_h"): [0, 0, 0]}
    warned = {("a-heavy", "arcsinh"): [True, True, False]}
    # Phase 1's location offset from the final draws, in posterior scales; far (> 3) on b-light for arcsinh only.
    offsets = {("b-light", "arcsinh"): [10.0, 12.0, 0.5]}
    rows = [
        {
            "posterior": p,
            "arm": a,
            "seed": s,
            "d": 4 if p == "a-heavy" else 3,
            "k": 4,
            "min_tail_ess_per_grad": v,
            "tail_ess_per_grad": v,
            "max_rhat": 1.0 if (p, a, s) != ("a-heavy", "stock", 1) else 1.2,
            "converged": (p, a, s) != ("a-heavy", "stock", 1),
            "divergences": divs.get((p, a), [0, 0, 0])[s],
            "total_grad_evals": 1000,
            "ref_tail_prob_max_abs_z": max_z.get((p, a), [1.5, 1.5, 1.5])[s],
            "ref_mean_max_abs_z": 1.0,
            "tukey_h_n_positive": h_pos.get((p, a), [0, 0, 0])[s] if a.startswith("tukey_h") else None,
            "tukey_h_ref_positive": h_ref.get((p, a), [0, 0, 0])[s] if a.startswith("tukey_h") else None,
            "tail_alignment_warning": warned.get((p, a), [False, False, False])[s],
            "calib_location_offset_max": None if a == "stock" else offsets.get((p, a), [0.2, 0.3, 0.1])[s],
            "error": None,
        }
        for (p, a), vals in mtpg.items()
        for s, v in enumerate(vals)
    ]
    (tmp_path / "reports").mkdir(exist_ok=True)
    payload = {
        "rows": rows,
        "config": {
            "seeds": [0, 1, 2],
            "chains": 4,
            "draws": 1000,
            "tune": 500,
            "arms": ["stock", "arcsinh", "tukey_h", "tukey_h0"],
        },
        "provenance": {"posteriordb_sha": "91027f9cdd13b7e76283fec3f2898f59d4b83ec4", "git_sha": "a" * 40},
    }
    (tmp_path / "reports" / "posteriordb_results.json").write_text(json.dumps(payload))
    return render_docs.Reports(tmp_path)


def test_posteriordb_findings_count_posteriors_where_each_arm_wins(tmp_path: Path) -> None:
    text = render_docs.render_posteriordb_findings(_synthetic_posteriordb(tmp_path))
    # Per-posterior medians of the paired ratios: arcsinh/stock 2 and 0.5; tukey_h/arcsinh 2 and 1.
    assert "**`arcsinh` vs stock NUTS**" in text
    assert "better on 1 of 2 posteriors" in text
    assert "**`tukey_h` vs `arcsinh`**" in text
    assert "`tukey_h` better on 1 of 2 posteriors, worse by more than a fifth on 0" in text


def test_posteriordb_findings_report_disagreement_with_the_reference(tmp_path: Path) -> None:
    text = render_docs.render_posteriordb_findings(_synthetic_posteriordb(tmp_path))
    assert "disagree with the reference" in text
    assert "`tukey_h`: 1 of 2 (b-light)" in text
    assert "`arcsinh`: 0 of 2" in text


def test_posteriordb_findings_report_the_guard_and_the_warning(tmp_path: Path) -> None:
    text = render_docs.render_posteriordb_findings(_synthetic_posteriordb(tmp_path))
    # Median over seeds of the coordinates with h > 0: 2 of 4 on a-heavy, 0 of 3 on b-light.
    assert "h > 0 on 2 of 7 unconstrained coordinates (1 of 2 posteriors)" in text
    # On exact draws (median over seeds): 1 coordinate of a-heavy, none of b-light.
    assert (
        "from Phase 1's draws it set h > 0 on 2 of 7 coordinates, and from as many exact posterior draws on 1" in text
    )
    assert "`TailAlignmentWarning`" in text and "1 of 2 posteriors" in text


def test_posteriordb_table_has_one_row_per_posterior(tmp_path: Path) -> None:
    text = render_docs.render_posteriordb_table(_synthetic_posteriordb(tmp_path))
    assert "91027f9cdd13" in text
    # Columns: d, stock's median, then arcsinh / stock, tukey_h / stock, tukey_h / arcsinh, tukey_h / tukey_h0
    # (median, seeds better), ..., then the guard's h > 0 count from Phase 1 and from exact draws (medians).
    assert "| a-heavy | 4 | 1 | 2.00× (3 of 3) | 4.00× (2 of 3) | 2.00× (2 of 3) | 1.00× (1 of 3) |" in text
    assert text.count("| 2 | 1 |") == 1
    assert "| b-light | 3 |" in text


def test_posteriordb_findings_relate_losses_to_phase1_calibrating_off_the_posterior(tmp_path: Path) -> None:
    text = render_docs.render_posteriordb_findings(_synthetic_posteriordb(tmp_path))
    assert "**Phase 1 calibrated off the posterior**" in text
    # b-light: arcsinh's median offset is 10 scales, and arcsinh / stock is 0.5 there.
    assert "`arcsinh`: 1 of 2 posteriors (b-light), worse than stock by more than a fifth on 1 of them" in text
    assert "`tukey_h`: 0 of 2 posteriors" in text


def test_posteriordb_findings_split_tukey_h_by_the_guards_decision(tmp_path: Path) -> None:
    text = render_docs.render_posteriordb_findings(_synthetic_posteriordb(tmp_path))
    # b-light: median h > 0 count 0 (affine everywhere); a-heavy: median 2 (some h > 0), tukey_h / tukey_h0 median 1.
    assert (
        "where it set h > 0 somewhere (1 of 2 posteriors), `tukey_h` vs `tukey_h0` 1.00× (better on 0, worse by more "
        "than a fifth on 0); where it kept h = 0 on every coordinate (1 of 2), 1.00×." in text
    )
    assert "**`tukey_h0` (h = 0 everywhere) vs stock NUTS**" in text
