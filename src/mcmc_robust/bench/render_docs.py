r"""Render the generated blocks of README.md and REPORT.md from the committed results files.

Every result number in the two documents sits between a pair of markers::

    <!-- BEGIN GENERATED: name -->
    ...rendered from reports/*.json...
    <!-- END GENERATED: name -->

and is produced by the renderer registered under ``name`` in :data:`BLOCKS`. Prose outside the
blocks may not contain hand-typed ratios or percentages (:func:`find_unsourced_numbers`), so a
sampler change that moves a reported number fails ``tests/unit/test_docs_consistency.py`` until
the blocks are regenerated.

Usage::

    uv run python -m mcmc_robust.bench.render_docs --write   # rewrite the blocks in place
    uv run python -m mcmc_robust.bench.render_docs --check   # exit 1 if any block is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

DOC_FILES: tuple[str, ...] = ("README.md", "REPORT.md")

_BLOCK_RE = re.compile(
    r"(?P<begin><!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\n)(?P<body>.*?)(?P<end><!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)
_FENCE_RE = re.compile(r"^```.*?^```[ \t]*$", re.DOTALL | re.MULTILINE)
# Inline code spans, except GitHub's $`...`$ math form, whose content is checked.
_CODE_SPAN_RE = re.compile(r"(?<!\$)`[^`\n]+`(?!\$)")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:×|x(?!\w)|\\%|%|\s?\\times)")

# Design constants (not results) that prose may state directly.
ALLOWED_LITERALS: frozenset[str] = frozenset({"15%", "65%", "20%"})

# Samplers that build no Hamiltonian trajectories, so a divergence count carries no information.
_NO_TRAJECTORY = frozenset(
    {
        "MALA",
        "Barker",
        "Tamed MALA",
        "Stereographic MALA (own implementation)",
        "AutoMALA (own re-implementation)",
        "GentleMALA",
        "AnisotropicTailMALA",
        "UnifiedRobustSampler",
    }
)

_DISPLAY_NAMES: dict[str, str] = {
    "Classical MALA": "MALA",
    "ClassicalMALA": "MALA",
    "BarkerLangevin (2022)": "Barker",
    "GloballyTamedMALA (2025)": "Tamed MALA",
    "Stereographic MALA (2024)": "Stereographic MALA (own implementation)",
    "AutoMALA (2024)": "AutoMALA (own re-implementation)",
    "AutoMALA": "AutoMALA (own re-implementation)",
    "BlackJAX NUTS (Neal)": "BlackJAX NUTS",
    "BlackJAX HMC (Standard)": "BlackJAX HMC",
    "BlackJAXNUTS (Stock)": "BlackJAX NUTS",
    "BlackJAXNUTS + JaxArcsinhTransform": "BlackJAX NUTS + fixed arcsinh (R = 2.5)",
    "UnifiedRobustSampler (Neal)": "UnifiedRobustSampler",
    "UnifiedRobustNUTS (Neal)": "UnifiedRobustNUTS",
}

_APPLIED_TARGETS: dict[str, str] = {
    "radon_hierarchical": "Radon-style hierarchical",
    "sparse_horseshoe_logistic": "Horseshoe logistic",
    "robust_student_t_regression": "Student-t regression",
    "hierarchical_irt_2pl": "IRT-2PL",
    "stochastic_volatility": "Stochastic volatility",
}

_SWEEP_SAMPLERS: dict[str, str] = {
    "mala": "MALA",
    "tamed_mala": "Tamed MALA",
    "barker": "Barker",
    "gentle_mala": "GentleMALA",
    "unified_robust": "UnifiedRobustSampler",
}

RHAT_CONVERGED = 1.01

_ABLATION_TARGETS: dict[str, str] = {
    "task1": "Task 1",
    "task3": "Task 3",
    "mvt_rot_nu1": "rotated t (ν = 1)",
    "mvt_rot_nu2": "rotated t (ν = 2)",
    "mvt_iso_nu2": "isotropic t (ν = 2)",
    "rot_prod_cauchy": "rotated product of Cauchys",
    "gauss_rot": "rotated Gaussian",
    "gauss_aniso": "anisotropic Gaussian",
}


# --- block mechanics ---------------------------------------------------------------------------


def find_blocks(text: str) -> list[str]:
    """Names of the generated blocks in ``text``, in order of appearance."""
    return [m.group("name") for m in _BLOCK_RE.finditer(text)]


def update_text(text: str, rendered: Mapping[str, str]) -> str:
    """Replace the body of every generated block in ``text`` with ``rendered[name]``."""

    def _sub(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in rendered:
            raise KeyError(f"no renderer for generated block {name!r}")
        return m.group("begin") + rendered[name] + m.group("end")

    return _BLOCK_RE.sub(_sub, text)


def find_unsourced_numbers(text: str) -> list[str]:
    """Ratios and percentages written outside generated blocks, code and design constants."""
    text = _BLOCK_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = _CODE_SPAN_RE.sub("", text)
    return [m.group(0) for m in _NUMBER_RE.finditer(text) if m.group(0) not in ALLOWED_LITERALS]


def render_all(root: Path, names: Sequence[str]) -> dict[str, str]:
    """Render the named blocks from the results files under ``root``."""
    reports = Reports(root)
    return {name: BLOCKS[name](reports) for name in names if name in BLOCKS}


def check(root: Path) -> list[str]:
    """Describe every generated block in the docs that differs from a fresh render."""
    problems: list[str] = []
    for doc in DOC_FILES:
        text = (root / doc).read_text()
        names = find_blocks(text)
        problems += [f"{doc}: unknown generated block {n!r}" for n in names if n not in BLOCKS]
        rendered = render_all(root, names)
        bodies = {m.group("name"): m.group("body") for m in _BLOCK_RE.finditer(text)}
        problems += [f"{doc}: block {n!r} is stale" for n, body in rendered.items() if bodies[n] != body]
    return problems


def write(root: Path) -> list[str]:
    """Regenerate every block in place; return the names of the docs that changed."""
    changed = []
    for doc in DOC_FILES:
        path = root / doc
        text = path.read_text()
        updated = update_text(text, render_all(root, find_blocks(text)))
        if updated != text:
            path.write_text(updated)
            changed.append(doc)
    return changed


# --- data access and formatting ----------------------------------------------------------------


class Reports:
    """Lazily loaded results files under ``root/reports`` plus ``root/formal``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _load(self, name: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.root / "reports" / name).read_text())
        return data

    @cached_property
    def sota(self) -> dict[str, Any]:
        return self._load("sota_comparison_results.json")

    @cached_property
    def applied(self) -> dict[str, Any]:
        return self._load("applied_benchmark_results.json")

    @cached_property
    def benchmark(self) -> dict[str, Any]:
        return self._load("benchmark_results.json")

    @cached_property
    def sweep(self) -> dict[str, Any]:
        return self._load("sweep_results.json")

    @cached_property
    def ablation(self) -> dict[str, Any]:
        return self._load("ablation_results.json")

    @cached_property
    def competitors(self) -> dict[str, Any]:
        return self._load("competitor_results.json")

    def sota_rows(self, task_prefix: str, target_name: str | None = None) -> list[dict[str, Any]]:
        return [
            r
            for r in self.sota["results"]
            if r["task_id"].startswith(task_prefix) and (target_name is None or r["target_name"] == target_name)
        ]

    def sota_row(self, task_prefix: str, sampler: str, target_name: str | None = None) -> dict[str, Any]:
        rows = [r for r in self.sota_rows(task_prefix, target_name) if display(r["sampler_name"]) == sampler]
        if len(rows) != 1:
            raise LookupError(f"expected one {sampler!r} row in {task_prefix}/{target_name}, found {len(rows)}")
        return rows[0]


def display(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


def _int(x: float) -> str:
    return f"{round(x):,}"


def _ess(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def _rate(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3g}"


def _rhat(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _converged(x: float | None) -> str:
    return "yes" if x is not None and x <= RHAT_CONVERGED else "no"


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}%"


def _num(x: float, digits: int = 2) -> str:
    """Fixed-point number with a typographic minus sign."""
    return f"{x:.{digits}f}".replace("-", "−")


def _ratio(x: float) -> str:
    return f"{x:.1f}×"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], numeric_from: int = 1) -> str:
    align = [":---" if i < numeric_from else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _sota_setup(reports: Reports) -> str:
    cfg = reports.sota["summary"]["config"]
    date = reports.sota["summary"]["timestamp"][:10]
    return (
        f"Seed {cfg['seed']} only, {cfg['n_chains']} chains × {cfg['n_samples']:,} draws after "
        f"{cfg['n_warmup']} warm-up steps per chain; gradient counts include warm-up. "
        f"Source: `reports/sota_comparison_results.json` (run {date}). "
        f"A run is converged when its largest rank-normalised split R-hat is at most {RHAT_CONVERGED}."
    )


def _divs(row: dict[str, Any]) -> str:
    return "–" if display(row["sampler_name"]) in _NO_TRAJECTORY else _int(row["divergences"])


def _sampler_cell(row: dict[str, Any]) -> str:
    name = display(row["sampler_name"])
    return f"`{name}`" if row["framework"] == "mcmc_robust" else name


# --- renderers ---------------------------------------------------------------------------------


def _render_sota_tails(reports: Reports, task_prefix: str) -> str:
    rows = reports.sota_rows(task_prefix)
    base = reports.sota_row(task_prefix, "BlackJAX NUTS")["tail_ess_per_grad"]
    body = [
        [
            _sampler_cell(r),
            _int(r["total_grad_evals"]),
            _ess(r["bulk_ess"]),
            _ess(r["tail_ess"]),
            _rhat(r["split_rhat"]),
            _converged(r["split_rhat"]),
            _rate(r["tail_ess_per_grad"]),
            _ratio(r["tail_ess_per_grad"] / base),
            _divs(r),
            _pct(r.get("q99_rel_error")),
        ]
        for r in rows
    ]
    header = [
        "Sampler",
        "Gradients",
        "Bulk-ESS",
        "Tail-ESS",
        "Max R-hat",
        "Converged",
        "Tail-ESS/grad",
        "vs BlackJAX NUTS",
        "Divergences",
        "q99 error",
    ]
    return _sota_setup(reports) + "\n\n" + _table(header, body)


def _eight_schools_truth() -> tuple[float, float]:
    from mcmc_robust.bench.sota_adapters import EightSchoolsTarget

    exact = EightSchoolsTarget().exact_moments_or_quantiles()
    return float(exact["prob_log_tau_lt_neg1"]), float(exact["q05_log_tau"])


def _neal_truth() -> tuple[float, float, float]:
    from mcmc_robust.targets.roughness import NealsFunnelTarget

    sigma_v = NealsFunnelTarget(d=10).sigma_v
    return sigma_v, float(stats.norm.cdf(-2.0 / sigma_v)), float(sigma_v * stats.norm.ppf(0.05))


def _render_funnel(reports: Reports, target_name: str, prob_key: str, q05_key: str, truth: str) -> str:
    rows = reports.sota_rows("task2", target_name)
    body = [
        [
            _sampler_cell(r),
            _int(r["total_grad_evals"]),
            _ess(r["bulk_ess"]),
            _ess(r["tail_ess"]),
            _rhat(r["split_rhat"]),
            _converged(r["split_rhat"]),
            _divs(r),
            _pct(r["extra_stats"][prob_key]),
            _num(r["extra_stats"][q05_key]),
        ]
        for r in rows
    ]
    label = "log τ" if "tau" in prob_key else "v"
    threshold = "−1" if "tau" in prob_key else "−2"
    header = [
        "Sampler",
        "Gradients",
        "Bulk-ESS",
        "Tail-ESS",
        "Max R-hat",
        "Converged",
        "Divergences",
        f"P({label} < {threshold})",
        f"q05({label})",
    ]
    return _sota_setup(reports) + " " + truth + "\n\n" + _table(header, body)


def render_sota_task1(reports: Reports) -> str:
    return _render_sota_tails(reports, "task1")


def render_sota_task3(reports: Reports) -> str:
    return _render_sota_tails(reports, "task3")


def render_sota_eight_schools(reports: Reports) -> str:
    p, q05 = _eight_schools_truth()
    truth = f"Exact reference: P(log τ < −1) = {_pct(p, 2)}, q05(log τ) = {_num(q05)}."
    return _render_funnel(reports, "eight_schools_centered", "prob_log_tau_lt_neg1", "q05_log_tau", truth)


def render_sota_neal(reports: Reports) -> str:
    sigma_v, p, q05 = _neal_truth()
    truth = f"Exact reference (v ~ N(0, {sigma_v:g}²)): P(v < −2) = {_pct(p, 2)}, q05(v) = {_num(q05)}."
    return _render_funnel(reports, "neals_funnel_d10", "prob_v_lt_neg2", "q05_v", truth)


def render_sota_task4(reports: Reports) -> str:
    rows = reports.sota_rows("task4")
    body = [
        [
            _sampler_cell(r),
            _int(r["total_grad_evals"]),
            _ess(r["bulk_ess"]),
            _ess(r["tail_ess"]),
            _rhat(r["split_rhat"]),
            _converged(r["split_rhat"]),
            _rate(r["tail_ess_per_grad"]),
            _divs(r),
            _int(r["extra_stats"]["burnin_grad_evals"]),
        ]
        for r in rows
    ]
    header = [
        "Sampler",
        "Gradients",
        "Bulk-ESS",
        "Tail-ESS",
        "Max R-hat",
        "Converged",
        "Tail-ESS/grad",
        "Divergences",
        "Gradients to reach the bulk",
    ]
    note = (
        ' Chains start at ‖x₀‖ = 10√d; "Gradients to reach the bulk" is the median over chains of the '
        "gradient evaluations until ‖x‖ ≤ 3.5, or the chain's whole budget when it never gets there."
    )
    return _sota_setup(reports) + note + "\n\n" + _table(header, body)


def render_applied(reports: Reports) -> str:
    meta = reports.applied["metadata"]
    rows = reports.applied["results"]
    body = [
        [
            f"{_APPLIED_TARGETS.get(r['target_name'], r['target_name'])} (d={r['dimension']})",
            display(r["sampler_name"]),
            _rhat(r["split_rhat"]),
            "yes" if r["converged"] else "no",
            _ess(r["min_bulk_ess"]),
            _ess(r["min_tail_ess"]),
            f"{1000.0 * r['tail_ess_per_grad']:.2f}",
            "–" if display(r["sampler_name"]) in _NO_TRAJECTORY else _int(r["divergences"]),
            _int(r["grad_evals"]),
            f"{r['wall_time_sec']:.1f}",
        ]
        for r in rows
    ]
    header = [
        "Model",
        "Sampler",
        "Max R-hat",
        "Converged",
        "Min Bulk-ESS",
        "Min Tail-ESS",
        "Tail-ESS/grad (×10⁻³)",
        "Divergences",
        "Gradients",
        "Wall time (s)",
    ]
    nuts = {"BlackJAX NUTS", "BlackJAX NUTS + fixed arcsinh (R = 2.5)", "UnifiedRobustNUTS"}
    n_conv = sum(bool(r["converged"]) for r in rows)
    all_nuts_converged = [
        t
        for t in dict.fromkeys(r["target_name"] for r in rows)
        if all(r["converged"] for r in rows if r["target_name"] == t and display(r["sampler_name"]) in nuts)
    ]
    first_order_converged = sum(bool(r["converged"]) for r in rows if display(r["sampler_name"]) not in nuts)
    summary = [
        f"- {n_conv} of {len(rows)} runs converge (largest split R-hat at most {meta['rhat_converged_threshold']}).",
        (
            f"- First-order samplers converge in {first_order_converged} of "
            f"{sum(display(r['sampler_name']) not in nuts for r in rows)} runs."
        ),
    ]
    for t in all_nuts_converged:
        stock = next(r for r in rows if r["target_name"] == t and display(r["sampler_name"]) == "BlackJAX NUTS")
        ours = next(r for r in rows if r["target_name"] == t and display(r["sampler_name"]) == "UnifiedRobustNUTS")
        summary.append(
            f"- {_APPLIED_TARGETS.get(t, t)} is a model on which every NUTS variant converges: `UnifiedRobustNUTS` "
            f"reaches {_ratio(ours['tail_ess_per_grad'] / stock['tail_ess_per_grad'])} the tail-ESS per gradient "
            f"of stock BlackJAX NUTS."
        )
    setup = (
        f"Seed {meta['seed']} only, {meta['n_chains']} chains × {meta['n_samples']:,} draws after "
        f"{meta['n_warmup']:,} warm-up steps, NUTS tree depth at most {meta['nuts_max_depth']}. "
        "Min ESS is over coordinates; Tail-ESS/grad is the coordinate mean divided by all gradients, "
        "warm-up included. Wall time includes JIT compilation for the BlackJAX rows, while "
        "`UnifiedRobustNUTS` runs in NumPy. Source: `reports/applied_benchmark_results.json`."
    )
    return setup + "\n\n" + _table(header, body, numeric_from=2) + "\n" + "\n".join(summary) + "\n"


def render_langevin_sweep(reports: Reports) -> str:
    meta = reports.sweep["metadata"]
    quartic = reports.sweep["summary"]["quartic"]
    body = []
    for key in sorted(quartic, key=lambda k: int(k.split("_")[1])):
        windows = quartic[key]["sampler_windows"]
        for s_key, s_name in _SWEEP_SAMPLERS.items():
            w = windows[s_key]
            burn = w["median_burnin_steps"]
            body.append(
                [
                    key.split("_")[1],
                    s_name,
                    f"{w['h_max']:.3g}",
                    f"{w['window_ratio']:,.0f}",
                    "not reached" if burn is None else f"{burn:g}",
                ]
            )
    header = ["d", "Sampler", "Largest stable h", "h_max / h_min", "Median burn-in steps"]
    setup = (
        f"Quartic target U(x) = ‖x‖⁴/4 started at ‖x₀‖ = 10√d; {meta['n_grid']} step sizes log-spaced on "
        f"[{meta['h_min']:g}, {meta['h_max']:g}]. A step size is stable when the run neither diverges nor "
        "accepts fewer than 1 in 100 proposals. Source: `reports/sweep_results.json`."
    )
    return setup + "\n\n" + _table(header, body, numeric_from=2)


def render_langevin_benign(reports: Reports) -> str:
    rows = [r for r in reports.benchmark["results"] if r["regime"] == "benign" and r["dimension"] == 50]
    samplers = ["mala", "barker", "tamed_mala", "gentle_mala", "unified_robust"]
    body = []
    for target in ("gaussian_isotropic", "gaussian_anisotropic"):
        for init in ("bulk", "tail"):
            cell = {r["sampler_name"]: r for r in rows if r["target_name"] == target and r["init_type"] == init}
            mala = cell["mala"]["ess_per_grad"]
            body += [
                [
                    target.removeprefix("gaussian_"),
                    init,
                    _SWEEP_SAMPLERS[s],
                    _rate(cell[s]["ess_per_grad"]),
                    _rate(cell[s]["tail_ess_per_grad"]),
                    _rhat(cell[s]["split_rhat"]),
                    _converged(cell[s]["split_rhat"]),
                    _pct(cell[s]["ess_per_grad"] / mala),
                ]
                for s in samplers
                if s in cell
            ]
    header = ["Gaussian", "Start", "Sampler", "ESS/grad", "Tail-ESS/grad", "R-hat", "Converged", "ESS/grad vs MALA"]
    meta = reports.benchmark["metadata"]
    setup = (
        f"d = 50, budget {meta['budget']:,} gradients per chain, {meta['n_chains']} chains, seed {meta['seed']}; "
        '"tail" starts at ‖x₀‖ = 10√d. Source: `reports/benchmark_results.json`.'
    )
    return setup + "\n\n" + _table(header, body, numeric_from=3)


def render_lean_inventory(reports: Reports) -> str:
    text = (reports.root / "formal" / "McmcRobustFormal.lean").read_text()
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        header = re.match(r"\s+(?:PROPERTY \d+:|§\d+\.)\s*(.+?)\s*$", line)
        if header and line.startswith("    "):
            sections.append((header.group(1), []))
            continue
        theorem = re.match(r"theorem (\w+)", line)
        if theorem and sections:
            sections[-1][1].append(theorem.group(1))
    body = [[title, ", ".join(f"`{t}`" for t in names)] for title, names in sections if names]
    total = sum(len(names) for _, names in sections)
    return (
        f"{total} theorems in `formal/McmcRobustFormal.lean`, listed under the file's own section headings:\n\n"
        + _table(["Section (as titled in the Lean file)", "Theorems"], body, numeric_from=2)
    )


def render_readme_headline(reports: Reports) -> str:
    t1 = {s: reports.sota_row("task1", s) for s in ("UnifiedRobustNUTS", "BlackJAX NUTS", "PyMC NUTS")}
    t3 = {s: reports.sota_row("task3", s) for s in ("UnifiedRobustNUTS", "BlackJAX NUTS")}
    es = {
        s: reports.sota_row("task2", s, "eight_schools_centered")
        for s in ("UnifiedRobustNUTS", "BlackJAX NUTS", "PyMC NUTS")
    }
    neal = {s: reports.sota_row("task2", s, "neals_funnel_d10") for s in ("UnifiedRobustNUTS", "BlackJAX NUTS")}
    es_p, _ = _eight_schools_truth()
    _, neal_p, _ = _neal_truth()
    cfg = reports.sota["summary"]["config"]
    applied = reports.applied["results"]

    def tg(row: dict[str, Any]) -> float:
        return float(row["tail_ess_per_grad"])

    ours1, ours3 = t1["UnifiedRobustNUTS"], t3["UnifiedRobustNUTS"]
    bullets = [
        (
            f"- **Heavy tails, product form (REPORT Table 1).** On a d=10 regression in which 7 of the 10 coefficients "
            f"sit at their independent Cauchy prior, `UnifiedRobustNUTS` has {_ratio(tg(ours1) / tg(t1['BlackJAX NUTS']))} "
            f"the tail-ESS per gradient of BlackJAX NUTS and {_ratio(tg(ours1) / tg(t1['PyMC NUTS']))} that of PyMC NUTS "
            f"(max R-hat {_rhat(ours1['split_rhat'])} vs {_rhat(t1['BlackJAX NUTS']['split_rhat'])} and "
            f"{_rhat(t1['PyMC NUTS']['split_rhat'])}); its 99% quantile error is {_pct(ours1['q99_rel_error'])} against "
            f"{_pct(t1['BlackJAX NUTS']['q99_rel_error'])} and {_pct(t1['PyMC NUTS']['q99_rel_error'])}."
        ),
        (
            f"- **Mixed tails (REPORT Table 3, d=20).** {_ratio(tg(ours3) / tg(t3['BlackJAX NUTS']))} the tail-ESS per "
            f"gradient of BlackJAX NUTS, using {_int(ours3['total_grad_evals'])} gradients against "
            f"{_int(t3['BlackJAX NUTS']['total_grad_evals'])}; 99% quantile error {_pct(ours3['q99_rel_error'])} against "
            f"{_pct(t3['BlackJAX NUTS']['q99_rel_error'])}."
        ),
        (
            f"- **Funnels are not solved (REPORT Table 2).** On Eight-Schools `UnifiedRobustNUTS` has "
            f"{_int(es['UnifiedRobustNUTS']['divergences'])} divergences against {_int(es['BlackJAX NUTS']['divergences'])} "
            f"(BlackJAX) and {_int(es['PyMC NUTS']['divergences'])} (PyMC), but only "
            f"{_pct(es['UnifiedRobustNUTS']['extra_stats']['prob_log_tau_lt_neg1'])} of its draws reach log τ < −1 "
            f"(exact: {_pct(es_p, 2)}), so it has fewer divergences because it stays out of the neck. On Neal's funnel it "
            f"reaches {_pct(neal['UnifiedRobustNUTS']['extra_stats']['prob_v_lt_neg2'])} neck mass (exact: {_pct(neal_p, 2)}) "
            f"at max R-hat {_rhat(neal['UnifiedRobustNUTS']['split_rhat'])}, with "
            f"{_int(neal['UnifiedRobustNUTS']['divergences'])} divergences against "
            f"{_int(neal['BlackJAX NUTS']['divergences'])} for BlackJAX NUTS. Non-centre funnel models instead."
        ),
        (
            f"- **Applied models on simulated data (REPORT Table 5).** {sum(bool(r['converged']) for r in applied)} of "
            f"{len(applied)} runs converge; see the table before reading anything into efficiency differences."
        ),
        (
            f"- **Scope.** Each number is one seed ({cfg['seed']}) of {cfg['n_chains']} chains × {cfg['n_samples']:,} "
            f"draws, with no error bars."
        ),
    ]
    return "\n".join(bullets) + "\n"


# --- multi-seed ablation (reports/ablation_results.json) ---------------------------------------


def _ablation_label(target: str) -> str:
    return _ABLATION_TARGETS.get(target, target)


def _ablation_by_seed(reports: Reports, target: str, arm: str) -> dict[int, dict[str, Any]]:
    return {r["seed"]: r for r in reports.ablation["rows"] if r["target"] == target and r["arm"] == arm}


def _paired_ratios(reports: Reports, target: str, arm: str, reference: str) -> list[float]:
    """Per-seed tail-ESS/grad of ``arm`` divided by that of ``reference``, over seeds where both exist."""
    a, b = _ablation_by_seed(reports, target, arm), _ablation_by_seed(reports, target, reference)
    return [
        a[s]["tail_ess_per_grad"] / b[s]["tail_ess_per_grad"]
        for s in sorted(a.keys() & b.keys())
        if a[s]["tail_ess_per_grad"] is not None and b[s]["tail_ess_per_grad"]
    ]


def _ablation_targets(reports: Reports) -> list[str]:
    return list(dict.fromkeys(r["target"] for r in reports.ablation["rows"]))


def _is_benign(target: str) -> bool:
    return target.startswith("gauss")


def _ablation_setup(reports: Reports) -> str:
    cfg = reports.ablation["config"]
    prov = reports.ablation.get("provenance", {})
    when = f" (run {prov['timestamp_utc'][:10]}, commit {prov['git_sha'][:7]})" if "git_sha" in prov else ""
    return (
        f"{len(cfg['seeds'])} seeds per cell; each run has {cfg['chains']} chains × {cfg['draws']:,} draws after "
        f"{cfg['tune']} warm-up steps. Every arm runs BlackJAX NUTS with window adaptation through "
        "`sample_blackjax_unified_nuts`; arms differ only in the coordinate transform (or, for `scale_mixture`, in "
        "the model's parameterisation), and share each seed's initial positions and random key, so ratios are "
        f"paired by seed. Source: `reports/ablation_results.json`{when}."
    )


def render_ablation_table(reports: Reports) -> str:
    cfg = reports.ablation["config"]
    body = []
    for target in _ablation_targets(reports):
        arms = list(dict.fromkeys(r["arm"] for r in reports.ablation["rows"] if r["target"] == target))
        for arm in arms:
            runs = list(_ablation_by_seed(reports, target, arm).values())
            if arm == "stock":
                ratio_cell, wins = "1 (reference)", "–"
            else:
                ratios = _paired_ratios(reports, target, arm, "stock")
                q25, med, q75 = np.percentile(ratios, [25, 50, 75])
                ratio_cell = f"{med:.2f}× [{q25:.2f}, {q75:.2f}]"
                wins = f"{sum(r > 1.0 for r in ratios)} of {len(ratios)}"
            converged = sum(bool(r["converged"]) for r in runs)
            warned = sum(bool(r.get("tail_alignment_warning")) for r in runs)
            errs = [r["tail_prob_abs_err"] for r in runs if r.get("tail_prob_abs_err") is not None]
            body.append(
                [
                    _ablation_label(target),
                    f"`{arm}`",
                    ratio_cell,
                    wins,
                    f"{converged} of {len(runs)}",
                    _int(float(np.median([r["total_grad_evals"] for r in runs]))),
                    f"{float(np.median([r['divergences'] for r in runs])):g}",
                    f"{float(np.median(errs)):.4f}" if errs else "n/a",
                    f"{warned} of {len(runs)}",
                ]
            )
    header = [
        "Target",
        "Arm",
        "Tail-ESS/grad vs stock, median [IQR]",
        "Seeds better than stock",
        "Seeds converged",
        "Median gradients",
        "Median divergences",
        "Median tail-probability error",
        "Seeds with tail-alignment warning",
    ]
    legend = "; ".join(f"**{_ablation_label(t)}**: {desc}" for t, desc in cfg.get("target_descriptions", {}).items())
    arms = cfg.get("arm_settings", {})
    arm_legend = "; ".join(f"`{a}`: {spec['description']}" for a, spec in arms.items() if "description" in spec)
    note = (
        "Tail-probability error is the mean over coordinates and p ∈ {0.01, 0.99} of |P̂(x_i < q_p) − p| at the "
        "exact marginal quantile q_p; it is less noisy than a relative error on the 0.99 quantile."
    )
    parts = [_ablation_setup(reports), note]
    if legend:
        parts.append("Targets: " + legend + ".")
    if arm_legend:
        parts.append("Arms: " + arm_legend + ".")
    return "\n\n".join(parts) + "\n\n" + _table(header, body, numeric_from=2)


def _never_visits_a_tail(row: dict[str, Any]) -> bool:
    """Some (coordinate, p) cell of the tail metric had no draw beyond the exact quantile."""
    return bool(row.get("tail_prob_n_degenerate") or 0)


def _nonzero(listing: str) -> str:
    """Keep the "target: k of n seeds" entries with k > 0; "none" if there are none."""
    kept = [part for part in listing.split("; ") if part and not part.split(": ", 1)[1].startswith("0 of ")]
    return "; ".join(kept) if kept else "none"


def render_ablation_findings(reports: Reports) -> str:
    targets = _ablation_targets(reports)

    def arms_of(target: str) -> set[str]:
        return {r["arm"] for r in reports.ablation["rows"] if r["target"] == target}

    def vs(arm: str, reference: str, verb: str) -> str:
        parts = []
        for t in targets:
            if not {arm, reference} <= arms_of(t):
                continue
            ratios = _paired_ratios(reports, t, arm, reference)
            if ratios:
                wins = sum(r > 1.0 for r in ratios)
                parts.append(
                    f"{_ablation_label(t)} {float(np.median(ratios)):.1f}× ({verb} on {wins} of {len(ratios)} seeds)"
                )
        return "; ".join(parts)

    def seeds_where(arm: str, predicate: Callable[[dict[str, Any]], bool]) -> str:
        parts = []
        for t in targets:
            runs = list(_ablation_by_seed(reports, t, arm).values())
            if runs:
                parts.append(f"{_ablation_label(t)}: {sum(predicate(r) for r in runs)} of {len(runs)} seeds")
        return "; ".join(parts)

    bullets = [
        "- **Arcsinh vs stock NUTS** (paired tail-ESS per gradient, median over seeds): "
        + vs("arcsinh", "stock", "wins")
        + ".",
        "- **Rescaling alone** (`affine`: the same calibration without the arcsinh) vs stock: "
        + vs("affine", "stock", "wins")
        + ".",
        "- **Reparameterising by hand** (`scale_mixture`, where the model is a Student-t scale mixture) vs `arcsinh`: "
        + vs("scale_mixture", "arcsinh", "better than arcsinh")
        + ".",
        *(
            [
                "- **Tukey h-map** (`tukey_h`) vs `arcsinh`: "
                + vs("tukey_h", "arcsinh", "better than arcsinh")
                + "; vs stock: "
                + vs("tukey_h", "stock", "wins")
                + "."
            ]
            if any(r["arm"] == "tukey_h" for r in reports.ablation["rows"])
            else []
        ),
        f"- **Seeds converged** (max R-hat at most {RHAT_CONVERGED}) with `arcsinh`: "
        + seeds_where("arcsinh", lambda r: bool(r["converged"]))
        + "; with `stock`: "
        + seeds_where("stock", lambda r: bool(r["converged"]))
        + ".",
        "- **Seeds with any post-warm-up divergence** with `arcsinh`: "
        + seeds_where("arcsinh", lambda r: r["divergences"] > 0)
        + ".",
        "- **Runs in which some coordinate never visits a tail** (no draw beyond its exact 0.01 or 0.99 quantile, "
        "which R-hat cannot detect) with `arcsinh`: "
        + _nonzero(seeds_where("arcsinh", _never_visits_a_tail))
        + "; with `stock`: "
        + _nonzero(seeds_where("stock", _never_visits_a_tail))
        + ".",
    ]
    rows = reports.ablation["rows"]
    if any("tail_alignment_warning" in r for r in rows):
        missed = [r for r in rows if _never_visits_a_tail(r)]
        bullets.append(
            "- **Tail-alignment warning** (`TailAlignmentWarning`, computed on each run's own draws; it flags heavy "
            "tails shared across coordinates) with `arcsinh`: "
            + seeds_where("arcsinh", lambda r: bool(r.get("tail_alignment_warning")))
            + f". Among runs of any arm in which some coordinate never visits a tail, it fired on "
            f"{sum(bool(r.get('tail_alignment_warning')) for r in missed)} of {len(missed)}."
        )
    return "\n".join(bullets) + "\n"


# --- direct competitors (reports/competitor_results.json) --------------------------------------

_COMPETITOR_TARGETS: dict[str, str] = {
    "task1": "Task 1",
    "task3": "Task 3",
    "corr_t1": "rotated t (ν = 1)",
    "corr_gauss": "rotated Gaussian",
}


def _competitor_runs(reports: Reports, target: str, arm: str) -> dict[int, dict[str, Any]]:
    return {
        r["seed"]: r
        for r in reports.competitors["rows"]
        if r["target"] == target and r["arm"] == arm and r.get("status", "ok") == "ok"
    }


def _competitor_ratios(reports: Reports, target: str, arm: str, reference: str, metric: str) -> list[float]:
    a, b = _competitor_runs(reports, target, arm), _competitor_runs(reports, target, reference)
    return [
        a[s][metric] / b[s][metric] for s in sorted(a.keys() & b.keys()) if a[s][metric] is not None and b[s][metric]
    ]


def _competitor_targets(reports: Reports) -> list[str]:
    return list(dict.fromkeys(r["target"] for r in reports.competitors["rows"]))


def render_competitor_table(reports: Reports) -> str:
    data = reports.competitors
    ref = data["reference_arm"]
    cfg = data["config"]
    body = []
    for target in _competitor_targets(reports):
        for arm in data["arms"]:
            runs = list(_competitor_runs(reports, target, arm).values())
            if not runs:
                continue
            if arm == ref:
                grad_cell, sec_cell = "1 (reference)", "1"
            else:
                q25, med, q75 = np.percentile(
                    _competitor_ratios(reports, target, arm, ref, "tail_ess_per_grad"), [25, 50, 75]
                )
                grad_cell = f"{med:.2f}× [{q25:.2f}, {q75:.2f}]"
                sec_cell = f"{float(np.median(_competitor_ratios(reports, target, arm, ref, 'tail_ess_per_sec'))):.2f}×"
            errs = [r["tail_prob_abs_err_p99"] for r in runs if r.get("tail_prob_abs_err_p99") is not None]
            mcse = [r["tail_prob_mcse_p99"] for r in runs if r.get("tail_prob_mcse_p99") is not None]
            rhats = [r["split_rhat"] for r in runs if r.get("split_rhat") is not None]
            body.append(
                [
                    _COMPETITOR_TARGETS.get(target, target),
                    f"`{arm}`",
                    grad_cell,
                    sec_cell,
                    f"{sum(bool(r['converged']) for r in runs)} of {len(runs)}",
                    _rhat(max(rhats)) if rhats else "n/a",
                    _int(sum(r["divergences"] or 0 for r in runs)),
                    f"{float(np.median(errs)):.4f} ({float(np.median(mcse)):.4f})" if errs and mcse else "n/a",
                ]
            )
    header = [
        "Target",
        "Arm",
        "Tail-ESS/grad vs stock, median [IQR]",
        "Tail-ESS/s vs stock, median",
        "Seeds converged",
        "Max R-hat",
        "Divergences, all seeds",
        "Tail-probability error at 0.99, median (MCSE)",
    ]
    arms = "; ".join(f"`{a}`: {spec['description']}" for a, spec in data["arms"].items())
    excluded = "; ".join(f"`{name}` ({spec['status']})" for name, spec in data.get("excluded_arms", {}).items())
    setup = (
        f"{len(cfg['seeds'])} seeds; each run has {cfg['chains']} chains × {cfg['draws']:,} draws. Ratios are paired by "
        "seed against stock BlackJAX NUTS; gradient counts include warm-up, and nutpie's flow training reuses "
        "warm-up gradients, so it adds wall time but no gradient evaluations. Wall-time ratios favour the "
        "jit-compiled BlackJAX arms over nutpie's Python callbacks. Source: `reports/competitor_results.json`."
    )
    parts = [setup, "Arms: " + arms + "."]
    if excluded:
        parts.append("Excluded: " + excluded + ".")
    return "\n\n".join(parts) + "\n\n" + _table(header, body, numeric_from=2)


def render_competitor_findings(reports: Reports) -> str:
    arms = reports.competitors["arms"]

    def vs(arm: str) -> str:
        parts = []
        for t in _competitor_targets(reports):
            ratios = _competitor_ratios(reports, t, "arcsinh_nuts", arm, "tail_ess_per_grad")
            if ratios:
                wins = sum(r > 1.0 for r in ratios)
                parts.append(
                    f"{_COMPETITOR_TARGETS.get(t, t)} {float(np.median(ratios)):.1f}× "
                    f"(better on {wins} of {len(ratios)} seeds)"
                )
        return "; ".join(parts)

    def converged(arm: str) -> str:
        return "; ".join(
            f"{_COMPETITOR_TARGETS.get(t, t)}: {sum(bool(r['converged']) for r in runs.values())} of {len(runs)}"
            for t in _competitor_targets(reports)
            if (runs := _competitor_runs(reports, t, arm))
        )

    bullets = []
    if "nutpie_diag" in arms:
        bullets.append(
            "- **Arcsinh vs nutpie's default** (gradient-based mass matrix), tail-ESS per gradient: "
            + vs("nutpie_diag")
            + "."
        )
    if "nutpie_flow" in arms:
        bullets.append(
            "- **Arcsinh vs nutpie's normalizing-flow adaptation** (same warm-up length), tail-ESS per gradient: "
            + vs("nutpie_flow")
            + "."
        )
    if "nutpie_flow_tune1500" in arms:
        bullets.append(
            "- **Arcsinh vs the flow with nutpie's own default warm-up** (1,500 draws, all counted): "
            + vs("nutpie_flow_tune1500")
            + "."
        )
    bullets.append(
        f"- **Seeds converged** (max R-hat at most {RHAT_CONVERGED}): `arcsinh_nuts` "
        + converged("arcsinh_nuts")
        + ("; `nutpie_flow` " + converged("nutpie_flow") if "nutpie_flow" in arms else "")
        + "."
    )
    return "\n".join(bullets) + "\n"


BLOCKS: dict[str, Callable[[Reports], str]] = {
    "readme-competitors": render_competitor_findings,
    "competitor-findings": render_competitor_findings,
    "competitor-table": render_competitor_table,
    "readme-ablation": render_ablation_findings,
    "ablation-findings": render_ablation_findings,
    "ablation-table": render_ablation_table,
    "readme-headline": render_readme_headline,
    "sota-task1": render_sota_task1,
    "sota-eight-schools": render_sota_eight_schools,
    "sota-neal": render_sota_neal,
    "sota-task3": render_sota_task3,
    "sota-task4": render_sota_task4,
    "applied": render_applied,
    "langevin-sweep": render_langevin_sweep,
    "langevin-benign": render_langevin_benign,
    "lean-inventory": render_lean_inventory,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="rewrite stale blocks in place")
    mode.add_argument("--check", action="store_true", help="exit 1 if any block is stale")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: cwd)")
    args = parser.parse_args(argv)
    if args.write:
        changed = write(args.root)
        print("updated: " + ", ".join(changed) if changed else "all generated blocks are current")
        return 0
    problems = check(args.root)
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
