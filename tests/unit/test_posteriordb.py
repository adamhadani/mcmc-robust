"""Tests for the posteriordb reader (`mcmc_robust.bench.posteriordb`), on a miniature database in tmp_path."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mcmc_robust.bench.posteriordb import (
    POSTERIORDB_SHA,
    PosteriorDatabase,
    fetch_posteriordb,
    stan_name_to_posteriordb,
)

STAN_CODE = "parameters { real mu; real<lower=0> tau; } model { mu ~ normal(0, 1); tau ~ lognormal(0, 1); }\n"


def _zip_json(path: Path, member: str, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, json.dumps(payload))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def make_database(root: Path) -> Path:
    """Two posteriors sharing a model: `toy-norm` has reference draws (3 chains x 4 draws), `toy2-norm` has none."""
    db = root / "posterior_database"
    _write_json(
        db / "models" / "info" / "norm.info.json",
        {"name": "norm", "model_implementations": {"stan": {"model_code": "models/stan/norm.stan"}}},
    )
    (db / "models" / "stan").mkdir(parents=True)
    (db / "models" / "stan" / "norm.stan").write_text(STAN_CODE)
    for data in ("toy", "toy2"):
        _zip_json(db / "data" / "data" / f"{data}.json.zip", f"{data}.json", {"N": 3, "y": [1.0, 2.0, 3.0]})
    _write_json(
        db / "posteriors" / "toy-norm.json",
        {"name": "toy-norm", "model_name": "norm", "data_name": "toy", "reference_posterior_name": "toy-norm"},
    )
    _write_json(
        db / "posteriors" / "toy2-norm.json",
        {"name": "toy2-norm", "model_name": "norm", "data_name": "toy2", "reference_posterior_name": None},
    )
    chains = [{"mu": [c + 0.1 * i for i in range(4)], "tau": [1.0 + c + 0.01 * i for i in range(4)]} for c in range(3)]
    _zip_json(db / "reference_posteriors" / "draws" / "draws" / "toy-norm.json.zip", "toy-norm.json", chains)
    return root


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stan", "pdb"),
    [("mu", "mu"), ("theta.1", "theta[1]"), ("Sigma.2.3", "Sigma[2,3]"), ("b_x.10", "b_x[10]")],
)
def test_stan_names_map_to_posteriordb_names(stan: str, pdb: str) -> None:
    assert stan_name_to_posteriordb(stan) == pdb


@pytest.mark.unit
def test_lists_only_posteriors_with_reference_draws(tmp_path: Path) -> None:
    db = PosteriorDatabase(make_database(tmp_path))
    assert db.reference_posterior_names() == ["toy-norm"]


@pytest.mark.unit
def test_loads_model_code_data_and_reference_draws(tmp_path: Path) -> None:
    spec = PosteriorDatabase(make_database(tmp_path)).posterior("toy-norm")
    assert (spec.name, spec.model_name, spec.data_name) == ("toy-norm", "norm", "toy")
    assert spec.stan_file.read_text() == STAN_CODE
    assert json.loads(spec.data_json) == {"N": 3, "y": [1.0, 2.0, 3.0]}
    assert spec.reference_names == ("mu", "tau")
    assert spec.reference_draws.shape == (3, 4, 2)
    np.testing.assert_allclose(spec.reference_draws[2, 1], [2.1, 3.01])


@pytest.mark.unit
def test_a_posterior_without_reference_draws_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no reference draws"):
        PosteriorDatabase(make_database(tmp_path)).posterior("toy2-norm")


@pytest.mark.unit
def test_a_missing_database_says_how_to_fetch_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch"):
        PosteriorDatabase(tmp_path / "nowhere")


@pytest.mark.unit
def test_fetch_checks_out_the_requested_commit_and_reports_it(tmp_path: Path) -> None:
    """fetch_posteriordb makes a shallow checkout of one commit; the database reports that commit."""
    origin = make_database(tmp_path / "origin")

    def git(*args: str, cwd: Path = origin) -> str:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "first")
    first = git("rev-parse", "HEAD")
    (origin / "posterior_database" / "posteriors" / "toy2-norm.json").unlink()
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "second")

    dest = fetch_posteriordb(tmp_path / "checkout", sha=first, remote=str(origin))
    db = PosteriorDatabase(dest)
    assert db.git_sha() == first
    assert (dest / "posterior_database" / "posteriors" / "toy2-norm.json").exists()
    # Fetching again into the same directory is a no-op that keeps the commit.
    assert fetch_posteriordb(dest, sha=first, remote=str(origin)) == dest
    assert PosteriorDatabase(dest).git_sha() == first


@pytest.mark.unit
def test_the_pinned_commit_is_a_full_sha() -> None:
    assert len(POSTERIORDB_SHA) == 40 and int(POSTERIORDB_SHA, 16) >= 0
