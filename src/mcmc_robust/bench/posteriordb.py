r"""Reader for a local checkout of posteriordb (Magnusson et al. 2025), pinned to one commit.

posteriordb pairs Stan models with data sets and, for some of the pairs, publishes reference
posterior draws (10 chains of 1000 thinned draws from long Stan runs). This module reads a
checkout of the repository directly; it does not use the `posteriordb` Python package.

- `fetch_posteriordb(dest)` makes a shallow checkout of `POSTERIORDB_SHA`.
- `PosteriorDatabase(root).reference_posterior_names()` lists the posteriors with reference draws.
- `PosteriorDatabase(root).posterior(name)` returns a `PosteriorSpec`: the Stan file, the data as
  a JSON string, and the reference draws as an array of shape `(chains, draws, k)` with posteriordb
  parameter names (`theta[1]`).

`stan_name_to_posteriordb` maps BridgeStan's names (`theta.1`) to posteriordb's.
"""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSTERIORDB_REPO: str = "https://github.com/stan-dev/posteriordb.git"

#: The posteriordb commit the reported results use (master, 2026-10-09).
POSTERIORDB_SHA: str = "91027f9cdd13b7e76283fec3f2898f59d4b83ec4"

#: Where the runner looks for the checkout unless `--posteriordb` or $POSTERIORDB_PATH says otherwise.
DEFAULT_POSTERIORDB_PATH: Path = Path(
    os.environ.get("POSTERIORDB_PATH", Path.home() / ".cache" / "mcmc-robust" / f"posteriordb-{POSTERIORDB_SHA[:12]}")
)


def stan_name_to_posteriordb(name: str) -> str:
    """Maps a BridgeStan parameter name (`Sigma.2.3`) to posteriordb's (`Sigma[2,3]`)."""
    base, *index = name.split(".")
    return f"{base}[{','.join(index)}]" if index else base


def _read_zipped_json(path: Path) -> object:
    with zipfile.ZipFile(path) as zf:
        (member,) = zf.namelist()
        return json.loads(zf.read(member))


@dataclass(frozen=True)
class PosteriorSpec:
    """One posteriordb posterior with reference draws."""

    name: str
    model_name: str
    data_name: str
    stan_file: Path
    data_json: str
    reference_names: tuple[str, ...]
    reference_draws: np.ndarray


class PosteriorDatabase:
    """A local posteriordb checkout (the directory that contains `posterior_database/`)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.db = self.root / "posterior_database"
        if not self.db.is_dir():
            raise FileNotFoundError(
                f"No posteriordb checkout at {self.root}. Run "
                "`uv run python -m mcmc_robust.bench.posteriordb_runner --fetch` or set $POSTERIORDB_PATH."
            )

    def _draws_file(self, reference_name: str) -> Path:
        return self.db / "reference_posteriors" / "draws" / "draws" / f"{reference_name}.json.zip"

    def _posterior_json(self, name: str) -> dict[str, object]:
        record = json.loads((self.db / "posteriors" / f"{name}.json").read_text())
        assert isinstance(record, dict)
        return record

    def reference_posterior_names(self) -> list[str]:
        """Names of the posteriors whose reference draws are present, sorted."""
        names = []
        for path in sorted((self.db / "posteriors").glob("*.json")):
            ref = json.loads(path.read_text()).get("reference_posterior_name")
            if ref and self._draws_file(ref).exists():
                names.append(path.stem)
        return names

    def posterior(self, name: str) -> PosteriorSpec:
        """Loads `name`; raises `ValueError` if it has no reference draws."""
        record = self._posterior_json(name)
        ref = record.get("reference_posterior_name")
        if not ref or not self._draws_file(str(ref)).exists():
            raise ValueError(f"Posterior {name!r} has no reference draws")
        model_name, data_name = str(record["model_name"]), str(record["data_name"])
        info = json.loads((self.db / "models" / "info" / f"{model_name}.info.json").read_text())
        stan_file = self.db / info["model_implementations"]["stan"]["model_code"]
        data = _read_zipped_json(self.db / "data" / "data" / f"{data_name}.json.zip")
        chains = _read_zipped_json(self._draws_file(str(ref)))
        assert isinstance(chains, list)
        names = tuple(chains[0])
        draws = np.stack([np.column_stack([np.asarray(c[n], dtype=np.float64) for n in names]) for c in chains])
        return PosteriorSpec(
            name=name,
            model_name=model_name,
            data_name=data_name,
            stan_file=stan_file,
            data_json=json.dumps(data),
            reference_names=names,
            reference_draws=draws,
        )

    def git_sha(self) -> str | None:
        """The checked-out commit, or None if the checkout is not a git repository."""
        try:
            out = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()


def fetch_posteriordb(dest: Path | str, sha: str = POSTERIORDB_SHA, remote: str = POSTERIORDB_REPO) -> Path:
    """Shallow-checks-out commit `sha` of `remote` into `dest` (a no-op if `dest` is already at `sha`)."""
    dest = Path(dest)
    if dest.is_dir() and (dest / "posterior_database").is_dir() and PosteriorDatabase(dest).git_sha() == sha:
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(dest), *args], check=True, capture_output=True, text=True)

    git("init", "-q")
    git("fetch", "-q", "--depth", "1", remote, sha)
    git("checkout", "-q", "--detach", "FETCH_HEAD")
    return dest


__all__ = [
    "DEFAULT_POSTERIORDB_PATH",
    "POSTERIORDB_REPO",
    "POSTERIORDB_SHA",
    "PosteriorDatabase",
    "PosteriorSpec",
    "fetch_posteriordb",
    "stan_name_to_posteriordb",
]
