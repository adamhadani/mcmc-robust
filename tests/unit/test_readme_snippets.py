"""Every ```python block in README.md must run as written."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"
SNIPPETS = re.findall(r"^```python\n(.*?)^```", README.read_text(), flags=re.DOTALL | re.MULTILINE)


def test_readme_has_python_snippets() -> None:
    assert SNIPPETS, "README.md should show at least one runnable usage example"


@pytest.mark.parametrize("code", SNIPPETS, ids=[f"snippet{i}" for i in range(len(SNIPPETS))])
def test_readme_snippet_runs(code: str) -> None:
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})  # noqa: S102 - running the README is the test
