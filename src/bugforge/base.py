"""The hand-written base modules that bugs are injected into.

Each base task lives in ``tasks/base/<name>/`` and has three files:

- ``<name>.py``        the correct module (also the reference at grading time)
- ``test_<name>.py``   unit tests; a random subset is shown to the agent
- ``fuzz_<name>.py``   a differential fuzz test against the reference, never shown
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

from .mutations import normalize


def tasks_root() -> Path:
    env = os.environ.get("BUGFORGE_TASKS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "tasks" / "base"


@dataclass(frozen=True)
class BaseTask:
    name: str
    source: str        # normalized correct module
    tests: str         # unit test file
    fuzz: str          # differential fuzz file

    @property
    def module_file(self) -> str:
        return f"{self.name}.py"

    @property
    def test_file(self) -> str:
        return f"test_{self.name}.py"

    @property
    def fuzz_file(self) -> str:
        return f"fuzz_{self.name}.py"

    def test_names(self) -> list[str]:
        tree = ast.parse(self.tests)
        return [n.name for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def load(name: str, root: Path | None = None) -> BaseTask:
    d = (root or tasks_root()) / name
    return BaseTask(
        name=name,
        source=normalize((d / f"{name}.py").read_text(encoding="utf-8")),
        tests=(d / f"test_{name}.py").read_text(encoding="utf-8"),
        fuzz=(d / f"fuzz_{name}.py").read_text(encoding="utf-8"),
    )


def available(root: Path | None = None) -> list[str]:
    r = root or tasks_root()
    return sorted(p.name for p in r.iterdir()
                  if p.is_dir() and (p / f"{p.name}.py").exists())


def subset_tests(test_source: str, keep: set[str]) -> str:
    """The test file with only the named test functions kept (helpers stay)."""
    tree = ast.parse(test_source)
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
                 or n.name in keep]
    return ast.unparse(tree) + "\n"
