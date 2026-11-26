"""The agent's workspace: a temporary directory plus the file and test tools.

``run_tests`` executes model-written code, so where it runs matters:

- **Docker runner**: the workspace runs as the agent left it (its own test
  files, conftest, anything) inside a network-less, resource-limited container.
- **Local runner (guarded)**: nothing the agent wrote runs unless it passes the
  same static policy as grading. Module files are policy-checked and copied,
  with the *original* visible tests, into a fresh directory; edited or added
  test files are ignored. The policy forbids imports outside a pure-stdlib
  allow-list, ``eval``/``getattr``/dunder access and import-time calls, but
  is not a security boundary. Use only with trusted code.
"""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import policy
from .runner import LocalRunner
from .verifier import parse_junit, run_pytest

MAX_FILE_BYTES = 200_000


class ToolError(Exception):
    """A tool call the agent made incorrectly; reported back as an observation."""


@dataclass
class TestRun:
    passed: int
    failed: int
    output: str
    timed_out: bool = False

    @property
    def all_passed(self) -> bool:
        return not self.timed_out and self.failed == 0 and self.passed > 0


class Workspace:
    def __init__(self, files: dict[str, str], module_files: list[str] | None = None,
                 runner=None, *, project_modules: frozenset[str] = frozenset()):
        self._initial = dict(files)
        self.module_files = module_files or sorted(
            f for f in files if f.endswith(".py") and not f.startswith("test_"))
        self.runner = runner or LocalRunner()
        self.project_modules = project_modules
        self.root: Path | None = None

    def __enter__(self):
        self.root = Path(tempfile.mkdtemp(prefix="bugforge-ws-"))
        for name, content in self._initial.items():
            self._path(name).write_text(content, encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None

    # -- path safety -------------------------------------------------------
    def _path(self, name: str) -> Path:
        if self.root is None:
            raise RuntimeError("workspace is not open")
        rel = PurePosixPath(name.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or ":" in name:
            raise ToolError(f"path '{name}' is outside the workspace")
        # The Docker workspace is writable by the candidate. Never follow
        # links it leaves behind when the host subsequently accesses files.
        p = self.root
        for i, part in enumerate(rel.parts):
            p = p / part
            last = i == len(rel.parts) - 1
            try:
                mode = p.lstat().st_mode
            except FileNotFoundError:
                if not last:
                    p.mkdir()
                continue
            if stat.S_ISLNK(mode) or (last and not stat.S_ISREG(mode)) or (
                    not last and not stat.S_ISDIR(mode)):
                raise ToolError(f"path '{name}' is not a regular workspace file")
        return p

    # -- tools -------------------------------------------------------------
    def list_files(self) -> list[str]:
        assert self.root is not None
        return sorted(str(p.relative_to(self.root).as_posix())
                      for p in self.root.rglob("*")
                      if not p.is_symlink() and p.is_file() and "__pycache__" not in p.parts
                      and p.name not in {"report.xml", "pytest.ini"})

    def read_file(self, name: str) -> str:
        p = self._path(name)
        if not p.is_file():
            raise ToolError(f"no such file: {name}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("file too large")
        return p.read_text(encoding="utf-8", errors="replace")

    def write_file(self, name: str, content: str) -> None:
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError("file too large")
        self._path(name).write_text(content, encoding="utf-8")

    def replace(self, name: str, old: str, new: str) -> None:
        text = self.read_file(name)
        n = text.count(old)
        if n != 1:
            raise ToolError(f"'old' must occur exactly once in {name} (found {n})")
        self.write_file(name, text.replace(old, new))

    def run_tests(self, timeout: float = 30.0) -> TestRun:
        assert self.root is not None
        if self.runner.isolates_code:
            return self._pytest(self.root, timeout)

        problems = {f: policy.check(self.read_file(f), project_modules=self.project_modules)
                    for f in self.module_files
                    if (self.root / f).is_file()}
        problems = {f: p for f, p in problems.items() if p}
        if problems:
            lines = [f"{f}: {msg}" for f, ps in problems.items() for msg in ps]
            return TestRun(0, 1, "tests not run: module code violates the sandbox policy\n"
                           + "\n".join(lines))
        with tempfile.TemporaryDirectory(prefix="bugforge-run-") as tmp:
            d = Path(tmp)
            for f in self.module_files:
                if (self.root / f).is_file():
                    (d / f).write_text(self.read_file(f), encoding="utf-8")
            notes = []
            for name, content in self._initial.items():
                if name.startswith("test_") and name.endswith(".py"):
                    (d / name).write_text(content, encoding="utf-8")
                    if self.read_file(name) != content:
                        notes.append(name)
            run = self._pytest(d, timeout)
        if notes:
            run.output += ("\n\nnote: edits to " + ", ".join(notes)
                           + " are ignored; the original tests were run")
        return run

    def _pytest(self, directory: Path, timeout: float) -> TestRun:
        tests = sorted(p.name for p in directory.glob("test_*.py"))
        if not tests:
            return TestRun(0, 0, "no test files found")
        ini = directory / "pytest.ini"
        if not ini.exists() and not ini.is_symlink():
            ini.write_text("[pytest]\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTEST_ADDOPTS"}}
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8")
        proc, report = run_pytest(directory, tests + ["--tb=short"], timeout, env, self.runner)
        if proc is None:
            return TestRun(0, 0, f"tests timed out after {timeout:.0f}s", timed_out=True)
        total, passed, _, failed = parse_junit(report)
        report.unlink(missing_ok=True)
        n_failed = max(len(failed), total - passed)
        if total == 0 and proc.returncode != 0:
            n_failed = max(n_failed, 1)
        return TestRun(passed, n_failed, (proc.stdout + proc.stderr).strip())

    def snapshot(self, names: list[str]) -> dict[str, str]:
        out = {}
        for name in names:
            try:
                out[name] = self.read_file(name)
            except ToolError:
                pass
        return out
