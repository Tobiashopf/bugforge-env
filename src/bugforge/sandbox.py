"""The agent's workspace: a temporary directory plus the file and test tools.

This backend runs the agent's test command as a local subprocess with a
timeout. It isolates *grading* (see verifier.py) but it is not a security
boundary for running untrusted, model-written code: use the Docker backend
for that (planned, see README).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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
    def __init__(self, files: dict[str, str]):
        self._initial = dict(files)
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
        p = self.root.joinpath(*rel.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- tools -------------------------------------------------------------
    def list_files(self) -> list[str]:
        assert self.root is not None
        return sorted(str(p.relative_to(self.root).as_posix())
                      for p in self.root.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts
                      and p.name not in {"report.xml", "pytest.ini"})

    def read_file(self, name: str) -> str:
        p = self._path(name)
        if not p.is_file():
            raise ToolError(f"no such file: {name}")
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
        tests = sorted(p.name for p in self.root.glob("test_*.py"))
        if not tests:
            return TestRun(0, 0, "no test files found")
        ini = self.root / "pytest.ini"
        if not ini.exists():
            ini.write_text("[pytest]\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTEST_ADDOPTS"}}
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8")
        proc, report = run_pytest(self.root, tests + ["--tb=short"], timeout, env)
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
