"""The verifier: grades a submission against hidden tests in a fresh directory.

Layered defenses, each covering a different way to fake a pass:

1. **File allow-list.** Only the module files named by the task are taken from
   the workspace. Edited tests, ``conftest.py``, ``pytest.ini`` or
   ``sitecustomize.py`` the agent wrote never reach the grading directory.
2. **Static policy** (``policy.py``). Code that would run at import time, touch
   the interpreter, or reach the reference is rejected before anything runs.
3. **Pristine hidden tests** in a new temporary directory with its own empty
   ``pytest.ini``, plugin autoloading off and a clean ``PYTHONPATH``.
4. **Counted results.** The JUnit report must list exactly the expected number
   of tests, all passed, none skipped. A process that exits 0 without running
   the tests scores zero.
5. **Differential fuzzing** against the original module, so hard-coding the
   answers to the visible tests does not pass.
6. **Timeout.** An infinite loop scores zero instead of hanging the grader.

With ``runner=DockerRunner()`` the grading run itself also happens inside a
network-less container (see runner.py).
"""
from __future__ import annotations

import ast
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from . import policy
from .base import BaseTask
from .runner import LocalRunner, Result


@dataclass
class VerifyResult:
    passed: int
    total: int
    expected: int
    failed: list[str] = field(default_factory=list)
    violations: dict[str, list[str]] = field(default_factory=dict)
    timed_out: bool = False
    note: str = ""

    @property
    def solved(self) -> bool:
        return (not self.violations and not self.timed_out
                and self.total == self.expected and self.passed == self.expected)

    @property
    def pass_fraction(self) -> float:
        if self.violations or self.timed_out or self.expected == 0:
            return 0.0
        return self.passed / self.expected


def _count_tests(source: str) -> int:
    tree = ast.parse(source)
    return sum(1 for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))


def expected_count(bases: list[BaseTask]) -> int:
    return sum(_count_tests(b.tests) + _count_tests(b.fuzz) for b in bases)


def _clean_env(fuzz_seed: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in {"PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PYTEST_ADDOPTS",
                        "PYTEST_PLUGINS"}}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", BUGFORGE_FUZZ_SEED=str(fuzz_seed))
    return env


def run_pytest(directory: Path, files: list[str], timeout: float, env: dict[str, str],
               runner=None) -> tuple[Result | None, Path]:
    """Run pytest on `files` (relative to `directory`) with its JUnit report in place."""
    runner = runner or LocalRunner()
    report = directory / "report.xml"
    args = ["-q", "-p", "no:cacheprovider", "-c", "pytest.ini", "--rootdir", ".",
            "--junitxml=report.xml", "-o", "junit_family=xunit2", *files]
    return runner.pytest(directory, args, timeout, env), report


def parse_junit(report: Path) -> tuple[int, int, int, list[str]]:
    """(tests, passed, skipped, failed names) from a JUnit XML report."""
    if not report.exists():
        return 0, 0, 0, []
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError:
        return 0, 0, 0, []
    tests = passed = skipped = 0
    failed: list[str] = []
    for case in root.iter("testcase"):
        tests += 1
        kinds = {child.tag for child in case}
        if "skipped" in kinds:
            skipped += 1
        elif kinds & {"failure", "error"}:
            failed.append(f"{case.get('classname', '')}::{case.get('name', '')}")
        else:
            passed += 1
    for suite in root.iter("testsuite"):  # collection errors appear only as counts
        errors = int(suite.get("errors", 0))
        if errors and not failed and tests == 0:
            failed.append("<collection error>")
    return tests, passed, skipped, failed


def verify(candidates: dict[str, str], bases: list[BaseTask], *, fuzz_seed: int = 0,
           timeout: float = 60.0, runner=None) -> VerifyResult:
    """Grade candidate module sources (keyed by file name) against the hidden tests."""
    expected = expected_count(bases)
    violations: dict[str, list[str]] = {}
    for base in bases:
        src = candidates.get(base.module_file)
        if src is None:
            violations[base.module_file] = ["file missing from submission"]
            continue
        problems = policy.check(src)
        if problems:
            violations[base.module_file] = problems
    if violations:
        return VerifyResult(0, 0, expected, violations=violations, note="policy violation")

    with tempfile.TemporaryDirectory(prefix="bugforge-verify-") as tmp:
        d = Path(tmp)
        (d / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        files: list[str] = []
        for base in bases:
            (d / base.module_file).write_text(candidates[base.module_file], encoding="utf-8")
            (d / f"ref_{base.module_file}").write_text(base.source, encoding="utf-8")
            (d / base.test_file).write_text(base.tests, encoding="utf-8")
            (d / base.fuzz_file).write_text(base.fuzz, encoding="utf-8")
            files += [base.test_file, base.fuzz_file]
        proc, report = run_pytest(d, files, timeout, _clean_env(fuzz_seed), runner)
        if proc is None:
            return VerifyResult(0, 0, expected, timed_out=True, note="timeout")
        tests, passed, skipped, failed = parse_junit(report)

    note = ""
    if tests != expected:
        note = f"expected {expected} test results, got {tests}"
    if skipped:
        note = (note + "; " if note else "") + f"{skipped} skipped"
    return VerifyResult(passed if not skipped else 0, tests, expected, failed=failed, note=note)
