"""Grade submissions against trusted tests.

Docker mode executes candidate functions in a separate, restricted worker with
only candidate sources mounted. Trusted pytest and reward calculation remain
outside the container. Local legacy mode is for trusted development only.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from . import policy
from .base import BaseTask
from .runner import DockerRunner, LocalRunner, Result


@dataclass
class VerifyResult:
    passed: int
    total: int
    expected: int
    failed: list[str] = field(default_factory=list)
    violations: dict[str, list[str]] = field(default_factory=dict)
    timed_out: bool = False
    note: str = ""

    report_valid: bool = True

    @property
    def valid(self) -> bool:
        return (self.report_valid and not self.violations and not self.timed_out
                and self.expected > 0 and self.total == self.expected
                and 0 <= self.passed <= self.total)

    @property
    def solved(self) -> bool:
        return self.valid and self.passed == self.expected

    @property
    def pass_fraction(self) -> float:
        return self.passed / self.expected if self.valid else 0.0


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
    if report.is_symlink() or not report.is_file():
        return 0, 0, 0, []
    try:
        if report.stat().st_size > 2_000_000:
            return 0, 0, 0, []
        root = ET.parse(report).getroot()
        if root.tag not in {"testsuite", "testsuites"}:
            return 0, 0, 0, []
        suites = list(root.iter("testsuite"))
        if not suites:
            return 0, 0, 0, []
        for suite in suites:
            cases = suite.findall("testcase")
            if int(suite.get("tests", "-1")) != len(cases):
                return 0, 0, 0, []
            for kind, attr in (("failure", "failures"), ("error", "errors"),
                               ("skipped", "skipped")):
                if int(suite.get(attr, "0")) != sum(c.find(kind) is not None for c in cases):
                    return 0, 0, 0, []
    except (ET.ParseError, OSError, ValueError):
        return 0, 0, 0, []
    tests = passed = skipped = 0
    seen: set[tuple[str, str]] = set()
    failed: list[str] = []
    for case in root.iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        if not all(key) or key in seen:
            return 0, 0, 0, []
        seen.add(key)
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
           timeout: float = 60.0, runner=None, isolated: bool = False) -> VerifyResult:
    """Grade candidate module sources (keyed by file name) against the hidden tests."""
    expected = expected_count(bases)
    violations: dict[str, list[str]] = {}
    for base in bases:
        project = frozenset(Path(f).stem for f in base.sources)
        for filename in base.sources:
            src = candidates.get(filename)
            if src is None:
                violations[filename] = ["file missing from submission"]
                continue
            problems = (policy.check(src, project_modules=project) if base.dependencies
                        else policy.check(src))
            if problems:
                violations[filename] = problems
    if violations:
        return VerifyResult(0, 0, expected, violations=violations, note="policy violation")

    if isinstance(runner, DockerRunner) or isolated:
        return _verify_split(candidates, bases, expected, fuzz_seed, timeout,
                             runner or LocalRunner())

    with tempfile.TemporaryDirectory(prefix="bugforge-verify-") as tmp:
        d = Path(tmp)
        (d / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        files: list[str] = []
        for base in bases:
            _stage_sources(base, candidates, d, d)
            (d / base.test_file).write_text(base.tests, encoding="utf-8")
            (d / base.fuzz_file).write_text(base.fuzz, encoding="utf-8")
            files += [base.test_file, base.fuzz_file]
        proc, report = run_pytest(d, files, timeout, _clean_env(fuzz_seed), runner)
        if proc is None:
            return VerifyResult(0, 0, expected, timed_out=True, note="timeout")
        tests, passed, skipped, failed = parse_junit(report)

    return _finish(proc, tests, passed, skipped, failed, expected)


def _finish(proc, tests, passed, skipped, failed, expected):
    note = ""
    if tests != expected:
        note = f"expected {expected} test results, got {tests}"
    if skipped:
        note = (note + "; " if note else "") + f"{skipped} skipped"
    # pytest returns 0 for all passed and 1 for ordinary test failures.
    # Collection errors, interruptions and internal errors are invalid runs.
    valid_exit = (proc.returncode == 0 and passed == tests and not failed
                  or proc.returncode == 1 and passed < tests and bool(failed))
    valid = tests == expected and expected > 0 and not skipped and valid_exit
    if not valid_exit:
        note = (note + "; " if note else "") + f"inconsistent pytest exit: {proc.returncode}"
    return VerifyResult(passed if valid else 0, tests, expected, failed=failed,
                        note=note, report_valid=valid)


def _reference_source(source: str, modules: frozenset[str]) -> str:
    """Bind internal imports to trusted copies, never candidate dependencies."""
    if not modules:
        return source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in modules and not node.level:
            node.module = "ref_" + node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    alias.asname = alias.asname or alias.name
                    alias.name = "ref_" + alias.name
    return ast.unparse(tree) + "\n"


def _stage_sources(base, candidates, candidate_dir, reference_dir):
    modules = frozenset(Path(f).stem for f in base.sources) if base.dependencies else frozenset()
    for filename, source in base.sources.items():
        (candidate_dir / filename).write_text(candidates[filename], encoding="utf-8")
        (reference_dir / f"ref_{filename}").write_text(
            _reference_source(source, modules), encoding="utf-8")


def _proxy_source(base: BaseTask) -> str:
    """Generate public API proxies from trusted reference declarations, not candidate code."""
    lines = ["from _bf_proxy import function, proxy_class"]
    for node in ast.parse(base.source).body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            lines.append(f"{node.name} = function({base.name!r}, {node.name!r})")
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)
                       and not n.name.startswith("_")]
            lines.append(f"{node.name} = proxy_class({base.name!r}, {node.name!r}, {methods!r})")
    return "\n".join(lines) + "\n"


def _verify_split(candidates, bases, expected, seed, timeout, runner):
    """Trusted tests see only typed values; workers never receive references or reports.

    Local transport is a test/development option, not an OS security boundary.
    Docker mode is required for hostile submissions.
    """
    if not isinstance(runner, (DockerRunner, LocalRunner)):
        raise TypeError("split verification requires a supported runner")
    names = []
    with tempfile.TemporaryDirectory(prefix="bugforge-split-") as tmp:
        root = Path(tmp)
        trusted, worker = root / "trusted", root / "candidate"
        trusted.mkdir()
        worker.mkdir(mode=0o755)
        sources = Path(__file__).resolve().parent
        codec = (sources / "rpc_worker.py").read_text(encoding="utf-8")
        (worker / "_bf_worker.py").write_text(codec, encoding="utf-8")
        (trusted / "_bf_codec.py").write_text(codec, encoding="utf-8")
        (trusted / "_bf_proxy.py").write_text(
            (sources / "rpc_proxy.py").read_text(encoding="utf-8"), encoding="utf-8")
        (trusted / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        config, files = {}, []
        for base in bases:
            _stage_sources(base, candidates, worker, trusted)
            (trusted / base.module_file).write_text(_proxy_source(base), encoding="utf-8")
            (trusted / base.test_file).write_text(base.tests, encoding="utf-8")
            (trusted / base.fuzz_file).write_text(base.fuzz, encoding="utf-8")
            files.extend([base.test_file, base.fuzz_file])
            if isinstance(runner, DockerRunner):
                name = f"bugforge-rpc-{uuid.uuid4().hex[:12]}"
                names.append(name)
                config[base.name] = runner.worker_command(worker, base.name, name)
            else:
                config[base.name] = [sys.executable, "-u", str(worker / "_bf_worker.py"),
                                     base.name]
        # The container uses an unprivileged UID; keep its mount readable.
        for path in worker.iterdir():
            path.chmod(0o644)
        worker.chmod(0o755)
        (trusted / "_bf_config.json").write_text(json.dumps(config), encoding="utf-8")
        try:
            proc, report = run_pytest(trusted, files, timeout, _clean_env(seed), LocalRunner())
            if proc is None:
                return VerifyResult(0, 0, expected, timed_out=True, note="split verifier timeout")
            if (trusted / "_bf_protocol_failed").exists():
                return VerifyResult(0, 0, expected, report_valid=False,
                                    note="candidate protocol failure")
            tests, passed, skipped, failed = parse_junit(report)
            return _finish(proc, tests, passed, skipped, failed, expected)
        finally:
            for name in names:
                runner.remove_worker(name)
