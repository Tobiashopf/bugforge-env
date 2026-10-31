"""Where test commands execute: the local interpreter or a locked-down container.

Both runners take a directory and pytest arguments that are *relative to that
directory*, so the same call works on the host (``cwd=directory``) and in a
container (directory mounted at ``/work``).

The Docker runner is the one to use whenever model-written code runs:

- ``--network none``: no exfiltration, no downloads;
- ``--read-only`` root filesystem, a small ``tmpfs`` for ``/tmp``, and only the
  task directory mounted writable;
- ``--cap-drop ALL`` and ``no-new-privileges``;
- memory, CPU and process-count limits (a fork bomb hits ``--pids-limit``);
- a wall-clock timeout that kills the container, not just the client.

Choose with ``BUGFORGE_SANDBOX=docker`` (or ``sandbox="docker"`` in code) and
build the image once with ``docker build -t bugforge-sandbox docker``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

PASSTHROUGH_ENV = ("BUGFORGE_FUZZ_SEED", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                   "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING")


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: list[str], cwd: Path | None, env: dict[str, str] | None,
         timeout: float) -> Result | None:
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return Result(proc.returncode, proc.stdout, proc.stderr)


class LocalRunner:
    """Runs pytest with the current interpreter. Isolates grading, not code."""

    name = "local"
    isolates_code = False

    def pytest(self, directory: Path, args: list[str], timeout: float,
               env: dict[str, str]) -> Result | None:
        return _run([sys.executable, "-m", "pytest", *args], directory, env, timeout)


class DockerRunner:
    """Runs pytest inside a throwaway, network-less, resource-limited container."""

    name = "docker"
    isolates_code = True

    def __init__(self, image: str | None = None, memory: str = "512m", cpus: str = "1",
                 pids: int = 256, docker: str = "docker"):
        self.image = image or os.environ.get("BUGFORGE_DOCKER_IMAGE", "bugforge-sandbox")
        self.memory, self.cpus, self.pids, self.docker = memory, cpus, pids, docker

    def command(self, directory: Path, args: list[str], env: dict[str, str],
                name: str) -> list[str]:
        cmd = [self.docker, "run", "--rm", "--name", name,
               "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", str(self.pids),
               "-v", f"{Path(directory).resolve()}:/work", "-w", "/work"]
        if hasattr(os, "getuid"):  # Linux/macOS: write files as the host user
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for key in PASSTHROUGH_ENV:
            if key in env:
                cmd += ["-e", f"{key}={env[key]}"]
        return cmd + [self.image, "python", "-m", "pytest", *args]

    def pytest(self, directory: Path, args: list[str], timeout: float,
               env: dict[str, str]) -> Result | None:
        name = f"bugforge-{uuid.uuid4().hex[:12]}"
        result = _run(self.command(directory, args, env, name), None, None, timeout + 15)
        if result is None:
            _run([self.docker, "kill", name], None, None, 30)
        return result

    @staticmethod
    def available(image: str | None = None) -> bool:
        if shutil.which("docker") is None:
            return False
        img = image or os.environ.get("BUGFORGE_DOCKER_IMAGE", "bugforge-sandbox")
        probe = _run(["docker", "image", "inspect", img], None, None, 20)
        return probe is not None and probe.returncode == 0


def get_runner(sandbox: str | None = None):
    """'local' or 'docker'; defaults to $BUGFORGE_SANDBOX, then 'local'."""
    choice = (sandbox or os.environ.get("BUGFORGE_SANDBOX") or "local").lower()
    if choice == "local":
        return LocalRunner()
    if choice == "docker":
        return DockerRunner()
    raise ValueError(f"unknown sandbox '{choice}' (use 'local' or 'docker')")
