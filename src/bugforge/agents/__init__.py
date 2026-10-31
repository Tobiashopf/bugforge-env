"""Reference agents: baselines that bracket the score range, and exploits.

- ``OracleAgent`` restores the original sources. It must always score 1.0:
  that is the proof every generated task is solvable.
- ``RandomAgent`` makes random local edits. It sets the floor.
- ``exploits.py`` holds agents that try to game the grader instead of fixing
  the bugs. Every one of them must score 0.0; that is a CI test.
"""
from __future__ import annotations

import random
import re
from typing import Protocol

from ..env import BugForgeEnv


class Agent(Protocol):
    name: str

    def reset(self, obs: str) -> None: ...

    def act(self, obs: str) -> dict: ...


def module_files_from(obs: str) -> list[str]:
    """Module names as listed in the initial observation (public information)."""
    files = re.findall(r"^\s+(\S+\.py)\s*$", obs, flags=re.MULTILINE)
    return [f for f in files if not f.startswith("test_")]


class OracleAgent:
    """Privileged: writes the correct sources. Used to prove solvability."""

    name = "oracle"

    def __init__(self, env: BugForgeEnv):
        self.env = env
        self.queue: list[dict] = []

    def reset(self, obs: str) -> None:
        assert self.env.task is not None
        self.queue = [{"tool": "write_file", "args": {"path": f, "content": src}}
                      for f, src in self.env.task.solution.items()]
        self.queue += [{"tool": "run_tests", "args": {}}, {"tool": "submit", "args": {}}]

    def act(self, obs: str) -> dict:
        return self.queue.pop(0)


_SWAPS = [("<=", "<"), ("<", "<="), (">=", ">"), (">", ">="), ("==", "!="), (" + ", " - "),
          (" - ", " + "), ("1", "2"), ("0", "1"), (" and ", " or ")]


class RandomAgent:
    """Reads a module, makes a random token swap, sometimes runs tests, submits."""

    name = "random"

    def __init__(self, seed: int = 0, edits: int = 3):
        self.rng = random.Random(seed)
        self.edits = edits

    def reset(self, obs: str) -> None:
        self.modules = module_files_from(obs)
        self.pending: str | None = None
        self.done_edits = 0

    def act(self, obs: str) -> dict:
        if self.pending is not None:
            text, path = obs, self.pending
            self.pending = None
            options = [(a, b) for a, b in _SWAPS if text.count(a) >= 1]
            self.rng.shuffle(options)
            for a, b in options:
                if text.count(a) == 1:
                    self.done_edits += 1
                    return {"tool": "replace_in_file", "args": {"path": path, "old": a, "new": b}}
        if self.done_edits >= self.edits:
            return {"tool": "submit", "args": {}}
        if self.rng.random() < 0.2:
            return {"tool": "run_tests", "args": {}}
        self.pending = self.rng.choice(self.modules)
        return {"tool": "read_file", "args": {"path": self.pending}}


def run_episode(env: BugForgeEnv, agent, seed: int, task=None) -> dict:
    obs, info = env.reset(seed, task=task)
    agent.reset(obs)
    while True:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        if terminated or truncated:
            return {"agent": agent.name, "seed": seed, "reward": reward, **info}
