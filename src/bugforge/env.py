"""BugForgeEnv: a Gymnasium-style, text-in/text-out environment for LLM agents.

    env = BugForgeEnv(difficulty="medium")
    obs, info = env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(
        {"tool": "read_file", "args": {"path": "intervals.py"}})

Actions are tool calls ``{"tool": name, "args": {...}}``; ``tool_schemas()``
returns them in the JSON-schema format that function-calling APIs accept.
Observations are strings. The reward is paid once, when the episode ends:
on ``submit``, or when the step budget runs out (the workspace is graded as
it stands). ``info`` carries grading details and ground truth for the harness;
never show it to the agent.
"""
from __future__ import annotations

from typing import Any

from .generator import Task, generate
from .sandbox import ToolError, Workspace
from .verifier import VerifyResult, verify

TOOLS: list[dict[str, Any]] = [
    {"name": "list_files", "description": "List the files in the workspace.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "read_file", "description": "Read a file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}},
    {"name": "write_file", "description": "Create or overwrite a file with the given content.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "replace_in_file",
     "description": "Replace one exact occurrence of `old` with `new` in a file.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                   "new": {"type": "string"}},
                    "required": ["path", "old", "new"]}},
    {"name": "run_tests", "description": "Run the visible tests and show the output.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "submit", "description": "Finish the episode and submit the modules for grading.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters truncated] ...\n{text[-half:]}"


class BugForgeEnv:
    def __init__(self, difficulty: str = "easy", max_steps: int = 30,
                 max_obs_chars: int = 6000, partial_credit: bool = False):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.max_obs_chars = max_obs_chars
        self.partial_credit = partial_credit
        self.task: Task | None = None
        self.ws: Workspace | None = None
        self.steps = 0
        self.done = True
        self.test_runs = 0

    # -- gym-style API -------------------------------------------------------
    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return [dict(t) for t in TOOLS]

    def reset(self, seed: int, task: Task | None = None) -> tuple[str, dict]:
        self.close()
        self.task = task or generate(seed, self.difficulty)
        self.ws = Workspace(self.task.workspace).__enter__()
        self.steps, self.test_runs, self.done = 0, 0, False
        obs = (self.task.workspace["README.md"] + "\nFiles:\n"
               + "\n".join(f"  {f}" for f in self.ws.list_files())
               + f"\n\nYou have {self.max_steps} tool calls.")
        return obs, {"seed": seed, "difficulty": self.task.difficulty,
                     "bases": list(self.task.bases)}

    def step(self, action: dict) -> tuple[str, float, bool, bool, dict]:
        if self.done or self.ws is None or self.task is None:
            raise RuntimeError("call reset() before step()")
        self.steps += 1
        tool = action.get("tool", "")
        args = action.get("args") or {}
        if tool == "submit":
            return self._finish("submitted", truncated=False)
        try:
            obs = self._run_tool(tool, args)
        except ToolError as exc:
            obs = f"error: {exc}"
        except (KeyError, TypeError) as exc:
            obs = f"error: bad or missing argument for {tool}: {exc}"
        if self.steps >= self.max_steps:
            final_obs, reward, term, trunc, info = self._finish("step budget exhausted",
                                                                truncated=True)
            obs = _truncate(obs, self.max_obs_chars) + "\n\n" + final_obs
            return obs, reward, term, trunc, info
        return _truncate(obs, self.max_obs_chars), 0.0, False, False, {"steps": self.steps}

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None
        self.done = True

    # -- internals -------------------------------------------------------------
    def _run_tool(self, tool: str, args: dict) -> str:
        ws = self.ws
        assert ws is not None
        if tool == "list_files":
            return "\n".join(ws.list_files())
        if tool == "read_file":
            return ws.read_file(args["path"])
        if tool == "write_file":
            ws.write_file(args["path"], args["content"])
            return f"wrote {args['path']}"
        if tool == "replace_in_file":
            ws.replace(args["path"], args["old"], args["new"])
            return f"edited {args['path']}"
        if tool == "run_tests":
            self.test_runs += 1
            run = ws.run_tests()
            head = "all visible tests passed" if run.all_passed else \
                f"{run.failed} failed, {run.passed} passed"
            return f"{head}\n\n{run.output}"
        raise ToolError(f"unknown tool '{tool}'")

    def grade(self) -> VerifyResult:
        assert self.ws is not None and self.task is not None
        submission = self.ws.snapshot(self.task.module_files)
        return verify(submission, self.task.load_bases(), fuzz_seed=self.task.seed)

    def _finish(self, why: str, truncated: bool) -> tuple[str, float, bool, bool, dict]:
        result = self.grade()
        reward = 1.0 if result.solved else (result.pass_fraction if self.partial_credit else 0.0)
        self.done = True
        info = {"steps": self.steps, "test_runs": self.test_runs, "solved": result.solved,
                "pass_fraction": result.pass_fraction, "violations": result.violations,
                "failed": result.failed, "note": result.note,
                "bugs": [b.__dict__ for b in self.task.bugs], "end": why}
        obs = f"Episode over ({why}). Reward: {reward:.2f}"
        self.close()
        return obs, reward, not truncated, truncated, info
