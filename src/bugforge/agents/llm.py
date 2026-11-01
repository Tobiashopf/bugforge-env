"""A tool-calling LLM agent, with a free local backend and a paid one.

- ``OllamaClient``: any model served by Ollama on your machine (free). Uses the
  native ``/api/chat`` endpoint because it lets us set the context window.
- ``AnthropicClient``: the Messages API with native tool use. Requires
  ``ANTHROPIC_API_KEY`` in the environment; never pass keys on the command line.

The agent keeps a provider-neutral transcript; each client converts it to its
own wire format. That keeps the agent logic (one tool call per env step, a
queue when the model asks for several at once, a nudge when it answers in
prose) independent of the provider, and makes it testable without a network.

Every request is checked against a shared ``Budget`` first, so a sweep stops
cleanly at a token ceiling instead of overspending.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..env import TOOLS

SYSTEM_PROMPT = """\
You are fixing bugs in a small Python project. Work only through the tools.

A good approach: run the tests, read the failing test and the module it covers,
find the faulty line, fix it with a minimal edit, and run the tests again.
Some modules may be bug-free, and the visible tests are only a sample of the
grading tests, so reason about the intended behaviour (docstrings help), not
just the visible failures. Do not edit test files; they are not graded.
When every bug is fixed, call submit."""

NUDGE = "Please continue by calling one of the tools. Call submit when you are done."


class BudgetExceeded(RuntimeError):
    pass


class ModelServerError(RuntimeError):
    """The model API failed (unreachable, timed out, or an HTTP error)."""


@dataclass
class Budget:
    """Token ceiling shared by every agent in a run (0 = unlimited)."""

    max_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def used(self) -> int:
        return self.input_tokens + self.output_tokens

    def check(self) -> None:
        if self.max_tokens and self.used >= self.max_tokens:
            raise BudgetExceeded(f"token budget of {self.max_tokens} reached ({self.used} used)")

    def add(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0


# A neutral transcript entry is one of:
#   {"role": "user", "text": str}
#   {"role": "assistant", "text": str, "calls": [ToolCall, ...]}
#   {"role": "tool", "results": [(ToolCall, str), ...]}


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _with_retries(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (408, 429, 500, 502, 503, 504, 529)
            if not retryable or i == attempts - 1:
                body = exc.read().decode("utf-8", "replace")[:500]
                raise ModelServerError(f"HTTP {exc.code}: {body}") from exc
        except TimeoutError as exc:
            # A request that timed out will usually time out again; one retry only.
            if i >= min(1, attempts - 1):
                raise ModelServerError(f"model server timed out: {exc}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) and i >= min(1, attempts - 1):
                raise ModelServerError(f"model server timed out: {exc}") from exc
            if i == attempts - 1:
                raise ModelServerError(f"cannot reach the model server: {exc}") from exc
        time.sleep(2 ** i)
    raise AssertionError("unreachable")


TOOL_NAMES = {t["name"] for t in TOOLS}


def calls_from_text(text: str, turn: int) -> list[ToolCall]:
    """Recover tool calls a model wrote as JSON in plain text.

    Small open models often know which tool they want but print
    ``{"name": "read_file", "arguments": {...}}`` (sometimes inside a code fence
    or ``<tool_call>`` tags) instead of emitting a structured call. Only objects
    naming a real tool are accepted, so ordinary JSON in prose is ignored.
    """
    decoder = json.JSONDecoder()
    calls: list[ToolCall] = []
    i = 0
    while (start := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        i = end
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = fn.get("name") or fn.get("tool")
        args = fn.get("arguments", fn.get("parameters", fn.get("args", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if name in TOOL_NAMES and isinstance(args, dict):
            calls.append(ToolCall(f"text_{turn}_{len(calls)}", name, args))
    return calls


class OllamaClient:
    """Local models via Ollama (free). Start it with Docker, see README."""

    provider = "ollama"

    def __init__(self, model: str = "qwen2.5-coder:3b", host: str | None = None,
                 num_ctx: int = 16384, temperature: float = 0.0, timeout: float = 1800):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx, self.temperature, self.timeout = num_ctx, temperature, timeout

    def payload(self, system: str, transcript: list[dict]) -> dict:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for t in transcript:
            if t["role"] == "user":
                messages.append({"role": "user", "content": t["text"]})
            elif t["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": t["text"]}
                if t["calls"]:
                    msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}}
                                         for c in t["calls"]]
                messages.append(msg)
            else:
                for call, content in t["results"]:
                    messages.append({"role": "tool", "content": content, "tool_name": call.name})
        tools = [{"type": "function", "function": {"name": t["name"],
                                                   "description": t["description"],
                                                   "parameters": t["parameters"]}}
                 for t in TOOLS]
        return {"model": self.model, "messages": messages, "tools": tools, "stream": False,
                "options": {"num_ctx": self.num_ctx, "temperature": self.temperature}}

    @staticmethod
    def parse(data: dict, turn: int) -> Reply:
        msg = data.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(f"call_{turn}_{i}", fn.get("name", ""), args))
        return Reply(msg.get("content") or "", calls,
                     data.get("prompt_eval_count", 0) or 0, data.get("eval_count", 0) or 0)

    def complete(self, system: str, transcript: list[dict], turn: int) -> Reply:
        payload = self.payload(system, transcript)
        data = _with_retries(lambda: _post(f"{self.host}/api/chat", payload, {}, self.timeout))
        return self.parse(data, turn)


class AnthropicClient:
    """Claude via the Messages API. Needs ANTHROPIC_API_KEY; costs money."""

    provider = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-haiku-5-5", max_tokens: int = 2048,
                 temperature: float = 0.0, timeout: float = 120):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("set ANTHROPIC_API_KEY in the environment first")
        self._key = key
        self.model, self.max_tokens = model, max_tokens
        self.temperature, self.timeout = temperature, timeout

    def payload(self, system: str, transcript: list[dict]) -> dict:
        messages: list[dict[str, Any]] = []
        for t in transcript:
            if t["role"] == "user":
                messages.append({"role": "user", "content": t["text"]})
            elif t["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if t["text"]:
                    blocks.append({"type": "text", "text": t["text"]})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                           for c in t["calls"]]
                messages.append({"role": "assistant", "content": blocks or t["text"]})
            else:
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call.id, "content": content}
                    for call, content in t["results"]]})
        tools = [{"name": t["name"], "description": t["description"],
                  "input_schema": t["parameters"]} for t in TOOLS]
        return {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "temperature": self.temperature, "messages": messages, "tools": tools}

    @staticmethod
    def parse(data: dict, turn: int) -> Reply:
        text, calls = [], []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        usage = data.get("usage") or {}
        return Reply("\n".join(text), calls, usage.get("input_tokens", 0),
                     usage.get("output_tokens", 0))

    def complete(self, system: str, transcript: list[dict], turn: int) -> Reply:
        payload = self.payload(system, transcript)
        headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01"}
        data = _with_retries(lambda: _post(self.url, payload, headers, self.timeout))
        return self.parse(data, turn)


def make_client(provider: str, model: str | None = None, timeout: float | None = None):
    if provider == "ollama":
        return OllamaClient(model or "qwen2.5-coder:3b", timeout=timeout or 1800)
    if provider == "anthropic":
        return AnthropicClient(model or "claude-haiku-5-5", timeout=timeout or 120)
    raise ValueError(f"unknown provider '{provider}' (use 'ollama' or 'anthropic')")


@dataclass
class LLMAgent:
    """Drives one model through an episode, one tool call per env step."""

    client: Any
    budget: Budget = field(default_factory=Budget)
    system: str = SYSTEM_PROMPT
    max_nudges: int = 2
    parse_text_calls: bool = True
    keep_recent_tool_turns: int = 2   # tool outputs shown in full
    clip_old_outputs: int = 600       # older outputs are cut to this many characters

    def __post_init__(self):
        self.name = f"llm:{self.client.provider}:{getattr(self.client, 'model', '?')}"
        self.transcript: list[dict] = []

    def reset(self, obs: str) -> None:
        self.transcript = [{"role": "user", "text": obs}]
        self.queue: list[ToolCall] = []
        self.pending: list[tuple[ToolCall, str]] = []
        self.last_call: ToolCall | None = None
        self.turns = 0
        self.input_tokens = self.output_tokens = 0
        self.text_calls = 0

    def act(self, obs: str) -> dict:
        if self.last_call is not None:
            self.pending.append((self.last_call, obs))
            self.last_call = None
        if not self.queue:
            if self.pending:
                self.transcript.append({"role": "tool", "results": self.pending})
                self.pending = []
            self.queue = self._ask()
        call = self.queue.pop(0)
        self.last_call = call
        return {"tool": call.name, "args": call.args}

    def _ask(self) -> list[ToolCall]:
        for _ in range(self.max_nudges + 1):
            self.budget.check()
            self.turns += 1
            reply = self.client.complete(self.system, self.view(), self.turns)
            self.budget.add(reply.input_tokens, reply.output_tokens)
            self.input_tokens += reply.input_tokens
            self.output_tokens += reply.output_tokens
            calls = list(reply.calls)
            text = reply.text
            if not calls and self.parse_text_calls:
                calls = calls_from_text(reply.text, self.turns)
                if calls:
                    self.text_calls += len(calls)
                    text = ""  # the JSON now lives in the structured call
            self.transcript.append({"role": "assistant", "text": text, "calls": calls})
            if calls:
                return list(calls)  # a copy: act() pops from the queue, not the transcript
            self.transcript.append({"role": "user", "text": NUDGE})
        give_up = ToolCall(f"auto_{self.turns}", "submit", {})
        return [give_up]

    def view(self) -> list[dict]:
        """What the model is sent: the transcript with old tool outputs clipped.

        Files read ten turns ago are re-sent on every request otherwise, and on a
        CPU the prompt length dominates the time per call. The last
        ``keep_recent_tool_turns`` tool turns stay whole; the model can re-read a
        file if it needs it again. The full outputs remain in ``transcript``.
        """
        tool_idx = [i for i, t in enumerate(self.transcript) if t["role"] == "tool"]
        old = set(tool_idx[:-self.keep_recent_tool_turns] if self.keep_recent_tool_turns
                  else tool_idx)
        out = []
        for i, t in enumerate(self.transcript):
            if i in old:
                lim = self.clip_old_outputs
                out.append({"role": "tool", "results": [
                    (c, o if len(o) <= lim else
                     o[:lim] + f"\n... [{len(o) - lim} more characters; read again if needed]")
                    for c, o in t["results"]]})
            else:
                out.append(t)
        return out

    def transcript_json(self, max_chars: int = 4000) -> list[dict]:
        """The transcript in a JSON-friendly form, long tool outputs clipped."""
        out = []
        for t in self.transcript:
            if t["role"] == "tool":
                out.append({"role": "tool", "results": [
                    {"tool": c.name, "args": c.args, "output": o[:max_chars]}
                    for c, o in t["results"]]})
            elif t["role"] == "assistant":
                out.append({"role": "assistant", "text": t["text"],
                            "calls": [{"tool": c.name, "args": c.args} for c in t["calls"]]})
            else:
                out.append(dict(t))
        return out
