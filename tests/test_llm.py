import io
import json
import urllib.error

import pytest

from bugforge.agents import llm, run_episode
from bugforge.agents.llm import (
    AnthropicClient,
    Budget,
    BudgetExceeded,
    LLMAgent,
    OllamaClient,
    Reply,
    ToolCall,
)
from bugforge.env import BugForgeEnv


class FakeClient:
    """Replays a script of replies; each item is a list of (tool, args) or a text string."""

    provider = "fake"
    model = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def complete(self, system, transcript, turn):
        self.seen.append([dict(t) for t in transcript])
        item = self.script.pop(0)
        if isinstance(item, str):
            return Reply(item, [], 10, 5)
        calls = [ToolCall(f"c{turn}_{i}", name, args) for i, (name, args) in enumerate(item)]
        return Reply("", calls, 100, 20)


def solving_script(env):
    """A script that fixes the task by writing the reference solution."""
    def script():
        task = env.task
        writes = [("write_file", {"path": f, "content": src}) for f, src in task.solution.items()]
        return [[("run_tests", {})], writes, [("run_tests", {})], [("submit", {})]]
    return script


def test_agent_solves_with_scripted_model():
    env = BugForgeEnv("medium")
    env.reset(1)
    client = FakeClient(solving_script(env)())
    agent = LLMAgent(client)
    res = run_episode(env, agent, 1, task=env.task)
    assert res["reward"] == 1.0
    assert agent.turns == 4 and agent.input_tokens == 400


def test_parallel_calls_are_queued_and_results_grouped():
    env = BugForgeEnv("medium")
    env.reset(1)
    client = FakeClient(solving_script(env)())
    agent = LLMAgent(client)
    run_episode(env, agent, 1, task=env.task)
    tool_turns = [t for t in agent.transcript if t["role"] == "tool"]
    assert [len(t["results"]) for t in tool_turns] == [1, 3, 1]  # 3 files written in one turn


def test_prose_answers_are_nudged_then_submitted():
    env = BugForgeEnv("easy")
    client = FakeClient(["thinking...", "still thinking", "hmm"])
    agent = LLMAgent(client, max_nudges=2)
    res = run_episode(env, agent, 0)
    assert res["end"] == "submitted" and res["reward"] == 0.0
    assert sum(1 for t in agent.transcript if t.get("text") == llm.NUDGE) == 3


def test_budget_stops_before_overspending():
    budget = Budget(max_tokens=100)
    env = BugForgeEnv("easy")
    agent = LLMAgent(FakeClient([[("list_files", {})], [("submit", {})]]), budget=budget)
    with pytest.raises(BudgetExceeded):
        run_episode(env, agent, 0)
    env.close()
    assert budget.used == 120


def _transcript():
    c1 = ToolCall("t1", "read_file", {"path": "a.py"})
    c2 = ToolCall("t2", "run_tests", {})
    return [{"role": "user", "text": "task"},
            {"role": "assistant", "text": "looking", "calls": [c1, c2]},
            {"role": "tool", "results": [(c1, "code"), (c2, "1 failed")]}]


def test_anthropic_payload_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    p = AnthropicClient().payload("sys", _transcript())
    assert p["system"] == "sys"
    assert [m["role"] for m in p["messages"]] == ["user", "assistant", "user"]
    blocks = p["messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "looking"}
    assert [b["id"] for b in blocks[1:]] == ["t1", "t2"]
    results = p["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert {t["name"] for t in p["tools"]} >= {"read_file", "submit"}
    assert "input_schema" in p["tools"][0]


def test_anthropic_parse():
    data = {"content": [{"type": "text", "text": "ok"},
                        {"type": "tool_use", "id": "x", "name": "run_tests", "input": {}}],
            "usage": {"input_tokens": 7, "output_tokens": 3}}
    r = AnthropicClient.parse(data, 1)
    assert r.text == "ok" and r.calls[0].name == "run_tests" and r.input_tokens == 7


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AnthropicClient()


def test_ollama_payload_and_parse():
    p = OllamaClient(num_ctx=8192).payload("sys", _transcript())
    roles = [m["role"] for m in p["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    assert p["messages"][2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert p["options"]["num_ctx"] == 8192 and p["stream"] is False
    data = {"message": {"content": "", "tool_calls": [
        {"function": {"name": "read_file", "arguments": '{"path": "b.py"}'}}]},
        "prompt_eval_count": 50, "eval_count": 9}
    r = OllamaClient.parse(data, 3)
    assert r.calls[0].args == {"path": "b.py"} and r.calls[0].id == "call_3_0"
    assert (r.input_tokens, r.output_tokens) == (50, 9)


def test_retries_on_rate_limit(monkeypatch):
    calls = {"n": 0}

    def flaky(url, payload, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(url, 429, "slow down", {}, io.BytesIO(b"{}"))
        return {"message": {"content": "hi"}}

    monkeypatch.setattr(llm, "_post", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    r = OllamaClient().complete("sys", [{"role": "user", "text": "x"}], 1)
    assert r.text == "hi" and calls["n"] == 3


def test_non_retryable_error_is_raised(monkeypatch):
    def bad(url, payload, headers, timeout):
        raise urllib.error.HTTPError(url, 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(llm, "_post", bad)
    with pytest.raises(RuntimeError, match="401"):
        OllamaClient().complete("sys", [{"role": "user", "text": "x"}], 1)


def test_transcript_json_is_serializable():
    env = BugForgeEnv("easy")
    env.reset(0)
    agent = LLMAgent(FakeClient(solving_script(env)()))
    run_episode(env, agent, 0, task=env.task)
    json.dumps(agent.transcript_json())


def test_ollama_over_real_http(monkeypatch):
    """End to end through urllib against a stand-in Ollama server."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    env = BugForgeEnv("easy")
    env.reset(3)
    task = env.task
    writes = [{"function": {"name": "write_file", "arguments": {"path": f, "content": s}}}
              for f, s in task.solution.items()]
    replies = [writes, [{"function": {"name": "submit", "arguments": {}}}]]
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            out = json.dumps({"message": {"content": "", "tool_calls": replies.pop(0)},
                              "prompt_eval_count": 1, "eval_count": 1}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OllamaClient(host=f"127.0.0.1:{server.server_port}")
        res = run_episode(env, LLMAgent(client), 3, task=task)
    finally:
        server.shutdown()
    assert res["reward"] == 1.0
    assert received[0]["model"] == "qwen2.5-coder:3b" and received[1]["messages"][-1]["role"] == "tool"


def test_calls_written_as_text_are_recovered():
    text = ('I will look first.\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
            '\n<tool_call>{"name": "run_tests", "arguments": "{}"}</tool_call>'
            ' and {"name": "rm", "arguments": {}} and {"x": 1}')
    calls = llm.calls_from_text(text, 2)
    assert [(c.name, c.args) for c in calls] == [("read_file", {"path": "a.py"}),
                                                 ("run_tests", {})]


def test_agent_uses_text_calls_like_qwen_does():
    env = BugForgeEnv("easy")
    env.reset(0)
    task = env.task
    script = [json.dumps({"name": "write_file", "arguments": {"path": f, "content": s}})
              for f, s in task.solution.items()] + ['{"name": "submit", "arguments": {}}']
    agent = LLMAgent(FakeClient(script))
    res = run_episode(env, agent, 0, task=task)
    assert res["reward"] == 1.0 and agent.text_calls == len(script)


class RecordingClient(FakeClient):
    """Like FakeClient, but builds the real Ollama payload each turn, as a server would see it."""

    def complete(self, system, transcript, turn):
        self.payloads = getattr(self, "payloads", [])
        self.payloads.append(OllamaClient().payload(system, transcript))
        return super().complete(system, transcript, turn)


@pytest.mark.parametrize("as_text", [False, True])
def test_history_sent_to_the_model_keeps_every_tool_call(as_text):
    """Regression: the queue once aliased the transcript, so executed calls vanished
    from the history and the model saw its own turns as empty."""
    env = BugForgeEnv("easy")
    env.reset(0)
    steps = [[("list_files", {})], [("run_tests", {})], [("submit", {})]]
    script = ([json.dumps({"name": s[0][0], "arguments": s[0][1]}) for s in steps]
              if as_text else steps)
    client = RecordingClient(script)
    agent = LLMAgent(client)
    run_episode(env, agent, 0, task=env.task)
    last = client.payloads[-1]["messages"]
    assistant = [m for m in last if m["role"] == "assistant"]
    assert [m["tool_calls"][0]["function"]["name"] for m in assistant] == ["list_files",
                                                                          "run_tests"]
    assert [m["tool_name"] for m in last if m["role"] == "tool"] == ["list_files", "run_tests"]
    for t in agent.transcript_json():
        if t["role"] == "assistant":
            assert t["calls"], "an assistant turn lost its tool calls"


def test_old_tool_outputs_are_clipped_in_what_the_model_sees():
    agent = LLMAgent(FakeClient([]), keep_recent_tool_turns=1, clip_old_outputs=10)
    agent.reset("task")
    c1, c2 = ToolCall("a", "read_file", {"path": "x.py"}), ToolCall("b", "run_tests", {})
    agent.transcript += [
        {"role": "assistant", "text": "", "calls": [c1]},
        {"role": "tool", "results": [(c1, "x" * 100)]},
        {"role": "assistant", "text": "", "calls": [c2]},
        {"role": "tool", "results": [(c2, "y" * 100)]},
    ]
    view = agent.view()
    old, recent = view[2]["results"][0][1], view[4]["results"][0][1]
    assert old.startswith("x" * 10) and "90 more characters" in old
    assert recent == "y" * 100
    assert agent.transcript[2]["results"][0][1] == "x" * 100  # full record kept


def test_timeouts_are_retried_once_then_reported(monkeypatch):
    calls = {"n": 0}

    def slow(url, payload, headers, timeout):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(llm, "_post", slow)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(llm.ModelServerError, match="timed out"):
        OllamaClient().complete("sys", [{"role": "user", "text": "x"}], 1)
    assert calls["n"] == 2
