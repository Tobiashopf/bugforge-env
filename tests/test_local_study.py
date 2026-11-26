import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bugforge import local_study as local
from bugforge.agents.llm import ToolCall, calls_from_text


def test_text_tool_roundtrip():
    client = local.TextClient("qwen2.5-coder:1.5b")
    call = ToolCall("id", "read_file", {"path": "invoice.py"})
    transcript = [
        {"role": "assistant", "calls": [call], "text": ""},
        {"role": "tool", "results": [(call, "contents")]},
    ]
    payload = client.payload("fix bugs", transcript)
    assert "tools" not in payload
    assert calls_from_text(payload["messages"][1]["content"], 2)[0].args == call.args
    assert payload["messages"][2] == {"role": "user", "content": "read_file result:\ncontents"}


def test_writer_and_freeze(tmp_path):
    with local.writer(tmp_path), pytest.raises(FileExistsError), local.writer(tmp_path):
        pass
    path = tmp_path / "protocol.json"
    local.frozen(path, {"model": 1})
    local.frozen(path, {"model": 1})
    with pytest.raises(ValueError):
        local.frozen(path, {"model": 2})


def test_bandit_learns_and_freezes_on_holdout(monkeypatch, tmp_path):
    from io import BytesIO

    monkeypatch.setattr(
        local.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="sha256:" + "a" * 64, returncode=0),
    )
    monkeypatch.setattr(
        local,
        "build_opener",
        lambda *a: SimpleNamespace(
            open=lambda *a, **kw: BytesIO(
                b'{"models":[{"name":"qwen2.5-coder:1.5b","digest":"test"}]}'
            )
        ),
    )
    monkeypatch.setenv("BUGFORGE_DOCKER_IMAGE", "test")
    monkeypatch.setenv("NO_PROXY", "")
    calls = []

    def episode(client, path, family, seed, plan, model, prompt, budget):
        arm = local.PROMPTS.index(prompt)
        calls.append((seed, arm))
        return {"reward": arm, "task_sha256": str(seed)}

    monkeypatch.setattr(local, "episode", episode)
    args = SimpleNamespace(
        plan=Path(__file__).resolve().parents[1] / "study_plan.json",
        stage="rl",
        output=tmp_path,
        transport="text",
        max_tokens=1000,
    )
    local.run(args)
    policy = json.loads((tmp_path / "rl/policy.json").read_text())
    assert [s for s, _ in calls[:8]] == list(range(100, 108))
    assert policy["arm"] == 1
    assert policy["paired_delta"] == [1] * 4
    assert calls[8:] == [(s, a) for s in range(200, 204) for a in (0, 1)]


def test_measured_client_and_report(tmp_path):
    from bugforge.local_report import summarize

    client = local.TextClient("qwen2.5-coder:1.5b")
    client.parse({"message": {"content": "done"}, "eval_count": 5, "eval_duration": 250000000}, 1)
    assert "No episode measurements" in summarize(tmp_path)
    folder = tmp_path / "metrics"
    folder.mkdir()
    sample = {
        "ollama_requests": client.requests,
        "family": "checkout",
        "seed": 0,
        "series": "small",
        "elapsed_seconds": 1.0,
        "min_available_bytes": 2**30,
        "completed": True,
        "text_calls": 1,
        "tool_errors": 0,
    }
    (folder / "one.json").write_text(json.dumps(sample))
    assert "| yes | 1.00 | 20.00 | 1.00 | — | 1 | 0 |" in summarize(tmp_path)
