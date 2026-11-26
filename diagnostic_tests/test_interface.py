import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

import diagnostic as d


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("I will read the file", "no_action"),
        ("{broken", "parse_failure"),
        ('{"tool":"module.py"}', "unknown_tool"),
        ("[]", "invalid_proposal"),
        ('{"name":"read_file","arguments":{"path":5}}', "invalid_arguments"),
        ('{"name":"submit","arguments":{}}', "accepted"),
    ],
)
def test_categories(raw, expected):
    assert d.decode(raw)[0] == expected


def test_native():
    assert d.decode("", [{"function": {"name": "run_tests", "arguments": {}}}])[0] == "accepted"
    assert d.decode("", [])[0] == "no_native_call"
    assert d.decode("", [{}, {}])[0] == "multiple_proposals"
    assert d.decode("", {"broken": True})[0] == "invalid_proposal"


def trusted_env(*args, **kwargs):
    return d.BugForgeEnv(*args, **{**kwargs, "sandbox": "local"})


def test_guided_probes(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "CheckedEnv", trusted_env)

    def scripted(model, ctx, mode, prompt, history, out, label, events):
        key = label.split("/")[-1]
        _, name, args = d.PROBES[key]
        return {"tool": name, "args": args}

    monkeypatch.setattr(d, "chat", scripted)
    rows = d.probes(d.SMALL, 4096, "text", d.EXPLICIT, tmp_path, "fixture", [])
    assert len(rows) == 6 and all(r["passed"] for r in rows.values())


def test_fallback_separate_from_execution_error(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "CheckedEnv", trusted_env)
    replies = iter([{"tool": "read_file", "args": {"path": "missing.py"}}, None, None, None])
    monkeypatch.setattr(d, "chat", lambda *args: next(replies))
    events = []
    r = d.pilot(d.EXPLICIT, tmp_path, "fixture", events)
    assert r["reward"] == 0 and r["steps"] == 2
    assert [e["event"] for e in events] == ["execution_error", "automatic_submit", "executed"]
