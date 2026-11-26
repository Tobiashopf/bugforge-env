import math
import random

from bugforge import experiments
from bugforge.agents.llm import Reply, ToolCall
from bugforge.experiments import failure_mode, fit_logistic, report, run_grid, wilson
from bugforge.generator import custom


def test_wilson_interval():
    lo, hi = wilson(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_logistic_fit_recovers_known_coefficients():
    rng = random.Random(0)
    true = [2.0, -1.5, 0.8]
    xs, ys = [], []
    for _ in range(3000):
        x = [1.0, rng.uniform(0, 3), rng.uniform(-1, 1)]
        p = 1 / (1 + math.exp(-sum(b * v for b, v in zip(true, x))))
        xs.append(x)
        ys.append(int(rng.random() < p))
    beta, se = fit_logistic(xs, ys, ridge=1e-6)
    for b, t, s in zip(beta, true, se):
        assert abs(b - t) < 4 * s


def test_logistic_fit_stays_finite_when_separable():
    xs = [[1.0, float(v)] for v in range(10)]
    ys = [1] * 5 + [0] * 5
    beta, _ = fit_logistic(xs, ys)
    assert all(math.isfinite(b) for b in beta) and beta[1] < 0


def rec(**kw):
    base = {"end": "submitted", "violations": False, "test_runs": 2, "last_visible_pass": True,
            "edited": ["a.py"], "bug_files": ["a.py"], "module_files": ["a.py", "b.py"]}
    base.update(kw)
    return base


def test_failure_modes():
    assert failure_mode(rec(end="step budget exhausted")) == "budget_exhausted"
    assert failure_mode(rec(violations=True)) == "policy_violation"
    assert failure_mode(rec(violations=["import of 'os' not allowed"])) == "policy_violation"
    assert failure_mode(rec(violations=["syntax error: bad indent (line 3)"])) == "broke_syntax"
    assert failure_mode(rec(test_runs=0)) == "never_ran_tests"
    assert failure_mode(rec(last_visible_pass=False)) == "submitted_red"
    assert failure_mode(rec(last_visible_pass=None)) == "untested_last_edit"
    assert failure_mode(rec(edited=["b.py"])) == "wrong_files"
    assert failure_mode(rec(bug_files=["a.py", "b.py"])) == "missed_bug_file"
    assert failure_mode(rec()) == "overfit_visible"


class SolveEasyClient:
    """Solves 1-bug tasks by writing the solution; gives up on anything harder."""

    provider = "fake"
    model = "half-good"

    def __init__(self):
        self.env_task = None

    def complete(self, system, transcript, turn):
        task = experiments._current_task
        if len(task.bugs) == 1 and turn == 1:
            calls = [ToolCall(f"w{i}", "write_file", {"path": f, "content": s})
                     for i, (f, s) in enumerate(task.solution.items())]
            calls += [ToolCall("r", "run_tests", {}), ToolCall("s", "submit", {})]
            return Reply("", calls, 50, 10)
        return Reply("", [ToolCall("s", "submit", {})], 50, 10)


def test_run_grid_writes_resumable_jsonl(tmp_path, monkeypatch):
    real_generate = experiments.generate

    def tracking_generate(seed, diff):
        task = real_generate(seed, diff)
        experiments._current_task = task
        return task

    monkeypatch.setattr(experiments, "generate", tracking_generate)
    grid = [custom(1, 1, 0.6), custom(3, 2, 0.4)]
    out = tmp_path / "runs.jsonl"
    records = run_grid(SolveEasyClient(), out, grid, [0, 1], sandbox="local", log=lambda *a: None)
    assert len(records) == 4
    by = {(r["difficulty"], r["seed"]): r["solved"] for r in records}
    assert by[("f1b1v60e", 0)] and not by[("f3b2v40e", 0)]
    assert records[0]["transcript"] and records[0]["input_tokens"] == 50
    again = run_grid(SolveEasyClient(), out, grid, [0, 1], sandbox="local", log=lambda *a: None)
    assert len(again) == 4  # nothing re-run

    text = report(again)
    assert "Pass rate by configuration" in text and "Why episodes failed" in text


def test_report_handles_no_variation():
    rs = [dict(rec(last_visible_pass=False), model="m", difficulty="d", seed=i, solved=False,
               config={"n_files": 1, "n_bugs": 1, "visible_frac": 0.6}, steps=3,
               tool_errors=0, tests_before_edit=True, input_tokens=1, output_tokens=1,
               seconds=1.0) for i in range(3)]
    text = report(rs)
    assert "Not enough variation" in text and "submitted_red" in text


def test_report_empty():
    assert "No episodes" in report([])


class BrokenClient:
    provider = "fake"
    model = "broken"

    def __init__(self, fail_first: int):
        self.fail_first = fail_first
        self.n = 0

    def complete(self, system, transcript, turn):
        from bugforge.agents.llm import ModelServerError
        self.n += 1
        if self.n <= self.fail_first:
            raise ModelServerError("model server timed out")
        return Reply("", [ToolCall("s", "submit", {})], 1, 1)


def test_run_grid_skips_server_errors_and_retries_on_rerun(tmp_path):
    out = tmp_path / "r.jsonl"
    grid = [custom(1, 1, 0.6)]
    log = []
    records = run_grid(BrokenClient(fail_first=1), out, grid, [0, 1], sandbox="local", log=log.append)
    assert len(records) == 1 and any("skipped" in m for m in log)
    records = run_grid(BrokenClient(fail_first=0), out, grid, [0, 1], sandbox="local", log=log.append)
    assert len(records) == 2


def test_run_grid_stops_when_the_server_is_down(tmp_path):
    log = []
    records = run_grid(BrokenClient(fail_first=99), tmp_path / "r.jsonl", [custom(1, 1, 0.6)],
                       [0, 1, 2, 3, 4], sandbox="local", log=log.append)
    assert records == [] and any("3 episodes in a row" in m for m in log)


def test_loop_detection():
    err = {"tool": "replace_in_file", "args": {"old": "a"}, "output": "error: not found"}
    looping = {"transcript": [{"role": "tool", "results": [err]}] * 3}
    assert experiments.stuck_in_loop(looping)
    assert not experiments.stuck_in_loop({"transcript": [{"role": "tool", "results": [err]}] * 2})


def test_gave_up_without_editing_mode():
    assert failure_mode(rec(edited=[])) == "gave_up_without_editing"


def test_enrich_restores_bug_details(tmp_path, monkeypatch):
    real_generate = experiments.generate

    def tracking_generate(seed, diff):
        task = real_generate(seed, diff)
        experiments._current_task = task
        return task

    monkeypatch.setattr(experiments, "generate", tracking_generate)
    grid = [custom(1, 1, 0.6), custom(3, 2, 0.4)]
    records = run_grid(SolveEasyClient(), tmp_path / "r.jsonl", grid, [0], sandbox="local", log=lambda *a: None)
    truth = {(r["difficulty"], r["seed"]): r["bugs"] for r in records}
    for r in records:
        del r["bugs"]
    assert experiments.enrich(records, grid) == len(records)
    for r in records:
        assert r["bugs"] == truth[(r["difficulty"], r["seed"])] and r["bugs_reconstructed"]
    assert "Pass rate by injected bug type" in report(records)


def test_default_calibration_fails_closed_without_docker(tmp_path, monkeypatch):
    import pytest

    from bugforge.runner import DockerRunner

    monkeypatch.setattr(DockerRunner, "available", staticmethod(lambda: False))
    out = tmp_path / "runs.jsonl"
    with pytest.raises(RuntimeError, match="Docker sandbox unavailable"):
        run_grid(BrokenClient(0), out, [custom(1, 1, 0.6)], [0])
    assert not out.exists()


def test_provenance_differentiates_settings():
    from bugforge.agents.llm import LLMAgent
    from bugforge.provenance import config, fingerprint, manifest

    client = BrokenClient(0)
    a = manifest(client, LLMAgent(client), 'local', 30)
    b = manifest(client, LLMAgent(client), 'local', 31)
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a) == fingerprint(manifest(client, LLMAgent(client), 'local', 30))
    assert fingerprint(a) != fingerprint(manifest(client, LLMAgent(client), 'local', 30, 'v2'))
    assert fingerprint(config(custom(1, 1, .601))) != fingerprint(config(custom(1, 1, .602)))


def test_resume_refuses_incompatible_and_legacy(tmp_path):
    import json

    import pytest

    from bugforge.agents.llm import LLMAgent
    from bugforge.provenance import config, fingerprint, manifest

    out = tmp_path / 'runs.jsonl'
    client = BrokenClient(0)
    diff = custom(1, 1, .6)
    m = manifest(client, LLMAgent(client), 'local', 30)
    row = {"provider": client.provider, "model": client.model, "sandbox": 'local',
               "experiment_id": fingerprint(m), "manifest": m, "difficulty": diff.name,
               "config": config(diff), "seed": 0}
    out.write_text(json.dumps(row))
    before = out.read_bytes()
    assert run_grid(client, out, [diff], [0], sandbox='local') == [row]
    with pytest.raises(ValueError, match='configuration changed'):
        run_grid(client, out, [diff], [0], sandbox='local', max_steps=31)
    assert out.read_bytes() == before
    out.write_text(json.dumps({'model': 'old'}))
    with pytest.raises(ValueError, match='legacy'):
        run_grid(client, out, [diff], [0], sandbox='local')
    assert out.read_text() == json.dumps({'model': 'old'})
    out.write_text('{broken')
    with pytest.raises(ValueError, match='invalid JSONL'):
        run_grid(client, out, [diff], [0], sandbox='local')
    assert out.read_text() == '{broken'


def test_reports_separate_and_deduplicate():
    import pytest

    from bugforge.agents.llm import LLMAgent
    from bugforge.provenance import config, fingerprint, manifest

    client = BrokenClient(0)
    diff = custom(1, 1, .6)
    def row(steps):
        m = manifest(client, LLMAgent(client), 'local', steps)
        return dict(rec(last_visible_pass=False), provider=client.provider, model=client.model,
                    sandbox='local', experiment_id=fingerprint(m), manifest=m,
                    difficulty=diff.name, config=config(diff), seed=0, solved=False,
                    steps=3, tool_errors=0, tests_before_edit=True, input_tokens=1,
                    output_tokens=1, seconds=1.0)
    a, b = row(30), row(31)
    text = report([a, a, b])
    assert text.count('1 episodes, 0 solved') == 2
    assert text.count('## Difficulty') == 0
    with pytest.raises(ValueError, match='conflicting duplicate'):
        report([a, dict(a, solved=True)])
    with pytest.raises(ValueError, match='invalid experiment fingerprint'):
        report([dict(a, experiment_id='bad')])
    legacy = {k: v for k, v in a.items() if k not in ('manifest', 'experiment_id')}
    assert 'provenance incomplete' in report([a, legacy])


def test_resume_append_and_config_collision(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from bugforge.provenance import fingerprint

    monkeypatch.setattr(experiments, 'generate', lambda seed, diff: SimpleNamespace(
        workspace={'x.py': str(seed)}, solution={'x.py': 'correct'}))
    def episode(env, agent, seed, task):
        agent.turns = agent.text_calls = agent.input_tokens = agent.output_tokens = 0
        return {"reward": 0, "solved": False, "steps": 1, "test_runs": 0, "end": 'submitted',
                    "edited": [], "bugs": [], "bases": [], "tool_errors": 0, "last_visible_pass": None,
                    "tests_before_edit": False, "violations": {}, "failed": [], "sandbox": 'local'}
    monkeypatch.setattr(experiments, 'run_episode', episode)
    out = tmp_path / 'r.jsonl'
    a, b = custom(1, 1, .601), custom(1, 1, .602)
    assert a.name == b.name
    rows = run_grid(BrokenClient(0), out, [a], [0, 0], sandbox='local')
    assert len(rows) == 1
    out.write_bytes(out.read_bytes().rstrip(b'\n'))
    rows = run_grid(BrokenClient(0), out, [a, b], [0, 1], sandbox='local')
    assert len(rows) == 4
    assert len({(r['seed'], fingerprint(r['config'])) for r in rows}) == 4
    assert len({r['experiment_id'] for r in rows}) == 1
    assert len(run_grid(BrokenClient(0), out, [a, b], [0, 1], sandbox='local')) == 4
    assert report(rows).count('`' + a.name + '@') == 2
