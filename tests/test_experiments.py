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
    records = run_grid(SolveEasyClient(), out, grid, [0, 1], log=lambda *a: None)
    assert len(records) == 4
    by = {(r["difficulty"], r["seed"]): r["solved"] for r in records}
    assert by[("f1b1v60e", 0)] and not by[("f3b2v40e", 0)]
    assert records[0]["transcript"] and records[0]["input_tokens"] == 50
    again = run_grid(SolveEasyClient(), out, grid, [0, 1], log=lambda *a: None)
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
