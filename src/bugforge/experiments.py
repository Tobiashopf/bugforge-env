"""Run LLM agents over a difficulty grid, fit a difficulty model, analyze rollouts.

    python -m bugforge calibrate --provider ollama --model qwen2.5-coder:3b --seeds 8
    python -m bugforge report runs/ollama_qwen2.5-coder-3b.jsonl

Runs append one JSON line per episode and skip episodes already in the file, so
a sweep can be stopped and resumed (or left running overnight).

**Calibration.** Pass/fail outcomes are fit with a lightly L2-regularized
logistic regression on the generator's knobs (bugs, files, share of tests
shown). The fit predicts the pass rate of any configuration, which is what lets
the generator be tuned to a target difficulty distribution *for a given model*.
Observed pass rates are reported with Wilson 95% intervals next to the
predictions, so a thin cell is visibly thin.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from .agents import run_episode
from .agents.llm import Budget, BudgetExceeded, LLMAgent
from .env import BugForgeEnv
from .generator import Difficulty, custom, generate

GRID = [
    custom(1, 1, 0.6), custom(1, 1, 0.25),
    custom(3, 1, 0.6), custom(3, 2, 0.4), custom(3, 2, 0.25),
    custom(5, 2, 0.4), custom(5, 3, 0.4), custom(5, 3, 0.25),
]
FEATURES = ["n_bugs", "n_files", "visible_frac"]


# --------------------------------------------------------------------------- runs
def _key(rec: dict) -> tuple:
    return rec["model"], rec["difficulty"], rec["seed"]


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short by an interrupted run
    return out


def run_grid(client, out: Path, grid: list[Difficulty], seeds: list[int], *,
             sandbox: str | None = None, max_steps: int = 30, budget: Budget | None = None,
             log=print) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    budget = budget or Budget()
    done = {_key(r) for r in load_records(out)}
    model = getattr(client, "model", "?")
    jobs = [(d, s) for s in seeds for d in grid if (model, d.name, s) not in done]
    log(f"{len(jobs)} episodes to run ({len(done)} already in {out})")
    t_start = time.time()
    for i, (diff, seed) in enumerate(jobs, 1):
        env = BugForgeEnv(diff, max_steps=max_steps, sandbox=sandbox)
        agent = LLMAgent(client, budget=budget)
        t0 = time.time()
        try:
            task = generate(seed, diff)
            res = run_episode(env, agent, seed, task=task)
        except BudgetExceeded as exc:
            env.close()
            log(f"stopping: {exc}")
            break
        rec = {
            "provider": client.provider, "model": model, "difficulty": diff.name,
            "config": {"n_files": diff.n_files, "n_bugs": diff.n_bugs,
                       "visible_frac": diff.visible_frac,
                       "each_bug_visible": diff.each_bug_visible},
            "seed": seed, "reward": res["reward"], "solved": res["solved"],
            "steps": res["steps"], "test_runs": res["test_runs"], "end": res["end"],
            "turns": agent.turns, "input_tokens": agent.input_tokens,
            "output_tokens": agent.output_tokens, "seconds": round(time.time() - t0, 1),
            "edited": res["edited"], "bug_files": sorted({b["file"] for b in res["bugs"]}),
            "module_files": [f"{b}.py" for b in res["bases"]],
            "tool_errors": res["tool_errors"], "last_visible_pass": res["last_visible_pass"],
            "tests_before_edit": res["tests_before_edit"],
            "violations": bool(res["violations"]), "hidden_failed": len(res["failed"]),
            "sandbox": res["sandbox"], "transcript": agent.transcript_json(),
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(jobs) - i)
        log(f"[{i}/{len(jobs)}] {diff.name:<10} seed {seed:<3} reward {res['reward']:.0f}  "
            f"{rec['seconds']:>6.1f}s  tokens {budget.used:,}  eta {eta / 60:.0f} min")
    return load_records(out)


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (tiny dense systems)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            raise ValueError("singular system")
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    cols = [_solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def fit_logistic(xs: list[list[float]], ys: list[int], ridge: float = 1.0,
                 iters: int = 50) -> tuple[list[float], list[float]]:
    """IRLS for logistic regression with an L2 penalty on non-intercept terms.

    Returns (coefficients, standard errors). ``xs`` rows start with 1.0.
    The penalty keeps the fit finite when a small sweep is perfectly separable
    (for example, every easy task solved and every hard one failed).
    """
    d = len(xs[0])
    beta = [0.0] * d
    hess: list[list[float]] = [[0.0] * d for _ in range(d)]
    for _ in range(iters):
        grad = [0.0] * d
        hess = [[0.0] * d for _ in range(d)]
        for x, y in zip(xs, ys):
            p = _sigmoid(sum(b * v for b, v in zip(beta, x)))
            w = p * (1 - p)
            for i in range(d):
                grad[i] += (y - p) * x[i]
                for j in range(d):
                    hess[i][j] += w * x[i] * x[j]
        for i in range(1, d):
            grad[i] -= ridge * beta[i]
            hess[i][i] += ridge
        step = _solve(hess, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    cov = _inverse(hess)
    return beta, [math.sqrt(max(cov[i][i], 0.0)) for i in range(d)]


def features(cfg: dict) -> list[float]:
    return [1.0] + [float(cfg[f]) for f in FEATURES]


def predict(beta: list[float], cfg: dict) -> float:
    return _sigmoid(sum(b * v for b, v in zip(beta, features(cfg))))


# -------------------------------------------------------------------- analysis
FAILURE_ORDER = [
    ("budget_exhausted", "ran out of tool calls before submitting"),
    ("policy_violation", "submitted code the sandbox policy rejects"),
    ("never_ran_tests", "submitted without ever running the tests"),
    ("submitted_red", "submitted while visible tests were still failing"),
    ("untested_last_edit", "edited after the last test run, then submitted"),
    ("wrong_files", "never edited a file that contained a bug"),
    ("missed_bug_file", "fixed some buggy files but not all"),
    ("overfit_visible", "visible tests green, hidden tests red"),
]


def failure_mode(r: dict) -> str:
    edited = set(r["edited"])
    bug_files = set(r["bug_files"])
    if r["end"] != "submitted":
        return "budget_exhausted"
    if r["violations"]:
        return "policy_violation"
    if r["test_runs"] == 0:
        return "never_ran_tests"
    if r["last_visible_pass"] is False:
        return "submitted_red"
    if r["last_visible_pass"] is None:
        return "untested_last_edit"
    if not edited & bug_files:
        return "wrong_files"
    if not bug_files <= edited:
        return "missed_bug_file"
    return "overfit_visible"


def report(records: list[dict], targets=(0.8, 0.5, 0.2)) -> str:
    if not records:
        return "No episodes recorded yet.\n"
    models = sorted({r["model"] for r in records})
    lines = [f"# Calibration report: {', '.join(models)}", ""]
    n = len(records)
    solved = sum(r["solved"] for r in records)
    toks = sum(r["input_tokens"] + r["output_tokens"] for r in records)
    secs = sum(r["seconds"] for r in records)
    lines += [(f"{n} episodes, {solved} solved ({solved / n:.0%}). "
               f"{toks:,} tokens, {secs / 3600:.1f} h of wall clock."), ""]

    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cfg[r["difficulty"]].append(r)
    xs = [features(r["config"]) for r in records]
    ys = [int(r["solved"]) for r in records]
    fitted = None
    if 0 < solved < n and len(by_cfg) >= 3:
        fitted = fit_logistic(xs, ys)

    lines += ["## Pass rate by configuration", "",
              "| config | files | bugs | tests shown | n | solved | observed (95% CI) | predicted |",
              "|---|---|---|---|---|---|---|---|"]
    order = sorted(by_cfg, key=lambda k: (by_cfg[k][0]["config"]["n_bugs"],
                                          by_cfg[k][0]["config"]["n_files"],
                                          -by_cfg[k][0]["config"]["visible_frac"]))
    for name in order:
        rs = by_cfg[name]
        c = rs[0]["config"]
        k = sum(x["solved"] for x in rs)
        lo, hi = wilson(k, len(rs))
        pred = f"{predict(fitted[0], c):.0%}" if fitted else "-"
        lines.append(f"| `{name}` | {c['n_files']} | {c['n_bugs']} | {c['visible_frac']:.0%} | "
                     f"{len(rs)} | {k} | {k / len(rs):.0%} ({lo:.0%}-{hi:.0%}) | {pred} |")
    lines.append("")

    if fitted:
        beta, se = fitted
        lines += ["## Difficulty model", "",
                  ("Logistic regression of *solved* on the generator knobs "
                   "(L2 penalty 1.0 on slopes). Coefficients are log-odds per unit."), "",
                  "| term | coefficient | std. error |", "|---|---|---|"]
        for name, b, s in zip(["intercept", *FEATURES], beta, se):
            lines.append(f"| {name} | {b:+.2f} | {s:.2f} |")
        lines += ["", "**Configurations closest to target pass rates** (from the grid):", ""]
        cfgs = {name: by_cfg[name][0]["config"] for name in by_cfg}
        for t in targets:
            best = min(cfgs, key=lambda k: abs(predict(beta, cfgs[k]) - t))
            lines.append(f"- target {t:.0%}: `{best}` (predicted {predict(beta, cfgs[best]):.0%})")
        lines.append("")
    else:
        lines += [("Not enough variation to fit a difficulty model (need some solved and "
                   "some failed episodes across at least 3 configurations)."), ""]

    fails = [r for r in records if not r["solved"]]
    if fails:
        counts = Counter(failure_mode(r) for r in fails)
        lines += ["## Why episodes failed", "", "| failure mode | meaning | episodes | share |",
                  "|---|---|---|---|"]
        for key, meaning in FAILURE_ORDER:
            if counts[key]:
                lines.append(f"| {key} | {meaning} | {counts[key]} | "
                             f"{counts[key] / len(fails):.0%} |")
        lines.append("")

    def share(pred) -> str:
        return f"{sum(1 for r in records if pred(r)) / n:.0%}"

    def mean(key: str) -> float:
        return sum(r[key] for r in records) / n

    def touched_clean(r: dict) -> bool:
        clean = set(r["module_files"]) - set(r["bug_files"])
        return bool(set(r["edited"]) & clean)

    def touched_tests(r: dict) -> bool:
        return any(e.startswith("test_") for e in r["edited"])

    rows = [
        ("ran the tests before the first edit", share(lambda r: r["tests_before_edit"])),
        ("edited a test file", share(touched_tests)),
        ("edited a bug-free module", share(touched_clean)),
        ("mean tool calls per episode", f"{mean('steps'):.1f}"),
        ("mean test runs per episode", f"{mean('test_runs'):.1f}"),
        ("mean tool-call errors per episode", f"{mean('tool_errors'):.1f}"),
        ("mean tokens per episode", f"{toks / n:,.0f}"),
        ("mean seconds per episode", f"{secs / n:.0f}"),
    ]
    lines += ["## Agent behaviour", "", "| behaviour | value |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    lines.append("")
    return "\n".join(lines)
