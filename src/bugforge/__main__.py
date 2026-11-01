"""Command line: inspect tasks and run the reference agents.

    python -m bugforge show --difficulty medium --seed 3
    python -m bugforge baselines --seeds 20
    python -m bugforge calibrate --provider ollama --model qwen2.5-coder:3b --seeds 8
    python -m bugforge report runs/ollama_qwen2.5-coder-3b.jsonl
"""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

from .agents import OracleAgent, RandomAgent, exploits, run_episode
from .env import BugForgeEnv
from .generator import DIFFICULTIES, generate


def cmd_show(args) -> None:
    task = generate(args.seed, args.difficulty)
    print(f"seed {task.seed}  difficulty {task.difficulty}  modules {', '.join(task.bases)}\n")
    print("injected bugs (hidden from the agent):")
    for b in task.bugs:
        print(f"  {b.file}:{b.line:<4} {b.operator:<8} {b.description}")
    print("\nvisible tests:")
    for name, tests in task.visible_tests.items():
        print(f"  {name}: {len(tests)} shown")
    if args.files:
        for name, content in task.workspace.items():
            print(f"\n===== {name} =====\n{content}")


def cmd_baselines(args) -> None:
    agents = ["oracle", "random"] + [e.name for e in exploits.ALL]
    rows: dict[tuple[str, str], list[float]] = defaultdict(list)
    t0 = time.time()
    for diff in args.difficulty or list(DIFFICULTIES):
        env = BugForgeEnv(diff)
        for seed in range(args.seeds):
            task = generate(seed, diff)
            for name in agents:
                if name == "oracle":
                    agent = OracleAgent(env)
                elif name == "random":
                    agent = RandomAgent(seed)
                else:
                    agent = next(e for e in exploits.ALL if e.name == name)()
                rows[(name, diff)].append(run_episode(env, agent, seed, task=task)["reward"])
    diffs = args.difficulty or list(DIFFICULTIES)
    print(f"mean reward over {args.seeds} seeds per difficulty ({time.time() - t0:.0f}s)\n")
    print(f"| agent | {' | '.join(diffs)} |")
    print(f"|---|{'---|' * len(diffs)}")
    for name in agents:
        cells = " | ".join(f"{sum(rows[(name, d)]) / len(rows[(name, d)]):.2f}" for d in diffs)
        print(f"| {name} | {cells} |")


def _default_out(provider: str, model: str) -> Path:
    return Path("runs") / f"{provider}_{re.sub(r'[^A-Za-z0-9.]+', '-', model)}.jsonl"


def cmd_calibrate(args) -> None:
    from . import experiments
    from .agents.llm import Budget, make_client
    client = make_client(args.provider, args.model, args.request_timeout)
    grid = ([DIFFICULTIES[n] for n in DIFFICULTIES] if args.named else experiments.GRID)
    out = Path(args.out) if args.out else _default_out(client.provider, client.model)
    records = experiments.run_grid(client, out, grid, list(range(args.seeds)),
                                   sandbox=args.sandbox, max_steps=args.max_steps,
                                   budget=Budget(args.max_tokens))
    report_path = out.with_name(out.stem + "_report.md")
    report_path.write_text(experiments.report(records), encoding="utf-8")
    print(f"\nreport written to {report_path}")


def cmd_enrich(args) -> None:
    import json

    from . import experiments
    path = Path(args.file)
    records = experiments.load_records(path)
    grid = experiments.GRID + list(DIFFICULTIES.values())
    n = experiments.enrich(records, grid)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"added bug details to {n} of {len(records)} records")


def cmd_report(args) -> None:
    from . import experiments
    records = []
    for path in args.files:
        records += experiments.load_records(Path(path))
    text = experiments.report(records)
    if args.write:
        out = Path(args.files[0])
        out.with_name(out.stem + "_report.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="bugforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="print a generated task")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--difficulty", default="easy", choices=list(DIFFICULTIES))
    s.add_argument("--files", action="store_true", help="also print the workspace files")
    s.set_defaults(fn=cmd_show)
    b = sub.add_parser("baselines", help="run reference and exploit agents")
    b.add_argument("--seeds", type=int, default=10)
    b.add_argument("--difficulty", action="append", choices=list(DIFFICULTIES))
    b.set_defaults(fn=cmd_baselines)
    c = sub.add_parser("calibrate", help="run an LLM agent over the difficulty grid")
    c.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    c.add_argument("--model", default=None, help="default: qwen2.5-coder:3b / claude-haiku-5-5")
    c.add_argument("--seeds", type=int, default=8, help="seeds per configuration")
    c.add_argument("--sandbox", default=None, choices=["local", "docker"],
                   help="where model-written code runs (default: $BUGFORGE_SANDBOX or local)")
    c.add_argument("--max-steps", type=int, default=30)
    c.add_argument("--request-timeout", type=float, default=None,
                   help="seconds per model call (default: 1800 for ollama, 120 for anthropic)")
    c.add_argument("--max-tokens", type=int, default=0,
                   help="stop the sweep after this many tokens in total (0 = no limit)")
    c.add_argument("--named", action="store_true", help="use easy/medium/hard instead of the grid")
    c.add_argument("--out", default=None, help="JSONL file (default: runs/<provider>_<model>.jsonl)")
    c.set_defaults(fn=cmd_calibrate)
    e = sub.add_parser("enrich", help="add bug details to runs recorded before they were logged")
    e.add_argument("file")
    e.set_defaults(fn=cmd_enrich)
    r = sub.add_parser("report", help="calibration and failure report from JSONL runs")
    r.add_argument("files", nargs="+")
    r.add_argument("--write", action="store_true", help="also save <file>_report.md")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
