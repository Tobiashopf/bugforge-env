"""Command line: inspect tasks and run the reference agents.

    python -m bugforge show --difficulty medium --seed 3
    python -m bugforge baselines --seeds 20
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

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
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
