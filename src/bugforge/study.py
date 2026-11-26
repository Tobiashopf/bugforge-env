"""Paired repository-family study. Planning/reporting are offline; real runs require Docker."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from statistics import mean

from .agents.llm import Budget, make_client
from .experiments import _unique, load_records, run_grid, wilson
from .generator import REPOSITORY_DIFFICULTIES
from .provenance import config
from .runner import DockerRunner


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != 1 or plan.get("sandbox") != "docker":
        raise ValueError("expected schema 1 and Docker isolation")
    seeds = plan["seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    families = plan["families"]
    if not families or len(set(families)) != len(families) or not set(families) <= set(REPOSITORY_DIFFICULTIES):
        raise ValueError("invalid family list")
    if type(plan["max_steps"]) is not int or plan["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    models = plan["models"]
    ids = [m["id"] for m in models]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("need at least two distinct model IDs")
    for m in models:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", m["id"]) or m["provider"] not in {"ollama", "anthropic"} or not m["model"]:
            raise ValueError("invalid model configuration")
    if len({(m["provider"], m["model"], m.get("revision")) for m in models}) != len(models):
        raise ValueError("duplicate model configuration")
    return plan


def expected_keys(plan):
    return {(f, s) for f in plan["families"] for s in plan["seeds"]}


def rows_for(plan, model, path):
    records = _unique(load_records(path))
    rows = {}
    manifest = None
    for r in records:
        key = r["difficulty"], r["seed"]
        m = r["manifest"]
        if key not in expected_keys(plan) or key in rows:
            raise ValueError(f"unexpected or repeated episode in {path}: {key}")
        if (r["provider"], r["model"], m.get("model_revision_claimed")) != (model["provider"], model["model"], model.get("revision")):
            raise ValueError(f"wrong model in {path}")
        if (r["sandbox"] != "docker" or m["agent"]["max_steps"] != plan["max_steps"]
                or m["inference"].get("temperature") != 0
                or r["config"] != config(REPOSITORY_DIFFICULTIES[key[0]])):
            raise ValueError(f"configuration mismatch in {path}")
        if manifest is not None and m != manifest:
            raise ValueError(f"mixed manifests in {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", r["task_sha256"]):
            raise ValueError("invalid task hash")
        if type(r["solved"]) is not bool or r["reward"] != int(r["solved"]):
            raise ValueError("expected binary reward consistent with solved")
        manifest = m
        rows[key] = r
    return rows, manifest


def report(plan, directory: Path) -> str:
    expected = expected_keys(plan)
    all_rows = {}
    common = None
    task_hashes = {}
    lines = ["# Paired repository study", "", "Descriptive only: seeds can repeat bug combinations; Wilson intervals are nominal, not adjusted for dependence.",
             "Missing episodes are not failures. Provider-specific inference settings may differ.", "",
             "| Model | Recorded / planned | Solved | Rate (nominal Wilson 95%) | Mean steps | Mean tokens | Mean seconds | Unique task hashes |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for model in plan["models"]:
        name = model["id"]
        rows, manifest = rows_for(plan, model, directory / f"{name}.jsonl")
        all_rows[name] = rows
        if manifest:
            signature = {k: manifest[k] for k in ("agent", "runner", "runtime", "source_sha256", "tasks_sha256", "generation_fuzz_seed")}
            if common is not None and signature != common:
                raise ValueError("models have different shared harness/runtime manifests")
            common = signature
        for key, row in rows.items():
            if key in task_hashes and task_hashes[key] != row["task_sha256"]:
                raise ValueError(f"task mismatch across models: {key}")
            task_hashes[key] = row["task_sha256"]
        n = len(rows)
        k = sum(r["solved"] for r in rows.values())
        lo, hi = wilson(k, n)
        rate = f"{k/n:.1%} ({lo:.1%}–{hi:.1%})" if n else "—"
        steps = f"{mean(r['steps'] for r in rows.values()):.1f}" if n else "—"
        tokens = f"{mean(r['input_tokens'] + r['output_tokens'] for r in rows.values()):.1f}" if n else "—"
        seconds = f"{mean(r['seconds'] for r in rows.values()):.1f}" if n else "—"
        lines.append(f"| {name} | {n}/{len(expected)} | {k} | {rate} | {steps} | {tokens} | {seconds} | {len({r['task_sha256'] for r in rows.values()})} |")
    lines += ["", "## Per family", "", "| Model | Family | Solved / recorded | Missing |", "|---|---|---:|---:|"]
    for name, rows in all_rows.items():
        for family in plan["families"]:
            subset = [r for (f, _), r in rows.items() if f == family]
            lines.append(f"| {name} | {family} | {sum(r['solved'] for r in subset)}/{len(subset)} | {len(plan['seeds'])-len(subset)} |")
    lines += ["", "## Pairwise common episodes", "", "| A | B | Matched / planned | Both solve | A only | B only | Neither | A−B rate (paired) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for a, b in itertools.combinations(all_rows, 2):
        ra, rb = all_rows[a], all_rows[b]
        keys = ra.keys() & rb.keys()
        both = sum(ra[k]["solved"] and rb[k]["solved"] for k in keys)
        ao = sum(ra[k]["solved"] and not rb[k]["solved"] for k in keys)
        bo = sum(rb[k]["solved"] and not ra[k]["solved"] for k in keys)
        neither = len(keys) - both - ao - bo
        delta = f"{(ao-bo)/len(keys):+.1%}" if keys else "—"
        lines.append(f"| {a} | {b} | {len(keys)}/{len(expected)} | {both} | {ao} | {bo} | {neither} | {delta} |")
    lines += ["", "No significance or causal claims are made. Model revisions are operator-supplied, not independently verified."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("study_plan.json"))
    parser.add_argument("--directory", type=Path, default=Path("runs/paired-study"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("report")
    run = sub.add_parser("run")
    run.add_argument("model_id")
    run.add_argument("--max-tokens", type=int, required=True, help="soft per-invocation ceiling; can overshoot one request")
    run.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args(argv)
    plan = read_plan(args.plan)
    if args.command == "plan":
        n = len(expected_keys(plan))
        print(f"{len(plan['models'])} models × {n} episodes = {len(plan['models'])*n} planned episodes")
        return
    if args.command == "report":
        print(report(plan, args.directory))
        return
    model = next((m for m in plan["models"] if m["id"] == args.model_id), None)
    if model is None or args.max_tokens <= 0:
        parser.error("choose a model from the plan and a positive token ceiling")
    if model["provider"] == "anthropic" and not args.allow_paid:
        parser.error("paid provider requires --allow-paid")
    if not DockerRunner.available():
        raise RuntimeError("Docker unavailable; build the sandbox before running")
    # Check all existing results before spending tokens; no concurrent writers.
    report(plan, args.directory)
    client = make_client(model["provider"], model["model"])
    run_grid(client, args.directory / f"{model['id']}.jsonl",
             [REPOSITORY_DIFFICULTIES[f] for f in plan["families"]], plan["seeds"],
             sandbox="docker", max_steps=plan["max_steps"], budget=Budget(args.max_tokens),
             model_revision=model.get("revision"))
    print(report(plan, args.directory))


if __name__ == "__main__":
    main()
