"""Sequential local-only pilot, paired calibration and prompt-policy bandit."""

import argparse
import json
import math
import os
import subprocess
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from .agents.llm import SYSTEM_PROMPT, Budget, OllamaClient
from .experiments import load_records, run_grid
from .generator import REPOSITORY_DIFFICULTIES
from .host_metrics import monitor
from .provenance import fingerprint
from .study import read_plan, report

HOST = "http://127.0.0.1:11434"
PROMPTS = (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT + "\nBefore editing, trace the failing input through the implementation.",
)
FAMILY = "repository-checkout"


class MeasuredClient(OllamaClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []

    def parse(self, data, turn):
        self.requests.append(
            {
                k: data[k]
                for k in (
                    "eval_count",
                    "eval_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "load_duration",
                    "total_duration",
                )
                if k in data
            }
        )
        return super().parse(data, turn)


class TextClient(MeasuredClient):
    """Plain chat transport for local models without native function calling."""

    def payload(self, system, transcript):
        payload = super().payload(system, transcript)
        schema = payload.pop("tools")
        messages = payload["messages"]
        messages[0]["content"] += (
            '\nReply with one JSON object {"name":"...", "arguments":{...}}. Available tools: '
            + json.dumps(schema)
        )
        for message in messages[1:]:
            if "tool_calls" in message:
                calls = message.pop("tool_calls")
                message["content"] += "\n" + "\n".join(json.dumps(c["function"]) for c in calls)
            if message["role"] == "tool":
                message["role"] = "user"
                message["content"] = message.pop("tool_name") + " result:\n" + message["content"]
        return payload


def choose_arm(history, explore):
    counts = [len(values) for values in history]
    if explore and 0 in counts:
        return counts.index(0)
    means = [sum(values) / len(values) if values else 0 for values in history]
    if explore:
        means = [v + math.sqrt(2 * math.log(sum(counts)) / n) for v, n in zip(means, counts)]
    return max(range(len(means)), key=means.__getitem__)


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Configuration changed. Choose a new output directory: {path}")
        return
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


@contextmanager
def writer(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "active.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def episode(client, path, family, seed, plan, model, prompt, budget):
    labels = {"family": family, "seed": seed, "model": model["model"], "series": path.stem}
    client.requests.clear()
    labels["ollama_requests"] = client.requests
    cached = any(r["difficulty"] == family and r["seed"] == seed for r in load_records(path))
    context = nullcontext({}) if cached else monitor(path.parent / "metrics", labels)
    with context as metrics:
        rows = run_grid(
            client,
            path,
            [REPOSITORY_DIFFICULTIES[family]],
            [seed],
            sandbox="docker",
            max_steps=plan["max_steps"],
            budget=budget,
            model_revision=model["revision"],
            system=prompt,
        )
        for row in rows:
            if (row["difficulty"], row["seed"]) == (family, seed):
                metrics.update(
                    completed=True, text_calls=row["text_calls"], tool_errors=row["tool_errors"]
                )
                return row
    raise RuntimeError(f"Incomplete episode ({family}, {seed}). Inspect logs before resuming.")


def run(args):
    plan = read_plan(args.plan)
    directory = args.output / args.stage
    with writer(directory):
        inspect = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", "bugforge-sandbox"],
            text=True,
            timeout=20,
            capture_output=True,
            check=False,
        )
        if inspect.returncode:
            raise RuntimeError(inspect.stderr.strip() or "Docker image inspection failed")
        inspected = inspect.stdout.strip()
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["BUGFORGE_DOCKER_IMAGE"] = inspected
        opener = build_opener(ProxyHandler({}))
        with opener.open(HOST + "/api/tags", timeout=10) as response:
            installed = {m["name"]: m["digest"] for m in json.load(response)["models"]}
        chosen = ["qwen-small"] if args.stage in {"first", "rl"} else args.models
        models = [m.copy() for m in plan["models"] if m["id"] in chosen]
        if len(models) != len(set(chosen)):
            raise ValueError("Unknown model ID")
        for model in models:
            if (
                model["provider"] != "ollama"
                or model["model"] not in installed
                or model["model"] not in {f"qwen2.5-coder:{size}" for size in ("1.5b", "3b", "7b")}
            ):
                raise ValueError(f"Local Ollama model unavailable: {model['model']}")
            model["revision"] = installed[model["model"]]
        # Keep all planned models visible in coverage reports.
        for model in models:
            next(m for m in plan["models"] if m["id"] == model["id"]).update(model)
        if args.stage in {"first", "pilot"}:
            plan["seeds"] = [0]
        if args.stage in {"first", "rl"}:
            plan["families"] = [FAMILY]
        frozen(
            directory / "protocol.json",
            {
                "plan": plan,
                "participants": chosen,
                "image": inspected,
                "transport": args.transport,
                "rl_prompts": [fingerprint(p) for p in PROMPTS],
                "train": list(range(100, 108)),
                "holdout": list(range(200, 204)),
            },
        )
        budget = Budget(args.max_tokens)
        for model in models:
            client_type = TextClient if args.transport == "text" else MeasuredClient
            client = client_type(model["model"], host=HOST)
            if args.stage == "rl":
                history = [[], []]
                for seed in range(100, 108):
                    arm = choose_arm(history, True)
                    row = episode(
                        client,
                        directory / f"train-{arm}.jsonl",
                        FAMILY,
                        seed,
                        plan,
                        model,
                        PROMPTS[arm],
                        budget,
                    )
                    history[arm].append(row["reward"])
                arm = choose_arm(history, False)
                paired = []
                for seed in range(200, 204):
                    outcomes = [
                        episode(
                            client,
                            directory / f"holdout-{a}.jsonl",
                            FAMILY,
                            seed,
                            plan,
                            model,
                            PROMPTS[a],
                            budget,
                        )
                        for a in sorted({0, arm})
                    ]
                    if len({r["task_sha256"] for r in outcomes}) != 1:
                        raise ValueError("Holdout task mismatch")
                    paired.append(outcomes[-1]["reward"] - outcomes[0]["reward"])
                frozen(
                    directory / "policy.json",
                    {"history": history, "arm": arm, "paired_delta": paired},
                )
            else:
                for seed in plan["seeds"]:
                    for family in plan["families"]:
                        episode(
                            client,
                            directory / (model["id"] + ".jsonl"),
                            family,
                            seed,
                            plan,
                            model,
                            SYSTEM_PROMPT,
                            budget,
                        )
                        (directory / "report.md").write_text(
                            report(plan, directory), encoding="utf-8"
                        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("first", "pilot", "calibration", "rl"))
    parser.add_argument("--plan", type=Path, default=Path("study_plan.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase13"))
    parser.add_argument("--models", nargs="+", default=["qwen-small"])
    parser.add_argument("--transport", choices=("native", "text"), default="text")
    parser.add_argument("--max-tokens", type=int, default=100000)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("provide a positive token ceiling")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    event = {"stage": args.stage, "status": "interrupted"}
    try:
        run(args)
        event["status"] = "completed"
    except Exception as exc:
        event.update(status="blocked", error=str(exc))
        raise
    finally:
        event["seconds"] = round(time.monotonic() - start, 3)
        (args.output / f"attempt-{uuid.uuid4().hex}.json").write_text(
            json.dumps(event, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
