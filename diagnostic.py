"""Bounded BugForge tool-interface diagnostic. Model-written actions always use Docker."""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from bugforge.agents import OracleAgent, run_episode
from bugforge.agents.llm import SYSTEM_PROMPT, OllamaClient
from bugforge.env import TOOLS, BugForgeEnv
from bugforge.generator import generate
from bugforge.host_metrics import monitor, sample
from bugforge.local_study import TextClient
from bugforge.provenance import fingerprint

HOST = "http://127.0.0.1:11434"
SMALL = "qwen2.5-coder:1.5b"
MEDIUM = "qwen2.5-coder:3b"
SCHEMA = {t["name"]: t["parameters"] for t in TOOLS}
EXPLICIT = SYSTEM_PROMPT + (
    "\nUse exactly one action per turn. Only list_files, read_file, write_file, "
    "replace_in_file, run_tests and submit are actions. Filenames and module names "
    "are arguments, never tool names. Inspect the workspace before proposing an edit. "
    "Use a visible test result to check the edit before submitting. "
    "If an action is unavailable, use a listed alternative."
)
PROBES = {
    "catalog": ("Discover which files are in the workspace.", "list_files", {}),
    "read": ("Read the project instructions in README.md.", "read_file", {"path": "README.md"}),
    "edit": (
        "Create probe_note.txt containing exactly OK. Use a single write action.",
        "write_file",
        {"path": "probe_note.txt", "content": "OK"},
    ),
    "test": ("Run the visible test suite.", "run_tests", {}),
    "submit": ("The isolated submission exercise is finished. Submit now.", "submit", {}),
    "unknown": (
        "A prior action named open_project was rejected as unknown. List the workspace files using a supported action.",
        "list_files",
        {},
    ),
}


def api(route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(HOST + route, data=data, headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(req, timeout=240) as response:
        return json.load(response)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def classify(obj):
    if not isinstance(obj, dict):
        return "invalid_proposal", None
    fn = obj.get("function", obj)
    if not isinstance(fn, dict):
        return "invalid_proposal", None
    name = fn.get("name", fn.get("tool"))
    if not isinstance(name, str):
        return "invalid_proposal", None
    if name not in SCHEMA:
        return "unknown_tool", None
    args = fn.get("arguments", fn.get("args", fn.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return "invalid_arguments", None
    params = SCHEMA[name]
    if (
        not isinstance(args, dict)
        or set(args) != set(params["required"])
        or any(not isinstance(v, str) for v in args.values())
    ):
        return "invalid_arguments", None
    return "accepted", {"tool": name, "args": args}


def decode(text, native=None):
    if native is not None:
        if not isinstance(native, list):
            return "invalid_proposal", None
        if not native:
            return "no_native_call", None
        if len(native) != 1:
            return "multiple_proposals", None
        return classify(native[0])
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence[1]
    try:
        obj = json.loads(stripped)
    except ValueError:
        return ("parse_failure" if any(c in stripped for c in "{[") else "no_action"), None
    return classify(obj)


class CheckedEnv(BugForgeEnv):
    def grade(self):
        result = super().grade()
        if not result.valid and not result.violations:
            raise RuntimeError("Terminal evaluation incomplete: " + result.note)
        return result


def controls(sandbox):
    env = CheckedEnv("repository-checkout", sandbox=sandbox)
    checks = {}
    try:
        env.reset(0)
        for key, (_, name, args) in PROBES.items():
            if key == "submit":
                continue
            if key == "unknown":
                obs, *_ = env.step({"tool": "open_project", "args": {}})
                assert obs.startswith("error: unknown tool")
            before = env.tool_errors
            obs, *_ = env.step({"tool": name, "args": args})
            checks[key] = before == env.tool_errors
            if key == "edit":
                checks[key] &= env.ws.read_file("probe_note.txt") == "OK"
            if key == "test":
                checks[key] &= env.test_runs == 1 and env.last_visible_pass is False
        _, reward, term, _, _ = env.step({"tool": "submit", "args": {}})
        checks["unedited_submit"] = term and reward == 0
        oracle = run_episode(env, OracleAgent(env), 0)
        checks["privileged_reference"] = oracle["solved"] and oracle["reward"] == 1
        assert all(checks.values()), checks
        return {"kind": "scripted_control", "sandbox": sandbox, "checks": checks}
    finally:
        env.close()


def chat(model, ctx, mode, prompt, history, out, label, events):
    client = (OllamaClient if mode == "native" else TextClient)(model, host=HOST, num_ctx=ctx)
    payload = client.payload(prompt, history)
    payload["options"]["num_predict"] = 256
    payload["keep_alive"] = 0 if label == "medium-catalog" else "2m"
    t = time.monotonic()
    with monitor(out / "memory", {"case": label, "window": ctx, "transport": mode, "model": model}):
        data = api("/api/chat", payload)
    msg = data["message"]
    disposition, action = decode(
        msg.get("content", ""), msg.get("tool_calls", []) if mode == "native" else None
    )
    event = {
        "case": label,
        "model": model,
        "window": ctx,
        "transport": mode,
        "event": disposition,
        "action": action["tool"] if action else None,
        "prompt_sha256": fingerprint(payload["messages"]),
        "instruction_sha256": fingerprint(prompt),
        "request_bytes": len(json.dumps(payload).encode()),
        "seconds": round(time.monotonic() - t, 2),
        "history_clipped": False,
        "input_at_window_boundary": (
            data["prompt_eval_count"] >= ctx - 1 if data.get("prompt_eval_count") else None
        ),
        "output_limit_reached": data.get("done_reason") == "length",
        "done_reason": data.get("done_reason"),
        **{
            k: data.get(k)
            for k in ("prompt_eval_count", "eval_count", "eval_duration", "load_duration")
        },
    }
    events.append(event)
    # Private traces retain proposals, results and request identity without the verifier's info.
    with (out / "private-events.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(dict(event, raw_message=msg, arguments=action["args"] if action else None))
            + "\n"
        )
    return action


def probes(model, ctx, mode, prompt, out, label, events):
    results = {}
    env = CheckedEnv("repository-checkout", sandbox="docker")
    try:
        for key, (instruction, name, args) in PROBES.items():
            env.reset(0)
            if key == "unknown":
                rejection, *_ = env.step({"tool": "open_project", "args": {}})
                instruction += " Environment feedback: " + rejection
            action = chat(
                model,
                ctx,
                mode,
                prompt,
                [{"role": "user", "text": instruction}],
                out,
                label + "/" + key,
                events,
            )
            correct = action == {"tool": name, "args": args}
            row = {"correct_request": correct, "executed": False, "passed": False}
            if correct:
                before = env.tool_errors
                obs, _, term, _, _ = env.step(action)
                success = env.tool_errors == before
                if key == "edit":
                    success &= env.ws.read_file("probe_note.txt") == "OK"
                if key == "read":
                    success &= obs == env.ws.read_file("README.md")[: len(obs)]
                if key in ("catalog", "unknown"):
                    success &= "README.md" in obs.splitlines()
                if key == "test":
                    success &= (
                        env.test_runs == 1 and env.last_visible_pass is False and "failed" in obs
                    )
                if key == "submit":
                    success &= term
                row.update(executed=True, passed=success, execution_error=env.tool_errors > before)
            results[key] = row
    finally:
        env.close()
    return results


def pilot(prompt, out, label, events):
    from bugforge.agents.llm import ToolCall

    env = CheckedEnv("repository-checkout", max_steps=8, sandbox="docker")
    task = generate(1, "repository-checkout")
    try:
        obs, _ = env.reset(1, task=task)
        history = [{"role": "user", "text": obs}]
        misses = 0
        for turn in range(24):
            action = chat(SMALL, 4096, "text", prompt, history, out, label, events)
            if action is None:
                misses += 1
                if misses < 3:
                    history.append(
                        {
                            "role": "user",
                            "text": "Your response did not contain one valid catalog action. Retry.",
                        }
                    )
                    continue
                action = {"tool": "submit", "args": {}}
                events.append({"case": label, "event": "automatic_submit"})
            misses = 0
            call = ToolCall(str(turn), action["tool"], action["args"])
            before = env.tool_errors
            obs, reward, term, trunc, info = env.step(action)
            events.append(
                {
                    "case": label,
                    "event": "execution_error" if env.tool_errors > before else "executed",
                    "tool": action["tool"],
                    "observation_clipped": "characters truncated]" in obs,
                }
            )
            history.extend(
                [
                    {"role": "assistant", "text": "", "calls": [call]},
                    {"role": "tool", "results": [(call, obs)]},
                ]
            )
            if term or trunc:
                return {
                    "task_hash": fingerprint(
                        {"workspace": task.workspace, "solution": task.solution}
                    ),
                    "seed": 1,
                    "reward": reward,
                    "steps": env.steps,
                    "edited": len(env.edited),
                    "test_runs": env.test_runs,
                    "end": info["end"],
                }
        raise RuntimeError("Pilot did not terminate")
    finally:
        env.close()


def context_check(ctx, out):
    # The raw input is byte-identical across windows and contains no repository content.
    text = "Remember the marker MINT. " + " blue" * 5000 + " Return the first marker only."
    with monitor(out / "memory", {"case": "context-stress", "window": ctx}):
        d = api(
            "/api/generate",
            {
                "model": SMALL,
                "prompt": text,
                "raw": True,
                "stream": False,
                "options": {"num_ctx": ctx, "temperature": 0, "num_predict": 12},
            },
        )
    return {
        "input_sha256": fingerprint(text),
        "prompt_tokens_evaluated": d.get("prompt_eval_count"),
        "marker_recalled": "MINT" in d.get("response", "").upper(),
        "done_reason": d.get("done_reason"),
        "resident": resident(),
    }


def resident():
    return [
        {k: m.get(k) for k in ("name", "context_length", "size", "size_vram")}
        for m in api("/api/ps")["models"]
        if m.get("name") in (SMALL, MEDIUM)
    ]


def report(result):
    lines = [
        "# Tool-interface diagnostic",
        "",
        f"Execution status: {result['status']}.",
        "",
        "| Condition | Probe | Correct request | Executed | Passed |",
        "|---|---|---:|---:|---:|",
    ]
    for label, rows in result["probes"].items():
        for key, row in rows.items():
            lines.append(
                f"| {label} | {key} | {row['correct_request']} | {row['executed']} | {row['passed']} |"
            )
    for name, row in result["pilots"].items():
        lines.append(
            f"\n{name}: reward {row['reward']}, steps {row['steps']}, edited files {row['edited']}, visible test runs {row['test_runs']}."
        )
    lines.extend(
        [
            "",
            "Probe results are guided. Pilot results are autonomous. Reference controls are privileged.",
            "Counts at the context boundary are indicators. See the matched raw-input stress test separately.",
            "The memory samples cover the entire Windows host.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("diagnostic-runs"))
    args = parser.parse_args()
    out = args.out / uuid.uuid4().hex
    out.mkdir(parents=True)
    events = []
    result = {
        "protocol": 1,
        "python": platform.python_version(),
        "events": events,
        "probes": {},
        "pilots": {},
        "contexts": {},
        "larger_model": {},
        "status": "incomplete",
    }
    try:
        if args.controls_only:
            result["controls"] = controls("local")
        else:
            if os.environ.get("DOCKER_HOST") and not os.environ["DOCKER_HOST"].startswith("npipe:"):
                raise RuntimeError("Expected a local Docker pipe")
            endpoint = subprocess.check_output(
                ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
                text=True,
            ).strip()
            if not endpoint.startswith("npipe:"):
                raise RuntimeError("Local Windows Docker context required")
            image = subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", "bugforge-sandbox"], text=True
            ).strip()
            os.environ["BUGFORGE_DOCKER_IMAGE"] = image
            result["image"] = image
            tags = {m["name"]: m for m in api("/api/tags")["models"]}
            result["digests"] = {m: tags[m]["digest"] for m in (SMALL, MEDIUM) if m in tags}
            large = result["larger_model"]
            disk = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
            while not disk.exists():
                disk = disk.parent
            large.update(
                installed=MEDIUM in tags,
                free_disk_gib=shutil.disk_usage(disk).free / 2**30,
                free_ram_gib_before=sample()[0] / 2**30,
                download_permitted=False,
            )
            large["disk_meets_6gib_estimate"] = large["free_disk_gib"] >= 6
            if SMALL not in tags:
                raise RuntimeError("Small model missing")
            result["controls"] = controls("docker")
            show = api("/api/show", {"model": SMALL})
            caps = show.get("capabilities")
            result["capabilities"] = caps
            for name, prompt in (("baseline", SYSTEM_PROMPT), ("explicit", EXPLICIT)):
                label = name + "-4096-text"
                result["probes"][label] = probes(SMALL, 4096, "text", prompt, out, label, events)
                result["pilots"][name] = pilot(prompt, out, name + "-pilot", events)
            for ctx in (4096, 8192):
                if ctx == 8192 and sample()[0] / 2**30 < 1:
                    result["contexts"][ctx] = {
                        "skipped": "Less than 1 GiB free before increasing context"
                    }
                    continue
                try:
                    result["contexts"][ctx] = context_check(ctx, out)
                except (TimeoutError, OSError) as err:
                    result["contexts"][ctx] = {"error_type": type(err).__name__}
                    continue
                if ctx == 8192:
                    for name, prompt in (("baseline", SYSTEM_PROMPT), ("explicit", EXPLICIT)):
                        label = name + "-8192-text"
                        result["probes"][label] = probes(
                            SMALL, ctx, "text", prompt, out, label, events
                        )
            counts = [result["contexts"][c].get("prompt_tokens_evaluated") for c in (4096, 8192)]
            result["stress_input_shortened_at_4096"] = (
                counts[0] < counts[1] if all(isinstance(x, int) and x > 0 for x in counts) else None
            )
            if caps is None or "tools" in caps:
                try:
                    result["probes"]["explicit-4096-native"] = probes(
                        SMALL, 4096, "native", EXPLICIT, out, "native", events
                    )
                except HTTPError as e:
                    detail = e.read().decode(errors="replace")
                    if e.code != 400 or "tool" not in detail.lower():
                        raise
                    result["native_status"] = "server rejected tools"
            else:
                result["native_status"] = "tools absent from advertised capabilities"
            free = sample()[0] / 2**30
            large["free_ram_gib_after"] = free
            large["required_free_gib_estimate"] = (
                3 + tags.get(MEDIUM, {}).get("size", 0) / 2**30 if MEDIUM in tags else None
            )
            large["memory_meets_estimate"] = (
                free >= large["required_free_gib_estimate"] if MEDIUM in tags else None
            )
            if MEDIUM in tags and large["memory_meets_estimate"]:
                a = chat(
                    MEDIUM,
                    4096,
                    "text",
                    EXPLICIT,
                    [{"role": "user", "text": PROBES["catalog"][0]}],
                    out,
                    "medium-catalog",
                    events,
                )
                large["catalog_passed"] = a == {"tool": "list_files", "args": {}}
            final = {m["name"]: m["digest"] for m in api("/api/tags")["models"]}
            if any(final.get(k) != v for k, v in result["digests"].items()):
                raise RuntimeError("Model identity changed")
        result["status"] = "complete"
    except Exception as err:
        result["failure_type"] = type(err).__name__
        raise
    finally:
        save(out / "results.json", result)
        (out / "REPORT.md").write_text(report(result), encoding="utf-8")
        print(out.resolve(), flush=True)


if __name__ == "__main__":
    main()
