"""Summarize resource samples from new episode attempts in a single stage."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    lines = [
        "# Episode resource and tool observations",
        "",
        "| Series | Family | Seed | Recorded outcome | Wall s | Output tok/s¹ | Min free GiB² | Peak CPU %² | Text calls³ | Tool errors³ |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted((directory / "metrics").glob("*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        completed = m.get("completed", False)
        pairs = [
            r
            for r in m.get("ollama_requests", [])
            if "eval_count" in r and r.get("eval_duration", 0) > 0
        ]
        duration = sum(r["eval_duration"] for r in pairs)
        speed = sum(r["eval_count"] for r in pairs) * 1e9 / duration if duration else None
        available = m.get("min_available_bytes")
        values = [
            m["series"],
            m["family"],
            m["seed"],
            "yes" if completed else "missing",
            m["elapsed_seconds"],
            speed,
            available / 2**30 if available is not None else None,
            m.get("peak_system_cpu_percent"),
            m.get("text_calls"),
            m.get("tool_errors"),
        ]
        cells = [
            "—" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v) for v in values
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if len(lines) == 4:
        lines.extend(["", "No episode measurements available."])
    lines.extend(
        [
            "",
            "¹ Output generation only, from responses with paired token and duration counters.",
            "² Whole Windows host, sampled every two seconds.",
            "³ From the completed episode record. Missing attempts have no score.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("expected an existing stage directory")
    print(summarize(args.directory))


if __name__ == "__main__":
    main()
