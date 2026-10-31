"""Build the kill matrix: for every mutation site, which hidden tests catch it.

Run once after editing a base task::

    python -m bugforge.index

It writes ``tasks/base/<name>/kill_matrix.json`` and prints each module's
*mutation score* -- the share of single-site mutants the hidden tests detect.
Surviving mutants are either equivalent (no observable change) or a gap in the
tests; both are excluded from task generation, and the list is printed so the
gaps can be closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import base as base_mod
from .mutations import apply, enumerate_sites
from .verifier import verify


def source_hash(b: base_mod.BaseTask, root: Path | None = None) -> str:
    """Hash of the raw files (not the normalized source, whose formatting
    depends on the Python version's ``ast.unparse``). Line endings are
    normalized so a Windows checkout hashes the same as a Linux one."""
    d = (root or base_mod.tasks_root()) / b.name
    h = hashlib.sha256()
    for f in (b.module_file, b.test_file, b.fuzz_file):
        h.update((d / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def build(name: str, root: Path | None = None, workers: int | None = None) -> dict:
    b = base_mod.load(name, root)
    sites = enumerate_sites(b.source)

    def one(site):
        mutant = apply(b.source, [site.id])
        res = verify({b.module_file: mutant}, [b], timeout=30)
        killed = sorted({f.split("::")[-1] for f in res.failed})
        if res.timed_out:
            killed = ["<timeout>"]
        elif not killed and not res.solved:
            killed = ["<collection error>"]
        return site, killed

    with ThreadPoolExecutor(max_workers=workers or max(2, os.cpu_count() or 2)) as pool:
        results = list(pool.map(one, sites))

    return {
        "name": name,
        "hash": source_hash(b, root),
        "tests": b.test_names(),
        "sites": {s.id: {"operator": s.operator, "line": s.line,
                         "description": s.description, "killed_by": killed}
                  for s, killed in results},
    }


def path_for(name: str, root: Path | None = None) -> Path:
    return (root or base_mod.tasks_root()) / name / "kill_matrix.json"


def load_index(name: str, root: Path | None = None) -> dict:
    p = path_for(name, root)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m bugforge.index` first")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data["hash"] != source_hash(base_mod.load(name, root), root):
        raise RuntimeError(f"{p} is stale: base task changed, rerun `python -m bugforge.index`")
    return data


def mutation_score(data: dict) -> tuple[int, int, list[str]]:
    total = len(data["sites"])
    survivors = [sid for sid, s in data["sites"].items() if not s["killed_by"]]
    return total - len(survivors), total, survivors


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="base tasks (default: all)")
    args = ap.parse_args(argv)
    for name in args.names or base_mod.available():
        data = build(name)
        path_for(name).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        killed, total, survivors = mutation_score(data)
        print(f"{name:<12} mutation score {killed}/{total} = {killed / total:.1%}")
        for sid in survivors:
            s = data["sites"][sid]
            print(f"    survivor {sid:<12} line {s['line']:<4} {s['description']}")


if __name__ == "__main__":
    main()
