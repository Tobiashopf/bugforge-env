"""Seeded task generator: pick modules, inject bugs, choose which tests to show.

Every task is a pure function of ``(seed, difficulty)``. The generator only
chooses mutation sites the kill matrix says are detectable, then checks the
combined task end to end:

- the bugged workspace must fail the hidden tests (bugs can mask each other);
- the shown tests must fail too, so the agent has a signal to start from;
- the original sources must pass the hidden tests (the task is solvable).

If a draw fails a check it is redrawn from the same seed stream, so the result
is still deterministic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import base as base_mod
from .index import load_index
from .mutations import apply
from .verifier import verify


@dataclass(frozen=True)
class Difficulty:
    name: str
    n_files: int            # modules in the workspace
    n_bugs: int             # mutation sites injected (spread over the modules)
    visible_frac: float     # share of each module's unit tests shown to the agent
    each_bug_visible: bool  # every bug is caught by at least one shown test


DIFFICULTIES = {
    "easy": Difficulty("easy", n_files=1, n_bugs=1, visible_frac=0.6, each_bug_visible=True),
    "medium": Difficulty("medium", n_files=3, n_bugs=2, visible_frac=0.4, each_bug_visible=True),
    "hard": Difficulty("hard", n_files=5, n_bugs=3, visible_frac=0.25, each_bug_visible=False),
}


@dataclass
class Bug:
    file: str
    site: str
    operator: str
    line: int
    description: str


@dataclass
class Task:
    seed: int
    difficulty: str
    bases: list[str]
    workspace: dict[str, str]                 # file name -> content shown to the agent
    solution: dict[str, str]                  # module file -> correct source
    bugs: list[Bug] = field(default_factory=list)
    visible_tests: dict[str, list[str]] = field(default_factory=dict)

    @property
    def module_files(self) -> list[str]:
        return [f"{b}.py" for b in self.bases]

    def load_bases(self) -> list[base_mod.BaseTask]:
        return [base_mod.load(b) for b in self.bases]


INSTRUCTIONS = """\
# Task

Some of the Python modules in this directory have bugs. The tests you can see
are a partial sample of the tests used for grading, and grading also checks
behaviour on inputs that no visible test covers.

Find and fix the bugs in the modules. You may run the tests as often as you
like. Editing or adding test files has no effect on your grade: only the module
files ({modules}) are graded, with the original hidden tests.
"""


def _draw(rng: random.Random, diff: Difficulty, names: list[str]) -> Task | None:
    chosen = sorted(rng.sample(names, diff.n_files))
    indexes = {n: load_index(n) for n in chosen}
    bases = {n: base_mod.load(n) for n in chosen}

    # spread bugs: every module gets one before any gets a second
    targets = (chosen * diff.n_bugs)[: diff.n_bugs] if diff.n_bugs >= diff.n_files else \
        rng.sample(chosen, diff.n_bugs)
    picked: dict[str, list[str]] = {n: [] for n in chosen}
    for name in targets:
        sites = indexes[name]["sites"]
        used_lines = {sites[s]["line"] for s in picked[name]}
        pool = [sid for sid, s in sites.items()
                if s["killed_by"] and sid not in picked[name] and s["line"] not in used_lines]
        if not pool:
            return None
        picked[name].append(rng.choice(sorted(pool)))

    # choose visible tests per module
    visible: dict[str, list[str]] = {}
    for name in chosen:
        tests = indexes[name]["tests"]
        k = max(1, round(diff.visible_frac * len(tests)))
        must: set[str] = set()
        for sid in picked[name]:
            killers = [t for t in indexes[name]["sites"][sid]["killed_by"] if t in tests]
            if diff.each_bug_visible:
                if not killers:
                    return None
                must.add(rng.choice(sorted(killers)))
        rest = [t for t in tests if t not in must]
        visible[name] = sorted(must | set(rng.sample(rest, max(0, k - len(must)))))

    if not diff.each_bug_visible:  # still guarantee one visible failure overall
        name = rng.choice([n for n in chosen if picked[n]])
        killers = [t for t in indexes[name]["sites"][picked[name][0]]["killed_by"]
                   if t in indexes[name]["tests"]]
        if not killers:
            return None
        visible[name] = sorted(set(visible[name]) | {rng.choice(sorted(killers))})

    workspace: dict[str, str] = {}
    bugs: list[Bug] = []
    for name in chosen:
        b = bases[name]
        workspace[b.module_file] = apply(b.source, picked[name]) if picked[name] else b.source
        workspace[b.test_file] = base_mod.subset_tests(b.tests, set(visible[name]))
        for sid in picked[name]:
            s = indexes[name]["sites"][sid]
            bugs.append(Bug(b.module_file, sid, s["operator"], s["line"], s["description"]))
    workspace["README.md"] = INSTRUCTIONS.format(
        modules=", ".join(f"`{n}.py`" for n in chosen))

    return Task(seed=0, difficulty=diff.name, bases=chosen, workspace=workspace,
                solution={bases[n].module_file: bases[n].source for n in chosen},
                bugs=bugs, visible_tests=visible)


def _visible_fails(task: Task) -> bool:
    """Run only the visible tests on the bugged modules (what the agent will see)."""
    from .sandbox import Workspace
    with Workspace(task.workspace) as ws:
        return not ws.run_tests().all_passed


def generate(seed: int, difficulty: str = "easy", *, check: bool = True,
             max_attempts: int = 50) -> Task:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
    diff = DIFFICULTIES[difficulty]
    names = base_mod.available()
    if diff.n_files > len(names):
        raise ValueError(f"{difficulty} needs {diff.n_files} base tasks, found {len(names)}")
    rng = random.Random(f"bugforge:{difficulty}:{seed}")
    for _ in range(max_attempts):
        task = _draw(rng, diff, names)
        if task is None:
            continue
        task.seed = seed
        if not check:
            return task
        bases = task.load_bases()
        bugged = {f: task.workspace[f] for f in task.module_files}
        if verify(bugged, bases).solved:
            continue  # bugs masked each other
        if not _visible_fails(task):
            continue
        if not verify(task.solution, bases).solved:
            raise RuntimeError("reference solution fails its own hidden tests")
        return task
    raise RuntimeError(f"no valid task after {max_attempts} draws (seed={seed})")
