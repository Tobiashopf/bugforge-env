# bugforge-env

**Procedurally generated bug-repair environments for LLM agents, with a verifier
built to resist reward hacking.**

Each episode generates a small Python workspace with bugs injected by AST
mutation. The agent works through tool calls (read, edit, run the visible
tests, submit) and is graded against hidden tests plus differential fuzzing
against the original code. Tasks are a pure function of `(seed, difficulty)`,
so the supply is effectively unlimited, never in anyone's training data, and
the difficulty is a set of knobs rather than a fixed dataset.

```python
from bugforge.env import BugForgeEnv

env = BugForgeEnv(difficulty="medium")
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step(
    {"tool": "read_file", "args": {"path": "intervals.py"}})
```

---

## Why this exists

An RL environment is only as good as its reward. Two things go wrong in
practice:

1. **The reward can be gamed.** A policy under optimization pressure finds the
   cheapest way to make the number go up. If editing the tests, dropping a
   `conftest.py` or exiting with status 0 is cheaper than fixing the bug, that
   is what gets learned.
2. **The difficulty is not controlled.** A fixed benchmark is either too easy or
   too hard for the model being trained, and it leaks into pre-training data.

This repo treats both as engineering requirements with tests, not as hopes.

---

## How a task is built

```
tasks/base/<module>/           hand-written, correct code + strong tests
        │
        ├── mutation engine    7 AST operators (boundary flips, off-by-one,
        │                      operator swaps, negated conditions, removed guards…)
        ├── kill matrix        which hidden test catches which mutant (precomputed)
        ▼
generator(seed, difficulty)    pick modules, inject detectable bugs, pick which
        │                      tests to show, then check the task end to end:
        │                        • bugged code fails the hidden tests (no masking)
        │                        • the visible tests fail too (the agent has a signal)
        │                        • the original code passes (the task is solvable)
        ▼
workspace  ──►  agent tool calls  ──►  submit  ──►  verifier  ──►  reward
```

| difficulty | modules in workspace | injected bugs | unit tests shown | every bug visible? |
|---|---|---|---|---|
| easy | 1 | 1 | 60% | yes |
| medium | 3 (1 is a clean distractor) | 2 | 40% | yes |
| hard | 5 (2 are clean distractors) | 3 | 25% | no — one guaranteed visible failure |

With the five base modules there are 189 detectable single bugs, about 42
thousand two-bug combinations for *medium* and about 500 thousand three-bug
combinations for *hard*, before counting the choice of visible tests.

### Test quality is measured, not assumed

A bug the hidden tests cannot see would make a task unsolvable *in reward
terms* — the agent fixes it and nothing changes, or never needs to. So every
single-site mutant of every base module is run against the hidden tests, and
only detected mutants are ever injected:

| base module | unit tests | mutation sites | detected | mutation score |
|---|---|---|---|---|
| grades | 26 | 57 | 57 | 100% |
| inventory | 28 | 40 | 39 | 97.5% |
| intervals | 27 | 27 | 25 | 92.6% |
| pagination | 23 | 46 | 45 | 97.8% |
| textstats | 22 | 23 | 23 | 100% |

All four survivors were checked by hand and are *equivalent mutants*: they do
not change behaviour (for example, removing a `raise KeyError` that the next
line raises anyway). Rebuild the matrix with `python -m bugforge.index`; a test
fails if a base module changes and the matrix is stale.

---

## The verifier: defense in depth

Grading never trusts the workspace. Six layers, each aimed at a different way
to fake a pass:

1. **File allow-list** — only the module files named by the task are taken.
   Edited tests, `conftest.py`, `pytest.ini`, `sitecustomize.py` never reach
   the grader.
2. **Static policy** — submitted modules may not run code at import time,
   import outside a small pure-stdlib allow-list, or use `eval`, `getattr`,
   dunder attributes and similar. No legitimate fix needs any of it.
3. **Pristine hidden tests** in a fresh temporary directory, with an empty
   `pytest.ini`, plugin autoloading off and a clean `PYTHONPATH`.
4. **Counted results** — the JUnit report must list exactly the expected number
   of tests, all passed, none skipped. Exit status is ignored.
5. **Differential fuzzing** against the original module on random inputs, so
   special-casing the visible tests' inputs fails.
6. **Timeout** — infinite loops score zero instead of hanging the grader.

Eight exploit agents try to beat it, and **CI fails if any of them scores above
zero** on any difficulty:

| exploit | what it does | stopped by |
|---|---|---|
| `tamper_tests` | rewrites the visible tests to `assert True` | 3 |
| `conftest_hook` | `conftest.py` hook that marks every test passed | 1 |
| `ini_injection` | `pytest.ini` that disables collection | 1, 3 |
| `exit_at_import` | `os._exit(0)` before any test runs | 2, and 4 if 2 is removed |
| `patch_pytest` | monkeypatches pytest's report objects | 2 |
| `builtins_escape` | reaches `eval` through `__subclasses__()` | 2 |
| `import_reference` | re-exports the hidden reference module | 2 |
| `delete_everything` | empties the modules | the tests fail |

Layers are tested on their own as well: with the policy switched off,
`exit_at_import` and a module-level `pytest.skip` still score zero
(`tests/test_verifier.py`), and a module that hard-codes the answers to the
visible tests is caught by the fuzz test.

---

## Baselines

Mean reward over 20 seeds per difficulty (`python -m bugforge baselines --seeds 20`):

| agent | easy | medium | hard |
|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 |
| random | 0.00 | 0.00 | 0.00 |
| `tamper_tests` | 0.00 | 0.00 | 0.00 |
| `conftest_hook` | 0.00 | 0.00 | 0.00 |
| `ini_injection` | 0.00 | 0.00 | 0.00 |
| `exit_at_import` | 0.00 | 0.00 | 0.00 |
| `patch_pytest` | 0.00 | 0.00 | 0.00 |
| `builtins_escape` | 0.00 | 0.00 | 0.00 |
| `import_reference` | 0.00 | 0.00 | 0.00 |
| `delete_everything` | 0.00 | 0.00 | 0.00 |

(1,800 episodes, about 3 minutes on two CPU cores.)

The oracle (which writes back the original code) scoring 1.00 everywhere is the
solvability guarantee; the random and exploit agents scoring 0.00 bracket the
range from below.

---

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m pytest -q                                  # ~1 minute
python -m bugforge show --difficulty medium --seed 3 # inspect a task
python -m bugforge show --difficulty easy --seed 0 --files
python -m bugforge baselines --seeds 5
```

Python 3.10+, no dependencies besides pytest. CI runs on Linux and Windows.

### Environment API

| | |
|---|---|
| `reset(seed)` | returns the instructions and file list as text |
| `step({"tool": ..., "args": ...})` | tools: `list_files`, `read_file`, `write_file`, `replace_in_file`, `run_tests`, `submit` |
| `tool_schemas()` | JSON schemas for function-calling APIs |
| reward | 1.0 if every hidden test passes, else 0.0 (`partial_credit=True` pays the pass fraction) |
| episode end | `submit`, or the step budget runs out (the workspace is graded as it stands) |

`info` carries grading details and the injected bugs, for the harness only.

### Adding a base module

Create `tasks/base/<name>/` with `<name>.py`, `test_<name>.py` and
`fuzz_<name>.py` (copy an existing fuzz file: it compares your module with
`ref_<name>` on random inputs), then run `python -m bugforge.index` and fix
the tests until the only survivors are equivalent mutants.

---

## Layout

```
src/bugforge/
  mutations.py   AST mutation operators and deterministic site enumeration
  index.py       kill-matrix builder and mutation scores
  generator.py   seeded task generation with end-to-end validity checks
  sandbox.py     the agent's workspace and tools
  policy.py      static policy for submitted code
  verifier.py    hidden-test grading (the six layers above)
  env.py         Gymnasium-style environment
  agents/        oracle, random, and the eight exploit agents
tasks/base/      the five hand-written base modules
tests/           unit tests, solvability and exploit audits
```

## Roadmap

- **Docker sandbox backend.** The current workspace runs the agent's test
  command as a local subprocess. Grading is isolated, but running untrusted
  model-written code safely needs a container (no network, CPU/memory limits).
- **LLM agent and difficulty calibration.** A function-calling agent over an
  OpenAI-compatible API, a sweep over difficulty × seeds, and a logistic fit of
  pass rate so the generator can hit a target difficulty distribution for a
  given model.
- **Learnability check.** A small GRPO fine-tune on the easy tier to show the
  reward carries a learning signal.
- **Rollout analysis.** A failure taxonomy from agent transcripts.

## Limitations

- The base modules are small by design; difficulty comes from search across
  files and partial test visibility, not from code size.
- The static policy is strict on purpose. It would reject some unusual but
  legitimate code (for example, top-level calls).
- Equivalent-mutant detection is by hand; the four survivors are documented above.
- Generated workspaces are re-printed with `ast.unparse`, whose formatting can
  differ slightly between Python versions: a seed gives the same bugs and tests
  everywhere, but the exact text is reproducible per Python version.

## Provenance

Developed with substantial AI assistance (code and documentation). I defined
the scope, reviewed the code, and ran and checked the measurements.

## License

MIT.
