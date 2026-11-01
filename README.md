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

Python 3.10+, no dependencies besides pytest (the LLM clients use the standard
library). CI runs on Linux and Windows, plus a Docker sandbox job.

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

## Sandboxing model-written code

`run_tests` executes code a model wrote, so it never runs unguarded:

| runner | what runs | isolation |
|---|---|---|
| `local` (default) | only module files that pass the static policy, with the **original** visible tests; edited or added test files are ignored | the policy: no imports outside a pure-stdlib allow-list, no `eval`/`getattr`/dunders, no import-time calls |
| `docker` | the workspace exactly as the agent left it | `--network none`, read-only root filesystem, `--cap-drop ALL`, `no-new-privileges`, memory/CPU/process limits, timeout that kills the container |

Grading can run in the container too (`BugForgeEnv(sandbox="docker")` does
both). A CI job builds the image and runs the oracle, all eight exploits and a
network probe inside it, and fails if those tests are skipped.

```bash
docker build -t bugforge-sandbox docker
python -m bugforge calibrate --sandbox docker ...   # or set BUGFORGE_SANDBOX=docker
```

---

## LLM agent and difficulty calibration

`agents/llm.py` drives a tool-calling model through the environment, one tool
call per step (parallel calls are queued, prose answers get a nudge). Two
backends share a provider-neutral transcript:

- **Ollama** (free, local): any tool-capable model on your machine.
- **Anthropic** (paid): the Messages API with native tool use, key read from
  `ANTHROPIC_API_KEY`. A `--max-tokens` ceiling stops a sweep before it overspends.

Small open models often know which tool they want but print the call as JSON
in plain text instead of emitting a structured tool call (the first smoke run
with Qwen2.5-Coder-3B did exactly this on every turn). The agent recovers
those calls, accepting only objects that name a real tool, and the report
counts how often it had to.

`python -m bugforge calibrate` runs the agent over a grid of generator
settings (files × bugs × share of tests shown), appending one JSON line per
episode with the full transcript; it resumes where it stopped. The report then:

1. gives the observed pass rate per configuration with a **Wilson 95% interval**;
2. fits a **logistic difficulty model** (IRLS, lightly L2-regularized so a small,
   perfectly separable sweep still gives finite coefficients) on the knobs, and
   picks the configurations closest to target pass rates (80% / 50% / 20%) for
   that model;
3. classifies every failure from its transcript and episode metrics:
   ran out of steps, policy violation, never ran tests, submitted with red
   tests, edited after the last test run, edited the wrong files, missed one of
   several buggy files, or passed the visible tests but not the hidden ones;
4. reports behaviour: tests before the first edit, edits to tests or to
   bug-free modules, tool calls, tool errors, tokens and time per episode.

### Running it for free

```bash
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
docker exec ollama ollama pull qwen2.5-coder:3b
docker build -t bugforge-sandbox docker
python -m bugforge calibrate --provider ollama --model qwen2.5-coder:3b --seeds 8 --sandbox docker
```

On a CPU this takes hours (it is meant to run overnight) and costs nothing.
Results go to `runs/` as JSONL plus a Markdown report.

### Results: Qwen2.5-Coder-7B, local, on a laptop CPU

63 episodes (8 seeds × 8 configurations; one lost to a model-server timeout),
766k tokens, about 20 hours of wall clock, $0. Full report:
[`runs/ollama_qwen2.5-coder-7b_report.md`](runs/ollama_qwen2.5-coder-7b_report.md);
every transcript is in the JSONL next to it.

| configuration | files | bugs | tests shown | solved | observed (95% CI) |
|---|---|---|---|---|---|
| `f1b1v25` | 1 | 1 | 25% | 4 / 8 | 50% (22%–78%) |
| `f3b1v60e` | 3 | 1 | 60% | 2 / 8 | 25% (7%–59%) |
| `f1b1v60e` | 1 | 1 | 60% | 0 / 7 | 0% (0%–35%) |
| all 2- and 3-bug configurations | 3–5 | 2–3 | 25–40% | 0 / 40 | 0% |

**1. The bug type matters more than the configuration knobs.** On single-bug
tasks:

| injected bug | solved |
|---|---|
| swapped arithmetic operator (`+`↔`-`, `*`↔`/`) | 4 / 4 |
| boundary comparison (`<`↔`<=`) | 1 / 2 |
| removed guard (`raise` deleted) | 1 / 5 |
| off-by-one literal | 0 / 5 |
| negated `if` condition | 0 / 4 |
| `and`↔`or` | 0 / 2 |

This also explains the oddest cell above: showing *more* tests (`f1b1v60e`,
0/7) did worse than showing fewer (`f1b1v25`, 4/8) only because none of the
`f1b1v60e` draws happened to contain an arithmetic bug. The logistic model on
files/bugs/visibility has wide standard errors for the same reason. The bug
type is now a generator knob (`custom(..., operators=("arith",))`), and the
report breaks pass rates down by it.

**2. The model gives up more than it fails.** 81% of failed episodes never
edited a file: the agent ran the tests, read the code, stopped calling tools,
and then submitted. It used 5.7 of its 30 tool calls on average and
stalled at least once in 94% of episodes. For this model a larger step budget
would not help; what the environment measures at this level is whether the
agent commits to an edit at all.

**3. The format layer alone can zero an evaluation.** 335 of 361 tool calls
(93%) arrived as JSON in plain text rather than as structured calls. Before the
agent learned to recover them, the same harness with Qwen2.5-Coder-3B scored
0/64 with exactly one action per episode
([`runs/before-text-recovery_qwen2.5-coder-3b.jsonl`](runs/before-text-recovery_qwen2.5-coder-3b.jsonl)):
a reading of "can't fix bugs" that was really "can't emit tool calls".

**4. The grid does not reach easy tiers for this model.** Predicted pass rates
span 1% to 28%, so 80% and 50% tiers are out of reach; the report says so
instead of picking the nearest wrong configuration. The next sweep should
build the easy tier from single-file arithmetic and comparison bugs.

No reward hacking was attempted: no test files were edited and nothing hit the
sandbox policy (two submissions failed because the model broke the module's
indentation).

### Lessons from running it

- **Read transcripts before trusting a zero.** Two sweeps scored 0 for reasons
  that had nothing to do with the tasks: unrecovered plain-text tool calls, and
  a bug in the agent where the action queue aliased the transcript, so the
  model saw its own turns as empty. Both were found by reading transcripts;
  the second now has a regression test that checks the exact payload sent to
  the model.
- **On a CPU, prompt length is the cost.** Re-sending every file the agent has
  read made single calls exceed ten minutes. Old tool outputs are now clipped
  in what the model sees (the full transcript is still logged).
- **A sweep must survive its infrastructure.** A model-server timeout now skips
  the episode (rerunning fills it in) instead of killing an overnight run.

---

## Layout

```
src/bugforge/
  mutations.py   AST mutation operators and deterministic site enumeration
  index.py       kill-matrix builder and mutation scores
  generator.py   seeded task generation with end-to-end validity checks
  sandbox.py     the agent's workspace and tools (guarded local mode)
  runner.py      local and Docker execution backends
  policy.py      static policy for submitted code
  verifier.py    hidden-test grading (the six layers above)
  env.py         Gymnasium-style environment
  experiments.py calibration sweeps, logistic difficulty model, failure analysis
  agents/        oracle, random, eight exploit agents, and the LLM agent
docker/          sandbox image
tasks/base/      the five hand-written base modules
tests/           unit tests, solvability and exploit audits
```

## Roadmap

- **Bug-type tiers.** Re-run the calibration with tiers built from the
  `operators` knob, to confirm finding 1 with more than a handful of tasks per
  bug type.
- **Learnability check.** A small GRPO fine-tune on the easiest tier (Colab) to
  show the reward carries a learning signal, not just a difficulty reading.
- **A stronger model** on the same grid, to see whether the difficulty ordering
  holds when the floor effect is gone.

## Limitations

- The base modules are small by design; difficulty comes from search across
  files and partial test visibility, not from code size.
- The static policy is strict on purpose. It would reject some unusual but
  legitimate code (for example, top-level calls).
- Equivalent-mutant detection is by hand; the four survivors are documented above.
- The calibration results come from one small local model and 63 episodes; the
  intervals are wide and are reported as such. The agent's nudge after a reply
  with no tool call ("... Call submit when you are done") may make giving up
  easier; that wording is part of the measured setup, and an ablation is open.
- Generated workspaces are re-printed with `ast.unparse`, whose formatting can
  differ slightly between Python versions: a seed gives the same bugs and tests
  everywhere, but the exact text is reproducible per Python version.

## Provenance

Developed with substantial AI assistance (code and documentation). I defined
the scope, reviewed the code, and ran and checked the measurements.

## License

MIT.
