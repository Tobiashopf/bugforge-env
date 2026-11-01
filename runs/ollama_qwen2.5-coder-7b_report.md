# Calibration report: qwen2.5-coder:7b

63 episodes, 6 solved (10%). 766,020 tokens, 20.2 h of wall clock.

## Pass rate by configuration

| config | files | bugs | tests shown | n | solved | observed (95% CI) | predicted |
|---|---|---|---|---|---|---|---|
| `f1b1v60e` | 1 | 1 | 60% | 7 | 0 | 0% (0%-35%) | 26% |
| `f1b1v25` | 1 | 1 | 25% | 8 | 4 | 50% (22%-78%) | 28% |
| `f3b1v60e` | 3 | 1 | 60% | 8 | 2 | 25% (7%-59%) | 12% |
| `f3b2v40e` | 3 | 2 | 40% | 8 | 0 | 0% (0%-32%) | 5% |
| `f3b2v25` | 3 | 2 | 25% | 8 | 0 | 0% (0%-32%) | 5% |
| `f5b2v40e` | 5 | 2 | 40% | 8 | 0 | 0% (0%-32%) | 2% |
| `f5b3v40e` | 5 | 3 | 40% | 8 | 0 | 0% (0%-32%) | 1% |
| `f5b3v25` | 5 | 3 | 25% | 8 | 0 | 0% (0%-32%) | 1% |

## Difficulty model

Logistic regression of *solved* on the generator knobs (L2 penalty 1.0 on slopes). Coefficients are log-odds per unit.

| term | coefficient | std. error |
|---|---|---|
| intercept | +0.71 | 1.11 |
| n_bugs | -1.11 | 0.80 |
| n_files | -0.46 | 0.41 |
| visible_frac | -0.35 | 0.95 |

**Configurations closest to target pass rates** (from the grid):

Across the grid this model's predicted pass rate spans 1% to 28%.

- target 80%: not reachable within the grid (closest: `f1b1v25`, predicted 28%)
- target 50%: not reachable within the grid (closest: `f1b1v25`, predicted 28%)
- target 20%: `f1b1v60e` (predicted 26%)

## Pass rate by injected bug type (single-bug tasks)

| bug type | n | solved | observed (95% CI) |
|---|---|---|---|
| arith | 4 | 4 | 100% (51%-100%) |
| cmp | 2 | 1 | 50% (9%-91%) |
| noraise | 5 | 1 | 20% (4%-62%) |
| const | 5 | 0 | 0% (0%-43%) |
| negate | 4 | 0 | 0% (0%-49%) |
| bool | 2 | 0 | 0% (0%-66%) |
| reverse | 1 | 0 | 0% (0%-79%) |

## Why episodes failed

| failure mode | meaning | episodes | share |
|---|---|---|---|
| broke_syntax | submitted a module that no longer parses | 2 | 4% |
| never_ran_tests | submitted without ever running the tests | 2 | 4% |
| gave_up_without_editing | ran the tests, never edited anything, submitted | 46 | 81% |
| submitted_red | submitted while visible tests were still failing | 4 | 7% |
| missed_bug_file | fixed some buggy files but not all | 3 | 5% |

## Agent behaviour

| behaviour | value |
|---|---|
| ran the tests before the first edit | 97% |
| edited a test file | 0% |
| edited a bug-free module | 0% |
| repeated the same failing call 3+ times | 0% |
| stalled at least once (a reply with no tool call) | 94% |
| mean tool calls per episode | 5.7 |
| mean test runs per episode | 2.0 |
| mean tool-call errors per episode | 0.1 |
| tool calls written as plain-text JSON (recovered) | 335 of 361 |
| mean tokens per episode | 12,159 |
| median seconds per episode | 377 |
