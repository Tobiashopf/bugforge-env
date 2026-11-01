# Calibration report: qwen2.5-coder:3b

64 episodes, 0 solved (0%). 135,207 tokens, 0.4 h of wall clock.

## Pass rate by configuration

| config | files | bugs | tests shown | n | solved | observed (95% CI) | predicted |
|---|---|---|---|---|---|---|---|
| `f1b1v60e` | 1 | 1 | 60% | 8 | 0 | 0% (0%-32%) | - |
| `f1b1v25` | 1 | 1 | 25% | 8 | 0 | 0% (0%-32%) | - |
| `f3b1v60e` | 3 | 1 | 60% | 8 | 0 | 0% (0%-32%) | - |
| `f3b2v40e` | 3 | 2 | 40% | 8 | 0 | 0% (0%-32%) | - |
| `f3b2v25` | 3 | 2 | 25% | 8 | 0 | 0% (0%-32%) | - |
| `f5b2v40e` | 5 | 2 | 40% | 8 | 0 | 0% (0%-32%) | - |
| `f5b3v40e` | 5 | 3 | 40% | 8 | 0 | 0% (0%-32%) | - |
| `f5b3v25` | 5 | 3 | 25% | 8 | 0 | 0% (0%-32%) | - |

Not enough variation to fit a difficulty model (need some solved and some failed episodes across at least 3 configurations).

## Why episodes failed

| failure mode | meaning | episodes | share |
|---|---|---|---|
| never_ran_tests | submitted without ever running the tests | 64 | 100% |

## Agent behaviour

| behaviour | value |
|---|---|
| ran the tests before the first edit | 0% |
| edited a test file | 0% |
| edited a bug-free module | 0% |
| repeated the same failing call 3+ times | 0% |
| stalled at least once (a reply with no tool call) | 100% |
| mean tool calls per episode | 1.0 |
| mean test runs per episode | 0.0 |
| mean tool-call errors per episode | 0.0 |
| tool calls written as plain-text JSON (recovered) | 0 of 64 |
| mean tokens per episode | 2,113 |
| median seconds per episode | 13 |
