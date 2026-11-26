# Measurement tables

This dataset describes local code-repair agent evaluations with Qwen2.5-Coder 1.5B. The historical cohort and the final diagnostic use different prompts, parsers and budgets. Keep their denominators separate.

| File | Unit | How to use it |
|---|---|---|
| episodes.csv | 19 completed historical episodes | One environment trajectory. Opening, pilot, training-arm and holdout labels identify the study phase. |
| assistant_turns.csv | 40 historical model turns | One first-disposition label per turn: unknown tool, parse failure or recognized submit. |
| requests.csv | 40 historical Ollama calls | Input/output token counts and evaluation durations in nanoseconds. Host memory and CPU aggregates repeat for all requests in an episode. |
| conditions.json | Historical experiment fingerprints | Selected settings, task and source hashes, prompt hash and immutable model/image identifiers. |
| scripted_controls.json | Trusted local control checks | Infrastructure evidence. Keep separate from model performance. |
| diagnostic.json | Final diagnostic run | Five conditions of six guided probes, two autonomous pilot outcomes, two raw-input stress measurements, Docker controls, model capabilities and ordered events. Includes 36 chat requests and four controller events. |
| diagnostic_memory.json | 38 monitored request windows | Host samples for 36 chat and two raw generation requests. Timings include monitoring overhead. |

**Keys and units.** Join episodes to turns and requests with `episode_id`. Within an episode, `turn` and `request` are one-based indices. Join the episode's `experiment_id` to the corresponding conditions object. The historical `difficulty` field identifies a repository family. The `seed` selects a generated instance; use `task_sha256` to compare task contents. `seconds` is whole-episode wall time. `steps` counts environment actions; `turns` counts model responses. `edits` counts distinct edited paths. `test_runs` counts tests initiated by the agent, excluding final grading. `auto_submit` identifies a fallback in these one-step submitted trajectories. `tool_errors` counts errors after dispatch. The reward is binary; `solved` records the verifier's success decision.

**Reading an observation.** The opening checkout row has two model turns, one environment step, zero edits, 1,481 input tokens and 149 output tokens. Its turn labels are `unknown_tool` and `recognized_submit`. Reconstruct totals by summing `prompt_eval_count` and `eval_count` by episode. Compute generation tokens per second as the sum of output tokens divided by the sum of output evaluation durations in seconds. Never add repeated host-level peaks or minima across requests.

**Limitations.** Seed reuse, fixed prompt order and small samples limit inference. No raw conversations, private test reports or workstation paths are distributed in the dataset. Counts from the fixed historical snapshot represent recorded behavior under the original adapter; direct comparison with the final diagnostic cannot identify a causal prompt effect. The final native condition had no structured calls. Only one model was evaluated in this study.
