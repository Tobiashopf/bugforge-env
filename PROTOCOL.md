# Evaluation protocol

## Historical cohort

The opening run used checkout seed 0. The pilot reused this checkout task and added five families at seed 0. The UCB1 experiment used checkout seeds 100–107 for instruction selection and 200–203 for the frozen evaluation. The release contains all 19 historical episodes and 33,915 tokens across 40 recorded requests. Treat the opening and pilot checkout as repeated observations. In the 19 episodes the agent took one environment step each. The model could make several inference requests before producing that step.

All reported model results use local Qwen2.5-Coder 1.5B. Historical text transport requested a 16,384-token context at temperature zero; the agent had up to 30 environment actions. In the opening episode, the model first proposed a repository module as a tool and then submitted. The episode used 1,481 input and 149 output tokens in 34.6 seconds. Minimum sampled free memory was 0.40 GiB across the Windows host. The 482-test result applies to the recorded Phase 13 validation and was reused after matching the source archive, Python environment and Docker image.

## Interface diagnostic

The six independently reset guided probes cover listing, reading the instructions, writing a harmless note, running visible tests, submitting and recovering from a deliberately rejected unknown tool. The catalog probe asks the agent to discover available files. A pass requires the expected action, correct arguments and the observed execution effect. The test probe expects a successful invocation that detects the defective baseline. A submit probe checks termination. Unexpected actions in guided probes are recorded before dispatch.

The baseline and explicit instructions were tested with text transport at 4,096 and 8,192 tokens. The explicit instruction names the tools and distinguishes paths from tool names. Neither prompt exposes the defect or hidden tests. The native condition used the explicit prompt at 4,096. Each condition contains six probes, giving 30 guided outcomes in total. Native success requires a structured tool-call field.

The two autonomous runs used the same checkout content hash, seed 1, text transport, 4,096 context tokens, temperature zero, a 256-token output cap and at most eight actions. Invalid proposals could trigger a retry. Three consecutive misses triggered automatic submission. The diagnostic parser accepts a single action object and checks argument names and types. This parser and the action limits differ from the historical setup. Compare prompts within the final pair. In that pair, the baseline proposed three unknown tools. The explicit prompt proposed `list_files` three times with an unsupported `path` argument. Both runs submitted automatically. The only graded environment step in each was the fallback.

## Stress test and controls

Two additional raw generation requests supplied byte-identical synthetic input. Ollama evaluated 2,050 input tokens at 4,096 and 5,013 at 8,192. Resident model size rose from 1,173,136,342 to 1,373,959,618 bytes (about 192 MiB). Both responses reached the output-length limit without reproducing the marker. This establishes relative shortening on this stress input. It does not establish truncation in shorter pilot conversations. Request-level host counters and Ollama's resident-model estimate measure different things.

A scripted control exercised the catalog, read, edit, visible-test and unknown-tool paths. An unchanged defective submission scored zero; a privileged agent with the reference implementation scored one. These checks passed in Docker. Local trusted-fixture controls are reported separately.

## Interpretation

The loss of a proposed call, the rejection of its arguments, execution of a tool, automatic submission and terminal grading are separate events. For the historical retrospective labels, the first observed rejection determines the category. The historical parser persisted structured calls for recovered text; their original wording is unavailable in the public tables. The holdout ran the selected baseline once per seed. It supplies no independently executed alternative for a treatment effect. The 3B tag was absent and downloads were disabled in the final diagnostic. The planned 180-cell calibration and any further UCB1 training episodes are outside the observed dataset.
