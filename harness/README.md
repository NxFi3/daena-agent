# Daena Baseline Harness

The baseline harness measures runtime behavior before Experience is enabled.

It intentionally keeps the task set, config, model provider and verification
contract separate from the Experience subsystem so later comparisons can use
the same workload without changing the evaluator.

Each case has:
- a user task;
- optional workspace setup;
- an independent verifier;
- tags for grouped analysis.

Run with the configured provider:

    python -m harness.runner --repetitions 3

Run one case:

    python -m harness.runner --case basic_file_repair

The JSON report records completion, independent verification, latency,
iterations, LLM calls, tokens, tool calls, failures and blocked actions.

For a fair Experience experiment, keep this directory and the baseline
configuration immutable, run the same cases with Experience disabled/enabled,
and compare the per-case distributions rather than only one aggregate number.
