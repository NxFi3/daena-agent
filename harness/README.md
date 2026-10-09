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

Run the isolated local Flask security audit with OpenRouter:

    set -a && source .env && set +a
    python -m harness.runner --provider openrouter --model deepseek/deepseek-v4.1-flash --case security_audit_test101 --output security-results.json

Use --provider and --model to compare providers with the same task and verifier.
The output-token cap is model/provider-native by default; the legacy fixed
Ollama num_predict=1024 value is removed from the shared config.

The JSON report records provider/model, independent verification, completion
rate, latency, iterations, LLM calls, tokens, tool calls and failures.

For a fair Experience experiment, keep this directory and the baseline
configuration immutable, run the same cases with Experience disabled/enabled,
and compare the per-case distributions rather than only one aggregate number.
