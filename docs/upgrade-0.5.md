# Upgrade to Daena 0.5.0

Use Python 3.11 or later. Keep the older checkout and workspace files; install the
new version from its own checkout to avoid importing an old local `cli` package.

```bash
conda activate AI
cd ~/Desktop
git clone --branch release-0.5.0 --single-branch https://github.com/NxFi3/daena-agent.git daena-0.5.0
cd daena-0.5.0
python -m pip install -e '.[desktop,ollama,openrouter,gemini,web]'
python -m cli --version
daena-studio --cwd ~/Desktop/test
```

For the terminal: `python -m cli --config config.json --cwd ~/Desktop/test`.
For the GUI without using the installed command:
`python -m src.studio --cwd ~/Desktop/test`.

The supplied model remains `qwen3.6:35b-a3b-q4_K_M`. Ollama must be running and the
model must already be installed on that machine. `OLLAMA_HOST` or the optional
`llm.provider_config.base_url` controls the endpoint.

## Changed defaults

- `runtime.completion_protocol` is now `implicit`, matching ordinary tool-calling
  chat APIs. The explicit protocol used in 0.4.0 is available as an option. This
  removes the runtime's repeated continuation nudges after conversational answers;
  action selection and completion still depend on the model following instructions.
- The prompt budget reserves space for schemas and output. Context uses at most
  80 recent events plus relevant retrieval, with incremental 2048-token summaries.
  Summaries and covered exchange IDs are stored per session in SQLite. Existing
  databases acquire the checkpoint table without discarding conversation events.
- Managed background commands are permitted in the supplied profile. They remain
  owned by the run and are stopped during cleanup. `security.allow_background=false`
  restores the previous explicit-approval policy.
- Logs use the user state directory (`$XDG_STATE_HOME/daena/logs`, normally
  `~/.local/state/daena/logs`), rather than creating logs in the launch directory.
  Override with `DAENA_LOG_DIR`; set console verbosity with `DAENA_LOG_LEVEL`.

Existing custom configs are preserved. In particular, a saved 0.4.0 config using
explicit completion keeps that behavior until changed. Studio saves a separate
workspace profile; its saved profile takes precedence over the initial `--config`.
Use Settings to change it. CLI `--config` continues to read the specified file.

## Provider changes

OpenRouter uses the OpenAI-compatible SSE endpoint through HTTPX. Gemini uses the
Google SDK's async stream and preserves opaque thought signatures during tool
round trips. Local-only options are filtered from remote calls, and switching
providers in the CLI keeps an independent configuration for each provider.
Cancellation also works while remote response headers are pending. Incomplete or
output-limited streams cannot execute partial tool calls.

No live cloud credentials or local Qwen inference are required by the test suite.
Passing transport fixtures does not establish availability or task quality on a
particular live model.
