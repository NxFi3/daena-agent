# Daena Agent

Daena is a local-first Python runtime for tool-using language models. The model
chooses the work; Daena executes tools, preserves their results, manages context,
and owns cancellation and resource cleanup.

## Install and run

Python 3.11+ and a running Ollama server are required for the default provider.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ollama,web]'
daena --config config.json --cwd /path/to/project
```

The default model is `qwen3.6:35b-a3b-q4_K_M`. Use `--model MODEL` or `/model MODEL` to select
another installed model. `python -m cli` also works from the source directory.
For OpenRouter or Gemini, install `.[openrouter]` or `.[gemini]` and configure
the provider and credentials. A successful package installation does not establish
model availability.

**Upgrading from 0.3.0:** see [upgrade instructions](docs/upgrade-0.4.md).
Use `python -m cli --version` to check which source version is running.

## Runtime behavior

- Canonical assistant/tool messages stay paired and chronological. The original
  request is not appended again after every tool result.
- Incremental history summaries cover older exchanges. Recent receipts, explicit
  user steering, and the current filesystem inventory remain visible. Large
  historical patches are bounded without modifying the original memory records.
- The supplied configuration uses explicit completion: ordinary assistant text is
  progress; the model calls `task_finish` with `completed`, `blocked`, or
  `needs_input` to end the turn. Evidence is reported by the model, not inferred
  from keywords. Library integrations can retain implicit text completion by
  omitting `runtime.completion_protocol` or setting it to `implicit`.
- Workspace plans and recent tool exchanges remain visible across follow-up turns.
  `plan.read` inspects the plan; `plan.replace` archives it before replacement.
- Repetition does not trigger action vetoes. The runtime does not infer task
  completion from command names or force a particular planning/verification flow.
  Iteration limits, tool contracts and host permissions still apply.
- Ollama requests stream through a per-request asynchronous client. Cancellation
  interrupts waits for response headers and subsequent chunks, including context
  compaction requests. Managed command waits and polling also observe cancellation.
- Final responses wait for owned processes and child results to be resolved.
  `metrics.outcome` distinguishes completion, blockers, questions and runtime stops.
  `metrics.completed` means the model reported completion, not that its claims were
  independently proven correct. Model output must still be evaluated against the task.

## Skills and parallel agents

Six bundled skills cover software engineering, systematic debugging, research,
code review, delegation and data analysis. Skill metadata enters the catalog;
`skill_read` loads instructions on demand. Custom skills live at
`<workspace>/.daena/skills/<name>/SKILL.md`. `/skills` lists them.

The `subagent` tool supports `spawn`, `status`, `wait`, `list`, `cancel`, `diff`,
`merge`, and `discard`. Each child has its own model client, conversation, memory
and tools. Read-only children inspect the parent workspace. Workers edit isolated
copies concurrently. The parent reviews changes and explicitly merges them;
conflicting files leave the whole merge unapplied. `/agents` shows capabilities
and integration status.

Workers are enabled in the supplied configuration. Explicit
`subagents.allow_writes: false` is respected and removes worker mode from the
advertised schema. There is no keyword-based automatic delegation: the model
selects the tool from the user's request and current capabilities.

See [skills and delegation](docs/skills-and-subagents.md) for the API, configuration
and snapshot limitations.

## CLI

Use `/help` for all commands. Common controls:

- `/cwd PATH`, `/model NAME`, `/provider NAME`
- `/skills`, `/agents`
- `/sessions`, `/session N`, `/session new`
- `/reasoning on|off|auto|low|medium|high`
- `/busy steer|interrupt|queue`, `/interrupt`, `/exit`

Reasoning defaults to the provider/model setting. `on` and `off` are booleans;
string levels are for models that support them. Ollama uses `OLLAMA_HOST` when
set. `request_timeout_seconds` defaults to `null`: slow local generation may wait
without a read timeout while cancellation remains available. Set a positive
number for a read-idle timeout, or `total_timeout_seconds` for an overall request
deadline. Connection timeout remains 10 seconds. The CLI shows waiting telemetry;
this indicates pending I/O, not proof that the server is progressing. Partial or
output-truncated streams never execute tool calls. Subagent deadlines separately
bound the delegated task.

`make_directory` creates trees from explicit paths, for example
`{"paths":["public/css","public/js","routes","data"]}`.
`command_exec` takes literal argument arrays, not shell syntax. For example:
`["mkdir", "-p", "css", "js", "images", "fonts"]`. Shell brace expansion inside
one array element does not create several directories.

## Boundaries

File tools enforce workspace paths and background commands are disabled by default.
These checks and worker snapshots are **not an OS sandbox**: programs can access
resources outside their working directory. Use an OS/container boundary for
untrusted execution. Other provider adapters may finish blocked I/O before they
observe cancellation; the prompt interruption guarantees above apply to Ollama.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m ruff check src cli harness tests
python -m compileall -q src cli harness tests
python -m pip check
python -m build
```

Tests include scripted model exchanges, a local HTTP server exercising the real
Ollama SDK, concurrent worker integration, skills, security and tool behavior.
They do not measure the reasoning quality of a particular live model. Run the
[benchmark harness](harness/README.md) against a configured model for that check.
CI targets Python 3.11, 3.12 and 3.13.
