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

**Upgrading to 0.6.0:** see [upgrade instructions](docs/upgrade-0.6.md).
Use `python -m cli --version` to check which source version is running.

## Desktop app

```bash
python -m pip install -e '.[desktop,ollama,openrouter,gemini,web]'
daena-studio --cwd /path/to/project
# Or: python -m cli --gui --cwd /path/to/project
```

Daena Studio is a PySide6 application sharing the CLI's agent runtime. Enter sends; Shift+Enter inserts a newline. Provider, model, reasoning and searchable tool selection are available above the conversation. It includes
conversation history, streaming responses/reasoning, cancellation and steering,
a live graph with child reports and diffs, skills selection, provider/model
settings, and tool scopes. All bundled tools have permission by default, including
web search/fetch; install the `web` extra for search backends. The optional
`task_finish` tool is shown only in explicit completion mode.

The automation canvas supports draggable agent nodes, output-to-input connections,
per-node tool scopes, JSON save/load, cycle validation, and real execution in
order of dependencies. Nodes have independent contexts and receive bounded
upstream reports. The first version executes nodes sequentially against the
selected workspace; it is a local DAG runner without scheduled or webhook triggers.
See the [Studio guide](docs/studio.md) and [example blueprint](docs/automation-example.json).

Profiles and run archives live in `<workspace>/.daena/studio/`. API credentials
remain in environment variables. The desktop extra requires a graphical desktop;
headless tests use `QT_QPA_PLATFORM=offscreen`.

## Runtime behavior

- Canonical assistant/tool messages stay paired and chronological. The original
  request is not appended again after every tool result.
- Incremental history summaries cover older exchanges. Recent receipts, explicit
  user steering, and the current filesystem inventory remain visible. Large
  historical patches are bounded without modifying the original memory records.
- The default uses standard conversational completion: text accompanying tool calls
  is progress; text without tool calls ends the turn. A stored plan is historical
  workspace state and does not by itself authorize restarting work. Explicit
  `task_finish` completion remains an opt-in integration protocol. No keyword or
  language classifier decides whether a request is a greeting or a coding task.
- Tool schemas, message framing, and an output reserve count against the context
  budget. Incremental summaries survive follow-up turns and are checkpointed in
  SQLite for reopening a session; old user messages are eligible for compaction.
- Workspace plans and recent tool exchanges remain visible across follow-up turns.
  `plan.read` inspects the plan; `plan.replace` archives it before replacement.
- Repetition does not trigger action vetoes. The runtime does not infer task
  completion from command names or force a particular planning/verification flow.
  Iteration limits, tool contracts and host permissions still apply.
- Ollama, OpenRouter and Gemini use per-request asynchronous transports. Cancellation
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

The `subagent` tool supports `spawn`, `status`, `wait`, `wait_all`, `list`, `cancel`, `diff`,
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
