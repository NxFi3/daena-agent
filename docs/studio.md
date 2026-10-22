# Daena Studio

Studio is an optional PySide6 client for the same `Agent` used by the terminal.
Install `.[desktop,ollama,openrouter,gemini,web]` and run
`daena-studio --cwd /path/to/workspace`.

On minimal Ubuntu/Debian systems, Qt also needs native libraries. The headless
Ubuntu CI image required the following installation before importing QtWidgets:

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libopengl0 libxcb-cursor0
```

On a normal graphical desktop these may already be installed. `QT_QPA_PLATFORM=offscreen`
is for headless tests, not the interactive desktop launcher.

## Conversation

The chat-first layout starts with suggested tasks; clicking a suggestion fills the
composer for review, without sending anything. The sidebar keeps conversation
history and the other Studio workspaces close at hand. Toggle it with **Ctrl+B**,
start a conversation with **Ctrl+N**, search history with **Ctrl+K**, or focus the
composer with **Ctrl+L**. The moon/sun control switches between persistent dark
and light appearances. Appearance changes do not reset an active agent session.
Right-click a saved chat to open or rename it; renaming updates the saved run
titles without changing the underlying conversation ID.

User messages use compact right-aligned bubbles; Daena's replies are shown in a
readable centered column. Debug receipts and model reasoning are stored but hidden
from the chat surface. Click the **Daena** model selector in the top bar, then
**Advanced controls & diagnostics…** to inspect tool activity, context, raw
output, reasoning and provider/tool controls. The same menu offers recent models,
provider switching and manual model entry. The **Reasoning** slider in the composer
sets Auto, Off, Low, Medium, or High; specialized On/Custom values remain available
in advanced controls. Configuration only changes between runs.

Enter a task and press the round send button or **Enter** (Shift+Enter inserts a
newline; Ctrl+Enter also sends). Replies stream into the conversation.
**Activity** in the diagnostics window shows tool receipts and runtime errors;
**Context** shows the actual request estimate including tool schemas.
**Stop** cooperatively cancels model I/O, processes and delegated work. Closing the
window during a run requests cancellation and waits for the worker thread to end.

While a conversation is busy, choose **Steer current task** to send new guidance,
or **Queue next message** for a follow-up after completion. No second turn runs
concurrently on the same Agent. New conversation keeps workspace files and plans,
but begins a new chat session. A plan is background state, not a new user request.

Select a saved conversation to restore its archived turns and continue its persistent session. The sidebar groups turns by session and supports search. Run
archives are JSON under `.daena/studio/runs`; the SQLite history and compaction
checkpoints live alongside them. Export a run with **Export run**. Archives are
local and may contain your prompts and tool output.

## Agent graph

Select a node to inspect its assignment, status, current tool, token estimates,
report, errors, file changes and recent events. Nodes move by dragging. Wheel zooms at the pointer; drag the background, use the middle button, or Space+drag to pan. Visible minus/plus/Reset zoom controls and **Fit graph** provide explicit navigation. Child contexts are independent: the
parent supplies their task and optional bounded reference context.

**Cancel child** requests cooperative stopping. **Inspect diff** reads a finished
worker's pending changes. **Request merge/discard** prepares guidance for the
coordinator, which integrates changes through the same conflict-checked tools.
Send that guidance while the run is active. Completed runs have already resolved
or discarded their pending snapshots during cleanup.

The `subagent` tool supports named assignments, narrower tool scopes, status,
inspection, `wait_all` and waits up to 300000 ms. These waits remain cancellable and
do not repeatedly call the parent model. Read-only children cannot execute commands;
use workers for command-based checks. Workers use isolated workspace copies,
independent memories and clients; their reports do not prove the parent task passed.
Children cannot increase the host's permissions or recursively delegate.

## Settings and skills

Choose provider, model, optional base URL and generation JSON. Use **Context &
agents** for context/output budgets, completion mode, skills, concurrency and child
deadlines. **Tools** configures the full tool allowlist, network tools and managed
background execution. An empty tool list is respected. Advanced JSON exposes the
remaining runtime options without forcing a particular task workflow.

Save settings between runs. The next turn creates a runtime with the new profile
and the same conversation ID. Credentials stay in `OPENROUTER_API_KEY`,
`GEMINI_API_KEY` / `GOOGLE_API_KEY`, or Ollama's existing local setup; profiles do
not need API keys. Applying a profile makes no paid generation request.

The Skills view lists instructions and lets you enable individual skills or add
search directories. Workspace skills live at `.daena/skills/<name>/SKILL.md` and
use YAML `name` and `description` metadata. Skills load on demand through
`skill_read`; they never grant tool permissions.

## Automation canvas

1. Add agent nodes and set each assignment. Switching to another agent preserves
   your edits; changes also save to a workspace-scoped draft after a short pause.
   **Update node** remains available for explicit validation.
2. Click an output port on the right of a node and then the input on the left of
   another node, or drag between the ports. Dependencies can also be edited in
   **Run after**.
3. Set each node's tool scope. The effective scope is its intersection with the
   workspace profile. In explicit completion mode it must include `task_finish`.
4. Save the versioned JSON blueprint, or open `docs/automation-example.json`.
5. Run it. Nodes execute sequentially in dependency order with independent agent
   sessions, sharing the selected workspace. Each receives bounded upstream reports
   as reference data. A failed node prevents later nodes from starting.

The live canvas shows state and reports, kept independently for each agent when
you switch selections. Automation execution never adds messages to the normal
chat, overwrites its graph, or changes its conversation ID. Automation receipts
are archived separately under `.daena/studio/automation_runs`; the editable
workspace draft lives at `.daena/studio/automation-draft.json`. **Stop** cancels
the active node; the editor is locked during execution. Loaded blueprints never
run automatically.
Cycles, self-dependencies and missing nodes are rejected. This is a local DAG
runner; scheduling, conditions, loops, webhooks and distributed execution are
future extensions. Workspace snapshots and tool policies are defense in depth,
not operating-system isolation for arbitrary generated code.

## Validation

The desktop tests run Qt offscreen against real runtime tools and SQLite, using
scripted model responses. They cover conversation execution, stop/close, history,
settings scopes, graph reports and worker merges, blueprint validation and actual
multi-node file work. Provider tests exercise the real Ollama SDK, HTTPX SSE and
Google SDK against local HTTP fixtures. Live model access and output quality must
be checked with the user's configured provider.

See [the 0.6 desktop guide](upgrade-0.6.md) for the keyboard, reasoning and tool controls.
