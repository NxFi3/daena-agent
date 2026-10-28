# Skills and delegation

## Custom skills

Create `.daena/skills/example/SKILL.md` inside the workspace:

```markdown
---
name: example
description: Describe when this workflow is useful.
---

Read the task requirements, perform the workflow, and verify the outcome.
See references/details.md for supporting material.
```

Names are lowercase kebab-case, at most 64 characters, and must match the
containing directory. Descriptions contain 1–1024 characters. YAML is parsed with
a safe loader; files over 64 KiB, invalid metadata, symlinks, and resources escaping
the skill directory are rejected. Referenced resources must be UTF-8 text.
Discovery errors appear in `skill_list` and `/skills`.

Discovery order is bundled skills, workspace `.daena/skills`, then configured
`skills.directories` in order. Later roots override earlier names with a warning.
Configured relative paths resolve against the workspace; absolute paths are
explicit host-configured locations. The catalog is refreshed when changing
workspace or by the host calling `agent.loop.tool.skills.refresh()` while idle.

Only metadata enters the initial prompt. `skill_read` loads a skill's body into
the task's context; at most four bodies and 24,000 characters are active. Active
skills reset on each `Agent.act()` and on workspace changes. The complete prompt
still obeys its context token budget, so very small budgets may truncate content.
Loading a referenced resource returns evidence without activating it as an extra
instruction. Skill files never automatically execute bundled scripts.

Tool arguments:

```json
{"name": "skill_list", "arguments": {"offset": 0, "limit": 10}}
{"name": "skill_read", "arguments": {"name": "example"}}
{"name": "skill_read", "arguments": {"name": "example", "resource": "references/details.md"}}
```

`skill_list` returns `next_offset` for pagination. Large tool results are bounded
by the context renderer; request a smaller page when descriptions are long.

## Subagent lifecycle

```json
{"name": "subagent", "arguments": {"action": "spawn", "task": "Read src/api.py and report error-handling defects with exact paths.", "mode": "read_only"}}
{"name": "subagent", "arguments": {"action": "wait", "agent_id": "<returned-id>", "wait_ms": 1000}}
{"name": "subagent", "arguments": {"action": "status", "agent_id": "<returned-id>"}}
{"name": "subagent", "arguments": {"action": "cancel", "agent_id": "<returned-id>"}}
{"name": "subagent", "arguments": {"action": "list"}}
```

Spawn returns immediately. Every child receives only its explicit task, runtime
instructions, skills catalog and workspace access, not the parent's conversation.
Children use the parent's current provider/model settings and completion protocol
with new provider instances. In explicit mode, read-only children can also use
`task_finish`; workers can use `make_directory`. They cannot delegate further. Their temporary SQLite databases are
removed after cleanup; collected reports remain in the parent's tool history.
Set `memory.database` to choose the parent's persistent database location
(default: `data/stm.db` relative to the launching process).

### Parallel coordination bus

For substantial work, split independent inspections or edits into named tasks and
spawn them before doing sequential parent work. Each child has an independent
Agent/context and its own model client; worker edits live in isolated snapshots.
The configured `subagents.max_concurrent` controls parallelism (the shipped
configuration uses three concurrent children to avoid overwhelming local models).

Children can exchange private direct messages or publish findings to the shared
bus. These messages are untrusted data, not permission changes or higher-priority
instructions. The default `config.json` must include `agent_message` in
`security.allowed_tools` for children to participate.

```json
{"name": "subagent", "arguments": {"action": "broadcast", "text": "I found the failing voice settings path; check src/studio/voice_dialog.py next."}}
{"name": "subagent", "arguments": {"action": "bus", "limit": 20}}
{"name": "subagent", "arguments": {"action": "message", "recipient": "<agent_id>", "text": "Please avoid editing voice_controller.py; I am fixing that file."}}
{"name": "subagent", "arguments": {"action": "inbox"}}
```

From a child, use `agent_message` with `action: "broadcast"` to share findings
with the other active agents and parent, `"bus"` to inspect recent shared
messages without consuming its inbox, `"inbox"` to collect newly received
direct/broadcast messages, or `"send"` for a targeted note. Broadcasts reach
running peers immediately and are also kept in bounded run history. Messages
sent before a child starts are delivered when it attaches. Use named task scopes,
the bus for coordination, `wait_all` for completion, and explicit diff/merge
for worker integration rather than having agents overwrite the same files.

Status contains `agent_id`, `mode`, `status`, `done`, `response`, `error`, and
metrics. Terminal statuses are `completed`, `failed`, `cancelled`, and `timed_out`.
`completed` means the child's runtime finished, not that independent verification
of its claims succeeded. A status-tool success does not imply child success.
Reports are capped at 16,000 characters and may be further trimmed by the context
budget; request focused reports and inspect referenced files for larger artifacts.

The parent must collect each terminal result using `status` or `wait` before
finishing. `list` and `/agents` show status without consuming reports. Children
still running at parent stop, failure, or close receive cancellation requests.
Waiting is bounded to 30 seconds per call and responds to parent cancellation.

Cancellation and deadlines use a shared stop token. Ollama's asynchronous HTTP
transport observes it while waiting for headers or chunks, including compaction.
Managed process waits and polls also observe it. Other provider adapters may wait
for their current network operation to finish. Such tasks remain `cancelling` and
consume capacity until they exit. New runs and workspace switches reject active
old children. Threads and snapshots are not OS security sandboxes.

### Worker integration

Workers use separate temporary filesystem copies, so multiple workers and the
parent can proceed concurrently. Assign relative paths and independent files.
Once a worker finishes:

```json
{"name": "subagent", "arguments": {"action": "diff", "agent_id": "<returned-id>"}}
{"name": "subagent", "arguments": {"action": "merge", "agent_id": "<returned-id>"}}
{"name": "subagent", "arguments": {"action": "discard", "agent_id": "<returned-id>"}}
```

Choose merge or discard for each pending snapshot. `diff` compares worker files
with the parent's current files; it is capped at 16,000 characters and marks
truncation. Large and binary files have summary entries. `merge` checks that each
changed parent file still matches the snapshot's initial hash and permissions.
A conflict applies none of the changes and leaves the snapshot pending. Review the
conflicting paths, then discard/reassign or resolve changes in the parent. There
is no forced overwrite. Merge receipts identify added, updated and deleted files.
I/O failures attempt rollback of files already written; integrations are not
crash-atomic transactions. Coordinate writes by external programs during a merge.

Snapshots accept regular files, up to 100 MiB and 10,000 files. They exclude Git
metadata, `.daena`, virtual environments, `node_modules`, Python/test/linter caches,
and the parent's configured SQLite memory files. Symlinks outside excluded
directories are rejected explicitly. Installed workspace dependencies are not
copied; provision dependencies inside the worker if needed, and verify integrated
changes in the parent environment. Workspace custom skills are passed as explicit
readable skill roots. Snapshots with pending changes persist until merge/discard;
parent cancellation or close discards them.

Status includes `changes` and `integration` (`none`, `pending`, `merged`, or
`discarded`). Collecting a report does not integrate its files. Model completion
requires all results collected and all pending snapshots resolved.

## Configuration

The shipped defaults are:

```json
{
  "skills": {"enabled": true, "include_builtin": true, "directories": []},
  "subagents": {
    "enabled": true,
    "allow_writes": true,
    "max_concurrent": 3,
    "max_tasks": 12,
    "max_iterations": 30,
    "timeout_seconds": 900
  }
}
```

`max_tasks` is the total spawn budget per parent run, including completed tasks;
`max_concurrent` counts running/cancelling tasks. Child iterations are capped by
both the parent limit and `subagents.max_iterations`. Configuration rejects invalid
boolean types and invalid budgets. Maximum configurable limits are 8 concurrent
tasks, 100 total tasks, 300 child iterations, and 3600 seconds per child.

The supplied configuration enables workers. Set `subagents.allow_writes` to
`false` to hide worker mode and reject worker calls. If the configuration omits
this option entirely, the library defaults to read-only delegation. Workers get
the intersection of the parent's allowlist with file, command, process, skill and
web tools. Workspace boundaries remain on, background execution stays off, and
plan/delegation tools are excluded. Tool schemas and execution state advertise
the enabled modes. Reports do not count as parent filesystem mutations; only a
successful merge does.

Turning a feature off hides its tools and rejects direct execution. Explicit
`security.allowed_tools: []` and `security.allowed_commands: []` deny all entries;
omitting those keys selects defaults. Skills cannot change runtime policy.

## Python API

```python
import json
from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType

config = json.load(open("config.json", encoding="utf-8"))
config["workspace"] = {"directory": "/path/to/project"}
with Agent(config) as agent:
    result = agent.act(ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content="Use the debugging skill and inspect the failing parser test.",
    ))
    print(result.response)
    print(agent.last_run_metrics)
```

`Agent(config, llm=...)` accepts a provider-compatible test double. An Agent allows
one active `act()` call; use `steer()` for live guidance or independent Agent
instances for concurrent host sessions. Stop a running task using its supplied
`threading.Event`, let it return, then close the Agent. The `src.*` import paths
are preserved for compatibility with existing integrations.

## Turn completion protocol

The bundled configuration sets `runtime.completion_protocol` to `explicit`.
Assistant text is a progress message; `task_finish` terminates the turn with
`outcome`, `response`, and an `evidence` list. Use `completed` for actual finished
work or a conversational answer, `blocked` for a concrete blocker, and
`needs_input` for a question. The tool is called alone, after pending process and
child resources are resolved. Completion evidence is model-reported; the runtime
does not verify its meaning. Empty evidence is valid when checks do not apply.

Library configurations that omit the setting retain legacy `implicit` completion.
Explicit custom tool allowlists must include `task_finish`; incompatible settings
fail early. No language, task-keyword, or command-name detector decides when work
is complete or when agents should be spawned.
