# Daena Agent Runtime

> **The model makes decisions. Daena manages what actually happens.**

Daena is a **local-first, model-agnostic runtime for tool-using LLM agents**.

The project is focused on the part that sits between an LLM and the real environment:

```text
Task
 ↓
Context
 ↓
LLM
 ↓
Tool Call
 ↓
Execution
 ↓
Observation
 ↓
Next Decision
```

The goal is to build a runtime that can keep an agent working through real tasks instead of treating a single LLM response as the whole agent.

Daena is actively under development.

---

## What Daena Is

Daena provides the runtime pieces needed to build a stateful software-engineering agent:

- agent execution loop
- structured context
- short-term memory
- tool registration and dispatch
- workspace state
- tool-result normalization
- local and remote LLM providers
- context compaction
- failure recovery and evidence gates
- duplicate/cycle-action handling
- managed process lifecycle and verification

The project is intentionally being built incrementally from real agent failures and end-to-end experiments.

---

# Architecture

The current architecture is roughly:

```text
                        User Task
                           │
                           ▼
                    ┌─────────────┐
                    │    Agent    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │     Loop    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          Context        Memory       Tools
           System         (STM)       System
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                          LLM
                           │
                           ▼
                      Tool Call
                           │
                           ▼
                        Tool
                           │
                           ▼
                        Result
                           │
                           ▼
                   State / WorkingSet
                           │
                           └──────→ next iteration
```

The runtime keeps the model, execution state, tools, context, and environment as separate concerns.

---

# Agent

The `Agent` is the top-level entry point.

It:

- creates the working directory
- creates an LLM provider
- creates the agent loop
- owns the session id
- sends user tasks to the loop

The actual execution happens inside `Loop`.

---

# Agent Loop

The main execution loop is implemented in:

```text
src/agent/agentloop.py
```

The loop repeatedly:

```text
1. Build context
2. Ask the LLM for the next action
3. Parse tool calls
4. Execute tools
5. Store tool calls/results
6. Update runtime state
7. Continue
```

The loop also contains basic runtime protection against:

- repeated identical successful actions
- repeated failure patterns
- empty model responses
- excessive iteration counts

The current default maximum is configurable through:

```json
{
  "max_agent_iterations": 100
}
```

The loop stores the execution trajectory as structured `ContextEvent` objects.

---

# Agent State

`AgentState` tracks the current runtime status of the agent.

It keeps information such as:

- current status
- current tool
- current action
- current target
- current iteration
- current error
- short progress history

Tool results are converted into normalized runtime effects such as:

```text
Created
Modified
Deleted
Inspected
Ran
Searched
Verified
```

This lets the runtime reason about execution state without hard-coding every concrete tool.

---

# Working Set

`WorkingSet` is the compact state of the current task.

It tracks things such as:

- recently touched artifacts
- file previews
- recent actions
- observations
- facts
- unresolved failures
- verification information

The idea is simple:

```text
Memory = longer-lived information

WorkingSet = information needed for the next decision
```

This prevents the runtime from having to reconstruct everything from raw history on every iteration.

---

# Context System

The context system is responsible for deciding what the model sees.

Main components:

```text
src/context/
├── contextbuilder.py
├── contextservice.py
├── contextwindow.py
├── tokenbudget.py
├── compactor.py
├── compactorprompt.py
└── workingset.py
```

## ContextService

`ContextService` currently retrieves:

1. recent STM events
2. relevant STM events using lexical search

Those events are merged and passed to `ContextBuilder`.

The current configuration uses recent context plus a small number of search results instead of replaying the entire session.

## ContextBuilder

`ContextBuilder` converts runtime events into provider-facing messages.

It preserves structured messages such as:

```text
user
assistant
assistant + tool_calls
tool + tool_call_id
```

It also handles:

- system instructions
- hand-written experience
- runtime information
- large tool-result truncation
- context-budget fitting

The builder intentionally keeps changing execution state in the conversation itself rather than duplicating it into the system prompt.

## Context Compaction

When context becomes too large, `Compactor` can turn older context into a smaller working-state representation.

The compaction prompt is designed to preserve:

- task requirements
- completed work
- current state
- important discoveries
- changes
- errors
- unresolved problems
- next steps

and remove repetitive history and unnecessary raw output.

---

# Memory

Daena's memory system is intentionally simple at the current stage.

The current persistent memory layer is:

```text
Session
   ↓
Context Events
   ↓
SQLite
   ↓
Recent retrieval / lexical search
```

The active implementation is **Short-Term Memory (STM)**.

---

## Short-Term Memory

STM is implemented in:

```text
src/memories/stm/
```

It stores `ContextEvent` objects in SQLite.

Each event contains information such as:

- id
- session id
- role
- type
- content
- priority
- step
- timestamp
- metadata

SQLite also contains an FTS5 index for lexical retrieval.

STM provides two basic access patterns:

```text
get_recent(...)
```

and

```text
search(...)
```

The search path currently uses SQLite FTS5/BM25.

STM itself does not build prompts and does not call the LLM.

---

# Experience

Daena also has a small experience mechanism:

```text
AgentInstruction/experience.md
```

This is currently a **hand-curated experience file** that is injected into the model context on each turn.

Its purpose is to store short, reusable instructions such as:

```text
- known tool usage patterns
- important operational rules
- lessons learned from previous failures
```

The current system is deliberately simple.

The longer-term direction is to make **Experience something Daena can produce from its own execution history**, rather than something that has to be written manually.

The intended flow is:

```text
Task
 ↓
Actions
 ↓
Tool Results
 ↓
Outcome
 ↓
Experience
```

That experience can later become useful to future runs.

The key distinction is:

```text
Memory
= stored information

Experience
= a useful lesson extracted from what happened
```

The experience-learning pipeline is not complete yet.

---

# Retrieval

The repository contains embedding and reranking components:

```text
src/engine/EmbeddingModel.py
src/engine/RerankerModel.py
```

The current active STM retrieval path, however, is based on SQLite FTS5/BM25.

The semantic retrieval and reranking components exist as infrastructure for future memory/retrieval work and are not currently the main STM retrieval path.

The intended direction is:

```text
Query
 ↓
Memory retrieval
 ↓
Relevant information
 ↓
Context
```

rather than inserting the entire history into every request.

---

# Tools

Tools are first-class runtime components.

The tool system contains:

```text
src/tools/
├── Tool.py
├── ToolRegistry.py
├── ToolManager.py
└── ToolDispatcher.py
```

## Tool Registry

`ToolRegistry` discovers builtin tools dynamically from the builtin tool packages.

Each tool provides:

- name
- description
- parameter schema
- execution method
- optional validation
- optional semantic action/target description

---

# Built-in Tools

Current builtin tools include:

### `read_file`

Reads UTF-8 text files.

Supports bounded output and optional line ranges.

```text
read_file
```

### `command_exec`

Runs local commands.

It supports:

- argument-array commands
- working directories
- timeouts
- bounded stdout/stderr
- foreground execution
- background execution

```text
command_exec
```

### `apply_patch`

Modifies files using a **Codex-style patch format**.

Supports:

```text
*** Add File
*** Update File
*** Delete File
```

Update operations use exact context matching.

### `web_search`

Searches the web through the available search backend.

The current implementation can use:

- DuckDuckGo through `ddgs`
- an optional self-hosted SearXNG instance

Search results are treated as untrusted external data.

### `plan`

Maintains the task-local `.daena/plan.md` lifecycle and exposes explicit create/update/delete operations.

### `process_poll`, `process_write`, `process_stop`

Manage long-running commands without treating process startup as task success.

### `web_fetch`

Fetches readable web pages and returns extracted text.

The implementation includes:

- HTML extraction
- text/JSON/XML handling
- paging for long pages
- bounded output
- link extraction
- public-network URL restrictions

---

# Tool Dispatch

Tool calls from providers are normalized into Daena's internal `ToolCall` representation.

The flow is:

```text
Provider Response
      ↓
ToolDispatcher
      ↓
Normalized ToolCall
      ↓
ToolManager
      ↓
ToolRegistry
      ↓
Concrete Tool
      ↓
ToolResult
```

The runtime therefore does not need each provider to use exactly the same tool-call representation.

---

# Tool Results

All tools return a common `ToolResult`.

A result contains:

```text
success
name
content
metadata
summary
evidence
effects
```

This gives the runtime a normalized representation of what happened.

For example, a file operation can produce an effect such as:

```text
modified → path/to/file.py
```

while command execution can produce evidence such as:

```text
exit code
stdout
stderr
duration
```

The goal is to let higher-level runtime components reason about outcomes without knowing the internal implementation of every tool.

---

# Providers

Daena separates the runtime from the underlying model provider.

Current provider implementations include:

```text
Ollama
OpenRouter
```

The provider abstraction lives under:

```text
src/engine/providers/
```

The runtime uses `LLMInput` and `LLMResult` as its provider-facing data structures.

This allows different providers to have different native message/tool formats while keeping the rest of the runtime provider-neutral.

---

# Ollama

The current local development configuration uses Ollama.

The provider converts Daena's canonical tool-call representation into the format expected by the Ollama SDK.

It also normalizes:

- model responses
- tool calls
- thinking/reasoning
- token usage

The active default configuration is defined in `config.json`.

---

# OpenRouter

OpenRouter is supported as another provider.

Its provider implementation normalizes OpenAI-compatible tool calls and messages into the same internal `LLMResult` representation used by the rest of Daena.

---

# Security

The repository contains a defense-in-depth security layer:

```text
src/security/
├── Policy.py
├── Sandbox.py
└── securityService.py
```

The security layer enforces:

- tool allowlists
- workspace boundaries
- blocked executables
- inline shell/interpreter restrictions
- background-process permissions

Background execution is denied by default. A host application can approve one exact background command for the current `Agent` session with `approve_background_command(...)`. The approval is scoped to the resolved workspace and exact argv, so changing the command or workspace requires a new approval.

For explicit trusted environments, `security.allow_background=true` can allow background commands without per-command approval. Session approvals are cleared when the `Agent` instance is discarded.

The security layer is a defense-in-depth control, not an operating-system sandbox. The long-term goal is to move important execution restrictions into the runtime instead of relying only on model instructions.

---

# Current Project Direction

Daena is currently focused on making the basic runtime reliable before adding large higher-level systems.

The immediate direction is:

```text
Reliable execution
        ↓
Simple persistent memory
        ↓
Experience extraction
        ↓
Better future decisions
```

The memory design is intentionally not overly complicated yet.

The current target is closer to:

```text
              ┌──────────────┐
              │ Main Memory  │
              └──────┬───────┘
                     │
                     ▼
                  Context
                     │
                     ▼
                    LLM
                     │
                 execution
                     │
                     ▼
                 Experience
                     │
                     └──────→ future use
```

The exact experience-learning mechanism is still being designed and evaluated.

---

# Research Direction

The main research idea behind Daena is **experience-driven agent improvement**.

Instead of only storing raw history, the runtime should eventually be able to identify useful lessons from previous execution.

For example:

```text
Task
 ↓
Tool failure
 ↓
Recovery
 ↓
Successful strategy
 ↓
Experience
```

A later task can then retrieve that experience:

```text
New Task
 ↓
Relevant Experience
 ↓
Better Action
```

The key research question is:

> Can an agent become more effective on recurring task classes by learning reusable strategies from its own past execution trajectories?

The goal is to measure this through real task execution rather than only through static prompt evaluation.

Possible measurements include:

- task success
- tool-call count
- repeated errors
- recovery success
- execution time
- token usage
- performance before and after relevant experience

---


This is an end-to-end generated project used to test the agent on a real software task.

The project contains:

- provider data models
- report generation
- provider comparison
- source information
- pytest tests
- CLI entry point

It is useful as a small benchmark for checking whether the agent can:

```text
Research
 ↓
Understand
 ↓
Create files
 ↓
Run tests
 ↓
Recover from mistakes
 ↓
Finish the task
```

More benchmark tasks will be added as the runtime evolves.

---

# Repository Structure

```text
daena-agent/
│
├── AgentInstruction/
│   ├── systeminstruction.md
│   └── experience.md
│
├── ai_provider_report_agent_test/
│
├── src/
│   ├── agent/
│   │   ├── agent.py
│   │   ├── agentloop.py
│   │   ├── agentstate.py
│   │   ├── planstate.py
│   │   ├── planprogress.py
│   │   └── toolguard.py
│   │
│   ├── context/
│   │   ├── compactor.py
│   │   ├── compactorprompt.py
│   │   ├── contextbuilder.py
│   │   ├── contextservice.py
│   │   ├── contextwindow.py
│   │   ├── tokenbudget.py
│   │   └── workingset.py
│   │
│   ├── engine/
│   │   ├── EmbeddingModel.py
│   │   ├── LlmProviderManager.py
│   │   ├── RerankerModel.py
│   │   └── providers/
│   │
│   ├── memories/
│   │   └── stm/
│   │       ├── STM.py
│   │       └── stmdatabase.py
│   │
│   ├── models/
│   │
│   ├── security/
│   │
│   ├── tools/
│   │   ├── Tool.py
│   │   ├── ToolDispatcher.py
│   │   ├── ToolManager.py
│   │   ├── ToolRegistry.py
│   │   ├── ToolManager.py
│   │   └── builtin/
│   │       ├── applypatch/
│   │       ├── command_exec/
│   │       ├── plan/
│   │       ├── process_poll/
│   │       ├── process_stop/
│   │       └── process_write/
│   │
│   └── utils/
│
├── tests/
│   ├── test_loop_e2e.py
│   ├── test_recovery_runtime.py
│   ├── test_process_runtime.py
│   └── ...
│
├── config.json
├── Agent.md
└── README.md
```

---

# Roadmap

## Runtime

- [x] Structured agent loop
- [x] Structured tool calls/results
- [x] Tool discovery and dispatch
- [x] Agent state
- [x] Working set
- [x] Context budgeting
- [x] Context compaction foundation
- [x] Stronger task completion and verification gates
- [x] Evidence-driven failure recovery
- [ ] Checkpoint/resume

## Memory

- [x] Persistent STM
- [x] Recent retrieval
- [x] Lexical retrieval
- [x] Hand-curated experience
- [ ] Main long-term memory layer
- [ ] Automatic experience extraction
- [ ] Experience retrieval/use
- [ ] Memory lifecycle improvements

## Tools

- [x] Local filesystem tools
- [x] Command execution
- [x] Codex-style patching
- [x] Web search
- [x] Web fetch
- [ ] More integrations
- [ ] MCP support

## Evaluation

- [x] Real end-to-end agent task
- [x] Core runtime regression suite
- [ ] Repeatable benchmark suite
- [ ] Learning-gain evaluation
- [ ] Failure-recovery evaluation
- [ ] Experience-ablation experiments

---

# Design Principles

### Runtime over prompt

Important execution guarantees should live in the runtime.

### Memory over raw history

Store useful information instead of replaying everything.

### Experience over repetition

Past failures should eventually become useful lessons.

### Verify real outcomes

A successful tool call does not automatically mean the task is complete.

### Build from real failures

The architecture should be driven by what actually breaks during agent execution.

### Keep the system modular

Agent, context, memory, tools, and providers should remain replaceable and independently testable.

---

# Status

**🚧 Active Development**

Daena is not a finished autonomous-agent framework.

The current project is a working foundation for experimenting with:

```text
LLMs
+
Tools
+
Context
+
State
+
Memory
+
Experience
```

with the longer-term goal of building an agent that can **learn useful strategies from its own experience instead of repeating the same mistakes forever**.
