# Context and future memory adapters

Daena builds each provider request from separate sources:

| Source | Purpose | Lifetime |
| --- | --- | --- |
| System and active runtime instructions | Application behavior and temporary modes such as Voice | Rebuilt on each model request; never truncated to fit |
| Current user messages and steering | The task and explicit corrections | Preserved throughout the active turn |
| Execution state and workspace plan | Current task state, artifact inventory and progress | Refreshed from current runtime/workspace |
| Tool call/result groups | Observed evidence and action outcomes | Kept together; older groups can be summarized or removed |
| Historical checkpoint | Compacted earlier conversation | Persisted in STM, invalidated when the session is cleared/deleted |
| Context contributions | Optional, selected retrieval or memory evidence | One request only; bounded and removed first when the budget is tight |
| Attached observations | Current images and observation data | Budgeted explicitly by the provider context builder |

`ContextService.get_context(..., contributions=records)` and
`ContextBuilder.build_context(..., contributions=records)` accept typed records:

```python
from src.context.contributions import ContextContribution

records = [
    ContextContribution(
        id="project-choice-17",
        source="memory://project/example/17",
        kind="project_note",
        content="The user previously selected a local-only deployment.",
    )
]
```

A future memory adapter can retrieve relevant records and pass this list alongside
ordinary context arguments. The adapter owns selection, permission checks, expiry
and source validation. This API performs no retrieval, model calls, consolidation
or persistent memory writes. It does not execute instructions found in records.

Records are rendered as explicitly untrusted data in a user-role context message,
with identifier, source and truncation status. Source labels describe provenance;
they are not proof that the content is correct. Current user instructions and fresh
observations take precedence. Records cannot supply a system role, tool call or
fake tool result. The caller's relevance order is preserved, duplicate source/ID
pairs are removed, and rendering inspects at most 64 records and includes at most
32. The text budget is at most 10% of the effective prompt allowance and 6,000
characters, and includes provenance/wrapper overhead. If the complete prompt is
still too large, this optional block is dropped before real conversation groups.
Contributions are neither put into pending history nor added to compaction/STM.

The legacy `learned_experience` parameter is a host-owned system section; do not
use it for arbitrary retrieved records. Use contributions for future memory data.
An impossible prompt budget raises `ContextBudgetError` with accounting details;
it must not silently remove pieces of system instructions or active Voice modes.

STM uses one reentrant transaction lock per connection, including history reads,
FTS updates, checkpoints, file-read ledgers and close. Agent and Studio threads
can share an STM instance without interleaving parts of separate transactions.
SQLite WAL still coordinates separate processes/connections. Direct use of the
raw `connection` attribute bypasses this contract and is intended only for
migrations and controlled diagnostics while the database is otherwise idle.
