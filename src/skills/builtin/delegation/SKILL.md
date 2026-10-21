---
name: delegation
description: Split a complex task into bounded subagent assignments, collect evidence, and integrate independently verified results.
---

Use subagent only when it is available and the work benefits from separate contexts.

1. Define independent, concrete assignments. Include the goal, necessary background, workspace paths, constraints and expected output in each task; children do not inherit conversation history.
2. Prefer read_only for research, inspection and review. These children cannot run commands or edit files. The parent can continue its own work while they run.
3. Use worker only when host configuration permits it. Workers edit isolated copies and can run concurrently. Assign independent files and relative paths; only a subset of the parent's tools is available.
4. Save each returned agent_id. Use wait with a meaningful bounded wait_ms, or status, to collect each result. Do not spawn duplicate replacements merely because a task takes time.
5. Treat child reports as evidence requiring review. Check failures, cancellations and timeouts explicitly. Review worker edits with diff, explicitly merge or discard each pending snapshot, and verify the integrated result before reporting success. A merge conflict leaves parent files unchanged; resolve it before continuing.
6. Cancel unnecessary tasks and wait until they stop. Cancellation is cooperative; cancelling is not a terminal status. Respect concurrency, task and iteration budgets.
7. Integrate results into one answer addressing the original request. Do not claim completion while children are still running or results remain uncollected.
