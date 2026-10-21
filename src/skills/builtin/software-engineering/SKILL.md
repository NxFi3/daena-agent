---
name: software-engineering
description: Implement or repair software in an existing repository, including setup, focused edits, and verification.
---

1. Inspect workspace inventory, repository instructions, manifests and version-control status. Preserve unrelated changes.
2. Identify the user's observable goal and the smallest coherent implementation. Use a plan for multi-step changes.
3. Read relevant code and its callers before editing. Follow existing conventions and public interfaces.
4. Use apply_patch for reviewable edits. Avoid broad rewrites, new frameworks, or dependencies without a concrete need.
5. Run focused checks first, then the project's required suite. Diagnose failures from evidence; do not repeat an unchanged failing command.
6. Review the diff for accidental files, credentials, incomplete work, and compatibility regressions.
7. Report what changed, the checks actually run, and any remaining limitations. A passing tool invocation alone does not establish correctness.

Use command_exec with argv arrays and an explicit workspace. If a command returns a live process ID, poll or stop it before continuing. Respect runtime permissions.
