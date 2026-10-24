# Runtime reliability audit

These changes repair runtime and persistence behavior without task-specific
keyword routing or model-output heuristics.

- Process output uses independent read/write file descriptions. Polling cannot
  seek a child's output cursor; UTF-8 characters split across reads remain intact.
  A completed command with unread output returns `output_pending: true` and a
  `process_id`; use `process_poll` to drain the remaining output. Closing the
  conversation releases retained output files and stdin handles.
- Context budgets refresh from the live provider/model generation settings.
  Desktop and browser screenshots also follow the active provider after a CLI
  switch, instead of the initial profile. Text token calibration resets on a
  model/provider change and survives context-size changes within one model.
- `explore` uses a temporary SQLite database, inherits parent cancellation and
  tool restrictions, closes its loop on failure, and reports incomplete runs as
  incomplete. Read-only subagents have the repository search tools; workers also
  have `write_file` and `edit_file`, subject to the parent allowlist. Delegation
  remains unavailable inside child agents.
- File overwrites preserve existing permissions; edits preserve UTF-16 byte
  order and BOM. Large generic tool receipts remain valid JSON. Default recursive
  repository searches skip `.daena`; explicitly selecting that directory still
  permits inspection.
- Opening a Studio conversation restores the original workspace profile. A
  missing workspace is reported before replacing the current conversation.
  Renaming/deleting/restoring a session includes turns older than the 500-entry
  history display. Deletion clears that session's SQLite events, search index,
  and checkpoint as well as its JSON archives; other sessions remain intact.

Regression coverage is in `tests/test_audit_regressions.py`, the Studio tests,
and `tests/computer_use/test_host_context.py`. The complete suite also exercises
real Chromium against local pages, native X11 input, provider serialization,
context continuation, isolated worker integration, and installed-wheel smoke
tests. These checks do not establish the quality of an arbitrary model's plans
or guarantee compatibility with every website, desktop application, or provider
account.
