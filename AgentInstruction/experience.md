<!--
This file holds durable, hand-curated lessons and preferences for the
agent. It is re-read on every turn (no restart needed) and shown to the
model inside an <experience> section, right after the base system
instruction.

Keep entries short, general and actionable — patterns the agent should
follow, not a log of specific past tasks.
-->

- apply_patch: "*** Add File: <path>" is for brand-new files; every content
  line must be prefixed with "+" and there are no hunks / "@@" markers.
  "*** Update File: <path>" is only for editing a file that already exists
  on disk; its hunks use " " / "+" / "-" prefixes, never a full-file
  replacement. Do not mix the two formats.
