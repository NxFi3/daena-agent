# File evidence and context continuity

`read_file` returns an exact contiguous source slice. It no longer joins the
beginning and end of a file while silently omitting its middle. Existing
`start_line`, `end_line`, and `max_output_chars` arguments remain supported.

Every successful read includes:

- `file_sha256`: SHA-256 of the UTF-8 source bytes read at that time.
- `source_start_offset`, `source_end_offset`: a half-open range in Unicode
  characters, not bytes. CRLF and Unicode text remain intact across pages.
- `requested_end_offset`: the end of the requested range.
- `next_read`: arguments for continuing from the first omitted character,
  including `expected_sha256`. It is null when that range is complete.
- `read_id`: identity of this read receipt.

Copy `next_read` into another `read_file` call when the omitted text is needed.
Its absolute `start_offset` must not be combined with `start_line`. A changed
source rejects the old continuation with `file_changed`; inspect the relevant
range of the new version instead. `end_line` describes the requested line cap;
source offsets describe the exact returned text, including partial long lines.

## Transport budgeting

The model-facing context layer may have a smaller budget than the tool output.
It reduces file text to a prefix, updates the delivered range and continuation,
and marks `transport_truncated`. It also replaces summaries that would otherwise
claim a complete read. If even metadata cannot fit, the receipt explicitly says
details were omitted. Canonical tool results and files on disk are not rewritten.
Older JSON receipts stay valid JSON during reduction. Duplicate-read comparison
uses source version and range; legacy receipts also compare their visible text.

## Persistent delivery history

After a confirmed model response, the context service records the file versions
and character intervals in that request. Adjacent and overlapping intervals are
merged; gaps remain gaps. Building a prompt alone does not record delivery.

SQLite's `context_file_reads` table stores this derived metadata by session,
path, and source hash. The table is created automatically in existing histories.
The bounded `file_read_history` execution-state catalog survives recent-event
window advancement and process restarts. Its ranges describe earlier delivery,
not source text still present in the current prompt or current disk contents.
The catalog reports omitted versions. Specific text can be recovered through
`read_file`; the full canonical event history remains in STM for context search.
Clearing a session clears its delivery catalog. Deleting an event invalidates
that session's derived catalog as well as its checkpoint. A persistent session
revision invalidates live prompt caches after either operation, including when
another connection performs the deletion.

## Sliding history and checkpoints

Recent retrieval extends a cutoff inside a tool batch to include its owning
assistant call. The ordinary `get_recent` API retains its original strict limit;
the context service uses the complete-exchange retrieval path.

Uncheckpointed exchanges remain available while the retrieval window advances.
That advancement triggers a checkpoint even below the token threshold. Stable
event IDs prevent already-covered receipts from being replayed after lexical
recall or restart. If semantic compaction of tool exchanges returns no summary, a bounded,
explicit receipt-only checkpoint is used; cancellation still propagates.

These mechanisms do not prohibit rereads, count reads to decide task progress,
select project-specific filenames, or infer whether a coding task is complete.
They make evidence loss visible and provide exact recovery coordinates. Model
planning quality still needs separate evaluation against real coding tasks.
