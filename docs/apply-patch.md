# Targeted file edits

`apply_patch` accepts exactly one of `patch` or `edits`. Both routes validate and
plan every operation before writing any files, and use the same workspace policy.

For a small replacement, structured edits avoid diff syntax:

```json
{"edits":[{"path":"drivers/framebuffer.c","operation":"replace","old":"return 0;","new":"return 1;"}]}
```

Choose enough original text to identify exactly one location. An optional
`expected_sha256` on each edit must match the original file bytes, including its
BOM and line endings. Obtain it from `read_file`; re-read if the version changes.
Supported operations are `replace`, `insert_before`, `insert_after`, `append`,
`create`, and `delete`. The first three require nonempty `old`; every operation
except `delete` requires `new`. `append`, `create`, and `delete` do not accept `old`.

The patch route uses explicit line prefixes:

```text
*** Begin Patch
*** Update File: drivers/framebuffer.c
@@
-    return 0;
+    return 1;
 }
*** End Patch
```

Here the closing brace starts with a space because it is unchanged context.
A bare brace cannot safely be interpreted as context, a deletion, or an addition.
Malformed or incomplete patches and context-only hunks are rejected before any
writes. Correct the reported hunk or submit a local structured replacement;
rewriting the entire file is unnecessary.

UTF-8 BOM, existing file permissions, and CRLF line endings are preserved. Sources
are checked again before commit. A filesystem failure triggers best-effort
rollback; concurrent changes are preserved and incomplete rollback is explicitly
reported. This is not a cross-process filesystem transaction: inspect any reported
rollback errors before retrying.
