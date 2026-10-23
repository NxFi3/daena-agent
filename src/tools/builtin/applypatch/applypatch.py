"""Daena apply_patch: safe, model-friendly text editing.

Supports classic Codex-style patches and simpler structured `edits`.
Requires unique matches; never guesses a target. Plans edits in memory first,
checks source snapshots, then commits with best-effort rollback.
"""
from __future__ import annotations

import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


@dataclass
class PatchOperation:
    operation: str
    path: str
    hunks: list[list[str]]
    content: str | None = None
    # Hint supplied by @@ (line number or section title), one per hunk.
    hints: list[str] | None = None


class PatchContextError(ValueError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class ApplyPatch(Tool):
    name = "apply_patch"
    action = "modify"
    MAX_CONTENT_PREVIEW_CHARS = 4000
    MAX_FILE_BYTES = 10 * 1024 * 1024

    description = (
        "Modify UTF-8 text files safely with `patch` (REQUIRED). "
        "Use Codex format: *** Begin Patch, *** Update File: path, @@, "
        "-EXACT_OLD_LINE, +NEW_LINE, *** End Patch. "
        "The '-' line is essential: NEVER put old text only in the @@ header. "
        "Convenience: for a UNIQUE existing full line you may use "
        "@@ EXACT_OLD_LINE @@ followed by +NEW_LINE as a replacement. "
        "Use one patch call with multiple Update File blocks. "
        "Ambiguous or stale edits are rejected rather than applied to the wrong place. "
        "Never use `edits` in tool calls: this deployment requires `patch`."
    )
    # Keep the external tool contract backward-compatible: Daena's dispatch/validation
    # layers may hard-require `patch` even if this tool advertises `edits`.
    # Direct Python callers can still use execute(edits=[...]).
    parameters = {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": (
                    "Required. Codex text patch. EXAMPLE: "
                    "*** Begin Patch\n*** Update File: main.c\n@@\n"
                    "-old_statement;\n+new_statement;\n*** End Patch. "
                    "Use full, exact old lines prefixed with '-'. "
                    "Convenience syntax: @@ old_statement; @@ followed by "
                    "+new_statement; replaces that uniquely matched old line."
                ),
            },
        },
        "required": ["patch"],
        "additionalProperties": False,
    }

    def execute(self, patch: str | None = None, edits: list[dict[str, Any]] | None = None) -> ToolResult:
        try:
            if (patch is None) == (edits is None):
                raise ValueError("Provide exactly one of `patch` or `edits`.")
            actions = self._parse_patch(patch) if patch is not None else self._parse_edits(edits)
            planned = self._plan(actions)
            results = self._commit(planned)
            stats = {
                "files_changed": len(results),
                "added": sum(r["added"] for r in results),
                "removed": sum(r["removed"] for r in results),
            }
            return ToolResult(
                success=True, name=self.name,
                content={"success": True, "operation": self.name,
                         "summary": f"Applied edits to {len(results)} file(s).",
                         "files": results, "statistics": stats}, metadata={},
            )
        except PatchContextError as exc:
            return self._failure("patch_context_mismatch", str(exc), exc.details)
        except (TypeError, ValueError) as exc:
            return self._failure("validation_error", str(exc))
        except OSError as exc:
            return self._failure("filesystem_error", str(exc))
        except Exception as exc:
            return self._failure("application_error", str(exc))

<<<<<<< HEAD
        already_applied = sum(
            1 for item in results if item.get("already_applied", False)
        )
        files_changed = len(results) - already_applied
        added = sum(item["added"] for item in results)
        removed = sum(item["removed"] for item in results)

        ops_summary = []
        for item in results:
            filename = Path(item["path"]).name
            suffix = " (already applied)" if item.get("already_applied") else ""
            ops_summary.append(f"{item['operation']}:{filename}{suffix}")

        idempotent_replay = bool(results) and already_applied == len(results)
        if idempotent_replay:
            summary = (
                "Patch is already fully applied; no files changed. "
                f"Operations: [{', '.join(ops_summary)}]. "
                "Do not retry this patch; continue with the next action."
            )
        else:
            replay_note = (
                f" {already_applied} operation(s) were already satisfied."
                if already_applied
                else ""
            )
            summary = (
                f"Applied patch to {files_changed} file(s): "
                f"{added} addition(s), {removed} deletion(s)."
                f"{replay_note} Operations: [{', '.join(ops_summary)}]. "
                "Use the returned file preview when available; "
                "re-read large files only when necessary."
            )

        return self._success(
            summary=summary,
            files=results,
            statistics={
                "operations": len(results),
                "files_changed": files_changed,
                "already_applied": already_applied,
                "added": added,
                "removed": removed,
            },
            idempotent_replay=idempotent_replay,
        )

    def _parse_patch(self, patch: str) -> list[PatchOperation]:
        normalized = self._normalize_patch_text(patch)

        lines = normalized.split("\n")

        if not lines:
            raise ValueError("Patch is empty.")

        operations = self._parse_operations(lines)

        if not operations:
            raise ValueError(
                "Patch contains no Add File, Update File, or Delete File operation."
            )

        return operations

    def _normalize_patch_text(self, patch: str) -> str:
        """
        Normalize common LLM formatting mistakes without changing the
        actual Codex-style operation syntax.
        """

        text = patch.replace("\r\n", "\n").replace("\r", "\n")

        # Remove BOM if present.
        text = text.lstrip("\ufeff")

        # Remove surrounding whitespace/newlines first.
        text = text.strip()

        if not text:
            raise ValueError("Patch is empty.")
=======
    def _failure(self, kind: str, message: str, details: dict | None = None) -> ToolResult:
        content = {"success": False, "operation": self.name,
                   "error": {"type": kind, "message": message}}
        if details:
            content.update(details)
        return ToolResult(success=False, name=self.name, content=content, metadata={})

    def _parse_edits(self, edits: Any) -> list[dict[str, Any]]:
        if not isinstance(edits, list) or not edits:
            raise ValueError("`edits` must be a nonempty list.")
        actions: list[dict[str, Any]] = []
        for idx, edit in enumerate(edits, 1):
            if not isinstance(edit, dict):
                raise ValueError(f"Edit {idx} must be an object.")
            if set(edit) - {"path", "operation", "old", "new"}:
                raise ValueError(f"Edit {idx} has unknown fields.")
            op, path = edit.get("operation"), edit.get("path")
            if op not in {"replace", "insert_before", "insert_after", "append", "create", "delete"}:
                raise ValueError(f"Edit {idx}: unsupported operation {op!r}.")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"Edit {idx}: invalid path.")
            if op in {"replace", "insert_before", "insert_after"} and (
                not isinstance(edit.get("old"), str) or not edit["old"]
                or not isinstance(edit.get("new"), str)
            ):
                raise ValueError(f"Edit {idx}: requires nonempty `old` and string `new`.")
            if op in {"append", "create"} and not isinstance(edit.get("new"), str):
                raise ValueError(f"Edit {idx}: requires string `new`.")
            actions.append(dict(edit))
        return actions
>>>>>>> 8904218 (checkpoint)

    def _parse_patch(self, patch: Any) -> list[dict[str, Any]]:
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("`patch` must be a nonempty string.")
        text = patch.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip("\n")
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines.pop(0)
            if lines and lines[-1].strip() == "```":
                lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
<<<<<<< HEAD
            raise ValueError("Patch is empty.")

        begin_count = sum(1 for line in lines if line.strip() == self._BEGIN)
        end_count = sum(1 for line in lines if line.strip() == self._END)

        # Some models concatenate multiple complete patch blocks into one tool
        # argument. The old parser stopped at the first End Patch and silently
        # ignored everything after it, which produced partial multi-file edits.
        # Collapse those wrappers into one transaction and parse every operation.
        if begin_count > 1 or end_count > 1:
            payload = [
                line
                for line in lines
                if line.strip() not in {self._BEGIN, self._END}
            ]
            lines = [self._BEGIN, *payload, self._END]
        else:
            # Be tolerant if the model omitted Begin Patch.
            if lines[0].strip() != self._BEGIN:
                if self._is_operation_line(lines[0]):
                    lines.insert(0, self._BEGIN)
                else:
                    raise ValueError(
                        f"Patch must start with '{self._BEGIN}' "
                        f"or directly with a file operation."
                    )

            # Be tolerant if the model omitted End Patch.
            end_positions = [
                index for index, line in enumerate(lines)
                if line.strip() == self._END
            ]
            if not end_positions:
                lines.append(self._END)
            elif end_positions[-1] != len(lines) - 1:
                raise ValueError(f"'{self._END}' must be the final patch marker.")

        return "\n".join(lines)

    def _parse_operations(
        self,
        lines: list[str],
    ) -> list[PatchOperation]:

        if lines[0].strip() != self._BEGIN:
            raise ValueError(f"Patch must start with '{self._BEGIN}'.")

        if lines[-1].strip() != self._END:
            raise ValueError(f"Patch must end with '{self._END}'.")

        operations: list[PatchOperation] = []

        i = 1

        while i < len(lines) - 1:
=======
            raise ValueError("Empty patch.")
        if lines[0].strip() == "*** Begin Patch":
            lines.pop(0)
        elif not lines[0].startswith(("*** Add File: ", "*** Update File: ", "*** Delete File: ")):
            raise ValueError("Expected *** Begin Patch or *** Add/Update/Delete File.")
        if lines and lines[-1].strip() == "*** End Patch":
            lines.pop()
        actions: list[dict[str, Any]] = []
        i = 0
        while i < len(lines):
>>>>>>> 8904218 (checkpoint)
            line = lines[i]
            if not line.strip():
                i += 1
                continue
<<<<<<< HEAD

            if line.startswith(self._UPDATE):
                path = line[len(self._UPDATE) :].strip()

                if not path:
                    raise ValueError("Update File path cannot be empty.")

                i += 1

                hunks, i = self._parse_hunks(
                    lines=lines,
                    start_index=i,
                )

                if not hunks:
                    raise ValueError(
                        f"Update File '{path}' has no hunks. "
                        "Start the update content with '@@'."
                    )

                operations.append(
                    PatchOperation(
                        operation="update",
                        path=path,
                        hunks=hunks,
                    )
                )

                continue

            if line.startswith(self._ADD):
                path = line[len(self._ADD) :].strip()

                if not path:
                    raise ValueError("Add File path cannot be empty.")

                i += 1

                content_lines: list[str] = []

                while i < len(lines) - 1:
                    current = lines[i]

                    if self._is_operation_line(current):
                        break

                    if current.startswith("@@"):
                        raise ValueError(
                            f"Add File '{path}' cannot contain hunks. "
                            "Prefix file content lines with '+'."
                        )

                    # Add File blocks are unambiguous once the
                    # parser has identified the operation boundary. Accept the
                    # canonical '+' prefix, but also tolerate raw content lines
                    # from models that omit the prefix for an entire file block.
                    if current == "":
                        content_lines.append("")
                    elif current.startswith("+"):
                        content_lines.append(current[1:])
                    else:
                        content_lines.append(current)

                    i += 1

                content = self._join_added_lines(content_lines)

                operations.append(
                    PatchOperation(
                        operation="add",
                        path=path,
                        hunks=[],
                        content=content,
                    )
                )

                continue

            if line.startswith(self._DELETE):
                path = line[len(self._DELETE) :].strip()

                if not path:
                    raise ValueError("Delete File path cannot be empty.")

                operations.append(
                    PatchOperation(
                        operation="delete",
                        path=path,
                        hunks=[],
                    )
                )

                i += 1
                continue

            if line.strip() == self._END:
                raise ValueError(
                    f"Unexpected '{self._END}' before the final patch marker."
                )

            raise ValueError(
                f"Unexpected patch line: {line!r}. "
                f"Expected Add File, Update File, Delete File, "
                f"or '{self._END}'."
            )

        return operations

    def _parse_hunks(
        self,
        lines: list[str],
        start_index: int,
    ) -> tuple[list[list[str]], int]:

        hunks: list[list[str]] = []

        current_hunk: list[str] | None = None

        i = start_index

        while i < len(lines) - 1:
            line = lines[i]

            if self._is_operation_line(line):
                break

            if line.strip() == self._END:
                break

            if line.startswith("@@"):
                if current_hunk is not None:
                    if not current_hunk:
                        raise ValueError("Empty patch hunk.")

                    hunks.append(current_hunk)

                current_hunk = []

                i += 1
                continue

            if current_hunk is None:
                raise ValueError("Update File content must begin with '@@'.")

            if line == "":
                current_hunk.append(" ")
                i += 1
                continue

            prefix = line[0]

            if prefix not in (" ", "+", "-"):
                raise ValueError(
                    f"Invalid hunk line: {line!r}. "
                    "Expected ' ' for context, '-' for removal, "
                    "or '+' for addition."
                )

            current_hunk.append(line)

=======
            m = re.match(r"\*\*\* (Add|Update|Delete) File: (.+)$", line)
            if not m:
                raise ValueError(f"Unexpected patch syntax at line {i + 1}: {line!r}.")
            op, path = m.group(1).lower(), m.group(2).strip()
>>>>>>> 8904218 (checkpoint)
            i += 1
            if op == "delete":
                actions.append({"path": path, "operation": "delete"})
                continue
            if op == "add":
                content: list[str] = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    content.append(lines[i][1:] if lines[i].startswith("+") else lines[i])
                    i += 1
                actions.append({"path": path, "operation": "create",
                                "new": "\n".join(content) + ("\n" if content else "")})
                continue
            hunks: list[tuple[str, list[str]]] = []
            hint = None
            block: list[str] = []
            while i < len(lines) and (not lines[i].startswith("*** ") or lines[i] == "*** End of File"):
                line = lines[i]
                i += 1
                if line.startswith("@@"):
                    if hint is not None:
                        if not block:
                            raise ValueError(f"Empty hunk in {path}.")
                        hunks.append((hint, block))
                    hint, block = line[2:].strip().removesuffix("@@").strip(), []
                elif line == "*** End of File":
                    continue
                elif hint is None:
                    if line.strip():
                        raise ValueError(f"Update for {path} needs an @@ hunk header.")
                elif not line:
                    block.append(" ")  # a raw blank context line
                elif line[0] in " +-":
                    block.append(line)
                elif line == "\\ No newline at end of file":
                    pass
                else:
                    raise ValueError(f"Malformed update line in {path}: {line!r}")
            if hint is not None:
                if not block:
                    raise ValueError(f"Empty hunk in {path}.")
                hunks.append((hint, block))
            if not hunks:
                raise ValueError(f"Update for {path} has no @@ hunks.")
            actions.append({"path": path, "operation": "patch", "hunks": hunks})
        if not actions:
            raise ValueError("Patch contains no file operations.")
        return actions

    @staticmethod
    def _path(value: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("File path cannot be empty.")
        path = Path(value).expanduser()
        if path.is_symlink():
            raise ValueError(f"Refusing to edit symlink: {path}")
        return path

    @classmethod
<<<<<<< HEAD
    def _is_operation_line(
        cls,
        line: str,
    ) -> bool:
        return (
            line.startswith(cls._UPDATE)
            or line.startswith(cls._ADD)
            or line.startswith(cls._DELETE)
        )

    def _build_plan(
        self,
        operations: list[PatchOperation],
    ) -> list[dict[str, Any]]:

        seen_paths: set[str] = set()

        plan: list[dict[str, Any]] = []

        for operation in operations:
            normalized_path = self._normalize_path(operation.path)

            if normalized_path in seen_paths:
                raise ValueError(
                    f"Duplicate operation for path " f"'{operation.path}'."
                )

            seen_paths.add(normalized_path)

            path = Path(normalized_path)

            if operation.operation == "add":
                plan.append(
                    self._plan_add(
                        path=path,
                        operation=operation,
                    )
                )

            elif operation.operation == "update":
                plan.append(
                    self._plan_update(
                        path=path,
                        operation=operation,
                    )
                )

            elif operation.operation == "delete":
                plan.append(
                    self._plan_delete(
                        path=path,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported patch operation " f"'{operation.operation}'."
                )

        return plan

    def _plan_add(
        self,
        path: Path,
        operation: PatchOperation,
    ) -> dict[str, Any]:
        content = operation.content or ""

        if path.exists():
            if not path.is_file():
                raise ValueError(f"Cannot add '{path}': path is not a file.")
            try:
                current = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Cannot add '{path}': existing file is not UTF-8 text."
                ) from exc
            if current == content:
                return {
                    "operation": "add",
                    "path": path,
                    "content": current,
                    "added": 0,
                    "removed": 0,
                    "already_applied": True,
                }
            raise ValueError(
                f"Cannot add '{path}': file already exists with different content."
            )

        return {
            "operation": "add",
            "path": path,
            "content": content,
            "added": self._count_lines(content),
            "removed": 0,
            "already_applied": False,
        }

    def _plan_update(
        self,
        path: Path,
        operation: PatchOperation,
    ) -> dict[str, Any]:

=======
    def _read(cls, path: Path) -> tuple[bytes | None, str]:
>>>>>>> 8904218 (checkpoint)
        if not path.exists():
            return None, ""
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Not a regular file: {path}")
        if path.stat().st_size > cls.MAX_FILE_BYTES:
            raise ValueError(f"File exceeds {cls.MAX_FILE_BYTES} bytes: {path}")
        data = path.read_bytes()
        if b"\0" in data:
            raise ValueError(f"Cannot edit binary file: {path}")
        try:
            return data, data.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError(f"File is not UTF-8: {path}") from exc

<<<<<<< HEAD
        try:
            updated, added, removed, replayed_hunks = self._apply_hunks(
                original=original,
                hunks=operation.hunks,
                path=str(path),
            )
        except ValueError as exc:
            message = str(exc)
            if (
                "Patch context did not match" in message
                or "Ambiguous patch" in message
            ):
                try:
                    current_content = path.read_text(
                        encoding="utf-8",
                    )
                except (OSError, UnicodeDecodeError):
                    current_content = ""

                raise PatchContextError(
                    message,
                    details={
                        "path": str(path),
                        "type": "file",
                        "content": self._bounded_preview(current_content),
                        "total_lines": self._count_lines(current_content),
                    },
                ) from exc
            raise

        already_applied = (
            updated == original
            and bool(operation.hunks)
            and replayed_hunks == len(operation.hunks)
        )

        return {
            "operation": "update",
            "path": path,
            "content": updated,
            "added": added,
            "removed": removed,
            "already_applied": already_applied,
        }

    def _plan_delete(
        self,
        path: Path,
    ) -> dict[str, Any]:

        if not path.exists():
            return {
                "operation": "delete",
                "path": path,
                "content": None,
                "added": 0,
                "removed": 0,
                "already_applied": True,
            }

        if not path.is_file():
            raise ValueError(f"Cannot delete '{path}': path is not a file.")

        try:
            file_content = path.read_text(encoding="utf-8")
            removed_lines = self._count_lines(file_content)
        except (UnicodeDecodeError, OSError):
            removed_lines = 0

        return {
            "operation": "delete",
            "path": path,
            "content": None,
            "added": 0,
            "removed": removed_lines,
            "already_applied": False,
        }

    def _apply_hunks(
        self,
        original: str,
        hunks: list[list[str]],
        path: str,
    ) -> tuple[str, int, int, int]:

        newline = self._detect_newline(original)
        file_lines = original.splitlines(keepends=True)
        additions = 0
        removals = 0
        replayed_hunks = 0
        search_from = 0

        for hunk_index, hunk in enumerate(hunks, start=1):
            old_lines: list[str] = []
            new_lines: list[str] = []

            for raw_line in hunk:
                prefix = raw_line[0]
                value = raw_line[1:]

                if prefix == " ":
                    old_lines.append(value)
                    new_lines.append(value)
                elif prefix == "-":
                    old_lines.append(value)
                elif prefix == "+":
                    new_lines.append(value)
=======
    @staticmethod
    def _newline(text: str) -> str:
        return "\r\n" if "\r\n" in text else "\n"

    @staticmethod
    def _match_positions(lines: list[str], needle: list[str], start: int = 0) -> tuple[list[int], str]:
        # Exact, trailing-space tolerant, then indentation tolerant. Each level must be unique.
        # Never fuzzy-match words, edit distance or semantic similarity.
        if not needle:
            return [], "none"
        for mode, transform in (
            ("exact", lambda x: x),
            ("trailing_whitespace", lambda x: x.rstrip()),
            ("indentation", lambda x: x.strip()),
        ):
            wanted = [transform(x) for x in needle]
            found = [i for i in range(start, len(lines) - len(needle) + 1)
                     if [transform(x) for x in lines[i:i + len(needle)]] == wanted]
            if found:
                return found, mode
        return [], "none"

    @staticmethod
    def _shift_indent(value: str, shift: int) -> str:
        if not value.strip() or shift == 0:
            return value
        if shift > 0:
            return " " * shift + value
        length = len(value) - len(value.lstrip(" "))
        return value[min(-shift, length):]

    def _patch_text(self, content: str, hunks: list[tuple[str, list[str]]], path: Path) -> tuple[str, int, int]:
        eol = self._newline(content)
        trailing = content.endswith(("\n", "\r"))
        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if trailing:
            lines.pop()
        cursor = 0
        adds = removes = 0
        for n, (hint, raw) in enumerate(hunks, 1):
            old = [r[1:] for r in raw if r[0] in " -"]
            # Model-friendly replacement shorthand:
            # @@ exact full old line @@
            # +new full line
            # Interprets the header as the removed line ONLY when the hunk
            # contains additions, no context/removals, and has a textual hint.
            # An ordinary @@ section header with a normal -/context hunk is
            # unaffected. Never use line number hints for blind insertion.
            shorthand_replace = not old
            if not old:
                if (hint and not hint.startswith("-") and not hint.startswith("+")
                        and all(r.startswith("+") for r in raw)):
                    raw = ["-" + hint, *raw]
                    old = [hint]
>>>>>>> 8904218 (checkpoint)
                else:
                    raise ValueError(
                        f"Hunk {n} in {path} needs '-' old text, context, "
                        "or a unique full existing line in '@@ old line @@'. "
                        "Blind insertion is not permitted."
                    )
            candidates, mode = self._match_positions(lines, old, cursor)
            # Section hints can choose a region in which the exact old text is unique.
            if len(candidates) > 1 and hint and not hint.startswith("-"):
                anchors = [i for i, line in enumerate(lines) if hint in line and i >= cursor]
                if len(anchors) == 1:
                    anchor = anchors[0]
                    next_section = next((i for i in range(anchor + 1, len(lines))
                                         if lines[i].lstrip().startswith(("def ", "class ", "async def "))), len(lines))
                    nearby = [pos for pos in candidates if anchor <= pos < next_section]
                    if len(nearby) == 1:
                        candidates = nearby
            # Unified @@ -N... hints are only used when the indexed position itself matches.
            if len(candidates) > 1:
                number = re.match(r"^-(\d+)(?:,\d+)?\s+\+\d+", hint)
                if number:
                    position = int(number.group(1)) - 1
                    if position in candidates:
                        candidates = [position]
            if len(candidates) != 1:
                reason = "did not match" if not candidates else f"matched {len(candidates)} locations"
                snippet = "\n".join(old)[:900]
                raise PatchContextError(
                    f"Patch context {reason} in '{path}', hunk {n}. "
                    "Use `edits` with a unique old string or re-read the relevant section.",
                    details={"path": str(path), "hunk": n, "old_snippet": snippet,
                             "matching_locations": len(candidates),
                             "content_preview": self._preview(content), "total_lines": len(lines)},
                )
            pos = candidates[0]
            # For indentation-tolerant matching, shift only new '+' lines, and
            # copy context lines from the actual file (never rewrite their formatting).
            shift = 0
            for k, original in enumerate(old):
                if original.strip():
                    actual = lines[pos + k]
                    if actual.strip() == original.strip():
                        shift = len(actual) - len(actual.lstrip(" ")) - (len(original) - len(original.lstrip(" ")))
                    break
            replacement: list[str] = []
            old_idx = 0
            for row in raw:
                prefix, value = row[0], row[1:]
                if prefix == " ":
                    replacement.append(lines[pos + old_idx])
                    old_idx += 1
                elif prefix == "-":
                    old_idx += 1
                    removes += 1
                elif prefix == "+":
                    replacement.append(
                        self._shift_indent(value, shift)
                        if mode == "indentation" and not shorthand_replace else value
                    )
                    adds += 1
            lines[pos:pos + len(old)] = replacement
            cursor = pos + len(replacement)
        output = eol.join(lines)
        if trailing and lines:
            output += eol
        return output, adds, removes

<<<<<<< HEAD
            old_normalized = [self._remove_newline(line) for line in old_lines]
            new_normalized = [self._remove_newline(line) for line in new_lines]

            matches = self._find_matches(
                file_lines=file_lines,
                target=old_normalized,
            )
            forward_matches = [
                position for position in matches if position >= search_from
            ]

            if not forward_matches:
                # Idempotent replay: if the desired post-patch sequence is
                # already present exactly once, this hunk succeeded earlier.
                # Treat it as satisfied instead of turning a successful prior
                # mutation into a misleading context-mismatch failure.
                replay_matches = (
                    self._find_matches(
                        file_lines=file_lines,
                        target=new_normalized,
                    )
                    if new_normalized
                    else []
                )
                replay_forward = [
                    position
                    for position in replay_matches
                    if position >= search_from
                ]
                if len(replay_forward) == 1:
                    position = replay_forward[0]
                    search_from = position + len(new_normalized)
                    replayed_hunks += 1
                    continue

                raise ValueError(
                    f"Patch context did not match in '{path}', "
                    f"hunk {hunk_index}. "
                    "The file may have changed or the context may be incorrect. "
                    "Re-read the relevant file section and retry."
                )

            if len(forward_matches) > 1:
                raise ValueError(
                    f"Ambiguous patch in '{path}', "
                    f"hunk {hunk_index}: "
                    f"context matched {len(forward_matches)} times. "
                    "Use more specific context."
                )

            position = forward_matches[0]
            replacement = [
                self._format_new_line(line=line, newline=newline)
                for line in new_lines
            ]
            file_lines[position : position + len(old_normalized)] = replacement
            search_from = position + len(replacement)

            additions += sum(1 for line in hunk if line.startswith("+"))
            removals += sum(1 for line in hunk if line.startswith("-"))

        return (
            "".join(file_lines),
            additions,
            removals,
            replayed_hunks,
        )

    def _find_matches(
        self,
        file_lines: list[str],
        target: list[str],
    ) -> list[int]:

        if not target:
            return []

        normalized_file = [self._remove_newline(line) for line in file_lines]

        def find(
            haystack: list[str],
            needle: list[str],
        ) -> list[int]:
            matches: list[int] = []
            limit = len(haystack) - len(needle) + 1
            if limit <= 0:
                return matches
            for position in range(limit):
                if haystack[position : position + len(needle)] == needle:
                    matches.append(position)
            return matches

        # Exact matching is preferred. Only when it produces no match do we
        # tolerate harmless trailing-whitespace drift in the model's context.
        exact = find(normalized_file, target)
        if exact:
            return exact

        whitespace_tolerant_file = [line.rstrip() for line in normalized_file]
        whitespace_tolerant_target = [line.rstrip() for line in target]
        return find(whitespace_tolerant_file, whitespace_tolerant_target)

    def _apply_plan(
        self,
        plan: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply a fully validated plan atomically.

        Every target is snapshotted before the first mutation. If any filesystem
        operation fails, previously changed targets are restored so a multi-file
        patch cannot leave the workspace half-applied.
        """
        results: list[dict[str, Any]] = []
        snapshots: list[tuple[Path, bool, bytes | None, int | None]] = []

        for item in plan:
            path: Path = item["path"]
            if path.exists() and path.is_file():
                snapshots.append(
                    (path, True, path.read_bytes(), path.stat().st_mode)
                )
            else:
                snapshots.append((path, False, None, None))

        try:
            for item in plan:
                path: Path = item["path"]
                operation = item["operation"]
                already_applied = bool(item.get("already_applied", False))

                if already_applied:
                    if operation in {"add", "update"}:
                        content = item.get("content") or ""
                        results.append(
                            self._build_file_result(
                                operation=operation,
                                path=path,
                                content=content,
                                added=0,
                                removed=0,
                                already_applied=True,
                            )
                        )
                    else:
                        results.append(
                            {
                                "path": str(path),
                                "operation": operation,
                                "added": 0,
                                "removed": 0,
                                "already_applied": True,
                            }
                        )
                    continue

                if operation == "add":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(item["content"], encoding="utf-8")
                    content = item.get("content") or ""
                    results.append(
                        self._build_file_result(
                            operation="add",
                            path=path,
                            content=content,
                            added=item["added"],
                            removed=item["removed"],
                            already_applied=False,
                        )
                    )

                elif operation == "update":
                    path.write_text(item["content"], encoding="utf-8")
                    content = item.get("content") or ""
                    results.append(
                        self._build_file_result(
                            operation="update",
                            path=path,
                            content=content,
                            added=item["added"],
                            removed=item["removed"],
                            already_applied=False,
                        )
                    )

                elif operation == "delete":
                    path.unlink()
                    results.append(
                        {
                            "path": str(path),
                            "operation": "delete",
                            "added": item["added"],
                            "removed": item["removed"],
                            "already_applied": False,
                        }
                    )

                else:
                    raise RuntimeError(
                        f"Unknown planned operation: {operation}"
                    )

            return results

        except Exception:
            for path, existed, original_bytes, mode in reversed(snapshots):
                try:
                    if existed:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(original_bytes or b"")
                        if mode is not None:
                            path.chmod(mode)
                    elif path.exists() and path.is_file():
                        path.unlink()
                except OSError:
                    # Preserve the original exception. The caller reports the
                    # filesystem failure; rollback is best-effort if the disk
                    # itself has become unavailable.
                    pass
            raise

    def _build_file_result(
        self,
        *,
        operation: str,
        path: Path,
        content: str,
        added: int,
        removed: int,
        already_applied: bool = False,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {
            "path": str(path),
            "operation": operation,
            "added": added,
            "removed": removed,
            "already_applied": already_applied,
            "content_preview": self._bounded_preview(content),
            "total_lines": self._count_lines(content),
        }

        # Do not send huge complete files back into the model context.
        #
        # Small files keep the previous convenient behavior.
        if len(content) <= self.MAX_CONTENT_PREVIEW_CHARS:
            result["content"] = content
            result["content_truncated"] = False
        else:
            result["content"] = None
            result["content_truncated"] = True

        return result

    @staticmethod
    def _normalize_path(
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError("File path cannot be empty.")

        return str(Path(value))

    @staticmethod
    def _detect_newline(
        content: str,
    ) -> str:

        if "\r\n" in content:
            return "\r\n"

        return "\n"

    @staticmethod
    def _remove_newline(
        value: str,
    ) -> str:

        if value.endswith("\r\n"):
            return value[:-2]

        if value.endswith("\n"):
            return value[:-1]

        if value.endswith("\r"):
            return value[:-1]

        return value

    @staticmethod
    def _format_new_line(
        line: str,
        newline: str,
    ) -> str:

        return ApplyPatch._remove_newline(line) + newline

    @staticmethod
    def _join_added_lines(
        lines: list[str],
    ) -> str:

        if not lines:
            return ""

        return "\n".join(lines) + "\n"

    @staticmethod
    def _count_lines(
        content: str,
    ) -> int:

        if not content:
            return 0

        return len(content.splitlines())

=======
    def _simple_text(self, text: str, edit: dict, path: Path) -> tuple[str, int, int]:
        op, old, new = edit["operation"], edit.get("old", ""), edit.get("new", "")
        # A model normally emits LF, even when the source is a Windows CRLF file.
        # Preserve the file's existing newline convention when injecting text.
        if "\r\n" in text:
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
        if op == "append":
            prefix = self._newline(text) if text and not text.endswith(("\n", "\r")) else ""
            return text + prefix + new, len(new.splitlines()), 0
        # Single unique substring matching is easiest for local models.
        starts = [m.start() for m in re.finditer(re.escape(old), text)]
        if len(starts) > 1:
            raise PatchContextError(
                f"Edit target matched {len(starts)} locations in '{path}'. Supply a longer unique `old` string.",
                details={"path": str(path), "old_snippet": old[:900],
                         "matching_locations": len(starts), "content_preview": self._preview(text)},
            )
        if not starts:
            # Fall back to whole-line whitespace/indentation tolerant matching.
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            source_lines = normalized.split("\n")
            old_lines = old.replace("\r\n", "\n").split("\n")
            if old_lines[-1] == "":
                old_lines.pop()
            positions, mode = self._match_positions(source_lines, old_lines)
            if len(positions) == 1 and old_lines:
                p = positions[0]
                actual = "\n".join(source_lines[p:p + len(old_lines)])
                replacement = new
                if mode == "indentation" and old_lines[0].strip():
                    observed = source_lines[p]
                    shift = (len(observed) - len(observed.lstrip(" "))) - (len(old_lines[0]) - len(old_lines[0].lstrip(" ")))
                    replacement = "\n".join(self._shift_indent(row, shift) for row in new.split("\n"))
                if op == "replace":
                    source_lines[p:p + len(old_lines)] = replacement.split("\n")
                elif op == "insert_before":
                    source_lines[p:p] = replacement.split("\n")
                else:
                    source_lines[p + len(old_lines):p + len(old_lines)] = replacement.split("\n")
                final = "\n".join(source_lines).replace("\n", self._newline(text))
                return final, len(replacement.splitlines()), len(actual.splitlines()) if op == "replace" else 0
            reason = "not found" if not starts and not positions else "ambiguous"
            raise PatchContextError(
                f"Edit target {reason} in '{path}'. Read file and supply a longer unique `old` string.",
                details={"path": str(path), "old_snippet": old[:900],
                         "matching_locations": len(starts) if starts else len(positions),
                         "content_preview": self._preview(text), "total_lines": len(text.splitlines())},
            )
        p = starts[0]
        if op == "replace":
            updated = text[:p] + new + text[p + len(old):]
        elif op == "insert_before":
            updated = text[:p] + new + text[p:]
        else:
            updated = text[:p + len(old)] + new + text[p + len(old):]
        return updated, len(new.splitlines()), len(old.splitlines()) if op == "replace" else 0

    @staticmethod
    def _encode(text: str, original: bytes | None) -> bytes:
        # Keep existing BOM if any.
        raw = text.encode("utf-8")
        return b"\xef\xbb\xbf" + raw if original and original.startswith(b"\xef\xbb\xbf") else raw

    def _plan(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        planned: dict[Path, dict[str, Any]] = {}
        for action in actions:
            path = self._path(action["path"])
            if path not in planned:
                data, content = self._read(path)
                planned[path] = {"path": path, "before": data, "text": content,
                                 "kind": "update", "added": 0, "removed": 0,
                                 "mode": (path.stat().st_mode & 0o777) if data is not None else None}
            item = planned[path]
            op = action["operation"]
            if item.get("deleted"):
                raise ValueError(f"Cannot edit after deletion: {path}")
            if op == "create":
                if item["before"] is not None or item.get("created"):
                    raise ValueError(f"Cannot create existing file: {path}")
                item["text"] = action["new"]
                item["created"] = True
                item["kind"] = "add"
                item["added"] += len(item["text"].splitlines())
            elif op == "delete":
                if item["before"] is None or item.get("created"):
                    raise ValueError(f"Cannot delete missing file: {path}")
                item["removed"] += len(item["text"].splitlines())
                item["deleted"] = True
                item["kind"] = "delete"
            else:
                if item["before"] is None and not item.get("created"):
                    raise ValueError(f"Cannot update missing file: {path}")
                if op == "patch":
                    result, added, removed = self._patch_text(item["text"], action["hunks"], path)
                else:
                    result, added, removed = self._simple_text(item["text"], action, path)
                item["text"] = result
                item["added"] += added
                item["removed"] += removed
        return list(planned.values())

    @staticmethod
    def _write_atomic(path: Path, contents: bytes, mode: int | None):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".daena-patch-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            if mode is not None:
                os.chmod(tmp, mode)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _commit(self, planned: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Recheck every source before modifying any files.
        for item in planned:
            current, _ = self._read(item["path"])
            if current != item["before"]:
                raise PatchContextError(f"File changed during patch: {item['path']}. Re-read it.")
        completed: list[dict[str, Any]] = []
        try:
            for item in planned:
                path = item["path"]
                if item.get("deleted"):
                    path.unlink()
                else:
                    self._write_atomic(path, self._encode(item["text"], item["before"]), item["mode"])
                completed.append(item)
        except Exception:
            # Best-effort rollback of previous successful changes.
            for item in reversed(completed):
                path = item["path"]
                try:
                    if item["before"] is None:
                        path.unlink(missing_ok=True)
                    else:
                        self._write_atomic(path, item["before"], item["mode"])
                except OSError:
                    pass
            raise
        results = []
        for item in planned:
            result = {"path": str(item["path"]), "operation": item["kind"],
                      "added": item["added"], "removed": item["removed"]}
            if not item.get("deleted"):
                content = item["text"]
                result.update({"content_preview": self._preview(content),
                               "total_lines": len(content.splitlines()),
                               "content": content if len(content) <= self.MAX_CONTENT_PREVIEW_CHARS else None,
                               "content_truncated": len(content) > self.MAX_CONTENT_PREVIEW_CHARS})
            results.append(result)
        return results

>>>>>>> 8904218 (checkpoint)
    @classmethod
    def _preview(cls, content: str) -> str:
        if len(content) <= cls.MAX_CONTENT_PREVIEW_CHARS:
            return content
        half = cls.MAX_CONTENT_PREVIEW_CHARS // 2
        return content[:half] + "\n... truncated ...\n" + content[-half:]

<<<<<<< HEAD
        head = int(limit * 0.70)

        tail = limit - head

        omitted = len(content) - head - tail

        return (
            content[:head].rstrip()
            + "\n... "
            + str(omitted)
            + " chars omitted ...\n"
            + content[-tail:].lstrip()
        )

    def _success(
        self,
        *,
        summary: str,
        files: list[dict[str, Any]],
        statistics: dict[str, int],
        idempotent_replay: bool = False,
    ) -> ToolResult:

        metadata: dict[str, Any] = {
            "idempotent_replay": idempotent_replay,
        }
        if idempotent_replay:
            metadata["recovery_hint"] = (
                "This exact patch is already reflected in the workspace. "
                "Do not retry it; continue with the next distinct action."
            )

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "operation": self.name,
                "summary": summary,
                "files": files,
                "statistics": statistics,
            },
            metadata=metadata,
        )

    def _failure(
        self,
        *,
        error_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ToolResult:

        content: dict[str, Any] = {
            "success": False,
            "operation": self.name,
            "error": {
                "type": error_type,
                "message": message,
            },
        }

        if isinstance(details, dict):
            content.update(details)

        return ToolResult(
            success=False,
            name=self.name,
            content=content,
            metadata={},
        )

    def describe_call(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, str]:

        patch = arguments.get(
            "patch",
            "",
        )

        if not isinstance(
            patch,
            str,
        ):
            return {
                "action": "modify",
                "target": "",
            }

=======
    def describe_call(self, arguments: dict[str, Any]) -> dict[str, str]:
>>>>>>> 8904218 (checkpoint)
        try:
            if isinstance(arguments.get("edits"), list):
                ops = arguments["edits"]
            elif isinstance(arguments.get("patch"), str):
                ops = self._parse_patch(arguments["patch"])
            else:
                return {"action": "modify", "target": ""}
            modes = {op.get("operation") for op in ops}
            action = "create" if modes <= {"create"} else "delete" if modes <= {"delete"} else "modify"
            return {"action": action, "target": ", ".join(str(op.get("path", "")) for op in ops)}
        except Exception:
            return {"action": "modify", "target": ""}

    def __repr__(self) -> str:
        return f"<Tool name='{self.name}'>"
