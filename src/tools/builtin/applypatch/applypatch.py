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

    def execute(
        self, patch: str | None = None, edits: list[dict[str, Any]] | None = None
    ) -> ToolResult:
        try:
            if (patch is None) == (edits is None):
                raise ValueError("Provide exactly one of `patch` or `edits`.")
            actions = (
                self._parse_patch(patch)
                if patch is not None
                else self._parse_edits(edits)
            )
            planned = self._plan(actions)
            results = self._commit(planned)
            already_applied = sum(bool(r.get("already_applied")) for r in results)
            stats = {
                "operations": len(results),
                "files_changed": len(results) - already_applied,
                "already_applied": already_applied,
                "added": sum(r["added"] for r in results),
                "removed": sum(r["removed"] for r in results),
            }
            return ToolResult(
                success=True,
                name=self.name,
                content={
                    "success": True,
                    "operation": self.name,
                    "summary": (
                        "Patch already applied. Do not retry; continue from the current files."
                        if already_applied == len(results)
                        else f"Applied edits to {len(results) - already_applied} file(s)."
                    ),
                    "files": results,
                    "statistics": stats,
                },
                metadata={"idempotent_replay": already_applied == len(results)},
            )
        except PatchContextError as exc:
            return self._failure("patch_context_mismatch", str(exc), exc.details)
        except (TypeError, ValueError) as exc:
            return self._failure("validation_error", str(exc))
        except OSError as exc:
            return self._failure("filesystem_error", str(exc))
        except Exception as exc:
            return self._failure("application_error", str(exc))

    def _failure(
        self, kind: str, message: str, details: dict | None = None
    ) -> ToolResult:
        content = {
            "success": False,
            "operation": self.name,
            "error": {"type": kind, "message": message},
        }
        if details:
            content.update(details)
            if "content_preview" in details:
                content["content"] = details["content_preview"]
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
            if op not in {
                "replace",
                "insert_before",
                "insert_after",
                "append",
                "create",
                "delete",
            }:
                raise ValueError(f"Edit {idx}: unsupported operation {op!r}.")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"Edit {idx}: invalid path.")
            if op in {"replace", "insert_before", "insert_after"} and (
                not isinstance(edit.get("old"), str)
                or not edit["old"]
                or not isinstance(edit.get("new"), str)
            ):
                raise ValueError(
                    f"Edit {idx}: requires nonempty `old` and string `new`."
                )
            if op in {"append", "create"} and not isinstance(edit.get("new"), str):
                raise ValueError(f"Edit {idx}: requires string `new`.")
            actions.append(dict(edit))
        return actions

    def _parse_patch(self, patch: Any) -> list[dict[str, Any]]:
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("`patch` must be a nonempty string.")
        text = (
            patch.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip("\n")
        )
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
            raise ValueError("Empty patch.")
        if lines[0].strip() == "*** Begin Patch":
            lines.pop(0)
        elif not lines[0].startswith(
            ("*** Add File: ", "*** Update File: ", "*** Delete File: ")
        ):
            raise ValueError("Expected *** Begin Patch or *** Add/Update/Delete File.")
        if lines and lines[-1].strip() == "*** End Patch":
            lines.pop()
        actions: list[dict[str, Any]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip() or line.strip() in {"*** Begin Patch", "*** End Patch"}:
                i += 1
                continue
            m = re.match(r"\*\*\* (Add|Update|Delete) File: (.+)$", line)
            if not m:
                raise ValueError(f"Unexpected patch syntax at line {i + 1}: {line!r}.")
            op, path = m.group(1).lower(), m.group(2).strip()
            i += 1
            if op == "delete":
                actions.append({"path": path, "operation": "delete"})
                continue
            if op == "add":
                content: list[str] = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    content.append(
                        lines[i][1:] if lines[i].startswith("+") else lines[i]
                    )
                    i += 1
                actions.append(
                    {
                        "path": path,
                        "operation": "create",
                        "new": "\n".join(content) + ("\n" if content else ""),
                    }
                )
                continue
            hunks: list[tuple[str, list[str]]] = []
            hint = None
            block: list[str] = []
            while i < len(lines) and (
                not lines[i].startswith("*** ") or lines[i] == "*** End of File"
            ):
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
    def _read(cls, path: Path) -> tuple[bytes | None, str]:
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

    @staticmethod
    def _newline(text: str) -> str:
        return "\r\n" if "\r\n" in text else "\n"

    @staticmethod
    def _match_positions(
        lines: list[str], needle: list[str], start: int = 0
    ) -> tuple[list[int], str]:
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
            found = [
                i
                for i in range(start, len(lines) - len(needle) + 1)
                if [transform(x) for x in lines[i : i + len(needle)]] == wanted
            ]
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
        return value[min(-shift, length) :]

    def _patch_text(
        self, content: str, hunks: list[tuple[str, list[str]]], path: Path
    ) -> tuple[str, int, int]:
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
                if (
                    hint
                    and not hint.startswith("-")
                    and not hint.startswith("+")
                    and all(r.startswith("+") for r in raw)
                ):
                    raw = ["-" + hint, *raw]
                    old = [hint]
                else:
                    raise ValueError(
                        f"Hunk {n} in {path} needs '-' old text, context, "
                        "or a unique full existing line in '@@ old line @@'. "
                        "Blind insertion is not permitted."
                    )
            candidates, mode = self._match_positions(lines, old, cursor)
            # Section hints can choose a region in which the exact old text is unique.
            if len(candidates) > 1 and hint and not hint.startswith("-"):
                anchors = [
                    i for i, line in enumerate(lines) if hint in line and i >= cursor
                ]
                if len(anchors) == 1:
                    anchor = anchors[0]
                    next_section = next(
                        (
                            i
                            for i in range(anchor + 1, len(lines))
                            if lines[i]
                            .lstrip()
                            .startswith(("def ", "class ", "async def "))
                        ),
                        len(lines),
                    )
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
                replacement = [row[1:] for row in raw if row[0] in " +"]
                if not candidates and replacement:
                    existing = [
                        i
                        for i in range(cursor, len(lines) - len(replacement) + 1)
                        if lines[i : i + len(replacement)] == replacement
                    ]
                    if len(existing) == 1:
                        cursor = existing[0] + len(replacement)
                        continue
                reason = (
                    "did not match"
                    if not candidates
                    else f"matched {len(candidates)} locations"
                )
                snippet = "\n".join(old)[:900]
                raise PatchContextError(
                    f"Patch context {reason} in '{path}', hunk {n}. "
                    "Re-read the relevant section and supply a unique patch context.",
                    details={
                        "path": str(path),
                        "hunk": n,
                        "old_snippet": snippet,
                        "matching_locations": len(candidates),
                        "content_preview": self._preview(content),
                        "total_lines": len(lines),
                    },
                )
            pos = candidates[0]
            # For indentation-tolerant matching, shift only new '+' lines, and
            # copy context lines from the actual file (never rewrite their formatting).
            shift = 0
            for k, original in enumerate(old):
                if original.strip():
                    actual = lines[pos + k]
                    if actual.strip() == original.strip():
                        shift = (
                            len(actual)
                            - len(actual.lstrip(" "))
                            - (len(original) - len(original.lstrip(" ")))
                        )
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
                        if mode == "indentation" and not shorthand_replace
                        else value
                    )
                    adds += 1
            lines[pos : pos + len(old)] = replacement
            cursor = pos + len(replacement)
        output = eol.join(lines)
        if trailing and lines:
            output += eol
        return output, adds, removes

    def _simple_text(self, text: str, edit: dict, path: Path) -> tuple[str, int, int]:
        op, old, new = edit["operation"], edit.get("old", ""), edit.get("new", "")
        # A model normally emits LF, even when the source is a Windows CRLF file.
        # Preserve the file's existing newline convention when injecting text.
        if "\r\n" in text:
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
        if op == "append":
            prefix = (
                self._newline(text) if text and not text.endswith(("\n", "\r")) else ""
            )
            return text + prefix + new, len(new.splitlines()), 0
        # Single unique substring matching is easiest for local models.
        starts = [m.start() for m in re.finditer(re.escape(old), text)]
        if len(starts) > 1:
            raise PatchContextError(
                f"Edit target matched {len(starts)} locations in '{path}'. Supply a longer unique `old` string.",
                details={
                    "path": str(path),
                    "old_snippet": old[:900],
                    "matching_locations": len(starts),
                    "content_preview": self._preview(text),
                },
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
                actual = "\n".join(source_lines[p : p + len(old_lines)])
                replacement = new
                if mode == "indentation" and old_lines[0].strip():
                    observed = source_lines[p]
                    shift = (len(observed) - len(observed.lstrip(" "))) - (
                        len(old_lines[0]) - len(old_lines[0].lstrip(" "))
                    )
                    replacement = "\n".join(
                        self._shift_indent(row, shift) for row in new.split("\n")
                    )
                if op == "replace":
                    source_lines[p : p + len(old_lines)] = replacement.split("\n")
                elif op == "insert_before":
                    source_lines[p:p] = replacement.split("\n")
                else:
                    source_lines[p + len(old_lines) : p + len(old_lines)] = (
                        replacement.split("\n")
                    )
                final = "\n".join(source_lines).replace("\n", self._newline(text))
                return (
                    final,
                    len(replacement.splitlines()),
                    len(actual.splitlines()) if op == "replace" else 0,
                )
            reason = "not found" if not starts and not positions else "ambiguous"
            raise PatchContextError(
                f"Edit target {reason} in '{path}'. Read file and supply a longer unique `old` string.",
                details={
                    "path": str(path),
                    "old_snippet": old[:900],
                    "matching_locations": len(starts) if starts else len(positions),
                    "content_preview": self._preview(text),
                    "total_lines": len(text.splitlines()),
                },
            )
        p = starts[0]
        if op == "replace":
            updated = text[:p] + new + text[p + len(old) :]
        elif op == "insert_before":
            updated = text[:p] + new + text[p:]
        else:
            updated = text[: p + len(old)] + new + text[p + len(old) :]
        return (
            updated,
            len(new.splitlines()),
            len(old.splitlines()) if op == "replace" else 0,
        )

    @staticmethod
    def _encode(text: str, original: bytes | None) -> bytes:
        # Keep existing BOM if any.
        raw = text.encode("utf-8")
        return (
            b"\xef\xbb\xbf" + raw
            if original and original.startswith(b"\xef\xbb\xbf")
            else raw
        )

    def _plan(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        planned: dict[Path, dict[str, Any]] = {}
        for action in actions:
            path = self._path(action["path"])
            if path not in planned:
                data, content = self._read(path)
                planned[path] = {
                    "path": path,
                    "before": data,
                    "text": content,
                    "kind": "update",
                    "added": 0,
                    "removed": 0,
                    "mode": (path.stat().st_mode & 0o777) if data is not None else None,
                }
            item = planned[path]
            op = action["operation"]
            if item.get("deleted"):
                raise ValueError(f"Cannot edit after deletion: {path}")
            if op == "create":
                if item["before"] is not None or item.get("created"):
                    if item["text"] == action["new"]:
                        item["kind"] = "add"
                        continue
                    raise ValueError(f"Cannot create existing file: {path}")
                item["text"] = action["new"]
                item["created"] = True
                item["kind"] = "add"
                item["added"] += len(item["text"].splitlines())
            elif op == "delete":
                if item["before"] is None or item.get("created"):
                    if item["before"] is None and not item.get("created"):
                        item.update(kind="delete", deleted=True, already_applied=True)
                        continue
                    raise ValueError(f"Cannot delete missing file: {path}")
                item["removed"] += len(item["text"].splitlines())
                item["deleted"] = True
                item["kind"] = "delete"
            else:
                if item["before"] is None and not item.get("created"):
                    raise ValueError(f"Cannot update missing file: {path}")
                if op == "patch":
                    result, added, removed = self._patch_text(
                        item["text"], action["hunks"], path
                    )
                else:
                    result, added, removed = self._simple_text(
                        item["text"], action, path
                    )
                item["text"] = result
                item["added"] += added
                item["removed"] += removed
        for item in planned.values():
            if not item.get("deleted") and item["before"] is not None:
                item["already_applied"] = (
                    self._encode(item["text"], item["before"]) == item["before"]
                )
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
                raise PatchContextError(
                    f"File changed during patch: {item['path']}. Re-read it."
                )
        completed: list[dict[str, Any]] = []
        try:
            for item in planned:
                path = item["path"]
                if item.get("already_applied"):
                    continue
                if item.get("deleted"):
                    path.unlink()
                else:
                    self._write_atomic(
                        path, self._encode(item["text"], item["before"]), item["mode"]
                    )
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
            result = {
                "path": str(item["path"]),
                "operation": item["kind"],
                "added": item["added"],
                "removed": item["removed"],
                "already_applied": item.get("already_applied", False),
            }
            if not item.get("deleted"):
                content = item["text"]
                result.update(
                    {
                        "content_preview": self._preview(content),
                        "total_lines": len(content.splitlines()),
                        "content": content
                        if len(content) <= self.MAX_CONTENT_PREVIEW_CHARS
                        else None,
                        "content_truncated": len(content)
                        > self.MAX_CONTENT_PREVIEW_CHARS,
                    }
                )
            results.append(result)
        return results

    @classmethod
    def _preview(cls, content: str) -> str:
        if len(content) <= cls.MAX_CONTENT_PREVIEW_CHARS:
            return content
        half = cls.MAX_CONTENT_PREVIEW_CHARS // 2
        return content[:half] + "\n... truncated ...\n" + content[-half:]

    def describe_call(self, arguments: dict[str, Any]) -> dict[str, str]:
        try:
            if isinstance(arguments.get("edits"), list):
                ops = arguments["edits"]
            elif isinstance(arguments.get("patch"), str):
                ops = self._parse_patch(arguments["patch"])
            else:
                return {"action": "modify", "target": ""}
            modes = {op.get("operation") for op in ops}
            action = (
                "create"
                if modes <= {"create"}
                else "delete"
                if modes <= {"delete"}
                else "modify"
            )
            return {
                "action": action,
                "target": ", ".join(str(op.get("path", "")) for op in ops),
            }
        except Exception:
            return {"action": "modify", "target": ""}

    def __repr__(self) -> str:
        return f"<Tool name='{self.name}'>"
