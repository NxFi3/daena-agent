from __future__ import annotations

import difflib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class EditFile(Tool):
    """Replace an exact string in an existing workspace text file."""

    name = "edit_file"
    action = "modify"

    description = (
        "Replace an exact string in an existing text file. The old string must "
        "match exactly once unless replace_all=true. Preserves encoding and line "
        "endings; returns a short diff and changed line range."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Workspace-relative file path."},
            "old_string": {"type": "string", "description": "Exact existing text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["file_path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._workspace_root: Path | None = None

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def validate(self, arguments: dict[str, Any]) -> bool:
        return (
            isinstance(arguments.get("file_path"), str)
            and bool(arguments["file_path"].strip())
            and isinstance(arguments.get("old_string"), str)
            and bool(arguments["old_string"])
            and isinstance(arguments.get("new_string"), str)
            and type(arguments.get("replace_all", False)) is bool
        )

    @staticmethod
    def _decode(raw: bytes) -> tuple[str, str]:
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw.decode("utf-8-sig"), "utf-8-sig"
        if raw.startswith(b"\xff\xfe"):
            return raw.decode("utf-16"), "utf-16"
        if raw.startswith(b"\xfe\xff"):
            return raw.decode("utf-16"), "utf-16"
        return raw.decode("utf-8"), "utf-8"

    @staticmethod
    def _line_number(text: str, index: int) -> int:
        return text.count("\n", 0, index) + 1

    @staticmethod
    def _closest_lines(text: str, needle: str) -> list[int]:
        first = next((line.strip() for line in needle.splitlines() if line.strip()), "")
        if not first:
            return []
        lines = text.splitlines()
        direct = [index for index, line in enumerate(lines, 1) if first in line]
        if direct:
            return direct[:5]
        scored = sorted(
            (
                (difflib.SequenceMatcher(None, first, line.strip()).ratio(), index)
                for index, line in enumerate(lines, 1)
                if line.strip()
            ),
            reverse=True,
        )
        return [index for score, index in scored[:3] if score >= 0.35]

    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResult:
        if not isinstance(file_path, str) or not file_path.strip():
            return self._error("invalid_argument", "file_path must be non-empty.")
        if not isinstance(old_string, str) or not old_string:
            return self._error("invalid_argument", "old_string must be a non-empty string.")
        if not isinstance(new_string, str):
            return self._error("invalid_argument", "new_string must be a string.")
        if old_string == new_string:
            return self._error("no_change", "old_string and new_string are identical.")
        if type(replace_all) is not bool:
            return self._error("invalid_argument", "replace_all must be a boolean.")
        if self._workspace_root is None:
            return self._error("workspace_unavailable", "Workspace root is not configured.")

        raw_path = Path(file_path).expanduser()
        path = (raw_path if raw_path.is_absolute() else self._workspace_root / raw_path).resolve(strict=False)
        try:
            path.relative_to(self._workspace_root)
        except ValueError:
            return self._error("workspace_boundary", "file_path escapes the configured workspace.")
        if not path.exists():
            return self._error("file_not_found", f"File does not exist: {file_path}")
        if not path.is_file():
            return self._error("invalid_target", f"Path is not a regular file: {file_path}")

        try:
            raw = path.read_bytes()
            original, encoding = self._decode(raw)
        except (OSError, UnicodeError) as exc:
            return self._error("read_error", f"Could not read text file: {exc}")

        count = original.count(old_string)
        if count == 0:
            closest = self._closest_lines(original, old_string)
            lines_text = ", ".join(str(value) for value in closest) if closest else "none found"
            return self._error(
                "old_string_not_found",
                f"old_string was not found in {file_path}. Closest matching line numbers: {lines_text}. Read the current file and retry with exact text.",
                extra={"closest_line_numbers": closest},
            )
        if count > 1 and not replace_all:
            positions = []
            start = 0
            while (index := original.find(old_string, start)) >= 0 and len(positions) < 3:
                positions.append(self._line_number(original, index))
                start = index + len(old_string)
            return self._error(
                "ambiguous_old_string",
                f"old_string matches {count} times in {file_path} (first matching line numbers: {', '.join(map(str, positions))}). Make the old string more specific or set replace_all=true.",
                extra={"match_count": count, "matching_line_numbers": positions},
            )

        updated = original.replace(old_string, new_string) if replace_all else original.replace(old_string, new_string, 1)
        first_pos = original.find(old_string)
        start_line = self._line_number(original, first_pos)
        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        changed_range = {
            "start_line": start_line,
            "end_line": start_line + max(old_lines, new_lines) - 1,
        }
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(file_path),
            tofile=str(file_path),
            n=2,
        ))
        diff_preview = diff[:3000]
        mode = stat.S_IMODE(path.stat().st_mode)
        payload = updated.encode(encoding)
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, path)
            temporary = None
        except OSError as exc:
            return self._error("write_error", f"Could not atomically update {file_path}: {exc}")
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

        return ToolResult(
            success=True,
            name=self.name,
            summary=f"Edited {path.name}; replaced {count if replace_all else 1} occurrence(s), lines {changed_range['start_line']}-{changed_range['end_line']}.",
            content={
                "success": True,
                "path": str(path),
                "match_count": count,
                "changed_line_range": changed_range,
                "diff": diff_preview,
                "diff_truncated": len(diff) > len(diff_preview),
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            },
            effects=[{"action": "modify", "target": str(path)}],
            evidence={"path": str(path), "content": updated[:2000]},
        )

    def _error(
        self,
        error_type: str,
        message: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> ToolResult:
        content: dict[str, Any] = {
            "success": False,
            "error": {"type": error_type, "message": message},
        }
        if extra:
            content.update(extra)
        return ToolResult(
            success=False,
            name=self.name,
            summary=message,
            content=content,
            evidence={"error_type": error_type, "message": message},
        )
