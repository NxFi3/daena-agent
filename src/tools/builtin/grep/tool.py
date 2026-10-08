from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.workspace_utils import iter_files, relative_path, workspace_path


class Grep(Tool):
    name = "grep"
    action = "search"
    allow_same_revision_repeat = True

    description = (
        "Search workspace text with a regular expression. Returns bounded "
        "file:line:snippet matches. Prefer this for locating symbols/usages; "
        "use read_file for the surrounding code."
    )

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "default": ".", "description": "Workspace-relative file or directory to search."},
            "include": {"type": "string", "description": "Optional glob filter such as '*.py' or '**/*.ts'."},
            "case_sensitive": {"type": "boolean", "default": True, "description": "Whether the regex match is case-sensitive."},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 2, "default": 0, "description": "Number of nearby lines to include around each match."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100, "description": "Maximum matches returned."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
        case_sensitive: bool = True,
        context_lines: int = 0,
        max_results: int = 100,
    ) -> ToolResult:
        root = getattr(self, "_workspace_root", None)
        if root is None:
            return self._error("workspace_missing", "No workspace is configured.")
        if not isinstance(pattern, str) or not pattern:
            return self._error("invalid_argument", "pattern must be a non-empty string.")
        if not isinstance(case_sensitive, bool):
            return self._error("invalid_argument", "case_sensitive must be boolean.")
        if not isinstance(context_lines, int) or isinstance(context_lines, bool) or not 0 <= context_lines <= 2:
            return self._error("invalid_argument", "context_lines must be between 0 and 2.")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= 200:
            return self._error("invalid_argument", "max_results must be between 1 and 200.")

        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
            start = workspace_path(root, path)
        except (re.error, PermissionError, OSError) as exc:
            return self._error("invalid_argument", str(exc))

        if not start.exists():
            return self._error("not_found", f"Search path does not exist: {start}")

        matches: list[dict[str, Any]] = []
        files_scanned = 0
        truncated = False

        for file_path in iter_files(root, start):
            if include:
                rel = relative_path(root, file_path)
                if not (fnmatch(rel, include) or fnmatch(file_path.name, include)):
                    continue

            try:
                data = file_path.read_bytes()
                if b"\x00" in data[:8192]:
                    continue
                text = data.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            files_scanned += 1
            lines = text.splitlines()
            for line_index, line in enumerate(lines):
                if regex.search(line) is None:
                    continue

                begin = max(0, line_index - context_lines)
                end = min(len(lines), line_index + context_lines + 1)
                matches.append({
                    "file": relative_path(root, file_path),
                    "line": line_index + 1,
                    "snippet": "\n".join(
                        f"{number + 1}: {lines[number]}"
                        for number in range(begin, end)
                    ),
                })
                if len(matches) >= max_results:
                    truncated = True
                    break

            if truncated:
                break

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "pattern": pattern,
                "path": relative_path(root, start),
                "matches": matches,
                "files_scanned": files_scanned,
                "truncated": truncated,
            },
            metadata={},
            summary=(
                f"Found {len(matches)} match(es) across {files_scanned} scanned file(s)."
                + (" Results were truncated." if truncated else "")
            ),
        )

    def _error(self, error_type: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={"success": False, "error": {"type": error_type, "message": message}},
            metadata={},
        )
