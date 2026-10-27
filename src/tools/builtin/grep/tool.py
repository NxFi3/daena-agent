from __future__ import annotations

import time
import regex as re
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
    MAX_FILE_BYTES = 4 * 1024 * 1024
    MAX_SNIPPET_CHARS = 2000
    MAX_PATTERN_CHARS = 4096
    MATCH_TIMEOUT_SECONDS = 0.1
    SEARCH_TIMEOUT_SECONDS = 10.0

    description = (
        "Search workspace text with a regular expression. Returns bounded "
        "file:line:snippet matches. Prefer this for locating symbols/usages; "
        "use read_file for the surrounding code. Skips binary, unreadable and "
        "over-4-MiB files, reporting skipped files; each snippet is limited to 2000 characters. "
        "Expensive regexes and searches have explicit time limits."
    )

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression to search for.",
                "maxLength": MAX_PATTERN_CHARS,
            },
            "path": {
                "type": "string",
                "default": ".",
                "description": "Workspace-relative file or directory to search.",
            },
            "include": {
                "type": "string",
                "description": "Optional glob filter such as '*.py' or '**/*.ts'.",
            },
            "case_sensitive": {
                "type": "boolean",
                "default": True,
                "description": "Whether the regex match is case-sensitive.",
            },
            "context_lines": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
                "default": 0,
                "description": "Number of nearby lines to include around each match.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 100,
                "description": "Maximum matches returned.",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def set_stop_event(self, event) -> None:
        self._stop_event = event

    def _cancelled(self):
        event = getattr(self, "_stop_event", None)
        return event is not None and event.is_set()

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
            return self._error(
                "invalid_argument", "pattern must be a non-empty string."
            )
        if len(pattern) > self.MAX_PATTERN_CHARS:
            return self._error("invalid_argument", "pattern exceeds 4096 characters.")
        if self._cancelled():
            return self._error("cancelled", "Search was cancelled.")
        if not isinstance(case_sensitive, bool):
            return self._error("invalid_argument", "case_sensitive must be boolean.")
        if (
            not isinstance(context_lines, int)
            or isinstance(context_lines, bool)
            or not 0 <= context_lines <= 2
        ):
            return self._error(
                "invalid_argument", "context_lines must be between 0 and 2."
            )
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or not 1 <= max_results <= 200
        ):
            return self._error(
                "invalid_argument", "max_results must be between 1 and 200."
            )

        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
            start = workspace_path(root, path)
        except (
            re.error,
            PermissionError,
            OSError,
            RuntimeError,
            RecursionError,
        ) as exc:
            return self._error("invalid_argument", str(exc))

        if not start.exists():
            return self._error("not_found", f"Search path does not exist: {start}")

        matches: list[dict[str, Any]] = []
        files_scanned = 0
        skipped_files = 0
        truncated = False
        deadline = time.monotonic() + self.SEARCH_TIMEOUT_SECONDS

        def interrupted(error_type, message):
            result = self._error(error_type, message)
            result.content.update(
                matches=matches,
                files_scanned=files_scanned,
                skipped_files=skipped_files,
                search_complete=False,
                truncated=True,
            )
            return result

        for file_path in iter_files(
            root, start, stop_event=getattr(self, "_stop_event", None)
        ):
            if self._cancelled():
                return interrupted(
                    "cancelled", "Search was cancelled; results are partial."
                )
            if time.monotonic() >= deadline:
                return interrupted(
                    "search_timeout",
                    "Search time limit reached; narrow path/include and retry. Results are partial.",
                )
            if include:
                rel = relative_path(root, file_path)
                if not (fnmatch(rel, include) or fnmatch(file_path.name, include)):
                    continue

            try:
                with file_path.open("rb") as handle:
                    data = handle.read(self.MAX_FILE_BYTES + 1)
                if len(data) > self.MAX_FILE_BYTES:
                    skipped_files += 1
                    continue
                if b"\x00" in data[:8192]:
                    skipped_files += 1
                    continue
                text = data.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                skipped_files += 1
                continue

            files_scanned += 1
            lines = text.splitlines()
            for line_index, line in enumerate(lines):
                if self._cancelled():
                    return interrupted(
                        "cancelled", "Search was cancelled; results are partial."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return interrupted(
                        "search_timeout",
                        "Search time limit reached; narrow path/include and retry. Results are partial.",
                    )
                try:
                    match = regex.search(
                        line,
                        timeout=min(self.MATCH_TIMEOUT_SECONDS, remaining),
                        concurrent=True,
                    )
                except TimeoutError:
                    return interrupted(
                        "search_timeout",
                        "Regular expression exceeded the search time limit; simplify the pattern. Results are partial.",
                    )
                if match is None:
                    continue

                begin = max(0, line_index - context_lines)
                end = min(len(lines), line_index + context_lines + 1)
                width = self.MAX_SNIPPET_CHARS // (end - begin) - 24
                excerpts = []
                snippet_truncated = False
                for number in range(begin, end):
                    original = lines[number]
                    offset = (
                        max(0, match.start() - width // 2)
                        if number == line_index
                        else 0
                    )
                    excerpt = original[offset : offset + width]
                    clipped = offset > 0 or offset + width < len(original)
                    snippet_truncated |= clipped
                    excerpts.append(
                        f"{number + 1}: "
                        + ("…" if offset else "")
                        + excerpt
                        + ("…" if offset + width < len(original) else "")
                    )
                snippet = "\n".join(excerpts)
                matches.append(
                    {
                        "file": relative_path(root, file_path),
                        "line": line_index + 1,
                        "snippet": snippet[: self.MAX_SNIPPET_CHARS],
                        "snippet_truncated": snippet_truncated
                        or len(snippet) > self.MAX_SNIPPET_CHARS,
                    }
                )
                if len(matches) >= max_results:
                    truncated = True
                    break

            if truncated:
                break

        if self._cancelled():
            return interrupted(
                "cancelled", "Search was cancelled; results are partial."
            )

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "pattern": pattern,
                "path": relative_path(root, start),
                "matches": matches,
                "files_scanned": files_scanned,
                "skipped_files": skipped_files,
                "truncated": truncated,
                "search_complete": not truncated and not skipped_files,
            },
            metadata={},
            summary=(
                f"Found {len(matches)} match(es) across {files_scanned} scanned file(s)."
                + (
                    f" Skipped {skipped_files} unreadable, binary or oversized file(s)."
                    if skipped_files
                    else ""
                )
                + (" Results were truncated." if truncated else "")
            ),
        )

    def _error(self, error_type: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={
                "success": False,
                "error": {"type": error_type, "message": message},
            },
            metadata={},
        )
