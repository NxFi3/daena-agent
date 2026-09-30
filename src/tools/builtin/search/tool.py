from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class Search(Tool):
    """Search text/regex across files inside the active workspace."""

    name = "search"
    action = "search"

    DEFAULT_MAX_RESULTS = 50
    MAX_RESULTS = 100
    DEFAULT_CONTEXT_LINES = 1
    MAX_CONTEXT_LINES = 3
    MAX_OUTPUT_CHARS = 7500

    _SKIP_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
    }

    description = (
        "Search text or a regular expression across files in the workspace. "
        "Use this to locate symbols, functions, error messages, or references "
        "instead of repeatedly reading the same file. Returns matching file "
        "paths, line numbers, and short context. This is local workspace search; "
        "use web_search for internet search and read_file for reading a known file."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text or regular expression to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search. Defaults to the workspace root.",
                "default": ".",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return.",
                "default": DEFAULT_MAX_RESULTS,
                "minimum": 1,
                "maximum": MAX_RESULTS,
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of surrounding lines to include for each match.",
                "default": DEFAULT_CONTEXT_LINES,
                "minimum": 0,
                "maximum": MAX_CONTEXT_LINES,
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional comma-separated glob patterns such as '*.py,*.md'.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(
        self,
        query: str,
        path: str = ".",
        max_results: int = DEFAULT_MAX_RESULTS,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        file_pattern: str | None = None,
    ) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return self._error("invalid_argument", "query is required.")

        if not isinstance(path, str) or not path.strip():
            return self._error("invalid_argument", "path must be a non-empty string.")

        if not isinstance(max_results, int) or isinstance(max_results, bool):
            return self._error("invalid_argument", "max_results must be an integer.")
        if not 1 <= max_results <= self.MAX_RESULTS:
            return self._error(
                "invalid_argument",
                f"max_results must be between 1 and {self.MAX_RESULTS}.",
            )

        if not isinstance(context_lines, int) or isinstance(context_lines, bool):
            return self._error("invalid_argument", "context_lines must be an integer.")
        if not 0 <= context_lines <= self.MAX_CONTEXT_LINES:
            return self._error(
                "invalid_argument",
                f"context_lines must be between 0 and {self.MAX_CONTEXT_LINES}.",
            )

        patterns = self._patterns(file_pattern)
        try:
            matcher = re.compile(query, re.IGNORECASE | re.MULTILINE)
        except re.error:
            matcher = re.compile(re.escape(query), re.IGNORECASE | re.MULTILINE)

        root = Path(path).expanduser().resolve(strict=False)
        if not root.exists():
            return self._error("not_found", f"Search path does not exist: {root}")

        files = self._iter_files(root, patterns)
        matches: list[dict[str, Any]] = []

        for file_path in files:
            if len(matches) >= max_results:
                break

            try:
                raw = file_path.read_bytes()
            except (OSError, PermissionError):
                continue

            if b"\x00" in raw[:8192]:
                continue

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue

            lines = text.splitlines()
            for index, line in enumerate(lines):
                if not matcher.search(line):
                    continue

                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)

                matches.append(
                    {
                        "path": str(file_path),
                        "line": index + 1,
                        "text": line.strip(),
                        "context_start": start + 1,
                        "context_end": end,
                        "context": lines[start:end],
                    }
                )

                if len(matches) >= max_results:
                    break

        content_parts: list[str] = []
        for item in matches:
            header = f"{item['path']}:{item['line']}"
            context = item["context"]
            lines_out = [header]
            for offset, value in enumerate(
                context,
                start=item["context_start"],
            ):
                marker = ">" if offset == item["line"] else " "
                lines_out.append(f"{marker}{offset:5d} | {value}")
            content_parts.append("\n".join(lines_out))

            if len("\n\n".join(content_parts)) >= self.MAX_OUTPUT_CHARS:
                break

        rendered = "\n\n".join(content_parts)
        truncated = len(matches) > len(content_parts)

        if len(rendered) > self.MAX_OUTPUT_CHARS:
            rendered = rendered[: self.MAX_OUTPUT_CHARS] + "\n...[output truncated]"

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "query": query,
                "path": str(root),
                "result_count": len(matches),
                "truncated": truncated,
                "content": rendered
                or f'No matches for "{query}" under {root}.',
            },
            metadata={
                "match_count": len(matches),
            },
        )

    @classmethod
    def _iter_files(
        cls,
        root: Path,
        patterns: list[str],
    ) -> list[Path]:
        if root.is_file():
            return [root] if cls._matches_pattern(root, patterns) else []

        found: list[Path] = []
        for current, dirs, filenames in os.walk(root, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs
                if d not in cls._SKIP_DIRS and not d.startswith(".daena")
            )

            for filename in sorted(filenames):
                candidate = Path(current) / filename
                if cls._matches_pattern(candidate, patterns):
                    found.append(candidate)

        return found

    @staticmethod
    def _matches_pattern(path: Path, patterns: list[str]) -> bool:
        if not patterns:
            return True
        return any(path.match(pattern) for pattern in patterns)

    @staticmethod
    def _patterns(value: str | None) -> list[str]:
        if not value:
            return []
        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    def _error(self, error_type: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={
                "success": False,
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
            metadata={},
        )

    def __repr__(self) -> str:
        return "<Tool name='search'>"
