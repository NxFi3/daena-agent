from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class Search(Tool):
    """Search text or regex across files inside the active workspace."""

    name = "search"
    action = "search"
    allow_same_revision_repeat = True

    DEFAULT_MAX_RESULTS = 50
    MAX_RESULTS = 100
    DEFAULT_CONTEXT_LINES = 1
    MAX_CONTEXT_LINES = 3
    MAX_OUTPUT_CHARS = 7_500

    _SKIP_DIRS: ClassVar[set[str]] = {
        ".git", ".hg", ".svn", "node_modules", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv",
        "venv", "dist", "build", ".next",
    }

    description = (
        "Search text or a regular expression across files in the active workspace. "
        "Use this to locate symbols, function names, error messages, references, or "
        "files before reading a known file. Results include file paths, line numbers, "
        "and short context. This is local repository search; use web_search for the "
        "internet."
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
            "use_regex": {
                "type": "boolean",
                "description": "Interpret query as a regular expression instead of literal text.",
                "default": False,
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

    def __init__(self) -> None:
        self._workspace_root: Path | None = None

    def set_workspace(self, directory: str) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def duplicate_key(
        self,
        arguments: dict[str, Any],
        *,
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        normalized = super().duplicate_key(
            arguments,
            workspace_root=workspace_root,
        )
        raw_path = normalized.get("path", ".")
        if isinstance(raw_path, str) and raw_path.strip():
            try:
                candidate = Path(raw_path).expanduser()
                if candidate.is_absolute():
                    normalized["path"] = str(candidate.resolve(strict=False))
                elif workspace_root:
                    normalized["path"] = str(
                        (Path(workspace_root).resolve() / candidate).resolve(strict=False)
                    )
            except (OSError, RuntimeError):
                pass
        return normalized

    def execute(
        self,
        query: str,
        path: str = ".",
        max_results: int = DEFAULT_MAX_RESULTS,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        file_pattern: str | None = None,
        use_regex: bool = False,
    ) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return self._error("invalid_argument", "query is required.")
        if not isinstance(path, str) or not path.strip():
            return self._error("invalid_argument", "path must be a non-empty string.")
        if type(max_results) is not int or not 1 <= max_results <= self.MAX_RESULTS:
            return self._error(
                "invalid_argument",
                f"max_results must be between 1 and {self.MAX_RESULTS}.",
            )
        if type(context_lines) is not int or not 0 <= context_lines <= self.MAX_CONTEXT_LINES:
            return self._error(
                "invalid_argument",
                f"context_lines must be between 0 and {self.MAX_CONTEXT_LINES}.",
            )

        root = Path(path).expanduser().resolve(strict=False)
        if self._workspace_root is not None:
            try:
                root.relative_to(self._workspace_root)
            except ValueError:
                return self._error(
                    "workspace_boundary",
                    f"Search path escapes the workspace: {root}",
                )

        if not root.exists():
            return self._error("not_found", f"Search path does not exist: {root}")

        patterns = self._patterns(file_pattern)
        regex_note: str | None = None

        if use_regex:
            try:
                matcher = re.compile(query, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                matcher = re.compile(re.escape(query), re.IGNORECASE | re.MULTILINE)
                regex_note = (
                    f"Invalid regular expression ({exc}); searched for the query "
                    "as literal text instead."
                )
        else:
            matcher = re.compile(re.escape(query), re.IGNORECASE | re.MULTILINE)

        matches: list[dict[str, Any]] = []
        for file_path in self._iter_files(root, patterns):
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
                        "context_start": start + 1,
                        "context": lines[start:end],
                    }
                )
                if len(matches) >= max_results:
                    break

        blocks: list[str] = []
        for item in matches:
            lines_out = [f"{item['path']}:{item['line']}"]
            for offset, value in enumerate(
                item["context"],
                start=item["context_start"],
            ):
                marker = ">" if offset == item["line"] else " "
                lines_out.append(f"{marker}{offset:5d} | {value}")
            blocks.append("\n".join(lines_out))
            if len("\n\n".join(blocks)) >= self.MAX_OUTPUT_CHARS:
                break

        rendered = "\n\n".join(blocks)
        if len(rendered) > self.MAX_OUTPUT_CHARS:
            rendered = rendered[: self.MAX_OUTPUT_CHARS] + "\n...[output truncated]"

        content: dict[str, Any] = {
            "success": True,
            "query": query,
            "path": str(root),
            "result_count": len(matches),
            "content": rendered or f'No matches for "{query}" under {root}.',
        }
        if regex_note:
            content["note"] = regex_note

        return ToolResult(
            success=True,
            name=self.name,
            content=content,
            metadata={"match_count": len(matches)},
        )

    @classmethod
    def _iter_files(cls, root: Path, patterns: list[str]) -> list[Path]:
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
        return not patterns or any(path.match(pattern) for pattern in patterns)

    @staticmethod
    def _patterns(value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

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

    def __repr__(self) -> str:
        return "<Tool name='search'>"
