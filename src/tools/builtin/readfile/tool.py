from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class ReadFile(Tool):
    """
    Read a UTF-8 text file.

    This tool is intentionally file-only.

    Use command_exec for:
        - listing directories
        - searching files
        - inspecting the workspace
        - running commands

    Output is structured through ToolResult.content and bounded to
    prevent unnecessarily large context payloads.
    """

    name = "read_file"
    action = "inspect"

    # Reads are observations and can be repeated when the model needs to
    # re-check evidence; the loop still bounds identical observations.
    allow_same_revision_repeat = True

    DEFAULT_MAX_OUTPUT_CHARS = 8_000
    MAX_OUTPUT_CHARS = 32_000
    MIN_OUTPUT_CHARS = 512

    description = (
        "Read a UTF-8 text file. "
        "Optional start_line and end_line can limit the returned range. "
        "Output is bounded to protect agent context. "
        "Use command_exec to inspect directories, search files, "
        "or run shell commands."
    )

    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the text file to read.",
            },
            "start_line": {
                "type": "integer",
                "description": (
                    "Optional 1-based first line to return. "
                    "If omitted, starts from the beginning."
                ),
                "minimum": 1,
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "Optional 1-based last line to return. "
                    "If omitted, reads through the end of the file."
                ),
                "minimum": 1,
            },
            "max_output_chars": {
                "type": "integer",
                "description": (
                    "Maximum number of characters returned. "
                    f"Default: {DEFAULT_MAX_OUTPUT_CHARS}."
                ),
                "default": DEFAULT_MAX_OUTPUT_CHARS,
                "minimum": MIN_OUTPUT_CHARS,
                "maximum": MAX_OUTPUT_CHARS,
            },
        },
        "required": ["file_path"],
        "additionalProperties": False,
    }

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

        raw_path = normalized.get("file_path")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path).expanduser()
            try:
                if candidate.is_absolute():
                    normalized["file_path"] = str(candidate.resolve(strict=False))
                elif workspace_root:
                    normalized["file_path"] = str(
                        (Path(workspace_root).resolve() / candidate).resolve(
                            strict=False
                        )
                    )
            except (OSError, RuntimeError):
                pass

        return normalized

    def normalize_arguments(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Strip fields that belong to this tool's result, not its input."""
        normalized = dict(arguments or {})
        notes: list[str] = []
        output_only = {
            "success",
            "type",
            "path",
            "content",
            "lines_requested",
            "lines_returned",
            "total_lines",
            "truncated",
        }
        removed = sorted(key for key in normalized if key in output_only)
        for key in removed:
            normalized.pop(key, None)
        if removed:
            notes.append(
                "Dropped read_file result fields from input: "
                + ", ".join(removed)
                + "."
            )
        return normalized, notes

    def execute(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> ToolResult:
        """
        ToolManager invokes this as:

            tool.execute(**toolcall.args)
        """

        validation_error = self._validate_arguments(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            max_output_chars=max_output_chars,
        )

        if validation_error is not None:
            return validation_error

        path = Path(file_path).expanduser()

        try:
            path = path.resolve()
        except OSError as exc:
            return self._error(
                error_type="path_error",
                message=f"Could not resolve path '{file_path}': {exc}",
            )

        if not path.exists():
            return self._error(
                error_type="not_found",
                message=f"File not found: {path}",
                extra={
                    "path": str(path),
                    "type": "file",
                },
            )

        if path.is_dir():
            return self._error(
                error_type="invalid_target",
                message=(
                    f"'{path}' is a directory, not a file. "
                    "Use command_exec to inspect directory contents."
                ),
                extra={
                    "path": str(path),
                    "type": "directory",
                },
            )

        if not path.is_file():
            return self._error(
                error_type="invalid_target",
                message=f"'{path}' is not a regular file.",
                extra={
                    "path": str(path),
                    "type": "other",
                },
            )

        return self._read_file(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_output_chars=max_output_chars,
        )

    def _validate_arguments(
        self,
        *,
        file_path: Any,
        start_line: Any,
        end_line: Any,
        max_output_chars: Any,
    ) -> ToolResult | None:

        if not isinstance(file_path, str):
            return self._error(
                error_type="invalid_argument",
                message="file_path must be a string.",
            )

        if not file_path.strip():
            return self._error(
                error_type="invalid_argument",
                message="file_path cannot be empty.",
            )

        if start_line is not None:
            if type(start_line) is not int:
                return self._error(
                    error_type="invalid_argument",
                    message="start_line must be an integer.",
                )

            if start_line < 1:
                return self._error(
                    error_type="invalid_argument",
                    message="start_line must be >= 1.",
                )

        if end_line is not None:
            if type(end_line) is not int:
                return self._error(
                    error_type="invalid_argument",
                    message="end_line must be an integer.",
                )

            if end_line < 1:
                return self._error(
                    error_type="invalid_argument",
                    message="end_line must be >= 1.",
                )

        if start_line is not None and end_line is not None and start_line > end_line:
            return self._error(
                error_type="invalid_argument",
                message="start_line must be <= end_line.",
            )

        if type(max_output_chars) is not int:
            return self._error(
                error_type="invalid_argument",
                message="max_output_chars must be an integer.",
            )

        if not (self.MIN_OUTPUT_CHARS <= max_output_chars <= self.MAX_OUTPUT_CHARS):
            return self._error(
                error_type="invalid_argument",
                message=(
                    "max_output_chars must be between "
                    f"{self.MIN_OUTPUT_CHARS} and "
                    f"{self.MAX_OUTPUT_CHARS}."
                ),
            )

        return None

    def _read_file(
        self,
        *,
        path: Path,
        start_line: int | None,
        end_line: int | None,
        max_output_chars: int,
    ) -> ToolResult:

        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="strict",
            ).splitlines(keepends=True)

        except UnicodeDecodeError:
            return self._error(
                error_type="binary_or_non_utf8",
                message=f"Cannot read non-UTF-8 or binary file: {path}",
                extra={
                    "path": str(path),
                    "type": "file",
                },
            )

        except PermissionError:
            return self._error(
                error_type="permission_error",
                message=f"Permission denied: {path}",
                extra={
                    "path": str(path),
                    "type": "file",
                },
            )

        except OSError as exc:
            return self._error(
                error_type="read_error",
                message=f"Could not read '{path}': {exc}",
                extra={
                    "path": str(path),
                    "type": "file",
                },
            )

        total_lines = len(lines)

        if total_lines == 0:
            return self._success(
                path=path,
                content="",
                start_line=None,
                end_line=None,
                lines_requested=0,
                lines_returned=0,
                total_lines=0,
                truncated=False,
                max_output_chars=max_output_chars,
            )

        actual_start = 1 if start_line is None else start_line
        actual_end = (
            total_lines
            if end_line is None
            else min(
                end_line,
                total_lines,
            )
        )

        if actual_start > total_lines:
            return self._error(
                error_type="line_out_of_range",
                message=(
                    f"start_line ({actual_start}) is beyond the end "
                    f"of the file ({total_lines} lines)."
                ),
                extra={
                    "path": str(path),
                    "type": "file",
                    "total_lines": total_lines,
                },
            )

        selected_lines = lines[actual_start - 1 : actual_end]

        raw_content = "".join(selected_lines)

        bounded_content, truncated = self._truncate(
            raw_content,
            max_output_chars,
        )

        if truncated:
            lines_returned = self._estimate_visible_lines(bounded_content)
        else:
            lines_returned = len(selected_lines)

        return self._success(
            path=path,
            content=bounded_content,
            start_line=actual_start,
            end_line=actual_end,
            lines_requested=actual_end - actual_start + 1,
            lines_returned=lines_returned,
            total_lines=total_lines,
            truncated=truncated,
            max_output_chars=max_output_chars,
        )

    @staticmethod
    def _truncate(
        value: str,
        limit: int,
    ) -> tuple[str, bool]:

        if len(value) <= limit:
            return value, False

        head = int(limit * 0.70)
        tail = limit - head

        omitted = len(value) - head - tail

        marker = (
            f"... {omitted} characters omitted; "
            "use start_line/end_line to inspect a narrower range ..."
        )

        bounded = (
            value[:head].rstrip() + "\n\n" + marker + "\n\n" + value[-tail:].lstrip()
        )

        return bounded, True

    @staticmethod
    def _estimate_visible_lines(
        content: str,
    ) -> int:
        if not content:
            return 0

        return len(content.splitlines())

    def _success(
        self,
        *,
        path: Path,
        content: str,
        start_line: int | None,
        end_line: int | None,
        lines_requested: int,
        lines_returned: int,
        total_lines: int,
        truncated: bool,
        max_output_chars: int,
    ) -> ToolResult:

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "type": "file",
                "path": str(path),
                "content": content,
                "start_line": start_line,
                "end_line": end_line,
                "lines_requested": lines_requested,
                "lines_returned": lines_returned,
                "total_lines": total_lines,
                "truncated": truncated,
                "max_output_chars": max_output_chars,
            },
            metadata={},
        )

    def _error(
        self,
        *,
        error_type: str,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> ToolResult:

        content: dict[str, Any] = {
            "success": False,
            "error": {
                "type": error_type,
                "message": message,
            },
        }

        if extra:
            content.update(extra)

        return ToolResult(
            success=False,
            name=self.name,
            content=content,
            metadata={},
        )

    def __repr__(self) -> str:
        return "<Tool name='read_file'>"
