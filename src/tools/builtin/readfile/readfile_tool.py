from __future__ import annotations

import hashlib
from bisect import bisect_right
from uuid import uuid4
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class ReadFile(Tool):
    """
    Read a UTF-8 text file.

    This tool is intentionally file-only.

    Use command_exec for:
        - running commands
        - direct workspace inspection when needed

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
        "Read a UTF-8 text file. Only use this for a file you know exists; "
        "do not use it to inspect directories or guess filenames. "
        "Optional start_line and end_line can limit the returned range. "
        "Output is an exact prefix, never a head/tail splice. Follow next_read to continue; offsets are Unicode character offsets and file_sha256 identifies the source version. "
        "Use glob/list_dir to discover paths and grep for relevant symbols; use command_exec for running commands."
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
            "start_offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Absolute Unicode character offset, normally copied from next_read. Do not combine with start_line.",
            },
            "expected_sha256": {
                "type": "string",
                "description": "Optional source version from next_read; reject continuation if the file changed.",
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
        normalized, notes = super().normalize_arguments(arguments)
        output_only = {
            "success",
            "type",
            "path",
            "content",
            "lines_requested",
            "lines_returned",
            "total_lines",
            "read_id",
            "file_sha256",
            "source_start_offset",
            "source_end_offset",
            "requested_end_offset",
            "offset_unit",
            "returned_end_line",
            "next_read",
            "transport_truncated",
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
        start_offset: int | None = None,
        expected_sha256: str | None = None,
    ) -> ToolResult:
        """
        ToolManager invokes this as:

            tool.execute(**toolcall.args)
        """

        if start_offset is not None and (
            type(start_offset) is not int or start_offset < 0 or start_line is not None
        ):
            return self._error(
                error_type="invalid_argument",
                message="start_offset must be nonnegative and cannot be combined with start_line.",
            )
        if expected_sha256 is not None and not isinstance(expected_sha256, str):
            return self._error(
                error_type="invalid_argument",
                message="expected_sha256 must be a string.",
            )
        # Models frequently send a reversed range (end_line < start_line).
        # Failing here used to trap the agent in an identical-retry loop that
        # the runtime then blocked. Recover instead: read from start_line to
        # the end of the file (still bounded by max_output_chars) and say so.
        range_note: str | None = None
        if (
            type(start_line) is int
            and type(end_line) is int
            and start_line >= 1
            and end_line >= 1
            and start_line > end_line
        ):
            range_note = (
                f"end_line ({end_line}) was before start_line ({start_line}); "
                f"returned lines from {start_line} to the end of the file "
                "instead. Use end_line >= start_line to limit the range."
            )
            end_line = None

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

        result = self._read_file(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_output_chars=max_output_chars,
            start_offset=start_offset,
            expected_sha256=expected_sha256,
        )

        if range_note and result.success and isinstance(result.content, dict):
            result.content["note"] = range_note
            result.evidence["note"] = range_note

        return result

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
        path,
        start_line,
        end_line,
        max_output_chars,
        start_offset=None,
        expected_sha256=None,
    ):
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return self._error(
                error_type="binary_or_non_utf8",
                message=f"Cannot read non-UTF-8 or binary file: {path}",
                extra={"path": str(path), "type": "file"},
            )
        except OSError as exc:
            return self._error(
                error_type="read_error",
                message=f"Could not read {path}: {exc}",
                extra={"path": str(path), "type": "file"},
            )
        version = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and expected_sha256 != version:
            return self._error(
                error_type="file_changed",
                message="The source changed since the previous page. Read the relevant range again without the old expected_sha256.",
                extra={"path": str(path), "file_sha256": version},
            )
        lines = text.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        first = start_line or 1
        if text and first > len(lines):
            return self._error(
                error_type="line_out_of_range",
                message=f"start_line ({first}) is beyond the end of the file ({len(lines)} lines).",
                extra={"path": str(path), "type": "file", "total_lines": len(lines)},
            )
        requested_end = min(end_line or len(lines), len(lines))
        start = (
            start_offset
            if start_offset is not None
            else offsets[min(first - 1, len(lines))]
        )
        stop = offsets[requested_end]
        if start > stop:
            return self._error(
                error_type="line_out_of_range",
                message="start_offset is beyond the requested range.",
                extra={"path": str(path), "total_lines": len(lines)},
            )
        end = min(stop, start + max_output_chars)
        body = text[start:end]
        next_read = (
            None
            if end == stop
            else {
                "file_path": str(path),
                "start_offset": end,
                "end_line": requested_end,
                "expected_sha256": version,
                "max_output_chars": max_output_chars,
            }
        )
        payload = {
            "success": True,
            "type": "file",
            "path": str(path),
            "content": body,
            "read_id": uuid4().hex,
            "file_sha256": version,
            "source_start_offset": start,
            "source_end_offset": end,
            "requested_end_offset": stop,
            "offset_unit": "unicode_characters",
            "start_line": min(bisect_right(offsets, start), len(lines))
            if lines
            else None,
            "end_line": requested_end if lines else None,
            "returned_end_line": bisect_right(offsets, end - 1) if body else None,
            "lines_requested": max(0, requested_end - first + 1),
            "lines_returned": len(body.splitlines()),
            "total_lines": len(lines),
            "truncated": end < stop,
            "next_read": next_read,
            "max_output_chars": max_output_chars,
        }
        # Explicit evidence avoids a second unrelated character cap in ToolResult.
        return ToolResult(
            True,
            self.name,
            content=payload,
            evidence=dict(payload),
            summary=f"Read {path}: characters [{start}, {end}) of requested end {stop}; "
            + (
                "follow next_read for the remaining text."
                if next_read
                else "requested range complete."
            ),
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
