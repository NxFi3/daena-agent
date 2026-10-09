from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class WriteFile(Tool):
    """Write a complete UTF-8 text file atomically."""

    name = "write_file"
    action = "write"

    MAX_CONTENT_CHARS = 512_000
    PREVIEW_CHARS = 5000

    description = (
        "Write complete UTF-8 text content to a file. Use this for new artifacts "
        "(CSV, JSON, Markdown, generated source, scripts) or intentional whole-file "
        "replacement. The default refuses to overwrite an existing file; set "
        "overwrite=true only when you intentionally need to replace it. Writes are "
        "atomic, and the result reports bytes, lines and SHA-256. Then read the file "
        "back or run the relevant parser/test when the task requires validation. "
        "For small targeted edits to an existing source file, prefer apply_patch."
    )

    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path relative to the active workspace, or an absolute path if policy permits.",
            },
            "content": {
                "type": "string",
                "description": "The complete UTF-8 text content to write.",
                "maxLength": MAX_CONTENT_CHARS,
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "Must be true to intentionally replace an existing file. "
                    "Defaults to false so existing files are preserved."
                ),
                "default": False,
            },
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._workspace_root: Path | None = None

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def validate(self, arguments: dict[str, Any]) -> bool:
        file_path = arguments.get("file_path")
        content = arguments.get("content")
        overwrite = arguments.get("overwrite", False)
        return (
            isinstance(file_path, str)
            and bool(file_path.strip())
            and isinstance(content, str)
            and len(content) <= self.MAX_CONTENT_CHARS
            and type(overwrite) is bool
        )

    def execute(
        self,
        file_path: str,
        content: str,
        overwrite: bool = False,
    ) -> ToolResult:
        if not isinstance(file_path, str) or not file_path.strip():
            return self._error("invalid_argument", "file_path must be a non-empty string.")
        if not isinstance(content, str):
            return self._error("invalid_argument", "content must be a string.")
        if len(content) > self.MAX_CONTENT_CHARS:
            return self._error(
                "content_too_large",
                f"content exceeds the {self.MAX_CONTENT_CHARS:,}-character limit.",
            )
        if type(overwrite) is not bool:
            return self._error("invalid_argument", "overwrite must be a boolean.")

        raw_path = Path(file_path).expanduser()
        if not raw_path.is_absolute():
            if self._workspace_root is None:
                return self._error(
                    "workspace_unavailable",
                    "Workspace root is not configured; provide an absolute path or configure the workspace.",
                )
            raw_path = self._workspace_root / raw_path

        try:
            path = raw_path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            return self._error("path_error", f"Could not resolve output path: {exc}")

        parent = path.parent
        if not parent.exists() or not parent.is_dir():
            return self._error(
                "parent_not_found",
                f"Parent directory does not exist: {parent}. Create it explicitly first.",
            )
        if path.exists() and path.is_dir():
            return self._error("invalid_target", f"Output path is a directory: {path}")
        if path.exists() and not overwrite:
            try:
                old_size = path.stat().st_size
            except OSError:
                old_size = None
            return self._error(
                "already_exists",
                f"File already exists: {path}. Set overwrite=true only if replacement is intended.",
                extra={"path": str(path), "bytes": old_size},
            )

        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        line_count = len(content.splitlines())
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

            if overwrite:
                os.replace(temp_path, path)
                temp_path = None
            else:
                # Atomic creation that cannot replace a concurrently-created file.
                os.link(temp_path, path)
                temp_path.unlink()
                temp_path = None

            try:
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass

        except FileExistsError:
            return self._error(
                "already_exists",
                f"File already exists: {path}. Set overwrite=true only if replacement is intended.",
                extra={"path": str(path)},
            )
        except OSError as exc:
            return self._error(
                "write_error",
                f"Could not write {path}: {type(exc).__name__}: {exc}",
                extra={"path": str(path)},
            )
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "path": str(path),
                "bytes_written": len(encoded),
                "characters_written": len(content),
                "lines_written": line_count,
                "sha256": digest,
                "overwrote_existing": overwrite,
                "preview": content[: self.PREVIEW_CHARS],
                "preview_truncated": len(content) > self.PREVIEW_CHARS,
            },
            summary=(
                f"Wrote {path.name} ({len(encoded)} bytes, {line_count} lines, "
                f"SHA-256 {digest[:12]})."
            ),
            evidence={
                "path": str(path),
                "bytes_written": len(encoded),
                "lines_written": line_count,
                "sha256": digest,
            },
            effects=[{"action": "write", "target": str(path)}],
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
            content=content,
            summary=f"Write failed: {message}",
            evidence={"error_type": error_type, "message": message},
        )
