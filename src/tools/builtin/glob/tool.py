from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.workspace_utils import SKIP_DIRECTORIES, relative_path, workspace_path


class Glob(Tool):
    name = "glob"
    action = "search"
    allow_same_revision_repeat = True

    description = (
        "Find workspace paths matching a glob pattern such as '**/*.py'. "
        "Returns bounded workspace-relative paths. Prefer this for discovering files."
    )

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path": {"type": "string", "default": ".", "description": "Workspace-relative directory to search from."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "default": 200, "description": "Maximum paths returned."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def execute(self, pattern: str, path: str = ".", max_results: int = 200) -> ToolResult:
        root = getattr(self, "_workspace_root", None)
        if root is None:
            return self._error("workspace_missing", "No workspace is configured.")
        if not isinstance(pattern, str) or not pattern.strip():
            return self._error("invalid_argument", "pattern must be a non-empty string.")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= 500:
            return self._error("invalid_argument", "max_results must be between 1 and 500.")

        try:
            start = workspace_path(root, path)
        except (PermissionError, OSError) as exc:
            return self._error("invalid_argument", str(exc))

        if not start.exists() or not start.is_dir():
            return self._error("invalid_target", f"Search root is not a directory: {start}")

        results: list[dict[str, Any]] = []
        try:
            candidates = start.glob(pattern)
        except (ValueError, OSError) as exc:
            return self._error("invalid_argument", str(exc))

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if not resolved.is_relative_to(root.resolve()):
                    continue
            except OSError:
                continue

            parts = candidate.relative_to(root).parts
            if any(part in SKIP_DIRECTORIES for part in parts):
                continue

            results.append({
                "path": relative_path(root, candidate),
                "type": "directory" if candidate.is_dir() else "file",
            })
            if len(results) >= max_results:
                break

        results.sort(key=lambda item: item["path"])
        truncated = len(results) >= max_results
        return ToolResult(
            success=True,
            name=self.name,
            content={
                "pattern": pattern,
                "path": relative_path(root, start),
                "matches": results,
                "truncated": truncated,
            },
            metadata={},
            summary=(
                f"Found {len(results)} path(s) matching {pattern!r}."
                + (" Results may be truncated." if truncated else "")
            ),
        )

    def _error(self, error_type: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={"success": False, "error": {"type": error_type, "message": message}},
            metadata={},
        )
