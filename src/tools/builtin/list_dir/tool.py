from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.workspace_utils import SKIP_DIRECTORIES, relative_path, workspace_path


class ListDir(Tool):
    name = "list_dir"
    action = "inspect"
    allow_same_revision_repeat = True

    description = (
        "List one workspace directory with bounded entries. Use this instead "
        "of command_exec for ordinary directory discovery."
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": ".", "description": "Workspace-relative directory."},
            "show_hidden": {"type": "boolean", "default": False, "description": "Include hidden entries such as .github."},
            "max_entries": {"type": "integer", "minimum": 1, "maximum": 500, "default": 200, "description": "Maximum entries returned."},
        },
        "required": [],
        "additionalProperties": False,
    }

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def execute(self, path: str = ".", show_hidden: bool = False, max_entries: int = 200) -> ToolResult:
        root = getattr(self, "_workspace_root", None)
        if root is None:
            return self._error("workspace_missing", "No workspace is configured.")
        if not isinstance(show_hidden, bool):
            return self._error("invalid_argument", "show_hidden must be boolean.")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or not 1 <= max_entries <= 500:
            return self._error("invalid_argument", "max_entries must be between 1 and 500.")

        try:
            target = workspace_path(root, path)
        except (PermissionError, OSError) as exc:
            return self._error("invalid_argument", str(exc))

        if not target.exists():
            return self._error("not_found", f"Directory does not exist: {target}")
        if not target.is_dir():
            return self._error("invalid_target", f"Not a directory: {target}")

        entries: list[dict[str, Any]] = []
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if not show_hidden and child.name.startswith("."):
                continue
            if child.name in SKIP_DIRECTORIES:
                continue

            item: dict[str, Any] = {
                "path": relative_path(root, child),
                "type": "directory" if child.is_dir() else "file",
            }
            if child.is_file():
                try:
                    item["size"] = child.stat().st_size
                except OSError:
                    pass
            entries.append(item)
            if len(entries) >= max_entries:
                break

        truncated = len(entries) >= max_entries
        return ToolResult(
            success=True,
            name=self.name,
            content={
                "path": relative_path(root, target),
                "entries": entries,
                "truncated": truncated,
            },
            metadata={},
            summary=(
                f"Listed {len(entries)} workspace entrie(s)."
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
