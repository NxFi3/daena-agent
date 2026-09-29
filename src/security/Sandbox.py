from __future__ import annotations

from pathlib import Path


class WorkspaceSandbox:
    """Resolve and validate filesystem targets against one workspace root."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else None

    def set_root(self, root: str | Path | None) -> None:
        self.root = Path(root).expanduser().resolve() if root else None

    def resolve(self, value: str | Path, *, must_exist: bool = False) -> Path:
        if self.root is None:
            raise ValueError("Workspace root is not configured.")

        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw

        resolved = candidate.resolve(strict=False)

        if not self.contains(resolved):
            raise PermissionError(
                f"Path escapes the configured workspace: {resolved}"
            )

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")

        return resolved

    def contains(self, path: str | Path) -> bool:
        if self.root is None:
            return False

        resolved_root = self.root.resolve()
        resolved_path = Path(path).expanduser().resolve(strict=False)

        try:
            resolved_path.relative_to(resolved_root)
            return True
        except ValueError:
            return False

    def validate_patch_paths(self, patch: str) -> None:
        import re

        if not isinstance(patch, str):
            raise TypeError("patch must be a string.")

        marker = re.compile(
            r"^\*\*\*\s+(?:Add|Update|Delete) File:\s*(.+?)\s*$"
        )

        found = False
        for raw_line in patch.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            match = marker.match(raw_line.strip())
            if not match:
                continue

            found = True
            path = match.group(1).strip()
            if not path:
                raise PermissionError("Patch contains an empty file path.")

            self.resolve(path, must_exist=False)

        if not found:
            raise ValueError("Patch contains no recognizable file operation.")
