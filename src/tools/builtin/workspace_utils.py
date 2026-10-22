from __future__ import annotations

import os
from pathlib import Path


SKIP_DIRECTORIES = frozenset({
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
})


def workspace_path(root: Path, value: str | None = ".") -> Path:
    root = root.expanduser().resolve()
    raw = Path(value or ".").expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not candidate.is_relative_to(root):
        raise PermissionError(f"Path escapes the workspace: {candidate}")
    return candidate


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def iter_files(root: Path, start: Path):
    root = root.resolve()
    start = workspace_path(root, str(start))
    if start.is_file():
        yield start
        return

    for current, dirs, files in os.walk(start):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRECTORIES)
        for name in sorted(files):
            path = Path(current) / name
            try:
                if path.is_symlink() and not path.resolve().is_relative_to(root):
                    continue
            except OSError:
                continue
            yield path
