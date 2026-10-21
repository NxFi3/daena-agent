"""Isolated worker copies with explicit, conflict-checked integration."""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


class WorkspaceSnapshot:
    EXCLUDED = {
        ".git",
        ".daena",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
    MAX_BYTES = 100 * 1024 * 1024
    MAX_FILES = 10_000

    def __init__(self, source: str, parent: str, excluded=()):
        self.source = Path(source).resolve()
        self.excluded = {
            p.relative_to(self.source).as_posix()
            for p in map(Path, excluded)
            if p.is_relative_to(self.source)
        }
        self.root = Path(parent) / "workspace"
        self.root.mkdir()
        self.base = self.manifest(self.source, self.excluded)
        for name in self.base:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source / name, destination)
        # A source changing while copying is not a reliable merge base.
        if self.manifest(self.root, self.excluded) != self.base:
            raise RuntimeError(
                "Workspace changed during snapshot; retry after the edit finishes"
            )

    @classmethod
    def manifest(cls, root: Path, excluded=()):
        result, total = {}, 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in cls.EXCLUDED
                and (Path(directory) / name).relative_to(root).as_posix()
                not in excluded
            )
            for name in dirs + files:
                if (Path(directory) / name).is_symlink():
                    raise ValueError(
                        "Worker snapshots do not support symlinks: "
                        + str(Path(directory) / name)
                    )
            for name in sorted(files):
                path = Path(directory) / name
                if path.relative_to(root).as_posix() in excluded:
                    continue
                if not path.is_file():
                    raise ValueError(
                        "Worker snapshots require regular files: " + str(path)
                    )
                size = path.stat().st_size
                total += size
                if total > cls.MAX_BYTES or len(result) >= cls.MAX_FILES:
                    raise ValueError(
                        "Worker snapshot exceeds the 100 MiB / 10000 file limit"
                    )
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(65536), b""):
                        digest.update(chunk)
                result[path.relative_to(root).as_posix()] = (
                    digest.hexdigest(),
                    path.stat().st_mode & 0o777,
                )
        return result

    def changes(self):
        current = self.manifest(self.root, self.excluded)
        return [
            {
                "path": name,
                "operation": "delete"
                if name not in current
                else "update"
                if name in self.base
                else "add",
            }
            for name in sorted(self.base.keys() | current.keys())
            if self.base.get(name) != current.get(name)
        ]

    def diff(self):
        """Bounded review against the parent's current files (including conflicts)."""
        changes = self.changes()
        sections = []
        remaining = 16000
        truncated = False
        for item in changes:
            if remaining <= 0:
                truncated = True
                break
            name = item["path"]
            parent, worker = self.source / name, self.root / name
            if (
                parent.is_symlink()
                or worker.is_symlink()
                or not parent.resolve().is_relative_to(self.source)
                or not worker.resolve().is_relative_to(self.root)
            ):
                raise ValueError("Cannot review links outside the workspace")
            # Bound reads as well as output; large or binary files are summarized.
            if any(
                path.is_file() and path.stat().st_size > 65536
                for path in (parent, worker)
            ):
                section = f"{name}: {item['operation']} (large file; verify after integration)\n"
            else:
                try:
                    before = (
                        parent.read_text(encoding="utf-8") if parent.is_file() else ""
                    )
                    after = (
                        worker.read_text(encoding="utf-8") if worker.is_file() else ""
                    )
                    section = "".join(
                        difflib.unified_diff(
                            before.splitlines(True),
                            after.splitlines(True),
                            fromfile="parent/" + name,
                            tofile="worker/" + name,
                        )
                    )
                    if not section:
                        section = f"{name}: permissions changed\n"
                except UnicodeDecodeError:
                    section = f"{name}: binary {item['operation']}\n"
            truncated = truncated or len(section) > remaining
            sections.append(section[:remaining])
            remaining -= len(section)
        return {"files": changes, "diff": "\n".join(sections), "truncated": truncated}

    def merge(self):
        changes = self.changes()
        current = self.manifest(self.source, self.excluded)
        conflicts = [
            item["path"]
            for item in changes
            if current.get(item["path"]) != self.base.get(item["path"])
        ]
        # Also reject path-type conflicts before writing any files.
        for item in changes:
            path = self.source / item["path"]
            if path.is_dir() or any(
                parent.exists() and not parent.is_dir()
                for parent in path.parents
                if parent != self.source
            ):
                conflicts.append(item["path"])
        if conflicts:
            return {"merged": False, "conflicts": sorted(set(conflicts)), "files": []}
        # Stage every replacement first; rollback already-written paths on I/O
        # failure. External writers should still be coordinated by the host.
        backups, written = {}, []
        with tempfile.TemporaryDirectory(
            prefix="daena-merge-", dir=self.source.parent
        ) as staging:
            for index, item in enumerate(changes):
                name = item["path"]
                target = self.source / name
                backups[name] = (
                    (target.read_bytes(), target.stat().st_mode & 0o777)
                    if target.exists()
                    else None
                )
                if item["operation"] != "delete":
                    shutil.copy2(self.root / name, Path(staging) / str(index))
            try:
                for index, item in enumerate(changes):
                    target = self.source / item["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if item["operation"] == "delete":
                        target.unlink()
                    else:
                        os.replace(Path(staging) / str(index), target)
                    written.append(item["path"])
            except Exception:
                for name in reversed(written):
                    target = self.source / name
                    if backups[name] is None:
                        target.unlink(missing_ok=True)
                    else:
                        data, mode = backups[name]
                        target.write_bytes(data)
                        target.chmod(mode)
                raise
        return {"merged": True, "conflicts": [], "files": changes}
