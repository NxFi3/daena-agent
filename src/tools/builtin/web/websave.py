from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.tools.builtin.workspace_utils import workspace_path

MAX_SAVED_BYTES = 5 * 1024 * 1024


def save_payload(
    workspace_root: Path | None,
    save_to: str,
    *,
    text_content: str,
    json_value: Any = None,
    save_format: str = "text",
) -> dict[str, Any]:
    """Write a bounded web result atomically under the active workspace."""
    if workspace_root is None:
        raise ValueError("Workspace root is not configured for save_to.")
    if not isinstance(save_to, str) or not save_to.strip():
        raise ValueError("save_to must be a non-empty workspace-relative path.")
    if save_format not in {"text", "json"}:
        raise ValueError("save_format must be 'text' or 'json'.")

    raw_path = Path(save_to).expanduser()
    if raw_path.is_absolute():
        target = workspace_path(workspace_root, str(raw_path))
    else:
        target = workspace_path(workspace_root, save_to)
    existed = target.exists()

    if save_format == "json":
        value = json_value if json_value is not None else {"content": text_content}
        serialized = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    else:
        serialized = text_content

    payload = serialized.encode("utf-8")
    if len(payload) > MAX_SAVED_BYTES:
        raise ValueError(
            f"save_to payload is {len(payload)} bytes; limit is {MAX_SAVED_BYTES} bytes."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise

    digest = hashlib.sha256(payload).hexdigest()
    relative = target.resolve().relative_to(Path(workspace_root).resolve()).as_posix()
    return {
        "path": relative,
        "absolute_path": str(target.resolve()),
        "bytes": len(payload),
        "sha256": digest,
        "preview": serialized[:1500],
        "total_chars": len(text_content),
        "existed": existed,
        "serialized_chars": len(serialized),
    }


def record_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("records", "results", "items", "listWidgets"):
            rows = value.get(key)
            if isinstance(rows, list):
                return len(rows)
        nb = value.get("nb")
        if isinstance(nb, dict) and isinstance(nb.get("listWidgets"), list):
            return len(nb["listWidgets"])
    return None
