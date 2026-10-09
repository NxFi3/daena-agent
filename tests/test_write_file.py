from __future__ import annotations

import hashlib
from pathlib import Path

from src.tools.builtin.write_file.tool import WriteFile


def test_write_file_creates_complete_utf8_artifact_and_reports_digest(tmp_path):
    tool = WriteFile()
    tool.set_workspace(tmp_path)
    content = "name,price\nخانه,۱۲۳۴۵\n"

    result = tool.execute("dataset.csv", content)

    target = tmp_path / "dataset.csv"
    assert result.success is True
    assert target.read_text(encoding="utf-8") == content
    assert result.content["path"] == str(target)
    assert result.content["bytes_written"] == len(content.encode("utf-8"))
    assert result.content["sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert result.content["lines_written"] == 2


def test_write_file_refuses_existing_file_without_explicit_overwrite(tmp_path):
    tool = WriteFile()
    tool.set_workspace(tmp_path)
    target = tmp_path / "data.csv"
    target.write_text("old,data\n", encoding="utf-8")

    result = tool.execute("data.csv", "new,data\n")

    assert result.success is False
    assert result.content["error"]["type"] == "already_exists"
    assert target.read_text(encoding="utf-8") == "old,data\n"


def test_write_file_explicitly_overwrites_existing_file_atomically(tmp_path):
    tool = WriteFile()
    tool.set_workspace(tmp_path)
    target = tmp_path / "data.csv"
    target.write_text("old,data\n", encoding="utf-8")

    result = tool.execute("data.csv", "new,data\n", overwrite=True)

    assert result.success is True
    assert result.content["overwrote_existing"] is True
    assert target.read_text(encoding="utf-8") == "new,data\n"
    assert not list(tmp_path.glob(".data.csv.*.tmp"))


def test_write_file_refuses_missing_parent_without_creating_directories(tmp_path):
    tool = WriteFile()
    tool.set_workspace(tmp_path)

    result = tool.execute("missing/data.csv", "x,y\n")

    assert result.success is False
    assert result.content["error"]["type"] == "parent_not_found"
    assert not (tmp_path / "missing").exists()


def test_write_file_rejects_oversized_content(tmp_path):
    tool = WriteFile()
    tool.set_workspace(tmp_path)

    result = tool.execute("too_big.txt", "x" * (tool.MAX_CONTENT_CHARS + 1))

    assert result.success is False
    assert result.content["error"]["type"] == "content_too_large"
    assert not (tmp_path / "too_big.txt").exists()


def test_write_file_requires_workspace_for_relative_paths():
    result = WriteFile().execute("data.txt", "hello")

    assert result.success is False
    assert result.content["error"]["type"] == "workspace_unavailable"


def test_tool_manager_runs_write_file_inside_workspace_with_policy(tmp_path):
    from src.models.ToolCall import ToolCall
    from src.tools.ToolManager import ToolManager

    manager = ToolManager({
        "security": {
            "workspace_only": True,
            "force_approve": True,
        }
    })
    manager.set_workspace(str(tmp_path))
    try:
        outcome = manager.execute([
            ToolCall(
                name="write_file",
                id="write-managed",
                valid=True,
                args={
                    "file_path": "nested.csv",
                    "content": "a,b\n1,2\n",
                },
            )
        ])
        assert outcome["results"][0].success is True
        assert (tmp_path / "nested.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    finally:
        manager.close()
