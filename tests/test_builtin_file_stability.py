"""File discovery/editing handles real workspace races and nonregular entries."""

import os
from pathlib import Path

import pytest

from src.tools.builtin.edit_file.tool import EditFile
from src.tools.builtin.glob.tool import Glob
from src.tools.builtin.grep.tool import Grep
from src.tools.builtin.list_dir.tool import ListDir
from src.tools.builtin.workspace_utils import iter_files


def configured(cls, root):
    tool = cls()
    tool.set_workspace(root)
    return tool


def test_edit_preserves_concurrent_change_and_removes_temporary_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "source.py"
    path.write_text("old content\n")
    tool = configured(EditFile, tmp_path)
    real_fsync = os.fsync

    def concurrent_save(fd):
        real_fsync(fd)
        path.write_text("a different editor saved this\n")

    monkeypatch.setattr(os, "fsync", concurrent_save)
    result = tool.execute("source.py", "old", "new")
    assert not result.success and result.content["error"]["type"] == "file_changed"
    assert path.read_text() == "a different editor saved this\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["source.py"]


@pytest.mark.skipif(
    os.name == "nt", reason="Portable symlink privileges vary on Windows"
)
def test_list_dir_reports_symlink_entry_without_following_external_target(tmp_path):
    external = tmp_path / "outside"
    external.mkdir()
    (external / "private.txt").write_text("private")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "external-link").symlink_to(external, target_is_directory=True)
    result = configured(ListDir, root).execute()
    assert result.success
    assert result.content["entries"] == [{"path": "external-link", "type": "symlink"}]


@pytest.mark.parametrize("pattern", ["/absolute/*", "**bad", "a/**suffix"])
def test_glob_returns_structured_error_for_lazy_invalid_pattern(tmp_path, pattern):
    result = configured(Glob, tmp_path).execute(pattern)
    assert not result.success
    assert result.content["error"]["type"] == "invalid_argument"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX only")
def test_workspace_search_skips_fifo_instead_of_waiting_for_a_writer(tmp_path):
    os.mkfifo(tmp_path / "named-pipe")
    (tmp_path / "regular.txt").write_text("needle")
    assert [path.name for path in iter_files(tmp_path, tmp_path)] == ["regular.txt"]
    result = configured(Grep, tmp_path).execute("needle")
    assert result.success and len(result.content["matches"]) == 1


def test_grep_bounds_reads_and_keeps_match_visible_in_long_line(tmp_path):
    tool = configured(Grep, tmp_path)
    (tmp_path / "large.txt").write_bytes(b"x" * (tool.MAX_FILE_BYTES + 1))
    (tmp_path / "line.txt").write_text("x" * 10000 + "NEEDLE" + "y" * 10000)
    result = tool.execute("NEEDLE")
    assert result.success and result.content["skipped_files"] == 1
    match = result.content["matches"][0]
    assert "NEEDLE" in match["snippet"] and len(match["snippet"]) <= 2000
    assert match["snippet_truncated"]


def test_grep_reports_regex_timeout_instead_of_hanging(tmp_path):
    import time

    (tmp_path / "text.txt").write_text("a" * 50000 + "!")
    tool = configured(Grep, tmp_path)
    started = time.monotonic()
    result = tool.execute(r"(a|aa)+$")
    assert time.monotonic() - started < 2
    assert not result.success
    assert result.content["error"]["type"] == "search_timeout"
    assert result.content["search_complete"] is False


def test_grep_honors_cancellation_before_reading_files(tmp_path, monkeypatch):
    from threading import Event

    (tmp_path / "text.txt").write_text("needle")
    cancelled = Event()
    cancelled.set()
    tool = configured(Grep, tmp_path)
    tool.set_stop_event(cancelled)
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: pytest.fail("No file should be opened")
    )
    result = tool.execute("needle")
    assert not result.success and result.content["error"]["type"] == "cancelled"
