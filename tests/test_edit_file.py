from __future__ import annotations

from src.tools.builtin.edit_file.tool import EditFile


def test_exact_edit_preserves_crlf_and_returns_diff(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"first\r\nvalue=1\r\nlast\r\n")
    tool = EditFile()
    tool.set_workspace(tmp_path)
    result = tool.execute("sample.txt", "value=1", "value=2")
    assert result.success
    assert path.read_bytes() == b"first\r\nvalue=2\r\nlast\r\n"
    assert result.content["changed_line_range"]["start_line"] == 2
    assert "-value=1" in result.content["diff"]
    assert "+value=2" in result.content["diff"]


def test_not_found_reports_closest_lines(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha value\nbeta value\ngamma\n", encoding="utf-8")
    tool = EditFile()
    tool.set_workspace(tmp_path)
    result = tool.execute("sample.txt", "beta values", "fixed")
    assert not result.success
    assert result.content["error"]["type"] == "old_string_not_found"
    assert result.content["closest_line_numbers"]


def test_multiple_matches_require_replace_all(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("x=1\nx=1\nx=1\n", encoding="utf-8")
    tool = EditFile()
    tool.set_workspace(tmp_path)
    result = tool.execute("sample.txt", "x=1", "x=2")
    assert not result.success
    assert result.content["error"]["type"] == "ambiguous_old_string"
    assert result.content["match_count"] == 3
    result = tool.execute("sample.txt", "x=1", "x=2", replace_all=True)
    assert result.success
    assert path.read_text(encoding="utf-8") == "x=2\nx=2\nx=2\n"


def test_nonexistent_and_identical_edits_are_rejected(tmp_path):
    tool = EditFile()
    tool.set_workspace(tmp_path)
    missing = tool.execute("missing.txt", "a", "b")
    same = tool.execute("missing.txt", "a", "a")
    assert missing.content["error"]["type"] == "file_not_found"
    assert same.content["error"]["type"] == "no_change"


def test_workspace_escape_is_rejected(tmp_path):
    tool = EditFile()
    tool.set_workspace(tmp_path)
    result = tool.execute("../outside.txt", "a", "b")
    assert not result.success
    assert result.content["error"]["type"] == "workspace_boundary"
