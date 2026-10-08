from pathlib import Path

from src.tools.builtin.glob.tool import Glob
from src.tools.builtin.grep.tool import Grep
from src.tools.builtin.list_dir.tool import ListDir


def test_grep_returns_file_line_and_snippet(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("first\nneedle = 42\nlast\n", encoding="utf-8")

    tool = Grep()
    tool.set_workspace(tmp_path)
    result = tool.execute("needle")

    assert result.success is True
    assert result.content["matches"][0]["file"] == "app.py"
    assert result.content["matches"][0]["line"] == 2
    assert "needle = 42" in result.content["matches"][0]["snippet"]


def test_grep_can_filter_files_and_reject_workspace_escape(tmp_path):
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    tool = Grep()
    tool.set_workspace(tmp_path)

    filtered = tool.execute("needle", include="*.py")
    assert [item["file"] for item in filtered.content["matches"]] == ["app.py"]

    escaped = tool.execute("needle", path="..")
    assert escaped.success is False
    assert escaped.content["error"]["type"] == "invalid_argument"


def test_glob_and_list_dir_are_workspace_bounded(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("a", encoding="utf-8")
    (src / "b.txt").write_text("b", encoding="utf-8")

    glob_tool = Glob()
    glob_tool.set_workspace(tmp_path)
    result = glob_tool.execute("**/*.py")
    assert [item["path"] for item in result.content["matches"]] == ["src/a.py"]

    list_tool = ListDir()
    list_tool.set_workspace(tmp_path)
    result = list_tool.execute("src")
    assert [item["path"] for item in result.content["entries"]] == ["src/a.py", "src/b.txt"]
