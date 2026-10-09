from pathlib import Path

from src.tools.builtin.applypatch.applypatch import ApplyPatch


def test_update_without_begin_or_end_markers_is_lenient(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\nnext\n", encoding="utf-8")
    result = ApplyPatch().execute(f"*** Update File: {path}\n-old\n+new")
    assert result.success
    assert path.read_text(encoding="utf-8") == "new\nnext\n"


def test_missing_update_file_says_add_file_or_write_file(tmp_path):
    path = tmp_path / "missing.txt"
    result = ApplyPatch().execute(
        f"*** Begin Patch\n*** Update File: {path}\n@@\n-a\n+b\n*** End Patch"
    )
    assert not result.success
    assert "file does not exist; use '*** Add File:' or write_file" in result.content["error"]["message"]


def test_invalid_implicit_hunk_has_minimal_example(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    result = ApplyPatch().execute(
        f"*** Begin Patch\n*** Update File: {path}\nold\nnew\n*** End Patch"
    )
    assert not result.success
    assert "Minimal example" in result.content["error"]["message"]
