from pathlib import Path

from src.tools.builtin.applypatch.applypatch import ApplyPatch


def test_patch_context_mismatch_returns_structured_failure(tmp_path):
    path = tmp_path / "app.js"
    path.write_text("const value = 2;\n", encoding="utf-8")

    result = ApplyPatch().execute(
        """*** Begin Patch
*** Update File: __PATH__
@@
-const value = 1;
+const value = 3;
*** End Patch""".replace("__PATH__", str(path))
    )

    assert result.success is False
    assert result.content["error"]["type"] == "patch_context_mismatch"
    assert result.content["path"] == str(path)
    assert "const value = 2;" in result.content["content"]


def test_patch_context_failure_does_not_apply_other_operations(tmp_path):
    good = tmp_path / "good.txt"
    target = tmp_path / "target.txt"
    good.write_text("before\n", encoding="utf-8")
    target.write_text("actual\n", encoding="utf-8")

    result = ApplyPatch().execute(
        """*** Begin Patch
*** Update File: __GOOD__
@@
-before
+after
*** Update File: __TARGET__
@@
-missing
+replacement
*** End Patch""".replace("__GOOD__", str(good)).replace("__TARGET__", str(target))
    )

    assert result.success is False
    assert result.content["error"]["type"] == "patch_context_mismatch"
    assert good.read_text(encoding="utf-8") == "before\n"
    assert target.read_text(encoding="utf-8") == "actual\n"


def test_patch_context_tolerates_trailing_whitespace_drift(tmp_path):
    path = tmp_path / "app.js"
    path.write_text("const value = 1;   \nnext();\n", encoding="utf-8")

    result = ApplyPatch().execute(
        f"""*** Begin Patch
*** Update File: {path}
@@
-const value = 1;
+const value = 2;
*** End Patch"""
    )

    assert result.success is True
    assert path.read_text(encoding="utf-8") == "const value = 2;   \nnext();\n"
