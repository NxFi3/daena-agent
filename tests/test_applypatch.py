from pathlib import Path
import os

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
    assert path.read_text(encoding="utf-8") == "const value = 2;\nnext();\n"


def test_add_file_tolerates_unprefixed_model_content(tmp_path):
    path = tmp_path / "public" / "style.css"

    result = ApplyPatch().execute(
        f"""*** Begin Patch
*** Add File: {path}
body {{
  margin: 0;
}}
.form {{
  display: block;
}}
*** End Patch"""
    )

    assert result.success is True
    assert path.read_text(encoding="utf-8") == (
        "body {\n  margin: 0;\n}\n.form {\n  display: block;\n}\n"
    )


def test_concatenated_patch_blocks_apply_every_operation(tmp_path):
    target = tmp_path / "target.txt"
    created = tmp_path / "created.txt"
    target.write_text("before\n", encoding="utf-8")

    result = ApplyPatch().execute(
        f"""*** Begin Patch
*** Update File: {target}
@@
-before
+after
*** End Patch
*** Begin Patch
*** Add File: {created}
+hello
*** End Patch"""
    )

    assert result.success is True
    assert result.content["statistics"]["operations"] == 2
    assert result.content["statistics"]["files_changed"] == 2
    assert target.read_text(encoding="utf-8") == "after\n"
    assert created.read_text(encoding="utf-8") == "hello\n"


def test_replaying_update_patch_is_successful_noop(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("before\n", encoding="utf-8")
    patch = f"""*** Begin Patch
*** Update File: {target}
@@
-before
+after
*** End Patch"""

    first = ApplyPatch().execute(patch)
    second = ApplyPatch().execute(patch)

    assert first.success is True
    assert second.success is True
    assert second.content["statistics"]["files_changed"] == 0
    assert second.content["statistics"]["already_applied"] == 1
    assert second.metadata["idempotent_replay"] is True
    assert "Do not retry" in second.summary
    assert target.read_text(encoding="utf-8") == "after\n"


def test_replaying_add_and_delete_are_idempotent(tmp_path):
    target = tmp_path / "item.txt"
    add_patch = f"""*** Begin Patch
*** Add File: {target}
+value
*** End Patch"""
    delete_patch = f"""*** Begin Patch
*** Delete File: {target}
*** End Patch"""

    assert ApplyPatch().execute(add_patch).success is True
    add_replay = ApplyPatch().execute(add_patch)
    assert add_replay.success is True
    assert add_replay.metadata["idempotent_replay"] is True

    assert ApplyPatch().execute(delete_patch).success is True
    delete_replay = ApplyPatch().execute(delete_patch)
    assert delete_replay.success is True
    assert delete_replay.metadata["idempotent_replay"] is True


def test_multi_file_application_rolls_back_on_filesystem_failure(
    tmp_path,
    monkeypatch,
):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one\n", encoding="utf-8")
    second.write_text("two\n", encoding="utf-8")

    original_replace = os.replace

    def flaky_replace(source, destination, *args, **kwargs):
        if Path(destination) == second:
            raise OSError("simulated disk failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", flaky_replace)

    result = ApplyPatch().execute(
        f"""*** Begin Patch
*** Update File: {first}
@@
-one
+ONE
*** Update File: {second}
@@
-two
+TWO
*** End Patch"""
    )

    assert result.success is False
    assert result.content["error"]["type"] == "filesystem_error"
    assert first.read_text(encoding="utf-8") == "one\n"
    assert second.read_text(encoding="utf-8") == "two\n"


def test_conflict_resolution_preserves_unique_header_shorthand(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n")
    result = ApplyPatch().execute(
        f"*** Begin Patch\n*** Update File: {target}\n@@ value = 1 @@\n+value = 2\n*** End Patch"
    )
    assert result.success and target.read_text() == "value = 2\n"


def test_structured_edit_rejects_ambiguous_match_without_writes(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("same\nsame\n")
    result = ApplyPatch().execute(
        edits=[
            {
                "operation": "replace",
                "path": str(target),
                "old": "same",
                "new": "changed",
            }
        ]
    )
    assert not result.success
    assert target.read_text() == "same\nsame\n"
