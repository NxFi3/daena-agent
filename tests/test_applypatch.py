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


def test_malformed_hunk_reports_location_without_touching_other_files(tmp_path):
    first = tmp_path / "first.c"
    second = tmp_path / "second.c"
    first.write_text("old\n")
    second.write_text("void f() {\n}\n")
    patch = f"*** Begin Patch\n*** Update File: {first}\n@@\n-old\n+new\n*** Update File: {second}\n@@\n void f() {{\n}}\n*** End Patch"
    result = ApplyPatch().execute(patch)
    assert (
        not result.success and result.content["error"]["type"] == "patch_syntax_error"
    )
    assert result.content["line"] == 9 and result.content["offending_line"] == "}"
    assert first.read_text() == "old\n" and second.read_text() == "void f() {\n}\n"


def test_context_only_and_truncated_patches_do_not_claim_success(tmp_path):
    target = tmp_path / "source.c"
    target.write_text("old\n")
    for patch in [
        f"*** Begin Patch\n*** Update File: {target}\n@@\n old\n*** End Patch",
        f"*** Begin Patch\n*** Update File: {target}\n@@\n-old\n+new",
    ]:
        result = ApplyPatch().execute(patch)
        assert (
            not result.success
            and result.content["error"]["type"] == "patch_syntax_error"
        )
        assert target.read_text() == "old\n"


def test_structured_edits_are_available_through_dispatch_and_workspace_policy(tmp_path):
    import hashlib
    from src.tools.ToolManager import ToolManager

    target = tmp_path / "source.c"
    before = b"\xef\xbb\xbfint a = 1;\r\nint b = 2;\r\n"
    target.write_bytes(before)
    target.chmod(0o751)
    manager = ToolManager({})
    try:
        manager.set_workspace(str(tmp_path))
        calls = manager.dispatcher.dispatch(
            {
                "name": "apply_patch",
                "arguments": {
                    "edits": [
                        {
                            "path": "source.c",
                            "operation": "replace",
                            "old": "int a = 1;\nint b = 2;",
                            "new": "int a = 3;\nint b = 4;",
                            "expected_sha256": hashlib.sha256(before).hexdigest(),
                        }
                    ]
                },
            }
        )
        assert calls[0].valid, calls[0].validation_error
        result = manager.execute(calls)["results"][0]
        assert result.success, result.content
        assert target.read_bytes() == b"\xef\xbb\xbfint a = 3;\r\nint b = 4;\r\n"
        if os.name != "nt":
            assert target.stat().st_mode & 0o777 == 0o751
    finally:
        manager.close()


def test_structured_edits_cannot_escape_workspace(tmp_path):
    from src.tools.ToolManager import ToolManager

    manager = ToolManager({})
    try:
        manager.set_workspace(str(tmp_path))
        calls = manager.dispatcher.dispatch(
            {
                "name": "apply_patch",
                "arguments": {
                    "edits": [
                        {
                            "path": "../outside.txt",
                            "operation": "create",
                            "new": "not allowed",
                        }
                    ]
                },
            }
        )
        result = manager.execute(calls)["results"][0]
        assert not result.success
        assert result.metadata["security_rule"] == "workspace_boundary"
        assert not (tmp_path.parent / "outside.txt").exists()
    finally:
        manager.close()


def test_stale_version_aborts_entire_structured_batch(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_text("a")
    second.write_text("b")
    result = ApplyPatch().execute(
        edits=[
            {"path": str(first), "operation": "replace", "old": "a", "new": "A"},
            {
                "path": str(second),
                "operation": "replace",
                "old": "b",
                "new": "B",
                "expected_sha256": "0" * 64,
            },
        ]
    )
    assert not result.success and result.content["actual_sha256"]
    assert first.read_text() == "a" and second.read_text() == "b"


def test_alias_paths_are_one_planned_file(tmp_path):
    middle = tmp_path / "middle"
    middle.mkdir()
    target = tmp_path / "source"
    target.write_text("a b")
    result = ApplyPatch().execute(
        edits=[
            {"path": str(target), "operation": "replace", "old": "a", "new": "A"},
            {
                "path": str(middle / ".." / "source"),
                "operation": "replace",
                "old": "b",
                "new": "B",
            },
        ]
    )
    assert result.success, result.content
    assert (
        target.read_text() == "A B"
        and result.content["statistics"]["files_changed"] == 1
    )


def test_external_change_between_commits_is_preserved(tmp_path, monkeypatch):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_text("a")
    second.write_text("b")
    tool = ApplyPatch()
    original = tool._write_atomic

    def write(path, contents, mode):
        original(path, contents, mode)
        if path == first and contents == b"A":
            second.write_text("external")

    monkeypatch.setattr(tool, "_write_atomic", write)
    result = tool.execute(
        edits=[
            {"path": str(first), "operation": "replace", "old": "a", "new": "A"},
            {"path": str(second), "operation": "replace", "old": "b", "new": "B"},
        ]
    )
    assert not result.success and result.content["rollback_complete"]
    assert first.read_text() == "a" and second.read_text() == "external"


def test_rollback_does_not_overwrite_concurrent_change(tmp_path, monkeypatch):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_text("a")
    second.write_text("b")
    tool = ApplyPatch()
    original = tool._write_atomic

    def write(path, contents, mode):
        if path == second:
            first.write_text("external")
            raise OSError("simulated disk failure")
        original(path, contents, mode)

    monkeypatch.setattr(tool, "_write_atomic", write)
    result = tool.execute(
        edits=[
            {"path": str(first), "operation": "replace", "old": "a", "new": "A"},
            {"path": str(second), "operation": "replace", "old": "b", "new": "B"},
        ]
    )
    assert not result.success and not result.content["rollback_complete"]
    assert result.content["rollback_errors"][0]["path"] == str(first)
    assert first.read_text() == "external" and second.read_text() == "b"


def test_eof_marker_limits_matching_to_file_end(tmp_path):
    target = tmp_path / "source"
    target.write_text("same\nmiddle\nsame\n")
    result = ApplyPatch().execute(
        f"*** Begin Patch\n*** Update File: {target}\n@@\n-same\n+last\n*** End of File\n*** End Patch"
    )
    assert result.success, result.content
    assert target.read_text() == "same\nmiddle\nlast\n"
    result = ApplyPatch().execute(
        f"*** Begin Patch\n*** Update File: {target}\n@@\n-middle\n+changed\n*** End of File\n*** End Patch"
    )
    assert not result.success and target.read_text() == "same\nmiddle\nlast\n"


def test_numeric_hints_account_for_previous_insertions(tmp_path):
    target = tmp_path / "source"
    target.write_text("start\nsame\nendA\nsame\nendB\n")
    result = ApplyPatch().execute(
        f"*** Begin Patch\n*** Update File: {target}\n@@ -1,1 +1,2 @@\n-start\n+start\n+insert\n@@ -4,1 +5,1 @@\n-same\n+changed\n*** End Patch"
    )
    assert result.success, result.content
    assert target.read_text() == "start\ninsert\nsame\nendA\nchanged\nendB\n"
