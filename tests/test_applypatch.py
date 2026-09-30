from src.tools.builtin.applypatch.applypatch import ApplyPatch


def test_patch_context_mismatch_returns_structured_failure(tmp_path):
    target = tmp_path / "server.js"
    target.write_text("const port = 3000;\napp.listen(port);\n", encoding="utf-8")

    patch = """*** Begin Patch
*** Update File: server.js
@@
 const port = 4000;
-app.listen(port);
+app.listen(port, () => console.log("started"));
*** End Patch"""

    result = ApplyPatch().execute(patch)

    assert result.success is False
    assert result.name == "apply_patch"
    assert result.content["error"]["type"] == "patch_context_mismatch"
    assert "Patch context did not match" in result.content["error"]["message"]
    assert result.content["path"] == str(target)
    assert "const port = 3000;" in result.content["content"]
    assert target.read_text(encoding="utf-8") == "const port = 3000;\napp.listen(port);\n"
