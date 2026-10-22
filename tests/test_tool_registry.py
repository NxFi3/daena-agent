from __future__ import annotations

import importlib

from src.tools.ToolRegistry import ToolRegistry


def test_registry_discovers_tool_from_package_without_all(tmp_path, monkeypatch):
    package_root = tmp_path / "fixture_builtin"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    bare = package_root / "bare"
    bare.mkdir()
    (bare / "tool.py").write_text(
        """from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class FallbackTool(Tool):
    name = "fallback_tool"
    description = "fixture"

    def execute(self, **kwargs):
        return ToolResult(True, self.name, {"success": True})
""",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    registry = ToolRegistry()
    registry.tools_path = package_root
    registry.package_root = "fixture_builtin"

    names = registry.discover()

    assert "fallback_tool" in names
    assert registry.get("fallback_tool").name == "fallback_tool"
