from __future__ import annotations

from src.tools.ToolManager import ToolManager
from src.tools.ToolRegistry import ToolRegistry


RECOVERED_TOOLS = {
    "computer_use",
    "context_search",
    "edit_file",
    "explore",
    "glob",
    "grep",
    "list_dir",
    "tool_search",
    "write_file",
}


def test_recovered_builtin_tools_are_discoverable():
    registry = ToolRegistry()
    names = set(registry.discover())

    assert RECOVERED_TOOLS <= names


def test_tool_search_activates_computer_use(tmp_path):
    manager = ToolManager({
        "security": {
            "allowed_tools": [
                "tool_search",
                "computer_use",
                "read_file",
                "apply_patch",
                "command_exec",
            ],
            "workspace_only": True,
            "allow_network_tools": False,
        },
        "skills": {"enabled": False},
        "subagents": {"enabled": False},
    })
    try:
        manager.set_workspace(str(tmp_path))
        initial_names = {
            item["function"]["name"]
            for item in manager.get_model_tools()
        }
        assert "tool_search" in initial_names
        assert "computer_use" not in initial_names

        search = manager.get_tool("tool_search")
        result = search.execute(
            query="control the desktop GUI and click visible controls",
            top_k=3,
        )

        assert result.success is True
        match_names = {item["name"] for item in result.content["matches"]}
        assert "computer_use" in match_names

        active_names = {
            item["function"]["name"]
            for item in manager.get_model_tools()
        }
        assert "computer_use" in active_names
    finally:
        manager.close()
