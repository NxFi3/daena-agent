from src.tools.builtin.search import Search
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolRegistry import ToolRegistry


def test_search_finds_matches_and_reports_line_numbers(tmp_path):
    (tmp_path / "a.py").write_text(
        "def first():\n    return 1\n\ndef target():\n    return 2\n",
        encoding="utf-8",
    )
    search = Search()
    search.set_workspace(str(tmp_path))

    result = search.execute(query="target", path=str(tmp_path), max_results=10, context_lines=1)

    assert result.success is True
    assert result.content["result_count"] == 1
    assert "a.py:4" in result.content["content"]


def test_search_rejects_workspace_escape(tmp_path):
    search = Search()
    search.set_workspace(str(tmp_path))

    result = search.execute(query="secret", path=str(tmp_path.parent))

    assert result.success is False
    assert result.content["error"]["type"] == "workspace_boundary"


def test_search_is_discoverable_and_exposed_to_dispatcher():
    registry = ToolRegistry()
    registry.discover()

    assert registry.is_available("search")

    dispatcher = ToolDispatcher(registry)
    calls = dispatcher.dispatch(
        [{"id": "1", "name": "search", "arguments": {"query": "ToolCall"}}]
    )

    assert len(calls) == 1
    assert calls[0].valid is True
    assert calls[0].name == "search"


def test_search_is_counted_as_observation():
    from src.agent.agentloop import Loop
    from src.models.ToolCall import ToolCall

    call = ToolCall(
        name="search",
        id="search-observation",
        valid=True,
        args={"query": "ToolCall"},
    )
    assert Loop._is_observation_call(call) is True


def test_tool_manager_hides_policy_disallowed_tools():
    from src.tools.ToolManager import ToolManager

    manager = ToolManager(
        {
            "security": {
                "allowed_tools": ["read_file"],
                "workspace_only": True,
            }
        }
    )
    try:
        definitions = manager.get_tools()
        names = {
            item["function"]["name"]
            for item in definitions
            if isinstance(item, dict)
        }
        assert names == {"read_file"}
    finally:
        manager.close()
