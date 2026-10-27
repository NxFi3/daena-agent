from types import SimpleNamespace

import pytest

from src.models.ToolCall import ToolCall
from src.tools.ToolManager import ToolManager
from src.tools.builtin.browser_use.tool import BrowserUseTool
from src.tools.builtin.browser_use.contracts import BrowserError, validate


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "fill", "text": "value"},
        {"action": "open", "url": "javascript:alert(1)"},
        {"action": "click", "target": {"role": "button", "text": "two selectors"}},
        {"action": "wait"},
        {"action": "upload", "target": {"label": "Upload"}, "paths": []},
        {"action": "dialog"},
        {"action": "click", "target": {}},
    ],
)
def test_invalid_requests_are_rejected_before_launch(arguments):
    with pytest.raises(BrowserError):
        validate(arguments)
    tool = BrowserUseTool()
    assert not tool.execute(**arguments).success
    assert tool._thread is None


def test_browser_discovery_permission_and_network_policy(tmp_path):
    config = {
        "security": {
            "allowed_tools": ["tool_search", "browser_use"],
            "allow_network_tools": True,
        }
    }
    manager = ToolManager(config)
    manager.set_workspace(str(tmp_path))
    try:
        manager.get_tools()
        found = manager._search_and_activate_tools(
            "control browser website forms login", 1
        )
        assert found["matches"][0]["name"] == "browser_use"
        manager.security.policy.allow_network_tools = False
        assert "browser_use" not in [d["function"]["name"] for d in manager.get_tools()]
        decision = manager.security.policy.evaluate(
            ToolCall(
                "browser_use", args={"action": "open", "url": "https://example.org"}
            ),
            None,
        )
        assert not decision.allowed
    finally:
        manager.close()


def test_finish_run_keeps_browser_until_conversation_closes(tmp_path):
    manager = ToolManager()
    manager.set_workspace(str(tmp_path))
    manager.get_tools()
    calls = []
    manager.toolregistry.tools["browser_use"] = SimpleNamespace(
        close=lambda: calls.append("closed")
    )
    manager.finish_run()
    assert calls == []
    manager.close()
    assert calls == ["closed"]
