"""Local MCP stdio transport tests: no network or external MCP server needed."""
import sys

import pytest

from src.tools.mcp_stdio import StdioMCPServer, MCPTool, configured_servers


MOCK = r'''
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    method = req["method"]
    if method == "initialize":
        value = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                 "serverInfo": {"name": "mock", "version": "1"}}
    elif method == "tools/list":
        value = {"tools": [{"name": "echo", "description": "Echo value",
                           "inputSchema": {"type": "object", "properties":
                               {"value": {"type": "string"}}, "required": ["value"]}}]}
    elif method == "tools/call":
        value = {"content": [{"type": "text", "text": req["params"]["arguments"]["value"]}]}
    else:
        value = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": value}), flush=True)
'''


def test_mcp_stdio_discovery_and_dispatch():
    server = StdioMCPServer("mock", [sys.executable, "-u", "-c", MOCK], timeout=3)
    try:
        remote = server.list_tools()
        assert len(remote) == 1
        tool = MCPTool(server, remote[0])
        assert tool.name == "mcp_mock_echo"
        assert tool.parameters["required"] == ["value"]
        result = tool.execute(value="hello")
        assert result.success
        assert result.content["result"]["content"][0]["text"] == "hello"
    finally:
        server.close()


def test_mcp_is_disabled_by_default():
    assert configured_servers({}) == []
    assert configured_servers({"mcp": {"servers": {"test": {
        "enabled": False, "command": [sys.executable]}}}}) == []


def test_bad_mcp_names_are_rejected():
    with pytest.raises(ValueError):
        configured_servers({"mcp": {"servers": {"bad/name": {
            "enabled": True, "command": [sys.executable]}}}})


def test_mcp_tools_join_agent_catalog_and_dispatch(tmp_path):
    from src.tools.ToolManager import ToolManager
    config = {"mcp": {"servers": {"mock": {
        "enabled": True, "command": [sys.executable, "-u", "-c", MOCK],
        "timeout_seconds": 3,
    }}}, "security": {
        "allowed_tools": ["tool_search", "mcp_mock_*"],
        "allow_network_tools": True,
    }}
    manager = ToolManager(config)
    try:
        manager.set_workspace(str(tmp_path))
        names = {d["function"]["name"] for d in manager.get_tools()}
        assert "mcp_mock_echo" in names
        calls = manager.dispatcher.dispatch([
            {"id": "mcp1", "type": "function",
             "function": {"name": "mcp_mock_echo", "arguments": '{"value":"ok"}'}}
        ])
        response = manager.execute(calls)
        assert response["results"][0].success
        assert response["results"][0].content["result"]["content"][0]["text"] == "ok"
    finally:
        manager.close()


def test_network_policy_prevents_starting_mcp_servers():
    from src.tools.ToolManager import ToolManager
    manager = ToolManager({"mcp": {"servers": {"mock": {
        "enabled": True, "command": ["non-existent-command"]}}},
        "security": {"allow_network_tools": False}})
    try:
        assert not any(n.startswith("mcp_") for n in
                       (d["function"]["name"] for d in manager.get_tools()))
        assert manager._mcp_servers == []
    finally:
        manager.close()
