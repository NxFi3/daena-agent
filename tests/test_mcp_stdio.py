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
