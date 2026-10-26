import pytest

from src.tools.contract import (
    ToolContractError, normalize_tool_definition, validate_catalog,
)


def test_native_and_mcp_schema_share_contract():
    schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    native = normalize_tool_definition({"type": "function", "function": {
        "name": "web_fetch", "description": "Fetch", "parameters": schema,
    }})
    mcp = normalize_tool_definition({"name": "web_fetch", "description": "Fetch", "inputSchema": schema})
    assert native == mcp


def test_reject_invalid_required_and_non_json_schema():
    with pytest.raises(ToolContractError):
        normalize_tool_definition({"name": "x", "inputSchema": {"type": "object", "required": ["missing"]}})
    with pytest.raises(ToolContractError):
        normalize_tool_definition({"name": "x", "inputSchema": {"type": "object", "properties": {"x": set()}}})


def test_reject_duplicate_names_case_insensitively():
    a = {"name": "Alpha", "inputSchema": {"type": "object"}}
    b = {"name": "alpha", "inputSchema": {"type": "object"}}
    with pytest.raises(ToolContractError):
        validate_catalog([a, b])


def test_does_not_mutate_input():
    original = {"name": "example", "inputSchema": {"type": "object"}}
    normalize_tool_definition(original)
    assert original["inputSchema"] == {"type": "object"}
