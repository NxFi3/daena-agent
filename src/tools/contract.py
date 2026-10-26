"""Provider-neutral tool contract validation for built-in and future MCP tools.

This module validates the public tool boundary; it does not decide which tool
an agent should use. Tool implementations retain their own runtime validation.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_TOOL_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


class ToolContractError(ValueError):
    """Invalid or conflicting public tool contract."""


def normalize_tool_definition(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Return an OpenAI-compatible function declaration with strict invariants.

    MCP inputSchema and native Tool.parameters both map to JSON Schema.
    No automatic parameter coercion or guessed defaults are introduced here.
    """
    if not isinstance(definition, Mapping):
        raise ToolContractError("Tool definition must be a mapping")
    fn = definition.get("function", definition)
    if not isinstance(fn, Mapping):
        raise ToolContractError("Tool function must be a mapping")
    name = fn.get("name")
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise ToolContractError(f"Invalid tool name: {name!r}")
    description = fn.get("description", "")
    if not isinstance(description, str):
        raise ToolContractError(f"{name}: description must be text")
    schema = fn.get("parameters", fn.get("inputSchema", None))
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ToolContractError(f"{name}: input schema must be an object schema")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping):
        raise ToolContractError(f"{name}: properties must be a mapping")
    if not isinstance(required, list) or any(not isinstance(x, str) or x not in properties for x in required):
        raise ToolContractError(f"{name}: required must name declared properties")
    try:
        copied = json.loads(json.dumps(schema, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ToolContractError(f"{name}: schema is not JSON serializable") from exc
    copied.setdefault("properties", {})
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": copied,
    }}


def validate_catalog(definitions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reject duplicate names rather than silently shadowing a capability."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for definition in definitions:
        normalized = normalize_tool_definition(definition)
        name = normalized["function"]["name"].casefold()
        if name in seen:
            raise ToolContractError(f"Duplicate tool name: {name}")
        seen.add(name)
        result.append(normalized)
    return result
