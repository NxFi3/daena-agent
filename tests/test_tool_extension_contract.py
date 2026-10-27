"""Host adapters use the same validation, policy and lifecycle as native tools."""

from copy import deepcopy
from threading import Event

import pytest

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolManager import ToolManager
from src.tools.ToolRegistry import ToolRegistry


class Adapter(Tool):
    name = "remote_echo"
    description = "A host-owned fixture, without a network transport"
    requires_network = True
    parameters = {"type": "object", "properties": {}}

    def __init__(self, schema=None):
        self.parameters = deepcopy(schema or self.parameters)
        self.received = []
        self.stop_event = None

    def set_stop_event(self, event):
        self.stop_event = event

    def execute(self, **kwargs):
        self.received.append(kwargs)
        return ToolResult(True, self.name, {"echo": kwargs})


def dispatcher(schema):
    registry = ToolRegistry()
    registry.register(Adapter(schema))
    return ToolDispatcher(registry)


def call(dispatch, arguments):
    return dispatch.dispatch({"name": "remote_echo", "arguments": arguments})[0]


def test_nested_schema_constraints_and_local_references_are_enforced():
    dispatch = dispatcher(
        {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/payload"}},
            "required": ["payload"],
            "$defs": {
                "payload": {
                    "type": "object",
                    "required": ["items"],
                    "additionalProperties": False,
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1, "maximum": 5},
                        }
                    },
                }
            },
        }
    )
    assert call(dispatch, {"payload": {"items": [1, 5]}}).valid
    for value in ({}, {"items": [0]}, {"items": ["1"]}, {"items": [1], "extra": 1}):
        bad = call(dispatch, {"payload": value})
        assert not bad.valid and "schema" in bad.validation_error


def test_schema_permitted_extra_and_pattern_properties_reach_execution():
    manager = ToolManager({"security": {"allowed_tools": ["remote_echo"]}})
    try:
        adapter = Adapter(
            {
                "type": "object",
                "properties": {},
                "patternProperties": {"^item_": {"type": "integer"}},
                "additionalProperties": {"type": "string"},
            }
        )
        manager.register_tool(adapter)
        valid = call(manager.dispatcher, {"item_count": 3, "label": "hello"})
        assert valid.valid
        assert manager.execute([valid])["results"][0].success
        assert adapter.received == [{"item_count": 3, "label": "hello"}]
        assert not call(manager.dispatcher, {"item_count": "bad"}).valid
        assert not call(manager.dispatcher, {"label": 5}).valid
    finally:
        manager.close()


def test_pattern_properties_are_allowed_with_additional_properties_false():
    dispatch = dispatcher(
        {
            "type": "object",
            "patternProperties": {"^item_": {"type": "integer"}},
            "additionalProperties": False,
        }
    )
    assert call(dispatch, {"item_count": 2}).valid
    assert not call(dispatch, {"other": 2}).valid


def test_range_clamping_is_opt_in_for_external_tool_schemas():
    strict = dispatcher(
        {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 5}},
        }
    )
    assert not call(strict, {"count": 8}).valid

    clamped = dispatcher(
        {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "x-daena-clamp": True,
                }
            },
        }
    )
    normalized = call(clamped, {"count": 8})
    assert normalized.valid and normalized.args["count"] == 5
    assert normalized.normalization_notes


def test_remote_schema_references_never_fetch_from_the_network():
    dispatch = dispatcher(
        {
            "type": "object",
            "properties": {"payload": {"$ref": "https://example.invalid/remote.json"}},
        }
    )
    result = call(dispatch, {"payload": "test"})
    assert not result.valid and "locally" in result.validation_error


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_non_json_argument_values_are_rejected(value):
    assert not call(dispatcher({"type": "object"}), {"value": value}).valid


def test_registered_adapter_requires_allowlist_and_respects_network_and_cancel():
    adapter = Adapter()
    manager = ToolManager(
        {"security": {"allowed_tools": [adapter.name], "allow_network_tools": False}}
    )
    try:
        manager.register_tool(adapter)
        assert not manager.get_tools()
        request = ToolCall(adapter.name, args={"value": "hello"}, valid=True)
        denied = manager.execute([request])["results"][0]
        assert denied.metadata["security_rule"] == "network_disabled"
        manager.security.policy.allow_network_tools = True
        manager.security.policy.allowed_tools.clear()
        assert not manager.execute([request])["results"][0].success
        manager.security.policy.allowed_tools.add(adapter.name)
        manager.stop_event = Event()
        manager.stop_event.set()
        canceled = manager.execute([request])["results"][0]
        assert canceled.metadata["security_rule"] == "task_cancelled"
        assert adapter.stop_event is manager.stop_event
        assert not adapter.received
    finally:
        manager.close()


def test_direct_canonical_calls_cannot_bypass_adapter_schema():
    adapter = Adapter(
        {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
    )
    manager = ToolManager({"security": {"allowed_tools": [adapter.name]}})
    try:
        manager.register_tool(adapter)
        result = manager.execute([ToolCall(adapter.name, valid=True, args={})])[
            "results"
        ][0]
        assert (
            not result.success
            and result.content["error"]["type"] == "invalid_tool_call"
        )
        assert not adapter.received
    finally:
        manager.close()


def test_registration_rejects_collisions_and_definition_mutation_cannot_change_schema():
    registry = ToolRegistry()
    first = Adapter({"type": "object", "properties": {"name": {"type": "string"}}})
    registry.register(first)
    assert registry.register(first) == first.name
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Adapter())
    definition = registry.get_definitions()[0]
    definition["function"]["parameters"]["properties"]["name"]["type"] = "integer"
    assert first.parameters["properties"]["name"]["type"] == "string"
    assert registry.get(first.name) is first


def test_cleanup_failure_cannot_abandon_other_resources():
    manager = ToolManager()
    closed = []

    def fail():
        closed.append("broken")
        raise RuntimeError("fixture failure")

    first, second = Adapter(), Adapter()
    first.close = fail
    second.close = lambda: closed.append("second")
    manager.toolregistry.tools = {"first": first, "alias": first, "second": second}
    manager.process_manager.close = lambda **kwargs: closed.append("processes")
    manager.subagents.close = lambda: closed.append("subagents")
    manager.close()
    assert closed == ["processes", "subagents", "broken", "second"]
    assert (
        len(manager.cleanup_errors) == 1
        and "fixture failure" in manager.cleanup_errors[0]
    )
