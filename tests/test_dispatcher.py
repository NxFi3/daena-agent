from src.models.ToolCall import ToolCall
from src.tools.ToolDispatcher import ToolDispatcher


class FakeTool:
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
        },
    }

    def validate(self, args):
        return isinstance(args.get("file_path"), str)

    def describe_call(self, args):
        return {"action": "inspect", "target": args["file_path"]}


class FakeRegistry:
    def is_available(self, name):
        return name == "read_file"

    def get(self, name):
        return FakeTool() if name == "read_file" else None


def test_dispatcher_normalizes_alias_and_generates_id():
    dispatcher = ToolDispatcher(FakeRegistry())
    calls = dispatcher.dispatch({
        "name": "read_file",
        "arguments": '{"path":"main.py","ignored":true}',
    })

    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call, ToolCall)
    assert call.valid is True
    assert call.args == {"file_path": "main.py"}
    assert call.target == "main.py"
    assert call.id

from src.tools.builtin.command_exec.tool import CommandExec


class FakeCommandTool:
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "array"},
            "yield_time_ms": {"type": "integer"},
        },
    }

    def validate(self, args):
        return isinstance(args.get("command"), list)

    def describe_call(self, args):
        return {"action": "run", "target": " ".join(args["command"])}


class CommandRegistry:
    def is_available(self, name):
        return name == "command_exec"

    def get(self, name):
        return FakeCommandTool() if name == "command_exec" else None


def test_dispatcher_keeps_legacy_command_timeout_alias():
    dispatcher = ToolDispatcher(CommandRegistry())
    calls = dispatcher.dispatch({
        "name": "command_exec",
        "arguments": {
            "command": ["pytest", "-q"],
            "timeout_ms": 5000,
        },
    })

    call = calls[0]
    assert call.valid is True
    assert call.args["yield_time_ms"] == 5000
    assert "timeout_ms" not in call.args


def test_dispatcher_clamps_numeric_command_options():
    dispatcher = ToolDispatcher(type("Registry", (), {
        "is_available": lambda self, name: name == "command_exec",
        "get": lambda self, name: CommandExec() if name == "command_exec" else None,
    })())

    calls = dispatcher.dispatch({
        "name": "command_exec",
        "arguments": {
            "command": ["npm", "test"],
            "timeout_ms": "120000",
            "max_output_chars": "64000",
        },
    })

    call = calls[0]
    assert call.valid is True
    assert call.args["yield_time_ms"] == 30_000
    assert call.args["max_output_chars"] == 32_000
    assert call.normalization_notes




def test_dispatcher_rejects_missing_required_args_before_execution():
    class RequiredTool:
        parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

        def validate(self, args):
            raise AssertionError("validate/execute must not run for missing required args")

    class Registry:
        def is_available(self, name):
            return name == "required_tool"

        def get(self, name):
            return RequiredTool() if name == "required_tool" else None

    dispatcher = ToolDispatcher(Registry())
    call = dispatcher.dispatch({
        "name": "required_tool",
        "arguments": "{}",
    })[0]

    assert call.valid is False
    assert "Missing required argument" in call.validation_error


class StrictTool(FakeTool):
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
        },
        "additionalProperties": False,
    }


class StrictRegistry:
    def is_available(self, name):
        return name == "read_file"

    def get(self, name):
        return StrictTool() if name == "read_file" else None


def test_dispatcher_rejects_unknown_args_for_strict_tools():
    dispatcher = ToolDispatcher(StrictRegistry())
    call = dispatcher.dispatch({
        "name": "read_file",
        "arguments": '{"file_path":"main.py","timeout":1000}',
    })[0]

    assert call.valid is False
    assert "Unknown argument" in call.validation_error
