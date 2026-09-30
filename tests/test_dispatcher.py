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
