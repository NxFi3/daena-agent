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



def test_dispatcher_recovers_without_registry_tools_map():
    class Registry:
        def is_available(self, name):
            return name == "read_file"

        def get(self, name):
            return StrictTool() if name == "read_file" else None

    dispatcher = ToolDispatcher(Registry())
    call = dispatcher.dispatch({
        "name": "read_file",
        "arguments": {"file_path": "main.py", "timeout": 1000},
    })[0]

    assert call.valid is False
    assert "Unknown argument" in call.validation_error

def test_dispatcher_rejects_unknown_args_for_strict_tools():
    dispatcher = ToolDispatcher(StrictRegistry())
    call = dispatcher.dispatch({
        "name": "read_file",
        "arguments": '{"file_path":"main.py","timeout":1000}',
    })[0]

    assert call.valid is False
    assert "Unknown argument" in call.validation_error


def test_command_exec_yield_time_alias_is_normalized():
    from src.tools.ToolDispatcher import ToolDispatcher
    from src.tools.ToolRegistry import ToolRegistry

    registry = ToolRegistry()
    registry.discover()
    dispatcher = ToolDispatcher(registry)

    calls = dispatcher.dispatch([{
        "id": "call-1",
        "name": "command_exec",
        "arguments": {
            "command": ["echo", "ok"],
            "yield_time": "1000",
        },
    }])

    assert len(calls) == 1
    assert calls[0].valid is True
    assert calls[0].args["yield_time_ms"] == 1000

def test_dispatcher_normalizes_trailing_punctuation_in_tool_name():
    dispatcher = ToolDispatcher(FakeRegistry())
    call = dispatcher.dispatch({
        "name": "read_file?",
        "arguments": {"file_path": "main.py"},
    })[0]

    assert call.valid is True
    assert call.name == "read_file"


def test_dispatcher_repairs_unique_misrouted_read_file_call():
    class PlanTool:
        parameters = {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
            },
            "additionalProperties": False,
        }

        def validate(self, args):
            return True

    class ReadTool(FakeTool):
        parameters = {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["file_path"],
            "additionalProperties": False,
        }

    class Registry:
        tools = {"plan": PlanTool(), "read_file": ReadTool()}

        def is_available(self, name):
            return name in self.tools

        def get(self, name):
            return self.tools.get(name)

    dispatcher = ToolDispatcher(Registry())
    call = dispatcher.dispatch({
        "name": "plan",
        "arguments": {
            "file_path": "agent.py",
            "start_line": 1,
            "end_line": 20,
        },
    })[0]

    assert call.valid is True
    assert call.name == "read_file"
    assert call.args["file_path"] == "agent.py"
    assert call.args["start_line"] == 1
    assert call.args["end_line"] == 20
    assert any("repaired" in note.lower() for note in call.normalization_notes)


def test_dispatcher_drops_read_file_result_fields_from_input():
    from src.tools.builtin.readfile import ReadFile

    class Registry:
        def is_available(self, name):
            return name == "read_file"

        def get(self, name):
            return ReadFile() if name == "read_file" else None

    dispatcher = ToolDispatcher(Registry())
    call = dispatcher.dispatch({
        "name": "read_file",
        "arguments": {
            "file_path": "agent.py",
            "start_line": 1,
            "lines_requested": 50,
            "lines_returned": 50,
            "total_lines": 100,
            "truncated": False,
            "content": "copied output",
        },
    })[0]

    assert call.valid is True
    assert call.args == {
        "file_path": "agent.py",
        "start_line": 1,
    }
    assert any("lines_requested" in note for note in call.normalization_notes)


def test_dispatcher_normalizes_wait_ms_alias():
    dispatcher = ToolDispatcher(CommandRegistry())
    calls = dispatcher.dispatch({
        "name": "command_exec",
        "arguments": {
            "command": ["pytest", "-q"],
            "wait_ms": 5000,
        },
    })

    call = calls[0]
    assert call.valid is True
    assert call.args["yield_time_ms"] == 5000
    assert "wait_ms" not in call.args
