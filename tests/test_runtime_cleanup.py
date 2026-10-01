from __future__ import annotations

from tempfile import TemporaryFile
from types import SimpleNamespace

from src.context.contextbuilder import ContextBuilder
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolManager import ToolManager
from src.tools.ToolRegistry import ToolRegistry
from src.tools.builtin.command_exec.process_manager import ProcessManager
from src.tools.builtin.plan.tool import PlanTool


def test_tool_dispatcher_timeout_is_seconds():
    dispatcher = ToolDispatcher(ToolRegistry())
    call = dispatcher._dispatch_call(
        {
            "name": "command_exec",
            "arguments": {
                "command": ["python", "-c", "print(1)"],
                "timeout": 2,
            },
        }
    )
    assert call.valid
    assert call.args["yield_time_ms"] == 2000
    assert any("seconds" in note for note in call.normalization_notes)


def test_canonical_tool_identity_matches_relative_and_absolute_paths(tmp_path):
    manager = ToolManager({"security": {"workspace_only": True}})
    manager.set_workspace(str(tmp_path))

    relative = ToolCall(
        name="command_exec",
        valid=True,
        args={"command": ["pytest", "-q"], "workdir": "."},
    )
    absolute = ToolCall(
        name="command_exec",
        valid=True,
        args={"command": ["pytest", "-q"], "workdir": str(tmp_path.resolve())},
    )

    manager.canonicalize_tool_call(relative)
    manager.canonicalize_tool_call(absolute)

    assert relative.args == absolute.args


def test_process_manager_preserves_head_and_tail_of_large_increment(tmp_path):
    manager = ProcessManager()
    payload = (b"HEAD\n" + b"x" * 500 + b"\nTAIL\n")
    with TemporaryFile(mode="w+b") as stream:
        stream.write(payload)
        stream.flush()
        text, offset, truncated = manager._read_file_incremental(stream, 0, 64)

    assert offset == len(payload)
    assert truncated is True
    assert "HEAD" in text
    assert "TAIL" in text


def test_tool_result_truncation_preserves_failure_tail():
    result = ToolResult(
        success=False,
        name="command_exec",
        content={"error": {"message": "HEAD " + ("x" * 1000) + " TAIL"}},
    )
    assert "HEAD" in result.summary
    assert "TAIL" in result.summary


def test_plan_blocked_step_advances_to_next_pending(tmp_path):
    tool = PlanTool()
    tool.set_workspace(tmp_path)

    created = tool.execute(
        operation="create",
        goal="test",
        steps=["first", "second"],
    )
    assert created.success

    blocked = tool.execute(
        operation="update",
        step=1,
        status="blocked",
    )
    assert blocked.success

    state = tool.snapshot()
    assert state.steps[0].status == "blocked"
    assert state.steps[1].status == "in_progress"


def test_context_fit_pins_original_task():
    fake_llm = SimpleNamespace(model=None)
    builder = ContextBuilder(
        {
            "context": {
                "safe_margin": 0,
                "compaction_enabled": False,
            }
        },
        fake_llm,
    )
    builder.tokenbudget.budget = 120
    builder.tokenbudget.chars_per_token = 1
    builder._task_text = "ORIGINAL TASK"

    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "ORIGINAL TASK"},
    ]
    messages.extend(
        {"role": "user", "content": f"nudge-{i} " + ("x" * 30)}
        for i in range(10)
    )

    fitted = builder._fit_messages(messages)

    assert any(
        message.get("content") == "ORIGINAL TASK"
        for message in fitted
        if message.get("role") == "user"
    )


def test_tool_nudge_queue_is_separate_from_immediate_messages():
    # The loop uses a deferred queue for tool-result nudges so a multi-call
    # assistant response is stored as assistant(tool_calls) -> tool(results)
    # before any corrective user turn is appended.
    from src.agent.agentloop import Loop

    loop = Loop.__new__(Loop)
    loop._pending_tool_nudges = []
    loop._queue_tool_nudge("one")
    loop._queue_tool_nudge("two")
    assert loop._pending_tool_nudges == ["one", "two"]
