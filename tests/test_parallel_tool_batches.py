"""Verify independent read batches and ordered receipts."""

from threading import Barrier
from types import SimpleNamespace

from src.agent.agentloop import Loop
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


def test_independent_read_calls_are_concurrent():
    barrier = Barrier(2)
    events = []
    results = []

    def execute(calls):
        barrier.wait(timeout=3)
        call = calls[0]
        return {"results": [ToolResult(True, call.name, {})]}

    runtime = SimpleNamespace(
        tool=SimpleNamespace(
            get_tool=lambda name: SimpleNamespace(parallel_safe=True),
            execute=execute,
        ),
        metrics={"tool_call_attempts": 0, "resource_blocks": 0},
        _stop_event=None,
        _emit_event=lambda kind, **kwargs: events.append((kind, kwargs)),
        _apply_result=lambda call, result, iteration: results.append(call.id),
    )
    calls = [
        ToolCall(name="read_file", id="first", valid=True),
        ToolCall(name="read_file", id="second", valid=True),
    ]
    Loop._execute_tool_calls(runtime, calls, 1)
    assert results == [call.id for call in calls]
    assert runtime.metrics["tool_call_attempts"] == 2


def test_third_identical_observation_is_blocked_without_execution():
    executions = []
    applied = []

    tool = SimpleNamespace(
        parallel_safe=True,
        duplicate_key=lambda arguments, workspace_root=None: dict(arguments),
    )

    def execute(calls):
        executions.append(calls[0].id)
        return {
            "results": [
                ToolResult(
                    True,
                    calls[0].name,
                    {"path": "/tmp/a", "content": "x"},
                )
            ]
        }

    runtime = SimpleNamespace(
        tool=SimpleNamespace(get_tool=lambda name: tool, execute=execute),
        metrics={"tool_call_attempts": 0, "resource_blocks": 0},
        _stop_event=None,
        _workspace_root=None,
        workspace_revision=0,
        _observation_counts={},
        _emit_event=lambda kind, **kwargs: None,
        _apply_result=lambda call, result, iteration: applied.append(result),
    )

    for index in range(3):
        call = ToolCall(
            name="read_file",
            id=f"read-{index}",
            valid=True,
            action="inspect",
            args={"file_path": "/tmp/a"},
        )
        Loop._execute_tool_calls(runtime, [call], index + 1)

    assert executions == ["read-0", "read-1"]
    assert len(applied) == 3
    assert applied[-1].success is False
    assert applied[-1].content["error"]["type"] == "duplicate_observation"
    assert applied[-1].metadata["duplicate_action"] is True


def test_observation_repeat_is_allowed_after_workspace_revision_changes():
    tool = SimpleNamespace(
        parallel_safe=True,
        duplicate_key=lambda arguments, workspace_root=None: dict(arguments),
    )
    runtime = SimpleNamespace(
        tool=SimpleNamespace(get_tool=lambda name: tool),
        _workspace_root=None,
        workspace_revision=0,
        _observation_counts={},
    )
    call = ToolCall(
        name="read_file",
        id="read",
        valid=True,
        action="inspect",
        args={"file_path": "/tmp/a"},
    )
    result = ToolResult(True, "read_file", {"path": "/tmp/a", "content": "x"})

    Loop._record_observation_success(runtime, call, result)
    Loop._record_observation_success(runtime, call, result)
    assert Loop._duplicate_observation_result(runtime, call) is not None

    runtime.workspace_revision = 1
    assert Loop._duplicate_observation_result(runtime, call) is None
