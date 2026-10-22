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
