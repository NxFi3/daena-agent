from src.agent.planprogress import PlanProgressTracker
from src.agent.planstate import PlanState, PlanStepState
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


def active_plan() -> PlanState:
    return PlanState(
        exists=True,
        goal="test",
        steps=(
            PlanStepState(
                number=1,
                status="in_progress",
                description="Do work and verify it.",
            ),
        ),
    )


def test_plan_progress_can_attach_to_existing_active_plan():
    tracker = PlanProgressTracker()
    state = active_plan()

    tracker.sync(state, iteration=7, workspace_revision=3)
    tracker.record(
        ToolCall(
            name="read_file",
            id="read-2",
            valid=True,
            action="inspect",
            target="server.js",
        ),
        ToolResult(
            success=True,
            name="read_file",
            content={"path": "server.js", "content": "app"},
        ),
        iteration=8,
    )

    assert tracker.can_complete(1) is True
    assert tracker.context()["last_result_tool"] == "read_file"


def test_plan_progress_retains_work_done_before_plan_creation():
    tracker = PlanProgressTracker()

    tracker.record(
        ToolCall(
            name="read_file",
            id="read-before-plan",
            valid=True,
            action="inspect",
            target="agent.py",
        ),
        ToolResult(
            success=True,
            name="read_file",
            content={"path": "agent.py", "content": "agent"},
        ),
        iteration=2,
    )
    tracker.sync(active_plan(), iteration=4, workspace_revision=0)

    assert tracker.can_complete(1) is True
    assert tracker.context()["last_result_tool"] == "read_file"


def test_failed_tool_result_is_recorded_as_evidence_not_plan_policy():
    tracker = PlanProgressTracker()
    tracker.sync(active_plan(), iteration=1, workspace_revision=0)

    tracker.record(
        ToolCall(
            name="command_exec",
            id="test-1",
            valid=True,
            action="run",
        ),
        ToolResult(
            success=False,
            name="command_exec",
            content={
                "status": "exited",
                "exit_code": 1,
                "stdout": "FAIL one",
            },
        ),
        iteration=2,
    )

    context = tracker.context()
    assert context["last_result_success"] is False
    assert context["last_result_tool"] == "command_exec"
