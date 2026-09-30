from src.agent.agentloop import Loop
from src.agent.planstate import PlanState, PlanStepState
from src.agent.planprogress import PlanProgressTracker
from src.context.workingset import WorkingSet
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


def plan_completion_call() -> ToolCall:
    return ToolCall(
        name="plan",
        id="plan-1",
        valid=True,
        args={
            "operation": "update",
            "step": 1,
            "status": "completed",
        },
    )


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


def _loop_for_validation() -> Loop:
    loop = Loop.__new__(Loop)
    loop._plan_progress = PlanProgressTracker()
    loop.working_set = WorkingSet()
    return loop


def test_plan_completion_requires_successful_step_work():
    loop = _loop_for_validation()

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is not None
    assert error[0] == "completion_requires_work"

    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )
    loop._plan_progress.record(
        ToolCall(
            name="read_file",
            id="read-1",
            valid=True,
            args={"file_path": "result.txt"},
        ),
        ToolResult(
            success=True,
            name="read_file",
            content={"path": "result.txt", "content": "ok"},
        ),
    )

    assert loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    ) is None


def test_failed_verification_blocks_step_completion_until_resolved():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )
    loop._plan_progress.record(
        ToolCall(
            name="command_exec",
            id="test-1",
            valid=True,
            args={"command": ["npm", "test"], "workdir": "."},
        ),
        ToolResult(
            success=False,
            name="command_exec",
            content={
                "command": ["npm", "test"],
                "workdir": ".",
                "status": "exited",
                "exit_code": 1,
                "stdout": "FAIL one",
            },
        ),
    )
    # Update WorkingSet too: completion should be governed by durable evidence,
    # not by whether the most recent unrelated action happened to succeed.
    loop.working_set.update(
        ToolCall(
            name="command_exec",
            id="test-1",
            valid=True,
            args={"command": ["npm", "test"], "workdir": "."},
        ),
        ToolResult(
            success=False,
            name="command_exec",
            content={
                "command": ["npm", "test"],
                "workdir": ".",
                "status": "exited",
                "exit_code": 1,
                "stdout": "FAIL one",
            },
        ),
        1,
    )

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )
    assert error is not None
    assert error[0] == "completion_has_unresolved_failure"


def test_incomplete_plan_blocks_final_response():
    loop = Loop.__new__(Loop)

    gate = loop._final_response_gate.__get__(loop)
    # The helper reads runtime plan state; replace it with a deterministic
    # snapshot for this unit test.
    loop._read_plan_state = lambda: active_plan()

    result = gate()

    assert result is not None
    assert result[0] == "plan_incomplete"


def test_plan_completion_is_allowed_after_failure_is_resolved():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )
    loop._plan_progress.record(
        ToolCall(
            name="read_file",
            id="read-1",
            valid=True,
            args={"file_path": "result.txt"},
        ),
        ToolResult(
            success=True,
            name="read_file",
            content={"path": "result.txt", "content": "ok"},
        ),
    )

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is None


def complete_plan() -> PlanState:
    return PlanState(
        exists=True,
        goal="test",
        steps=(
            PlanStepState(
                number=1,
                status="completed",
                description="Do work and verify it.",
            ),
        ),
    )


def regular_tool_call() -> ToolCall:
    return ToolCall(
        name="read_file",
        id="read-1",
        valid=True,
        args={"file_path": "result.txt"},
    )


def test_completed_plan_does_not_block_follow_up_tools():
    loop = Loop.__new__(Loop)

    gate = loop._plan_gate_message(
        regular_tool_call(),
        complete_plan(),
    )

    assert gate is None

    
def test_redundant_start_of_current_step_is_allowed():
    loop = Loop.__new__(Loop)

    error = loop._validate_plan_transition(
        ToolCall(
            name="plan",
            id="plan-2",
            valid=True,
            args={
                "operation": "update",
                "step": 1,
                "status": "in_progress",
            },
        ),
        active_plan(),
    )

    assert error is None
