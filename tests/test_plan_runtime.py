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


def test_plan_completion_is_structural_not_tool_specific():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )

    assert loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    ) is None


def test_failed_verification_does_not_create_a_runtime_semantic_gate():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )

    failed = ToolResult(
        success=False,
        name="command_exec",
        content={
            "command": ["npm", "test"],
            "workdir": ".",
            "status": "exited",
            "exit_code": 1,
            "stdout": "FAIL one",
        },
    )
    loop.working_set.update(
        ToolCall(
            name="command_exec",
            id="test-1",
            valid=True,
            args={"command": ["npm", "test"], "workdir": "."},
        ),
        failed,
        1,
    )

    assert loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    ) is None


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


def test_missing_active_step_does_not_block_recovery_work():
    loop = Loop.__new__(Loop)
    state = PlanState(
        exists=True,
        goal="test",
        steps=(
            PlanStepState(number=1, status="completed", description="Done"),
            PlanStepState(number=2, status="pending", description="Continue"),
        ),
    )

    assert loop._plan_gate_message(regular_tool_call(), state) is None
