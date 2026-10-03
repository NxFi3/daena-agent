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
        args={"action": "complete"},
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


def test_plan_completion_requires_successful_terminal_work():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is not None
    assert error[0] == "completion_requires_success"


def test_plan_completion_allows_successful_terminal_work():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )
    loop._plan_progress.record(
        ToolCall(
            name="apply_patch",
            id="patch-1",
            valid=True,
            action="modify",
            target="app.py",
        ),
        ToolResult(
            success=True,
            name="apply_patch",
            content={"files": [{"path": "app.py", "operation": "update", "content": "x"}]},
        ),
        iteration=2,
    )

    assert loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    ) is None


def test_plan_completion_rejects_running_work():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )
    loop._plan_progress.record(
        ToolCall(
            name="command_exec",
            id="proc-1",
            valid=True,
            action="run",
        ),
        ToolResult(
            success=True,
            name="command_exec",
            content={
                "status": "running",
                "process_id": "proc-test",
            },
        ),
        iteration=2,
    )

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is not None
    assert error[0] == "completion_requires_terminal_result"


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
    loop._plan_progress.record(
        ToolCall(
            name="command_exec",
            id="test-1",
            valid=True,
            action="run",
        ),
        failed,
        iteration=1,
    )

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is not None
    assert error[0] == "completion_requires_success"


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



def test_plan_progress_can_attach_to_an_existing_active_plan():
    loop = _loop_for_validation()
    state = active_plan()

    loop._plan_progress.sync(
        state,
        iteration=7,
        workspace_revision=3,
    )
    loop._plan_progress.record(
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
            content={
                "path": "server.js",
                "content": "app",
            },
        ),
        iteration=8,
    )

    assert loop._plan_progress.can_complete(1) is True



def test_plan_completion_rejects_live_process_receipt():
    loop = _loop_for_validation()
    loop._plan_progress.sync(
        active_plan(),
        iteration=1,
        workspace_revision=0,
    )
    loop._plan_progress.record(
        ToolCall(
            name="process_write",
            id="write-1",
            valid=True,
            action="modify",
        ),
        ToolResult(
            success=True,
            name="process_write",
            content={
                "status": "accepted",
                "process_id": "proc-live",
            },
        ),
        iteration=2,
    )

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is not None
    assert error[0] == "completion_requires_terminal_result"


def test_plan_progress_retains_successful_work_done_before_plan_creation():
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

    tracker.sync(
        active_plan(),
        iteration=4,
        workspace_revision=0,
    )

    assert tracker.can_complete(1) is True
    assert tracker.context()["last_result_tool"] == "read_file"


def test_plan_file_observation_is_blocked_at_runtime():
    loop = Loop.__new__(Loop)

    call = ToolCall(
        name="read_file",
        id="plan-read",
        valid=True,
        args={"file_path": ".daena/plan.md"},
    )

    gate = loop._plan_gate_message(call, active_plan())

    assert gate is not None
    assert gate[0] == "plan_internal_state"


def test_plan_actions_do_not_require_model_selected_step_numbers():
    loop = _loop_for_validation()

    error = loop._validate_plan_transition(
        ToolCall(
            name="plan",
            id="complete-current",
            valid=True,
            args={"action": "complete"},
        ),
        active_plan(),
    )

    assert error is not None
    assert error[0] == "completion_requires_success"
