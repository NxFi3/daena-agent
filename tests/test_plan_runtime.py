from src.agent.agentloop import Loop
from src.agent.planstate import PlanState, PlanStepState
from src.models.ToolCall import ToolCall


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


def test_plan_completion_is_blocked_after_unresolved_failure():
    loop = Loop.__new__(Loop)
    loop._plan_step_work_started = True
    loop._plan_step_has_unresolved_failure = True

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is not None
    assert error[0] == "completion_has_unresolved_failure"


def test_plan_completion_is_allowed_after_failure_is_resolved():
    loop = Loop.__new__(Loop)
    loop._plan_step_work_started = True
    loop._plan_step_has_unresolved_failure = False

    error = loop._validate_plan_transition(
        plan_completion_call(),
        active_plan(),
    )

    assert error is None
