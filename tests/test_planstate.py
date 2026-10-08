from pathlib import Path

from src.agent.planstate import PlanState
from src.tools.builtin.plan.tool import Plan


def test_plan_state_snapshot(tmp_path, monkeypatch):
    plan_path = tmp_path / ".daena" / "plan.md"
    monkeypatch.setattr(Plan, "PLAN_PATH", Path(plan_path))

    tool = Plan()
    assert tool.snapshot() == PlanState.empty()

    result = tool.execute(
        operation="create",
        goal="Build authentication",
        steps=["Inspect files", "Implement login"],
    )
    assert result.success is True

    state = tool.snapshot()
    assert state.exists is True
    assert state.goal == "Build authentication"
    assert state.current_step is not None
    assert state.current_step.number == 1
    assert state.current_step.status == "in_progress"
    assert state.next_pending_step is not None
    assert state.next_pending_step.number == 2
    assert state.is_complete is False
