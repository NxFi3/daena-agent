from pathlib import Path

from src.tools.builtin.plan.tool import Plan


def test_plan_create_update_delete(tmp_path, monkeypatch):
    plan_path = tmp_path / "AgentInstruction" / "plan.md"
    monkeypatch.setattr(Plan, "PLAN_PATH", Path(plan_path))

    tool = Plan()

    result = tool.execute(
        operation="create",
        goal="Build authentication",
        steps=[
            "Inspect authentication architecture",
            "Implement login",
            "Run tests",
        ],
    )

    assert result.success is True
    assert result.content["summary"] == "Plan created with 3 step(s)."

    assert plan_path.read_text(encoding="utf-8") == (
        "# Plan\n\n"
        "## Goal\n"
        "Build authentication\n\n"
        "## Steps\n\n"
        "1. [pending] Inspect authentication architecture\n"
        "2. [pending] Implement login\n"
        "3. [pending] Run tests\n"
    )

    result = tool.execute(
        operation="update",
        action="set_step",
        step=2,
        status="completed",
    )
    assert result.success is True
    assert "step 2 completed." in result.content["summary"]

    result = tool.execute(
        operation="update",
        action="set_step",
        step=2,
        description_text="Implement JWT login",
    )
    assert result.success is True
    assert "Implement JWT login" in plan_path.read_text(
        encoding="utf-8"
    )

    result = tool.execute(
        operation="update",
        action="add_step",
        after=2,
        description_text="Add integration tests",
    )
    assert result.success is True
    assert "3. [pending] Add integration tests" in plan_path.read_text(
        encoding="utf-8"
    )

    result = tool.execute(
        operation="update",
        action="remove_step",
        step=1,
    )
    assert result.success is True
    assert "1. [completed] Implement JWT login" in plan_path.read_text(
        encoding="utf-8"
    )

    result = tool.execute(
        operation="update",
        action="set_goal",
        goal="Secure authentication",
    )
    assert result.success is True
    assert "Secure authentication" in plan_path.read_text(
        encoding="utf-8"
    )

    result = tool.execute(
        operation="create",
        goal="Another plan",
        steps=["Should fail"],
    )
    assert result.success is False
    assert result.content["error"]["type"] == "plan_exists"

    result = tool.execute(operation="delete")
    assert result.success is True
    assert not plan_path.exists()

    result = tool.execute(operation="delete")
    assert result.success is True
