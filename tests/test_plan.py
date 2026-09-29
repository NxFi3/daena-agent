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

    content = plan_path.read_text(encoding="utf-8")
    assert "# Plan" in content
    assert "## Instructions" in content
    assert "The agent MUST keep this plan synchronized with actual work." in content
    assert "1. [pending] Inspect authentication architecture" in content
    assert "2. [pending] Implement login" in content
    assert "3. [pending] Run tests" in content

    result = tool.execute(
        operation="update",
        step=1,
        status="in_progress",
    )
    assert result.success is True

    result = tool.execute(
        operation="update",
        step=1,
        status="completed",
    )
    assert result.success is True

    result = tool.execute(
        operation="update",
        step=2,
        description="Implement JWT login",
    )
    assert result.success is True
    assert "2. [pending] Implement JWT login" in plan_path.read_text(
        encoding="utf-8"
    )

    result = tool.execute(
        operation="update",
        add_step="Add integration tests",
    )
    assert result.success is True
    assert "4. [pending] Add integration tests" in plan_path.read_text(
        encoding="utf-8"
    )

    result = tool.execute(
        operation="update",
        remove_step=4,
    )
    assert result.success is True

    result = tool.execute(
        operation="update",
        goal="Secure authentication",
    )
    assert result.success is True
    assert "Secure authentication" in plan_path.read_text(
        encoding="utf-8"
    )

    duplicate = tool.execute(
        operation="update",
        add_step="Implement JWT login",
    )
    assert duplicate.success is False
    assert duplicate.content["error"]["type"] == "duplicate_step"

    exists = tool.execute(
        operation="create",
        goal="Another plan",
        steps=["Should fail"],
    )
    assert exists.success is False
    assert exists.content["error"]["type"] == "plan_exists"

    deleted = tool.execute(operation="delete")
    assert deleted.success is True
    assert not plan_path.exists()

    deleted_again = tool.execute(operation="delete")
    assert deleted_again.success is True
