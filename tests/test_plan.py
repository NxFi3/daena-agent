from pathlib import Path

from src.tools.builtin.plan.tool import Plan


def test_plan_create_update_delete(tmp_path, monkeypatch):
    plan_path = tmp_path / ".daena" / "plan.md"
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
    assert "1. [in_progress] Inspect authentication architecture" in content
    assert "2. [pending] Implement login" in content
    assert "3. [pending] Run tests" in content

    result = tool.execute(
        operation="update",
        step=1,
        status="completed",
    )
    assert result.success is True

    content = plan_path.read_text(encoding="utf-8")
    assert "1. [completed] Inspect authentication architecture" in content
    assert "2. [in_progress] Implement login" in content

    result = tool.execute(
        operation="update",
        step=2,
        description="Implement JWT login",
    )
    assert result.success is True
    assert "2. [in_progress] Implement JWT login" in plan_path.read_text(
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

    result = tool.execute(
        operation="update",
        step=2,
        status="completed",
    )
    assert result.success is True

    content = plan_path.read_text(encoding="utf-8")
    assert "2. [completed] Implement JWT login" in content
    assert "3. [in_progress] Run tests" in content

    result = tool.execute(
        operation="update",
        step=3,
        status="blocked",
    )
    assert result.success is True

    plan_state = tool.snapshot()
    assert plan_state.is_complete is True

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


def test_plan_accepts_inferred_operation_when_omitted(tmp_path, monkeypatch):
    plan_path = tmp_path / ".daena" / "plan.md"
    monkeypatch.setattr(Plan, "PLAN_PATH", Path(plan_path))
    tool = Plan()

    result = tool.execute(
        goal="Build authentication",
        steps=["Inspect code", "Implement change"],
    )

    assert result.success is True
    assert "1. [in_progress] Inspect code" in plan_path.read_text(encoding="utf-8")

    result = tool.execute(
        step=1,
        status="completed",
    )

    assert result.success is True
    assert "2. [in_progress] Implement change" in plan_path.read_text(encoding="utf-8")
    assert "step 2 is now in_progress" in result.summary


def test_plan_normalizes_inferred_operation_for_runtime_dispatch(tmp_path, monkeypatch):
    plan_path = tmp_path / ".daena" / "plan.md"
    monkeypatch.setattr(Plan, "PLAN_PATH", Path(plan_path))
    tool = Plan()

    normalized, notes = tool.normalize_arguments({
        "step": 1,
        "status": "completed",
    })

    assert normalized["operation"] == "update"
    assert "operation inferred as 'update'." in notes

    normalized, notes = tool.normalize_arguments({
        "goal": "Ship CLI",
        "steps": ["Implement", "Test"],
    })

    assert normalized["operation"] == "create"
    assert "operation inferred as 'create'." in notes
