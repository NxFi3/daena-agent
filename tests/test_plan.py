from pathlib import Path

from src.tools.builtin.plan.tool import Plan


def make_tool(tmp_path, monkeypatch):
    plan_path = tmp_path / ".daena" / "plan.md"
    monkeypatch.setattr(Plan, "PLAN_PATH", Path(plan_path))
    return Plan(), plan_path


def test_plan_create_complete_block_add(tmp_path, monkeypatch):
    tool, plan_path = make_tool(tmp_path, monkeypatch)

    result = tool.execute(
        action="create",
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
    assert "1. [in_progress] Inspect authentication architecture" in content
    assert "2. [pending] Implement login" in content
    assert "3. [pending] Run tests" in content

    # The runtime validates whether completion has real work evidence; the
    # storage primitive itself only applies the requested lifecycle transition.
    result = tool.execute(action="block", reason="Architecture is unavailable")
    assert result.success is True
    assert "2. [in_progress] Implement login" in plan_path.read_text(encoding="utf-8")

    result = tool.execute(action="add", step="Add integration tests")
    assert result.success is True
    assert "4. [pending] Add integration tests" in plan_path.read_text(encoding="utf-8")

    duplicate = tool.execute(action="add", step="Add integration tests")
    assert duplicate.success is False
    assert duplicate.content["error"]["type"] == "duplicate_step"


def test_plan_complete_action_advances_current_step(tmp_path, monkeypatch):
    tool, plan_path = make_tool(tmp_path, monkeypatch)
    tool.execute(
        action="create",
        goal="Ship CLI",
        steps=["Implement", "Verify"],
    )

    # Unit-test the storage primitive directly by calling the helper after the
    # runtime has established that the step is complete.
    result = tool._set_current_status("completed")
    assert result.success is True

    content = plan_path.read_text(encoding="utf-8")
    assert "1. [completed] Implement" in content
    assert "2. [in_progress] Verify" in content


def test_plan_public_validation_is_minimal():
    tool = Plan()

    assert tool.validate({"action": "create", "goal": "Ship", "steps": ["Build"]}) is True
    assert tool.validate({"action": "complete"}) is True
    assert tool.validate({"action": "block", "reason": "dependency unavailable"}) is True
    assert tool.validate({"action": "add", "step": "Run tests"}) is True

    assert tool.validate({"action": "complete", "step": 2}) is False
    assert tool.validate({"operation": "update", "step": 2, "status": "completed"}) is False
    assert tool.validate({"action": "delete"}) is False


def test_plan_create_is_rejected_when_plan_already_exists(tmp_path, monkeypatch):
    tool, _ = make_tool(tmp_path, monkeypatch)
    first = tool.execute(
        action="create",
        goal="Ship",
        steps=["Build"],
    )
    assert first.success is True

    second = tool.execute(
        action="create",
        goal="Replace",
        steps=["Something else"],
    )
    assert second.success is False
    assert second.content["error"]["type"] == "plan_exists"
