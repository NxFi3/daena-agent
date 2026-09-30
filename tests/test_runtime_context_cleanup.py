from pathlib import Path

from src.context.contextbuilder import ContextBuilder, PlanReader
from src.context.contextwindow import ContextWindow
from src.context.workingset import WorkingSet
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.tools.ToolManager import ToolManager
from src.tools.builtin.plan.tool import Plan


def test_plan_uses_active_workspace(tmp_path):
    tool = Plan()
    tool.set_workspace(tmp_path)

    result = tool.execute(
        operation="create",
        goal="Workspace-scoped plan",
        steps=["Create a file"],
    )

    assert result.success is True

    plan_path = tmp_path / ".daena" / "plan.md"
    assert plan_path.exists()
    assert tool.describe_call({})["target"] == str(plan_path)


def test_tool_manager_propagates_workspace_to_plan(tmp_path):
    manager = ToolManager()
    manager.set_workspace(str(tmp_path))

    plan = manager.get_tool("plan")
    result = plan.execute(
        operation="create",
        goal="Manager workspace",
        steps=["Create a file"],
    )

    assert result.success is True
    assert (tmp_path / ".daena" / "plan.md").exists()


def test_plan_reader_uses_workspace_scope(tmp_path):
    plan_path = tmp_path / ".daena" / "plan.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("workspace plan", encoding="utf-8")

    assert PlanReader(tmp_path) == "workspace plan"


def test_context_builder_loads_workspace_plan_automatically(tmp_path):
    plan_path = tmp_path / ".daena" / "plan.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        "# Plan\n\n## Goal\nKeep the project organized\n",
        encoding="utf-8",
    )

    builder = ContextBuilder.__new__(ContextBuilder)
    builder.window = ContextWindow()
    builder.system_instruction = ""
    builder.experience_enabled = False

    builder._populate_window(
        events=[],
        task=None,
        workspace=str(tmp_path),
        learned_experience=None,
        execution_state=None,
    )

    assert "Keep the project organized" in builder.window.plan


def test_working_set_resolves_relative_paths_from_workspace(tmp_path):
    working_set = WorkingSet()
    working_set.set_workspace(str(tmp_path))

    call = ToolCall(
        name="plan",
        action="modify",
        target=".daena/plan.md",
    )
    result = ToolResult(
        success=True,
        name="plan",
        content={"summary": "Plan updated."},
        metadata={
            "effects": [
                {
                    "action": "modify",
                    "target": ".daena/plan.md",
                }
            ]
        },
        summary="Plan updated.",
    )

    working_set.update(call, result, iteration=1)

    assert str(tmp_path / ".daena" / "plan.md") in working_set.context()[
        "artifacts"
    ]


def test_recovered_validation_failure_is_removed_from_unresolved(tmp_path):
    working_set = WorkingSet(str(tmp_path))

    failed_call = ToolCall(
        name="plan",
        action="modify",
        valid=False,
        validation_error="Missing required argument(s): operation.",
    )
    failed_result = ToolResult(
        success=False,
        name="plan",
        content={
            "error": {
                "type": "invalid_tool_call",
                "message": "Missing required argument(s): operation.",
            }
        },
        summary="Missing required argument(s): operation.",
    )

    working_set.update(failed_call, failed_result, iteration=1)
    assert working_set.context()["unresolved"]

    successful_call = ToolCall(
        name="plan",
        action="modify",
        valid=True,
        target=str(tmp_path / ".daena" / "plan.md"),
    )
    successful_result = ToolResult(
        success=True,
        name="plan",
        content={"summary": "Plan updated."},
        metadata={},
        summary="Plan updated.",
    )

    working_set.update(successful_call, successful_result, iteration=2)

    assert working_set.context()["unresolved"] == []


def test_context_does_not_feed_provider_thinking_back_to_model():
    builder = ContextBuilder.__new__(ContextBuilder)

    message = builder._assistant_message(
        "",
        {
            "llm_message": {
                "content": "",
                "thinking": "private reasoning that should stay internal",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "plan",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        },
    )

    assert message is not None
    assert "thinking" not in message
    assert "tool_calls" in message
