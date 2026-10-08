from uuid import uuid4

from src.agent.agentloop import Loop
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.models.ContextEvent import ContextRole, ContextType


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


class FakeLLM:
    def __init__(self):
        self.model = FakeModel()


def make_loop(tmp_path):
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {"safe_margin": 0, "compaction_enabled": False},
        "retrieval": {"top_k": 1},
        "security": {"workspace_only": True, "force_approve": True},
        "max_agent_iterations": 3,
        "experience": {"enabled": False},
    }
    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    return loop


def test_policy_gate_does_not_touch_loop_guard_or_failure_memory(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="command_exec",
            id="before-plan",
            valid=True,
            args={"command": ["python", "-m", "pytest", "-q"]},
        )
        loop._plan_required_this_run = True
        loop._plan_active_this_run = False

        # The legacy plan-required flag no longer gates repository work.
        # A plan policy check must never become a fake tool failure/recovery event.
        allowed, blocked = loop._classify_calls([call])
        assert allowed == [0]
        assert blocked == {}

        result = ToolResult(
            success=True,
            name="command_exec",
            content={"status": "exited", "exit_code": 0},
            metadata={},
        )
        loop._apply_result(call, result, 1)

        assert loop._failed_call_keys == {}
        assert loop._semantic_failure_counts == {}
        assert loop._recovery_mode is False
    finally:
        loop.close()


def test_plan_updates_do_not_advance_code_workspace_revision(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="plan",
            id="plan-create",
            valid=True,
            args={
                "action": "create",
                "goal": "Fix and verify",
                "steps": ["Inspect", "Implement", "Verify"],
            },
        )
        result = ToolResult(
            success=True,
            name="plan",
            content={"success": True},
            metadata={},
        )
        loop._apply_result(call, result, 1)

        assert loop._plan_active_this_run is True
        assert loop._verification_required is False
        assert loop.workspace_revision == 0
    finally:
        loop.close()


def test_verification_is_a_completion_gate_not_an_exploration_gate(tmp_path):
    loop = make_loop(tmp_path)
    try:
        loop._verification_required = True
        loop._recovery_mode = False

        read_call = ToolCall(
            name="read_file",
            id="post-mutation-read",
            valid=True,
            args={"file_path": "src/app.py"},
        )
        allowed, blocked = loop._classify_calls([read_call])
        assert allowed == [0]
        assert blocked == {}

        failed_test = ToolCall(
            name="command_exec",
            id="verify-failed",
            valid=True,
            args={"command": ["python", "-m", "pytest", "-q"]},
        )
        failed_result = ToolResult(
            success=False,
            name="command_exec",
            content={
                "status": "exited",
                "exit_code": 1,
                "error": {"type": "test_failure", "message": "one test failed"},
            },
            metadata={},
        )
        loop._apply_result(failed_test, failed_result, 1)

        assert loop._recovery_mode is True
        allowed, blocked = loop._classify_calls([read_call])
        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()
from pathlib import Path

from src.tools.builtin.plan.tool import Plan


def test_verified_active_plan_step_is_finalized_without_extra_llm_turn(tmp_path, monkeypatch):
    from src.agent.agentloop import Loop
    from src.models.ToolCall import ToolCall
    from src.models.ToolResult import ToolResult

    plan_path = tmp_path / "AgentInstruction" / "plan.md"
    monkeypatch.setattr(Plan, "PLAN_PATH", plan_path)

    loop = make_loop(tmp_path)
    try:
        plan_call = ToolCall(
            name="plan",
            id="plan-create",
            valid=True,
            args={
                "action": "create",
                "goal": "Fix and verify",
                "steps": ["Run verification"],
            },
        )
        plan_tool = loop.tool.get_tool("plan")
        plan_result = plan_tool.execute(
            action="create",
            goal="Fix and verify",
            steps=["Run verification"],
        )
        loop._apply_result(plan_call, plan_result, 1)

        verification_call = ToolCall(
            name="command_exec",
            id="verify",
            valid=True,
            args={"command": ["python", "-m", "pytest", "-q"]},
        )
        verification_result = ToolResult(
            success=True,
            name="command_exec",
            content={
                "command": ["python", "-m", "pytest", "-q"],
                "status": "exited",
                "exit_code": 0,
            },
            metadata={},
        )
        loop._apply_result(verification_call, verification_result, 2)

        assert loop._final_verification_satisfied is True
        assert loop._verification_required is False
        assert loop._auto_finalize_verified_plan(3) is True
        assert loop._read_plan_state().is_complete is True
    finally:
        loop.close()
