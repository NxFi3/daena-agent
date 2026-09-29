from pathlib import Path
from uuid import uuid4

from src.agent.agentloop import Loop
from src.agent.planstate import PlanState
from src.models.ToolCall import ToolCall
from src.tools.builtin.plan.tool import Plan


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


class FakeLLM:
    def __init__(self):
        self.model = FakeModel()

    def generate(self, messages, tools=None):
        raise AssertionError("LLM generation is not used by these unit tests")


def config():
    return {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 4096},
            }
        },
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 40,
            "compaction_enabled": False,
        },
        "retrieval": {"top_k": 3},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": True,
        },
        "max_agent_iterations": 5,
        "experience": {"enabled": False},
    }


def command_call():
    return ToolCall(
        name="command_exec",
        id=str(uuid4()),
        args={
            "command": ["echo", "hello"],
            "workdir": ".",
        },
        valid=True,
        action="run",
        target="echo hello",
    )


def plan_call(step: int, status: str):
    return ToolCall(
        name="plan",
        id=str(uuid4()),
        args={
            "operation": "update",
            "step": step,
            "status": status,
        },
        valid=True,
        action="modify",
        target="AgentInstruction/plan.md",
    )


def test_plan_state_snapshot(tmp_path, monkeypatch):
    plan_path = tmp_path / "AgentInstruction" / "plan.md"
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
    assert state.current_step is None
    assert state.next_pending_step is not None
    assert state.next_pending_step.number == 1
    assert state.is_complete is False

    result = tool.execute(
        operation="update",
        step=1,
        status="in_progress",
    )
    assert result.success is True

    state = tool.snapshot()
    assert state.current_step is not None
    assert state.current_step.number == 1
    assert state.current_step.status == "in_progress"


def test_loop_plan_gate_enforces_step_lifecycle(tmp_path, monkeypatch):
    plan_path = tmp_path / "AgentInstruction" / "plan.md"
    monkeypatch.setattr(Plan, "PLAN_PATH", Path(plan_path))

    plan_tool = Plan()
    assert plan_tool.execute(
        operation="create",
        goal="Build something",
        steps=["Inspect files", "Implement changes"],
    ).success

    loop = Loop(config(), FakeLLM())
    loop.session_id = uuid4()

    try:
        command = command_call()

        allowed, blocked = loop._classify_calls([command])
        assert allowed == []
        assert blocked[0].content["error"]["type"] == "step_start_required"

        start = plan_call(1, "in_progress")
        allowed, blocked = loop._classify_calls([start, command])

        assert allowed == [0]
        assert 1 in blocked

        assert plan_tool.execute(
            operation="update",
            step=1,
            status="in_progress",
        ).success
        loop._plan_step_work_started = False

        complete = plan_call(1, "completed")
        allowed, blocked = loop._classify_calls([complete])
        assert allowed == []
        assert blocked[0].content["error"]["type"] == "completion_requires_work"

        loop._plan_step_work_started = True
        allowed, blocked = loop._classify_calls([complete, command])
        assert allowed == [0]
        assert blocked[1].content["error"]["type"] == "plan_update_required_first"
    finally:
        loop.close()
