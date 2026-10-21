from pathlib import Path
from uuid import uuid4

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


def plan_call(action: str):
    return ToolCall(
        name="plan",
        id=str(uuid4()),
        args={"action": action},
        valid=True,
        action="modify",
        target=".daena/plan.md",
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
    assert state.current_step is not None
    assert state.current_step.number == 1
    assert state.current_step.status == "in_progress"
    assert state.next_pending_step is not None
    assert state.next_pending_step.number == 2
    assert state.is_complete is False
