from uuid import uuid4

from src.agent.agentloop import Loop
from src.models.ToolCall import ToolCall


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
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": True,
        },
        "max_agent_iterations": 3,
        "experience": {"enabled": False},
    }
    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    return loop


def test_agent_loop_does_not_turn_security_into_strategy_gating(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="command_exec",
            id="blocked-command",
            valid=True,
            args={"command": ["rm", "important.txt"], "workdir": "."},
        )

        allowed, blocked = loop._classify_calls([call])

        # Strategy layer: valid model decision remains executable.
        assert allowed == [0]
        assert blocked == {}

        # Security layer: actual execution is still fail-closed.
        result = loop.tool.execute([call])["results"][0]
        assert result.success is False
        assert result.content["error"]["type"] == "security_denied"
        assert result.metadata["security_rule"] == "blocked_executable"
    finally:
        loop.close()


def test_workspace_boundary_remains_runtime_enforced(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="read_file",
            id="outside",
            valid=True,
            args={"file_path": "../outside.txt"},
        )

        result = loop.tool.execute([call])["results"][0]

        assert result.success is False
        assert result.content["error"]["type"] == "security_denied"
        assert result.metadata["security_rule"] == "workspace_boundary"
    finally:
        loop.close()
