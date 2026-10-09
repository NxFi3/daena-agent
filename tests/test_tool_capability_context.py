from pathlib import Path
from uuid import uuid4

from src.agent.agentloop import Loop
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


class FakeLLM:
    def __init__(self):
        self.model = FakeModel()
        self.captured = None
        self.captured_tools = None

    def generate(self, messages, tools=None, **_kwargs):
        self.captured = messages
        self.captured_tools = tools
        return LLMResult(
            response="done",
            message={"role": "assistant", "content": "done"},
            tool_calls=[],
            thinking=None,
            usage=1,
        )


def test_context_exposes_specialized_web_tools_and_failure_recovery_guidance(tmp_path):
    llm = FakeLLM()
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 20,
            "compaction_enabled": False,
        },
        "retrieval": {"top_k": 1},
        "security": {
            "workspace_only": True,
            "allow_background": True,
            "allow_network_tools": True,
            "force_approve": True,
        },
        "memory": {"stm_db_path": str(tmp_path / "stm.db")},
        "max_agent_iterations": 5,
        "experience": {"enabled": False},
    }
    loop = Loop(config, llm)
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Find current house-sale listing URLs in Kashan.",
        )
        result = loop._generate_next_action(task, str(tmp_path))

        assert result is not None
        rendered = "\n".join(str(message.get("content", "")) for message in llm.captured)
        assert "web_search" in rendered
        assert "web_fetch" in rendered
        assert any(
            item.get("function", {}).get("name") == "write_file"
            for item in (llm.captured_tools or [])
        )
        assert any(
            item.get("function", {}).get("name") == "context_search"
            for item in (llm.captured_tools or [])
        )
        assert "A 404 or empty extraction only describes that URL/response" in rendered
        assert "choose a materially different approach" in rendered
    finally:
        loop.close()
