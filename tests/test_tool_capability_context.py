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
        self.captured_options = None

    def generate(self, messages, tools=None, **_kwargs):
        self.captured = messages
        self.captured_tools = tools
        self.captured_options = _kwargs.get("options")
        return LLMResult(
            response="done",
            message={"role": "assistant", "content": "done"},
            tool_calls=[],
            thinking=None,
            usage=1,
        )


def test_context_exposes_tool_search_without_eagerly_sending_every_schema(tmp_path):
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
        captured_names = {
            item.get("function", {}).get("name")
            for item in (llm.captured_tools or [])
        }
        assert "tool_search" in captured_names
        assert {
            "command_exec", "read_file", "write_file", "edit_file", "apply_patch"
        }.issubset(captured_names)
        assert "tool_search" in rendered
        assert "available on the next turn" in rendered
        assert "Shell, file, patch, and managed-process tools marked always-exposed" in rendered
        assert "web_search" not in captured_names
        assert "context_search" not in captured_names
        assert "computer_use" not in captured_names
        assert "If a search finds nothing, try a concrete alternate query" in rendered
    finally:
        loop.close()


def test_computer_use_mode_disables_long_reasoning_by_default(tmp_path):
    llm = FakeLLM()
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096, "think": "low"}}},
        "context": {"safe_margin": 0, "recent_event_limit": 10, "compaction_enabled": False},
        "retrieval": {"top_k": 1},
        "security": {"workspace_only": True, "force_approve": True},
        "memory": {"stm_db_path": str(tmp_path / "stm-fast-cua.db")},
        "max_agent_iterations": 5,
        "experience": {"enabled": False},
    }
    loop = Loop(config, llm)
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    loop._computer_use_mode_active = True
    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Continue the current desktop task.",
        )
        result = loop._generate_next_action(task, str(tmp_path))
        assert result is not None
        assert llm.captured_options == {"think": False}
        rendered = "\n".join(str(message.get("content", "")) for message in llm.captured)
        assert "COMPUTER USE FAST PATH" in rendered
    finally:
        loop.close()
