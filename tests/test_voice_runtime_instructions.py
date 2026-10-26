"""Voice is a transient system mode, never another message in task history."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.voice.instructions import VOICE_SYSTEM_INSTRUCTION


class Model:
    defaultConfig = {"num_ctx": 32768}


class RecordingLLM:
    model = Model()

    def __init__(self):
        self.requests = []
        self.transition = None

    def generate(self, messages, tools=None, **kwargs):
        self.requests.append(deepcopy(messages))
        if self.transition is not None:
            transition, self.transition = self.transition, None
            transition()
            calls = [
                {
                    "id": "read_mode_test",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path":"sample.txt"}',
                    },
                }
            ]
            return LLMResult("", {"role": "assistant"}, calls, None, 1)
        return LLMResult("Done.", {"role": "assistant"}, [], None, 1)


def config_for(tmp_path):
    return {
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "memory.db")},
        "llm": {"provider_config": {"generation_config": {"num_ctx": 32768}}},
        "runtime_instruction": "Keep the configured application instruction.",
        "security": {"allowed_tools": ["read_file"]},
        "context": {"compaction_enabled": False},
        "experience": {"enabled": False},
    }


def task():
    return ContextEvent(
        role=ContextRole.USER, type=ContextType.MESSAGE, content="Read sample.txt."
    )


def test_voice_mode_changes_next_request_during_run_and_never_persists(tmp_path):
    (tmp_path / "sample.txt").write_text("Example input", encoding="utf-8")
    config = config_for(tmp_path)
    original = deepcopy(config)
    llm = RecordingLLM()
    agent = Agent(config, llm=llm)
    try:
        agent.set_runtime_instruction("other_mode", "Preserve this other mode.")
        agent.set_runtime_instruction("voice_mode", VOICE_SYSTEM_INSTRUCTION)
        llm.transition = lambda: agent.set_runtime_instruction("voice_mode", None)
        assert agent.act(task()).response == "Done."
        assert len(llm.requests) == 2
        assert VOICE_SYSTEM_INSTRUCTION.strip() in llm.requests[0][0]["content"]
        assert VOICE_SYSTEM_INSTRUCTION.strip() not in llm.requests[1][0]["content"]
        assert "Preserve this other mode." in llm.requests[1][0]["content"]
        assert original["runtime_instruction"] in llm.requests[1][0]["content"]
        assert config == original
        stored = agent.loop.stm.get_recent(agent.session_id, limit=100)
        assert stored and all(
            "Voice conversation mode" not in str(event.content) for event in stored
        )
        assert "voice_mode" not in agent.loop.tool.subagents.config
        assert agent.act(task()).response == "Done."
        assert "Voice conversation mode" not in llm.requests[-1][0]["content"]
    finally:
        agent.close()


def test_transient_modes_update_independently_and_validate_inputs(tmp_path):
    agent = Agent(config_for(tmp_path), llm=SimpleNamespace(model=Model()))
    try:
        agent.set_runtime_instruction("voice_mode", "Old style")
        agent.set_runtime_instruction("other_mode", "Other style")
        agent.set_runtime_instruction("voice_mode", "New style")
        assert agent.runtime_instructions() == "New style\n\nOther style"
        agent.set_runtime_instruction("voice_mode", " ")
        assert agent.runtime_instructions() == "Other style"
        with pytest.raises(ValueError, match="nonempty"):
            agent.set_runtime_instruction("", "Text")
        with pytest.raises(TypeError, match="text"):
            agent.set_runtime_instruction("bad", {"content": "Text"})
        assert agent.runtime_instructions() == "Other style"
    finally:
        agent.close()


def test_worker_applies_voice_style_before_first_agent_request(qt_app):
    from src.studio.worker import RuntimeWorker

    calls = []

    class StubAgent:
        last_run_metrics = {"completed": True}

        def set_runtime_instruction(self, key, text):
            calls.append(("instruction", key, text))

        def act(self, event, **kwargs):
            calls.append(("request", event.content))
            return SimpleNamespace(response="Spoken reply")

    worker = RuntimeWorker(
        {},
        "Hello",
        agent=StubAgent(),
        runtime_instructions={"voice_mode": VOICE_SYSTEM_INSTRUCTION},
    )
    worker.ready.connect(lambda agent: calls.append(("ready",)))
    worker.run()
    assert calls == [
        ("instruction", "voice_mode", VOICE_SYSTEM_INSTRUCTION),
        ("ready",),
        ("request", "Hello"),
    ]
