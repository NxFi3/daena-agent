from uuid import uuid4

from src.agent.agentstate import AgentState
from src.context.contextservice import ContextService
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.memories.stm import STM


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


class FakeLLM:
    def __init__(self):
        self.model = FakeModel()
        self.calls = 0

    def generate(self, messages, tools=None, options=None):
        self.calls += 1
        return LLMResult(
            response="Persistent checkpoint summary.",
            message={
                "role": "assistant",
                "content": "Persistent checkpoint summary.",
            },
            tool_calls=[],
            thinking=None,
            usage=20,
        )


def config():
    return {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 4096},
            }
        },
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 80,
            "compaction_enabled": True,
            "compaction_target_tokens": 512,
            "max_prompt_tokens": 3072,
            "compaction_trigger_ratio": 0.70,
        },
        "retrieval": {"top_k": 2},
        "experience": {"enabled": False},
    }


def event(content: str, step: int) -> ContextEvent:
    return ContextEvent(
        id=uuid4(),
        role=ContextRole.ASSISTANT,
        type=ContextType.MESSAGE,
        content=content,
        step=step,
    )


def test_checkpoint_persists_and_prevents_repeat_compaction(tmp_path):
    llm = FakeLLM()
    stm = STM(tmp_path / "stm.db")
    service = ContextService(config(), llm, stm)
    session_id = uuid4()

    try:
        for step in range(1, 61):
            stm.add(session_id, event(f"state-{step}-" + ("x" * 180), step))

        first_task = ContextEvent(
            id=uuid4(),
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="continue the implementation",
            step=0,
        )

        first = service.get_context(
            session_id=session_id,
            user_task=first_task,
            agent_state=AgentState(),
            workspace_directory=str(tmp_path),
        )

        checkpoint = stm.get_latest_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint.metadata["checkpoint"] is True
        assert checkpoint.metadata["covered_through_step"] > 0
        assert llm.calls == 1

        second_task = ContextEvent(
            id=uuid4(),
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="make the next change",
            step=0,
        )

        second = service.get_context(
            session_id=session_id,
            user_task=second_task,
            agent_state=AgentState(),
            workspace_directory=str(tmp_path),
        )

        assert llm.calls == 1
        assert second[-1] == {"role": "user", "content": "make the next change"}
        assert sum(
            "<checkpoint_summary>" in str(message.get("content", ""))
            for message in second
            if message.get("role") == "user"
        ) == 1
    finally:
        stm.close()


def test_system_prompt_is_stable_while_runtime_state_changes(tmp_path):
    llm = FakeLLM()
    stable_config = config()
    stable_config["context"]["max_prompt_tokens"] = 4096
    stm = STM(tmp_path / "stm.db")
    service = ContextService(stable_config, llm, stm)
    session_id = uuid4()

    try:
        task = ContextEvent(
            id=uuid4(),
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="inspect this",
            step=1,
        )
        first = service.contextbuilder.build_context(
            events=[task],
            task={"id": str(task.id), "content": task.content, "step": 1},
            agent_state={"status": "running", "iteration": 1},
            working_set={"facts": ["alpha"]},
            workspace=str(tmp_path),
        )
        second = service.contextbuilder.build_context(
            events=[task],
            task={"id": str(task.id), "content": task.content, "step": 1},
            agent_state={"status": "running", "iteration": 2},
            working_set={"facts": ["beta"]},
            workspace=str(tmp_path),
        )

        assert first[0]["role"] == "system"
        assert second[0]["role"] == "system"
        assert first[0]["content"] == second[0]["content"]
        assert first[-1] == {"role": "user", "content": "inspect this"}
        assert second[-1] == {"role": "user", "content": "inspect this"}
        assert '"iteration": 2' in str(
            next(
                message["content"]
                for message in second
                if message.get("role") == "user"
                and str(message.get("content", "")).startswith("<runtime_state>")
            )
        )
    finally:
        stm.close()



def test_context_service_keeps_retrieved_memory_outside_recent_history_window(
    tmp_path,
    monkeypatch,
):
    llm = FakeLLM()
    stm = STM(tmp_path / "stm.db")
    service = ContextService(config(), llm, stm)
    session_id = uuid4()
    important = event(
        "Previous investigation proved the cache key must include checkpoint_step.",
        1,
    )

    try:
        stm.add(session_id, important)
        for step in range(2, 18):
            stm.add(session_id, event(f"routine historical event {step}", step))

        task = ContextEvent(
            id=uuid4(),
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="continue context debugging",
            step=20,
        )
        stm.add(session_id, task)
        monkeypatch.setattr(stm, "search", lambda **_kwargs: [important])

        messages = service.get_context(
            session_id=session_id,
            user_task=task,
            agent_state=AgentState(),
            workspace_directory=str(tmp_path),
        )

        rendered = "\\n".join(str(message.get("content", "")) for message in messages)
        assert "<retrieved_context>" in rendered
        assert "Previous investigation proved the cache key must include checkpoint_step." in rendered
        assert messages[-1] == {
            "role": "user",
            "content": "continue context debugging",
        }
    finally:
        stm.close()


def test_checkpoint_boundary_uses_recent_message_steps_not_event_count(tmp_path):
    llm = FakeLLM()
    stm = STM(tmp_path / "step-coverage.db")
    service = ContextService(config(), llm, stm)
    builder = service.contextbuilder
    messages = [{"role": "system", "content": "system"}]
    for step in range(1, 16):
        messages.append({
            "role": "assistant",
            "content": f"state at step {step} " + ("x" * 100),
            "_step": step,
        })
    messages.append({"role": "user", "content": "latest task", "_step": 20})
    try:
        compacted = builder._compact_messages(messages)
        assert compacted is not None
        assert builder._pending_checkpoint["covered_through_step"] == 5
        assert all(
            not any(key.startswith("_") for key in message)
            for message in builder._strip_internal_message_keys(compacted)
        )
    finally:
        stm.close()


def test_build_context_never_leaks_internal_step_to_provider(tmp_path):
    llm = FakeLLM()
    stm = STM(tmp_path / "strip-step.db")
    service = ContextService(config(), llm, stm)
    task = ContextEvent(
        id=uuid4(), role=ContextRole.USER, type=ContextType.MESSAGE,
        content="inspect the source", step=1,
    )
    try:
        messages = service.contextbuilder.build_context(
            events=[task],
            task={"id": str(task.id), "content": task.content, "step": 1},
            workspace=str(tmp_path),
        )
        assert messages[-1] == {"role": "user", "content": "inspect the source"}
        assert all(not any(key.startswith("_") for key in message) for message in messages)
    finally:
        stm.close()
