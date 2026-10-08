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
            stm.add(session_id, event(f"state-{step}-" + ("x" * 60), step))

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
