import json
from uuid import uuid4

from src.agent.agentloop import Loop
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


def llm_result(*, response="", tool_calls=None):
    message = {"role": "assistant", "content": response}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return LLMResult(
        response=response,
        message=message,
        tool_calls=tool_calls or [],
        thinking=None,
        usage=2,
    )


class RepeatedActionLLM:
    def __init__(self):
        self.model = FakeModel()
        self.main_calls = 0
        self.recovery_review_seen_calls = []
        self.final_review_calls = 0

    def generate(self, messages, tools=None, **_kwargs):
        # Reviewer calls explicitly set tools=[] and have a distinct system prompt.
        if tools == []:
            system = messages[0]["content"]
            payload = json.loads(messages[1]["content"])
            if "execution-recovery reviewer" in system:
                self.recovery_review_seen_calls = payload["recent_observed_actions_and_results"]
                return llm_result(response=json.dumps({
                    "decision": "continue",
                    "reason": "Three read actions produced no new artifact.",
                    "next_action": "Use the observed result to decide the next useful step.",
                }))
            self.final_review_calls += 1
            return llm_result(response=json.dumps({
                "decision": "complete",
                "reason": "The final answer is accepted for this loop test.",
                "next_action": "",
            }))

        self.main_calls += 1
        if self.main_calls <= 3:
            call = {
                "id": f"read-{self.main_calls}",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"file_path": "hello.txt"}),
                },
            }
            return llm_result(tool_calls=[call])
        return llm_result(response="done")


class SequentialReadLLM:
    """A healthy multi-tool task should not trigger a periodic second opinion."""

    def __init__(self):
        self.model = FakeModel()
        self.main_calls = 0
        self.recovery_review_calls = 0
        self.final_review_calls = 0

    def generate(self, messages, tools=None, **_kwargs):
        if tools == []:
            system = messages[0]["content"]
            if "execution-recovery reviewer" in system:
                self.recovery_review_calls += 1
                return llm_result(response=json.dumps({
                    "decision": "continue",
                    "reason": "This call should not happen during healthy progress.",
                    "next_action": "",
                }))
            self.final_review_calls += 1
            return llm_result(response=json.dumps({
                "decision": "complete",
                "reason": "The task result is accepted.",
                "next_action": "",
            }))

        self.main_calls += 1
        if self.main_calls <= 6:
            call = {
                "id": f"unique-read-{self.main_calls}",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({
                        "file_path": f"note_{self.main_calls}.txt",
                    }),
                },
            }
            return llm_result(tool_calls=[call])
        return llm_result(response="Read all six notes successfully.")


def test_healthy_tool_streak_does_not_trigger_periodic_recovery_review(tmp_path):
    for index in range(1, 7):
        (tmp_path / f"note_{index}.txt").write_text(
            f"contents {index}", encoding="utf-8"
        )
    llm = SequentialReadLLM()
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 40,
            "compaction_enabled": False,
        },
        "retrieval": {"top_k": 1},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": False,
        },
        "memory": {"stm_db_path": str(tmp_path / "healthy-streak.db")},
        "max_agent_iterations": 10,
        "completion_review": {
            "enabled": True,
            "recovery_interval": 3,
            "max_recovery_checks": 2,
            "max_retries": 1,
        },
        "experience": {"enabled": False},
    }
    loop = Loop(config, llm)
    loop.session_id = uuid4()

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Read note_1.txt through note_6.txt and summarize them.",
        )
        result = loop.run(task, workspace_directory=str(tmp_path))

        assert result is not None
        assert result.response == "Read all six notes successfully."
        assert llm.main_calls == 7
        assert llm.recovery_review_calls == 0
        assert llm.final_review_calls == 1
        metrics = loop.get_metrics()
        assert metrics["tool_successes"] == 6
        assert metrics["recovery_review_calls"] == 0
        assert metrics["completion_review_calls"] == 1
        assert metrics["completed"] is True
    finally:
        loop.close()


def test_model_driven_recovery_review_runs_after_repeated_observation_loop(tmp_path):
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    llm = RepeatedActionLLM()
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 30,
            "compaction_enabled": False,
        },
        "retrieval": {"top_k": 1},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": False,
        },
        "memory": {"stm_db_path": str(tmp_path / "stm.db")},
        "max_agent_iterations": 8,
        "completion_review": {
            "enabled": True,
            "recovery_interval": 3,
            "max_recovery_checks": 2,
            "max_retries": 2,
        },
        "experience": {"enabled": False},
    }
    loop = Loop(config, llm)
    loop.session_id = uuid4()

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Read hello.txt and report the contents.",
        )
        result = loop.run(task, workspace_directory=str(tmp_path))

        assert result is not None
        assert result.response == "done"
        assert llm.main_calls == 4
        assert llm.final_review_calls == 1
        metrics = loop.get_metrics()
        assert metrics["recovery_review_calls"] == 1
        assert metrics["recovery_review_count"] == 1
        assert metrics["completed"] is True
        assert any(
            any(call.get("tool") == "read_file" for call in event.get("calls", []))
            for event in llm.recovery_review_seen_calls
            if event.get("kind") == "tool_calls"
        )
    finally:
        loop.close()
