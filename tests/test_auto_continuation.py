import json
from pathlib import Path
from uuid import uuid4

from src.agent.agentloop import Loop
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.tools.builtin.plan.tool import Plan


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


def read_tool_result(call_number: int) -> LLMResult:
    tool_call = {
        "id": f"call_read_{call_number}",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"file_path": "hello.txt"}),
        },
    }
    return LLMResult(
        response="",
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        },
        tool_calls=[tool_call],
        thinking=None,
        usage=2,
    )


def final_result(text: str = "done") -> LLMResult:
    return LLMResult(
        response=text,
        message={"role": "assistant", "content": text},
        tool_calls=[],
        thinking=None,
        usage=2,
    )


class ScriptedLLM:
    provider_name = "fake"

    def __init__(self, scripted_results):
        self.model = FakeModel()
        self.scripted_results = list(scripted_results)
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.scripted_results:
            result = self.scripted_results.pop(0)
            if callable(result):
                return result(self.calls)
            return result
        return read_tool_result(self.calls)


def config(**overrides):
    values = {
        "llm": {
            "provider": "fake",
            "provider_config": {
                "generation_config": {"num_ctx": 4096},
            },
        },
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 40,
            "compaction_enabled": False,
            "compaction_target_tokens": 1024,
        },
        "retrieval": {"top_k": 2},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": False,
            "force_approve": False,
        },
        "max_agent_iterations": 1,
        "max_auto_continuations": 4,
        "max_no_progress_segments": 2,
        "experience": {"enabled": False},
    }
    values.update(overrides)
    return values


def setup_loop(tmp_path, monkeypatch, llm, cfg=None):
    (tmp_path / "hello.txt").write_text("hello baseline", encoding="utf-8")
    monkeypatch.setattr(
        Plan,
        "PLAN_PATH",
        Path(tmp_path) / "AgentInstruction" / "plan.md",
    )
    loop = Loop(cfg or config(), llm)
    loop.session_id = uuid4()
    return loop


def task():
    return ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content="Read hello.txt, continue the task if an iteration budget is reached, and finish.",
    )


def test_iteration_budget_continues_same_task_automatically(tmp_path, monkeypatch):
    llm = ScriptedLLM([read_tool_result(1), final_result("finished after continuation")])
    loop = setup_loop(tmp_path, monkeypatch, llm)

    try:
        result = loop.run(task(), workspace_directory=str(tmp_path))

        assert result is not None
        assert result.response == "finished after continuation"
        assert llm.calls == 2
        metrics = loop.get_metrics()
        assert metrics["completed"] is True
        assert metrics["iterations"] == 2
        assert metrics["continuation_segments"] == 1
    finally:
        loop.close()


def test_auto_continuation_stops_after_repeated_no_progress(tmp_path, monkeypatch):
    llm = ScriptedLLM([lambda n: read_tool_result(n)])
    loop = setup_loop(
        tmp_path,
        monkeypatch,
        llm,
        config(max_agent_iterations=1, max_auto_continuations=5, max_no_progress_segments=2),
    )

    try:
        result = loop.run(task(), workspace_directory=str(tmp_path))

        assert result is not None
        metrics = loop.get_metrics()
        assert metrics["completed"] is False
        assert "No verified workspace progress" in metrics["stop_reason"]
        assert llm.calls == 2
        assert metrics["iterations"] == 2
        assert metrics["continuation_segments"] == 1
    finally:
        loop.close()
