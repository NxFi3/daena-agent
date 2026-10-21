from pathlib import Path
from uuid import uuid4

from src.tools.builtin.plan.tool import Plan

from src.agent.agentloop import Loop
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


class FakeLLM:
    def __init__(self):
        self.model = FakeModel()
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1

        if self.calls == 1:
            return LLMResult(
                response="",
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_read_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"file_path":"hello.txt"}',
                            },
                        }
                    ],
                },
                tool_calls=[
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"file_path":"hello.txt"}',
                        },
                    }
                ],
                thinking=None,
                usage=25,
            )

        return LLMResult(
            response="done",
            message={"role": "assistant", "content": "done"},
            tool_calls=[],
            thinking=None,
            usage=15,
        )


def test_loop_executes_tool_through_security_and_context(tmp_path, monkeypatch):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello baseline", encoding="utf-8")

    config = {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 4096},
            }
        },
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 40,
            "compaction_enabled": True,
            "compaction_target_tokens": 256,
        },
        "retrieval": {"top_k": 3},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": False,
        },
        "max_agent_iterations": 5,
        "experience": {"enabled": False},
    }

    monkeypatch.setattr(
        Plan,
        "PLAN_PATH",
        Path(tmp_path) / "AgentInstruction" / "plan.md",
    )

    llm = FakeLLM()
    loop = Loop(config, llm)
    loop.session_id = uuid4()

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Read hello.txt and finish.",
        )

        result = loop.run(task, workspace_directory=str(tmp_path))

        assert result is not None
        assert result.response == "done"
        assert llm.calls == 2

        metrics = loop.get_metrics()
        assert metrics["completed"] is True
        assert metrics["iterations"] == 2
        assert metrics["tool_successes"] == 1
        assert metrics["tool_failures"] == 0
        assert metrics["llm_calls"] == 2
        assert metrics["tokens"] == 40
    finally:
        loop.close()


def test_loop_honors_configured_context_limits(tmp_path):
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 120000}}},
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 17,
            "max_prompt_tokens": 9000,
            "compaction_enabled": True,
            "compaction_target_tokens": 4096,
        },
        "retrieval": {"top_k": 2},
        "security": {
            "workspace_only": True,
            "allow_background": True,
            "allow_network_tools": True,
            "force_approve": True,
        },
        "max_agent_iterations": 5,
        "experience": {"enabled": False},
    }

    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()

    try:
        assert loop.recent_context_limit == 17
        assert loop.search_context_top_k == 2
        assert loop.context.contextbuilder.tokenbudget.budget == 9000
    finally:
        loop.close()


def test_apply_result_tracks_only_foreground_running_processes(tmp_path):
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 10,
            "compaction_enabled": False,
            "compaction_target_tokens": 256,
        },
        "retrieval": {"top_k": 3},
        "security": {
            "workspace_only": True,
            "allow_background": True,
            "allow_network_tools": True,
            "force_approve": True,
        },
        "max_agent_iterations": 5,
        "experience": {"enabled": False},
    }

    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))

    try:
        background_call = ToolCall(
            name="command_exec",
            id="background-1",
            valid=True,
            args={"command": ["python", "-m", "http.server"], "background": True},
        )
        loop._apply_result(
            background_call,
            ToolResult(
                success=False,
                name="command_exec",
                content={
                    "status": "running",
                    "process_id": "proc-background",
                    "background": True,
                },
                metadata={},
            ),
            1,
        )
        assert loop._active_process_ids == set()

        foreground_call = ToolCall(
            name="command_exec",
            id="foreground-1",
            valid=True,
            args={"command": ["python", "-m", "pytest"], "background": False},
        )
        loop._apply_result(
            foreground_call,
            ToolResult(
                success=False,
                name="command_exec",
                content={
                    "status": "running",
                    "process_id": "proc-foreground",
                    "background": False,
                },
                metadata={},
            ),
            2,
        )
        assert loop._active_process_ids == {"proc-foreground"}
    finally:
        loop.close()
