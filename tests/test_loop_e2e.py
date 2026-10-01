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
                    "tool_calls": [{
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"file_path":"hello.txt"}',
                        },
                    }],
                },
                tool_calls=[{
                    "id": "call_read_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path":"hello.txt"}',
                    },
                }],
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

def test_duplicate_detector_canonicalizes_read_paths_and_allows_dynamic_polling(tmp_path):
    target = tmp_path / "hello.txt"
    target.write_text("hello", encoding="utf-8")

    config = {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 4096},
            }
        },
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

    from src.models.ToolCall import ToolCall

    llm = FakeLLM()
    loop = Loop(config, llm)
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))

    try:
        first_read = ToolCall(
            name="read_file",
            id="read-1",
            valid=True,
            args={"file_path": "hello.txt"},
        )
        second_read = ToolCall(
            name="read_file",
            id="read-2",
            valid=True,
            args={"file_path": str(target)},
        )

        key = loop._tool_call_key(first_read)
        loop._successful_tool_calls[key] = loop.workspace_revision

        # Observations may be repeated a few times without a workspace edit so
        # the model can re-check evidence, but the runtime still bounds them.
        allowed, blocked = loop._classify_calls([second_read])
        assert allowed == [0]
        assert blocked == {}

        loop._same_revision_call_counts[key] = (
            loop.workspace_revision,
            loop.OBSERVATION_REPEAT_LIMIT,
        )
        allowed, blocked = loop._classify_calls([second_read])
        assert allowed == []
        assert 0 in blocked

        poll = ToolCall(
            name="process_poll",
            id="poll-1",
            valid=True,
            args={"process_id": "proc-demo"},
        )
        poll_key = loop._tool_call_key(poll)
        loop._successful_tool_calls[poll_key] = loop.workspace_revision

        allowed, blocked = loop._classify_calls([poll])
        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()


def test_runtime_blocks_plan_while_process_is_active_and_repeats_failed_action(tmp_path):
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
        loop._active_process_ids.add("proc-live")

        plan_call = ToolCall(
            name="plan",
            id="plan-live",
            valid=True,
            args={"operation": "update", "step": 1, "status": "completed"},
        )
        blocked = loop._runtime_recovery_gate(plan_call)
        assert blocked is not None
        assert blocked.metadata["runtime_gate"] is True

        loop._active_process_ids.clear()

        failed_call = ToolCall(
            name="command_exec",
            id="cmd-1",
            valid=True,
            args={
                "command": ["python", "-c", "raise SystemExit(1)"],
                "workdir": ".",
            },
        )
        failed_result = ToolResult(
            success=False,
            name="command_exec",
            content={
                "success": False,
                "status": "exited",
                "process_id": None,
                "exit_code": 1,
                "error": {"type": "test_assertion", "message": "Expected 200"},
            },
            metadata={},
        )

        loop._apply_result(failed_call, failed_result, 1)

        blocked_retry = loop._runtime_recovery_gate(failed_call)
        assert blocked_retry is not None
        assert blocked_retry.metadata["runtime_gate"] is True
        assert "exact failed action" in blocked_retry.summary
    finally:
        loop.close()


def test_failed_work_requires_different_success_before_plan_completion():
    loop = _loop_for_validation()
    loop._recovery_required = True
    loop._recovery_failed_key = "failed-key"

    state = active_plan()
    error = loop._validate_plan_transition(
        plan_completion_call(),
        state,
    )

    assert error is not None
    assert error[0] == "recovery_required"

    loop._recovery_required = False
    loop._recovery_failed_key = None
    loop._plan_progress.record(
        ToolCall(
            name="read_file",
            id="recovery-read",
            valid=True,
            action="inspect",
            target="package.json",
        ),
        ToolResult(
            success=True,
            name="read_file",
            content={"path": "package.json", "content": "{}"},
        ),
        iteration=3,
    )

    assert loop._validate_plan_transition(
        plan_completion_call(),
        state,
    ) is None


def test_control_plane_plan_failure_does_not_enter_recovery_state():
    loop = _loop_for_validation()
    loop._recovery_required = False
    loop._recovery_failed_key = None
    loop._failed_call_keys = {}
    loop._active_process_ids = set()
    loop._recovery_mode = False
    loop.workspace_revision = 0
    loop._store_event = lambda event: None
    loop.agent_state = type("State", (), {
        "begin": lambda *args, **kwargs: None,
        "update_from_result": lambda *args, **kwargs: None,
    })()
    loop._tool_call_event = lambda call: None
    loop._tool_result_event = lambda call, result: None
    loop.working_set.update = lambda **kwargs: False
    loop._plan_progress.record = lambda **kwargs: None
    loop.tool = type("Tools", (), {
        "get_tool": lambda self, name: None,
    })()

    plan = plan_completion_call()
    result = ToolResult(
        success=False,
        name="plan",
        content={
            "success": False,
            "error": {
                "type": "completion_requires_success",
                "message": "previous work failed",
            },
        },
        metadata={"plan_gate": True},
    )

    loop._apply_result(plan, result, 2)

    assert loop._recovery_required is False
