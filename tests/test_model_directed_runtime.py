from pathlib import Path
from uuid import uuid4

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


def make_loop(tmp_path):
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
        "max_agent_iterations": 5,
        "experience": {"enabled": False},
    }
    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    return loop


def test_valid_duplicate_calls_are_model_owned(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call_a = ToolCall(
            name="read_file",
            id="read-a",
            valid=True,
            args={"file_path": "app.py"},
        )
        call_b = ToolCall(
            name="read_file",
            id="read-b",
            valid=True,
            args={"file_path": "app.py"},
        )

        allowed, blocked = loop._classify_calls([call_a, call_b])

        assert allowed == [0, 1]
        assert blocked == {}
    finally:
        loop.close()


def test_repeated_failure_does_not_create_a_runtime_retry_gate(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="read_file",
            id="bad-read",
            valid=True,
            args={"file_path": ""},
        )
        failed = ToolResult(
            success=False,
            name="read_file",
            content={
                "success": False,
                "error": {
                    "type": "invalid_argument",
                    "message": "file_path is required.",
                },
            },
        )

        loop._apply_result(call, failed, 1)
        loop._apply_result(call, failed, 2)

        allowed, blocked = loop._classify_calls([call])

        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()


def test_foreground_process_state_is_evidence_not_a_strategy_gate(tmp_path):
    loop = make_loop(tmp_path)
    try:
        loop._active_process_ids.add("proc-live")

        calls = [
            ToolCall(
                name="read_file",
                id="read",
                valid=True,
                args={"file_path": "app.py"},
            ),
            ToolCall(
                name="grep",
                id="grep",
                valid=True,
                args={"pattern": "Agent"},
            ),
            ToolCall(
                name="plan",
                id="plan",
                valid=True,
                args={"action": "create", "goal": "inspect"},
            ),
            ToolCall(
                name="command_exec",
                id="cmd",
                valid=True,
                args={"command": ["python", "script.py"]},
            ),
        ]

        allowed, blocked = loop._classify_calls(calls)

        assert allowed == [0, 1, 2, 3]
        assert blocked == {}
    finally:
        loop.close()


def test_plan_is_optional_and_full_tool_vocabulary_remains_available(tmp_path):
    loop = make_loop(tmp_path)
    captured = []

    def generate(messages, tools=None):
        captured.append(
            [
                item.get("function", {}).get("name")
                for item in (tools or [])
                if isinstance(item, dict)
            ]
        )
        return LLMResult(
            response="done",
            message={"role": "assistant", "content": "done"},
            tool_calls=[],
            thinking=None,
            usage=1,
        )

    loop.llm.generate = generate

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Inspect and improve the project.",
        )
        result = loop._generate_next_action(task, str(tmp_path))

        assert result is not None
        assert captured
        assert "plan" in captured[0]
        assert "read_file" in captured[0]
        assert "apply_patch" in captured[0]
        assert "command_exec" in captured[0]
        assert "process_poll" in captured[0]
    finally:
        loop.close()


def test_successful_mutation_does_not_force_verification_before_next_action(tmp_path):
    loop = make_loop(tmp_path)
    try:
        changed_call = ToolCall(
            name="apply_patch",
            id="patch",
            valid=True,
            action="modify",
            target="app.py",
            args={"patch": "*** Begin Patch\n*** Update File: app.py\n@@\n-old\n+new\n*** End Patch"},
        )
        changed_result = ToolResult(
            success=True,
            name="apply_patch",
            content={
                "success": True,
                "files": [{"path": "app.py", "operation": "update", "content": "new\n"}],
            },
            metadata={},
        )

        loop._apply_result(changed_call, changed_result, 1)

        read_call = ToolCall(
            name="read_file",
            id="read-after-edit",
            valid=True,
            args={"file_path": "app.py"},
        )
        allowed, blocked = loop._classify_calls([read_call])

        assert loop.workspace_revision == 1
        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()


def test_invalid_calls_are_rejected_but_not_followed_by_runtime_strategy(tmp_path):
    loop = make_loop(tmp_path)
    try:
        invalid = ToolCall(
            name="read_file",
            id="invalid",
            valid=False,
            validation_error="Missing required argument(s): file_path.",
        )

        allowed, blocked = loop._classify_calls([invalid])

        assert allowed == []
        assert blocked[0].content["error"]["type"] == "invalid_tool_call"
        assert "recovery_hint" in blocked[0].metadata
    finally:
        loop.close()


def test_natural_language_completion_is_not_blocked_after_mutation(tmp_path):
    class MutateThenFinishLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                patch = (
                    "*** Begin Patch\n"
                    "*** Add File: result.txt\n"
                    "+done\n"
                    "*** End Patch"
                )
                raw = {
                    "id": "patch-1",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": {"patch": patch},
                    },
                }
                return LLMResult(
                    response="",
                    message={"role": "assistant", "content": "", "tool_calls": [raw]},
                    tool_calls=[raw],
                    thinking=None,
                    usage=1,
                )

            assert Path(tmp_path, "result.txt").read_text(encoding="utf-8") == "done\n"
            return LLMResult(
                response="done",
                message={"role": "assistant", "content": "done"},
                tool_calls=[],
                thinking=None,
                usage=1,
            )

    loop = Loop(
        {
            "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
            "context": {
                "safe_margin": 0,
                "recent_event_limit": 20,
                "compaction_enabled": False,
            },
            "retrieval": {"top_k": 1},
            "security": {
                "workspace_only": True,
                "allow_background": False,
                "allow_network_tools": True,
                "force_approve": False,
            },
            "max_agent_iterations": 4,
            "experience": {"enabled": False},
        },
        MutateThenFinishLLM(),
    )
    loop.session_id = uuid4()

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Create result.txt and finish.",
        )
        result = loop.run(task, workspace_directory=str(tmp_path))

        assert result is not None
        assert result.response == "done"
        assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "done\n"
    finally:
        loop.close()
