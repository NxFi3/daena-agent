import sys
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


def test_runtime_blocks_repeated_semantic_tool_failure_across_unrelated_success(tmp_path):
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
        first_bad_call = ToolCall(
            name="read_file",
            id="read-bad-1",
            valid=False,
            validation_error="Missing required argument(s): file_path. Provide every required parameter.",
        )
        first_bad_result = ToolResult(
            success=False,
            name="read_file",
            content={
                "success": False,
                "error": {
                    "type": "invalid_tool_call",
                    "message": first_bad_call.validation_error,
                },
            },
        )
        loop._apply_result(first_bad_call, first_bad_result, 1)

        read_call = ToolCall(
            name="command_exec",
            id="command-between",
            valid=True,
            args={"command": ["pwd"], "workdir": "."},
        )
        read_result = ToolResult(
            success=True,
            name="command_exec",
            content={
                "success": True,
                "command": ["pwd"],
                "workdir": ".",
                "status": "exited",
                "exit_code": 0,
                "stdout": ".",
            },
        )
        loop._apply_result(read_call, read_result, 2)

        second_bad_call = ToolCall(
            name="read_file",
            id="read-bad-2",
            valid=True,
            args={"file_path": ""},
        )
        second_bad_result = ToolResult(
            success=False,
            name="read_file",
            content={
                "success": False,
                "error": {
                    "type": "invalid_argument",
                    "message": "Missing required argument(s): file_path. Provide every required parameter.",
                },
            },
        )
        loop._apply_result(second_bad_call, second_bad_result, 3)

        matching_key = "read_file::missing_required:file_path"
        assert loop._semantic_failure_counts[matching_key] == 2

        retry_call = ToolCall(
            name="read_file",
            id="read-bad-3",
            valid=False,
            validation_error="Missing required argument(s): file_path. Provide every required parameter.",
        )
        blocked = loop._runtime_recovery_gate(retry_call)

        assert blocked is not None
        assert blocked.metadata["runtime_gate"] is True
        assert blocked.metadata["recovery_required"] is True
        assert blocked.content["error"]["type"] == "semantic_failure_repeat"
        assert blocked.metadata["failure_signature"] == matching_key
    finally:
        loop.close()


def test_run_routes_around_no_unrelated_work_while_foreground_process_runs(tmp_path):
    class ProcessGateLLM:
        def __init__(self):
            self.model = FakeModel()
            self.calls = 0

        def generate(self, messages, tools=None):
            self.calls += 1

            if self.calls == 1:
                raw = {
                    "id": "call-command-running",
                    "type": "function",
                    "function": {
                        "name": "command_exec",
                        "arguments": {
                            "command": [
                                sys.executable,
                                str(tmp_path / "sleep_test.py"),
                            ],
                            "workdir": ".",
                            "yield_time_ms": 25,
                        },
                    },
                }
                return LLMResult(
                    response="",
                    message={"role": "assistant", "content": "", "tool_calls": [raw]},
                    tool_calls=[raw],
                    thinking=None,
                    usage=10,
                )

            if self.calls == 2:
                raw = {
                    "id": "call-read-blocked",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"file_path": "hello.txt"},
                    },
                }
                return LLMResult(
                    response="",
                    message={"role": "assistant", "content": "", "tool_calls": [raw]},
                    tool_calls=[raw],
                    thinking=None,
                    usage=10,
                )

            if self.calls == 3:
                import re
                process_ids = re.findall(r"proc-[0-9a-f]+", str(messages))
                assert process_ids, "the running process id must remain model-visible"
                raw = {
                    "id": "call-process-poll",
                    "type": "function",
                    "function": {
                        "name": "process_poll",
                        "arguments": {"process_id": process_ids[-1], "wait_ms": 1000},
                    },
                }
                return LLMResult(
                    response="",
                    message={"role": "assistant", "content": "", "tool_calls": [raw]},
                    tool_calls=[raw],
                    thinking=None,
                    usage=10,
                )

            return LLMResult(
                response="done",
                message={"role": "assistant", "content": "done"},
                tool_calls=[],
                thinking=None,
                usage=10,
            )

    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 40,
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
        "max_agent_iterations": 6,
        "experience": {"enabled": False},
    }

    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sleep_test.py").write_text(
        "import time\ntime.sleep(0.4)\n",
        encoding="utf-8",
    )
    llm = ProcessGateLLM()
    loop = Loop(config, llm)
    loop.session_id = uuid4()

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Run the command and then finish the task.",
        )
        result = loop.run(task, workspace_directory=str(tmp_path))

        assert result is not None
        assert result.response == "done"
        assert llm.calls == 4
        assert loop.get_metrics()["recovery_blocks"] >= 1
        assert loop._active_process_ids == set()
        assert not any(
            item.get("tool") == "read_file" and item.get("outcome") == "success"
            for item in loop.working_set.recent_actions
        )
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


def test_runtime_requires_process_observation_while_foreground_process_is_active(tmp_path):
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

        read_call = ToolCall(
            name="read_file",
            id="read-live",
            valid=True,
            args={"file_path": "hello.txt"},
        )
        blocked = loop._runtime_recovery_gate(read_call)
        assert blocked is not None
        assert blocked.metadata["runtime_gate"] is True
        assert blocked.content["error"]["type"] == "active_process_requires_observation"

        web_call = ToolCall(
            name="web_search",
            id="web-live",
            valid=True,
            args={"query": "hello"},
        )
        assert loop._runtime_recovery_gate(web_call) is not None

        command_call = ToolCall(
            name="command_exec",
            id="command-live",
            valid=True,
            args={"command": ["python", "-c", "print('unrelated')"], "workdir": "."},
        )
        assert loop._runtime_recovery_gate(command_call) is not None

        plan_call = ToolCall(
            name="plan",
            id="plan-live",
            valid=True,
            args={"action": "complete"},
        )
        plan_block = loop._runtime_recovery_gate(plan_call)
        assert plan_block is not None
        assert plan_block.content["error"]["type"] == "active_process"

        poll_call = ToolCall(
            name="process_poll",
            id="poll-live",
            valid=True,
            args={"process_id": "proc-live"},
        )
        assert loop._runtime_recovery_gate(poll_call) is None

        write_call = ToolCall(
            name="process_write",
            id="write-live",
            valid=True,
            args={"process_id": "proc-live", "input_text": "ok\\n"},
        )
        assert loop._runtime_recovery_gate(write_call) is None

        stop_call = ToolCall(
            name="process_stop",
            id="stop-live",
            valid=True,
            args={"process_id": "proc-live"},
        )
        assert loop._runtime_recovery_gate(stop_call) is None
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

def test_runtime_does_not_exhaust_observation_from_counters(tmp_path):
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
        "max_agent_iterations": 20,
        "experience": {"enabled": False},
    }

    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")

    try:
        loop._same_revision_read_count = 100
        loop._observation_action_count = 100

        read_call = ToolCall(
            name="read_file",
            id="read-after-former-limit",
            valid=True,
            args={"file_path": "hello.txt"},
        )

        assert loop._runtime_recovery_gate(read_call) is None
    finally:
        loop.close()

def test_final_verification_detects_whole_suite_only(tmp_path):
    loop = Loop(
        {
            "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
            "context": {"safe_margin": 0, "compaction_enabled": False},
            "retrieval": {"top_k": 1},
            "security": {"workspace_only": True, "force_approve": True},
            "max_agent_iterations": 3,
            "experience": {"enabled": False},
        },
        FakeLLM(),
    )
    try:
        whole_suite = ToolCall(
            name="command_exec",
            id="verify-all",
            valid=True,
            args={"command": ["pytest", "-q"], "workdir": "."},
        )
        whole_suite_result = ToolResult(
            success=True,
            name="command_exec",
            content={
                "command": ["pytest", "-q"],
                "status": "exited",
                "exit_code": 0,
            },
        )

        targeted = ToolCall(
            name="command_exec",
            id="verify-one",
            valid=True,
            args={"command": ["pytest", "-q", "test_workspace_stats.py"], "workdir": "."},
        )
        targeted_result = ToolResult(
            success=True,
            name="command_exec",
            content={
                "command": ["pytest", "-q", "test_workspace_stats.py"],
                "status": "exited",
                "exit_code": 0,
            },
        )

        assert loop._is_final_verification_call(whole_suite, whole_suite_result) is True
        assert loop._is_final_verification_call(targeted, targeted_result) is False
    finally:
        loop.close()


def test_finalization_mode_blocks_non_plan_tools(tmp_path):
    loop = Loop(
        {
            "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
            "context": {"safe_margin": 0, "compaction_enabled": False},
            "retrieval": {"top_k": 1},
            "security": {"workspace_only": True, "force_approve": True},
            "max_agent_iterations": 3,
            "experience": {"enabled": False},
        },
        FakeLLM(),
    )
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    try:
        loop._plan_active_this_run = True
        loop._final_verification_satisfied = True

        read_call = ToolCall(
            name="read_file",
            id="read-after-verify",
            valid=True,
            args={"file_path": "workspace_stats.py"},
        )
        allowed, blocked = loop._classify_calls([read_call])

        assert allowed == []
        assert blocked[0].content["error"]["type"] == "finalization_only"
    finally:
        loop.close()


def test_final_verification_state_resets_between_runs(tmp_path):
    loop = Loop(
        {
            "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
            "context": {"safe_margin": 0, "compaction_enabled": False},
            "retrieval": {"top_k": 1},
            "security": {"workspace_only": True, "force_approve": True},
            "max_agent_iterations": 3,
            "experience": {"enabled": False},
        },
        FakeLLM(),
    )
    try:
        loop._final_verification_satisfied = True
        loop._reset_run_state()
        assert loop._final_verification_satisfied is False
    finally:
        loop.close()



def test_final_verification_supports_common_project_runners():
    loop = Loop(
        {
            "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
            "context": {"safe_margin": 0, "compaction_enabled": False},
            "retrieval": {"top_k": 1},
            "security": {"workspace_only": True, "force_approve": True},
            "max_agent_iterations": 3,
            "experience": {"enabled": False},
        },
        FakeLLM(),
    )
    try:
        commands = {
            "npm": (["npm", "test"], True),
            "cargo": (["cargo", "test"], True),
            "go_all": (["go", "test", "./..."], True),
            "go_targeted": (["go", "test", "./pkg/foo"], False),
            "gradle": (["./gradlew", "test"], True),
        }
        for _, (command, expected) in commands.items():
            call = ToolCall(
                name="command_exec",
                id="verify-" + command[0],
                valid=True,
                args={"command": command, "workdir": "."},
            )
            result = ToolResult(
                success=True,
                name="command_exec",
                content={"command": command, "exit_code": 0},
            )
            assert loop._is_final_verification_call(call, result) is expected
    finally:
        loop.close()


def test_multiphase_task_requires_plan_before_repository_work(tmp_path):
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {"safe_margin": 0, "compaction_enabled": False},
        "retrieval": {"top_k": 1},
        "security": {"workspace_only": True, "force_approve": True},
        "max_agent_iterations": 3,
        "experience": {"enabled": False},
    }
    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    try:
        loop._plan_required_this_run = True

        read_call = ToolCall(
            name="read_file",
            id="read-before-plan",
            valid=True,
            args={"file_path": "app.py"},
        )
        allowed, blocked = loop._classify_calls([read_call])
        assert allowed == []
        assert blocked[0].content["error"]["type"] == "plan_required_first"

        plan_call = ToolCall(
            name="plan",
            id="plan-first",
            valid=True,
            args={
                "action": "create",
                "goal": "Fix and verify",
                "steps": ["Inspect", "Implement", "Verify"],
            },
        )
        allowed, blocked = loop._classify_calls([plan_call])
        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()


def test_strict_observation_repeat_is_bounded_after_one_success(tmp_path):
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {"safe_margin": 0, "compaction_enabled": False},
        "retrieval": {"top_k": 1},
        "security": {"workspace_only": True, "force_approve": True},
        "max_agent_iterations": 3,
        "experience": {"enabled": False},
    }
    loop = Loop(config, FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))
    try:
        call = ToolCall(
            name="grep",
            id="grep-1",
            valid=True,
            args={"pattern": "class Foo"},
        )
        loop._successful_tool_calls[loop._tool_call_key(call)] = loop.workspace_revision
        loop._same_revision_call_counts[loop._tool_call_key(call)] = (
            loop.workspace_revision,
            1,
        )

        allowed, blocked = loop._classify_calls([call])
        assert allowed == []
        assert blocked[0].content["error"]["type"] == "duplicate_action"
    finally:
        loop.close()
