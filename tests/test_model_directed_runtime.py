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


def test_exact_duplicate_observation_calls_are_blocked_in_one_batch(tmp_path):
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

        assert allowed == [0]
        assert set(blocked) == {1}
        assert blocked[1].content["error"]["type"] == "repeated_observation_no_progress"
    finally:
        loop.close()


def test_identical_failed_command_is_blocked_after_two_failures_without_progress(tmp_path):
    loop = make_loop(tmp_path)
    command = ToolCall(
        name="command_exec",
        id="failed-command",
        valid=True,
        args={
            "command": ["python", "-c", "raise SystemExit('same error')"],
            "working_directory": str(tmp_path),
        },
    )
    failed_result = ToolResult(
        success=False,
        name="command_exec",
        content={
            "exit_code": 1,
            "stderr": "same error",
            "error": {"type": "command_failed", "message": "exit code 1"},
        },
        summary="Command exited with status 1",
    )
    try:
        loop._apply_result(command, failed_result, iteration=1)
        loop._apply_result(command, failed_result, iteration=2)

        allowed, blocked = loop._classify_calls([command])

        assert allowed == []
        assert 0 in blocked
        assert blocked[0].content["error"]["type"] == "repeated_command_failure_no_progress"
        assert loop.get_metrics()["repeated_command_failure_blocks"] == 1
    finally:
        loop.close()


def test_failed_command_can_run_again_after_workspace_progress(tmp_path):
    loop = make_loop(tmp_path)
    command = ToolCall(
        name="command_exec",
        id="failed-command",
        valid=True,
        args={
            "command": ["python", "-c", "raise SystemExit('same error')"],
            "working_directory": str(tmp_path),
        },
    )
    failed_result = ToolResult(
        success=False,
        name="command_exec",
        content={
            "exit_code": 1,
            "stderr": "same error",
            "error": {"type": "command_failed", "message": "exit code 1"},
        },
        summary="Command exited with status 1",
    )
    write = ToolCall(
        name="write_file",
        id="write-after-failure",
        valid=True,
        args={"file_path": "recovery.txt", "content": "corrected"},
    )
    try:
        loop._apply_result(command, failed_result, iteration=1)
        loop._apply_result(command, failed_result, iteration=2)
        loop._apply_result(
            write,
            ToolResult(
                success=True,
                name="write_file",
                content={"path": str(tmp_path / "recovery.txt"), "bytes_written": 9},
                summary="Wrote recovery.txt",
            ),
            iteration=3,
        )

        allowed, blocked = loop._classify_calls([command])

        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()


def test_observation_can_repeat_after_a_successful_write(tmp_path):
    loop = make_loop(tmp_path)
    try:
        read_call = ToolCall(
            name="read_file",
            id="read-a",
            valid=True,
            args={"file_path": "app.py"},
        )
        loop._apply_result(
            read_call,
            ToolResult(success=True, name="read_file", content={"content": "print('hi')"}),
            1,
        )
        allowed, blocked = loop._classify_calls([read_call])
        assert allowed == []
        assert set(blocked) == {0}

        write_call = ToolCall(
            name="write_file",
            id="write-a",
            valid=True,
            args={"file_path": "generated.txt", "content": "new state"},
        )
        loop._apply_result(
            write_call,
            ToolResult(success=True, name="write_file", content={"success": True}),
            2,
        )

        allowed, blocked = loop._classify_calls([read_call])
        assert allowed == [0]
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


def test_successful_remote_fetch_is_not_unlocked_by_a_local_write(tmp_path):
    loop = make_loop(tmp_path)
    try:
        fetch_call = ToolCall(
            name="web_fetch",
            id="fetch-a",
            valid=True,
            args={"url": "https://example.invalid/public"},
        )
        loop._apply_result(
            fetch_call,
            ToolResult(success=True, name="web_fetch", content={"content": "observed data"}),
            1,
        )

        write_call = ToolCall(
            name="write_file",
            id="write-a",
            valid=True,
            args={"file_path": "dataset.csv", "content": "header"},
        )
        loop._apply_result(
            write_call,
            ToolResult(success=True, name="write_file", content={"success": True}),
            2,
        )

        allowed, blocked = loop._classify_calls([fetch_call])
        assert allowed == []
        assert set(blocked) == {0}
        assert blocked[0].success is True
        assert blocked[0].content["cached"] is True
        assert blocked[0].summary.startswith("(cached: identical earlier call, workspace unchanged)")
    finally:
        loop.close()


def test_csv_artifact_validator_reports_inconsistent_record_width(tmp_path):
    loop = make_loop(tmp_path)
    try:
        path = tmp_path / "dataset.csv"
        path.write_text(
            "price,area,rooms,parking,elevator,neighborhood,url,retrieved_at\n"
            "1000000,,2,,,\n",
            encoding="utf-8",
        )
        call = ToolCall(
            name="write_file",
            id="write-csv",
            valid=True,
            args={"file_path": "dataset.csv", "content": "irrelevant-to-the-validator"},
        )
        result = ToolResult(
            success=True,
            name="write_file",
            content={"path": str(path), "success": True},
        )

        findings = loop._validate_written_artifact(call, result)
        assert len(findings) == 1
        assert "line 2 has 6 fields (expected 8)" in findings[0]
    finally:
        loop.close()



def test_csv_validator_rejects_date_only_value_in_timestamp_column(tmp_path):
    loop = make_loop(tmp_path)
    try:
        path = tmp_path / "dataset.csv"
        path.write_text(
            "retrieved_at,listing_url\n"
            "2026-10-09,https://example.com/listing\n",
            encoding="utf-8",
        )
        call = ToolCall(
            name="write_file",
            id="write-timestamp",
            valid=True,
            args={"file_path": "dataset.csv", "content": "irrelevant-to-validator"},
        )
        result = ToolResult(
            success=True,
            name="write_file",
            content={"path": str(path), "success": True},
        )

        findings = loop._validate_written_artifact(call, result)
        assert any("date without a time" in finding for finding in findings)
    finally:
        loop.close()


def test_write_result_itself_exposes_artifact_validation_failure_to_next_turn(tmp_path):
    import json

    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="write_file",
            id="write-invalid-csv",
            valid=True,
            args={
                "file_path": "dataset.csv",
                "content": (
                    "price,area,rooms,url\n"
                    "1000000,90,2\n"
                ),
            },
        )
        loop._execute_tool_calls([call], iteration=1)

        events = loop.stm.get_recent(loop.session_id, limit=30)
        result_events = [
            event for event in events
            if str(getattr(event.type, "value", event.type)) == "tool_result"
        ]
        assert result_events, "Tool execution should persist a result event."
        payload = json.loads(result_events[-1].content)
        assert payload["name"] == "write_file"
        assert payload["success"] is True  # write succeeded; the artifact format did not.
        assert "ARTIFACT VALIDATION FAILED" in payload["summary"]
        findings = payload["content"]["artifact_validation"]["findings"]
        assert any("line 2 has 3 fields (expected 4)" in finding for finding in findings)
        assert payload["metadata"]["validation_findings"] == findings
    finally:
        loop.close()


def test_identical_successful_write_is_blocked_until_content_changes(tmp_path):
    loop = make_loop(tmp_path)
    try:
        content = "a,b\n1,2\n"
        write_call = ToolCall(
            name="write_file",
            id="write-once",
            valid=True,
            args={"file_path": "dataset.csv", "content": content, "overwrite": True},
        )
        # ToolManager resolves the relative workspace path before result handling.
        executed_call = ToolCall(
            name="write_file",
            id="write-once",
            valid=True,
            args={
                "file_path": str(tmp_path / "dataset.csv"),
                "content": content,
                "overwrite": True,
            },
        )
        loop._apply_result(
            executed_call,
            ToolResult(
                success=True,
                name="write_file",
                content={"success": True, "path": str(tmp_path / "dataset.csv")},
            ),
            1,
        )

        allowed, blocked = loop._classify_calls([write_call])
        assert allowed == []
        assert set(blocked) == {0}
        assert blocked[0].content["error"]["type"] == "repeated_mutation_no_progress"

        corrected_call = ToolCall(
            name="write_file",
            id="write-corrected",
            valid=True,
            args={"file_path": "dataset.csv", "content": "a,b\n1,2,3\n", "overwrite": True},
        )
        allowed, blocked = loop._classify_calls([corrected_call])
        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()


def test_repeated_observation_returns_cached_success_before_block_limit(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="read_file", id="cached-read", valid=True,
            args={"file_path": "app.py"},
        )
        result = ToolResult(
            success=True, name="read_file",
            content={"success": True, "path": "app.py", "content": "observed bytes"},
        )
        loop._apply_result(call, result, 1)
        allowed, cached = loop._classify_calls([call])
        assert allowed == []
        assert cached[0].success is True
        assert cached[0].content["cached"] is True
        assert cached[0].summary.startswith("(cached: identical earlier call, workspace unchanged)")
        assert cached[0] is not result
        assert loop.get_metrics()["duplicate_observation_cache_hits"] == 1
    finally:
        loop.close()


def test_observation_cache_blocks_after_three_consecutive_hits(tmp_path):
    loop = make_loop(tmp_path)
    try:
        call = ToolCall(
            name="read_file", id="cached-read", valid=True,
            args={"file_path": "app.py"},
        )
        loop._apply_result(
            call,
            ToolResult(success=True, name="read_file", content={"success": True, "content": "observed"}),
            1,
        )
        for iteration in (2, 3):
            allowed, results = loop._classify_calls([call])
            assert allowed == []
            assert results[0].success
            loop._apply_result(call, results[0], iteration)
        allowed, results = loop._classify_calls([call])
        assert allowed == []
        assert results[0].content["error"]["type"] == "repeated_observation_no_progress"
        assert loop.get_metrics()["duplicate_observation_cache_hits"] == 3
    finally:
        loop.close()


def test_observation_cache_is_invalidated_by_workspace_mutation(tmp_path):
    loop = make_loop(tmp_path)
    try:
        read_call = ToolCall(
            name="read_file", id="cached-read", valid=True,
            args={"file_path": "app.py"},
        )
        loop._apply_result(
            read_call,
            ToolResult(success=True, name="read_file", content={"content": "old"}),
            1,
        )
        allowed, results = loop._classify_calls([read_call])
        assert results[0].content.get("cached") is True
        loop._apply_result(
            ToolCall(
                name="write_file", id="write-after-read", valid=True,
                args={"file_path": "new.txt", "content": "new"},
            ),
            ToolResult(success=True, name="write_file", content={"success": True}),
            3,
        )
        allowed, blocked = loop._classify_calls([read_call])
        assert allowed == [0]
        assert blocked == {}
    finally:
        loop.close()


def test_csv_validator_emits_identical_column_as_advisory_not_failure(tmp_path):
    loop = make_loop(tmp_path)
    try:
        path = tmp_path / "dataset.csv"
        path.write_text(
            "listing_id,price,source\n"
            "a,100,Divar\n"
            "b,200,Divar\n"
            "c,300,Divar\n"
            "d,400,Divar\n"
            "e,500,Divar\n",
            encoding="utf-8",
        )
        call = ToolCall(
            name="write_file", id="advisory-csv", valid=True,
            args={"file_path": "dataset.csv", "content": "ignored"},
        )
        result = ToolResult(
            success=True, name="write_file",
            content={"path": str(path), "success": True},
        )
        findings = loop._validate_written_artifact(call, result)
        assert len([f for f in findings if f.startswith("ADVISORY:")]) == 1
        advisory = next(f for f in findings if f.startswith("ADVISORY:"))
        assert "column 'source' has the same value in all 5 rows" in advisory
        assert loop._validate_written_artifact(call, result) == []
    finally:
        loop.close()


def test_csv_identical_column_advisory_ignores_empty_or_varied_values(tmp_path):
    loop = make_loop(tmp_path)
    try:
        path = tmp_path / "dataset.csv"
        path.write_text(
            "id,price\na,100\nb,100\nc,\nd,100\ne,100\n",
            encoding="utf-8",
        )
        call = ToolCall(name="write_file", id="csv", valid=True, args={"file_path": "dataset.csv"})
        result = ToolResult(success=True, name="write_file", content={"path": str(path)})
        assert not any(item.startswith("ADVISORY:") for item in loop._validate_written_artifact(call, result))
    finally:
        loop.close()
