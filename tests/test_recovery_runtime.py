from pathlib import Path
from uuid import uuid4

from src.agent.agentloop import Loop
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


def _config():
    return {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 4096},
            }
        },
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 20,
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
        "max_agent_iterations": 10,
        "experience": {"enabled": False},
    }


class _FakeLLM:
    def __init__(self):
        self.model = type("Model", (), {"defaultConfig": {"num_ctx": 4096}})()

    def generate(self, messages, tools=None):
        raise AssertionError("LLM should not be called in this runtime-gate unit test")


def _failed_test_call():
    return ToolCall(
        name="command_exec",
        id="test-1",
        action="run",
        valid=True,
        args={"command": ["npm", "test"], "workdir": "."},
    )


def _failed_test_result():
    return ToolResult(
        success=False,
        name="command_exec",
        content={
            "success": False,
            "status": "exited",
            "exit_code": 1,
            "command": ["npm", "test"],
            "workdir": ".",
            "stdout": "FAIL tests/api.test.js\\nExpected: 200\\nReceived: 404",
            "stderr": "",
            "error": {
                "type": "test_failure",
                "message": "Expected 200, received 404",
            },
        },
        metadata={},
        summary="Tests failed: Expected 200, received 404.",
    )


def test_new_mutation_is_blocked_until_fresh_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    loop = Loop(_config(), _FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))

    try:
        failed_call = _failed_test_call()
        loop._apply_result(failed_call, _failed_test_result(), 1)

        # A different patch is still the same blind recovery strategy.
        patch = ToolCall(
            name="apply_patch",
            id="patch-1",
            action="modify",
            target="server.js",
            valid=True,
            args={
                "patch": "*** Update File: server.js\\n@@\\n-old\\n+new",
            },
        )
        blocked = loop._runtime_recovery_gate(patch)
        assert blocked is not None
        assert blocked.metadata["recovery_required"] is True
        assert blocked.content["error"]["type"] == "diagnosis_required"

        # Observation is explicitly allowed and unlocks a corrective mutation.
        read = ToolCall(
            name="read_file",
            id="read-1",
            action="inspect",
            target="server.js",
            valid=True,
            args={"file_path": "server.js"},
        )
        assert loop._runtime_recovery_gate(read) is None

        read_result = ToolResult(
            success=True,
            name="read_file",
            content={
                "success": True,
                "path": str(Path(tmp_path) / "server.js"),
                "content": "app.post('/api/projects/:id/tasks', handler);",
            },
            metadata={},
            summary="Read server.js.",
        )
        loop._apply_result(read, read_result, 2)

        assert loop._runtime_recovery_gate(patch) is None
        state = loop._recovery_context()
        assert state["status"] == "evidence_collected"
    finally:
        loop.close()


def test_recovery_state_is_model_visible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    loop = Loop(_config(), _FakeLLM())
    loop.session_id = uuid4()
    loop.set_workspace(str(tmp_path))

    try:
        loop._apply_result(_failed_test_call(), _failed_test_result(), 1)
        state = loop._recovery_context()

        assert state["status"] == "diagnosis_required"
        assert "Expected 200" in state["failure"]["summary"]
        assert "Inspect/search/reproduce" in state["next_action"]
    finally:
        loop.close()
