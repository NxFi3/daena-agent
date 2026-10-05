from uuid import uuid4

from src.context.contextbuilder import ContextBuilder
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


class FakeLLM:
    def __init__(self):
        self.model = FakeModel()
        self.calls = 0
        self.options = []

    def generate(self, messages, tools=None, options=None):
        self.calls += 1
        self.options.append(dict(options or {}))
        return LLMResult(
            response="Compact state.",
            message={"role": "assistant", "content": "Compact state."},
            tool_calls=[],
            thinking=None,
            usage=10,
        )


def event(role, event_type, content, step):
    return ContextEvent(
        id=uuid4(),
        role=role,
        type=event_type,
        content=content,
        step=step,
    )


def base_config():
    return {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 4096},
            }
        },
        "context": {
            "safe_margin": 0,
            "compaction_enabled": True,
            "compaction_target_tokens": 256,
        },
    }


def test_current_task_is_not_duplicated():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)
    task = event(ContextRole.USER, ContextType.MESSAGE, "fix the parser", 1)

    messages = builder.build_context(
        events=[task],
        task={
            "id": str(task.id),
            "content": task.content,
        },
    )

    user_messages = [
        m for m in messages if m.get("role") == "user"
    ]
    assert [m["content"] for m in user_messages] == ["fix the parser"]


def test_tool_call_and_id_are_preserved():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)
    task = event(ContextRole.USER, ContextType.MESSAGE, "inspect file", 1)
    assistant = event(
        ContextRole.ASSISTANT,
        ContextType.MESSAGE,
        "",
        2,
    )
    assistant.metadata = {
        "llm_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"file_path":"main.py"}',
                },
            }],
        }
    }
    tool = event(
        ContextRole.TOOL,
        ContextType.TOOL_RESULT,
        '{"name":"read_file","tool_call_id":"call_123","success":true,"content":{"path":"main.py","content":"ok"}}',
        3,
    )

    messages = builder.build_context(
        events=[task, assistant, tool],
        task={"id": str(task.id), "content": task.content},
    )

    assistant_messages = [
        m for m in messages if m.get("role") == "assistant"
    ]
    tool_messages = [
        m for m in messages if m.get("role") == "tool"
    ]

    assert assistant_messages[0]["tool_calls"][0]["id"] == "call_123"
    assert tool_messages[0]["tool_call_id"] == "call_123"


def test_compactor_disables_reasoning():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)
    task = event(
        ContextRole.USER,
        ContextType.MESSAGE,
        "finish the task",
        1,
    )

    events = [task]
    for i in range(2, 35):
        events.append(
            event(
                ContextRole.ASSISTANT if i % 2 == 0 else ContextRole.TOOL,
                ContextType.MESSAGE if i % 2 == 0 else ContextType.TOOL_RESULT,
                "x" * 1000,
                i,
            )
        )

    messages = builder.build_context(
        events=events,
        task={"id": str(task.id), "content": task.content},
    )

    assert llm.calls == 1
    assert llm.options[0]["think"] is False
    assert any(
        "<compacted_context>" in str(m.get("content", ""))
        for m in messages
        if m.get("role") == "user"
    )


class EmptyCompactorLLM(FakeLLM):
    def generate(self, messages, tools=None, options=None):
        self.calls += 1
        self.options.append(dict(options or {}))
        return LLMResult(
            response="",
            message={"role": "assistant", "content": ""},
            tool_calls=[],
            thinking="hidden reasoning",
            usage=10,
        )


def test_compactor_failure_keeps_deterministic_recent_history():
    llm = EmptyCompactorLLM()
    builder = ContextBuilder(base_config(), llm)
    task = event(
        ContextRole.USER,
        ContextType.MESSAGE,
        "continue the implementation",
        1,
    )

    events = [task]
    for i in range(2, 35):
        events.append(
            event(
                ContextRole.ASSISTANT if i % 2 == 0 else ContextRole.TOOL,
                ContextType.MESSAGE if i % 2 == 0 else ContextType.TOOL_RESULT,
                f"important state {i} " + ("x" * 1000),
                i,
            )
        )

    messages = builder.build_context(
        events=events,
        task={"id": str(task.id), "content": task.content},
    )

    assert llm.calls == 1
    fallback = next(
        m["content"]
        for m in messages
        if m.get("role") == "user" and "<deterministic_context>" in str(m.get("content", ""))
    )
    assert "important state 34" in fallback
    assert messages[-1] == {"role": "user", "content": "continue the implementation"}


def test_compaction_is_used_when_latest_task_history_does_not_fit():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)
    task = event(
        ContextRole.USER,
        ContextType.MESSAGE,
        "finish the task",
        1,
    )
    events = [task]

    for i in range(2, 35):
        events.append(
            event(
                ContextRole.ASSISTANT if i % 2 == 0 else ContextRole.TOOL,
                ContextType.MESSAGE if i % 2 == 0 else ContextType.TOOL_RESULT,
                "x" * 1000,
                i,
            )
        )

    messages = builder.build_context(
        events=events,
        task={"id": str(task.id), "content": task.content},
    )

    assert llm.calls == 1
    assert any(
        "<compacted_context>" in str(m.get("content", ""))
        for m in messages
        if m.get("role") == "user"
    )
    assert messages[-1] == {"role": "user", "content": "finish the task"}


def test_hard_fit_keeps_prompt_under_budget_without_compaction():
    llm = FakeLLM()
    config = base_config()
    config["context"]["compaction_enabled"] = False
    config["context"]["max_prompt_tokens"] = 512

    builder = ContextBuilder(config, llm)
    task = event(
        ContextRole.USER,
        ContextType.MESSAGE,
        "finish the task",
        1,
    )
    events = [task]
    for i in range(2, 30):
        events.append(
            event(
                ContextRole.ASSISTANT,
                ContextType.MESSAGE,
                "x" * 2000,
                i,
            )
        )

    messages = builder.build_context(
        events=events,
        task={"id": str(task.id), "content": task.content},
    )

    assert builder.tokenbudget.estimate_messages_tokens(messages) <= 512
    assert messages[-1]["role"] == "user"


def test_token_budget_uses_configured_num_ctx():
    llm = FakeLLM()
    config = base_config()
    builder = ContextBuilder(config, llm)

    assert builder.tokenbudget.context_length == 4096
    assert builder.tokenbudget.budget == 4096

def test_token_budget_applies_working_prompt_cap():
    llm = FakeLLM()
    config = base_config()
    config["llm"]["provider_config"]["generation_config"]["num_ctx"] = 120000
    config["context"]["max_prompt_tokens"] = 8192

    builder = ContextBuilder(config, llm)

    assert builder.tokenbudget.context_length == 120000
    assert builder.tokenbudget.budget == 8192


def test_duplicate_read_results_are_collapsed():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)

    def read(path, step, content):
        return {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": f"call-{step}",
            "content": (
                '{"success":true,"path":"'
                + path
                + '","start_line":null,"end_line":null}\n'
                + content
            ),
        }

    messages = [
        read("src/a.py", 1, "A" * 7000),
        read("src/b.py", 2, "B" * 7000),
        read("src/a.py", 3, "A" * 7000),
        read("src/c.py", 4, "C" * 7000),
        read("src/a.py", 5, "A" * 7000),
        read("src/d.py", 6, "D" * 7000),
    ]

    builder._shrink_old_tool_results(messages)

    assert len(messages[4]["content"]) == len(
        read("src/a.py", 3, "A" * 7000)["content"]
    )
    assert len(messages[5]["content"]) == len(read("src/d.py", 6, "D" * 7000)["content"])
    assert len(messages[0]["content"]) <= builder.OLD_TOOL_CHARS + len(builder.OLD_RESULT_MARKER)



def test_untrusted_orphan_tool_result_is_removed():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)

    task = event(ContextRole.USER, ContextType.MESSAGE, "continue", 1)
    orphan = event(
        ContextRole.TOOL,
        ContextType.TOOL_RESULT,
        '{"name":"read_file","tool_call_id":"orphan","success":true,"content":{"path":"x"}}',
        2,
    )

    messages = builder.build_context(
        events=[task, orphan],
        task={"id": str(task.id), "content": task.content},
    )

    assert not any(m.get("role") == "tool" for m in messages)


def test_experience_is_disabled_for_baseline():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)

    task = event(ContextRole.USER, ContextType.MESSAGE, "continue", 1)

    messages = builder.build_context(
        events=[task],
        task={"id": str(task.id), "content": task.content},
    )

    system = messages[0]["content"]
    assert "<experience>" not in system

def test_execution_state_is_visible_and_compact():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)
    task = event(ContextRole.USER, ContextType.MESSAGE, "continue implementation", 1)

    messages = builder.build_context(
        events=[task],
        task={"id": str(task.id), "content": task.content},
        agent_state={
            "status": "succeeded",
            "tool": "read_file",
            "action": "inspect",
            "target": "server.js",
            "iteration": 4,
        },
        progress={"items": ["Read server.js"]},
        working_set={
            "artifacts": {
                "/workspace/server.js": {
                    "status": "known",
                    "known": True,
                    "preview": "const app = express();",
                }
            },
            "verification": {},
            "facts": ["server.js exists"],
            "unresolved": [],
        },
        recent_actions={
            "items": [
                {
                    "iteration": 4,
                    "tool": "read_file",
                    "action": "inspect",
                    "target": "server.js",
                    "outcome": "success",
                    "summary": "Read file server.js.",
                }
            ]
        },
    )

    system = messages[0]["content"]
    assert "<execution_state>" in system
    assert "server.js" in system
    assert "Read server.js" in system


def test_last_failed_verification_survives_unrelated_command_and_clears_on_success():
    from src.context.workingset import WorkingSet
    from src.models.ToolCall import ToolCall
    from src.models.ToolResult import ToolResult

    ws = WorkingSet()

    test_call = ToolCall(
        name="command_exec",
        action="run",
        args={"command": ["npm", "test"], "workdir": "."},
        valid=True,
    )
    failed = ToolResult(
        success=False,
        name="command_exec",
        content={
            "command": ["npm", "test"],
            "workdir": ".",
            "status": "exited",
            "exit_code": 1,
            "stdout": "FAIL tests/api.test.js\nExpected: 200\nReceived: 500",
            "stderr": "",
        },
    )
    ws.update(test_call, failed, 1)
    assert ws.context()["last_failed_verification"]["exit_code"] == 1

    unrelated_call = ToolCall(
        name="command_exec",
        action="run",
        args={"command": ["pwd"], "workdir": "."},
        valid=True,
    )
    unrelated = ToolResult(
        success=True,
        name="command_exec",
        content={
            "command": ["pwd"],
            "workdir": ".",
            "status": "exited",
            "exit_code": 0,
            "stdout": "/workspace",
        },
    )
    ws.update(unrelated_call, unrelated, 2)
    assert ws.context()["last_failed_verification"]["exit_code"] == 1

    fixed = ToolResult(
        success=True,
        name="command_exec",
        content={
            "command": ["npm", "test"],
            "workdir": ".",
            "status": "exited",
            "exit_code": 0,
            "stdout": "PASS all tests",
        },
    )
    ws.update(test_call, fixed, 3)
    assert ws.context()["last_failed_verification"] == {}


def test_failed_verification_evidence_is_visible_to_model():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)
    task = event(
        ContextRole.USER,
        ContextType.MESSAGE,
        "fix failing tests",
        1,
    )

    messages = builder.build_context(
        events=[task],
        task={"id": str(task.id), "content": task.content},
        working_set={
            "last_failed_verification": {
                "tool": "command_exec",
                "command": ["npm", "test"],
                "workdir": "/workspace",
                "success": False,
                "exit_code": 1,
                "output_excerpt": (
                    "FAIL tests/api.test.js\n"
                    "Expected: 200\n"
                    "Received: 500"
                ),
            }
        },
    )

    system = messages[0]["content"]
    assert "last_failed_verification" in system
    assert "Expected: 200" in system
    assert "Received: 500" in system


def test_tool_result_keeps_compact_diagnostic_evidence():
    from src.models.ToolResult import ToolResult

    result = ToolResult(
        success=False,
        name="command_exec",
        content={
            "command": ["npm", "test"],
            "status": "exited",
            "exit_code": 1,
            "stdout": "FAIL tests/api.test.js\\nExpected: 200\\nReceived: 500\\nRan all test suites.",
            "stderr": "",
        },
    )

    assert "diagnostic_excerpt" in result.evidence
    assert "Expected: 200" in result.evidence["diagnostic_excerpt"]
    assert "Received: 500" in result.evidence["diagnostic_excerpt"]


def test_workspace_inventory_is_visible_even_without_old_creation_events(tmp_path):
    from src.context.workingset import WorkingSet

    (tmp_path / "public").mkdir()
    (tmp_path / "public" / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "server.js").write_text("console.log('ok')", encoding="utf-8")

    ws = WorkingSet(str(tmp_path))
    assert ws.refresh_workspace() is True

    state = ws.context()
    paths = {item["path"] for item in state["workspace_inventory"]}

    assert "public" in paths
    assert "public/index.html" in paths
    assert "src/server.js" in paths
    assert state["workspace_file_count"] == 2
    assert state["workspace_directory_count"] == 2

    builder = ContextBuilder(base_config(), FakeLLM())
    task = event(ContextRole.USER, ContextType.MESSAGE, "continue implementation", 1)
    messages = builder.build_context(
        events=[task],
        task={"id": str(task.id), "content": task.content},
        working_set=state,
    )

    system = messages[0]["content"]
    assert "workspace" in system
    assert "public/index.html" in system
    assert "src/server.js" in system


def test_workspace_inventory_refresh_detects_external_changes(tmp_path):
    from src.context.workingset import WorkingSet

    target = tmp_path / "server.js"
    target.write_text("one", encoding="utf-8")

    ws = WorkingSet(str(tmp_path))
    assert ws.refresh_workspace() is True
    assert ws.refresh_workspace() is False

    target.write_text("two", encoding="utf-8")
    assert ws.refresh_workspace() is True

    target.unlink()
    assert ws.refresh_workspace() is True
    paths = {item["path"] for item in ws.context()["workspace_inventory"]}
    assert "server.js" not in paths


def test_historical_tool_events_and_unavailable_tools_are_excluded():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)

    old_task = event(
        ContextRole.USER,
        ContextType.MESSAGE,
        "old task",
        1,
    )
    old_assistant = event(
        ContextRole.ASSISTANT,
        ContextType.MESSAGE,
        "",
        2,
    )
    old_assistant.metadata = {
        "llm_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "old-search",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": {"query": "old"},
                },
            }],
        }
    }
    old_tool = event(
        ContextRole.TOOL,
        ContextType.TOOL_RESULT,
        (
            '{"name":"search","tool_call_id":"old-search","success":false,'
            '"summary":"Tool search is unavailable."}'
        ),
        3,
    )
    current_task = event(
        ContextRole.USER,
        ContextType.MESSAGE,
        "new task",
        10,
    )

    messages = builder.build_context(
        events=[old_task, old_assistant, old_tool, current_task],
        task={
            "id": str(current_task.id),
            "content": current_task.content,
            "step": current_task.step,
        },
        available_tool_names={"read_file", "apply_patch", "command_exec"},
    )

    rendered = "\n".join(str(message) for message in messages)
    assert "old-search" not in rendered
    assert '"name": "search"' not in rendered
    assert all(message.get("role") != "tool" for message in messages)
    assert messages[-1] == {"role": "user", "content": "new task"}


def test_tool_payload_prefers_normalized_observation_state():
    llm = FakeLLM()
    builder = ContextBuilder(base_config(), llm)

    payload = builder._tool_payload({
        "name": "read_file",
        "success": True,
        "summary": "Read src/main.py.",
        "evidence": {"path": "src/main.py", "content": "important finding"},
        "effects": [{"action": "inspect", "target": "src/main.py"}],
        "content": {"path": "src/main.py", "content": "huge raw content"},
    })

    assert "Read src/main.py." in payload
    assert "important finding" in payload
    assert "huge raw content" not in payload
