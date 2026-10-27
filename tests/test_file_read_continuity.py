import hashlib
import json
from uuid import uuid4

from src.context.file_reads import delivered_reads
from src.context.payloads import fit_json
from src.tools.builtin.readfile import ReadFile
from src.memories.stm import STM
from src.context.contextservice import ContextService
from src.agent.agentstate import AgentState
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.context.contextbuilder import ContextBuilder


def test_unicode_long_lines_and_crlf_pages_reconstruct_exact_source(tmp_path):
    path = tmp_path / "source.c"
    text = "/* " + "سلام😀" * 1500 + " */\r\nint main() {\r\n  return 0;\r\n}\r\n"
    path.write_bytes(text.encode())
    result = ReadFile().execute(str(path), max_output_chars=512)
    parts = []
    while True:
        assert result.success, result.content
        payload = result.content
        assert payload["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert payload["source_end_offset"] - payload["source_start_offset"] == len(
            payload["content"]
        )
        assert (
            payload["content"]
            == text[payload["source_start_offset"] : payload["source_end_offset"]]
        )
        assert len(payload["content"]) <= 512
        parts.append(payload["content"])
        if payload["next_read"] is None:
            break
        result = ReadFile().execute(**payload["next_read"])
    assert "".join(parts) == text


def test_changed_file_rejects_old_continuation(tmp_path):
    path = tmp_path / "source"
    path.write_text("a" * 2000)
    first = ReadFile().execute(str(path), max_output_chars=512)
    path.write_text("b" * 2000)
    result = ReadFile().execute(**first.content["next_read"])
    assert not result.success and result.content["error"]["type"] == "file_changed"


def test_transport_bounds_are_exact_and_continuation_is_truthful(tmp_path):
    path = tmp_path / "source"
    text = "line abc سلام\n" * 3000
    path.write_bytes(text.encode("utf-8"))
    read = ReadFile().execute(str(path), max_output_chars=32000)
    canonical = json.dumps(read.evidence)
    original = {"success": True, "evidence": read.evidence, "summary": read.summary}
    for limit in (8000, 6000, 2500, 1500, 256):
        encoded = fit_json(original, limit)
        assert len(encoded) <= limit
        payload = json.loads(encoded)
        receipts = list(delivered_reads([{"role": "tool", "content": encoded}]))
        if receipts:
            evidence = receipts[0]
            assert (
                evidence["content"]
                == text[evidence["source_start_offset"] : evidence["source_end_offset"]]
            )
            assert evidence["source_end_offset"] < len(text)
            continuation = ReadFile().execute(**evidence["next_read"])
            assert continuation.success
            assert (
                continuation.content["source_start_offset"]
                == evidence["source_end_offset"]
            )
        else:
            assert payload.get("details_omitted")
    assert json.dumps(read.evidence) == canonical


class Model:
    model = None

    def generate(self, *args, **kwargs):
        return LLMResult("Historical summary", {}, [], None, 1)


def test_delivered_history_survives_restart_and_recent_window_eviction(tmp_path):
    path = tmp_path / "source"
    path.write_text("x" * 16000)
    read = ReadFile().execute(str(path), max_output_chars=32000)
    message = {
        "role": "tool",
        "content": fit_json({"success": True, "evidence": read.evidence}, 2000),
    }
    actual = list(delivered_reads([message]))[0]
    session = uuid4()
    database = tmp_path / "history.sqlite3"
    stm = STM(database)
    service = ContextService({}, Model(), stm)
    service.context = [message]
    assert not stm.get_file_reads(session)[
        "items"
    ]  # Building a prompt is not delivery.
    service.record_delivery(session)
    stm.save_checkpoint(session, "Historical summary", [])
    stm.close()
    stm = STM(database)
    try:
        service = ContextService({}, Model(), stm)
        task = ContextEvent(
            role=ContextRole.USER, type=ContextType.MESSAGE, content="Continue fixing"
        )
        messages = service.get_context(
            session,
            task,
            AgentState(),
            workspace_directory=str(tmp_path),
            recent_limit=1,
        )
        system = (
            json.loads(messages[0]["content"])
            if messages[0]["content"].startswith("{")
            else messages[0]["content"]
        )
        assert "file_read_history" in str(system)
        assert str(path) in str(system)
        rows = stm.get_file_reads(session)["items"]
        assert rows[0]["delivered_character_ranges"] == [
            [0, actual["source_end_offset"]]
        ]
        assert rows[0]["delivered_character_ranges"] != [[0, len(path.read_text())]]
        assert not stm.get_file_reads(uuid4())["items"]
        stm.clear_session(session)
        assert not stm.get_file_reads(session)["items"]
    finally:
        stm.close()


def test_versions_and_ranges_merge_without_claiming_unseen_text(tmp_path):
    path = tmp_path / "source"
    path.write_text("a" * 3000)
    one = ReadFile().execute(str(path), max_output_chars=512)
    two = ReadFile().execute(str(path), start_offset=1024, max_output_chars=512)
    with STM(tmp_path / "memory.sqlite3") as stm:
        session = uuid4()
        stm.record_file_reads(session, [one.evidence, two.evidence, one.evidence])
        assert stm.get_file_reads(session)["items"][0][
            "delivered_character_ranges"
        ] == [[0, 512], [1024, 1536]]
        path.write_text("b" * 3000)
        new = ReadFile().execute(str(path), max_output_chars=512)
        stm.record_file_reads(session, [new.evidence])
        assert stm.get_file_reads(session)["total_versions"] == 2


def test_read_deduplication_includes_source_version(tmp_path):
    path = tmp_path / "source"
    path.write_text("old")
    one = ReadFile().execute(str(path))
    path.write_text("new")
    two = ReadFile().execute(str(path))

    def message(read):
        return {
            "role": "tool",
            "tool_name": "read_file",
            "content": json.dumps({"success": True, "evidence": read.evidence}),
        }

    assert ContextBuilder._read_signature(
        message(one)
    ) != ContextBuilder._read_signature(message(two))


def test_real_loop_records_only_delivered_pages(tmp_path):
    from src.agent.agentloop import Loop

    path = tmp_path / "main.c"
    path.write_text("/* source */\n" * 1000)

    class Reader:
        model = None

        def __init__(self):
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                args = {"file_path": str(path), "max_output_chars": 512}
            elif self.calls == 2:
                receipt = list(delivered_reads(messages))[-1]
                assert len(receipt["content"]) == 512
                args = receipt["next_read"]
            else:
                assert "file_read_history" in messages[0]["content"]
                return LLMResult("Read the first two source pages.", {}, [], None, 1)
            return LLMResult(
                "",
                {},
                [
                    {
                        "id": str(self.calls),
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(args),
                        },
                    }
                ],
                None,
                1,
            )

    loop = Loop(
        {
            "memory": {"database": str(tmp_path / "history.db")},
            "max_agent_iterations": 4,
        },
        Reader(),
    )
    try:
        result = loop.run(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Inspect the first two pages",
            ),
            str(tmp_path),
        )
        assert result.response == "Read the first two source pages."
        rows = loop.context.stm.get_file_reads(loop.session_id)["items"]
        assert rows[0]["delivered_character_ranges"] == [[0, 1024]]
        assert loop.get_metrics()["tool_successes"] == 2
    finally:
        loop.close()


def test_legacy_reads_with_changed_text_are_not_duplicates():
    header = json.dumps({"path": "/src/main.c", "start_line": 1, "end_line": 2})
    old = {
        "role": "tool",
        "tool_name": "read_file",
        "content": header + "\n[content]\nold value",
    }
    new = {**old, "content": header + "\n[content]\nnew value"}
    assert ContextBuilder._read_signature(old) != ContextBuilder._read_signature(new)


def test_small_recent_window_preserves_complete_parallel_read_exchange(tmp_path):
    from src.agent.agentloop import Loop

    paths = []
    for i in range(6):
        path = tmp_path / f"{i}.h"
        path.write_bytes(f"// header {i}\n".encode("utf-8"))
        paths.append(path)

    class BatchReader:
        model = None

        def __init__(self):
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    "",
                    {},
                    [
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"file_path": str(path)}),
                            },
                        }
                        for i, path in enumerate(paths)
                    ],
                    None,
                    1,
                )
            delivered = list(delivered_reads(messages))
            assert len(delivered) == 6
            assert {r["content"] for r in delivered} == {p.read_text() for p in paths}
            return LLMResult("All six headers were delivered.", {}, [], None, 1)

    loop = Loop(
        {
            "memory": {"database": str(tmp_path / "history.db")},
            "context": {"recent_event_limit": 2},
            "max_agent_iterations": 3,
        },
        BatchReader(),
    )
    try:
        result = loop.run(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Inspect these headers",
            ),
            str(tmp_path),
        )
        assert result.response == "All six headers were delivered."
        assert loop.context.stm.get_file_reads(loop.session_id)["total_versions"] == 6
    finally:
        loop.close()


def test_clearing_session_invalidates_in_memory_exchanges(tmp_path):
    from src.agent.agentloop import Loop

    path = tmp_path / "source"
    path.write_text("unique previously read source text")

    class Reader:
        model = None

        def __init__(self):
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    "",
                    {},
                    [
                        {
                            "id": "read",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"file_path": str(path)}),
                            },
                        }
                    ],
                    None,
                    1,
                )
            if self.calls == 3:
                assert "unique previously read source text" not in json.dumps(messages)
                assert "file_read_history" not in messages[0]["content"]
            return LLMResult("Ready", {}, [], None, 1)

    loop = Loop({"memory": {"database": str(tmp_path / "history.db")}}, Reader())
    try:
        task = ContextEvent(
            role=ContextRole.USER, type=ContextType.MESSAGE, content="Read source"
        )
        assert loop.run(task, str(tmp_path)).response == "Ready"
        loop.context.stm.clear_session(loop.session_id)
        followup = ContextEvent(
            role=ContextRole.USER, type=ContextType.MESSAGE, content="Start fresh"
        )
        assert loop.run(followup, str(tmp_path)).response == "Ready"
    finally:
        loop.close()


def test_read_catalog_yields_space_to_current_failure_without_invalid_json():
    builder = ContextBuilder.__new__(ContextBuilder)
    value = builder._compact_execution_state(
        agent_state=None,
        progress=None,
        observation=None,
        recent_actions=None,
        working_set={
            "last_failed_verification": {
                "command": ["build"],
                "exit_code": 1,
                "output_excerpt": "failure " * 400,
            },
            "file_read_history": {
                "items": [
                    {
                        "path": f"file-{i}.c",
                        "file_sha256": "a" * 64,
                        "delivered_character_ranges": [[0, 100]],
                    }
                    for i in range(32)
                ],
                "total_versions": 32,
                "shown_versions": 32,
                "truncated": False,
            },
        },
    )
    state = json.loads(value)
    assert len(value) <= builder.MAX_EXECUTION_STATE_CHARS
    assert state["last_failed_verification"]["exit_code"] == 1
    assert state["file_read_history"]["truncated"]
