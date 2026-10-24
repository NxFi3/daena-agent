import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace

import pytest

from src.models.ToolResult import ToolResult
from src.tools.builtin.command_exec.process_manager import ProcessManager
from src.tools.builtin.edit_file.tool import EditFile
from src.tools.builtin.write_file.tool import WriteFile
from src.tools.builtin.grep.tool import Grep
from src.tools.builtin.explore.tool import Explore


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable mode")
def test_overwriting_executable_preserves_mode(tmp_path):
    path = tmp_path / "run.sh"
    path.write_text("old")
    path.chmod(0o751)
    tool = WriteFile()
    tool.set_workspace(tmp_path)
    assert tool.execute("run.sh", "new", overwrite=True).success
    assert stat.S_IMODE(path.stat().st_mode) == 0o751


def test_edit_preserves_utf16_big_endian_and_bom(tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes(b"\xfe\xff" + "سلام old\r\n".encode("utf-16-be"))
    tool = EditFile()
    tool.set_workspace(tmp_path)
    assert tool.execute("text.txt", "old", "new").success
    assert path.read_bytes() == b"\xfe\xff" + "سلام new\r\n".encode("utf-16-be")


def test_generic_tool_evidence_stays_parseable_when_large():
    result = ToolResult(
        True,
        "grep",
        content={
            "matches": [{"file": f"{i}.py", "snippet": "x" * 300} for i in range(60)]
        },
    )
    value = json.loads(result.evidence["data"])
    assert value["matches"] and value["matches_truncated"]


def test_project_search_does_not_walk_browser_profile_by_default(tmp_path):
    private = tmp_path / ".daena" / "browser_use"
    private.mkdir(parents=True)
    (private / "cookies.txt").write_text("needle private")
    (tmp_path / "project.txt").write_text("needle public")
    tool = Grep()
    tool.set_workspace(tmp_path)
    result = tool.execute("needle")
    assert [r["file"] for r in result.content["matches"]] == ["project.txt"]
    assert tool.execute("needle", path=".daena/browser_use").content["matches"]


def test_process_reads_do_not_corrupt_utf8_or_move_writer_cursor():
    text = "سلام دنیا 😀"
    with tempfile.TemporaryFile("w+b") as file:
        file.write(text.encode())
        file.flush()
        end = file.tell()
        offset = 0
        parts = []
        for _ in range(30):
            part, offset = ProcessManager._read_file_incremental(file, offset, 5)
            parts.append(part)
            if offset == end:
                break
        assert "".join(parts) == text
        assert file.tell() == end


def test_reading_process_output_cannot_overwrite_later_output(tmp_path):
    manager = ProcessManager()
    try:
        first = manager.start(
            command=[
                sys.executable,
                "-u",
                "-c",
                "print('x'*20000); input(); print('TAIL')",
            ],
            workdir=tmp_path,
            yield_time_ms=100,
            max_output_chars=512,
        )
        assert first["process_id"]
        parts = [first["stdout"]]
        manager.write(process_id=first["process_id"], input_text="go\n")
        for _ in range(60):
            row = manager.poll(
                process_id=first["process_id"], wait_ms=100, max_output_chars=512
            )
            parts.append(row["stdout"])
            if row["status"] != "running" and not row["stdout"]:
                break
        assert "".join(parts) == "x" * 20000 + "\nTAIL\n"
    finally:
        manager.close()


def test_fast_process_keeps_unread_output_available(tmp_path):
    manager = ProcessManager()
    try:
        row = manager.start(
            command=[sys.executable, "-c", "print('x'*20000)"],
            workdir=tmp_path,
            yield_time_ms=1000,
            max_output_chars=512,
        )
        assert row["status"] == "exited" and row["process_id"]
        parts = [row["stdout"]]
        while row["output_pending"]:
            row = manager.poll(
                process_id=row["process_id"], wait_ms=0, max_output_chars=512
            )
            parts.append(row["stdout"])
        assert "".join(parts) == "x" * 20000 + "\n"
    finally:
        manager.close()


def test_explore_uses_temporary_database_and_reports_incomplete_run(
    monkeypatch, tmp_path
):
    seen = {}

    class Loop:
        def __init__(self, config, llm):
            seen["config"] = config

        def set_workspace(self, path):
            pass

        def run(self, **kwargs):
            return SimpleNamespace(response="Stopped before completion")

        def get_metrics(self):
            return {"completed": False, "stop_reason": "cancelled"}

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr("src.agent.agentloop.Loop", Loop)
    tool = Explore()
    tool.set_workspace(tmp_path)
    tool.set_llm_provider(SimpleNamespace())
    result = tool.execute("Inspect repository")
    assert "database" in seen["config"]["memory"]
    assert not Path(seen["config"]["memory"]["database"]).parent.exists()
    assert seen["closed"]
    assert not result.success


def test_budget_tracks_live_provider_settings_without_mutating_profile():
    from src.context.tokenbudget import TokenBudget

    config = {
        "llm": {"provider": "ollama"},
        "context": {"safe_margin": 0, "max_prompt_tokens": 50000},
    }
    llm = SimpleNamespace(
        provider_name="ollama",
        llm_config={"model_name": "first"},
        generation_config={"num_ctx": 16000, "num_predict": 2000},
    )
    budget = TokenBudget(config, llm)
    assert budget.budget == 14000
    budget.chars_per_token = 3.2
    budget.reserve_tools([{"name": "tool"}])
    llm.generation_config["num_ctx"] = 8000
    budget.refresh(config)
    assert budget.budget == 6000 and budget.chars_per_token == 3.2
    llm.provider_name = "gemini"
    llm.llm_config = {"model_name": "second"}
    llm.generation_config = {"num_ctx": 32000, "max_output_tokens": 4000}
    budget.refresh(config)
    assert budget.budget == 28000 and budget.chars_per_token == 4
    assert budget.tools == [{"name": "tool"}]
    assert config["llm"] == {"provider": "ollama"}


def test_browser_images_follow_live_provider(tmp_path):
    from src.tools.builtin.browser_use.integration import browser_messages

    folder = tmp_path / ".daena" / "browser_use"
    folder.mkdir(parents=True)
    image = folder / "screen.png"
    image.write_bytes(b"PNG bytes")
    tool = SimpleNamespace(
        workspace_root=tmp_path, current_observation=lambda: {"image_ref": str(image)}
    )
    loop = SimpleNamespace(
        config={"llm": {"provider": "ollama"}},
        llm=SimpleNamespace(provider_name="openrouter"),
        tool=SimpleNamespace(get_tool=lambda name: tool),
    )
    message = browser_messages(loop, {"browser_use"})[0]
    assert message["content"][1]["type"] == "image_url" and "images" not in message
    loop.llm.provider_name = "gemini"
    assert browser_messages(loop, {"browser_use"})[0]["computer_image"]


def test_explorer_inherits_cancellation_policy_and_closes_on_failure(
    monkeypatch, tmp_path
):
    from threading import Event

    stop = Event()
    stop.set()
    seen = {}

    class Loop:
        def __init__(self, config, llm):
            seen["config"] = config

        def set_workspace(self, path):
            pass

        def run(self, **kwargs):
            assert kwargs["stop_event"] is stop
            raise InterruptedError("cancelled")

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr("src.agent.agentloop.Loop", Loop)
    tool = Explore()
    tool.set_workspace(tmp_path)
    tool.set_llm_provider(SimpleNamespace())
    tool.set_runtime_config({"security": {"allowed_tools": ["read_file"]}})
    tool.set_process_manager(SimpleNamespace(stop_event=stop))
    assert not tool.execute("Inspect").success
    assert seen["closed"]
    assert seen["config"]["security"]["allowed_tools"] == ["read_file"]
    assert not tool.execute("Inspect", path="../").success


def test_history_session_deletion_is_not_limited_to_recent_500(monkeypatch, tmp_path):
    from uuid import uuid4
    from src.studio.state import StudioStore
    from src.memories.stm.stmdatabase import STMDatabase
    from src.models.ContextEvent import ContextEvent, ContextRole, ContextType

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "global"))
    store = StudioStore(tmp_path / "project")
    session = uuid4()
    other = uuid4()
    # Archive old target turns behind the recent display window.
    first = store.save_run(session, "old", [], {})
    path = store.chat_root / "runs" / f"{first}.json"
    os.utime(path, (1, 1))
    for i in range(501):
        store.save_run(other, str(i), [], {})
    database = store.root / "history.sqlite3"
    with STMDatabase(database) as db:
        db.add(
            session,
            ContextEvent(
                role=ContextRole.USER, type=ContextType.MESSAGE, content="remove me"
            ),
        )
        db.add(
            other,
            ContextEvent(
                role=ContextRole.USER, type=ContextType.MESSAGE, content="keep me"
            ),
        )
        db.save_checkpoint(session, "old summary", [])
    assert len(store.runs()) == 500 and len(store.runs(limit=None)) == 502
    assert store.delete_session(session) == 1
    with STMDatabase(database) as db:
        assert (
            db.connection.execute(
                "SELECT COUNT(*) FROM context_events WHERE session_id=?",
                (str(session),),
            ).fetchone()[0]
            == 0
        )
        assert (
            db.connection.execute(
                "SELECT COUNT(*) FROM context_events WHERE session_id=?", (str(other),)
            ).fetchone()[0]
            == 1
        )
        assert db.get_checkpoint(session) is None
    assert len(store.runs(limit=None)) == 501


def test_process_disposal_releases_output_but_keeps_status(tmp_path):
    manager = ProcessManager()
    row = manager.start(
        command=[sys.executable, "-c", "print('x'*20000)"],
        workdir=tmp_path,
        yield_time_ms=1000,
        max_output_chars=512,
    )
    entry = manager._processes[row["process_id"]]
    manager.close(discard_output=True)
    assert entry.stdout_file.closed and entry.stderr_file.closed
    assert (
        manager.poll(process_id=row["process_id"], wait_ms=0, max_output_chars=512)[
            "status"
        ]
        == "exited"
    )


def test_subagent_file_tools_respect_role_and_parent_policy(tmp_path):
    from src.agent.subagents import SubagentManager

    allowed = [
        "read_file",
        "grep",
        "glob",
        "list_dir",
        "write_file",
        "edit_file",
        "subagent",
        "explore",
    ]
    manager = SubagentManager({"security": {"allowed_tools": allowed}})
    manager.workspace = str(tmp_path)
    child = SimpleNamespace(mode="read_only", allowed_tools=None, snapshot=None)
    config = manager._child_config(child, str(tmp_path))
    assert set(config["security"]["allowed_tools"]) == {
        "read_file",
        "grep",
        "glob",
        "list_dir",
    }
    child.mode = "worker"
    assert set(
        manager._child_config(child, str(tmp_path))["security"]["allowed_tools"]
    ) == set(allowed) - {"subagent", "explore"}
    child.allowed_tools = ["edit_file"]
    assert manager._child_config(child, str(tmp_path))["security"]["allowed_tools"] == [
        "edit_file"
    ]


def test_partial_utf8_waits_for_remaining_bytes_and_invalid_eof_flushes():
    with tempfile.TemporaryFile("w+b") as file:
        file.write(b"\xf0\x9f")
        file.flush()
        assert ProcessManager._read_file_incremental(file, 0, 4) == ("", 0)
        file.write(b"\x98\x80")
        file.flush()
        assert ProcessManager._read_file_incremental(file, 0, 4) == ("😀", 4)
        file.write(b"\xe2")
        file.flush()
        assert ProcessManager._read_file_incremental(file, 4, 4, finished=True) == (
            "�",
            5,
        )


def test_explore_real_loop_with_read_tools_and_isolated_memory(tmp_path):
    from src.models.LLMResult import LLMResult

    (tmp_path / "README.md").write_text("Repository evidence")

    class Model:
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
                                "arguments": json.dumps({"file_path": "README.md"}),
                            },
                        }
                    ],
                    None,
                    1,
                )
            assert "Repository evidence" in json.dumps(messages)
            return LLMResult("README.md contains Repository evidence.", {}, [], None, 1)

    tool = Explore()
    tool.set_workspace(tmp_path)
    tool.set_llm_provider(Model())
    result = tool.execute("Read README.md")
    assert result.success and result.content["completed"]
    assert "Repository evidence" in result.content["report"]
    assert result.content["tool_successes"] == 1
