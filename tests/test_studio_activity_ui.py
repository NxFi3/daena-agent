"""Offline regression tests for Studio chat experience."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from src.studio.window import StudioWindow
from src.studio.widgets import MessageCard


@pytest.fixture(scope="module")
def app(qt_app):
    return qt_app


def test_activity_link_requires_actual_events(app):
    card = MessageCard("Daena", "Answer")
    assert card.previous_activity.isHidden()
    card.add_activity("Read README.md")
    assert not card.previous_activity.isHidden()
    assert card.activity_log.isHidden()
    card._toggle_activity()
    assert not card.activity_log.isHidden()
    card._toggle_activity()
    assert card.activity_log.isHidden()


def test_one_stop_control_and_subagent_timeline(app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    window = StudioWindow(tmp_path / "project")
    try:
        assert window.stop_button.parentWidget() is not None
        assert window.stop_button not in [window.send_button]
        assert window.stop_button.isHidden()
        window.graph.upsert("parent", "Daena", "Coordinator", "running")
        window.on_child_event(
            {
                "agent_id": "research",
                "name": "Researcher",
                "mode": "read_only",
                "event": {"type": "spawned", "task": "Analyze requirements"},
            }
        )
        window.inspect_child("research")
        assert "Started" in window.child_events.toPlainText()
        assert "Analyze requirements" in window.agent_summary.toPlainText()
        assert "{" not in window.child_events.toPlainText()
    finally:
        window.close()


def test_history_survives_workspace_changes(app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    from src.studio.state import StudioStore
    from uuid import uuid4

    first = StudioStore(tmp_path / "project-a")
    session = uuid4()
    first.save_run(
        session,
        "A chat from project A",
        [
            {"type": "user_message", "text": "Hello"},
            {"type": "final_response", "text": "Hi"},
        ],
        {"completed": True},
    )
    second = StudioStore(tmp_path / "project-b")
    archives = [run for _, run in second.runs()]
    assert any(run["session_id"] == str(session) for run in archives)
    assert any(
        run.get("workspace") == str((tmp_path / "project-a").resolve())
        for run in archives
    )


def test_reasoning_and_tools_are_inline_disclosures(app):
    card = MessageCard("Daena", "Answer")
    assert card.reasoning_button.isHidden()
    assert card.tools.isHidden()

    card.append_reasoning("Thinking about the task")
    assert not card.reasoning_button.isHidden()
    assert card.reasoning_body.isHidden()
    card._toggle_reasoning()
    assert not card.reasoning_body.isHidden()

    card.add_tool_call("read_file", {"file_path": "README.md"})
    assert not card.tools.isHidden()
    assert card.tools.details.isHidden()
    card.tools._toggle()
    assert not card.tools.details.isHidden()
    card.add_tool_result("read_file", True, "Read README.md")
    assert "README.md" in card.tools.details.toPlainText()


def test_intermediate_agent_text_remains_in_conversation_order(
    app, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data-stream"))
    window = StudioWindow(tmp_path / "stream-project")
    try:
        window.transcript.begin_turn()
        window.transcript.append_stream("I will inspect the project first.")
        assert "I will inspect the project first." in window.transcript.live.text
        window.on_event({"type": "generation_start"})
        assert (
            window.transcript.live.phases[0][0] == "I will inspect the project first."
        )
        assert not window.transcript.live.phases[0][1].isHidden()
        window.transcript.append_stream("Final answer begins.")
        window.transcript.finish("", "Completed")
        assert "Final answer begins." in window.transcript.live.text
    finally:
        window.close()
