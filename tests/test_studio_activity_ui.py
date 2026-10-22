"""Offline regression tests for Studio chat experience."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from src.studio.window import StudioWindow
from src.studio.widgets import MessageCard

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])

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
        window.on_child_event({"agent_id": "research", "name": "Researcher", "mode": "read_only",
            "event": {"type": "spawned", "task": "Analyze requirements"}})
        window.inspect_child("research")
        assert "Started" in window.child_events.toPlainText()
        assert "Analyze requirements" in window.agent_summary.toPlainText()
        assert "{" not in window.child_events.toPlainText()
    finally:
        window.close()
