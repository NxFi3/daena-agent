"""Studio conversation UI regression tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from src.studio.widgets import MessageCard, ToolActivity


def test_tool_receipts_are_correlated_and_previewed():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = ToolActivity()
    panel.add_call("read_file", {"file_path": "a.py"}, "a")
    panel.add_call("read_file", {"file_path": "b.py"}, "b")
    panel.finish_call("read_file", True, "ok", "b")
    assert panel.calls[0]["status"] == "running"
    assert panel.calls[1]["status"] == "complete"
    assert "a.py" in panel.preview.text()
    assert "b.py" in panel.preview.text()
    panel.close()


def test_markdown_is_rendered_during_streaming():
    app = QApplication.instance() or QApplication([])
    card = MessageCard("Daena")
    card.append_stream("**bold**")
    card._render_stream_markdown()
    assert card.body.toPlainText() == "bold"
    card.close()


def test_activity_keeps_narration_and_parallel_receipts_in_order(qt_app):
    from src.studio.widgets import ConversationView

    view = ConversationView()
    view.begin_turn()
    card = view.live
    view.append_stream("Inspect first.")
    card.add_tool_call("read_file", {"path": "a.py"}, "a")
    card.add_tool_call("read_file", {"path": "b.py"}, "b")
    card.add_tool_result("read_file", True, "Read B", "b")
    card.add_tool_result("read_file", False, "Missing A", "a")
    view.append_stream("Now verify.")
    card.add_tool_call("command_exec", {"command": ["pytest"]}, "c")
    card.add_tool_result("command_exec", True, "Passed", "c")
    view.append_stream("**Finished.**")
    view.finish("**Finished.**")
    assert len(card.tool_groups) == 2
    blocks = [
        card.work_layout.itemAt(i).widget() for i in range(card.work_layout.count())
    ]
    assert blocks.index(card.phases[0][1]) < blocks.index(card.tool_groups[0])
    assert blocks.index(card.tool_groups[0]) < blocks.index(card.phases[1][1])
    assert blocks.index(card.phases[1][1]) < blocks.index(card.tool_groups[1])
    assert card.work.isHidden()
    assert card.body.toPlainText() == "Finished."
    assert "1 failed" in card.previous_activity.text()
    card._toggle_activity()
    assert not card.work.isHidden()
    assert card.tool_groups[0].details.isHidden()
    card.tool_groups[0]._toggle()
    assert "Read B" in card.tool_groups[0].details.toPlainText()
    view.close()


def test_replay_uses_ordered_activity_and_preserves_interrupted_draft(qt_app):
    from src.studio.widgets import ConversationView

    view = ConversationView()
    events = [
        {"type": "user_message", "text": "Inspect"},
        {"type": "content_delta", "text": "First update"},
        {"type": "tool_call", "name": "read_file", "call_id": "r"},
        {"type": "tool_result", "name": "read_file", "call_id": "r", "success": True},
        {"type": "generation_start"},
        {"type": "content_delta", "text": "Second update"},
        {"type": "tool_call", "name": "command_exec", "call_id": "c"},
        {"type": "generation_start"},
        {"type": "content_delta", "text": "Partial draft"},
    ]
    view.restore_turn(events)
    card = view.cards[-1]
    assert [text for text, _ in card.phases] == ["First update", "Second update"]
    assert card.text == "Partial draft"
    assert len(card.tool_groups) == 2
    assert card.tool_groups[-1].calls[0]["status"] == "unfinished"
    assert card.work.isHidden()
    assert view.live is None
    view.close()


def test_missing_window_module_has_actionable_startup_error(monkeypatch, capsys):
    import builtins
    from src.studio.app import main

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "src.studio.window":
            raise ModuleNotFoundError("missing window", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    import pytest

    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
    assert "git restore --source=HEAD" in capsys.readouterr().err


def test_launcher_constructs_and_shows_studio(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from src.studio.app import main
    from src.studio.window import StudioWindow

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    shown = []

    def run(self):
        windows = [
            w
            for w in self.topLevelWidgets()
            if isinstance(w, StudioWindow) and w.isVisible()
        ]
        shown.extend(windows)
        for window in windows:
            window.close()
        return 0

    monkeypatch.setattr(QApplication, "exec", run)
    assert main(["--cwd", str(tmp_path)]) == 0
    assert len(shown) == 1
