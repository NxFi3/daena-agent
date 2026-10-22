"""User input and view interaction regression tests; no live model calls."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QPoint, QPointF, QEvent, QThread, Signal
from PySide6.QtGui import QWheelEvent, QMouseEvent, QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog

from src.agent.agent import Agent
from src.models.LLMResult import LLMResult
from src.studio.window import StudioWindow
from src.studio.graph import GraphCanvas
from src.studio.widgets import Composer
from tests.test_studio import Script, wait_until


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ui(tmp_path, app, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    window = StudioWindow(tmp_path)
    window.show()
    app.processEvents()
    yield window
    if window.busy:
        window.stop()
        wait_until(lambda: not window.busy)
    window.close()
    app.processEvents()
    assert not errors


def wheel(canvas, delta):
    point = canvas.viewport().rect().center()
    event = QWheelEvent(
        QPointF(point),
        QPointF(canvas.viewport().mapToGlobal(point)),
        QPoint(),
        QPoint(0, delta),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(canvas.viewport(), event)
    QApplication.processEvents()


def drag(canvas, start, end, button=Qt.LeftButton):
    viewport = canvas.viewport()
    QTest.mousePress(viewport, button, Qt.NoModifier, start)
    move = QMouseEvent(
        QEvent.MouseMove,
        QPointF(end),
        QPointF(viewport.mapToGlobal(end)),
        Qt.NoButton,
        button,
        Qt.NoModifier,
    )
    QApplication.sendEvent(viewport, move)
    QTest.mouseRelease(viewport, button, Qt.NoModifier, end)
    QApplication.processEvents()


def test_enter_sends_shift_enter_inserts_a_line_and_ctrl_enter_still_sends(ui):
    model = Script(
        [
            LLMResult("First answer", {}, [], None, 1),
            LLMResult("Second answer", {}, [], None, 1),
        ]
    )
    ui.agent_factory = lambda c: Agent(c, llm=model)
    ui.input.setFocus()
    QTest.keyClicks(ui.input, "hello")
    QTest.keyClick(ui.input, Qt.Key_Return)
    wait_until(lambda: not ui.busy)
    assert "First answer" in ui.transcript.toPlainText() and not ui.input.toPlainText()
    QTest.keyClicks(ui.input, "first line")
    QTest.keyClick(ui.input, Qt.Key_Return, Qt.ShiftModifier)
    QTest.keyClicks(ui.input, "second line")
    assert ui.input.toPlainText() == "first line\nsecond line" and not ui.busy
    QTest.keyClick(ui.input, Qt.Key_Return, Qt.ControlModifier)
    wait_until(lambda: not ui.busy)
    assert "Second answer" in ui.transcript.toPlainText()


def test_composer_respects_ime_preedit(app):
    composer = Composer()
    calls = []
    composer.submitted.connect(lambda: calls.append(1))
    QApplication.sendEvent(composer, QInputMethodEvent("pending", []))
    QTest.keyClick(composer, Qt.Key_Return)
    assert not calls
    QApplication.sendEvent(composer, QInputMethodEvent("", []))
    QTest.keyClick(composer, Qt.Key_Return)
    assert calls == [1]
    composer.close()


def test_welcome_suggestions_sidebar_toggle_and_keyboard_focus(ui):
    assert ui.transcript.welcome.isVisible()
    ui.transcript.suggestedPrompt.emit("Explain this project")
    assert ui.input.toPlainText() == "Explain this project"
    assert ui.input.hasFocus()
    assert ui.sidebar.isVisible()
    ui.toggle_sidebar()
    assert not ui.sidebar.isVisible()
    ui.toggle_sidebar()
    assert ui.sidebar.isVisible()
    ui.add_message("You", "A real message")
    assert not ui.transcript.welcome.isVisible()
    ui.new_session()
    assert ui.transcript.welcome.isVisible()


def test_theme_switch_persists_without_resetting_session(ui):
    original_session = ui.session_id
    assert ui.config.get("studio", {}).get("theme", "dark") == "dark"
    ui.toggle_theme()
    assert ui.config["studio"]["theme"] == "light"
    assert ui.store.load_config()["studio"]["theme"] == "light"
    assert ui.session_id == original_session
    ui.toggle_theme()
    assert ui.config["studio"]["theme"] == "dark"


def test_tool_receipts_restore_from_archived_conversation(ui):
    session = ui.session_id
    ui.store.save_run(
        session,
        "Check tools",
        [
            {"type": "user_message", "text": "Check tools"},
            {
                "type": "tool_call",
                "name": "read_file",
                "arguments": {"file_path": "README.md"},
            },
            {
                "type": "tool_result",
                "name": "read_file",
                "success": True,
                "summary": "Read README.md",
            },
            {"type": "thinking_delta", "text": "Reasoning summary"},
            {"type": "final_response", "text": "All set"},
        ],
        {"completed": True},
    )
    ui.refresh_history()
    ui.open_history(ui.history.item(0))
    card = ui.transcript.cards[-1]
    assert card.text == "All set"
    assert len(card.tools.calls) == 1
    assert card.tools.calls[0]["status"] == "complete"
    assert "README.md" in card.tools.details.toPlainText()
    assert card.reasoning == "Reasoning summary"
    assert not card.tools.isHidden()
    assert not card.reasoning_button.isHidden()
    assert not card.details_button.isVisible()


def test_rename_history_updates_every_turn_without_changing_session(ui, monkeypatch):
    original_session = ui.session_id
    for title in ("First", "Second"):
        ui.store.save_run(
            original_session,
            title,
            [{"type": "user_message", "text": title}],
            {"completed": True},
        )
    ui.refresh_history()
    monkeypatch.setattr(
        "src.studio.window.QInputDialog.getText",
        lambda *args, **kwargs: ("  My research chat  ", True),
    )
    ui.rename_history(ui.history.item(0))
    assert ui.history.item(0).toolTip() == "My research chat"
    assert {run["title"] for _, run in ui.store.runs()} == {"My research chat"}
    assert ui.session_id == original_session



def test_delete_history_removes_every_turn_for_session(ui, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from uuid import uuid4

    session = uuid4()
    ui.store.save_run(
        session,
        "Delete me",
        [
            {"type": "user_message", "text": "one"},
            {"type": "final_response", "text": "1"},
        ],
        {"completed": True},
    )
    ui.store.save_run(
        session,
        "Delete me",
        [
            {"type": "user_message", "text": "two"},
            {"type": "final_response", "text": "2"},
        ],
        {"completed": True},
    )
    ui.refresh_history()
    target = next(
        ui.history.item(index)
        for index in range(ui.history.count())
        if "Delete me" in ui.history.item(index).text()
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )
    ui.delete_history(target)
    assert all(run["session_id"] != str(session) for _, run in ui.store.runs())
    assert all(
        "Delete me" not in ui.history.item(index).text()
        for index in range(ui.history.count())
    )

def test_wheel_zoom_without_modifiers_and_visible_reset(app):
    canvas = GraphCanvas()
    canvas.resize(800, 550)
    canvas.show()
    canvas.upsert("a", "Coordinator", "Working", x=0, y=0)
    canvas.upsert("b", "Reviewer", "Read only", x=450, y=0)
    canvas.connect_nodes("a", "b")
    canvas.fit_graph()
    app.processEvents()
    initial = canvas.transform().m11()
    wheel(canvas, 120)
    assert canvas.transform().m11() > initial
    wheel(canvas, -120)
    assert abs(canvas.transform().m11() - initial) < 0.01
    canvas.zoom_to(99)
    assert canvas.transform().m11() <= 3.001
    canvas.zoom_to(0)
    assert canvas.transform().m11() >= 0.149
    QTest.keyClick(canvas, Qt.Key_0)
    assert abs(canvas.transform().m11() - 1) < 0.001
    canvas.close()


def test_panning_node_dragging_and_middle_drag_are_distinct(app):
    canvas = GraphCanvas()
    canvas.resize(850, 600)
    canvas.show()
    a = canvas.upsert("a", "A", "Task", x=0, y=0)
    b = canvas.upsert("b", "B", "Task", x=450, y=0)
    canvas.connect_nodes("a", "b")
    canvas.fit_graph()
    app.processEvents()
    position = QPointF(a.pos())
    center = canvas.mapToScene(canvas.viewport().rect().center())
    drag(canvas, QPoint(760, 500), QPoint(680, 440))
    assert (
        canvas.mapToScene(canvas.viewport().rect().center()) != center
        and a.pos() == position
    )
    start = canvas.mapFromScene(a.pos() + QPointF(120, 45))
    path_before = canvas.edges[0][2].path()
    drag(canvas, start, start + QPoint(65, 35))
    assert a.pos() != position and canvas.edges[0][2].path() != path_before
    position = QPointF(a.pos())
    center = canvas.mapToScene(canvas.viewport().rect().center())
    start = canvas.mapFromScene(a.pos() + QPointF(120, 45))
    drag(canvas, start, start + QPoint(35, 25), Qt.MiddleButton)
    assert (
        a.pos() == position
        and canvas.mapToScene(canvas.viewport().rect().center()) != center
    )
    canvas.close()


def test_quick_provider_and_reasoning_controls_persist_independent_profiles(ui):
    ui.execution.reasoning.setCurrentIndex(ui.execution.reasoning.findData("high"))
    assert ui.config["llm"]["provider_config"]["generation_config"]["think"] == "high"
    ui.execution.provider.setCurrentText("openrouter")
    assert ui.config["llm"]["provider"] == "openrouter"
    assert "num_ctx" not in ui.config["llm"]["provider_config"]["generation_config"]
    ui.execution.reasoning.setCurrentIndex(ui.execution.reasoning.findData("low"))
    ui.execution.provider.setCurrentText("ollama")
    assert ui.config["llm"]["provider_config"]["generation_config"]["think"] == "high"
    ui.settings.reasoning.setCurrentIndex(ui.settings.reasoning.findData("off"))
    ui.settings.apply_form()
    assert ui.config["llm"]["provider_config"]["generation_config"]["think"] is False
    assert ui.execution.reasoning.currentData() == "off"


def test_chat_header_model_selector_and_reasoning_slider(ui):
    ui._refresh_model_menu()
    assert "Choose another model" in " ".join(a.text() for a in ui.model_menu.actions())
    ui._select_model("qwen-test:9b")
    assert ui.config["llm"]["provider_config"]["model_name"] == "qwen-test:9b"
    assert "qwen-test:9b" in ui.model_hint.text()
    ui.reasoning_slider.setValue(3)
    assert ui.execution.reasoning.currentData() == "medium"
    assert ui.config["llm"]["provider_config"]["generation_config"]["think"] == "medium"
    assert ui.reasoning_label.text() == "Medium"
    ui.open_diagnostics()
    assert ui.stream_tabs.isVisible() and ui.execution.isVisible()
    ui.diagnostics.close()


def test_automation_keeps_each_agents_edits_and_results_separate(ui):
    panel = ui.automation
    panel.add_node()
    a = panel.selected
    panel.title.setText("First agent")
    panel.prompt.setPlainText("Investigate the API")
    panel.add_node()  # Commits agent 1 even without pressing Update.
    b = panel.selected
    panel.title.setText("Second agent")
    panel.prompt.setPlainText("Write tests for the API")
    panel.select_node(a)
    assert panel.title.text() == "First agent"
    assert panel.prompt.toPlainText() == "Investigate the API"
    panel.select_node(b)
    assert panel.title.text() == "Second agent"
    assert panel.prompt.toPlainText() == "Write tests for the API"
    panel.handle_event(
        {
            "type": "automation_node",
            "node_id": a,
            "status": "completed",
            "response": "API reviewed",
        }
    )
    panel.handle_event(
        {
            "type": "automation_node",
            "node_id": b,
            "status": "completed",
            "response": "Tests written",
        }
    )
    assert panel.report.toPlainText() == "Tests written"
    panel.select_node(a)
    assert panel.report.toPlainText() == "API reviewed"
    panel.select_node(b)
    assert panel.report.toPlainText() == "Tests written"
    assert panel.graph.nodes[0].title == "First agent"
    assert panel.graph.nodes[1].prompt == "Write tests for the API"


def test_automation_draft_survives_panel_reload(ui):
    from src.studio.automation import AutomationPanel

    panel = ui.automation
    panel.add_node()
    panel.title.setText("Draft researcher")
    panel.prompt.setPlainText("Find the requirements")
    assert panel.flush_draft()
    assert (ui.store.root / "automation-draft.json").exists()
    restored = AutomationPanel()
    restored.set_draft_path(ui.store.root / "automation-draft.json")
    assert len(restored.graph.nodes) == 1
    assert restored.graph.nodes[0].title == "Draft researcher"
    assert restored.graph.nodes[0].prompt == "Find the requirements"
    restored.select_node(restored.graph.nodes[0].id)
    assert restored.prompt.toPlainText() == "Find the requirements"
    restored.close()


def test_automation_runs_away_from_chat_history(ui, monkeypatch):
    chat = ui.add_message("You", "Original conversation")
    ui.journal = [{"type": "user_message", "text": "Original conversation"}]
    first_session = ui.session_id
    ui.graph.upsert("parent", "Daena", "Original graph", "completed")

    class Worker(QThread):
        ready = Signal(object)
        event = Signal(dict)
        result = Signal(dict)

        def __init__(
            self,
            config,
            prompt,
            *,
            agent=None,
            session_id=None,
            blueprint=None,
            parent=None,
        ):
            super().__init__(parent)
            from threading import Event

            self.stop = Event()
            self.blueprint = blueprint

        def run(self):
            self.event.emit(
                {
                    "type": "automation_node",
                    "node_id": self.blueprint.nodes[0].id,
                    "status": "running",
                }
            )
            self.event.emit(
                {
                    "type": "automation_event",
                    "node_id": self.blueprint.nodes[0].id,
                    "event": {"type": "content_delta", "text": "Progress"},
                }
            )
            self.event.emit(
                {
                    "type": "automation_node",
                    "node_id": self.blueprint.nodes[0].id,
                    "status": "completed",
                    "response": "Done",
                }
            )
            self.result.emit(
                {
                    "text": "Automation completed",
                    "metrics": {"completed": True, "outputs": {}},
                }
            )

    monkeypatch.setattr("src.studio.window.RuntimeWorker", Worker)
    panel = ui.automation
    panel.add_node()
    panel.title.setText("Independent agent")
    panel.prompt.setPlainText("Do independent work")
    panel.start()
    wait_until(lambda: not ui.busy)
    assert ui.session_id == first_session
    assert ui.transcript.cards == [chat]
    assert ui.journal == [{"type": "user_message", "text": "Original conversation"}]
    assert set(ui.graph.nodes) == {"parent"}
    assert ui.store.runs() == []
    assert list((ui.store.root / "automation_runs").glob("*.json"))
    assert panel.node_results[panel.graph.nodes[0].id] == "Done"
    assert panel.report.toPlainText() == "Done"
    assert panel.run_state.text() == "Automation completed"
    assert not panel.stop_button.isVisible()


def test_automation_ports_support_drag_to_connect(app):
    from src.studio.graph import AgentNode

    canvas = GraphCanvas(editable=True)
    canvas.resize(900, 520)
    canvas.show()
    canvas.upsert("first", "First", "Draft", x=0, y=0)
    canvas.upsert("second", "Second", "Review", x=520, y=0)
    canvas.fit_graph()
    app.processEvents()
    emitted = []
    canvas.wireRequested.connect(
        lambda source, target: emitted.append((source, target))
    )
    start = canvas.mapFromScene(
        canvas.nodes["first"].pos() + QPointF(AgentNode.WIDTH, AgentNode.HEIGHT / 2)
    )
    end = canvas.mapFromScene(
        canvas.nodes["second"].pos() + QPointF(0, AgentNode.HEIGHT / 2)
    )
    drag(canvas, start, end)
    assert emitted == [("first", "second")]
    assert canvas.pending_wire is None and canvas.wire_preview is None
    canvas.close()


def test_tool_picker_saves_real_names_and_searches_descriptions(ui, monkeypatch):
    from src.studio.controls import ToolScopeDialog

    def accept(dialog):
        dialog.editor.set_scope(["read_file", "web_search"])
        dialog.editor.search.setText("web")
        assert any(
            not dialog.editor.list.item(i).isHidden()
            for i in range(dialog.editor.list.count())
        )
        return QDialog.Accepted

    monkeypatch.setattr(ToolScopeDialog, "exec", accept)
    ui.choose_tools()
    assert set(ui.config["security"]["allowed_tools"]) == {"read_file", "web_search"}
    assert "2" in ui.execution.tools.text()


def test_streaming_updates_a_single_message_and_restores_full_conversation(ui):
    class Streaming:
        model = None

        def generate(self, messages, on_event=None, **kwargs):
            for event in [
                {"type": "generation_start"},
                {"type": "thinking_delta", "text": "A short thought"},
                {"type": "content_delta", "text": "A complete "},
                {"type": "content_delta", "text": "answer"},
                {"type": "generation_done"},
            ]:
                on_event(event)
            return LLMResult("A complete answer", {}, [], None, 1)

    ui.agent_factory = lambda c: Agent(c, llm=Streaming())
    ui.input.setPlainText("First request")
    ui.send()
    wait_until(lambda: not ui.busy)
    assert ui.transcript.toPlainText().count("A complete answer") == 1
    assert ui.transcript.cards[-1].reasoning == "A short thought"
    ui.input.setPlainText("Followup request")
    ui.send()
    wait_until(lambda: not ui.busy)
    assert ui.history.count() == 1 and len(ui.store.runs()) == 2
    ui.open_history(ui.history.item(0))
    text = ui.transcript.toPlainText()
    assert (
        "First request" in text
        and "Followup request" in text
        and text.count("A complete answer") == 2
    )
