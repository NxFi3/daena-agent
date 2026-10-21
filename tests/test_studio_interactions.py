"""User input and view interaction regression tests; no live model calls."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QPoint, QPointF, QEvent
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
