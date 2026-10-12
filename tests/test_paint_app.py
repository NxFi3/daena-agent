"""Regression tests for the PySide6 Paint app; run with QT_QPA_PLATFORM=offscreen."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from paint_app import PaintCanvas, PaintWindow


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def click(widget, point):
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=point)
    QApplication.processEvents()


def test_window_constructs_and_initial_history_actions_are_disabled(app):
    window = PaintWindow()
    try:
        assert len(window._tool_actions) == 9
        assert len(window.color_buttons) == 16
        assert window.canvas.image.size().width() == 1000
        assert window.canvas.image.size().height() == 700
        assert not window.action_undo.isEnabled()
        assert not window.action_redo.isEnabled()
        assert window._dirty is False
    finally:
        window.close()


def test_freehand_draw_undo_redo(app):
    canvas = PaintCanvas(40, 40)
    canvas.show()
    canvas.set_color(QColor("#ff0000"))
    canvas.set_brush_size(1)
    QApplication.processEvents()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(5, 10))
    QTest.mouseMove(canvas, QPoint(25, 10), delay=5)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(25, 10))
    QApplication.processEvents()

    assert canvas.image.pixelColor(5, 10).name() == "#ff0000"
    assert canvas.image.pixelColor(15, 10).name() == "#ff0000"
    assert canvas._undo_stack

    canvas.undo()
    assert canvas.image.pixelColor(5, 10).name() == "#ffffff"
    assert not canvas._undo_stack
    assert canvas._redo_stack

    canvas.redo()
    assert canvas.image.pixelColor(15, 10).name() == "#ff0000"
    canvas.close()


def test_rectangle_preview_and_commit(app):
    canvas = PaintCanvas(40, 40)
    canvas.set_tool("rect")
    canvas.set_color(QColor("#0000ff"))
    canvas.set_brush_size(1)
    canvas.show()
    QApplication.processEvents()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    QTest.mouseMove(canvas, QPoint(25, 25), delay=5)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(25, 25))
    QApplication.processEvents()
    border_color = canvas.image.pixelColor(5, 15)
    assert border_color.blue() > 200 and border_color.red() < 160
    assert canvas.image.pixelColor(15, 15).name() == "#ffffff"
    assert len(canvas._undo_stack) == 1
    canvas.close()


def test_flood_fill_respects_boundaries(app):
    canvas = PaintCanvas(30, 30)
    image = QImage(30, 30, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setPen(QPen(QColor("black"), 1))
    painter.drawRect(5, 5, 19, 19)
    painter.end()
    canvas.set_image(image)
    canvas.set_tool("fill")
    canvas.set_color(QColor("#00ff00"))
    canvas.show()
    click(canvas, QPoint(10, 10))
    assert canvas.image.pixelColor(10, 10).name() == "#00ff00"
    assert canvas.image.pixelColor(0, 0).name() == "#ffffff"
    assert canvas.image.pixelColor(5, 5).name() == "#000000"
    canvas.close()


def test_save_and_open_round_trip(tmp_path, app):
    window = PaintWindow()
    path = tmp_path / "round_trip.png"
    try:
        window.canvas.image.setPixelColor(8, 9, QColor("#7a22ee"))
        assert window._save_to_path(str(path)) is True
        assert path.exists() and path.stat().st_size > 0
        window.canvas.new_image(20, 18)
        window._load_from_path(str(path))
        assert window.canvas.image.size().width() == 1000
        assert window.canvas.image.size().height() == 700
        assert window.canvas.image.pixelColor(8, 9).name() == "#7a22ee"
        assert window.current_file == str(path.resolve())
        assert window._dirty is False
    finally:
        window.close()


def test_invalid_image_is_rejected_without_replacing_canvas(tmp_path, app):
    window = PaintWindow()
    invalid = tmp_path / "not_an_image.png"
    invalid.write_text("not an image")
    try:
        original_size = window.canvas.image.size()
        with pytest.raises(ValueError, match="could not open"):
            window._load_from_path(str(invalid))
        assert window.canvas.image.size() == original_size
    finally:
        window.close()
