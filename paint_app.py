#!/usr/bin/env python3
"""A practical, self-contained Paint-style raster editor built with PySide6."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QCursor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


PALETTE = [
    "#000000", "#FFFFFF", "#808080", "#C0C0C0",
    "#800000", "#FF0000", "#808000", "#FFFF00",
    "#008000", "#00FF00", "#008080", "#00FFFF",
    "#000080", "#0000FF", "#800080", "#FF00FF",
]
TOOLS = (
    ("pencil", "Pencil"),
    ("brush", "Brush"),
    ("eraser", "Eraser"),
    ("line", "Line"),
    ("rect", "Rectangle"),
    ("ellipse", "Ellipse"),
    ("fill", "Fill"),
    ("text", "Text"),
    ("eyedropper", "Pick color"),
)
MAX_HISTORY = 40
MAX_IMAGE_PIXELS = 40_000_000


class PaintCanvas(QWidget):
    """Canvas widget: drawing input is handled on the same widget that paints it."""

    changed = Signal()
    colorPicked = Signal(QColor)
    cursorPositionChanged = Signal(QPoint)
    historyChanged = Signal(bool, bool)

    def __init__(self, width: int = 1000, height: int = 700, parent=None):
        super().__init__(parent)
        self.setObjectName("paintCanvas")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.image = QImage(width, height, QImage.Format.Format_ARGB32)
        self.image.fill(QColor("white"))
        self.setFixedSize(width, height)

        self.tool = "pencil"
        self.color = QColor("#000000")
        self.brush_size = 3

        self._drawing = False
        self._start = QPoint()
        self._last = QPoint()
        self._preview_end: QPoint | None = None
        self._before_stroke: QImage | None = None
        self._undo_stack: list[QImage] = []
        self._redo_stack: list[QImage] = []
        self._update_cursor()

    def sizeHint(self) -> QSize:
        return self.image.size()

    def set_tool(self, tool: str) -> None:
        if tool not in {name for name, _ in TOOLS}:
            raise ValueError(f"Unknown paint tool: {tool}")
        self.tool = tool
        self._update_cursor()

    def set_color(self, color: QColor) -> None:
        if color.isValid():
            self.color = QColor(color)
            self._update_cursor()

    def set_brush_size(self, size: int) -> None:
        self.brush_size = max(1, min(100, int(size)))
        self._update_cursor()

    def set_image(self, image: QImage) -> None:
        if image.isNull():
            raise ValueError("Cannot set the canvas to an invalid image.")
        if image.width() * image.height() > MAX_IMAGE_PIXELS:
            raise ValueError("Image is too large to edit safely (maximum 40 megapixels).")
        self.image = image.convertToFormat(QImage.Format.Format_ARGB32).copy()
        self.setFixedSize(self.image.size())
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._drawing = False
        self._preview_end = None
        self._before_stroke = None
        self.update()
        self.changed.emit()
        self._emit_history_state()

    def new_image(self, width: int, height: int) -> None:
        width = max(1, min(10000, int(width)))
        height = max(1, min(10000, int(height)))
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Canvas dimensions exceed the 40-megapixel limit.")
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(QColor("white"))
        self.set_image(image)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#bdbdbd"))
        painter.drawImage(0, 0, self.image)

        if self._drawing and self._preview_end is not None and self.tool in {
            "line", "rect", "ellipse"
        }:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._draw_shape(painter, self._start, self._preview_end)
        painter.setPen(QPen(QColor("#777777"), 1))
        painter.drawRect(QRect(0, 0, self.image.width() - 1, self.image.height() - 1))
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        point = self._point(event.position().toPoint(), clamp=False)
        if point is None:
            event.ignore()
            return
        self.setFocus()
        self.cursorPositionChanged.emit(point)

        if self.tool == "eyedropper":
            self.colorPicked.emit(self.image.pixelColor(point))
            event.accept()
            return

        if self.tool == "text":
            self._insert_text(point)
            event.accept()
            return

        if self.tool == "fill":
            before = self.image.copy()
            if self._flood_fill(point):
                self._commit_edit(before)
                self.update()
            event.accept()
            return

        self._drawing = True
        self._start = point
        self._last = point
        self._preview_end = point
        self._before_stroke = self.image.copy()

        if self.tool in {"line", "rect", "ellipse"}:
            self.update()
        else:
            painter = QPainter(self.image)
            self._configure_pen(painter)
            painter.drawPoint(point)
            painter.end()
            self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        point = self._point(event.position().toPoint(), clamp=self._drawing)
        if point is None:
            return
        self.cursorPositionChanged.emit(point)
        if not self._drawing:
            return

        if self.tool in {"line", "rect", "ellipse"}:
            self._preview_end = point
        else:
            painter = QPainter(self.image)
            self._configure_pen(painter)
            painter.drawLine(self._last, point)
            painter.end()
            self._last = point
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            event.ignore()
            return
        point = self._point(event.position().toPoint(), clamp=True)
        if point is None:
            self._drawing = False
            self._before_stroke = None
            self._preview_end = None
            return

        if self.tool in {"line", "rect", "ellipse"}:
            painter = QPainter(self.image)
            self._configure_pen(painter)
            self._draw_shape(painter, self._start, point)
            painter.end()

        before = self._before_stroke
        self._drawing = False
        self._preview_end = None
        self._before_stroke = None
        if before is not None:
            self._commit_edit(before)
        self.update()
        event.accept()

    def _point(self, point: QPoint, *, clamp: bool) -> QPoint | None:
        width, height = self.image.width(), self.image.height()
        if clamp:
            return QPoint(
                max(0, min(width - 1, point.x())),
                max(0, min(height - 1, point.y())),
            )
        if 0 <= point.x() < width and 0 <= point.y() < height:
            return point
        return None

    def _configure_pen(self, painter: QPainter) -> None:
        color = QColor("white") if self.tool == "eraser" else self.color
        width = self.brush_size * 2 if self.tool == "eraser" else self.brush_size
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            self.tool in {"brush", "eraser", "line", "rect", "ellipse"},
        )
        pen = QPen(color, width, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

    def _draw_shape(self, painter: QPainter, start: QPoint, end: QPoint) -> None:
        self._configure_pen(painter)
        rect = QRect(start, end).normalized()
        if self.tool == "line":
            painter.drawLine(start, end)
        elif self.tool == "rect":
            painter.drawRect(rect)
        elif self.tool == "ellipse":
            painter.drawEllipse(rect)

    def _flood_fill(self, start: QPoint) -> bool:
        """Fill a contiguous same-color region using a scanline queue."""
        width, height = self.image.width(), self.image.height()
        x0, y0 = start.x(), start.y()
        target = self.image.pixel(x0, y0)
        replacement = self.color.rgba()
        if target == replacement:
            return False

        pixels = self.image
        stack = deque([(x0, y0)])
        while stack:
            x, y = stack.pop()
            if not (0 <= x < width and 0 <= y < height):
                continue
            if pixels.pixel(x, y) != target:
                continue

            left = x
            while left > 0 and pixels.pixel(left - 1, y) == target:
                left -= 1

            x = left
            span_up = False
            span_down = False
            while x < width and pixels.pixel(x, y) == target:
                pixels.setPixel(x, y, replacement)
                if y > 0:
                    same_up = pixels.pixel(x, y - 1) == target
                    if same_up and not span_up:
                        stack.append((x, y - 1))
                    span_up = same_up
                if y < height - 1:
                    same_down = pixels.pixel(x, y + 1) == target
                    if same_down and not span_down:
                        stack.append((x, y + 1))
                    span_down = same_down
                x += 1
        return True

    def _insert_text(self, point: QPoint) -> None:
        from PySide6.QtWidgets import QInputDialog

        text, accepted = QInputDialog.getMultiLineText(
            self, "Insert text", "Text:", ""
        )
        if not accepted or not text:
            return
        size, accepted = QInputDialog.getInt(
            self, "Text size", "Font size:", 18, 6, 200, 1
        )
        if not accepted:
            return
        before = self.image.copy()
        painter = QPainter(self.image)
        painter.setPen(self.color)
        painter.setFont(QFont("Sans Serif", size))
        line_height = painter.fontMetrics().height()
        for index, line in enumerate(text.splitlines() or [text]):
            painter.drawText(point.x(), point.y() + index * line_height, line)
        painter.end()
        self._commit_edit(before)
        self.update()

    def _update_cursor(self) -> None:
        if self.tool == "eyedropper":
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self.tool == "text":
            self.setCursor(Qt.CursorShape.IBeamCursor)
            return
        if self.tool in {"fill", "line", "rect", "ellipse"}:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        diameter = max(8, min(32, self.brush_size + 4))
        cursor_pixmap = QPixmap(diameter, diameter)
        cursor_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(cursor_pixmap)
        color = QColor("white") if self.tool == "eraser" else self.color
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.setBrush(color)
        painter.drawEllipse(2, 2, diameter - 5, diameter - 5)
        painter.end()
        self.setCursor(QCursor(cursor_pixmap))

    def _commit_edit(self, before: QImage) -> None:
        self._undo_stack.append(before.copy())
        if len(self._undo_stack) > MAX_HISTORY:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self.changed.emit()
        self._emit_history_state()

    def _emit_history_state(self) -> None:
        self.historyChanged.emit(bool(self._undo_stack), bool(self._redo_stack))

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self.image.copy())
        self.image = self._undo_stack.pop()
        self.setFixedSize(self.image.size())
        self.update()
        self.changed.emit()
        self._emit_history_state()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self.image.copy())
        self.image = self._redo_stack.pop()
        self.setFixedSize(self.image.size())
        self.update()
        self.changed.emit()
        self._emit_history_state()


class PaintWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("paintWindow")
        self.current_file: str | None = None
        self._dirty = False
        self.setWindowTitle("Paint")
        self.resize(1280, 850)
        self.setMinimumSize(760, 560)

        self.canvas = PaintCanvas()
        self.canvas.changed.connect(self._on_canvas_changed)
        self.canvas.colorPicked.connect(self._set_color)
        self.canvas.cursorPositionChanged.connect(self._show_position)
        self.canvas.historyChanged.connect(self._update_history_actions)

        self._build_actions()
        self._build_ui()
        self._update_dirty_state(False)

    def _build_actions(self) -> None:
        self._tool_actions: dict[str, QAction] = {}
        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
        for key, label in TOOLS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData(key)
            action.triggered.connect(
                lambda checked=False, name=key: self.canvas.set_tool(name)
            )
            self._tool_group.addAction(action)
            self._tool_actions[key] = action
        self._tool_actions["pencil"].setChecked(True)

        self.action_new = self._make_action("New", "Ctrl+N", self._new_canvas)
        self.action_open = self._make_action("Open…", "Ctrl+O", self._open_image)
        self.action_save = self._make_action("Save", "Ctrl+S", self._save_image)
        self.action_save_as = self._make_action("Save As…", "Ctrl+Shift+S", self._save_image_as)
        self.action_undo = self._make_action("Undo", "Ctrl+Z", self.canvas.undo)
        self.action_redo = self._make_action("Redo", "Ctrl+Y", self.canvas.redo)
        self.action_clear = self._make_action("Clear canvas", None, self._clear_canvas)
        self.action_exit = self._make_action("Exit", "Ctrl+Q", self.close)
        self.action_undo.setEnabled(False)
        self.action_redo.setEnabled(False)

    def _make_action(self, text: str, shortcut: str | None, slot) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        self.addAction(action)
        return action

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self.setCentralWidget(central)

        menu_file = self.menuBar().addMenu("&File")
        for action in (self.action_new, self.action_open, self.action_save,
                       self.action_save_as):
            menu_file.addAction(action)
        menu_file.addSeparator()
        menu_file.addAction(self.action_exit)
        menu_edit = self.menuBar().addMenu("&Edit")
        menu_edit.addAction(self.action_undo)
        menu_edit.addAction(self.action_redo)
        menu_edit.addSeparator()
        menu_edit.addAction(self.action_clear)

        tool_bar = QToolBar("Tools", self)
        tool_bar.setMovable(False)
        tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for key, label in TOOLS:
            tool_bar.addAction(self._tool_actions[key])
        layout.addWidget(tool_bar)

        file_bar = QToolBar("File and editing", self)
        file_bar.setMovable(False)
        for action in (self.action_new, self.action_open, self.action_save,
                       self.action_save_as, self.action_undo, self.action_redo,
                       self.action_clear):
            file_bar.addAction(action)
        layout.addWidget(file_bar)

        options = QHBoxLayout()
        palette_box = QGroupBox("Colors")
        palette_layout = QHBoxLayout(palette_box)
        palette_layout.setContentsMargins(6, 4, 6, 4)
        palette_layout.setSpacing(3)
        self.color_buttons: dict[str, QPushButton] = {}
        for value in PALETTE:
            button = QPushButton()
            button.setFixedSize(25, 25)
            button.setToolTip(value.upper())
            button.setStyleSheet(
                f"background-color: {value}; border: 1px solid #666; border-radius: 2px;"
            )
            button.clicked.connect(
                lambda checked=False, color=value: self._set_color(QColor(color))
            )
            self.color_buttons[value] = button
            palette_layout.addWidget(button)

        custom_button = QPushButton("Custom…")
        custom_button.clicked.connect(self._choose_color)
        palette_layout.addWidget(custom_button)
        options.addWidget(palette_box, 0)

        size_box = QGroupBox("Brush")
        size_layout = QHBoxLayout(size_box)
        size_layout.setContentsMargins(8, 4, 8, 4)
        size_layout.addWidget(QLabel("Width:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(3)
        self.size_spin.setSuffix(" px")
        self.size_spin.valueChanged.connect(self.canvas.set_brush_size)
        size_layout.addWidget(self.size_spin)
        options.addWidget(size_box, 0)

        self.current_color_label = QLabel("    ")
        self.current_color_label.setToolTip("Current drawing color")
        self.current_color_label.setFixedSize(30, 25)
        options.addWidget(QLabel("Current:"))
        options.addWidget(self.current_color_label)
        options.addStretch(1)
        layout.addLayout(options)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("canvasScrollArea")
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet(
            "QScrollArea { background: #777; border: 1px solid #555; }"
            "QScrollArea > QWidget > QWidget { background: #777; }"
        )
        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.tool_status = QLabel("Tool: Pencil")
        self.position_status = QLabel("X: 0, Y: 0")
        self.dimensions_status = QLabel("1000 × 700 px")
        self.status.addWidget(self.tool_status)
        self.status.addPermanentWidget(self.position_status)
        self.status.addPermanentWidget(self.dimensions_status)

        self._tool_group.triggered.connect(
            lambda action: self.tool_status.setText(f"Tool: {action.text()}")
        )
        self._set_color(QColor("#000000"))

    def _set_color(self, color: QColor) -> None:
        if not color.isValid():
            return
        self.canvas.set_color(color)
        self.current_color_label.setStyleSheet(
            f"background: {color.name()}; border: 1px solid #444;"
        )
        self.setToolTip(f"Current color: {color.name().upper()}")

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(self.canvas.color, self, "Choose drawing color")
        if color.isValid():
            self._set_color(color)

    def _show_position(self, point: QPoint) -> None:
        self.position_status.setText(f"X: {point.x()}, Y: {point.y()}")

    def _update_history_actions(self, can_undo: bool, can_redo: bool) -> None:
        self.action_undo.setEnabled(can_undo)
        self.action_redo.setEnabled(can_redo)

    def _on_canvas_changed(self) -> None:
        self._update_dirty_state(True)
        self.dimensions_status.setText(
            f"{self.canvas.image.width()} × {self.canvas.image.height()} px"
        )

    def _update_dirty_state(self, dirty: bool) -> None:
        self._dirty = dirty
        name = Path(self.current_file).name if self.current_file else "Untitled"
        self.setWindowTitle(f"{name}{' *' if dirty else ''} — Paint")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Discard your unsaved changes?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _new_canvas(self) -> None:
        if not self._confirm_discard():
            return
        width, ok = QInputDialog.getInt(
            self, "New canvas", "Width (pixels):", 1000, 1, 10000, 1
        )
        if not ok:
            return
        height, ok = QInputDialog.getInt(
            self, "New canvas", "Height (pixels):", 700, 1, 10000, 1
        )
        if not ok:
            return
        if width * height > MAX_IMAGE_PIXELS:
            QMessageBox.warning(
                self, "Canvas too large", "The canvas may contain at most 40 megapixels."
            )
            return
        self.canvas.new_image(width, height)
        self.current_file = None
        self._update_dirty_state(False)
        self.dimensions_status.setText(f"{width} × {height} px")

    def _save_to_path(self, file_path: str) -> bool:
        if not file_path:
            return False
        path = Path(file_path).expanduser()
        if not self.canvas.image.save(str(path)):
            raise OSError(f"Qt could not save the image to: {path}")
        self.current_file = str(path.resolve())
        self._update_dirty_state(False)
        return True

    def _save_image(self) -> None:
        if self.current_file:
            try:
                self._save_to_path(self.current_file)
            except OSError as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._save_image_as()

    def _save_image_as(self) -> None:
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save image",
            self.current_file or "untitled.png",
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;BMP image (*.bmp)",
        )
        if not file_path:
            return
        if not Path(file_path).suffix:
            suffix = ".jpg" if "JPEG" in selected_filter else ".bmp" if "BMP" in selected_filter else ".png"
            file_path += suffix
        try:
            self._save_to_path(file_path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load_from_path(self, file_path: str) -> None:
        path = Path(file_path).expanduser()
        image = QImage(str(path))
        if image.isNull():
            raise ValueError(f"Qt could not open this image: {path}")
        if image.width() * image.height() > MAX_IMAGE_PIXELS:
            raise ValueError("Image is too large to edit safely (maximum 40 megapixels).")
        self.canvas.set_image(image)
        self.current_file = str(path.resolve())
        self._update_dirty_state(False)
        self.dimensions_status.setText(
            f"{image.width()} × {image.height()} px"
        )

    def _open_image(self) -> None:
        if not self._confirm_discard():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff);;All files (*)",
        )
        if not file_path:
            return
        try:
            self._load_from_path(file_path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def _clear_canvas(self) -> None:
        answer = QMessageBox.question(
            self,
            "Clear canvas",
            "Fill the canvas with white?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        before = self.canvas.image.copy()
        self.canvas.image.fill(QColor("white"))
        self.canvas._commit_edit(before)
        self.canvas.update()

    def closeEvent(self, event) -> None:
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Paint")
    app.setStyle("Fusion")
    window = PaintWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
