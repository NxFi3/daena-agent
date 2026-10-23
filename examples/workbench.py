"""A real desktop fixture for reproducible input tests; no network/model calls."""

import json
from pathlib import Path
import sys

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DrawingArea(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(550, 230)
        self.strokes, self.active = [], False
        self.scroll_total = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.active = True
            self.strokes.append([[event.position().x(), event.position().y()]])
            self.update()

    def mouseMoveEvent(self, event):
        if self.active:
            self.strokes[-1].append([event.position().x(), event.position().y()])
            self.update()

    def mouseReleaseEvent(self, event):
        self.active = False
        self.update()

    def wheelEvent(self, event):
        self.scroll_total += event.angleDelta().y()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101827"))
        painter.setPen(QPen(QColor("#29374f"), 1))
        for x in range(0, self.width(), 24):
            for y in range(0, self.height(), 24):
                painter.drawPoint(x, y)
        painter.setPen(QPen(QColor("#a594ff"), 4, Qt.SolidLine, Qt.RoundCap))
        for stroke in self.strokes:
            for left, right in zip(stroke, stroke[1:]):
                painter.drawLine(QPoint(*map(round, left)), QPoint(*map(round, right)))


class Workbench(QWidget):
    def __init__(self, state_path):
        super().__init__()
        self.path, self.saved = Path(state_path), 0
        self.setWindowTitle("Daena Computer Use · Test Workbench")
        self.resize(820, 650)
        self.setStyleSheet("""
            QWidget {background:#0c101b;color:#eef0fa;font-family:'DejaVu Sans';font-size:14px;}
            QLineEdit {background:#172136;border:1px solid #3b4960;border-radius:8px;padding:14px;}
            QPushButton {background:#a594ff;color:#121125;padding:13px 26px;border-radius:8px;font-weight:700;}
            QLabel#title {font-size:26px;font-weight:700;} QLabel#sub {color:#a3b0c7;}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 26, 30, 26)
        title = QLabel("◈  DAENA  /  INPUT WORKBENCH")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Local desktop fixture · live mouse and keyboard events")
        subtitle.setObjectName("sub")
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        self.text = QLineEdit()
        self.text.setPlaceholderText("Type here — Unicode supported")
        self.text.setAccessibleName("Name")
        layout.addWidget(self.text)
        self.check = QCheckBox("Enable review")
        layout.addWidget(self.check)
        self.canvas = DrawingArea()
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.label = QLabel("Ready · draw a stroke, enter a name, and save")
        row.addWidget(self.label)
        self.save = QPushButton("Save")
        self.save.clicked.connect(self.on_save)
        row.addWidget(self.save)
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.record)
        self.timer.start(40)

    def on_save(self):
        self.saved += 1
        self.label.setText(f"Saved {self.saved} time(s)")
        self.record()

    def record(self):
        def center(widget):
            p = widget.mapTo(self, widget.rect().center())
            return [p.x(), p.y()]

        p = self.canvas.mapTo(self, QPoint(0, 0))
        data = {
            "text": self.text.text(),
            "checked": self.check.isChecked(),
            "saved": self.saved,
            "stroke_count": len(self.canvas.strokes),
            "point_count": sum(map(len, self.canvas.strokes)),
            "mouse_down": self.canvas.active,
            "scroll_total": self.canvas.scroll_total,
            "points": {
                "name": center(self.text),
                "save": center(self.save),
                "check": [
                    self.check.mapTo(self, QPoint(12, self.check.height() // 2)).x(),
                    center(self.check)[1],
                ],
            },
            "canvas": [p.x(), p.y(), self.canvas.width(), self.canvas.height()],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False))
        temporary.replace(self.path)


if __name__ == "__main__":
    app = QApplication([])
    window = Workbench(sys.argv[1])
    window.show()
    raise SystemExit(app.exec())
