"""Read-only desktop trace viewer. It never sends input or starts model calls."""

import json
from pathlib import Path
import sys
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class Canvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(Qt.GlobalColor.black)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if 0.05 <= self.transform().m11() * factor <= 12:
            self.scale(factor, factor)
        event.accept()

    def fit(self):
        if not self.scene().itemsBoundingRect().isEmpty():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def show_image(self, path):
        self.scene().clear()
        if path and Path(path).is_file():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self.scene().addPixmap(pixmap)
                self.scene().setSceneRect(pixmap.rect())
                self.fit()
                return
        text = self.scene().addText(
            "No retained screenshot for this event.\nExecution receipts remain available below."
        )
        text.setDefaultTextColor(QColor("#a9b4ca"))
        self.scene().setSceneRect(text.boundingRect().adjusted(-80, -80, 80, 80))
        self.fit()


class Viewer(QMainWindow):
    def __init__(self, workspace):
        super().__init__()
        self.root = Path(workspace).expanduser().resolve() / ".daena" / "computer_use"
        self.setWindowTitle("Daena · Computer Use Observatory")
        self.resize(1360, 900)
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#0c101a; color:#e9edf7; font-family:'DejaVu Sans'; font-size:13px; }
            QLabel#title { font-size:24px; font-weight:700; }
            QLabel#muted { color:#99a8c0; }
            QComboBox, QPushButton { background:#1b2435; border:1px solid #34435e; border-radius:7px; padding:9px 14px; }
            QPushButton:hover { background:#303852; }
            QListWidget { background:#121b2a; border:1px solid #27344b; border-radius:9px; padding:8px; }
            QListWidget::item { padding:14px; border-radius:6px; margin:3px; }
            QListWidget::item:selected { background:#423571; }
            QSplitter::handle { background:#253249; width:2px; }
        """)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 20, 22, 18)
        header = QHBoxLayout()
        title = QLabel("◈  DAENA  /  COMPUTER USE")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.sessions = QComboBox()
        self.sessions.setMinimumWidth(280)
        self.sessions.currentTextChanged.connect(self.refresh_events)
        header.addWidget(self.sessions)
        layout.addLayout(header)
        subtitle = QLabel("OBSERVATORY   ·   Local execution receipts   ·   Read-only")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        split = QSplitter()
        self.timeline = QListWidget()
        self.timeline.currentItemChanged.connect(self.inspect)
        split.addWidget(self.timeline)
        right = QWidget()
        panel = QVBoxLayout(right)
        controls = QHBoxLayout()
        self.status = QLabel("Select an observation")
        controls.addWidget(self.status)
        controls.addStretch()
        self.canvas = Canvas()
        for label, fn in (
            ("Fit", self.canvas.fit),
            ("100%", self.canvas.resetTransform),
        ):
            button = QPushButton(label)
            button.clicked.connect(fn)
            controls.addWidget(button)
        panel.addLayout(controls)
        panel.addWidget(self.canvas, 1)
        self.detail = QLabel("Wheel to zoom · drag to pan. Dispatched is not verified.")
        self.detail.setWordWrap(True)
        self.detail.setObjectName("muted")
        panel.addWidget(self.detail)
        split.addWidget(right)
        split.setSizes([330, 990])
        layout.addWidget(split, 1)
        self.setCentralWidget(root)
        self.signature = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    def refresh(self):
        names = sorted(p.parent.name for p in self.root.glob("*/events.jsonl"))
        if names != [self.sessions.itemText(i) for i in range(self.sessions.count())]:
            current = self.sessions.currentText()
            self.sessions.blockSignals(True)
            self.sessions.clear()
            self.sessions.addItems(names)
            if current in names:
                self.sessions.setCurrentText(current)
            self.sessions.blockSignals(False)
        self.refresh_events()

    def refresh_events(self, *_):
        path = self.root / self.sessions.currentText() / "events.jsonl"
        try:
            signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            if signature == self.signature:
                return
            lines = path.read_text(encoding="utf-8").splitlines()[-500:]
        except OSError:
            return
        self.signature = signature
        selected = self.timeline.currentRow()
        following = selected < 0 or selected == self.timeline.count() - 1
        self.timeline.clear()
        for line in lines:
            try:
                event = json.loads(line)
                stamp = datetime.fromtimestamp(event["time"]).strftime("%H:%M:%S")
            except (ValueError, KeyError, TypeError, OverflowError):
                continue
            status = (event.get("verification") or {}).get("status") or event.get(
                "dispatch_status", "observed"
            )
            mark = "✓" if event.get("success") else "!"
            item = QListWidgetItem(
                f"{mark}  {event['action']}    {stamp}\n{status}  ·  {event.get('duration_ms', 0)} ms"
            )
            item.setData(Qt.UserRole, event)
            self.timeline.addItem(item)
        self.timeline.setCurrentRow(
            self.timeline.count() - 1
            if following
            else min(selected, self.timeline.count() - 1)
        )

    def inspect(self, item, _previous=None):
        if item is None:
            return
        event = item.data(Qt.UserRole)
        verification = (event.get("verification") or {}).get("status")
        status = (
            "Condition verified"
            if verification == "verified"
            else "Dispatched · review result"
            if event.get("dispatch_status") == "dispatched"
            else "Observed"
            if event.get("success")
            else "Needs attention"
        )
        self.status.setText(f"{event['action'].upper()}  ·  {status}")
        path = event.get("image_ref")
        if path:
            candidate = self.root / self.sessions.currentText() / Path(path).name
            if not Path(path).is_absolute() or not Path(path).exists():
                path = candidate
        self.canvas.show_image(path)
        self.detail.setText(
            json.dumps(
                {k: v for k, v in event.items() if k not in {"image_ref", "time"}},
                ensure_ascii=False,
                indent=2,
            )
        )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication([])
    window = Viewer(args[0] if args else Path.cwd())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
