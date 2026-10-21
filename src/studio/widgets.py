"""Conversation widgets with explicit keyboard and streaming behavior."""

from __future__ import annotations

from datetime import datetime
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class Composer(QPlainTextEdit):
    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._composing = False
        self.setObjectName("composer")
        self.setPlaceholderText(
            "Ask Daena to build, investigate, or improve something…"
        )
        self.setMinimumHeight(72)
        self.setMaximumHeight(168)
        self.document().documentLayout().documentSizeChanged.connect(self._resize)

    def inputMethodEvent(self, event):
        self._composing = bool(event.preeditString())
        super().inputMethodEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._composing:
                super().keyPressEvent(event)
                return
            if not event.modifiers() & (Qt.ShiftModifier | Qt.AltModifier):
                if not event.isAutoRepeat():
                    self.submitted.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def _resize(self, *args):
        height = (
            int(self.document().size().height() * self.fontMetrics().lineSpacing()) + 24
        )
        self.setFixedHeight(max(72, min(168, height)))


class MarkdownBody(QTextBrowser):
    """Size short messages to their contents; long code keeps an internal scroll."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._open_link)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.document().setDefaultStyleSheet(
            "body {line-height:1.55;} p {margin-top:6px;margin-bottom:12px;}"
            "pre {background:#0d1320;color:#d9e7fa;padding:14px;white-space:pre-wrap;}"
            "code {font-family:monospace;color:#c3b6ff;}"
            "a {color:#a795ff;} h1,h2,h3 {color:#f0f3fa;}"
        )
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._apply_reflow)
        self.document().documentLayout().documentSizeChanged.connect(self.reflow)
        self.setMinimumHeight(36)

    def _open_link(self, url):
        if url.scheme() in {"https", "http"}:
            QDesktopServices.openUrl(url)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reflow()

    def reflow(self, *args):
        if hasattr(self, "_resize_timer") and not self._resize_timer.isActive():
            self._resize_timer.start(0)

    def _apply_reflow(self):
        width = max(80, self.viewport().width())
        if abs(self.document().textWidth() - width) > 1:
            self.document().setTextWidth(width)
        height = max(36, min(780, int(self.document().size().height()) + 16))
        if self.height() != height:
            self.setFixedHeight(height)

    def append_text(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.reflow()


class MessageCard(QFrame):
    def __init__(self, speaker, text="", *, user=False):
        super().__init__()
        self.speaker, self.text, self.reasoning = speaker, str(text), ""
        self.setObjectName("userMessage" if user else "assistantMessage")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(9)
        head = QHBoxLayout()
        avatar = QLabel("Y" if user else "◈")
        avatar.setObjectName("userAvatar" if user else "agentAvatar")
        avatar.setFixedSize(30, 30)
        avatar.setAlignment(Qt.AlignCenter)
        head.addWidget(avatar)
        name = QLabel(speaker)
        name.setObjectName("messageSpeaker")
        head.addWidget(name)
        time = QLabel(datetime.now().strftime("%H:%M"))
        time.setObjectName("quiet")
        head.addWidget(time)
        head.addStretch()
        copy = QToolButton()
        copy.setText("Copy")
        copy.setToolTip("Copy message text")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text))
        head.addWidget(copy)
        layout.addLayout(head)
        self.status = QLabel()
        self.status.setObjectName("messageStatus")
        self.status.hide()
        layout.addWidget(self.status)
        self.body = MarkdownBody()
        layout.addWidget(self.body)
        self.reasoning_button = QToolButton()
        self.reasoning_button.setText("▸ Reasoning")
        self.reasoning_button.setCheckable(True)
        self.reasoning_button.hide()
        self.reasoning_body = QPlainTextEdit()
        self.reasoning_body.setReadOnly(True)
        self.reasoning_body.setMaximumHeight(220)
        self.reasoning_body.hide()
        self.reasoning_button.toggled.connect(self._toggle_reasoning)
        layout.addWidget(self.reasoning_button)
        layout.addWidget(self.reasoning_body)
        self.details_button = QToolButton()
        self.details_button.setCheckable(True)
        self.details_button.setText("▸ Execution details")
        self.details_button.hide()
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(180)
        self.details.document().setMaximumBlockCount(300)
        self.details.hide()
        self.details_button.toggled.connect(self.details.setVisible)
        layout.addWidget(self.details_button)
        layout.addWidget(self.details)
        self.set_text(text)

    def _toggle_reasoning(self, visible):
        self.reasoning_body.setVisible(visible)
        self.reasoning_button.setText(("▾" if visible else "▸") + " Reasoning")

    def set_text(self, text, *, markdown=True):
        self.text = str(text)
        if markdown:
            self.body.setMarkdown(self.text)
        else:
            self.body.setPlainText(self.text)
        self.body.setVisible(bool(self.text))
        self.body.reflow()

    def append_stream(self, text):
        self.text += text
        self.body.show()
        self.body.append_text(text)

    def append_reasoning(self, text):
        self.reasoning += text
        # UI preview is bounded; canonical history belongs to the runtime.
        self.reasoning = self.reasoning[-100000:]
        self.reasoning_button.show()
        cursor = self.reasoning_body.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        if self.reasoning_body.document().characterCount() > 110000:
            self.reasoning_body.setPlainText(self.reasoning)

    def set_status(self, text):
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def add_detail(self, text):
        self.details_button.show()
        self.details.appendPlainText(text)


class ConversationView(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setObjectName("conversation")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.container.setObjectName("conversationCanvas")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(8, 18, 8, 24)
        self.layout.setSpacing(18)
        self.layout.addStretch()
        self.setWidget(self.container)
        self.cards = []
        self.live = None
        self._follow = True
        self.verticalScrollBar().valueChanged.connect(self._track_scroll)

    def _track_scroll(self, value):
        bar = self.verticalScrollBar()
        self._follow = bar.maximum() - value < 70

    def scroll_to_end(self):
        if self._follow:
            QTimer.singleShot(
                0,
                lambda: self.verticalScrollBar().setValue(
                    self.verticalScrollBar().maximum()
                ),
            )

    def add_message(self, speaker, text):
        self._follow = True
        card = MessageCard(
            speaker, text, user=speaker.startswith("You") or speaker == "Queued"
        )
        self.layout.insertWidget(self.layout.count() - 1, card)
        self.cards.append(card)
        self.scroll_to_end()
        return card

    def begin_turn(self):
        self.live = self.add_message("Daena", "")
        self.live.set_status("● Working")

    def append_stream(self, text):
        if self.live is None:
            self.begin_turn()
        self.live.append_stream(text)
        self.scroll_to_end()

    def append_reasoning(self, text):
        if self.live is None:
            self.begin_turn()
        self.live.append_reasoning(text)

    def set_status(self, text):
        if self.live is not None:
            self.live.set_status(text)

    def add_detail(self, text):
        if self.live is not None:
            self.live.add_detail(text)

    def finish(self, text, status="Completed"):
        if self.live is None:
            self.live = self.add_message("Daena", text)
        else:
            self.live.set_text(text)
        self.live.set_status(status)
        self.scroll_to_end()

    def end_turn(self):
        self.live = None

    def clear(self):
        for card in self.cards:
            self.layout.removeWidget(card)
            card.deleteLater()
        self.cards = []
        self.live = None

    def toPlainText(self):
        return "\n\n".join(f"{card.speaker}\n{card.text}" for card in self.cards)
