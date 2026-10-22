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
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QPushButton,
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
        self.setFixedHeight(72)
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
        # QTextDocument.size() is already measured in pixels. Multiplying it by
        # lineSpacing kept the composer permanently at its maximum height.
        height = int(self.document().size().height()) + 24
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


class ToolActivity(QFrame):
    """Compact, expandable receipts for the tools used in one assistant turn."""

    def __init__(self):
        super().__init__()
        self.setObjectName("toolActivity")
        self.calls = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(4)
        self.toggle = QToolButton()
        self.toggle.setObjectName("toolToggle")
        self.toggle.setCheckable(True)
        self.toggle.setText("▸  Activity")
        self.toggle.toggled.connect(self._toggle)
        outer.addWidget(self.toggle, alignment=Qt.AlignLeft)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(190)
        self.details.setObjectName("toolReceipts")
        self.details.hide()
        outer.addWidget(self.details)
        self.hide()

    def _toggle(self, visible):
        self.details.setVisible(visible)
        self.toggle.setText(("▾" if visible else "▸") + self._label())

    def _label(self):
        pending = sum(call["status"] == "running" for call in self.calls)
        failed = sum(call["status"] == "failed" for call in self.calls)
        unfinished = sum(call["status"] == "unfinished" for call in self.calls)
        count = len(self.calls)
        if pending:
            return f"  Working with {count} tool{'s' if count != 1 else ''} · {pending} running"
        return (
            f"  {count} tool{'s' if count != 1 else ''} used"
            + (f" · {failed} failed" if failed else "")
            + (f" · {unfinished} unfinished" if unfinished else "")
        )

    def add_call(self, name, arguments=None):
        self.calls.append(
            {
                "name": str(name),
                "arguments": arguments,
                "status": "running",
                "summary": "",
            }
        )
        self.show()
        self._refresh()

    def finish_call(self, name, success, summary=""):
        # The runtime emits results in dispatch order and does not expose a call ID.
        for call in self.calls:
            if call["status"] == "running" and (not name or call["name"] == name):
                call["status"] = "complete" if success else "failed"
                call["summary"] = str(summary)
                break
        self._refresh()

    def finalize(self):
        for call in self.calls:
            if call["status"] == "running":
                call["status"] = "unfinished"
                call["summary"] = "No result receipt before the run ended"
        if self.calls:
            self._refresh()

    def _refresh(self):
        import json

        lines = []
        for index, call in enumerate(self.calls, 1):
            symbol = {
                "running": "…",
                "complete": "✓",
                "failed": "×",
                "unfinished": "!",
            }[call["status"]]
            lines.append(f"{index}. {symbol} {call['name']}")
            args = call["arguments"]
            if args:
                try:
                    rendered = json.dumps(args, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    rendered = str(args)
                lines.append("   " + rendered[:1200])
            if call["summary"]:
                lines.append("   " + call["summary"][:1200])
        self.details.setPlainText("\n".join(lines))
        self.toggle.setText(("▾" if self.toggle.isChecked() else "▸") + self._label())


class MessageCard(QFrame):
    def __init__(self, speaker, text="", *, user=False):
        super().__init__()
        self.speaker, self.text, self.reasoning = speaker, str(text), ""
        self.setObjectName("userMessage" if user else "assistantMessage")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20 if user else 8, 14, 20 if user else 8, 14)
        layout.setSpacing(12)
        head = QHBoxLayout()
        avatar = QLabel("Y" if user else "✦")
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
        self.tools = ToolActivity()
        layout.addWidget(self.tools)
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

    def add_tool_call(self, name, arguments=None):
        self.tools.add_call(name, arguments)

    def add_tool_result(self, name, success, summary=""):
        self.tools.finish_call(name, success, summary)


class ConversationView(QScrollArea):
    suggestedPrompt = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("conversation")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.container.setObjectName("conversationCanvas")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(16, 22, 16, 28)
        self.layout.setSpacing(22)
        self.welcome = self._make_welcome()
        self.layout.addWidget(self.welcome, 1)
        self.layout.addStretch()
        self.setWidget(self.container)
        self.cards = []
        self.live = None
        self._follow = True
        self.verticalScrollBar().valueChanged.connect(self._track_scroll)

    def _make_welcome(self):
        host = QWidget()
        host.setObjectName("welcomeHost")
        root = QVBoxLayout(host)
        root.setContentsMargins(12, 8, 12, 8)
        root.addStretch(1)
        hero = QWidget()
        self.welcome_hero = hero
        hero.setMaximumWidth(640)
        hero.setMinimumWidth(300)
        panel = QVBoxLayout(hero)
        panel.setSpacing(12)
        symbol = QLabel("✦")
        symbol.setObjectName("welcomeSymbol")
        panel.addWidget(symbol)
        title = QLabel("What can we build together?")
        title.setObjectName("welcomeTitle")
        title.setWordWrap(True)
        title.setMinimumHeight(48)
        panel.addWidget(title)
        subtitle = QLabel(
            "Code, research, automate, or explore your workspace. Daena can plan and use tools while you stay in control."
        )
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setMinimumHeight(42)
        panel.addWidget(subtitle)
        self.welcome_buttons = []
        for heading, prompt in [
            (
                "⌘   Explore my codebase",
                "Explore this workspace and explain its architecture, entry points, and important files.",
            ),
            (
                "✦   Build a new feature",
                "Help me plan and implement a useful feature in this project. Start by inspecting the codebase.",
            ),
            (
                "◇   Find and fix a bug",
                "Inspect this project for a high-impact bug, fix it, and run relevant tests.",
            ),
        ]:
            button = QPushButton(heading)
            button.setObjectName("suggestion")
            button.setToolTip(prompt)
            button.clicked.connect(
                lambda checked=False, text=prompt: self.suggestedPrompt.emit(text)
            )
            panel.addWidget(button)
            self.welcome_buttons.append(button)
        root.addWidget(hero, 0, Qt.AlignHCenter)
        root.addStretch(1)
        return host

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "welcome_hero"):
            self.welcome_hero.setFixedWidth(
                max(300, min(640, self.viewport().width() - 76))
            )
            available = self.viewport().height()
            for index, button in enumerate(self.welcome_buttons):
                button.setVisible(
                    index == 0 or available >= (385 if index == 1 else 445)
                )

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
        self.welcome.hide()
        card = MessageCard(
            speaker, text, user=speaker.startswith("You") or speaker == "Queued"
        )
        card.setMaximumWidth(720 if card.objectName() == "userMessage" else 860)
        row = QWidget()
        row.setObjectName("messageRow")
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        if card.objectName() == "userMessage":
            line.addStretch(2)
            line.addWidget(card, 7)
            line.addStretch(1)
        else:
            line.addStretch(1)
            line.addWidget(card, 8)
            line.addStretch(1)
        self.layout.insertWidget(self.layout.count() - 1, row)
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
        self.live.tools.finalize()
        self.scroll_to_end()

    def end_turn(self):
        self.live = None

    def clear(self):
        for card in self.cards:
            row = card.parentWidget()
            self.layout.removeWidget(row)
            row.deleteLater()
        self.cards = []
        self.live = None
        self.welcome.show()

    def toPlainText(self):
        return "\n\n".join(f"{card.speaker}\n{card.text}" for card in self.cards)
