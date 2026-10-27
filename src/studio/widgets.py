"""Conversation widgets with explicit keyboard and streaming behavior."""

from __future__ import annotations

from pathlib import Path
from html import escape

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor, QTextBlockFormat, QColor, QDesktopServices
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

from src.studio.theme import markdown_css, palette_for_theme


class Composer(QPlainTextEdit):
    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._composing = False
        self.setObjectName("composer")
        self.setPlaceholderText("Message Daena")
        self.setMinimumHeight(56)
        self.setMaximumHeight(168)
        self.setFixedHeight(56)
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
        height = int(self.document().size().height()) + 19
        self.setFixedHeight(max(56, min(168, height)))


class MarkdownBody(QTextBrowser):
    """Size short messages to their contents; long code keeps an internal scroll."""

    def __init__(self, theme="dark"):
        super().__init__()
        self._theme = theme
        self.setFrameShape(QFrame.NoFrame)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._open_link)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.document().setDefaultStyleSheet(markdown_css(theme))
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._apply_reflow)
        self.document().documentLayout().documentSizeChanged.connect(self.reflow)
        self.setMinimumHeight(36)

    def setMarkdown(self, text):
        self._markdown_text = text
        super().setMarkdown(text)
        # Qt's Markdown importer does not apply paragraph CSS consistently.
        # Set document metrics explicitly for live and restored messages alike.
        block = self.document().begin()
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        while block.isValid():
            cursor.setPosition(block.position())
            fmt = block.blockFormat()
            fmt.setLineHeight(140, QTextBlockFormat.ProportionalHeight.value)
            fmt.setTopMargin(12 if fmt.headingLevel() else 3)
            fmt.setBottomMargin(9)
            if fmt.nonBreakableLines():
                fmt.setBackground(QColor(palette_for_theme(self._theme)["code"]))
                fmt.setLeftMargin(12)
                fmt.setRightMargin(12)
                fmt.setTopMargin(5)
                fmt.setBottomMargin(5)
            cursor.setBlockFormat(fmt)
            block = block.next()
        cursor.endEditBlock()
        self.reflow()

    def set_theme(self, theme):
        self._theme = theme
        self.document().setDefaultStyleSheet(markdown_css(theme))
        if hasattr(self, "_markdown_text"):
            self.setMarkdown(self._markdown_text)

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
        # Avoid a nested scrollbar for normal long answers/code blocks.
        height = max(36, min(16000, int(self.document().size().height()) + 20))
        if self.height() != height:
            self.setFixedHeight(height)

    def append_text(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.reflow()


class ToolActivity(QFrame):
    """Collect per-turn receipts for diagnostics without cluttering the chat."""

    def __init__(self):
        super().__init__()
        self.setObjectName("toolActivity")
        self.calls = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(4)
        self.toggle = QLabel()
        self.toggle.setObjectName("activityLink")
        self.toggle.setTextFormat(Qt.RichText)
        self.toggle.setOpenExternalLinks(False)
        self.toggle.setText(
            '<a href="toggle" style="color: #8d8d85; text-decoration: none;">› &nbsp; Tool calls</a>'
        )
        self.toggle.linkActivated.connect(self._toggle)
        outer.addWidget(self.toggle, alignment=Qt.AlignLeft)
        self._expanded = False
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(190)
        self.details.setObjectName("toolReceipts")
        self.details.hide()
        outer.addWidget(self.details)
        self.preview = QLabel()
        self.preview.setObjectName("toolPreview")
        self.preview.setTextFormat(Qt.PlainText)
        self.preview.setWordWrap(True)
        self.preview.hide()
        self.hide()

    def _toggle(self, *args):
        self._expanded = not self._expanded
        self.details.setVisible(self._expanded)
        arrow = "⌄" if self._expanded else "›"
        self.toggle.setText(
            f'<a href="toggle" style="color: #8d8d85; text-decoration: none;">{arrow} &nbsp; {escape(self._label().strip())}</a>'
        )

    def _label(self):
        pending = sum(call["status"] == "running" for call in self.calls)
        failed = sum(call["status"] == "failed" for call in self.calls)
        unfinished = sum(call["status"] == "unfinished" for call in self.calls)
        count = len(self.calls)
        if pending:
            return f"  Working with {count} tool{'s' if count != 1 else ''} · {pending} running"
        names = list(
            dict.fromkeys(call["name"].replace("_", " ") for call in self.calls)
        )
        summary = ", ".join(names[:3])
        if len(names) > 3:
            summary += f" +{len(names) - 3} more"
        return (
            f"Used {summary} · {count} call{'s' if count != 1 else ''}"
            + (f" · {failed} failed" if failed else "")
            + (f" · {unfinished} unfinished" if unfinished else "")
        )

    def add_call(self, name, arguments=None, call_id=None):
        self.calls.append(
            {
                "name": str(name),
                "call_id": call_id,
                "arguments": arguments,
                "status": "running",
                "summary": "",
            }
        )
        self.show()
        self._refresh()

    def finish_call(self, name, success, summary="", call_id=None):
        # The runtime emits results in dispatch order and does not expose a call ID.
        for call in self.calls:
            if call["status"] == "running" and (
                call["call_id"] == call_id
                if call_id
                else (not name or call["name"] == name)
            ):
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
        previews = []
        for index, call in enumerate(self.calls, 1):
            symbol = {
                "running": "…",
                "complete": "✓",
                "failed": "×",
                "unfinished": "!",
            }[call["status"]]
            lines.append(f"{index}. {symbol} {call['name']}")
            args = call["arguments"]
            target = ""
            if isinstance(args, dict):
                target = next(
                    (
                        str(args[key])
                        for key in ("file_path", "path", "query", "url")
                        if args.get(key)
                    ),
                    "",
                )
            previews.append(f"{symbol} {Path(target).name if target else call['name']}")
            if args:
                try:
                    rendered = json.dumps(args, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    rendered = str(args)
                lines.append("   " + rendered[:1200])
            if call["summary"]:
                lines.append("   " + call["summary"][:1200])
        self.details.setPlainText("\n".join(lines))
        visible = previews[-5:]
        self.preview.setText(
            "   ·   ".join(visible)
            + (f"   ·   +{len(previews) - 5} earlier" if len(previews) > 5 else "")
        )
        self.show()
        arrow = "⌄" if self._expanded else "›"
        self.toggle.setText(
            f'<a href="toggle" style="color: #8d8d85; text-decoration: none;">{arrow} &nbsp; {escape(self._label().strip())}</a>'
        )


class MessageCard(QFrame):
    def __init__(self, speaker, text="", *, user=False, theme="dark"):
        super().__init__()
        self.speaker, self.text, self.reasoning = speaker, str(text), ""
        self.phases = []
        self.theme = theme
        self._user = user
        self._status_base = ""
        self._dots = 0
        self.setObjectName("userMessage" if user else "assistantMessage")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            18 if user else 0, 11 if user else 2, 18 if user else 0, 11 if user else 2
        )
        layout.setSpacing(6)
        self.status = QLabel()
        self.status.setObjectName("thinkingDots")
        self.status.hide()
        layout.addWidget(self.status)
        self._dots_timer = QTimer(self)
        self._dots_timer.setInterval(420)
        self._dots_timer.timeout.connect(self._tick_status)
        self.previous_activity = QLabel(self)
        self.previous_activity.setObjectName("activityLink")
        self.previous_activity.setTextFormat(Qt.RichText)
        self.previous_activity.setOpenExternalLinks(False)
        self.previous_activity.setText(
            '<a href="toggle" style="color: #8d8d85; text-decoration: none;">› &nbsp; Previous activity</a>'
        )
        self.previous_activity.hide()
        layout.addWidget(self.previous_activity)
        self._activity_expanded = False
        self.activity_log = QPlainTextEdit(self)
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumHeight(170)
        self.activity_log.document().setMaximumBlockCount(150)
        self.activity_log.hide()
        layout.addWidget(self.activity_log)
        self.previous_activity.linkActivated.connect(self._toggle_activity)
        self.body = MarkdownBody(theme)
        layout.addWidget(self.body)
        self.work = QFrame()
        self.work.setObjectName("workTimeline")
        self.work_layout = QVBoxLayout(self.work)
        self.work_layout.setContentsMargins(14, 4, 0, 8)
        self.work_layout.setSpacing(12)
        layout.removeWidget(self.activity_log)
        self.work_layout.addWidget(self.activity_log)
        layout.insertWidget(layout.indexOf(self.body), self.work)
        self.work.hide()
        self._work_finished = False
        self._tools_open = False
        self.tool_groups = []
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self.copy_button = QToolButton()
        self.copy_button.setObjectName("messageCopy")
        self.copy_button.setText("⧉  Copy")
        self.copy_button.setToolTip("Copy message")
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.clicked.connect(self._copy)
        actions.addWidget(self.copy_button)
        actions.addStretch(1)
        if user:
            self.copy_button.hide()
        else:
            layout.addLayout(actions)
        # Receipts/reasoning remain available to history and diagnostics APIs;
        # they are deliberately not rendered in the conversational surface.
        self.tools = ToolActivity()
        self.tool_groups.append(self.tools)
        self.work_layout.addWidget(self.tools)
        self.reasoning_button = QLabel(self)
        self.reasoning_button.setObjectName("activityLink")
        self.reasoning_button.setTextFormat(Qt.RichText)
        self.reasoning_button.setOpenExternalLinks(False)
        self.reasoning_button.setText(
            '<a href="toggle" style="color: #8d8d85; text-decoration: none;">› &nbsp; Reasoning</a>'
        )
        self.reasoning_button.hide()
        self._reasoning_expanded = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_stream_markdown)
        self.work_layout.insertWidget(0, self.reasoning_button)
        self.reasoning_body = QPlainTextEdit(self)
        self.reasoning_body.setReadOnly(True)
        self.reasoning_body.setMaximumHeight(220)
        self.reasoning_body.hide()
        self.work_layout.insertWidget(1, self.reasoning_body)
        self.reasoning_button.linkActivated.connect(self._toggle_reasoning)
        self.details_button = QToolButton(self)
        self.details_button.setCheckable(True)
        self.details_button.setText("▸ Execution details")
        self.details_button.hide()
        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(180)
        self.details.document().setMaximumBlockCount(300)
        self.details.hide()
        self.details_button.toggled.connect(self.details.setVisible)
        self.set_text(text)

    def archive_phase(self):
        """Keep completed narration in its original position during streaming."""
        self._render_timer.stop()
        if not self.text.strip():
            return
        phase = MarkdownBody(self.theme)
        phase.setMarkdown(self.text)
        self.work_layout.addWidget(phase)
        self.work.show()
        self.previous_activity.show()
        self.phases.append((self.text, phase))
        self._activity_expanded = True
        self._update_activity_label()
        self.set_text("")

    def _copy(self, *args):
        QApplication.clipboard().setText(self.text)
        self.copy_button.setText("✓  Copied")
        QTimer.singleShot(1500, self._reset_copy_label)

    def _reset_copy_label(self):
        try:
            self.copy_button.setText("⧉  Copy")
        except RuntimeError:  # the card was deleted before the timer fired
            pass

    def set_theme(self, theme):
        self.theme = theme
        for _, phase in self.phases:
            phase.set_theme(theme)
        self.body.set_theme(theme)
        if self.text:
            self.body.setMarkdown(self.text)
            self.body.reflow()

    def _tick_status(self):
        self._dots = (self._dots + 1) % 4
        self.status.setText(self._status_base + "." * self._dots)

    def _toggle_activity(self, *args):
        self._activity_expanded = not self._activity_expanded
        self.work.setVisible(self._activity_expanded)
        self.activity_log.setVisible(
            self._activity_expanded and not self.activity_log.document().isEmpty()
        )
        self._update_activity_label()

    def _update_activity_label(self):
        count = sum(len(group.calls) for group in self.tool_groups)
        failed = sum(
            call["status"] == "failed"
            for group in self.tool_groups
            for call in group.calls
        )
        label = "View activity" if self._work_finished else "Working"
        if count:
            label += f" · {count} tool call{'s' if count != 1 else ''}"
        if failed:
            label += f" · {failed} failed"
        arrow = "⌄" if self._activity_expanded else "›"
        self.previous_activity.setText(
            f'<a href="toggle" style="color: #8d8d85; text-decoration: none;">{arrow} &nbsp; {label}</a>'
        )

    def finish_work(self):
        self._render_timer.stop()
        self._render_stream_markdown()
        self._work_finished = True
        for group in self.tool_groups:
            group.finalize()
        self._activity_expanded = False
        self.work.hide()
        self._update_activity_label()

    def add_activity(self, label):
        if not label:
            return
        self.activity_log.appendPlainText(str(label)[:550])
        self.previous_activity.show()

    def _toggle_reasoning(self, *args):
        self._reasoning_expanded = not self._reasoning_expanded
        self.reasoning_body.setVisible(self._reasoning_expanded)
        arrow = "⌄" if self._reasoning_expanded else "›"
        self.reasoning_button.setText(
            f'<a href="toggle" style="color: #8d8d85; text-decoration: none;">{arrow} &nbsp; Reasoning</a>'
        )

    def set_text(self, text, *, markdown=True):
        self._render_timer.stop()
        self.text = str(text)
        if markdown:
            self.body.setMarkdown(self.text)
        else:
            self.body.setPlainText(self.text)
        self.body.setVisible(bool(self.text))
        self.body.reflow()

    def append_stream(self, text):
        self._tools_open = False
        self.text += text
        self.body.show()
        if not self._render_timer.isActive():
            self._render_timer.start(90)

    def _render_stream_markdown(self):
        self.body.setMarkdown(self.text)
        self.body.reflow()

    def append_reasoning(self, text):
        self.reasoning += text
        # UI preview is bounded; canonical history belongs to the runtime.
        self.reasoning = self.reasoning[-100000:]
        self.reasoning_button.show()
        self.previous_activity.show()
        if not self._work_finished:
            self.work.show()
            self._activity_expanded = True
        cursor = self.reasoning_body.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        if self.reasoning_body.document().characterCount() > 110000:
            self.reasoning_body.setPlainText(self.reasoning)

    def set_status(self, text):
        working = str(text).startswith("●")
        self._status_base = str(text).rstrip(".… ") if working else ""
        self._dots = 0
        show_outcome = bool(text) and str(text).lower() not in {"completed", "complete"}
        self.status.setText(str(text) if working or show_outcome else "")
        self.status.setVisible(working or show_outcome)
        if working:
            self._dots_timer.start()
        else:
            self._dots_timer.stop()

    def add_detail(self, text):
        self.details.appendPlainText(text)

    def add_tool_call(self, name, arguments=None, call_id=None):
        self.archive_phase()
        if not self._tools_open:
            if self.tools.calls:
                self.tools = ToolActivity()
                self.tool_groups.append(self.tools)
                self.work_layout.addWidget(self.tools)
            self.work_layout.removeWidget(self.tools)
            self.work_layout.addWidget(self.tools)
            self._tools_open = True
        self.tools.add_call(name, arguments, call_id)
        self.previous_activity.show()
        self._activity_expanded = True
        self.work.show()
        self._update_activity_label()

    def add_tool_result(self, name, success, summary="", call_id=None):
        for group in self.tool_groups:
            if any(
                call["status"] == "running"
                and (
                    call["call_id"] == call_id
                    if call_id
                    else (not name or call["name"] == name)
                )
                for call in group.calls
            ):
                group.finish_call(name, success, summary, call_id)
                break
        self._update_activity_label()


class ConversationView(QScrollArea):
    suggestedPrompt = Signal(str)

    def __init__(self, theme="dark"):
        super().__init__()
        self.theme = theme
        self.setObjectName("conversation")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.container.setObjectName("conversationCanvas")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(24, 12, 24, 32)
        self.layout.setSpacing(26)
        self.welcome = self._make_welcome()
        self.layout.addWidget(self.welcome, 1)
        self.layout.addStretch()
        self.setWidget(self.container)
        self.cards = []
        self.live = None
        self._follow = True
        self.verticalScrollBar().valueChanged.connect(self._track_scroll)
        self.verticalScrollBar().rangeChanged.connect(lambda *_: self.scroll_to_end())

    def _make_welcome(self):
        host = QWidget()
        host.setObjectName("welcomeHost")
        root = QVBoxLayout(host)
        root.setContentsMargins(12, 8, 12, 8)
        root.addStretch(1)
        hero = QWidget()
        self.welcome_hero = hero
        hero.setMaximumWidth(600)
        hero.setMinimumWidth(300)
        panel = QVBoxLayout(hero)
        panel.setSpacing(12)
        symbol = QLabel("✳")
        symbol.setObjectName("welcomeSymbol")
        symbol.setAlignment(Qt.AlignCenter)
        panel.addWidget(symbol)
        title = QLabel("How can I help today?")
        title.setObjectName("welcomeTitle")
        title.setWordWrap(True)
        title.setMinimumHeight(48)
        title.setAlignment(Qt.AlignCenter)
        panel.addWidget(title)
        subtitle = QLabel(
            "Ask Daena to explore, build or fix things in your workspace."
        )
        subtitle.setObjectName("welcomeSub")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        panel.addWidget(subtitle)
        self.welcome_buttons = []
        for heading, prompt in [
            (
                "Explore my codebase  ↗",
                "Explore this workspace and explain its architecture, entry points, and important files.",
            ),
            (
                "Build a new feature  ↗",
                "Help me plan and implement a useful feature in this project. Start by inspecting the codebase.",
            ),
            (
                "Find and fix a bug  ↗",
                "Inspect this project for a high-impact bug, fix it, and run relevant tests.",
            ),
        ]:
            button = QPushButton(heading)
            button.setObjectName("suggestion")
            button.setCursor(Qt.PointingHandCursor)
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
        if hasattr(self, "layout"):
            margin = max(20, (self.viewport().width() - 850) // 2)
            self.layout.setContentsMargins(margin, 20, margin, 36)
        if hasattr(self, "welcome_hero"):
            self.welcome_hero.setFixedWidth(
                max(300, min(600, self.viewport().width() - 76))
            )
            available = self.viewport().height()
            for index, button in enumerate(self.welcome_buttons):
                button.setVisible(
                    index == 0 or available >= (385 if index == 1 else 445)
                )

    def apply_theme(self, theme):
        self.theme = theme
        for card in self.cards:
            card.set_theme(theme)

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
            speaker,
            text,
            user=speaker.startswith("You") or speaker == "Queued",
            theme=self.theme,
        )
        card.setMaximumWidth(640 if card.objectName() == "userMessage" else 780)
        row = QWidget()
        row.setObjectName("messageRow")
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(14)
        if card.objectName() == "userMessage":
            line.addStretch(1)
            line.addWidget(card, 0)
        else:
            avatar = QLabel("✳")
            avatar.setObjectName("avatar")
            avatar.setFixedSize(28, 28)
            avatar.setAlignment(Qt.AlignCenter)
            line.addWidget(avatar, 0, Qt.AlignTop)
            line.addWidget(card, 1)
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
        final_text = str(text or "")
        if self.live is None:
            self.live = self.add_message("Daena", final_text)
        elif final_text:
            # Never erase visible streamed content with an empty final payload.
            # Canonical non-empty final text may replace the live draft.
            self.live.set_text(final_text)
        self.live.set_status(status)
        self.live.finish_work()
        self.scroll_to_end()

    def restore_turn(self, events):
        """Replay the journal in order, using the same cards as live streaming."""
        self.end_turn()
        completed = False
        fallback = None
        for event in events:
            kind = event.get("type")
            if kind == "user_message":
                self.add_message("You", event.get("text", ""))
                continue
            if kind == "studio_result":
                fallback = event.get("text", "")
                continue
            if kind not in {
                "generation_start",
                "generation_retry",
                "content_delta",
                "assistant_message",
                "thinking_delta",
                "tool_call",
                "tool_result",
                "final_response",
            }:
                continue
            if self.live is None:
                self.begin_turn()
            card = self.live
            if kind == "generation_start":
                card.archive_phase()
            elif kind == "generation_retry":
                card.set_text("")
                card.reasoning = ""
                card.reasoning_body.clear()
                card.reasoning_button.hide()
            elif kind == "content_delta":
                card.append_stream(event.get("text", ""))
            elif kind == "assistant_message":
                card.set_text(event.get("text", ""))
            elif kind == "thinking_delta":
                card.append_reasoning(event.get("text", ""))
            elif kind == "tool_call":
                card.add_tool_call(
                    event.get("name", "tool"),
                    event.get("arguments"),
                    event.get("call_id"),
                )
            elif kind == "tool_result":
                card.add_tool_result(
                    event.get("name", ""),
                    bool(event.get("success")),
                    event.get("summary", ""),
                    event.get("call_id"),
                )
            elif kind == "final_response":
                self.finish(
                    event.get("text", ""),
                    event.get("outcome", "Completed").replace("_", " ").title(),
                )
                completed = True
        if not completed and (self.live is not None or fallback is not None):
            self.finish(fallback or "", "Stopped")
        self.end_turn()

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
