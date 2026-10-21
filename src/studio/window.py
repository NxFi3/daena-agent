"""Daena Studio desktop shell. All execution uses the same runtime as the CLI."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4, UUID

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QPlainTextEdit,
    QTextBrowser,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QComboBox,
    QFrame,
    QToolButton,
    QDialog,
)

from src import __version__
from src.skills.registry import SkillRegistry
from src.studio.state import StudioStore, write_json
from src.studio.graph import GraphCanvas
from src.studio.settings import SettingsPanel
from src.studio.automation import AutomationPanel
from src.studio.worker import RuntimeWorker
from src.studio.widgets import Composer, ConversationView
from src.studio.controls import ExecutionBar, ToolScopeDialog
from src.studio.theme import STYLE


class SkillsPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        root = QVBoxLayout(self)
        title = QLabel("A library of working knowledge")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(
            QLabel(
                "Skills load on demand. Local instructions never expand tool permissions."
            )
        )
        row = QHBoxLayout()
        self.directories = QLineEdit()
        self.directories.setPlaceholderText(
            "Additional skill directories, separated by ;"
        )
        row.addWidget(self.directories)
        save = QPushButton("Apply directories")
        save.clicked.connect(self.save_directories)
        row.addWidget(save)
        enabled = QPushButton("Apply skill selection")
        enabled.clicked.connect(self.save_selection)
        row.addWidget(enabled)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        root.addLayout(row)
        split = QSplitter()
        self.list = QListWidget()
        self.body = QPlainTextEdit()
        self.body.setReadOnly(True)
        split.addWidget(self.list)
        split.addWidget(self.body)
        split.setSizes([300, 700])
        root.addWidget(split)
        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        root.addWidget(self.warnings)
        self.list.currentItemChanged.connect(self.show_skill)
        self.refresh()

    def refresh(self):
        catalog = deepcopy(self.window.config)
        catalog.setdefault("skills", {}).update(enabled=True, disabled=[])
        self.registry = SkillRegistry(catalog)
        self.registry.set_workspace(str(self.window.workspace))
        self.list.clear()
        for skill in self.registry.list():
            item = QListWidgetItem(skill["name"])
            item.setToolTip(skill["description"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Unchecked
                if skill["name"]
                in self.window.config.get("skills", {}).get("disabled", [])
                else Qt.Checked
            )
            self.list.addItem(item)
        self.directories.setText(
            "; ".join(self.window.config.get("skills", {}).get("directories", []))
        )
        self.warnings.setText("\n".join(self.registry.warnings))
        if self.list.count():
            self.list.setCurrentRow(0)

    def show_skill(self, item, previous=None):
        if item:
            try:
                self.body.setPlainText(
                    self.registry.skills[item.text()].path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                self.window.error(str(exc))

    def save_directories(self):
        if self.window.busy:
            self.window.error("Stop the current run before changing skills")
            return
        config = deepcopy(self.window.config)
        config.setdefault("skills", {})["directories"] = [
            s.strip() for s in self.directories.text().split(";") if s.strip()
        ]
        self.window.apply_config(config)
        self.refresh()

    def save_selection(self):
        if self.window.busy:
            self.window.error("Stop the current run before changing skills")
            return
        config = deepcopy(self.window.config)
        config.setdefault("skills", {})["disabled"] = [
            self.list.item(i).text()
            for i in range(self.list.count())
            if self.list.item(i).checkState() != Qt.Checked
        ]
        self.window.apply_config(config)


class StudioWindow(QMainWindow):
    def __init__(self, workspace, config=None, *, agent_factory=None):
        super().__init__()
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.store = StudioStore(self.workspace)
        self.config = self.store.load_config(config)
        self.agent = None
        self.worker = None
        self.busy = False
        self.session_id = uuid4()
        self.journal = []
        self.children = {}
        self.agent_factory = agent_factory
        self.pending_text = ""
        self.pending_thought = ""
        self._result = None
        self._closing = False
        self.setWindowTitle(f"Daena Studio {__version__}")
        self.resize(1440, 940)
        self.setMinimumSize(1000, 680)
        self.setStyleSheet(STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 8)
        layout.setSpacing(14)
        header = QFrame()
        header.setObjectName("header")
        h = QHBoxLayout(header)
        brand = QLabel("◈ DAENA")
        brand.setObjectName("brand")
        h.addWidget(brand)
        edition = QLabel("STUDIO  /  " + __version__)
        edition.setObjectName("muted")
        h.addWidget(edition)
        h.addStretch()
        self.model_badge = QLabel()
        h.addWidget(self.model_badge)
        self.workspace_edit = QLineEdit(str(self.workspace))
        self.workspace_edit.setReadOnly(True)
        self.workspace_edit.setMinimumWidth(250)
        h.addWidget(self.workspace_edit)
        choose = QPushButton("Workspace…")
        choose.clicked.connect(self.choose_workspace)
        h.addWidget(choose)
        layout.addWidget(header)
        main = QSplitter()
        layout.addWidget(main, 1)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 6, 0)
        new = QPushButton("+ New conversation")
        new.clicked.connect(self.new_session)
        side.addWidget(new)
        self.nav = QListWidget()
        self.nav.setMaximumHeight(260)
        self.nav.addItems(
            [
                "◌  Conversation",
                "◇  Agent graph",
                "⌘  Automations",
                "▤  Skills",
                "⚙  Settings",
            ]
        )
        side.addWidget(self.nav)
        label = QLabel("CONVERSATIONS")
        label.setObjectName("muted")
        side.addWidget(label)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search conversations…")
        self.history_search.textChanged.connect(self.filter_history)
        side.addWidget(self.history_search)
        self.history = QListWidget()
        self.history.itemActivated.connect(self.open_history)
        self.history.itemClicked.connect(self.open_history)
        side.addWidget(self.history)
        export = QPushButton("Export run…")
        export.clicked.connect(self.export_run)
        side.addWidget(export)
        hint = QLabel(
            "Enter to send\nShift+Enter for a new line\nCtrl+L to focus the composer"
        )
        hint.setObjectName("muted")
        side.addWidget(hint)
        main.addWidget(sidebar)
        self.pages = QStackedWidget()
        main.addWidget(self.pages)
        main.setSizes([220, 1150])
        self._build_chat()
        self._build_graph()
        self.automation = AutomationPanel()
        self.automation.runRequested.connect(self.run_automation)
        self.automation.error.connect(self.error)
        self.pages.addWidget(self.automation)
        self.skills = SkillsPanel(self)
        self.pages.addWidget(self.skills)
        self.settings = SettingsPanel(self.config)
        self.settings.applied.connect(self.apply_config)
        self.settings.error.connect(self.error)
        self.pages.addWidget(self.settings)
        self.nav.currentRowChanged.connect(self.page_changed)
        self.nav.setCurrentRow(0)
        footer = QHBoxLayout()
        self.state_label = QLabel("Ready · your workspace, your tools")
        footer.addWidget(self.state_label)
        footer.addStretch()
        self.context_bar = QProgressBar()
        self.context_bar.setFixedWidth(140)
        self.context_bar.setRange(0, 100)
        self.context_bar.setValue(0)
        footer.addWidget(self.context_bar)
        self.context_label = QLabel("Context —")
        footer.addWidget(self.context_label)
        layout.addLayout(footer)
        self.flush_timer = QTimer(self)
        self.flush_timer.setInterval(70)
        self.flush_timer.timeout.connect(self.flush_stream)
        self.flush_timer.start()
        self._update_badge()
        self.refresh_history()
        self.add_message(
            "Daena Studio",
            "Ready when you are. Choose a workspace, check your model, and give Daena a task.\n\nUse Agent graph to inspect delegated work, or Automations to connect independent agent steps.",
        )
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.input.setFocus)
        QShortcut(
            QKeySequence("Ctrl+,"), self, activated=lambda: self.nav.setCurrentRow(4)
        )

    def _build_chat(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 0, 0, 0)
        root.setSpacing(14)
        head = QHBoxLayout()
        title = QLabel("Conversation")
        title.setObjectName("pageTitle")
        head.addWidget(title)
        head.addStretch()
        self.details_toggle = QToolButton()
        self.details_toggle.setText("Run details")
        self.details_toggle.setCheckable(True)
        head.addWidget(self.details_toggle)
        root.addLayout(head)
        self.execution = ExecutionBar()
        self.execution.changed.connect(self.apply_config)
        self.execution.toolsRequested.connect(self.choose_tools)
        root.addWidget(self.execution)
        split = QSplitter(Qt.Horizontal)
        self.transcript = ConversationView()
        split.addWidget(self.transcript)
        self.stream_tabs = QTabWidget()
        self.stream_tabs.setMinimumWidth(260)
        self.stream = QPlainTextEdit()
        self.stream.setReadOnly(True)
        self.thought = QPlainTextEdit()
        self.thought.setReadOnly(True)
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.document().setMaximumBlockCount(1200)
        self.context_details = QPlainTextEdit()
        self.context_details.setReadOnly(True)
        for name, widget in [
            ("Activity", self.activity),
            ("Context", self.context_details),
            ("Raw stream", self.stream),
            ("Reasoning", self.thought),
        ]:
            self.stream_tabs.addTab(widget, name)
        self.stream_tabs.hide()
        self.details_toggle.toggled.connect(self.stream_tabs.setVisible)
        split.addWidget(self.stream_tabs)
        split.setSizes([900, 300])
        root.addWidget(split, 1)
        panel = QFrame()
        panel.setObjectName("composerPanel")
        composer_layout = QVBoxLayout(panel)
        composer_layout.setContentsMargins(10, 8, 10, 10)
        self.input = Composer()
        self.input.submitted.connect(self.send)
        composer_layout.addWidget(self.input)
        row = QHBoxLayout()
        hint = QLabel("Enter to send  ·  Shift+Enter for a new line")
        hint.setObjectName("quiet")
        row.addWidget(hint)
        row.addStretch()
        self.busy_mode = QComboBox()
        self.busy_mode.addItems(["Steer current task", "Queue next message"])
        self.busy_mode.setToolTip("What a new message does while Daena is working")
        row.addWidget(self.busy_mode)
        self.stop_button = QPushButton("■ Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        self.send_button = QPushButton("Send  ↑")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self.send)
        row.addWidget(self.send_button)
        composer_layout.addLayout(row)
        root.addWidget(panel)
        self.queued = []
        self.pages.addWidget(page)

    def choose_tools(self):
        if self.busy:
            return
        dialog = ToolScopeDialog(self.config, self)
        if dialog.exec() == QDialog.Accepted:
            config = deepcopy(self.config)
            config.setdefault("security", {})["allowed_tools"] = dialog.editor.scope()
            self.apply_config(config)

    def _build_graph(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 0, 0, 0)
        row = QHBoxLayout()
        title = QLabel("Agent workspace")
        title.setObjectName("pageTitle")
        row.addWidget(title)
        row.addStretch()
        fit = QPushButton("Fit graph")
        fit.clicked.connect(lambda: self.graph.fit_graph())
        row.addWidget(fit)
        root.addLayout(row)
        root.addWidget(
            QLabel(
                "Wheel to zoom · Drag the background to pan · Drag a node to move it · Space+drag or middle button also pans."
            )
        )
        split = QSplitter()
        self.graph = GraphCanvas()
        toolbar = QHBoxLayout()
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("quiet")
        for label, slot in [
            ("−", self.graph.zoom_out),
            ("+", self.graph.zoom_in),
            ("100%", self.graph.reset_view),
        ]:
            button = QPushButton(label)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch()
        self.graph.zoomChanged.connect(
            lambda value: self.zoom_label.setText(f"{value}%")
        )
        root.addLayout(toolbar)
        split.addWidget(self.graph)
        pane = QWidget()
        p = QVBoxLayout(pane)
        self.agent_title = QLabel("Select an agent")
        p.addWidget(self.agent_title)
        self.agent_detail = QPlainTextEdit()
        self.agent_detail.setReadOnly(True)
        self.agent_summary = QTextBrowser()
        self.agent_summary.setOpenLinks(False)
        self.agent_summary.anchorClicked.connect(self.open_link)
        self.child_events = QPlainTextEdit()
        self.child_events.setReadOnly(True)
        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.addTab(self.agent_summary, "Overview")
        self.inspector_tabs.addTab(self.agent_detail, "Details")
        self.inspector_tabs.addTab(self.child_events, "Events")
        p.addWidget(self.inspector_tabs)
        row = QHBoxLayout()
        cancel = QPushButton("Cancel child")
        cancel.clicked.connect(self.cancel_child)
        row.addWidget(cancel)
        diff = QPushButton("Inspect diff")
        diff.clicked.connect(self.inspect_diff)
        row.addWidget(diff)
        p.addLayout(row)
        review = QHBoxLayout()
        for label, action in [
            ("Request merge", "merge"),
            ("Request discard", "discard"),
        ]:
            button = QPushButton(label)
            button.clicked.connect(
                lambda checked=False, action=action: self.request_integration(action)
            )
            review.addWidget(button)
        p.addLayout(review)
        delegate = QPushButton("Ask Daena to delegate…")
        delegate.clicked.connect(self.request_delegation)
        p.addWidget(delegate)
        split.addWidget(pane)
        split.setSizes([740, 350])
        root.addWidget(split, 1)
        self.graph.nodeSelected.connect(self.inspect_child)
        self.selected_child = None
        self.pages.addWidget(page)

    def _update_badge(self):
        self.execution.populate(self.config)
        llm = self.config.get("llm", {})
        self.model_badge.setText(
            llm.get("provider", "ollama")
            + "  /  "
            + llm.get("provider_config", {}).get("model_name", "default")
        )

    def page_changed(self, index):
        self.pages.setCurrentIndex(index)
        if index in {1, 2}:
            canvas = self.graph if index == 1 else self.automation.canvas
            QTimer.singleShot(0, canvas.fit_graph)

    def error(self, text):
        QMessageBox.warning(self, "Daena Studio", str(text))

    def open_link(self, url):
        if url.scheme() in {"https", "http"}:
            QDesktopServices.openUrl(url)

    def add_message(self, speaker, text):
        self.transcript.add_message(speaker, text)

    def send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        if self.busy:
            if self.worker.blueprint is not None:
                self.error(
                    "Use Stop to cancel the automation before sending a conversation message"
                )
                self.input.setPlainText(text)
                return
            if self.busy_mode.currentIndex() == 1:
                self.queued.append(text)
                self.add_message("Queued", text)
                return
            if self.agent is None:
                self.error("The model is still initializing; try again shortly")
                self.input.setPlainText(text)
                return
            self.agent.steer(text)
            self.add_message("You · steering", text)
            self.record({"type": "user_message", "text": text})
            return
        self.start_worker(text)

    def start_worker(self, prompt, blueprint=None):
        if self.busy:
            return
        self.journal = []
        self.children = {}
        self.selected_child = None
        self.graph.clear_graph()
        self.graph.upsert(
            "parent",
            "Daena",
            "Coordinator · current conversation",
            "running",
            x=0,
            y=170,
        )
        self.stream.clear()
        self.thought.clear()
        self.activity.clear()
        self.pending_text = ""
        self.pending_thought = ""
        self._result = None
        self.last_prompt = prompt
        self.record({"type": "user_message", "text": prompt})
        if blueprint is not None:
            self.record({"type": "blueprint", "graph": blueprint.dump()})
        self.add_message("You" if blueprint is None else "Automation", prompt)
        self.transcript.begin_turn()
        self.execution.setEnabled(False)
        self.busy = True
        self.send_button.setText("Send guidance  ↑")
        self.stop_button.setEnabled(True)
        self.automation.set_running(True)
        self.settings.setEnabled(False)
        config = self.store.runtime_config(self.config)
        self.worker = RuntimeWorker(
            config,
            prompt,
            agent=self.agent if blueprint is None else None,
            session_id=self.session_id,
            blueprint=blueprint,
            parent=self,
        )
        if self.agent_factory is not None and blueprint is None and self.agent is None:
            # Dependency injection is used by offline end-to-end tests.
            self.worker.agent = self.agent_factory(config)
        self.worker.ready.connect(self.agent_ready)
        self.worker.event.connect(self.on_event)
        self.worker.result.connect(self.got_result)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.state_label.setText(
            "Running · Stop remains available during model generation"
        )

    def agent_ready(self, agent):
        self.agent = agent

    def record(self, event):
        # Preserve useful receipts; streaming fragments are coalesced into bounded entries.
        kind = event.get("type")
        if kind == "generation_wait":
            return
        if kind == "subagent_event" and event.get("child_event", {}).get(
            "event", {}
        ).get("type") in {"content_delta", "thinking_delta", "generation_wait"}:
            return
        if kind in {"content_delta", "thinking_delta"}:
            if (
                self.journal
                and self.journal[-1].get("type") == kind
                and len(self.journal[-1].get("text", "")) < 24000
            ):
                self.journal[-1]["text"] += event.get("text", "")
                return
        bounded = deepcopy(event)
        self.journal.append(bounded)
        if len(self.journal) > 5000:
            del self.journal[:100]

    def on_event(self, event):
        kind = event.get("type")
        self.record(event)
        if kind == "generation_start":
            self.flush_stream()
            if self.transcript.live and self.transcript.live.text:
                self.transcript.add_detail(
                    "Model update: " + self.transcript.live.text[:1200]
                )
                self.transcript.live.set_text("")
            self.transcript.set_status("● Generating")
        if kind == "content_delta":
            self.pending_text += event.get("text", "")
        elif kind == "thinking_delta":
            self.pending_thought += event.get("text", "")
        elif kind == "context":
            used, budget = event.get("estimated_tokens", 0), event.get("budget", 1)
            self.context_bar.setValue(min(100, int(100 * used / max(1, budget))))
            self.context_label.setText(f"Context {used:,} / {budget:,}")
            self.context_details.setPlainText(json.dumps(event, indent=2))
        elif kind == "subagent_event":
            self.on_child_event(event["child_event"])
        elif kind in {"automation_event", "automation_node"}:
            self.automation.handle_event(event)
            if kind == "automation_event":
                inner = event["event"]
                if inner.get("type") == "content_delta":
                    self.pending_text += inner.get("text", "")
                elif inner.get("type") == "subagent_event":
                    self.on_child_event(inner["child_event"])
        elif kind == "final_response":
            self.flush_stream()
            self.transcript.finish(
                event.get("text", ""),
                event.get("outcome", "Completed").replace("_", " ").title(),
            )
        elif kind == "assistant_message":
            self.flush_stream()
            if self.transcript.live:
                self.transcript.live.set_text(event.get("text", ""))
        if kind not in {
            "content_delta",
            "thinking_delta",
            "subagent_event",
            "generation_wait",
        }:
            summary = (
                event.get("summary") or event.get("name") or event.get("reason") or ""
            )
            self.activity.appendPlainText(f"{kind}  {summary}")
        if kind == "generation_wait":
            self.state_label.setText(
                f"Model generating · {event.get('elapsed_seconds', 0):.0f}s · Stop to interrupt"
            )
        elif kind == "tool_call":
            self.state_label.setText("Using tool · " + event.get("name", ""))
            self.transcript.set_status("Using " + event.get("name", "tool"))
            self.transcript.add_detail("→ " + event.get("name", "tool"))
        if kind == "tool_result":
            self.transcript.add_detail(
                ("✓ " if event.get("success") else "× ")
                + event.get("summary", event.get("name", ""))
            )

    def on_child_event(self, data):
        ident = data["agent_id"]
        event = data["event"]
        kind = event.get("type")
        child = self.children.setdefault(
            ident,
            {
                "agent_id": ident,
                "name": data.get("name", "Agent"),
                "mode": data.get("mode", ""),
                "events": [],
                "status": "running",
            },
        )
        if kind not in {"content_delta", "thinking_delta", "generation_wait"}:
            child["events"].append(event)
            child["events"] = child["events"][-160:]
        if kind == "spawned":
            child["task"] = event.get("task", "")
        if kind == "finished":
            child.update({k: v for k, v in event.items() if k != "type"})
        if kind == "tool_call":
            child["tool"] = event.get("name", "")
        index = list(self.children).index(ident)
        self.graph.upsert(
            ident,
            child["name"],
            child.get("tool") or child["mode"],
            child["status"],
            x=380,
            y=index * 155,
        )
        self.graph.connect_nodes("parent", ident)
        if kind == "spawned":
            self.graph.fit_graph()
        if self.selected_child == ident:
            self.inspect_child(ident)

    def flush_stream(self):
        for attr, widget in [
            ("pending_text", self.stream),
            ("pending_thought", self.thought),
        ]:
            text = getattr(self, attr)
            if text:
                cursor = widget.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(text)
                widget.setTextCursor(cursor)
                widget.ensureCursorVisible()
                if attr == "pending_text":
                    self.transcript.append_stream(text)
                else:
                    self.transcript.append_reasoning(text)
                setattr(self, attr, "")

    def got_result(self, result):
        self._result = result
        self.flush_stream()
        self.transcript.finish(
            result["text"],
            "Completed"
            if result["metrics"].get("completed")
            else result["metrics"].get("outcome", "Stopped").replace("_", " ").title(),
        )
        self.record({"type": "studio_result", **result})
        metrics = result["metrics"]
        node = self.graph.nodes.get("parent")
        if node:
            node.status = (
                "completed"
                if metrics.get("completed")
                else metrics.get("outcome", "stopped")
            )
            node.update()
        try:
            self.store.save_run(
                self.session_id, self.last_prompt, self.journal, metrics
            )
        except Exception as exc:
            self.activity.appendPlainText("Archive error: " + str(exc))
        self.refresh_history()

    def worker_finished(self):
        self.busy = False
        self.stop_button.setEnabled(False)
        self.send_button.setText("Send  ↑")
        self.execution.setEnabled(True)
        self.transcript.end_turn()
        self.automation.set_running(False)
        self.settings.setEnabled(True)
        metrics = (self._result or {}).get("metrics", {})
        self.state_label.setText(
            ("Completed" if metrics.get("completed") else "Stopped")
            + " · "
            + str(
                metrics.get("stop_reason")
                or metrics.get("reason")
                or f"{metrics.get('tool_successes', 0)} tools · {metrics.get('iterations', 0)} iterations"
            )
        )
        old = self.worker
        self.worker = None
        old.deleteLater()
        if self._closing:
            self.close()
            return
        if self.queued:
            text = self.queued.pop(0)
            QTimer.singleShot(0, lambda: self.start_worker(text))

    def stop(self):
        self.queued.clear()
        if self.worker:
            self.worker.stop.set()
            self.state_label.setText("Stopping · waiting for resource cleanup")

    def inspect_child(self, ident):
        self.selected_child = ident
        if ident == "parent":
            self.agent_title.setText("Daena · coordinator")
            self.agent_detail.setPlainText(
                json.dumps((self._result or {}).get("metrics", {}), indent=2)
            )
            self.agent_summary.setMarkdown(
                "## Coordinator\n\n"
                + getattr(
                    self, "last_prompt", "Select a task to inspect its execution."
                )
                + "\n\n"
                + self.state_label.text()
            )
            return
        child = deepcopy(self.children.get(ident, {}))
        if self.agent and ident in self.agent.loop.tool.subagents.tasks:
            child.update(self.agent.loop.tool.subagents.inspect(ident))
        events = child.pop("events", [])
        self.agent_title.setText(child.get("name", ident))
        self.agent_detail.setPlainText(
            json.dumps(child, ensure_ascii=False, indent=2, default=str)
        )
        changes = "\n".join(
            f"- {item.get('operation', '')} `{item.get('path', '')}`"
            for item in child.get("changes", [])
        )
        metrics = child.get("metrics", {})
        self.agent_summary.setMarkdown(
            f"**{child.get('status', 'running').upper()}** · {child.get('mode', '')}\n\n"
            f"### Assignment\n{child.get('task', '')}\n\n"
            f"### Report\n{child.get('response') or 'No final report yet.'}\n\n"
            + (f"### Error\n{child['error']}\n\n" if child.get("error") else "")
            + f"### File changes · {child.get('integration', 'none')}\n{changes or 'No file changes recorded.'}\n\n"
            + f"Iterations: {metrics.get('iterations', '—')} · Tokens: {metrics.get('tokens', '—')}"
        )
        self.child_events.setPlainText(
            "\n".join(
                json.dumps(e, ensure_ascii=False, default=str) for e in events[-50:]
            )
        )

    def cancel_child(self):
        if self.agent and self.selected_child and self.selected_child != "parent":
            try:
                self.agent.loop.tool.subagents.cancel(self.selected_child)
            except Exception as exc:
                self.error(str(exc))

    def inspect_diff(self):
        if self.agent and self.selected_child and self.selected_child != "parent":
            try:
                self.agent_detail.setPlainText(
                    json.dumps(
                        self.agent.loop.tool.subagents.diff(self.selected_child),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                self.inspector_tabs.setCurrentWidget(self.agent_detail)
            except Exception as exc:
                self.error(str(exc))

    def request_integration(self, action):
        if not self.selected_child or self.selected_child == "parent":
            return
        self.nav.setCurrentRow(0)
        self.input.setPlainText(
            f"Review subagent {self.selected_child} and {action} its pending changes. Report the actual result and any conflicts."
        )
        self.input.setFocus()

    def request_delegation(self):
        self.nav.setCurrentRow(0)
        self.input.setPlainText(
            "Delegate independent parts of the current task to named subagents. Give each a clear scope, inspect and integrate their results, and verify the combined work."
        )
        self.input.setFocus()

    def run_automation(self, graph):
        self.start_worker("Run automation: " + graph.name, blueprint=graph)

    def release_agent(self):
        if self.agent:
            self.agent.close()
            self.agent = None

    def new_session(self):
        if self.busy:
            self.error("Stop the running task before opening another conversation")
            return
        self.release_agent()
        self.session_id = uuid4()
        self.journal = []
        self.children = {}
        self._result = None
        self.selected_child = None
        self.graph.clear_graph()
        self.transcript.clear()
        self.stream.clear()
        self.thought.clear()
        self.activity.clear()
        self.context_details.clear()
        self.context_bar.setValue(0)
        self.context_label.setText("Context —")
        self.nav.setCurrentRow(0)
        self.state_label.setText(
            "New conversation · workspace files and stored plan remain available as background"
        )

    def apply_config(self, config):
        if self.busy:
            self.error("Stop the current run before applying a profile")
            return
        try:
            self.store.save_config(config)
            self.release_agent()
            self.config = deepcopy(config)
            self.settings.populate(config)
            self.skills.refresh()
            self._update_badge()
            self.state_label.setText("Profile saved · applies to the next message")
        except Exception as exc:
            self.error(str(exc))

    def choose_workspace(self):
        if self.busy:
            self.error("Stop the current run before changing workspace")
            return
        path = QFileDialog.getExistingDirectory(
            self, "Choose workspace", str(self.workspace)
        )
        if not path:
            return
        try:
            store = StudioStore(path)
            config = store.load_config(self.config)
            self.new_session()
            self.workspace = Path(path).resolve()
            self.store = store
            self.config = config
            self.workspace_edit.setText(str(self.workspace))
            self.settings.populate(config)
            self.skills.refresh()
            self.refresh_history()
            self._update_badge()
        except Exception as exc:
            self.error(str(exc))

    def refresh_history(self):
        self.history.clear()
        try:
            groups = {}
            for path, run in self.store.runs():
                groups.setdefault(run["session_id"], []).append((path, run))
            for turns in groups.values():
                path, latest = turns[0]
                title = turns[-1][1]["title"]
                item = QListWidgetItem(
                    title[:55] + (f"\n{len(turns)} turns" if len(turns) > 1 else "")
                )
                item.setToolTip(title)
                item.setData(Qt.UserRole, str(path))
                self.history.addItem(item)
            self.filter_history(self.history_search.text())
        except Exception as exc:
            self.activity.appendPlainText("History error: " + str(exc))

    def filter_history(self, text):
        if not hasattr(self, "history"):
            return
        for i in range(self.history.count()):
            item = self.history.item(i)
            item.setHidden(
                text.casefold() not in (item.text() + " " + item.toolTip()).casefold()
            )

    def open_history(self, item):
        if self.busy:
            self.error("Stop the running task before opening another conversation")
            return
        try:
            run = json.loads(Path(item.data(Qt.UserRole)).read_text())
            turns = [
                data
                for _, data in reversed(self.store.runs())
                if data["session_id"] == run["session_id"]
            ]
            self.new_session()
            self.session_id = UUID(run["session_id"])
            self.last_prompt = run["title"]
            self._result = {"metrics": run["metrics"]}
            self.graph.upsert(
                "parent",
                "Daena",
                "Archived run",
                "completed" if run["metrics"].get("completed") else "stopped",
                x=0,
                y=170,
            )
            for turn in turns:
                for event in turn["events"]:
                    kind = event.get("type")
                    if kind == "user_message":
                        self.add_message("You", event["text"])
                    elif kind == "final_response":
                        self.add_message("Daena", event["text"])
                    elif kind == "studio_result" and not any(
                        e.get("type") == "final_response" for e in turn["events"]
                    ):
                        self.add_message("Daena", event["text"])
            for event in run["events"]:
                kind = event.get("type")
                if kind == "subagent_event":
                    self.on_child_event(event["child_event"])
                elif kind == "blueprint":
                    from src.studio.blueprint import Blueprint

                    self.automation.graph = Blueprint.load(event["graph"])
                    self.automation.name.setText(self.automation.graph.name)
                    self.automation.draw()
                elif kind == "automation_node":
                    self.automation.handle_event(event)
            self.journal = run["events"]
            self.state_label.setText(
                f"Conversation restored · {len(turns)} turns · next message continues this session"
            )
        except Exception as exc:
            self.error(str(exc))

    def export_run(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export run", "daena-run.json", "JSON (*.json)"
        )
        if path:
            try:
                write_json(
                    path,
                    {
                        "schema_version": 1,
                        "session_id": str(self.session_id),
                        "events": self.journal,
                        "result": self._result,
                    },
                )
            except Exception as exc:
                self.error(str(exc))

    def closeEvent(self, event):
        if self.busy:
            self._closing = True
            self.stop()
            event.ignore()
            return
        self.release_agent()
        event.accept()
