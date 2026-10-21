"""Daena Studio desktop shell. All execution uses the same runtime as the CLI."""

from __future__ import annotations

from copy import deepcopy
import html
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
)

from src import __version__
from src.skills.registry import SkillRegistry
from src.studio.state import StudioStore, write_json
from src.studio.graph import GraphCanvas
from src.studio.settings import SettingsPanel
from src.studio.automation import AutomationPanel
from src.studio.worker import RuntimeWorker

STYLE = """
QMainWindow, QWidget {background:#0c1220;color:#dce6fa;font-family:'Inter','Segoe UI',sans-serif;font-size:13px;}
QLabel#brand {font-size:25px;font-weight:800;color:#c7b8ff;letter-spacing:3px;}
QLabel#pageTitle {font-size:27px;font-weight:700;color:#f2f5ff;margin-top:4px;margin-bottom:10px;}
QLabel#muted {color:#8fa1bc;}
QFrame#header {border-bottom:1px solid #26334a;background:#101929;}
QPushButton {background:#1b2940;border:1px solid #30425e;border-radius:8px;padding:9px 15px;font-weight:600;}
QPushButton:hover {background:#2b3b57;border-color:#7a71c9;}
QPushButton:disabled {color:#63718a;border-color:#223047;}
QPushButton#primary {background:#8a6aef;color:#fff;border-color:#a68bfa;}
QPushButton#danger {color:#ff9aac;}
QLineEdit,QPlainTextEdit,QTextBrowser,QListWidget,QComboBox,QSpinBox {background:#111c2e;border:1px solid #293b55;border-radius:9px;padding:8px;selection-background-color:#5e4ea2;}
QPlainTextEdit,QTextBrowser {font-size:14px;}
QListWidget::item {padding:10px 8px;border-radius:7px;}
QListWidget::item:selected {background:#332b56;color:#e4d9ff;}
QTabWidget::pane {border:1px solid #293b55;border-radius:10px;}
QTabBar::tab {padding:10px 17px;background:#111b2c;color:#8fa1bc;}
QTabBar::tab:selected {color:#c7b8ff;border-bottom:2px solid #a78bfa;}
QSplitter::handle {background:#223047;width:2px;height:2px;}
QProgressBar {border:0;border-radius:4px;background:#1b2940;height:7px;text-align:center;}
QProgressBar::chunk {background:#9b80f5;border-radius:4px;}
QScrollBar:vertical {background:#101929;width:9px;}
QScrollBar::handle:vertical {background:#3b4b68;border-radius:4px;min-height:25px;}
QCheckBox {spacing:9px;padding:3px;}
QStatusBar {background:#101929;color:#9baccc;}
QToolTip {background:#24324a;color:#f1f5ff;border:1px solid #8a75cf;padding:7px;}
"""


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
        sidebar = QWidget()
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 6, 0)
        new = QPushButton("＋ New conversation")
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
        label = QLabel("RECENT RUNS")
        label.setObjectName("muted")
        side.addWidget(label)
        self.history = QListWidget()
        self.history.itemActivated.connect(self.open_history)
        side.addWidget(self.history)
        export = QPushButton("Export run…")
        export.clicked.connect(self.export_run)
        side.addWidget(export)
        hint = QLabel("Local workspace\nIndependent agent contexts\nCtrl+Enter to send")
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
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.send)

    def _build_chat(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 0, 0, 0)
        title = QLabel("From an idea to working software")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "Talk, steer, inspect. One shared runtime across terminal and desktop."
        )
        subtitle.setObjectName("muted")
        root.addWidget(subtitle)
        split = QSplitter(Qt.Vertical)
        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.anchorClicked.connect(self.open_link)
        split.addWidget(self.transcript)
        self.stream_tabs = QTabWidget()
        self.stream = QPlainTextEdit()
        self.stream.setReadOnly(True)
        self.stream.setPlaceholderText("Live model response appears here")
        self.thought = QPlainTextEdit()
        self.thought.setReadOnly(True)
        self.thought.setPlaceholderText("Provider reasoning, when supplied")
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.document().setMaximumBlockCount(1200)
        self.context_details = QPlainTextEdit()
        self.context_details.setReadOnly(True)
        for name, widget in [
            ("Response stream", self.stream),
            ("Reasoning", self.thought),
            ("Activity", self.activity),
            ("Context", self.context_details),
        ]:
            self.stream_tabs.addTab(widget, name)
        split.addWidget(self.stream_tabs)
        split.setSizes([430, 180])
        root.addWidget(split, 1)
        self.input = QPlainTextEdit()
        self.input.setMaximumHeight(105)
        self.input.setPlaceholderText(
            "Describe your task…  Ctrl+Enter to send. While busy, messages steer the active task."
        )
        root.addWidget(self.input)
        row = QHBoxLayout()
        self.send_button = QPushButton("Send message")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self.send)
        self.stop_button = QPushButton("■ Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.busy_mode = QComboBox()
        self.busy_mode.addItems(["Steer current task", "Queue next message"])
        row.addWidget(self.send_button)
        row.addWidget(self.stop_button)
        row.addStretch()
        row.addWidget(self.busy_mode)
        root.addLayout(row)
        self.queued = []
        self.pages.addWidget(page)

    def _build_graph(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 0, 0, 0)
        row = QHBoxLayout()
        title = QLabel("A clear view of every agent")
        title.setObjectName("pageTitle")
        row.addWidget(title)
        row.addStretch()
        fit = QPushButton("Fit graph")
        fit.clicked.connect(lambda: self.graph.fit_graph())
        row.addWidget(fit)
        root.addLayout(row)
        root.addWidget(
            QLabel(
                "Select a node for its assignment, live tools, report, errors and file changes. Ctrl+wheel zooms."
            )
        )
        split = QSplitter()
        self.graph = GraphCanvas()
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
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(
            f'<p style="color:#b8a5ff;font-weight:700">{html.escape(speaker)}</p>'
        )
        cursor.insertBlock()
        cursor.insertMarkdown(str(text))
        cursor.insertBlock()
        cursor.insertBlock()
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

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
        self.busy = True
        self.send_button.setText("Send guidance")
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
            self.add_message("Daena", event.get("text", ""))
        elif kind == "assistant_message":
            self.add_message("Daena · progress", event.get("text", ""))
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
                setattr(self, attr, "")

    def got_result(self, result):
        self._result = result
        self.flush_stream()
        if not any(e.get("type") == "final_response" for e in self.journal):
            self.add_message("Daena", result["text"])
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
        self.send_button.setText("Send message")
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
            for path, run in self.store.runs():
                item = QListWidgetItem(run["title"][:70])
                item.setToolTip(run["title"])
                item.setData(Qt.UserRole, str(path))
                self.history.addItem(item)
        except Exception as exc:
            self.activity.appendPlainText("History error: " + str(exc))

    def open_history(self, item):
        if self.busy:
            self.error("Stop the running task before opening history")
            return
        try:
            run = json.loads(Path(item.data(Qt.UserRole)).read_text())
            self.new_session()
            self.session_id = UUID(run["session_id"])
            self._result = {"metrics": run["metrics"]}
            self.graph.upsert(
                "parent",
                "Daena",
                "Archived run",
                "completed" if run["metrics"].get("completed") else "stopped",
                x=0,
                y=170,
            )
            for event in run["events"]:
                if event.get("type") == "user_message":
                    self.add_message("You", event["text"])
                elif event.get("type") == "final_response":
                    self.add_message("Daena", event["text"])
                elif event.get("type") == "subagent_event":
                    self.on_child_event(event["child_event"])
                elif event.get("type") == "blueprint":
                    from src.studio.blueprint import Blueprint

                    self.automation.graph = Blueprint.load(event["graph"])
                    self.automation.name.setText(self.automation.graph.name)
                    self.automation.draw()
                elif event.get("type") == "automation_node":
                    self.automation.handle_event(event)
                elif event.get("type") == "studio_result" and not any(
                    e.get("type") == "final_response" for e in run["events"]
                ):
                    self.add_message("Daena", event["text"])
            self.journal = run["events"]
            self.state_label.setText(
                "History opened · your next message continues this session"
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
