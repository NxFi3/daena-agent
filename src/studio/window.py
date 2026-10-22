"""Daena Studio desktop shell. All execution uses the same runtime as the CLI."""

from __future__ import annotations

from copy import deepcopy
from html import escape
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
    QInputDialog,
    QMenu,
    QSlider,
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
from src.studio.theme import style_for_theme


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
        self._automation_result = None
        self._automation_run = False
        self.automation_journal = []
        self.automation_blueprint = None
        self._closing = False
        self.setWindowTitle(f"Daena Studio {__version__}")
        self.resize(1460, 960)
        self.setMinimumSize(900, 640)
        self.setStyleSheet(
            style_for_theme(self.config.get("studio", {}).get("theme", "dark"))
        )
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame()
        header.setObjectName("header")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 10, 24, 10)
        h.setSpacing(8)
        self.sidebar_toggle = QToolButton()
        self.sidebar_toggle.setObjectName("sidebarToggle")
        self.sidebar_toggle.setText("☰")
        self.sidebar_toggle.setToolTip("Show or hide the sidebar · Ctrl+B")
        self.sidebar_toggle.clicked.connect(self.toggle_sidebar)
        h.addWidget(self.sidebar_toggle)
        self.model_badge = QPushButton()
        self.model_badge.setObjectName("modelBadge")
        self.model_badge.setToolTip("Switch models and providers")
        self.model_menu = QMenu(self.model_badge)
        self.model_menu.aboutToShow.connect(self._refresh_model_menu)
        self.model_badge.setMenu(self.model_menu)
        h.addWidget(self.model_badge)
        h.addStretch()
        self.model_hint = QLabel()
        self.model_hint.setObjectName("modelHint")
        h.addWidget(self.model_hint)
        self.theme_button = QToolButton()
        self.theme_button.setObjectName("sidebarToggle")
        self.theme_button.setToolTip("Switch between light and dark appearance")
        self.theme_button.clicked.connect(self.toggle_theme)
        h.addWidget(self.theme_button)
        self._refresh_theme_button()
        self.workspace_edit = QLineEdit(str(self.workspace))
        self.workspace_edit.setReadOnly(True)
        self.workspace_edit.setObjectName("workspacePath")
        self.workspace_edit.setToolTip(str(self.workspace))
        choose = QPushButton("Workspace")
        choose.setObjectName("workspaceButton")
        choose.setToolTip(f"Change workspace\n{self.workspace}")
        choose.clicked.connect(self.choose_workspace)
        h.addWidget(choose)
        self.workspace_button = choose
        layout.addWidget(header)
        main = QSplitter(Qt.Horizontal)
        main.setChildrenCollapsible(False)
        layout.addWidget(main, 1)
        sidebar = QFrame()
        self.sidebar = sidebar
        sidebar.setObjectName("sidebar")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 16, 12, 12)
        side.setSpacing(8)
        sidebar_brand = QLabel("Daena")
        sidebar_brand.setObjectName("sidebarBrand")
        side.addWidget(sidebar_brand)
        new = QPushButton("+   New chat")
        new.setObjectName("newChat")
        new.clicked.connect(self.new_session)
        side.addWidget(new)
        self.history_search = QLineEdit()
        self.history_search.setObjectName("historySearch")
        self.history_search.setPlaceholderText("⌕   Search chats    Ctrl+K")
        self.history_search.textChanged.connect(self.filter_history)
        side.addWidget(self.history_search)
        label = QLabel("Chats")
        label.setObjectName("sidebarHeading")
        side.addWidget(label)
        self.history = QListWidget()
        self.history.setObjectName("historyList")
        self.history.itemActivated.connect(self.open_history)
        self.history.itemClicked.connect(self.open_history)
        self.history.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history.customContextMenuRequested.connect(self.history_menu)
        side.addWidget(self.history, 1)
        self.nav = QListWidget()
        self.nav.setObjectName("navigation")
        self.nav.setFixedHeight(180)
        self.nav.addItems(
            [
                "Chat",
                "Agent graph",
                "Automations",
                "Skills",
                "Settings",
            ]
        )
        side.addWidget(self.nav)
        export = QPushButton("Export conversation")
        export.setObjectName("sidebarFooter")
        export.clicked.connect(self.export_run)
        side.addWidget(export)
        main.addWidget(sidebar)
        sidebar.setMinimumWidth(210)
        sidebar.setMaximumWidth(310)
        self.pages = QStackedWidget()
        main.addWidget(self.pages)
        self.main_splitter = main
        main.setSizes([265, 1195])
        self._build_chat()
        self._build_graph()
        self.automation = AutomationPanel()
        self.automation.runRequested.connect(self.run_automation)
        self.automation.stopRequested.connect(self.stop)
        self.automation.error.connect(self.error)
        self.automation.set_draft_path(self.store.root / "automation-draft.json")
        self.pages.addWidget(self.automation)
        self.skills = SkillsPanel(self)
        self.pages.addWidget(self.skills)
        self.settings = SettingsPanel(self.config)
        self.settings.applied.connect(self.apply_config)
        self.settings.error.connect(self.error)
        self.pages.addWidget(self.settings)
        self.nav.currentRowChanged.connect(self.page_changed)
        self.nav.setCurrentRow(0)
        footer_panel = QFrame()
        footer_panel.setObjectName("statusBar")
        footer = QHBoxLayout(footer_panel)
        footer.setContentsMargins(22, 8, 22, 8)
        self.state_label = QLabel("●  Ready to work")
        self.state_label.setObjectName("runtimeStatus")
        footer.addWidget(self.state_label)
        footer.addStretch()
        self.context_bar = QProgressBar()
        self.context_bar.setFixedWidth(90)
        self.context_bar.setRange(0, 100)
        self.context_bar.setValue(0)
        footer.addWidget(self.context_bar)
        self.context_label = QLabel("Context —")
        footer.addWidget(self.context_label)
        # Runtime metrics live in Diagnostics, not in the conversation shell.
        self.diagnostics_layout.addWidget(footer_panel)
        self.flush_timer = QTimer(self)
        self.flush_timer.setInterval(70)
        self.flush_timer.timeout.connect(self.flush_stream)
        self.flush_timer.start()
        self._update_badge()
        self._apply_appearance()
        self.refresh_history()
        if self.history.count():
            self.open_history(self.history.item(0))
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.input.setFocus)
        QShortcut(QKeySequence("Ctrl+B"), self, activated=self.toggle_sidebar)
        QShortcut(QKeySequence("Ctrl+N"), self, activated=self.new_session)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.focus_history)
        QShortcut(
            QKeySequence("Ctrl+,"), self, activated=lambda: self.nav.setCurrentRow(4)
        )

    def _build_chat(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(14, 4, 14, 16)
        root.setSpacing(0)
        self.diagnostics = QDialog(self)
        self.diagnostics.setWindowTitle("Daena · Advanced controls and diagnostics")
        self.diagnostics.resize(940, 650)
        self.diagnostics_layout = QVBoxLayout(self.diagnostics)
        self.diagnostics_layout.setContentsMargins(18, 18, 18, 18)
        self.diagnostics_layout.setSpacing(12)
        diagnostics_title = QLabel("Advanced controls & diagnostics")
        diagnostics_title.setObjectName("sectionTitle")
        self.diagnostics_layout.addWidget(diagnostics_title)
        self.execution = ExecutionBar()
        self.execution.changed.connect(self.apply_config)
        self.execution.toolsRequested.connect(self.choose_tools)
        self.diagnostics_layout.addWidget(self.execution)
        centered = QHBoxLayout()
        centered.setContentsMargins(0, 0, 0, 0)
        centered.addStretch(1)
        column = QWidget()
        column.setObjectName("chatColumn")
        column.setMaximumWidth(960)
        content = QVBoxLayout(column)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(10)
        self.transcript = ConversationView()
        self.transcript.suggestedPrompt.connect(self.use_suggestion)
        content.addWidget(self.transcript, 1)
        self.agents_shortcut = QPushButton("◇  Agents")
        self.agents_shortcut.setObjectName("sidebarFooter")
        self.agents_shortcut.setToolTip("Open the live agent graph")
        self.agents_shortcut.clicked.connect(lambda: self.nav.setCurrentRow(1))
        self.agents_shortcut.hide()
        content.addWidget(self.agents_shortcut)
        self.stream_tabs = QTabWidget()
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
        self.diagnostics_layout.addWidget(self.stream_tabs, 1)
        panel = QFrame()
        panel.setObjectName("composerPanel")
        composer_layout = QVBoxLayout(panel)
        composer_layout.setContentsMargins(14, 10, 12, 10)
        composer_layout.setSpacing(0)
        self.input = Composer()
        self.input.submitted.connect(self.send)
        composer_layout.addWidget(self.input)
        row = QHBoxLayout()
        self.tools_shortcut = QToolButton()
        self.tools_shortcut.setObjectName("composerAccessory")
        self.tools_shortcut.setText("+")
        self.tools_shortcut.setToolTip("Choose tools")
        self.tools_shortcut.clicked.connect(self.choose_tools)
        row.addWidget(self.tools_shortcut)
        reasoning_title = QLabel("Reasoning")
        reasoning_title.setObjectName("composerHint")
        row.addWidget(reasoning_title)
        self.reasoning_slider = QSlider(Qt.Horizontal)
        self.reasoning_slider.setObjectName("reasoningSlider")
        self.reasoning_slider.setRange(0, 4)
        self.reasoning_slider.setSingleStep(1)
        self.reasoning_slider.setPageStep(1)
        self.reasoning_slider.setTickInterval(1)
        self.reasoning_slider.setFixedWidth(150)
        self.reasoning_slider.setToolTip("Auto · Off · Low · Medium · High")
        self.reasoning_slider.valueChanged.connect(self._reasoning_slider_changed)
        row.addWidget(self.reasoning_slider)
        self.reasoning_label = QLabel("Auto")
        self.reasoning_label.setObjectName("composerHint")
        self.reasoning_label.setMinimumWidth(48)
        row.addWidget(self.reasoning_label)
        row.addStretch()
        self.busy_mode = QComboBox()
        self.busy_mode.addItems(["Steer current task", "Queue next message"])
        self.busy_mode.setToolTip("What a new message does while Daena is working")
        row.addWidget(self.busy_mode)
        self.busy_mode.hide()
        self.stop_button = QPushButton("■")
        self.stop_button.setParent(panel)
        self.stop_button.setObjectName("danger")
        self.stop_button.setToolTip("Stop generation")
        self.stop_button.setEnabled(False)
        self.stop_button.hide()
        self.stop_button.clicked.connect(self.stop)
        # Stop is represented by the send button itself; no duplicate control.
        self.send_button = QPushButton("↑")
        self.send_button.setObjectName("sendMessage")
        self.send_button.setToolTip("Send message (Enter)")
        self.send_button.setFixedSize(34, 34)
        self.send_button.clicked.connect(self.send)
        row.addWidget(self.send_button)
        composer_layout.addLayout(row)
        content.addWidget(panel)
        note = QLabel("Daena can make mistakes. Check important changes.")
        note.setObjectName("composerNote")
        note.setAlignment(Qt.AlignCenter)
        content.addWidget(note)
        centered.addWidget(column, 8)
        centered.addStretch(1)
        root.addLayout(centered, 1)
        self.queued = []
        self.pages.addWidget(page)

    def _refresh_model_menu(self):
        menu = self.model_menu
        menu.clear()
        llm = self.config.get("llm", {})
        provider = llm.get("provider", "ollama")
        current = llm.get("provider_config", {}).get("model_name", "")
        title = menu.addAction(current or "Current model")
        title.setEnabled(False)
        menu.addSeparator()
        for model in (
            self.config.get("studio", {}).get("recent_models", {}).get(provider, [])
        ):
            if model == current:
                continue
            action = menu.addAction(model)
            action.setEnabled(not self.busy)
            action.triggered.connect(
                lambda checked=False, model=model: self._select_model(model)
            )
        manual = menu.addAction("Choose another model…")
        manual.setEnabled(not self.busy)
        manual.triggered.connect(self._prompt_model)
        providers = menu.addMenu("Provider")
        for name in ("ollama", "openrouter", "gemini"):
            action = providers.addAction(name.title())
            action.setCheckable(True)
            action.setChecked(name == provider)
            action.setEnabled(not self.busy)
            action.triggered.connect(
                lambda checked=False, name=name: self.execution.provider.setCurrentText(
                    name
                )
            )
        menu.addSeparator()
        menu.addAction("Advanced controls & diagnostics…", self.open_diagnostics)

    def _select_model(self, model):
        if not self.busy:
            self.execution.model.setCurrentText(model)
            self.execution.commit()

    def _reasoning_slider_changed(self, value):
        if self.busy:
            return
        mode = ("auto", "off", "low", "medium", "high")[value]
        self.execution.reasoning.setCurrentIndex(
            self.execution.reasoning.findData(mode)
        )

    def _prompt_model(self):
        if self.busy:
            return
        model, accepted = QInputDialog.getText(
            self,
            "Choose model",
            "Model name or provider model ID:",
            text=self.config.get("llm", {})
            .get("provider_config", {})
            .get("model_name", ""),
        )
        if accepted and model.strip():
            self._select_model(model.strip())

    def open_diagnostics(self):
        self.diagnostics.show()
        self.diagnostics.raise_()
        self.diagnostics.activateWindow()

    def toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def focus_history(self):
        self.sidebar.show()
        self.history_search.setFocus()
        self.history_search.selectAll()

    def _refresh_theme_button(self):
        self.theme_button.setText(
            "☀" if self.config.get("studio", {}).get("theme") == "light" else "☾"
        )

    def _apply_appearance(self):
        theme = self.config.get("studio", {}).get("theme", "dark")
        self.setStyleSheet(style_for_theme(theme))
        self.graph.apply_theme(theme)
        self.automation.canvas.apply_theme(theme)
        self._refresh_theme_button()

    def toggle_theme(self):
        config = deepcopy(self.config)
        current = config.setdefault("studio", {}).get("theme", "dark")
        config["studio"]["theme"] = "light" if current == "dark" else "dark"
        try:
            self.store.save_config(config)
        except Exception as exc:
            self.error(str(exc))
            return
        self.config = config
        self._apply_appearance()

    def use_suggestion(self, prompt):
        self.nav.setCurrentRow(0)
        self.input.setPlainText(prompt)
        self.input.setFocus()

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
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(10)
        row = QHBoxLayout()
        title = QLabel("Agent workspace")
        title.setObjectName("pageTitle")
        row.addWidget(title)
        row.addStretch()
        fit = QPushButton("Fit graph")
        fit.clicked.connect(lambda: self.graph.fit_graph())
        row.addWidget(fit)
        root.addLayout(row)
        graph_help = QLabel(
            "Live agent activity. Select a node to inspect its work; drag the canvas to pan and scroll to zoom."
        )
        graph_help.setObjectName("muted")
        graph_help.setWordWrap(True)
        root.addWidget(graph_help)
        split = QSplitter()
        self.graph = GraphCanvas()
        toolbar = QHBoxLayout()
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("quiet")
        for label, slot in [
            ("−", self.graph.zoom_out),
            ("+", self.graph.zoom_in),
            ("Reset zoom", self.graph.reset_view),
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
        self.inspector_tabs.addTab(self.agent_detail, "Raw data")
        self.inspector_tabs.addTab(self.child_events, "Activity")
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
        provider = llm.get("provider", "ollama")
        model = llm.get("provider_config", {}).get("model_name", "default")
        self.model_badge.setText("Daena")
        self.model_hint.setText(model)
        self.model_hint.setToolTip(f"{provider} · {model}")
        reasoning = self.execution.reasoning.currentData()
        slider_modes = ("auto", "off", "low", "medium", "high")
        self.reasoning_slider.blockSignals(True)
        self.reasoning_slider.setValue(
            slider_modes.index(reasoning) if reasoning in slider_modes else 0
        )
        self.reasoning_slider.blockSignals(False)
        self.reasoning_label.setText(reasoning.title() if reasoning else "Auto")

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
        return self.transcript.add_message(speaker, text)

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
        self._automation_run = blueprint is not None
        if self._automation_run:
            self.automation_journal = []
            self.automation_blueprint = deepcopy(blueprint)
            self.automation.reset_run()
            self._automation_result = None
        else:
            self.journal = []
            self.children = {}
            self.agents_shortcut.hide()
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
            self._result = None
        self.pending_text = ""
        self.pending_thought = ""
        if not self._automation_run:
            self.last_prompt = prompt
        self.record({"type": "user_message", "text": prompt})
        if blueprint is not None:
            self.record({"type": "blueprint", "graph": blueprint.dump()})
        else:
            self.add_message("You", prompt)
            self.transcript.begin_turn()
            self.transcript.set_status("● Thinking")
        self.execution.setEnabled(False)
        self.reasoning_slider.setEnabled(False)
        self.busy = True
        self.send_button.setText("■" if not self._automation_run else "↑")
        self.send_button.setToolTip("Stop current task" if not self._automation_run else "Send message")
        try:
            self.send_button.clicked.disconnect(self.send)
        except (RuntimeError, TypeError):
            pass
        if not self._automation_run:
            self.send_button.clicked.connect(self.stop)
        else:
            self.send_button.clicked.connect(self.send)
        self.stop_button.setEnabled(True)
        if not self._automation_run:
            self.busy_mode.show()
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
        journal = self.automation_journal if self._automation_run else self.journal
        if kind in {"content_delta", "thinking_delta"}:
            if (
                journal
                and journal[-1].get("type") == kind
                and len(journal[-1].get("text", "")) < 24000
            ):
                journal[-1]["text"] += event.get("text", "")
                return
        bounded = deepcopy(event)
        journal.append(bounded)
        if len(journal) > 5000:
            del journal[:100]

    def on_event(self, event):
        kind = event.get("type")
        self.record(event)
        if kind == "generation_start":
            self.flush_stream()
            if self.transcript.live and self.transcript.live.text:
                # Preserve intermediate agent narration instead of silently erasing it
                # when a new model/tool phase begins. Keep the final answer surface
                # clean by folding earlier narration into Previous activity.
                previous = self.transcript.live.text.strip()
                self.transcript.add_detail("Model update: " + previous[:1200])
                self.transcript.live.add_activity("Model update\n" + previous)
                self.transcript.live.set_text("")
            if not self._automation_run:
                self.transcript.set_status("● Generating")
        if kind == "generation_retry" and not self._automation_run:
            # The provider explicitly discarded the previous generation, so no
            # partial prose/reasoning from that attempt should look authoritative.
            self.pending_text = ""
            self.pending_thought = ""
            if self.transcript.live:
                self.transcript.live.set_text("")
                self.transcript.live.reasoning = ""
                self.transcript.live.reasoning_body.clear()
                self.transcript.live.reasoning_button.hide()
            attempt = event.get("attempt", 1)
            maximum = event.get("max_attempts", 1)
            self.transcript.set_status(f"● Retrying smaller response · {attempt}/{maximum}")
        if kind == "content_delta" and not self._automation_run:
            self.transcript.set_status("● Generating")
        elif kind == "thinking_delta" and not self._automation_run:
            self.transcript.set_status("● Thinking")
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
            if not self._automation_run and self.transcript.live:
                child_event = event["child_event"]
                stage = child_event.get("event", {}).get("type", "")
                if stage in {"spawned", "finished"}:
                    self.transcript.live.add_activity(
                        "◇  " + child_event.get("name", "Subagent") + " · " + stage
                    )
            self.on_child_event(event["child_event"])
        elif kind in {"automation_event", "automation_node"}:
            self.automation.handle_event(event)
            if kind == "automation_event":
                inner = event["event"]
                if inner.get("type") == "content_delta":
                    self.automation.append_stream(
                        event["node_id"], inner.get("text", "")
                    )
                elif inner.get("type") == "subagent_event" and not self._automation_run:
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
                message = event.get("text", "")
                if message:
                    if self.transcript.live.text and self.transcript.live.text != message:
                        self.transcript.live.add_activity(
                            "Model update\n" + self.transcript.live.text.strip()
                        )
                    self.transcript.live.set_text(message)
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
            if not self._automation_run:
                self.transcript.set_status("● Waiting for model")
            self.state_label.setText(
                f"Model generating · {event.get('elapsed_seconds', 0):.0f}s · Stop to interrupt"
            )
        elif kind == "tool_call":
            self.state_label.setText("Using tool · " + event.get("name", ""))
            if not self._automation_run:
                self.transcript.set_status("● Working · " + event.get("name", "tool"))
            self.transcript.add_detail("→ " + event.get("name", "tool"))
            if self.transcript.live:
                self.transcript.live.add_tool_call(
                    event.get("name", "tool"),
                    event.get("arguments"),
                    event.get("call_id"),
                )
        if kind == "tool_result" and not self._automation_run:
            self.transcript.set_status("● Thinking")
        if kind == "tool_result":
            self.transcript.add_detail(
                ("✓ " if event.get("success") else "× ")
                + event.get("summary", event.get("name", ""))
            )
            if self.transcript.live:
                self.transcript.live.add_tool_result(
                    event.get("name", ""),
                    bool(event.get("success")),
                    event.get("summary", ""),
                    event.get("call_id"),
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
        active = sum(child.get("status", "running") == "running" for child in self.children.values())
        self.agents_shortcut.setText(f"◇  {len(self.children)} Agents · {active} working   ›")
        self.agents_shortcut.setVisible(bool(self.children))
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
                if self._automation_run:
                    pass
                elif attr == "pending_text":
                    self.transcript.append_stream(text)
                else:
                    self.transcript.append_reasoning(text)
                setattr(self, attr, "")

    def got_result(self, result):
        if self._automation_run:
            self._automation_result = result
            self.record({"type": "studio_result", **result})
            summary = (
                "Automation completed"
                if result["metrics"].get("completed")
                else "Automation stopped: "
                + str(
                    result["metrics"].get("reason")
                    or result["metrics"].get("stop_reason", "")
                )
            )
            self.automation.run_state.setText(summary)
            try:
                self.store.save_automation_run(
                    self.automation_blueprint.name,
                    self.automation_blueprint.dump(),
                    self.automation_journal,
                    result["metrics"],
                )
            except Exception as exc:
                self.activity.appendPlainText("Automation archive error: " + str(exc))
            return
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
        self.send_button.setText("↑")
        self.send_button.setEnabled(True)
        self.send_button.setToolTip("Send message (Enter)")
        try:
            self.send_button.clicked.disconnect(self.stop)
        except (RuntimeError, TypeError):
            pass
        try:
            self.send_button.clicked.disconnect(self.send)
        except (RuntimeError, TypeError):
            pass
        self.send_button.clicked.connect(self.send)
        self.busy_mode.hide()
        self.execution.setEnabled(True)
        self.reasoning_slider.setEnabled(True)
        if not self._automation_run:
            self.transcript.end_turn()
        self.automation.set_running(False)
        self.settings.setEnabled(True)
        metrics = (
            self._automation_result if self._automation_run else self._result
        ) or {}
        metrics = metrics.get("metrics", {})
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
        self._automation_run = False
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
            self.send_button.setEnabled(False)
            self.send_button.setToolTip("Stopping…")
            self.worker.stop.set()
            self.state_label.setText("Stopping · waiting for resource cleanup")
            if self._automation_run:
                self.automation.run_state.setText("Stopping automation…")

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
            self.child_events.setPlainText("Select a subagent to see its activity.")
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
        status = str(child.get("status", "running")).replace("_", " ").title()
        mode = str(child.get("mode", "Agent"))
        task = str(child.get("task") or "No assignment recorded.")
        report = str(
            child.get("response") or "The agent has not returned a final report."
        )
        error = str(child.get("error") or "")
        integration = str(child.get("integration") or "none")
        def block(value):
            return escape(str(value)).replace("\n", "<br>")
        self.agent_summary.setHtml(
            "<div style='line-height:1.55'>"
            "<h2>" + block(child.get("name", ident)) + "</h2>"
            "<p><b>● " + block(status) + "</b> &nbsp; · &nbsp; " + block(mode) + "</p>"
            "<h3>Assignment</h3><p>" + block(task) + "</p>"
            "<h3>Result</h3><p>"
            + block(report)
            + "</p>"
            + ("<h3>Error</h3><p>" + block(error) + "</p>" if error else "")
            + "<h3>Files &amp; integration</h3><p>"
            + block(changes or "No file changes recorded.")
            + "</p>"
            + "<p>Integration: "
            + block(integration)
            + "</p>"
            + "<p>Iterations: "
            + block(metrics.get("iterations", "—"))
            + " &nbsp; · &nbsp; Tokens: "
            + block(metrics.get("tokens", "—"))
            + "</p>"
            + "</div>"
        )
        # Human-readable event timeline. Full JSON remains available in Raw data.
        event_rows = []
        for item in events[-80:]:
            event_type = item.get("type", "update")
            names = {
                "spawned": "Started",
                "finished": "Finished",
                "tool_call": "Using a tool",
                "tool_result": "Tool finished",
                "context": "Preparing context",
                "generation_start": "Generating",
                "generation_done": "Generated response",
                "error": "Error",
            }
            title = names.get(event_type, event_type.replace("_", " ").title())
            description = (
                item.get("summary")
                or item.get("name")
                or item.get("task")
                or item.get("response")
                or item.get("error")
                or ""
            )
            if not isinstance(description, str):
                description = str(description)
            event_rows.append(
                f"• {title}" + (f" — {description[:350]}" if description else "")
            )
        self.child_events.setPlainText(
            "\n\n".join(event_rows) if event_rows else "No activity recorded yet."
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
        self.agents_shortcut.hide()
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
            self._apply_appearance()
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
            if not self.automation.flush_draft():
                return
            store = StudioStore(path)
            config = store.load_config(self.config)
            self.new_session()
            self.workspace = Path(path).resolve()
            self.store = store
            self.config = config
            self.workspace_edit.setText(str(self.workspace))
            self.workspace_edit.setToolTip(str(self.workspace))
            self.workspace_button.setToolTip(f"Change workspace\n{self.workspace}")
            self.settings.populate(config)
            self.skills.refresh()
            self.automation.set_draft_path(self.store.root / "automation-draft.json")
            self.refresh_history()
            self._update_badge()
            self._apply_appearance()
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

    def history_menu(self, point):
        item = self.history.itemAt(point)
        if item is None or self.busy:
            return
        menu = QMenu(self)
        open_action = menu.addAction("Open chat")
        rename_action = menu.addAction("Rename chat…")
        menu.addSeparator()
        delete_action = menu.addAction("Delete chat")
        action = menu.exec(self.history.mapToGlobal(point))
        if action == open_action:
            self.open_history(item)
        elif action == rename_action:
            self.rename_history(item)
        elif action == delete_action:
            self.delete_history(item)

    def delete_history(self, item):
        if self.busy:
            self.error("Stop the current run before deleting a conversation")
            return
        try:
            selected = json.loads(Path(item.data(Qt.UserRole)).read_text())
            session_id = selected["session_id"]
            answer = QMessageBox.question(
                self,
                "Delete conversation",
                "Delete this conversation and all of its turns?\n\nThis cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            active = str(self.session_id) == str(session_id)
            deleted = self.store.delete_session(session_id)
            self.refresh_history()
            if active:
                self.new_session()
            self.state_label.setText(
                f"Deleted conversation · {deleted} archived turn{'s' if deleted != 1 else ''} removed"
            )
        except Exception as exc:
            self.error(str(exc))

    def rename_history(self, item):
        if self.busy:
            self.error("Stop the current run before renaming a conversation")
            return
        try:
            selected = json.loads(Path(item.data(Qt.UserRole)).read_text())
            title, accepted = QInputDialog.getText(
                self, "Rename conversation", "Conversation title:", text=item.toolTip()
            )
            title = title.strip()[:160]
            if not accepted or not title:
                return
            for path, run in self.store.runs():
                if run["session_id"] == selected["session_id"]:
                    run["title"] = title
                    write_json(path, run)
            self.refresh_history()
        except Exception as exc:
            self.error(str(exc))

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
            # Restore the project that originally owned this conversation.
            original_workspace = run.get("workspace")
            if (
                original_workspace
                and Path(original_workspace).is_dir()
                and Path(original_workspace).resolve() != self.workspace
            ):
                self.workspace = Path(original_workspace).resolve()
                self.store = StudioStore(self.workspace)
                self.workspace_edit.setText(str(self.workspace))
                self.workspace_edit.setToolTip(str(self.workspace))
                self.workspace_button.setToolTip(f"Change workspace\n{self.workspace}")
                self.automation.set_draft_path(
                    self.store.root / "automation-draft.json"
                )
                self.skills.refresh()
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
                receipts = []
                reasoning = []
                for event in turn["events"]:
                    kind = event.get("type")
                    if kind == "user_message":
                        self.add_message("You", event["text"])
                    elif kind == "tool_call":
                        receipts.append(("call", event))
                    elif kind == "tool_result":
                        receipts.append(("result", event))
                    elif kind == "thinking_delta":
                        reasoning.append(event.get("text", ""))
                    elif kind == "final_response":
                        card = self.add_message("Daena", event["text"])
                        self._restore_receipts(card, receipts, reasoning)
                    elif kind == "studio_result" and not any(
                        e.get("type") == "final_response" for e in turn["events"]
                    ):
                        card = self.add_message("Daena", event["text"])
                        self._restore_receipts(card, receipts, reasoning)
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

    @staticmethod
    def _restore_receipts(card, receipts, reasoning):
        for action, event in receipts:
            if action == "call":
                card.add_tool_call(event.get("name", "tool"), event.get("arguments"))
            else:
                card.add_tool_result(
                    event.get("name", ""),
                    bool(event.get("success")),
                    event.get("summary", ""),
                )
        if reasoning:
            card.append_reasoning("".join(reasoning))

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
        self.automation.flush_draft(silent=True)
        event.accept()
