"""Form-based workspace settings; raw JSON remains an optional advanced editor."""

from __future__ import annotations
from copy import deepcopy
import json
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPlainTextEdit,
    QPushButton,
    QLabel,
    QScrollArea,
    QListWidget,
    QStackedWidget,
    QFrame,
)
from src.security.Policy import SecurityPolicy
from src.studio.state import validate_config
from src.studio.controls import ToolScopeEditor
from src.studio.profiles import (
    REASONING,
    reasoning_value,
    set_reasoning,
    switch_profile,
)


class SettingsPanel(QWidget):
    applied = Signal(dict)
    error = Signal(str)
    voice_requested = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        self._syncing = False
        self._runtime_busy = False
        self._voice_widget = None
        root = QVBoxLayout(self)
        root.setSpacing(16)
        title = QLabel("Workspace settings")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        hint = QLabel(
            "Clear controls for the model, memory, delegation and permissions. Saved between runs."
        )
        hint.setObjectName("muted")
        root.addWidget(hint)
        content = QHBoxLayout()
        root.addLayout(content, 1)
        self.sections = QListWidget()
        self.sections.addItems(
            [
                "Model & reasoning",
                "Context & memory",
                "Delegation & skills",
                "Tool permissions",
                "MCP Servers",
                "Advanced profile",
                "Voice",
            ]
        )
        self.sections.setFixedWidth(210)
        content.addWidget(self.sections)
        self.pages = QStackedWidget()
        content.addWidget(self.pages, 1)
        self.sections.currentRowChanged.connect(self.pages.setCurrentIndex)
        self._forms = []

        model, form = self.page(
            "Model & reasoning",
            "Choose the provider and model; adjust reasoning without editing JSON.",
        )
        self.provider = QComboBox()
        self.provider.addItems(["ollama", "openrouter", "gemini"])
        self.model = QLineEdit()
        self.model.setPlaceholderText("Installed model name or provider model ID")
        self.reasoning = QComboBox()
        for label, value in REASONING:
            self.reasoning.addItem(label, value)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 2)
        self.temperature.setSingleStep(0.1)
        self.temperature.setDecimals(2)
        self.native_context = QSpinBox()
        self.native_context.setRange(1024, 2000000)
        self.native_context.setSingleStep(1024)
        self.output_tokens = QSpinBox()
        self.output_tokens.setRange(0, 131072)
        self.output_tokens.setSpecialValueText("Provider default")
        self.output_tokens.setSingleStep(1024)
        self.base = QLineEdit()
        self.base.setPlaceholderText("Default endpoint / OLLAMA_HOST")
        for label, widget in [
            ("Provider", self.provider),
            ("Model", self.model),
            ("Reasoning", self.reasoning),
            ("Temperature", self.temperature),
            ("Model context · Ollama", self.native_context),
            ("Max output tokens", self.output_tokens),
            ("Custom endpoint", self.base),
        ]:
            form.addRow(label, widget)
        note = QLabel(
            "Ollama uses think; OpenRouter uses reasoning effort; Gemini uses thinking budgets. Model support varies; custom native options can be edited below."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        form.addRow(note)
        self.generation = QPlainTextEdit()
        self.generation.setMaximumHeight(170)
        self.generation.setPlaceholderText("Advanced provider options · JSON")
        self.options_toggle = QPushButton("Show advanced provider options")
        self.options_toggle.setCheckable(True)
        self.options_toggle.toggled.connect(self.generation.setVisible)
        self.generation.hide()
        form.addRow(self.options_toggle)
        form.addRow(self.generation)
        credentials = QLabel(
            "Cloud credentials stay in environment variables:\nOPENROUTER_API_KEY · GEMINI_API_KEY / GOOGLE_API_KEY\nApplying settings does not contact a paid model."
        )
        credentials.setWordWrap(True)
        credentials.setObjectName("muted")
        form.addRow(credentials)

        context, f = self.page(
            "Context & memory",
            "Reserve space for output and schemas; keep useful recent evidence.",
        )
        self.context = QSpinBox()
        self.context.setRange(1024, 2000000)
        self.context.setSingleStep(1024)
        self.recent = QSpinBox()
        self.recent.setRange(4, 1000)
        self.reserve = QSpinBox()
        self.reserve.setRange(0, 131072)
        self.reserve.setSingleStep(1024)
        self.image_reserve = QSpinBox()
        self.image_reserve.setRange(128, 32768)
        self.image_reserve.setSingleStep(512)
        self.image_reserve.setToolTip(
            "Estimated tokens reserved per screenshot before fitting conversation history. Actual image costs depend on the model."
        )
        self.summary = QSpinBox()
        self.summary.setRange(128, 32768)
        self.summary.setSingleStep(256)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 1000)
        self.protocol = QComboBox()
        self.protocol.addItem("Standard conversation", "implicit")
        self.protocol.addItem("Explicit task_finish integration", "explicit")
        self.compact = QCheckBox("Summarize older exchanges and persist checkpoints")
        for label, widget in [
            ("Prompt budget", self.context),
            ("Output reserve", self.reserve),
            ("Tokens per screenshot · estimate", self.image_reserve),
            ("Recent events", self.recent),
            ("Summary target", self.summary),
            ("Iteration limit", self.iterations),
            ("Completion mode", self.protocol),
            ("History compaction", self.compact),
        ]:
            f.addRow(label, widget)

        agents, f = self.page(
            "Delegation & skills",
            "Give children independent contexts and keep their capabilities bounded.",
        )
        self.agents = QCheckBox("Enable delegated agents")
        self.writes = QCheckBox("Allow workers to edit isolated copies")
        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.timeout = QSpinBox()
        self.timeout.setRange(1, 3600)
        self.timeout.setSuffix(" seconds")
        self.skills_enabled = QCheckBox("Enable skill discovery and reading")
        self.builtin_skills = QCheckBox("Include the six bundled skills")
        for label, widget in [
            ("Subagents", self.agents),
            ("Worker edits", self.writes),
            ("Concurrent children", self.concurrent),
            ("Child deadline", self.timeout),
            ("Skills", self.skills_enabled),
            ("Built-in library", self.builtin_skills),
        ]:
            f.addRow(label, widget)
        note = QLabel(
            "Use the Skills page to inspect instructions and choose individual skills. Use Agent graph to inspect live delegated work."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        f.addRow(note)

        tools = QWidget()
        t = QVBoxLayout(tools)
        title = QLabel("Tool permissions")
        title.setObjectName("sectionTitle")
        t.addWidget(title)
        self.network = QCheckBox("Allow web search, web fetch and browser control")
        self.background = QCheckBox("Allow managed background processes")
        self.workspace_only = QCheckBox("Keep file tools inside the workspace")
        for widget in (self.network, self.background, self.workspace_only):
            t.addWidget(widget)
        self.scope_editor = ToolScopeEditor()
        self.tools = self.scope_editor.list
        t.addWidget(self.scope_editor)
        browser_title = QLabel("Browser")
        browser_title.setObjectName("sectionTitle")
        t.addWidget(browser_title)
        browser_form = QFormLayout()
        self.browser_headless = QCheckBox("Run without a visible browser window")
        self.browser_executable = QLineEdit()
        self.browser_executable.setPlaceholderText("Blank: Playwright Chromium")
        self.browser_cdp = QLineEdit()
        self.browser_cdp.setPlaceholderText("Optional: http://localhost:9222")
        browser_form.addRow(self.browser_headless)
        browser_form.addRow("Chromium executable", self.browser_executable)
        browser_form.addRow("Connect to existing browser · CDP", self.browser_cdp)
        t.addLayout(browser_form)
        browser_note = QLabel(
            "Install the browser extra and run: python -m playwright install chromium. Blank CDP uses a separate browser profile for this conversation."
        )
        browser_note.setWordWrap(True)
        browser_note.setObjectName("muted")
        t.addWidget(browser_note)
        self.pages.addWidget(tools)

        _, mcp_form = self.page(
            "MCP Servers",
            "Configure trusted, already-installed MCP stdio servers. "
            "Each enabled command runs with your OS user permissions. "
            "Save settings and restart Daena to discover its tools.",
        )
        self.mcp_servers = QPlainTextEdit()
        self.mcp_servers.setPlaceholderText(
            '{"example": {"enabled": false, "command": ["python", "-m", "server"]}}'
        )
        self.mcp_servers.setMinimumHeight(260)
        mcp_form.addRow("Servers · JSON", self.mcp_servers)
        mcp_notice = QLabel(
            "Only enable commands you installed and trust. "
            "MCP tools use the normal tool search catalog. "
            "Network permissions apply. Editing does not install packages."
        )
        mcp_notice.setWordWrap(True)
        mcp_form.addRow(mcp_notice)

        advanced = QWidget()
        a = QVBoxLayout(advanced)
        title = QLabel("Advanced profile")
        title.setObjectName("sectionTitle")
        a.addWidget(title)
        note = QLabel(
            "Load the form before editing JSON. Apply JSON separately; other pages handle ordinary settings."
        )
        note.setWordWrap(True)
        a.addWidget(note)
        self.advanced = QPlainTextEdit()
        a.addWidget(self.advanced)
        row = QHBoxLayout()
        load = QPushButton("Load form into JSON")
        load.clicked.connect(self.load_advanced)
        use = QPushButton("Apply JSON profile")
        use.clicked.connect(self.apply_advanced)
        row.addWidget(load)
        row.addWidget(use)
        a.addLayout(row)
        self.pages.addWidget(advanced)
        # Voice preferences belong to a separate client configuration, not the
        # Agent profile. Construct its model/device controls only on demand.
        self.voice_index = self.pages.count()
        self.voice_page = QWidget()
        self.voice_layout = QVBoxLayout(self.voice_page)
        self.voice_layout.setContentsMargins(24, 10, 24, 24)
        self.voice_layout.setSpacing(16)
        voice_title = QLabel("Voice settings")
        voice_title.setObjectName("sectionTitle")
        self.voice_layout.addWidget(voice_title)
        self.voice_hint = QLabel(
            "Choose local speech models, voices, audio devices and acoustic perception. "
            "Voice preferences save separately for this workspace and remain available while the Agent works."
        )
        self.voice_hint.setWordWrap(True)
        self.voice_hint.setObjectName("muted")
        self.voice_layout.addWidget(self.voice_hint)
        self.voice_placeholder = QWidget()
        placeholder = QVBoxLayout(self.voice_placeholder)
        placeholder.setContentsMargins(0, 0, 0, 0)
        self.open_voice_button = QPushButton("Open voice settings")
        self.open_voice_button.clicked.connect(self.voice_requested)
        placeholder.addWidget(self.open_voice_button)
        placeholder.addStretch()
        self.voice_layout.addWidget(self.voice_placeholder, 1)
        self.pages.addWidget(self.voice_page)
        bottom = QHBoxLayout()
        self.notice = QLabel("Changes apply on the next turn.")
        self.notice.setObjectName("muted")
        bottom.addWidget(self.notice)
        bottom.addStretch()
        self.apply_button = QPushButton("Save workspace settings")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(self.apply_form)
        bottom.addWidget(self.apply_button)
        root.addLayout(bottom)
        self.provider.currentTextChanged.connect(self.change_provider)
        self.generation.textChanged.connect(self.sync_generation_controls)
        for signal in (
            self.reasoning.currentIndexChanged,
            self.temperature.valueChanged,
            self.native_context.valueChanged,
            self.output_tokens.valueChanged,
        ):
            signal.connect(self.update_generation)
        self.populate(config)
        self.sections.currentRowChanged.connect(self._section_changed)
        self.sections.setCurrentRow(0)

    def page(self, title, description):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 10, 24, 24)
        layout.setSpacing(18)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        text = QLabel(description)
        text.setWordWrap(True)
        text.setObjectName("muted")
        layout.addWidget(text)
        form = QFormLayout()
        form.setVerticalSpacing(18)
        form.setHorizontalSpacing(30)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        layout.addStretch()
        scroll.setWidget(body)
        self.pages.addWidget(scroll)
        return body, form

    def show_section(self, index):
        self.sections.setCurrentRow(index)

    def show_voice(self):
        """Navigate without constructing audio devices or speech models here."""
        if self.sections.currentRow() == self.voice_index:
            self._request_voice()
        else:
            self.show_section(self.voice_index)

    def _request_voice(self):
        if self._voice_widget is None:
            self.voice_requested.emit()

    def _section_changed(self, index):
        voice = index == self.voice_index
        self.apply_button.setVisible(not voice)
        if voice:
            self.notice.setText(
                "Voice preferences save separately from the Agent profile."
            )
            self._request_voice()
        else:
            self.notice.setText(
                "Agent running · workspace controls are locked; Voice settings remain available."
                if self._runtime_busy
                else "Changes apply on the next turn."
            )

    def mount_voice_settings(self, widget):
        """Host the controller's canonical configuration widget, without copying it.

        The owner retains configuration/lifecycle responsibility. Repeated calls
        with the same widget are harmless; replacing a widget detaches the old
        one rather than deleting a live controller-owned client.
        """
        if widget is self._voice_widget:
            return
        if not isinstance(widget, QWidget):
            raise TypeError("Voice settings must be a QWidget")
        if self._voice_widget is not None:
            self.voice_layout.removeWidget(self._voice_widget)
            self._voice_widget.hide()
            self._voice_widget.setParent(None)
        self.voice_placeholder.hide()
        widget.setParent(self.voice_page, Qt.Widget)
        self.voice_layout.addWidget(widget, 1)
        self._voice_widget = widget
        widget.show()

    def set_runtime_busy(self, busy):
        """Lock runtime mutation without disabling navigation or Voice controls."""
        self._runtime_busy = bool(busy)
        for index in range(self.voice_index):
            self.pages.widget(index).setEnabled(not busy)
        self.apply_button.setEnabled(not busy)
        self._section_changed(self.sections.currentRow())

    def change_provider(self, provider):
        if self._syncing:
            return
        try:
            current = self.value(
                provider_name=self.config.get("llm", {}).get("provider", "ollama")
            )
            self.populate(switch_profile(current, provider))
        except Exception as exc:
            self.error.emit(str(exc))

    def populate(self, config):
        self._syncing = True
        self.config = deepcopy(config)
        llm = config.get("llm", {})
        pc = llm.get("provider_config", {})
        c = config.get("context", {})
        s = config.get("subagents", {})
        self.provider.setCurrentText(llm.get("provider", "ollama"))
        self.model.setText(pc.get("model_name", ""))
        self.base.setText(pc.get("base_url", ""))
        self.generation.setPlainText(
            json.dumps(pc.get("generation_config", {}), indent=2)
        )
        self.context.setValue(c.get("max_prompt_tokens", 24000))
        self.recent.setValue(c.get("recent_event_limit", 80))
        self.reserve.setValue(c.get("output_reserve_tokens", 4096))
        self.image_reserve.setValue(c.get("image_token_reserve", 4096))
        self.summary.setValue(c.get("compaction_target_tokens", 2048))
        self.iterations.setValue(config.get("max_agent_iterations", 100))
        self.protocol.setCurrentIndex(
            max(
                0,
                self.protocol.findData(
                    config.get("runtime", {}).get("completion_protocol", "implicit")
                ),
            )
        )
        self.compact.setChecked(c.get("compaction_enabled", True))
        self.agents.setChecked(s.get("enabled", True))
        self.writes.setChecked(s.get("allow_writes", False))
        self.concurrent.setValue(s.get("max_concurrent", 3))
        self.timeout.setValue(s.get("timeout_seconds", 900))
        self.skills_enabled.setChecked(config.get("skills", {}).get("enabled", True))
        self.builtin_skills.setChecked(
            config.get("skills", {}).get("include_builtin", True)
        )
        policy = SecurityPolicy.from_config(config.get("security"))
        self.network.setChecked(policy.allow_network_tools)
        self.background.setChecked(policy.allow_background)
        self.workspace_only.setChecked(policy.workspace_only)
        self.scope_editor.set_scope(policy.allowed_tools)
        browser = config.get("browser", {})
        self.browser_headless.setChecked(browser.get("headless", False))
        self.browser_executable.setText(browser.get("executable_path", ""))
        self.browser_cdp.setText(browser.get("cdp_url", ""))
        self.mcp_servers.setPlainText(
            json.dumps(config.get("mcp", {}).get("servers", {}), indent=2)
        )
        self._syncing = False
        self.sync_generation_controls()

    def sync_generation_controls(self):
        if self._syncing:
            return
        try:
            options = json.loads(self.generation.toPlainText())
        except ValueError:
            return
        if not isinstance(options, dict):
            return
        self._syncing = True
        self.reasoning.setCurrentIndex(
            max(0, self.reasoning.findData(reasoning_value(options)))
        )
        self.temperature.setValue(options.get("temperature", 0.3))
        self.native_context.setValue(options.get("num_ctx", 32768))
        self.output_tokens.setValue(
            options.get(
                "num_predict",
                options.get("max_tokens", options.get("max_output_tokens", 0)),
            )
            or 0
        )
        self.native_context.setEnabled(self.provider.currentText() == "ollama")
        self._syncing = False

    def update_generation(self, *args):
        if self._syncing:
            return
        try:
            options = json.loads(self.generation.toPlainText())
        except ValueError:
            return
        if not isinstance(options, dict):
            return
        options = set_reasoning(options, self.reasoning.currentData())
        options["temperature"] = self.temperature.value()
        if self.provider.currentText() == "ollama":
            options["num_ctx"] = self.native_context.value()
        for key in (
            "num_predict",
            "max_tokens",
            "max_output_tokens",
            "max_completion_tokens",
        ):
            options.pop(key, None)
        if self.output_tokens.value():
            key = {
                "ollama": "num_predict",
                "openrouter": "max_tokens",
                "gemini": "max_output_tokens",
            }[self.provider.currentText()]
            options[key] = self.output_tokens.value()
        self._syncing = True
        self.generation.setPlainText(json.dumps(options, indent=2))
        self._syncing = False

    def select_tools(self, checked):
        self.scope_editor.select_all(checked)

    def value(self, provider_name=None):
        result = deepcopy(self.config)
        options = json.loads(self.generation.toPlainText())
        if not isinstance(options, dict):
            raise ValueError("Provider options must be a JSON object")
        pc = {
            **result.get("llm", {}).get("provider_config", {}),
            "model_name": self.model.text().strip(),
            "generation_config": options,
        }
        pc.pop("base_url", None)
        if self.base.text().strip():
            pc["base_url"] = self.base.text().strip()
        result["llm"] = {
            "provider": provider_name or self.provider.currentText(),
            "provider_config": pc,
        }
        result.setdefault("context", {}).update(
            max_prompt_tokens=self.context.value(),
            recent_event_limit=self.recent.value(),
            output_reserve_tokens=self.reserve.value(),
            image_token_reserve=self.image_reserve.value(),
            compaction_target_tokens=self.summary.value(),
            compaction_enabled=self.compact.isChecked(),
        )
        result["max_agent_iterations"] = self.iterations.value()
        result.setdefault("runtime", {})["completion_protocol"] = (
            self.protocol.currentData()
        )
        result.setdefault("subagents", {}).update(
            enabled=self.agents.isChecked(),
            allow_writes=self.writes.isChecked(),
            max_concurrent=self.concurrent.value(),
            timeout_seconds=self.timeout.value(),
        )
        result.setdefault("skills", {}).update(
            enabled=self.skills_enabled.isChecked(),
            include_builtin=self.builtin_skills.isChecked(),
        )
        result.setdefault("security", {}).update(
            allow_network_tools=self.network.isChecked(),
            allow_background=self.background.isChecked(),
            workspace_only=self.workspace_only.isChecked(),
            allowed_tools=self.scope_editor.scope(),
        )
        result.setdefault("browser", {}).update(
            headless=self.browser_headless.isChecked(),
            executable_path=self.browser_executable.text().strip(),
            cdp_url=self.browser_cdp.text().strip(),
        )
        servers = json.loads(self.mcp_servers.toPlainText() or "{}")
        if not isinstance(servers, dict):
            raise ValueError("MCP Servers must be a JSON object")
        from src.tools.mcp_stdio import configured_servers

        configured_servers({"mcp": {"servers": servers}})
        result.setdefault("mcp", {})["servers"] = servers
        # Studio saves an explicit tool scope. Enable only the MCP namespaces
        # for servers that the user explicitly opted into.
        scope = set(result["security"]["allowed_tools"])
        scope = {name for name in scope if not name.startswith("mcp_")}
        scope.update(
            f"mcp_{name}_*"
            for name, server in servers.items()
            if isinstance(server, dict) and server.get("enabled", False)
        )
        result["security"]["allowed_tools"] = sorted(scope)
        return validate_config(result)

    def apply_form(self):
        if self._runtime_busy:
            self.error.emit(
                "Wait for the Agent to finish before changing its workspace profile."
            )
            return
        try:
            self.applied.emit(self.value())
            self.notice.setText("Workspace settings saved.")
        except Exception as exc:
            self.error.emit(str(exc))

    def load_advanced(self):
        try:
            self.advanced.setPlainText(
                json.dumps(self.value(), ensure_ascii=False, indent=2)
            )
        except Exception as exc:
            self.error.emit(str(exc))

    def apply_advanced(self):
        if self._runtime_busy:
            self.error.emit(
                "Wait for the Agent to finish before changing its workspace profile."
            )
            return
        try:
            value = validate_config(json.loads(self.advanced.toPlainText()))
            self.populate(value)
            self.applied.emit(value)
        except Exception as exc:
            self.error.emit(str(exc))
