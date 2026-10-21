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
    QCheckBox,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QLabel,
    QScrollArea,
)
from src.security.Policy import SecurityPolicy
from src.tools.ToolRegistry import ToolRegistry
from src.studio.state import validate_config


class SettingsPanel(QWidget):
    applied = Signal(dict)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        root = QVBoxLayout(self)
        title = QLabel("Make Daena yours")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(
            QLabel("Profiles belong to this workspace. Apply changes between runs.")
        )
        tabs = QTabWidget()
        root.addWidget(tabs)
        model = QWidget()
        form = QFormLayout(model)
        self.provider = QComboBox()
        self.provider.addItems(["ollama", "openrouter", "gemini"])
        self.model = QLineEdit()
        self.base = QLineEdit()
        self.base.setPlaceholderText("Default endpoint / OLLAMA_HOST")
        self.generation = QPlainTextEdit()
        self.generation.setMinimumHeight(180)
        self.generation.setPlaceholderText(
            '{"temperature": 0.3, "request_timeout_seconds": null}'
        )
        form.addRow("Provider", self.provider)
        form.addRow("Model", self.model)
        form.addRow("Base URL (optional)", self.base)
        form.addRow("Generation options · JSON", self.generation)
        form.addRow(
            QLabel(
                "Remote credentials stay in environment variables:\nOPENROUTER_API_KEY · GEMINI_API_KEY / GOOGLE_API_KEY\nNo API request is made when applying settings."
            )
        )
        tabs.addTab(model, "Model")
        runtime = QWidget()
        f = QFormLayout(runtime)
        self.context = QSpinBox()
        self.context.setRange(1024, 2000000)
        self.recent = QSpinBox()
        self.recent.setRange(4, 1000)
        self.reserve = QSpinBox()
        self.reserve.setRange(0, 131072)
        self.summary = QSpinBox()
        self.summary.setRange(128, 32768)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 1000)
        self.protocol = QComboBox()
        self.protocol.addItems(["implicit", "explicit"])
        self.compact = QCheckBox("Summarize older exchanges; keep recent receipts")
        self.agents = QCheckBox("Enable delegation")
        self.skills_enabled = QCheckBox("Enable skill discovery and reading")
        self.builtin_skills = QCheckBox("Include bundled skills")
        self.writes = QCheckBox("Workers may edit isolated copies")
        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.timeout = QSpinBox()
        self.timeout.setRange(1, 3600)
        self.timeout.setSuffix(" seconds")
        for label, widget in [
            ("Prompt budget", self.context),
            ("Output reserve", self.reserve),
            ("Recent events", self.recent),
            ("Summary target", self.summary),
            ("Max iterations", self.iterations),
            ("Completion protocol", self.protocol),
            ("Compaction", self.compact),
            ("Skills", self.skills_enabled),
            ("Built-in skills", self.builtin_skills),
            ("Subagents", self.agents),
            ("Worker writes", self.writes),
            ("Concurrent children", self.concurrent),
            ("Child deadline", self.timeout),
        ]:
            f.addRow(label, widget)
        runtimeScroll = QScrollArea()
        runtimeScroll.setWidgetResizable(True)
        runtimeScroll.setWidget(runtime)
        tabs.addTab(runtimeScroll, "Context & agents")
        tools = QWidget()
        t = QVBoxLayout(tools)
        t.addWidget(
            QLabel(
                "Checked tools are available to the model. Child permissions are always intersected with this scope."
            )
        )
        self.network = QCheckBox("Allow web search and web fetch")
        self.background = QCheckBox(
            "Allow managed background commands (stopped when the run ends)"
        )
        self.workspace_only = QCheckBox("Resolve file tools within the workspace")
        t.addWidget(self.network)
        t.addWidget(self.background)
        t.addWidget(self.workspace_only)
        self.tools = QListWidget()
        t.addWidget(self.tools)
        row = QHBoxLayout()
        for title, checked in [("Enable all", True), ("Disable all", False)]:
            b = QPushButton(title)
            b.clicked.connect(lambda _, c=checked: self.select_tools(c))
            row.addWidget(b)
        t.addLayout(row)
        registry = ToolRegistry()
        registry.discover()
        for name, tool in sorted(registry.tools.items()):
            item = QListWidgetItem(name)
            item.setToolTip(tool.description)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.tools.addItem(item)
        tabs.addTab(tools, "Tools")
        self.advanced = QPlainTextEdit()
        self.advanced.setPlaceholderText("Full profile JSON")
        advanced = QWidget()
        a = QVBoxLayout(advanced)
        a.addWidget(
            QLabel(
                "Advanced profile editor. Load the current form before editing; apply JSON separately."
            )
        )
        a.addWidget(self.advanced)
        load = QPushButton("Load form into JSON")
        load.clicked.connect(self.load_advanced)
        use = QPushButton("Apply JSON profile")
        use.clicked.connect(self.apply_advanced)
        a.addWidget(load)
        a.addWidget(use)
        tabs.addTab(advanced, "Advanced")
        apply = QPushButton("Save and apply profile")
        apply.setObjectName("primary")
        apply.clicked.connect(self.apply_form)
        root.addWidget(apply)
        self.provider.currentTextChanged.connect(self.change_provider)
        self.populate(config)

    def change_provider(self, provider):
        defaults = {
            "ollama": (
                "qwen3.6:35b-a3b-q4_K_M",
                {"temperature": 0.3, "num_ctx": 32768, "request_timeout_seconds": None},
            ),
            "openrouter": (
                "openrouter/free",
                {"temperature": 0.3, "request_timeout_seconds": None},
            ),
            "gemini": (
                "gemini-2.5-flash",
                {"temperature": 0.3, "request_timeout_seconds": 600},
            ),
        }
        model, options = defaults[provider]
        self.model.setText(model)
        self.base.clear()
        self.generation.setPlainText(json.dumps(options, indent=2))

    def populate(self, config):
        self.config = deepcopy(config)
        llm = config.get("llm", {})
        pc = llm.get("provider_config", {})
        c = config.get("context", {})
        s = config.get("subagents", {})
        self.provider.blockSignals(True)
        self.provider.setCurrentText(llm.get("provider", "ollama"))
        self.provider.blockSignals(False)
        self.model.setText(pc.get("model_name", ""))
        self.base.setText(pc.get("base_url", ""))
        self.generation.setPlainText(
            json.dumps(pc.get("generation_config", {}), indent=2)
        )
        self.context.setValue(c.get("max_prompt_tokens", 24000))
        self.recent.setValue(c.get("recent_event_limit", 80))
        self.reserve.setValue(c.get("output_reserve_tokens", 4096))
        self.summary.setValue(c.get("compaction_target_tokens", 2048))
        self.iterations.setValue(config.get("max_agent_iterations", 100))
        self.protocol.setCurrentText(
            config.get("runtime", {}).get("completion_protocol", "implicit")
        )
        self.compact.setChecked(c.get("compaction_enabled", True))
        self.agents.setChecked(s.get("enabled", True))
        self.writes.setChecked(s.get("allow_writes", False))
        self.skills_enabled.setChecked(config.get("skills", {}).get("enabled", True))
        self.builtin_skills.setChecked(
            config.get("skills", {}).get("include_builtin", True)
        )
        self.concurrent.setValue(s.get("max_concurrent", 3))
        self.timeout.setValue(s.get("timeout_seconds", 900))
        policy = SecurityPolicy.from_config(config.get("security"))
        self.network.setChecked(policy.allow_network_tools)
        self.background.setChecked(policy.allow_background)
        self.workspace_only.setChecked(policy.workspace_only)
        for i in range(self.tools.count()):
            item = self.tools.item(i)
            item.setCheckState(
                Qt.Checked if item.text() in policy.allowed_tools else Qt.Unchecked
            )

    def select_tools(self, checked):
        for i in range(self.tools.count()):
            self.tools.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def value(self):
        result = deepcopy(self.config)
        options = json.loads(self.generation.toPlainText())
        if not isinstance(options, dict):
            raise ValueError("Generation options must be an object")
        pc = {
            **result.get("llm", {}).get("provider_config", {}),
            "model_name": self.model.text().strip(),
            "generation_config": options,
        }
        pc.pop("base_url", None)
        if self.base.text().strip():
            pc["base_url"] = self.base.text().strip()
        result["llm"] = {"provider": self.provider.currentText(), "provider_config": pc}
        result.setdefault("context", {}).update(
            max_prompt_tokens=self.context.value(),
            recent_event_limit=self.recent.value(),
            output_reserve_tokens=self.reserve.value(),
            compaction_target_tokens=self.summary.value(),
            compaction_enabled=self.compact.isChecked(),
        )
        result["max_agent_iterations"] = self.iterations.value()
        result.setdefault("runtime", {})["completion_protocol"] = (
            self.protocol.currentText()
        )
        result.setdefault("skills", {}).update(
            enabled=self.skills_enabled.isChecked(),
            include_builtin=self.builtin_skills.isChecked(),
        )
        result.setdefault("subagents", {}).update(
            enabled=self.agents.isChecked(),
            allow_writes=self.writes.isChecked(),
            max_concurrent=self.concurrent.value(),
            timeout_seconds=self.timeout.value(),
        )
        result.setdefault("security", {}).update(
            allow_network_tools=self.network.isChecked(),
            allow_background=self.background.isChecked(),
            workspace_only=self.workspace_only.isChecked(),
            allowed_tools=[
                self.tools.item(i).text()
                for i in range(self.tools.count())
                if self.tools.item(i).checkState() == Qt.Checked
            ],
        )
        return validate_config(result)

    def apply_form(self):
        try:
            self.applied.emit(self.value())
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
        try:
            value = validate_config(json.loads(self.advanced.toPlainText()))
            self.populate(value)
            self.applied.emit(value)
        except Exception as exc:
            self.error.emit(str(exc))
