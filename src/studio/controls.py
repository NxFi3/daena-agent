"""Discoverable desktop controls for execution settings and tool scopes."""

from copy import deepcopy
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from src.security.Policy import SecurityPolicy
from src.tools.ToolRegistry import ToolRegistry
from src.studio.profiles import (
    REASONING,
    reasoning_value,
    switch_profile,
    set_reasoning,
)

LABELS = {
    "read_file": "Read files",
    "apply_patch": "Edit files",
    "make_directory": "Create folders",
    "command_exec": "Run commands",
    "process_poll": "Read process output",
    "process_write": "Write process input",
    "process_stop": "Stop processes",
    "web_search": "Search the web",
    "web_fetch": "Read web pages",
    "browser_use": "Control browser tabs and forms",
    "computer_use": "Control desktop applications",
    "subagent": "Delegate to agents",
    "plan": "Manage a plan",
    "skill_list": "Discover skills",
    "skill_read": "Read a skill",
    "task_finish": "Explicit completion",
}


class ToolScopeEditor(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a tool by name or capability…")
        root.addWidget(self.search)
        self.list = QListWidget()
        self.list.setSpacing(3)
        root.addWidget(self.list)
        registry = ToolRegistry()
        registry.discover()
        for name, tool in sorted(registry.tools.items()):
            item = QListWidgetItem(f"{LABELS.get(name, name)}   ·   {name}")
            item.setData(Qt.UserRole, name)
            item.setToolTip(tool.description)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list.addItem(item)
        self.search.textChanged.connect(self.filter)
        row = QHBoxLayout()
        for label, check in [("Select all", True), ("Clear selection", False)]:
            button = QPushButton(label)
            button.clicked.connect(lambda _, checked=check: self.select_all(checked))
            row.addWidget(button)
        root.addLayout(row)

    def filter(self, text):
        text = text.casefold()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(text not in (item.text() + " " + item.toolTip()).casefold())

    def select_all(self, checked):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def set_scope(self, names):
        names = set(names)
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setCheckState(
                Qt.Checked if item.data(Qt.UserRole) in names else Qt.Unchecked
            )

    def scope(self):
        return [
            self.list.item(i).data(Qt.UserRole)
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.Checked
        ]


class ToolScopeDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Daena's tools")
        self.resize(600, 650)
        root = QVBoxLayout(self)
        title = QLabel("Tools for this workspace")
        title.setObjectName("sectionTitle")
        root.addWidget(title)
        hint = QLabel(
            "Select the capabilities available on the next turn. Changes also limit delegated agents."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.editor = ToolScopeEditor()
        root.addWidget(self.editor)
        self.editor.set_scope(
            SecurityPolicy.from_config(config.get("security")).allowed_tools
        )
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)


class ExecutionBar(QWidget):
    changed = Signal(dict)
    toolsRequested = Signal()

    def __init__(self):
        super().__init__()
        self.config = {}
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.provider = QComboBox()
        self.provider.addItems(["ollama", "openrouter", "gemini"])
        self.provider.setMinimumWidth(120)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setToolTip(
            "Type an installed model name or provider model ID; Enter applies it."
        )
        self.model.setMinimumWidth(230)
        self.model.setInsertPolicy(QComboBox.NoInsert)
        self.model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(18)
        self.reasoning = QComboBox()
        for label, value in REASONING:
            self.reasoning.addItem(label, value)
        self.reasoning.setToolTip(
            "Reasoning setting sent to the provider. Effort levels require a model that supports them."
        )
        self.tools = QPushButton("Tools")
        self.tools.clicked.connect(self.toolsRequested)
        for label, widget in [
            ("Provider", self.provider),
            ("Model", self.model),
            ("Reasoning", self.reasoning),
        ]:
            box = QWidget()
            column = QVBoxLayout(box)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(4)
            text = QLabel(label)
            text.setObjectName("controlLabel")
            column.addWidget(text)
            column.addWidget(widget)
            row.addWidget(box, 2 if widget is self.model else 1)
        row.addWidget(self.tools)
        self.provider.currentTextChanged.connect(self.provider_changed)
        self.model.lineEdit().editingFinished.connect(self.commit)
        self.model.activated.connect(lambda _: self.commit())
        self.reasoning.currentIndexChanged.connect(lambda _: self.commit())

    def populate(self, config):
        self.config = deepcopy(config)
        pc = config.get("llm", {}).get("provider_config", {})
        for widget in (self.provider, self.model, self.reasoning):
            widget.blockSignals(True)
        self.provider.setCurrentText(config.get("llm", {}).get("provider", "ollama"))
        self.model.clear()
        self.model.addItem(pc.get("model_name", ""))
        for model in (
            config.get("studio", {})
            .get("recent_models", {})
            .get(self.provider.currentText(), [])
        ):
            if model and self.model.findText(model) < 0:
                self.model.addItem(model)
        self.model.setCurrentText(pc.get("model_name", ""))
        self.reasoning.setCurrentIndex(
            max(
                0,
                self.reasoning.findData(
                    reasoning_value(pc.get("generation_config", {}))
                ),
            )
        )
        for widget in (self.provider, self.model, self.reasoning):
            widget.blockSignals(False)
        allowed = SecurityPolicy.from_config(config.get("security")).allowed_tools
        self.tools.setText(f"Tools · {len(allowed)}")
        self.tools.setToolTip(", ".join(sorted(allowed)) or "No tools selected")

    def provider_changed(self, provider):
        config = switch_profile(self.config, provider)
        self.populate(config)
        self.changed.emit(config)

    def commit(self):
        if not self.config:
            return
        config = deepcopy(self.config)
        pc = config.setdefault("llm", {}).setdefault("provider_config", {})
        pc["model_name"] = self.model.currentText().strip()
        pc["generation_config"] = set_reasoning(
            pc.get("generation_config", {}), self.reasoning.currentData()
        )
        recent = (
            config.setdefault("studio", {})
            .setdefault("recent_models", {})
            .setdefault(config["llm"]["provider"], [])
        )
        if pc["model_name"] and pc["model_name"] not in recent:
            recent.insert(0, pc["model_name"])
        del recent[20:]
        self.config = deepcopy(config)
        self.changed.emit(config)
