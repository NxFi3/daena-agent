from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QLabel,
)
from src.studio.blueprint import Blueprint, Node
from src.studio.graph import GraphCanvas
from PySide6.QtWidgets import QGraphicsItem
from src.studio.state import write_json
from src.security.Policy import DEFAULT_ALLOWED_TOOLS


class AutomationPanel(QWidget):
    runRequested = Signal(object)
    stopRequested = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.graph = Blueprint()
        self.selected = None
        self._loading_editor = False
        self._dirty = False
        self.node_results = {}
        self.node_statuses = {}
        self.draft_path = None
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(1200)
        self.autosave_timer.timeout.connect(self._autosave_edit)
        root = QVBoxLayout(self)
        title = QLabel("Connect work into a flow")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        description = QLabel(
            "Create agents, give each a task, then connect dependencies. Edits are saved when switching agents."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        root.addWidget(description)
        row = QHBoxLayout()
        self.name = QLineEdit(self.graph.name)
        row.addWidget(self.name)
        for label, slot in [
            ("Add agent", self.add_node),
            ("Open…", self.open_graph),
            ("Save…", self.save_graph),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        self.run = QPushButton("Run automation")
        self.run.setObjectName("primary")
        self.run.clicked.connect(self.start)
        row.addWidget(self.run)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stopRequested)
        self.stop_button.hide()
        row.addWidget(self.stop_button)
        root.addLayout(row)
        self.run_state = QLabel("Ready to run")
        self.run_state.setObjectName("muted")
        root.addWidget(self.run_state)
        split = QSplitter()
        root.addWidget(split)
        self.canvas = GraphCanvas(editable=True)
        split.addWidget(self.canvas)
        editor = QWidget()
        f = QVBoxLayout(editor)
        self.title = QLineEdit()
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("What should this agent accomplish and verify?")
        self.dependencies = QListWidget()
        self.dependencies.setMaximumHeight(130)
        self.tools = QListWidget()
        self.tools.setMaximumHeight(160)
        form = QFormLayout()
        form.addRow("Name", self.title)
        f.addLayout(form)
        f.addWidget(QLabel("Assignment"))
        f.addWidget(self.prompt)
        f.addWidget(QLabel("Run after"))
        f.addWidget(self.dependencies)
        f.addWidget(QLabel("Tool scope (intersected with workspace profile)"))
        f.addWidget(self.tools)
        row = QHBoxLayout()
        apply = QPushButton("Update node")
        apply.clicked.connect(self.apply_node)
        row.addWidget(apply)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_node)
        row.addWidget(remove)
        f.addLayout(row)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(130)
        f.addWidget(self.report)
        split.addWidget(editor)
        split.setSizes([760, 340])
        self.canvas.nodeSelected.connect(self.select_node)
        self.canvas.wireRequested.connect(self.wire)
        self._editor = editor
        editor.setEnabled(False)
        self.title.textChanged.connect(self._mark_dirty)
        self.prompt.textChanged.connect(self._mark_dirty)
        self.dependencies.itemChanged.connect(self._mark_dirty)
        self.tools.itemChanged.connect(self._mark_dirty)

    def _mark_dirty(self, *args):
        if not self._loading_editor and self.selected is not None:
            self._dirty = True
            self.autosave_timer.start()

    def set_draft_path(self, path):
        """Drafts belong to workspaces and remain separate from chat history."""
        self.autosave_timer.stop()
        self.draft_path = Path(path)
        self.graph = Blueprint()
        self.selected = None
        self._dirty = False
        self.node_results.clear()
        self.node_statuses.clear()
        self._editor.setEnabled(False)
        try:
            if self.draft_path.is_file():
                self.graph = Blueprint.load(
                    json.loads(self.draft_path.read_text(encoding="utf-8"))
                )
        except Exception as exc:
            self.error.emit(f"Could not restore automation draft: {exc}")
        self.name.setText(self.graph.name)
        self.run_state.setText("Draft restored" if self.graph.nodes else "Ready to run")
        self.draw()
        self.canvas.fit_graph()

    def _save_draft(self):
        if self.draft_path is not None:
            try:
                write_json(self.draft_path, self.graph.dump())
            except Exception as exc:
                self.error.emit(f"Could not save automation draft: {exc}")

    def _autosave_edit(self):
        self._commit_current_editor(silent=True)

    def flush_draft(self, *, silent=False):
        if not self._commit_current_editor(silent=silent):
            return False
        self._save_draft()
        return True

    def sync_positions(self):
        self.graph.name = self.name.text().strip() or "Untitled automation"
        for node in self.graph.nodes:
            item = self.canvas.nodes.get(node.id)
            if item:
                node.x, node.y = item.pos().x(), item.pos().y()

    def draw(self):
        # Editing a node must not reset zoom or pan every time the DAG redraws.
        self.canvas.clear_graph(reset_view=False)
        for node in self.graph.nodes:
            self.canvas.upsert(
                node.id,
                node.title,
                "Independent context · tool scope",
                self.node_statuses.get(node.id, "idle"),
                x=node.x,
                y=node.y,
            )
        for node in self.graph.nodes:
            for parent in node.depends_on:
                self.canvas.connect_nodes(parent, node.id)

    def _refresh_graph_content(self):
        """Refresh labels/edges without destroying nodes during a mouse event."""
        for node in self.graph.nodes:
            self.canvas.upsert(
                node.id,
                node.title,
                "Independent context · tool scope",
                self.node_statuses.get(node.id, "idle"),
            )
        for _, _, edge in self.canvas.edges:
            self.canvas.scene().removeItem(edge)
        self.canvas.edges.clear()
        for node in self.graph.nodes:
            for parent in node.depends_on:
                self.canvas.connect_nodes(parent, node.id)

    def add_node(self):
        if not self._commit_current_editor():
            return
        self.sync_positions()
        node = Node(
            uuid4().hex,
            f"Agent {len(self.graph.nodes) + 1}",
            "Describe the task and report what you verified.",
            x=len(self.graph.nodes) * 310,
            y=(len(self.graph.nodes) % 2) * 150,
        )
        self.graph.nodes.append(node)
        self.draw()
        self.select_node(node.id)
        self.canvas.fit_graph()
        self._save_draft()

    def select_node(self, ident, *, force=False):
        if self._loading_editor:
            return
        node = next((n for n in self.graph.nodes if n.id == ident), None)
        if node is None:
            return
        if self.selected == ident and self._editor.isEnabled() and not force:
            return
        if not self._commit_current_editor():
            if self.selected in self.canvas.nodes:
                self.canvas.nodes[self.selected].setSelected(True)
            return
        node = next(n for n in self.graph.nodes if n.id == ident)
        self._loading_editor = True
        try:
            self.selected = ident
            self._editor.setEnabled(True)
            self.title.setText(node.title)
            self.prompt.setPlainText(node.prompt)
            self.dependencies.clear()
            self.tools.clear()
            for other in self.graph.nodes:
                if other.id == ident:
                    continue
                item = QListWidgetItem(other.title)
                item.setData(Qt.UserRole, other.id)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.Checked if other.id in node.depends_on else Qt.Unchecked
                )
                self.dependencies.addItem(item)
            for name in sorted(DEFAULT_ALLOWED_TOOLS):
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.Checked
                    if node.tools is None or name in node.tools
                    else Qt.Unchecked
                )
                self.tools.addItem(item)
            self.report.setPlainText(self.node_results.get(ident, "No run result yet"))
            if ident in self.canvas.nodes:
                self.canvas.nodes[ident].setSelected(True)
            self._dirty = False
        finally:
            self._loading_editor = False

    def _commit_current_editor(self, *, silent=False):
        """Persist the active editor before switching nodes, saving or running."""
        self.sync_positions()
        if self.selected is None or not self._dirty:
            return True
        try:
            candidate = deepcopy(self.graph)
            node = next(n for n in candidate.nodes if n.id == self.selected)
            node.title = self.title.text().strip()
            node.prompt = self.prompt.toPlainText()
            node.depends_on = [
                self.dependencies.item(i).data(Qt.UserRole)
                for i in range(self.dependencies.count())
                if self.dependencies.item(i).checkState() == Qt.Checked
            ]
            node.tools = [
                self.tools.item(i).text()
                for i in range(self.tools.count())
                if self.tools.item(i).checkState() == Qt.Checked
            ]
            if not node.title:
                raise ValueError("Agent name cannot be empty")
            candidate.order()
            self.graph = candidate
            self._dirty = False
            self._refresh_graph_content()
            self.autosave_timer.stop()
            self._save_draft()
            return True
        except Exception as exc:
            if not silent:
                self.error.emit(str(exc))
            return False

    def apply_node(self):
        ident = self.selected
        if ident is None or not self._commit_current_editor():
            return
        self.select_node(ident, force=True)
        self.report.setPlainText("Changes saved")

    def remove_node(self):
        if self.selected is None:
            return
        self.sync_positions()
        self.graph.nodes = [n for n in self.graph.nodes if n.id != self.selected]
        self.node_results.pop(self.selected, None)
        self.node_statuses.pop(self.selected, None)
        for n in self.graph.nodes:
            n.depends_on = [d for d in n.depends_on if d != self.selected]
        self.selected = None
        self._dirty = False
        self._editor.setEnabled(False)
        self.draw()
        self._save_draft()

    def wire(self, source, target):
        if not self._commit_current_editor():
            return
        try:
            self.sync_positions()
            candidate = deepcopy(self.graph)
            node = next(n for n in candidate.nodes if n.id == target)
            if source not in node.depends_on:
                node.depends_on.append(source)
            candidate.order()
            self.graph = candidate
            self._refresh_graph_content()
            self.select_node(target, force=True)
            self._save_draft()
        except Exception as exc:
            self.error.emit(str(exc))

    def open_graph(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open blueprint", filter="JSON (*.json)"
        )
        if path:
            try:
                with open(path, encoding="utf-8") as stream:
                    self.graph = Blueprint.load(json.load(stream))
                self.name.setText(self.graph.name)
                self.selected = None
                self._dirty = False
                self.autosave_timer.stop()
                self.node_results.clear()
                self.node_statuses.clear()
                self._editor.setEnabled(False)
                self.draw()
                self.canvas.fit_graph()
                self._save_draft()
            except Exception as exc:
                self.error.emit(str(exc))

    def save_graph(self):
        if not self._commit_current_editor():
            return
        try:
            self.graph.order()
        except Exception as exc:
            self.error.emit(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save blueprint", "automation.json", "JSON (*.json)"
        )
        if path:
            try:
                write_json(path, self.graph.dump())
            except Exception as exc:
                self.error.emit(str(exc))

    def start(self):
        if not self._commit_current_editor():
            return
        try:
            self.graph.order()
            if not self.graph.nodes:
                raise ValueError("Add an agent node first")
            self.runRequested.emit(deepcopy(self.graph))
        except Exception as exc:
            self.error.emit(str(exc))

    def handle_event(self, event):
        if event.get("type") == "automation_node":
            ident = event["node_id"]
            self.node_statuses[ident] = event["status"]
            agent = next((n for n in self.graph.nodes if n.id == ident), None)
            if agent:
                self.run_state.setText(
                    f"{agent.title} · {event['status'].replace('_', ' ').title()}"
                )
            node = self.canvas.nodes.get(event["node_id"])
            if node:
                node.status = event["status"]
                node.update()
            if event.get("response"):
                self.node_results[ident] = event["response"]
                if self.selected == ident:
                    self.report.setPlainText(event["response"])

    def append_stream(self, ident, text):
        # Only show the selected node's output; keep separate buffers per agent.
        self.node_results[ident] = (self.node_results.get(ident, "") + text)[-50000:]
        if self.selected == ident:
            self.report.setPlainText(self.node_results[ident])

    def reset_run(self):
        self.node_results.clear()
        self.node_statuses.clear()
        for node in self.canvas.nodes.values():
            node.status = "idle"
            node.update()
        self.run_state.setText("Automation is starting…")
        self.report.setPlainText("Waiting for this agent's result…")

    def set_running(self, running):
        self.name.setEnabled(not running)
        for button in self.findChildren(QPushButton):
            button.setEnabled(not running)
        self.stop_button.setVisible(running)
        self.stop_button.setEnabled(running)
        self._editor.setEnabled(not running and self.selected is not None)
        self.canvas.editable = not running
        for node in self.canvas.nodes.values():
            node.setFlag(QGraphicsItem.ItemIsMovable, not running)
