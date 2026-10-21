from __future__ import annotations

import json
from copy import deepcopy
from uuid import uuid4
from PySide6.QtCore import Qt, Signal
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
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.graph = Blueprint()
        self.selected = None
        root = QVBoxLayout(self)
        title = QLabel("Connect work into a flow")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(
            QLabel(
                "Connect an output port to an input port. Drag nodes to arrange. Runs execute in dependency order with independent contexts."
            )
        )
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
        root.addLayout(row)
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

    def sync_positions(self):
        self.graph.name = self.name.text().strip() or "Untitled automation"
        for node in self.graph.nodes:
            item = self.canvas.nodes.get(node.id)
            if item:
                node.x, node.y = item.pos().x(), item.pos().y()

    def draw(self):
        self.canvas.clear_graph()
        for node in self.graph.nodes:
            self.canvas.upsert(
                node.id,
                node.title,
                "Independent context · tool scope",
                x=node.x,
                y=node.y,
            )
        for node in self.graph.nodes:
            for parent in node.depends_on:
                self.canvas.connect_nodes(parent, node.id)

    def add_node(self):
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

    def select_node(self, ident):
        node = next((n for n in self.graph.nodes if n.id == ident), None)
        if node is None:
            return
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
                Qt.Checked if node.tools is None or name in node.tools else Qt.Unchecked
            )
            self.tools.addItem(item)

    def apply_node(self):
        if not self.selected:
            return
        try:
            self.sync_positions()
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
            candidate.order()
            self.graph = candidate
            self.draw()
            self.report.setPlainText("Node updated")
        except Exception as exc:
            self.error.emit(str(exc))

    def remove_node(self):
        self.sync_positions()
        self.graph.nodes = [n for n in self.graph.nodes if n.id != self.selected]
        for n in self.graph.nodes:
            n.depends_on = [d for d in n.depends_on if d != self.selected]
        self.selected = None
        self._editor.setEnabled(False)
        self.draw()

    def wire(self, source, target):
        try:
            self.sync_positions()
            candidate = deepcopy(self.graph)
            node = next(n for n in candidate.nodes if n.id == target)
            if source not in node.depends_on:
                node.depends_on.append(source)
            candidate.order()
            self.graph = candidate
            self.draw()
            self.select_node(target)
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
                self._editor.setEnabled(False)
                self.draw()
                self.canvas.fit_graph()
            except Exception as exc:
                self.error.emit(str(exc))

    def save_graph(self):
        self.sync_positions()
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
        self.sync_positions()
        try:
            self.graph.order()
            if not self.graph.nodes:
                raise ValueError("Add an agent node first")
            self.runRequested.emit(deepcopy(self.graph))
        except Exception as exc:
            self.error.emit(str(exc))

    def handle_event(self, event):
        if event.get("type") == "automation_node":
            node = self.canvas.nodes.get(event["node_id"])
            if node:
                node.status = event["status"]
                node.update()
            if event.get("response"):
                self.report.setPlainText(event["response"])

    def set_running(self, running):
        self.name.setEnabled(not running)
        for button in self.findChildren(QPushButton):
            button.setEnabled(not running)
        self._editor.setEnabled(not running and self.selected is not None)
        self.canvas.editable = not running
        for node in self.canvas.nodes.values():
            node.setFlag(QGraphicsItem.ItemIsMovable, not running)
