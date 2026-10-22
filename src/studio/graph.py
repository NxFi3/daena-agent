"""Interactive Qt graph used for live delegation and editable automation DAGs."""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath, QFont
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsItem,
)

COLORS = {
    "idle": "#7c8aa5",
    "queued": "#7c8aa5",
    "running": "#a78bfa",
    "completed": "#40d9b4",
    "failed": "#ff7d93",
    "stopped": "#ffba6b",
    "blocked": "#ffba6b",
    "needs_input": "#ffba6b",
    "cancelled": "#ffba6b",
    "timed_out": "#ff7d93",
    "cancelling": "#ffba6b",
}

LIGHT_COLORS = {
    "idle": "#777386",
    "queued": "#777386",
    "running": "#7953bc",
    "completed": "#087d6c",
    "failed": "#b53d5c",
    "stopped": "#ad6918",
    "blocked": "#ad6918",
    "needs_input": "#ad6918",
    "cancelled": "#ad6918",
    "timed_out": "#b53d5c",
    "cancelling": "#ad6918",
}


class AgentNode(QGraphicsObject):
    WIDTH, HEIGHT = 280, 132

    def __init__(self, canvas, ident, title, subtitle, status="idle"):
        super().__init__()
        self.canvas, self.ident = canvas, ident
        self.title, self.subtitle, self.status = title, subtitle, status
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(1)

    def boundingRect(self):
        return QRectF(-8, -8, self.WIDTH + 16, self.HEIGHT + 16)

    def paint(self, p, option, widget=None):
        palette = LIGHT_COLORS if self.canvas.theme == "light" else COLORS
        color = QColor(palette.get(self.status, palette["idle"]))
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(self.canvas.color("#292929", "#ffffff"))
        p.setPen(
            QPen(
                color if self.isSelected() else self.canvas.color("#484848", "#d6d6d6"),
                2 if self.isSelected() else 1,
            )
        )
        p.drawRoundedRect(QRectF(0, 0, self.WIDTH, self.HEIGHT), 14, 14)
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(14, 16, 4, 27), 2, 2)
        p.setPen(self.canvas.color("#f4f4f4", "#242424"))
        p.setFont(QFont("Sans Serif", 11, QFont.DemiBold))
        p.drawText(
            QRectF(28, 14, 230, 26), Qt.AlignLeft | Qt.AlignVCenter, self.title[:30]
        )
        p.setPen(self.canvas.color("#b4b4b4", "#666666"))
        p.setFont(QFont("Sans Serif", 9))
        p.drawText(QRectF(28, 46, 230, 38), Qt.TextWordWrap, self.subtitle[:90])
        p.setPen(color)
        p.setFont(QFont("Sans Serif", 8, QFont.Bold))
        p.drawText(
            QRectF(28, 103, 230, 16),
            Qt.AlignLeft,
            self.status.upper().replace("_", " "),
        )
        p.setBrush(self.canvas.color("#1c1c1c", "#ffffff"))
        p.setPen(QPen(color, 2))
        for x in (0, self.WIDTH):
            p.drawEllipse(QPointF(x, self.HEIGHT / 2), 5, 5)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.update_edges()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        x = event.pos().x()
        if self.canvas.editable and abs(event.pos().y() - self.HEIGHT / 2) < 14:
            if x > self.WIDTH - 14:
                self.canvas.begin_wire(self.ident)
                self.canvas.setToolTip(
                    "Select an input port on another node to connect"
                )
                event.accept()
                return
            if x < 14 and self.canvas.pending_wire:
                source = self.canvas.pending_wire
                self.canvas.cancel_wire()
                self.canvas.wireRequested.emit(source, self.ident)
                event.accept()
                return
        super().mousePressEvent(event)


class GraphCanvas(QGraphicsView):
    nodeSelected = Signal(str)
    wireRequested = Signal(str, str)
    zoomChanged = Signal(int)

    def __init__(self, editable=False):
        super().__init__()
        self.theme = "dark"
        self.editable, self.pending_wire = editable, None
        self.wire_preview = None
        self._wire_dragging = False
        self.nodes, self.edges = {}, []
        self.setScene(QGraphicsScene(self))
        self.scene().selectionChanged.connect(self._selected)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setBackgroundBrush(QBrush(QColor("#1c1c1c")))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.StrongFocus)
        self._panning = False
        self._space = False
        self._pan_last = None
        self.setMinimumSize(300, 240)
        self.setSceneRect(-20000, -20000, 40000, 40000)
        self.centerOn(0, 0)
        self.setStyleSheet(
            "QGraphicsView {border: 1px solid #404040; border-radius: 12px;}"
        )

    def drawBackground(self, p, rect):
        super().drawBackground(p, rect)
        p.setPen(QPen(self.color("#323232", "#ededed"), 1))
        startx, starty = int(rect.left()) // 24 * 24, int(rect.top()) // 24 * 24
        if rect.width() * rect.height() < 50_000_000:
            for x in range(startx, int(rect.right()) + 1, 24):
                for y in range(starty, int(rect.bottom()) + 1, 24):
                    p.drawPoint(x, y)

    def color(self, dark, light):
        return QColor(light if self.theme == "light" else dark)

    def apply_theme(self, theme):
        self.theme = "light" if theme == "light" else "dark"
        self.setBackgroundBrush(self.color("#1c1c1c", "#fafafa"))
        border = "#d6d6d6" if self.theme == "light" else "#404040"
        self.setStyleSheet(
            f"QGraphicsView {{border:1px solid {border};border-radius:12px;}}"
        )
        for _, _, edge in self.edges:
            edge.setPen(QPen(self.color("#8794a6", "#8596a9"), 2))
        if self.wire_preview is not None:
            self.wire_preview.setPen(
                QPen(self.color("#b6a0ff", "#7761b2"), 2, Qt.DashLine)
            )
        self.scene().update()

    def _selected(self):
        nodes = [n for n in self.scene().selectedItems() if isinstance(n, AgentNode)]
        if nodes:
            self.nodeSelected.emit(nodes[0].ident)

    def clear_graph(self, *, reset_view=True):
        self.nodes, self.edges = {}, []
        self.scene().clear()
        self.pending_wire = None
        self.wire_preview = None
        self._wire_dragging = False
        if reset_view:
            self.reset_view()

    def begin_wire(self, source):
        """Click the output port, then click or drag to a target input port."""
        self.cancel_wire()
        self.pending_wire = source
        self._wire_dragging = False
        self.wire_preview = QGraphicsPathItem()
        self.wire_preview.setZValue(-1)
        self.wire_preview.setPen(QPen(self.color("#b6a0ff", "#7761b2"), 2, Qt.DashLine))
        self.scene().addItem(self.wire_preview)
        self._update_wire_preview(
            self.nodes[source].pos() + QPointF(AgentNode.WIDTH, AgentNode.HEIGHT / 2)
        )

    def _update_wire_preview(self, target):
        if self.pending_wire is None or self.wire_preview is None:
            return
        origin = self.nodes[self.pending_wire].pos() + QPointF(
            AgentNode.WIDTH, AgentNode.HEIGHT / 2
        )
        path = QPainterPath(origin)
        bend = max(60, abs(target.x() - origin.x()) / 2)
        path.cubicTo(origin + QPointF(bend, 0), target - QPointF(bend, 0), target)
        self.wire_preview.setPath(path)

    def cancel_wire(self):
        if self.wire_preview is not None:
            self.scene().removeItem(self.wire_preview)
            self.wire_preview = None
        self.pending_wire = None
        self._wire_dragging = False
        self.setToolTip("")

    def upsert(self, ident, title, subtitle, status="idle", x=None, y=None):
        if ident not in self.nodes:
            node = AgentNode(self, ident, title, subtitle, status)
            self.nodes[ident] = node
            self.scene().addItem(node)
            node.setPos(x or 0, y or 0)
        node = self.nodes[ident]
        node.title, node.subtitle, node.status = title, subtitle, status
        node.setToolTip(f"{title}\n{subtitle}\n{status}")
        node.update()
        return node

    def connect_nodes(self, source, target):
        if (
            source not in self.nodes
            or target not in self.nodes
            or any(s == source and t == target for s, t, _ in self.edges)
        ):
            return
        edge = QGraphicsPathItem()
        edge.setPen(QPen(self.color("#8794a6", "#8596a9"), 2))
        self.scene().addItem(edge)
        self.edges.append((source, target, edge))
        self.update_edges()

    def update_edges(self):
        for source, target, edge in self.edges:
            a = self.nodes[source].pos() + QPointF(
                AgentNode.WIDTH, AgentNode.HEIGHT / 2
            )
            b = self.nodes[target].pos() + QPointF(0, AgentNode.HEIGHT / 2)
            path = QPainterPath(a)
            bend = max(70, abs(b.x() - a.x()) / 2)
            path.cubicTo(a + QPointF(bend, 0), b - QPointF(bend, 0), b)
            edge.setPath(path)

    def fit_graph(self):
        if self.nodes:
            bounds = self.scene().itemsBoundingRect().adjusted(-65, -65, 65, 65)
            self.fitInView(bounds, Qt.KeepAspectRatio)
            if self.transform().m11() > 1.15:
                self.scale(1.15 / self.transform().m11(), 1.15 / self.transform().m11())
            self.zoomChanged.emit(round(self.transform().m11() * 100))

    def zoom_to(self, scale, anchor=None):
        scale = max(0.15, min(3.0, scale))
        anchor = anchor or self.viewport().rect().center()
        before = self.mapToScene(anchor)
        self.scale(scale / self.transform().m11(), scale / self.transform().m11())
        after = self.mapToScene(anchor)
        delta = after - before
        self.translate(delta.x(), delta.y())
        self.zoomChanged.emit(round(self.transform().m11() * 100))

    def zoom_in(self):
        self.zoom_to(self.transform().m11() * 1.2)

    def zoom_out(self):
        self.zoom_to(self.transform().m11() / 1.2)

    def reset_view(self):
        center = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(center)
        self.zoomChanged.emit(100)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if event.modifiers() & Qt.ShiftModifier:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta)
            )
        elif delta:
            self.zoom_to(
                self.transform().m11() * 1.15 ** (delta / 120),
                event.position().toPoint(),
            )
        event.accept()

    def mousePressEvent(self, event):
        self.setFocus(Qt.MouseFocusReason)
        item = self.itemAt(event.position().toPoint())
        background = not isinstance(item, AgentNode)
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and (self._space or background)
        ):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            if background and not event.modifiers() & Qt.ControlModifier:
                self.scene().clearSelection()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self.editable
            and self.pending_wire is not None
            and event.buttons() & Qt.LeftButton
        ):
            self._wire_dragging = True
            self._update_wire_preview(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        if self._panning:
            pos = event.position().toPoint()
            delta = pos - self._pan_last
            self._pan_last = pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self.editable
            and self.pending_wire is not None
            and event.button() == Qt.LeftButton
        ):
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, AgentNode) and item.ident != self.pending_wire:
                pos = item.mapFromScene(self.mapToScene(event.position().toPoint()))
                if pos.x() < 16 and abs(pos.y() - AgentNode.HEIGHT / 2) < 17:
                    source = self.pending_wire
                    self.cancel_wire()
                    self.wireRequested.emit(source, item.ident)
                    event.accept()
                    return
            if self._wire_dragging:
                self.cancel_wire()
                event.accept()
                return
        if self._panning:
            self._panning = False
            self.viewport().setCursor(
                Qt.OpenHandCursor if self._space else Qt.ArrowCursor
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif event.key() == Qt.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key_0:
            self.reset_view()
        elif event.key() == Qt.Key_F:
            self.fit_graph()
        elif event.key() == Qt.Key_Escape:
            self.cancel_wire()
            self.scene().clearSelection()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space = False
            self.viewport().setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space = self._panning = False
        self.viewport().setCursor(Qt.ArrowCursor)
        super().focusOutEvent(event)
