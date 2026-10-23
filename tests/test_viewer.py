import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_viewer_loads_real_receipts_and_zoom_is_bounded(engine):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPointF, QPoint, Qt
    from PySide6.QtGui import QWheelEvent
    from computer_use.viewer import Viewer

    tool, _ = engine
    assert tool.execute("capture").success
    app = QApplication.instance() or QApplication([])
    viewer = Viewer(tool.workspace_root)
    viewer.show()
    app.processEvents()
    assert viewer.sessions.count() == 1
    assert viewer.timeline.count() == 1
    assert len(viewer.canvas.scene().items()) == 1
    old = viewer.canvas.transform().m11()
    event = QWheelEvent(
        QPointF(100, 100),
        QPointF(100, 100),
        QPoint(),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    viewer.canvas.wheelEvent(event)
    assert viewer.canvas.transform().m11() > old
    viewer.canvas.resetTransform()
    assert viewer.canvas.transform().m11() == 1
    viewer.close()
    app.processEvents()
