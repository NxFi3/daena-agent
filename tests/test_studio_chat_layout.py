"""Studio conversation UI regression tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from src.studio.widgets import MessageCard, ToolActivity


def test_tool_receipts_are_correlated_and_previewed():
    app = QApplication.instance() or QApplication([])
    panel = ToolActivity()
    panel.add_call("read_file", {"file_path": "a.py"}, "a")
    panel.add_call("read_file", {"file_path": "b.py"}, "b")
    panel.finish_call("read_file", True, "ok", "b")
    assert panel.calls[0]["status"] == "running"
    assert panel.calls[1]["status"] == "complete"
    assert "a.py" in panel.preview.text()
    assert "b.py" in panel.preview.text()
    panel.close()


def test_markdown_is_rendered_during_streaming():
    app = QApplication.instance() or QApplication([])
    card = MessageCard("Daena")
    card.append_stream("**bold**")
    card._render_stream_markdown()
    assert card.body.toPlainText() == "bold"
    card.close()
