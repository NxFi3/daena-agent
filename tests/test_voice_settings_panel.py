"""Voice settings stay available while runtime profile changes are locked."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QWidget, QLineEdit, QVBoxLayout

from src.studio.settings import SettingsPanel
from src.studio.state import default_config


def test_voice_controls_remain_editable_while_agent_profile_is_locked(qt_app):
    panel = SettingsPanel(default_config())
    voice = QWidget()
    layout = QVBoxLayout(voice)
    device = QLineEdit()
    layout.addWidget(device)
    panel.mount_voice_settings(voice)
    applied, errors = [], []
    panel.applied.connect(applied.append)
    panel.error.connect(errors.append)
    panel.set_runtime_busy(True)
    panel.show_voice()
    panel.show()
    qt_app.processEvents()

    assert panel.isEnabled() and panel.sections.isEnabled()
    assert not panel.provider.isEnabled() and not panel.advanced.isEnabled()
    assert not panel.apply_button.isEnabled()
    assert device.isEnabled() and panel.pages.currentWidget() is panel.voice_page
    QTest.keyClicks(device, "Selected speakers")
    assert device.text() == "Selected speakers"
    panel.apply_form()
    panel.apply_advanced()
    assert not applied and len(errors) == 2
    assert "separately" in panel.notice.text()

    panel.set_runtime_busy(False)
    panel.show_section(0)
    assert panel.provider.isEnabled() and panel.apply_button.isEnabled()
    panel.apply_form()
    assert len(applied) == 1
    assert "Selected speakers" not in str(applied[0])
    panel.close()


def test_voice_editor_is_requested_lazily_and_uses_one_canonical_widget(qt_app):
    panel = SettingsPanel(default_config())
    requests = []
    voice = QWidget()

    def load():
        requests.append(1)
        panel.mount_voice_settings(voice)

    panel.voice_requested.connect(load)
    for index in range(panel.voice_index):
        panel.show_section(index)
    assert not requests
    panel.show_voice()
    assert requests == [1] and voice.parentWidget() is panel.voice_page
    panel.show_section(0)
    panel.show_voice()
    panel.show_voice()
    panel.mount_voice_settings(voice)
    assert requests == [1]
    assert panel.voice_layout.indexOf(voice) >= 0
    assert panel.voice_placeholder.isHidden()
    panel.close()


def test_unconnected_voice_placeholder_is_actionable(qt_app):
    panel = SettingsPanel(default_config())
    requests = []
    panel.voice_requested.connect(lambda: requests.append(1))
    panel.show_voice()
    assert requests == [1]
    panel.open_voice_button.click()
    assert requests == [1, 1]
    assert panel.voice_index == 5
    panel.close()


def test_mount_converts_dialog_editor_to_embedded_widget(qt_app):
    panel = SettingsPanel(default_config())
    editor = QDialog()
    panel.mount_voice_settings(editor)
    panel.show_voice()
    panel.show()
    qt_app.processEvents()
    assert editor.parentWidget() is panel.voice_page
    assert not editor.isWindow()
    assert editor.isVisible()
    panel.close()


def test_settings_construction_does_not_import_voice_or_open_devices():
    source = """
import sys
from PySide6.QtWidgets import QApplication
from src.studio.settings import SettingsPanel
from src.studio.state import default_config
app = QApplication([])
panel = SettingsPanel(default_config())
panel.set_runtime_busy(True)
for index in range(panel.voice_index):
    panel.show_section(index)
assert not any(name.startswith(('src.voice', 'src.studio.voice', 'PySide6.QtMultimedia', 'faster_whisper', 'piper', 'kokoro_onnx', 'sherpa_onnx')) for name in sys.modules)
panel.close()
print('SETTINGS_WITHOUT_VOICE_IMPORTS_OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SETTINGS_WITHOUT_VOICE_IMPORTS_OK" in result.stdout
