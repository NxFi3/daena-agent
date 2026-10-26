"""Focused Voice surface stays responsive without loading a speech engine."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from src.studio.voice_screen import VoiceScreen


def test_microphone_transcript_is_visible_for_review_without_changing_input_mode(
    qt_app,
):
    screen = VoiceScreen(input_mode="microphone")
    screen.show()
    screen.set_transcript("Review these recognized words")
    assert screen.composer_frame.isVisible()
    assert screen.composer.toPlainText() == "Review these recognized words"
    assert screen._mode == "microphone"
    screen.acknowledge_submission("Review these recognized words")
    assert screen.composer_frame.isHidden()
    screen.close()


from src.studio.theme import style_for_theme


@pytest.fixture
def screen(qt_app):
    view = VoiceScreen()
    view.setStyleSheet(style_for_theme("dark"))
    view.show()
    qt_app.processEvents()
    yield view
    view.close()
    view.deleteLater()
    qt_app.processEvents()


def test_orb_opens_typed_conversation_and_enter_sends_to_agent(screen, qt_app):
    modes, requests = [], []
    screen.input_mode_requested.connect(modes.append)
    screen.send_requested.connect(requests.append)
    assert not screen.composer_frame.isVisible()
    QTest.mouseClick(screen.orb, Qt.LeftButton)
    qt_app.processEvents()
    assert modes == ["text"] and screen.composer_frame.isVisible()
    assert screen.composer.hasFocus()
    QTest.keyClicks(screen.composer, "Hello Daena")
    QTest.keyClick(screen.composer, Qt.Key_Return, Qt.ShiftModifier)
    QTest.keyClicks(screen.composer, "Please reply")
    QTest.keyClick(screen.composer, Qt.Key_Return)
    assert requests == ["Hello Daena\nPlease reply"]
    # Submission is only cleared after the host acknowledges acceptance.
    assert screen.composer.toPlainText() == requests[0]
    screen.acknowledge_submission(requests[0])
    assert not screen.composer.toPlainText()
    QTest.mouseClick(screen.input_button, Qt.LeftButton)
    assert modes[-1] == "microphone" and not screen.composer_frame.isVisible()


def test_busy_agent_keeps_settings_typing_interrupt_and_end_accessible(screen):
    requests, settings, interrupts, stops = [], [], [], []
    screen.send_requested.connect(requests.append)
    screen.settings_requested.connect(lambda: settings.append(1))
    screen.interrupt_requested.connect(lambda: interrupts.append(1))
    screen.stop_requested.connect(lambda: stops.append(1))
    screen.set_input_mode("text")
    screen.set_active(True)
    screen.set_busy(True)
    screen.set_state("Thinking about your request")
    screen.composer.setPlainText("Another message")
    assert screen.send_button.isEnabled() and screen.settings_button.isEnabled()
    assert screen.interrupt_button.isEnabled() and screen.end_button.isEnabled()
    screen.send_button.click()
    screen.settings_button.click()
    screen.interrupt_button.click()
    screen.end_button.click()
    assert requests == ["Another message"]
    assert settings == interrupts == stops == [1]
    assert screen.heading.text() == "Daena is thinking"


def test_hands_free_switch_is_live_and_host_sync_does_not_emit_requests(screen):
    requests = []
    screen.auto_send_changed.connect(requests.append)
    screen.set_auto_send(False)
    assert not requests and not screen.hands_free_button.isChecked()
    assert screen.hands_free_button.text() == "Review speech"
    screen.set_active(True)
    screen.set_busy(True)
    assert screen.hands_free_button.isVisible()
    assert screen.hands_free_button.isEnabled()
    screen.hands_free_button.click()
    assert requests == [True] and screen.hands_free_button.isChecked()
    screen.set_input_mode("text")
    assert screen.hands_free_button.isHidden()
    screen.set_input_mode("microphone")
    assert screen.hands_free_button.isVisible()
    assert screen.hands_free_button.isChecked()


def test_recognition_and_old_acknowledgements_do_not_replace_edited_draft(screen):
    screen.set_input_mode("text")
    screen.set_transcript("Recognized words")
    assert screen.composer.toPlainText() == "Recognized words"
    screen.composer.setPlainText("My edited draft")
    screen.set_transcript("Late recognition")
    screen.acknowledge_submission("Previously submitted message")
    assert screen.composer.toPlainText() == "My edited draft"
    screen.send_button.click()
    screen.composer.setPlainText("A draft after sending")
    screen.acknowledge_submission("My edited draft")
    assert screen.composer.toPlainText() == "A draft after sending"


def test_mode_changes_from_host_do_not_request_microphone_and_mute_is_explicit(screen):
    modes, muted = [], []
    screen.input_mode_requested.connect(modes.append)
    screen.mute_changed.connect(muted.append)
    screen.set_input_mode("text")
    screen.set_active(True)
    assert not modes and not screen.mute_button.isVisible()
    assert "Microphone off" in screen.footer.text()
    screen.set_input_mode("microphone")
    screen.mute_button.click()
    assert muted == [True] and "muted" in screen.footer.text()
    screen.set_active(False)
    assert muted == [True], "Stopping must not emit a spurious unmute request"


def test_small_screen_keeps_bottom_controls_and_bounded_reply_visible(screen, qt_app):
    screen.set_input_mode("text")
    screen.set_active(True)
    screen.set_reply("<b>" + "LongWord" * 200 + "</b>")
    screen.show_error("Connection error: " + "A" * 1000)
    screen.resize(500, 560)
    qt_app.processEvents()
    assert screen.width() == 500 and screen.height() == 560
    assert len(screen.reply.text()) <= 261
    assert screen.reply.textFormat() == screen.error_label.textFormat() == Qt.PlainText
    assert screen.center_scroll.horizontalScrollBar().maximum() == 0
    assert screen.end_button.isVisible() and screen.settings_button.isVisible()
    assert screen.rect().contains(
        screen.end_button.mapTo(screen, screen.end_button.rect().center())
    )


def test_animation_stops_when_hidden_and_keyboard_can_switch_input(screen, qt_app):
    screen.set_active(True)
    qt_app.processEvents()
    assert screen.orb._timer.isActive()
    screen.orb.setFocus()
    QTest.keyClick(screen.orb, Qt.Key_Space)
    assert screen.composer_frame.isVisible()
    screen.hide()
    assert not screen.orb._timer.isActive()


def test_close_stops_once_per_opening(screen, qt_app):
    stops = []
    screen.stop_requested.connect(lambda: stops.append(1))
    screen.close()
    qt_app.processEvents()
    assert stops == [1]
    screen.show()
    qt_app.processEvents()
    QTest.keyClick(screen, Qt.Key_Escape)
    assert stops == [1, 1]


def test_screen_import_and_construction_do_not_load_audio_or_models():
    code = """
from PySide6.QtWidgets import QApplication
from src.studio.voice_screen import VoiceScreen
import sys
app = QApplication([])
screen = VoiceScreen()
assert not any(name in sys.modules for name in (
    'src.studio.voice_audio', 'PySide6.QtMultimedia', 'faster_whisper',
    'piper', 'kokoro_onnx', 'sherpa_onnx'))
screen.close()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
        check=True,
        timeout=15,
        capture_output=True,
        text=True,
    )
