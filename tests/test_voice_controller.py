"""Voice is opt-in and uses the existing Agent entry point without hardware."""

import os
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget, QPlainTextEdit
from PySide6.QtTest import QTest
from src.studio.voice_controller import VoiceController
from src.studio.voice_dialog import VoiceDialog
from src.voice.session import VoiceEvent
from src.voice.providers import AudioChunk


class Audio(QObject):
    pcm_ready = Signal(bytes)
    level = Signal(float)
    fault = Signal(str)
    playback_started = Signal()
    playback_stopped = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.played = []
        self.accept = True
        self.closed = False
        self.flushed = 0

    def start_capture(self, *args):
        return True

    def set_output_device(self, *args):
        pass

    def set_muted(self, *args):
        pass

    def play_pcm(self, pcm, rate):
        self.played.append((pcm, rate))
        return True

    def can_accept(self, *args):
        return self.accept

    def stop_playback(self):
        self.played.clear()

    def finish_playback(self):
        self.flushed += 1

    def close(self):
        self.closed = True

    @property
    def queued_seconds(self):
        return 0.0


class Session:
    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.events = Queue()
        self.audio = Queue()
        self.spoken = []
        self.retired = True
        self.segmenter = SimpleNamespace(reset=lambda: None)

    def speak(self, text, language):
        self.spoken.append(text)

    def interrupt(self):
        self.epoch += 1

    def close(self):
        self.epoch += 1

    def feed(self, data):
        pass

    def finish_utterance(self):
        pass


@pytest.fixture
def controller(qt_app, tmp_path):
    window = QWidget()
    window.workspace = tmp_path
    window.session_id = "session-a"
    window.store = SimpleNamespace(root=tmp_path / "studio")
    window.input = QPlainTextEdit(window)
    window.busy = False
    window._automation_run = False
    window.sent = []

    def send():
        window.sent.append(window.input.toPlainText())
        window.input.clear()

    window.send = send
    result = VoiceController(
        window,
        audio_factory=Audio,
        session_factory=Session,
        provider_factory=lambda s: (object(), object()),
    )
    window.voice_controller = result
    window.stop = lambda: result.cancel_speech()
    result.dialog = VoiceDialog(parent=window)
    result.start({"language": "auto", "auto_send": False})
    assert result.session is not None
    yield result, window
    result.close()
    window.close()
    qt_app.processEvents()


def test_review_does_not_duplicate_transcript_or_overwrite_draft(controller):
    c, w = controller
    w.input.setPlainText("Existing draft")
    c.session.events.put(
        VoiceEvent("transcript", {"text": "سلام", "truncated": False, "stt_ms": 12}, 0)
    )
    c.poll()
    assert not w.sent and w.input.toPlainText() == "Existing draft\nسلام"
    c.send_review("سلام اصلاح شده")
    assert w.sent == ["Existing draft\nسلام اصلاح شده"]


def test_autosend_requires_complete_utterance_idle_agent_and_empty_draft(controller):
    c, w = controller
    c._settings["auto_send"] = True
    c.session.events.put(
        VoiceEvent("transcript", {"text": "Hello", "truncated": True, "stt_ms": 1}, 0)
    )
    c.poll()
    assert not w.sent and w.input.toPlainText() == "Hello"
    w.input.clear()
    w.busy = True
    c.session.events.put(
        VoiceEvent(
            "transcript",
            {"text": "Review while busy", "truncated": False, "stt_ms": 1},
            0,
        )
    )
    c.poll()
    assert not w.sent
    w.input.clear()
    w.busy = False
    c.session.events.put(
        VoiceEvent(
            "transcript",
            {"text": "Complete request", "truncated": False, "stt_ms": 1},
            0,
        )
    )
    c.poll()
    assert w.sent == ["Complete request"]


def test_only_committed_responses_are_spoken_and_result_is_not_duplicated(controller):
    c, w = controller
    for kind in ("thinking_delta", "content_delta", "tool_result"):
        c.on_event({"type": kind, "text": "Do not speak drafts or reasoning"})
    assert not c.session.spoken
    c.on_event({"type": "final_response", "text": "سلام، آماده‌ام."})
    c.on_result({"text": "سلام، آماده‌ام.", "metrics": {"completed": True}})
    assert c.session.spoken == ["سلام، آماده‌ام."]


def test_audio_backpressure_preserves_chunk_and_flushes_tail(controller):
    c, w = controller
    chunk = AudioChunk(16000, b"\0\0" * 100)
    c.session.audio.put((0, chunk))
    c.session.audio.put((0, None))
    c.audio.accept = False
    c.poll()
    assert c._pending_pcm == (0, chunk) and not c.audio.played
    c.audio.accept = True
    c.poll()
    assert c.audio.played == [(chunk.pcm16, 16000)] and c.audio.flushed == 1


def test_context_change_and_stop_discard_stale_transcripts(controller):
    c, w = controller
    c.session.events.put(
        VoiceEvent(
            "transcript",
            {"text": "Must not cross sessions", "truncated": False, "stt_ms": 1},
            0,
        )
    )
    w.session_id = "session-b"
    c.poll()
    assert c.session is None and not w.input.toPlainText()


def test_ordinary_studio_does_not_create_voice_or_load_speech_modules(
    qt_app, tmp_path, monkeypatch
):
    from src.studio.window import StudioWindow

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    w = StudioWindow(tmp_path / "project")
    try:
        assert w.voice_controller is None
        assert w.voice_button.toolTip()
        w.voice_button.click()
        assert w.voice_controller is not None and w.voice_controller.session is None
        assert w.voice_controller.audio is None
    finally:
        w.close()


def test_interrupt_suppresses_late_agent_reply_until_actual_next_turn(controller):
    c, w = controller
    c.cancel_speech()
    c.on_event({"type": "run_start"})  # queued old event must not re-enable speech
    c.on_event({"type": "final_response", "text": "Late canceled response"})
    c.on_result({"text": "Late canceled response", "metrics": {"completed": True}})
    assert not c.session.spoken
    c.begin_turn()
    c.on_event({"type": "final_response", "text": "A new requested reply"})
    assert c.session.spoken == ["A new requested reply"]


def test_new_agent_turn_acknowledges_voice_review_already_sent_from_composer(
    controller,
):
    c, w = controller
    c.session.events.put(
        VoiceEvent(
            "transcript",
            {"text": "Reviewed request", "truncated": False, "stt_ms": 1},
            0,
        )
    )
    c.poll()
    assert c._review_inserted and c.dialog.transcript.toPlainText()
    w.send()
    c.begin_turn()  # Studio synchronously acknowledges accepted composer submits
    assert c._review_inserted is None and not c.dialog.transcript.toPlainText()


def test_restart_waits_for_canceled_native_work_without_loading_second_models(
    controller,
):
    c, w = controller
    old = c.session
    old.retired = False
    c.stop()
    c.start({"language": "auto"})
    assert c.session is None and c.audio is None
    assert "still finishing" in c.dialog.error_label.text()
    old.retired = True
    c.start({"language": "auto"})
    assert c.session is not None
