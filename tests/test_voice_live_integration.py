"""Live Voice configuration never stalls or silently replaces the Agent task."""

import json
import os
from threading import Event
from types import SimpleNamespace
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QWidget, QPlainTextEdit, QListWidget

from src.studio.voice_controller import VoiceController
from src.studio.settings import SettingsPanel
from src.studio.state import default_config
from src.voice.providers import AudioChunk
from src.voice.streaming import Utterance
from tests.test_voice_controller import Audio, Session


def wait_for(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


class Output(Audio):
    def __init__(self, parent):
        super().__init__(parent)
        self.device = ""
        self.prepared = 0

    def prepare_output(self):
        self.prepared += 1
        return {"name": "Scripted speakers"}

    def set_output_device(self, device):
        self.device = device


class Synthesizer:
    def __init__(self):
        self.spoken = []

    def synthesize(self, text, *args, **kwargs):
        self.spoken.append(text)
        yield AudioChunk(24000, b"\1\0" * 2400)


@pytest.fixture
def voice(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.studio.voice_audio.AudioDevices.devices",
        lambda **kwargs: {"inputs": [], "outputs": []},
    )
    w = QWidget()
    w.workspace, w.session_id = tmp_path, "live-session"
    w.store = SimpleNamespace(root=tmp_path / "studio")
    w.input = QPlainTextEdit(w)
    w.settings = SettingsPanel(default_config())
    w.settings.setParent(w)
    w.nav = QListWidget(w)
    w.nav.addItems(["Chat", "Graph", "Automation", "Skills", "Settings"])
    w.busy = w._automation_run = False
    w.sent, w.stops = [], []
    synthesizer = Synthesizer()
    c = VoiceController(
        w,
        audio_factory=Output,
        provider_factory=lambda settings: (object(), synthesizer),
    )
    w.voice_controller = c
    c.ensure_dialog()
    c.dialog.populate({"input_mode": "text", "language": "en"})

    def send():
        w.sent.append(w.input.toPlainText())
        w.input.clear()
        if not w.busy:
            c.begin_turn()

    w.send = send
    w.stop = lambda: w.stops.append("Agent stop")
    yield c, w, synthesizer
    c.close()
    wait_for(lambda: all(session.retired for session in c._retiring))
    w.settings.close()
    w.close()
    qt_app.processEvents()


def test_inactive_send_starts_voice_and_reads_the_committed_answer(voice):
    c, w, synth = voice
    assert c.session is None
    c.send_review("A typed Voice request")
    assert c.session is not None and c.session.recognizer is None
    assert w.sent == ["A typed Voice request"]
    c.on_result({"text": "A spoken answer", "metrics": {"completed": True}})
    wait_for(lambda: bool(c.audio.played))
    assert synth.spoken == ["A spoken answer"] and not w.stops


def test_speaker_diagnostic_never_feeds_its_tone_to_the_active_microphone(voice):
    c, w, _ = voice
    c.start({"input_mode": "microphone", "language": "en"})
    current = c.session
    changes = []
    c.audio.set_muted = changes.append
    c.test_output()
    assert c.session is current and changes == [True] and c._test_audio.played
    c._close_test_audio()
    assert changes == [True, False] and not w.stops


def test_busy_voice_message_uses_studio_submission_and_preserves_unrelated_draft(voice):
    c, w, _ = voice
    w.busy = True
    w.input.setPlainText("An unrelated unsent chat draft")
    c.send_review("Steer or queue this Voice request")
    assert w.sent == ["Steer or queue this Voice request"]
    assert w.input.toPlainText() == "An unrelated unsent chat draft"
    assert w.busy and not w.stops
    assert not c.dialog.transcript.toPlainText()


def test_preview_during_agent_work_does_not_suppress_its_later_reply(voice):
    c, w, synth = voice
    c.start(c.dialog.values())
    w.busy = True
    c.speak_text("Voice preview while the Agent is working")
    wait_for(lambda: bool(c.audio.played))
    assert not c._speech_suppressed and w.busy and not w.stops
    c.on_event({"type": "final_response", "text": "The Agent reply after the preview"})
    wait_for(lambda: "The Agent reply after the preview" in synth.spoken)
    assert w.busy and not w.stops


def test_apply_while_agent_busy_waits_for_native_retirement_and_keeps_final_reply(
    voice,
):
    c, w, _ = voice
    created = []

    class ControlledSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def close(self):
            super().close()
            self.retired = False

    c._session_factory = ControlledSession
    c.start(c.dialog.values())
    old = c.session
    w.busy = True
    c.apply_settings(
        {**c.dialog.values(), "speech_speed": 1.2, "output_device_id": "new-speakers"}
    )
    assert c.session is None and len(created) == 1
    assert w.busy and not w.stops
    c.on_event(
        {"type": "final_response", "text": "Answer during Voice reconfiguration"}
    )
    c.poll()
    assert len(created) == 1
    old.retired = True
    c.poll()
    assert c.session is not None and c.session is not old
    assert c._settings["speech_speed"] == 1.2
    assert c.audio.device == "new-speakers"
    assert c.session.spoken == ["Answer during Voice reconfiguration"]
    assert w.busy and not w.stops
    # The fake explicitly completes after close so fixture cleanup remains real.
    c.session.close = lambda owner=c.session: setattr(owner, "retired", True)


def test_invalid_active_settings_leave_working_voice_running(voice):
    c, w, _ = voice
    c.start(c.dialog.values())
    active = c.session
    before = dict(c._settings)
    saved = json.loads(c._settings_path().read_text())
    c.dialog.kokoro_model_path.setText("/missing/model.onnx")
    assert json.loads(c._settings_path().read_text()) == saved

    def invalid(settings):
        raise ValueError("selected voice file is missing")

    c._provider_factory = invalid
    c.apply_settings({**c.dialog.values(), "kokoro_model_path": "/missing/model.onnx"})
    assert c.session is active and c._settings == before
    assert json.loads(c._settings_path().read_text()) == saved
    assert c.dialog.kokoro_model_path.text() == "/missing/model.onnx"
    assert c.dialog.model_page.isEnabled()
    assert "not applied" in c.dialog.error_label.text()
    assert not w.stops


def test_pending_restart_cannot_cross_workspace_or_conversation(voice, tmp_path):
    c, w, _ = voice
    created = []

    class ControlledSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def close(self):
            super().close()
            self.retired = False

    c._session_factory = ControlledSession
    c.start(c.dialog.values())
    old = c.session
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.15})
    assert c.session is None and c._restart_settings
    c.on_event({"type": "final_response", "text": "Old workspace answer"})
    w.workspace = tmp_path / "different-workspace"
    w.session_id = "different-conversation"
    w.input.setPlainText("Keep the new workspace draft")
    old.retired = True
    c.poll()
    assert c.session is None and len(created) == 1
    assert c._restart_settings is None and not c._pending_replies
    assert w.input.toPlainText() == "Keep the new workspace draft"
    assert not w.stops


@pytest.mark.parametrize("agent_busy", [False, True])
def test_send_during_native_restart_dispatches_now_and_replays_only_current_reply(
    voice, agent_busy
):
    c, w, _ = voice

    class ControlledSession(Session):
        def close(self):
            super().close()
            self.retired = False

    c._session_factory = ControlledSession
    c.start(c.dialog.values())
    old = c.session
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.1})
    assert c.session is None and c._restart_settings
    w.busy = agent_busy
    if not agent_busy:
        # A real new turn must drop queued speech and cancellation state from
        # the old task, even though its native Voice worker is still retiring.
        c.on_event({"type": "final_response", "text": "Outdated previous answer"})
    w.input.setPlainText("Keep the unrelated chat draft")
    c.send_review("Send while Voice settings are applying")
    assert w.sent == ["Send while Voice settings are applying"]
    assert w.input.toPlainText() == "Keep the unrelated chat draft"
    assert c.session is None and w.busy == agent_busy and not w.stops
    c.on_event({"type": "final_response", "text": "Current answer"})
    c.on_result({"text": "Current answer", "metrics": {"completed": True}})
    old.retired = True
    c.poll()
    assert c.session is not None and c.session is not old
    assert c.session.spoken == ["Current answer"]
    assert not w.stops
    c.session.close = lambda owner=c.session: setattr(owner, "retired", True)


def test_explicit_model_download_does_not_interrupt_live_voice_or_agent(
    voice, monkeypatch, tmp_path
):
    c, w, _ = voice
    entered, release = Event(), Event()

    def download(key, **kwargs):
        entered.set()
        release.wait(3)
        return {"key": key, "path": str(tmp_path / "new-english.onnx")}

    monkeypatch.setattr("src.studio.voice_controller.download_model", download)
    c.start(c.dialog.values())
    active = c.session
    selected = c._settings["tts_en_model"]
    w.busy = True
    try:
        c.download("piper-en")
        assert entered.wait(1)
        assert c.session is active and w.busy and not w.stops
        release.set()
        wait_for(lambda: c._download is not None and not c._download.is_alive())
        c.poll()
        assert c.session is active and w.busy and not w.stops
        assert c._settings["tts_en_model"] == selected
        assert c.dialog.tts_en_model.text() == str(tmp_path / "new-english.onnx")
    finally:
        release.set()


def test_microphone_to_text_discards_native_recognition_finishing_late(voice):
    c, w, synth = voice
    entered, release = Event(), Event()

    class SlowRecognition:
        def transcribe(self, *args, **kwargs):
            entered.set()
            release.wait(4)
            return "Late microphone text must not replace typing"

    c._provider_factory = lambda settings: (SlowRecognition(), synth)
    c.dialog.populate({"input_mode": "microphone", "language": "en"})
    c.start(c.dialog.values())
    old = c.session
    old.submit_audio(Utterance(b"\1\0" * 24000))
    assert entered.wait(1)
    w.input.setPlainText("Preserve this typed draft")
    try:
        c.switch_input_mode("text")
        assert c.session is None and old in c._retiring
        release.set()
        wait_for(lambda: c.session is not None and c.session is not old)
        assert c.session.recognizer is None
        assert w.input.toPlainText() == "Preserve this typed draft"
        assert not w.sent and not w.stops
        assert not c.dialog.transcript.toPlainText()
    finally:
        release.set()


def test_voice_settings_mount_is_accessible_while_agent_runs_without_starting_capture(
    voice,
):
    c, w, _ = voice
    w.busy = True
    w.settings.set_runtime_busy(True)
    c.show_settings()
    assert w.settings.isEnabled() and w.settings.voice_page.isEnabled()
    assert w.settings.pages.currentWidget() is w.settings.voice_page
    assert c.dialog.model_page.isEnabled() and c.dialog.audio_page.isEnabled()
    assert c.session is None and c.audio is None
    assert not w.settings.provider.isEnabled() and not w.stops


@pytest.mark.parametrize("busy_mode", [0, 1])
def test_actual_studio_keeps_voice_settings_and_busy_submission_usable(
    qt_app, tmp_path, monkeypatch, busy_mode
):
    from PySide6.QtWidgets import QMessageBox
    from src.agent.agent import Agent
    from src.models.LLMResult import LLMResult
    from src.studio.window import StudioWindow

    entered, release = Event(), Event()
    errors = []
    calls = []
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    monkeypatch.setattr(
        "src.studio.voice_audio.AudioDevices.devices",
        lambda **kwargs: {"inputs": [], "outputs": []},
    )

    class BlockingModel:
        model = None

        def generate(self, messages, **kwargs):
            calls.append(messages)
            entered.set()
            release.wait(5)
            return LLMResult("A committed reply", {}, [], None, 1)

    w = StudioWindow(
        tmp_path / "project",
        agent_factory=lambda config: Agent(config, llm=BlockingModel()),
    )
    c = VoiceController(
        w, audio_factory=Output, provider_factory=lambda settings: (None, Synthesizer())
    )
    w.voice_controller = c
    c.ensure_dialog()
    c.dialog.populate({"input_mode": "text", "language": "en"})
    try:
        w.input.setPlainText("First active task")
        w.send()
        wait_for(lambda: entered.is_set() and w.busy)
        c.show_settings()
        assert w.settings.isEnabled() and w.settings.voice_page.isEnabled()
        assert not w.settings.provider.isEnabled()
        assert c.session is None, (
            "Opening settings must not start the microphone or a voice model"
        )
        assert w.settings.pages.currentWidget() is w.settings.voice_page
        w.busy_mode.setCurrentIndex(busy_mode)
        w.input.setPlainText("Preserved chat draft")
        c.send_review("Voice message during active work")
        assert c.session is not None and w.busy
        assert c.dialog.model_page.isEnabled() and c.dialog.audio_page.isEnabled()
        assert w.input.toPlainText() == "Preserved chat draft"
        assert "Voice message during active work" in w.transcript.toPlainText()
        if busy_mode == 1:
            assert w.queued == ["Voice message during active work"]
        else:
            assert not w.queued and "steering" in w.transcript.toPlainText().lower()
        release.set()
        wait_for(
            lambda: not w.busy and not w.queued and (busy_mode != 1 or len(calls) >= 2)
        )
        assert w.settings.provider.isEnabled()
        wait_for(lambda: c.audio and bool(c.audio.played))
        assert not errors
    finally:
        release.set()
        wait_for(lambda: not w.busy)
        w.close()
        qt_app.processEvents()
