"""Output-only Voice works without Whisper, VAD, microphone or Agent inference."""

import os
from types import SimpleNamespace
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget, QPlainTextEdit

from src.studio.voice_controller import VoiceController
from src.studio.voice_dialog import VoiceDialog
from src.voice.providers import AudioChunk
from src.voice.session import VoiceSession
from tests.test_voice_controller import Audio


def wait_for(predicate, seconds=3):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        QApplication.processEvents()
        time.sleep(0.01)  # Let native inference workers acquire the Python GIL.
        if predicate():
            return
    assert predicate()


class NoMicrophone(Audio):
    def start_capture(self, *args):
        pytest.fail("Text-only Voice must never request microphone capture")


class NoRecognition:
    def prepare(self, **kwargs):
        pytest.fail("Text-only Voice must never warm STT")

    def transcribe(self, *args, **kwargs):
        pytest.fail("Text-only Voice must never transcribe")


class Synthesizer:
    def __init__(self):
        self.spoken = []

    def synthesize(self, text, *args, **kwargs):
        self.spoken.append(text)
        yield AudioChunk(24000, b"\1\0" * 2400)


@pytest.fixture
def output(qt_app, tmp_path):
    w = QWidget()
    w.workspace = tmp_path
    w.session_id = "text-session"
    w.store = SimpleNamespace(root=tmp_path / "studio")
    w.input = QPlainTextEdit(w)
    w.busy = w._automation_run = False
    w.sent = []
    synth = Synthesizer()
    c = VoiceController(
        w,
        audio_factory=NoMicrophone,
        provider_factory=lambda s: (NoRecognition(), synth),
    )
    c.dialog = VoiceDialog({"input_mode": "text", "language": "en"}, w)
    c.dialog.speak_requested.connect(c.speak_text)
    c.dialog.send_requested.connect(c.send_review)
    w.send = lambda: (
        w.sent.append(w.input.toPlainText()),
        w.input.clear(),
        c.begin_turn(),
    )
    w.stop = c.cancel_speech
    yield c, w, synth
    c.close()
    wait_for(lambda: all(s.retired for s in c._retiring))
    w.close()
    qt_app.processEvents()


def test_speak_text_starts_only_tts_and_preserves_chat_draft(output, monkeypatch):
    c, w, synth = output
    monkeypatch.setattr(
        "src.voice.session.SpeechSegmenter",
        lambda **kwargs: pytest.fail("VAD is unnecessary"),
    )
    w.input.setPlainText("Keep my draft")
    c.dialog.set_transcript("Hello Daena.")
    c.dialog.speak_button.click()
    assert c.session and c.session.recognizer is None and c.session.segmenter is None
    assert [thread.name for thread in c.session._threads] == ["daena-tts"]
    wait_for(lambda: bool(c.audio.played))
    assert synth.spoken == ["Hello Daena."]
    assert not w.sent and w.input.toPlainText() == "Keep my draft"
    assert c.dialog.transcript.toPlainText() == "Hello Daena."
    assert not c.dialog.mute_button.isEnabled()
    assert not c.dialog.stt_download.isEnabled()
    assert "microphone off" in c.dialog.status.text()


def test_enter_speaks_shift_enter_adds_line_and_send_uses_existing_agent(output):
    c, w, synth = output
    c.dialog.set_transcript("A preview")
    QTest.keyClick(c.dialog.transcript, Qt.Key_Return, Qt.ShiftModifier)
    assert "\n" in c.dialog.transcript.toPlainText()
    c.dialog.set_transcript("A preview")
    QTest.keyClick(c.dialog.transcript, Qt.Key_Return)
    wait_for(lambda: bool(c.audio.played))
    assert not w.sent and synth.spoken == ["A preview"]
    c.dialog.set_transcript("A request")
    c.dialog.send_button.click()
    assert w.sent == ["A request"]
    c.on_event({"type": "final_response", "text": "The Agent answer."})
    wait_for(lambda: "The Agent answer." in synth.spoken)


def test_typed_send_can_start_output_without_whisper(output):
    c, w, synth = output
    c.dialog.set_transcript("Typed request")
    c.dialog.send_button.click()
    assert c.session and c.session.recognizer is None and w.sent == ["Typed request"]
    c.on_result({"text": "Spoken reply", "metrics": {"completed": True}})
    wait_for(lambda: "Spoken reply" in synth.spoken)


def test_text_mode_bypasses_missing_whisper_and_unused_persian_voice(
    output, monkeypatch, tmp_path
):
    c, _, _ = output
    c._provider_factory = None
    monkeypatch.setattr(
        "src.studio.voice_controller.FasterWhisperProvider",
        lambda *a, **k: pytest.fail("Whisper instantiated"),
    )
    monkeypatch.setattr("src.studio.voice_controller.installed_models", lambda: {})
    model, voices = tmp_path / "kokoro.onnx", tmp_path / "voices.bin"
    model.touch()
    voices.touch()
    recognizer, router = c._providers(
        {
            "input_mode": "text",
            "language": "auto",
            "stt_model": "/missing",
            "english_engine": "kokoro",
            "kokoro_model_path": str(model),
            "kokoro_voices_path": str(voices),
        }
    )
    assert recognizer is None and router.providers["fa"] is None
    assert router.providers["en"].model_path == model
    router.providers["en"]._engine = object()
    router.prepare()  # English-only output must not warm an absent Persian voice.
    assert (
        c._perception(
            {
                "input_mode": "text",
                "perception_enabled": True,
                "speaker_model_path": "/missing",
            }
        )
        is None
    )


def test_output_device_enumeration_does_not_touch_input_devices(monkeypatch):
    from src.studio import voice_audio

    class Media:
        @staticmethod
        def audioInputs():
            pytest.fail("Text-only device enumeration must not inspect microphones")

        @staticmethod
        def defaultAudioInput():
            pytest.fail("Text-only device enumeration must not inspect microphones")

        @staticmethod
        def audioOutputs():
            return []

        @staticmethod
        def defaultAudioOutput():
            return object()

    monkeypatch.setattr(
        voice_audio, "_qt_audio", lambda: (None, None, None, None, Media)
    )
    assert voice_audio.AudioDevices.devices(include_inputs=False) == {
        "inputs": [],
        "outputs": [],
    }


def test_text_mode_controls_round_trip_and_microphone_mode_remains_available(qt_app):
    dialog = VoiceDialog({"input_mode": "text"})
    assert dialog.values()["input_mode"] == "text"
    assert not dialog.stt_model.isEnabled() and not dialog.input_device.isEnabled()
    assert not dialog.auto_send.isEnabled() and not dialog.perception_page.isEnabled()
    dialog._select(dialog.input_mode, "microphone")
    assert dialog.stt_model.isEnabled() and dialog.input_device.isEnabled()
    assert dialog.auto_send.isEnabled() and dialog.perception_page.isEnabled()
    dialog.close()


def test_open_without_whisper_offers_text_and_only_enumerates_speakers(
    output, monkeypatch
):
    c, _, _ = output
    c.dialog.deleteLater()
    c.dialog = None
    seen = []
    monkeypatch.setattr("src.studio.voice_controller.installed_models", lambda: {})
    monkeypatch.setattr(
        "src.studio.voice_audio.AudioDevices.devices",
        lambda **kwargs: seen.append(kwargs) or {"inputs": [], "outputs": []},
    )
    c.open()
    assert c.dialog.values()["input_mode"] == "text"
    assert seen == [{"include_inputs": False}]
    c.dialog._select(c.dialog.input_mode, "microphone")
    assert seen[-1] == {"include_inputs": True}


def test_new_agent_turn_interrupts_preview_without_narrating_old_audio(output):
    c, w, synth = output
    c.speak_text("Preview before request")
    wait_for(lambda: bool(c.audio.played))
    epoch = c.session.epoch
    c.begin_turn()
    assert c.session.epoch > epoch and not c.audio.played
    c.on_event({"type": "final_response", "text": "Fresh answer"})
    wait_for(lambda: "Fresh answer" in synth.spoken)


def test_native_kokoro_gui_text_output_needs_no_whisper_or_microphone(output):
    directory = os.environ.get("DAENA_TEST_VOICE_MODELS")
    if not directory:
        pytest.skip(
            "Explicitly provision local Kokoro model files to run native speech"
        )
    from pathlib import Path

    root = Path(directory)
    c, w, _ = output
    c._provider_factory = None
    c.dialog.populate(
        {
            "input_mode": "text",
            "language": "auto",
            "english_engine": "kokoro",
            "kokoro_model_path": str(root / "kokoro-v1.0.onnx"),
            "kokoro_voices_path": str(root / "voices-v1.0.bin"),
        }
    )
    c.speak_text("Hello, I am Daena.")
    assert c.session and c.session.recognizer is None
    wait_for(
        lambda: c.session is None or (c.audio and bool(c.audio.played)), seconds=15
    )
    assert c.session is not None, c.dialog.error_label.text()
    assert any(any(pcm) for pcm, _ in c.audio.played)
    assert "first_pcm_ms" in c._metrics and not w.sent
