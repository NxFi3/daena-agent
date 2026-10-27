"""Installed package: speech→real Agent worker→speech, scripted models/no devices."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
import src.voice.session
from src.agent.agent import Agent
from src.models.LLMResult import LLMResult
from src.studio.window import StudioWindow
from src.studio.voice_controller import VoiceController
from src.studio.voice_dialog import VoiceDialog
from src.voice.providers import AudioChunk
from src.voice.streaming import Utterance

checkout = Path(sys.argv[1]).resolve()
assert not Path(src.voice.session.__file__).resolve().is_relative_to(checkout)
spoken = []
prompts = []


class Model:
    model = None

    def generate(self, messages, **kwargs):
        prompts.append(messages)
        return LLMResult("سلام. Hello, I am Daena.", {}, [], None, 1)


class Recognizer:
    def transcribe(self, *args, **kwargs):
        return "سلام داینا"


class Synthesizer:
    def synthesize(self, text, *args, **kwargs):
        spoken.append(text)
        yield AudioChunk(16000, b"\0\0" * 1600)


class Perception:
    def analyze(self, pcm16, sample_rate=16000, **kwargs):
        assert pcm16 and sample_rate == 16000
        return {
            "speaker": {"id": "speaker_1", "status": "new", "similarity": None},
            "sounds": {"status": "unknown", "events": []},
        }


class Audio(QObject):
    pcm_ready = Signal(bytes)
    level = Signal(float)
    fault = Signal(str)
    playback_started = Signal()
    playback_stopped = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.played = []

    def start_capture(self, *args):
        return True

    def set_output_device(self, *args):
        pass

    def set_muted(self, *args):
        pass

    def can_accept(self, *args):
        return True

    def play_pcm(self, pcm, rate):
        self.played.append(pcm)
        return True

    def stop_playback(self):
        self.played.clear()

    def finish_playback(self):
        pass

    def close(self):
        pass

    @property
    def queued_seconds(self):
        return 0.0


app = QApplication([])
with TemporaryDirectory() as directory:
    os.environ["XDG_DATA_HOME"] = str(Path(directory) / "data")
    window = StudioWindow(
        directory, agent_factory=lambda config: Agent(config, llm=Model())
    )
    controller = VoiceController(
        window,
        audio_factory=Audio,
        provider_factory=lambda config: (Recognizer(), Synthesizer()),
        perception_factory=lambda config: Perception(),
    )
    window.voice_controller = controller
    controller.dialog = VoiceDialog(parent=window)
    controller.start(
        {"language": "auto", "auto_send": True, "perception_enabled": True}
    )
    controller.session.submit_audio(Utterance(b"\0\0" * 1600))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if window._result and controller.audio.played and not window.busy:
            break
        time.sleep(0.01)
    assert window._result and window._result["metrics"]["completed"], window._result
    assert "سلام داینا" in window.transcript.toPlainText()
    assert controller.audio.played and spoken
    assert prompts and any(
        "anonymous_session_label" in str(messages) and "speaker_1" in str(messages)
        for messages in prompts
    )
    assert window.agent is not None
    window.close()
    app.processEvents()

with TemporaryDirectory() as directory:
    os.environ["XDG_DATA_HOME"] = str(Path(directory) / "data")
    window = StudioWindow(
        directory, agent_factory=lambda config: Agent(config, llm=Model())
    )

    class OutputOnlyAudio(Audio):
        def start_capture(self, *args):
            raise AssertionError("Text-only installed client accessed microphone")

    controller = VoiceController(
        window,
        audio_factory=OutputOnlyAudio,
        provider_factory=lambda config: (None, Synthesizer()),
    )
    window.voice_controller = controller
    controller.dialog = VoiceDialog(
        {"input_mode": "text", "language": "en"}, parent=window
    )
    controller.speak_text("Hello, this is a direct preview.")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not controller.audio.played:
        app.processEvents()
        time.sleep(0.01)
    assert controller.audio.played and window.agent is None
    assert (
        controller.session.recognizer is None and controller.session.segmenter is None
    )
    controller.open()
    controller.screen.composer.setPlainText("Please answer this typed request.")
    controller.screen.send_button.click()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if window._result and controller.audio.played and not window.busy:
            break
        time.sleep(0.01)
    assert window._result and window._result["metrics"]["completed"]
    assert controller.audio.played and window.agent is not None
    assert "Please answer this typed request." in window.transcript.toPlainText()
    window.close()
    app.processEvents()
print("INSTALLED_LOCAL_VOICE_WHEEL_OK")
